/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import { UploadCloud, FileUp, X, Play, Pause, Brain, Activity, AlertTriangle, Download, Sparkles, CheckCircle2 } from "lucide-react";
import { api, ApiError, API_BASE_URL, UploadPredictionResponse } from "@/lib/api";
import { parseFilesLocal, ParsedData } from "@/lib/parser";
import { scores, readingAt, ReadingSeries, Scores } from "@/lib/metrics";
import { PageHeader, Panel, Callout, Stat, Segmented, Expand } from "@/components/blocks";
import { ThemedPlot, usePlotColors } from "@/components/plot";
import { ClarkeGrid } from "@/components/clarke-grid";
import { Button } from "@/components/ui/button";
import { DISCLAIMER } from "@/components/layout/disclaimer";
import { cn } from "@/lib/utils";

const HOUR = 3600 * 1000;
const toLocalIso = (ms: number) => format(new Date(ms), "yyyy-MM-dd'T'HH:mm:ss");

interface Row { tPred: number; tFc: number; pred: number; actual: number | null; current: number | null; }

const MAX_FILES = 4;
const START_CMD = "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860";

export default function LabPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [local, setLocal] = useState<ParsedData | null>(null);
  const [res, setRes] = useState<UploadPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [windowH, setWindowH] = useState<"6" | "24" | "72" | "all">("24");
  const [elapsed, setElapsed] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const c = usePlotColors();

  // Backend status: checked on load, every 5 s while offline and every 30 s while online.
  const health = useQuery({
    queryKey: ["api-health"], queryFn: api.getApiHealth, retry: false, refetchOnWindowFocus: true,
    refetchInterval: (q) => (q.state.status === "error" ? 5000 : 30000), refetchIntervalInBackground: true,
  });
  const backendDown = health.isError;

  const run = useMutation({
    mutationFn: async (fs: File[]) => {
      const [parsed, out] = await Promise.all([parseFilesLocal(fs), api.predictUpload(fs, "all")]);
      return { parsed, out };
    },
    onSuccess: ({ parsed, out }) => {
      setLocal(parsed); setRes(out); setError(null);
      setCursor(Math.max(0, out.predictions.length - 1));
    },
    onError: (e: any) => {
      setRes(null);
      if (e instanceof ApiError && e.code === "PREDICTION_TIMEOUT") setError("PREDICTION_TIMEOUT: the forecast took longer than the backend allows. Restart the backend with start-local.bat (it allows 3 minutes), or upload one file at a time.");
      else if (e instanceof ApiError) setError(`${e.code}: ${e.message}`);
      else if (e instanceof TypeError) {
        setError(`Cannot reach the backend at ${API_BASE_URL}. Start it from the repository root with: ${START_CMD}`);
        health.refetch();
      } else setError(e?.message ?? "Unexpected error while forecasting.");
    },
  });

  useEffect(() => {
    if (!run.isPending) return;
    const t0 = Date.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [run.isPending]);

  const reset = () => { setRes(null); setLocal(null); setError(null); };
  /** Adds files (same name = replaced), keeping at most 4; the upload starts only with "Run forecast". */
  const addFiles = (incoming: File[]) => {
    if (!incoming.length) return;
    setFiles((prev) => {
      const merged = [...prev.filter((p) => !incoming.some((f) => f.name === p.name)), ...incoming];
      if (merged.length > MAX_FILES) setError(`At most ${MAX_FILES} files of one dataset can be uploaded together; keeping the last ${MAX_FILES}.`);
      else setError(null);
      return merged.slice(-MAX_FILES);
    });
    setRes(null); setLocal(null);
  };
  const removeFile = (name: string) => { setFiles((prev) => prev.filter((f) => f.name !== name)); reset(); };
  const start = (fs: File[]) => { reset(); if (fs.length) run.mutate(fs); };
  const useDemo = async () => {
    const blob = await (await fetch("/samples/synthetic_01.csv")).blob();
    const demo = [new File([blob], "synthetic_01.csv", { type: "text/csv" })];
    setFiles(demo);
    start(demo);
  };

  // ------------------------------------------------------------- forecasts vs real readings
  const series: ReadingSeries | null = useMemo(() =>
    local ? { t: local.glucose.map((g) => g.t), v: local.glucose.map((g) => g.value) } : null, [local]);

  const rows: Row[] = useMemo(() => {
    if (!res || !series) return [];
    return res.predictions.map((p) => {
      const tPred = new Date(p.prediction_time).getTime(), tFc = new Date(p.forecast_time).getTime();
      return { tPred, tFc, pred: p.predicted_glucose_mg_dl, actual: readingAt(series, tFc), current: readingAt(series, tPred) };
    });
  }, [res, series]);

  const scored = useMemo(() => rows.filter((r) => r.actual != null && r.current != null), [rows]);
  const model: Scores | null = useMemo(() => scores(scored.map((r) => r.actual!), scored.map((r) => r.pred)), [scored]);
  const persist: Scores | null = useMemo(() => scores(scored.map((r) => r.actual!), scored.map((r) => r.current!)), [scored]);

  useEffect(() => {
    if (!playing || !rows.length) return;
    const id = setInterval(() => setCursor((i) => Math.min(i + 1, rows.length - 1)), 120);
    return () => clearInterval(id);
  }, [playing, rows.length]);
  useEffect(() => { if (playing && cursor >= rows.length - 1) setPlaying(false); }, [playing, cursor, rows.length]);

  const cur = rows[cursor];
  const latest = rows[rows.length - 1];

  // ------------------------------------------------------------- main chart
  const chart = useMemo(() => {
    if (!local || !cur) return null;
    const data: any[] = [
      { type: "scattergl", mode: "lines", name: "Real glucose", x: local.glucose.map((g) => g.timestamp), y: local.glucose.map((g) => g.value),
        line: { color: c.accent, width: 2 }, hovertemplate: "%{x|%b %d %H:%M}<br>Real %{y:.0f} mg/dL<extra></extra>" },
      { type: "scattergl", mode: "lines", name: "30-min forecast (drawn at its target time)", x: rows.map((r) => toLocalIso(r.tFc)), y: rows.map((r) => r.pred),
        line: { color: c.warn, width: 2, dash: "dot" }, hovertemplate: "Forecast for %{x|%b %d %H:%M}<br>%{y:.1f} mg/dL<extra></extra>" },
    ];
    if (local.carbs.length) data.push({ type: "scatter", mode: "markers", name: "Carbs", x: local.carbs.map((e) => e.timestamp), y: local.carbs.map(() => 48),
      marker: { symbol: "triangle-up", size: 9, color: "#22c55e" }, text: local.carbs.map((e) => `${e.value.toFixed(0)} g carbs`), hovertemplate: "%{text}<br>%{x|%H:%M}<extra></extra>" });
    if (local.bolus.length) data.push({ type: "scatter", mode: "markers", name: "Insulin bolus", x: local.bolus.map((e) => e.timestamp), y: local.bolus.map(() => 44),
      marker: { symbol: "triangle-down", size: 9, color: "#a855f7" }, text: local.bolus.map((e) => `${e.value} U bolus`), hovertemplate: "%{text}<br>%{x|%H:%M}<extra></extra>" });
    // the replay cursor: "now", the real value now, and the forecast made now for 30 min later
    data.push({ type: "scatter", mode: "lines+markers", name: "Forecast made at the cursor", showlegend: false,
      x: [toLocalIso(cur.tPred), toLocalIso(cur.tFc)], y: [cur.current, cur.pred],
      line: { color: c.red, width: 2, dash: "dash" }, marker: { size: [8, 14], color: c.red, symbol: ["circle", "diamond"] },
      hovertemplate: "%{x|%H:%M}: %{y:.1f}<extra></extra>" });
    if (cur.actual != null) data.push({ type: "scatter", mode: "markers", showlegend: false, x: [toLocalIso(cur.tFc)], y: [cur.actual],
      marker: { size: 13, color: c.accent, line: { width: 2, color: c.text } }, hovertemplate: "Real at target: %{y:.0f}<extra></extra>" });

    const end = cur.tFc + HOUR;
    const range = windowH === "all" ? undefined : [toLocalIso(cur.tPred - +windowH * HOUR), toLocalIso(end)];
    const layout = {
      margin: { t: 16, r: 16, l: 56, b: 40 },
      xaxis: { type: "date", range },
      yaxis: { title: { text: "Glucose (mg/dL)" }, range: [40, 400] },
      shapes: [
        { type: "rect", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 180, fillcolor: c.band, line: { width: 0 }, layer: "below" },
        { type: "rect", x0: toLocalIso(cur.tPred), x1: toLocalIso(cur.tFc), yref: "paper", y0: 0, y1: 1,
          fillcolor: c.dark ? "rgba(248,113,113,0.08)" : "rgba(220,38,38,0.06)", line: { width: 0 }, layer: "below" },
        { type: "line", x0: toLocalIso(cur.tPred), x1: toLocalIso(cur.tPred), yref: "paper", y0: 0, y1: 1, line: { color: c.red, width: 1.5 } },
      ],
      annotations: [{ x: toLocalIso(cur.tPred), y: 1, yref: "paper", text: "now", showarrow: false, yanchor: "bottom", font: { color: c.red, size: 11 } },
        { x: toLocalIso(cur.tFc), y: 1, yref: "paper", text: "+30 min", showarrow: false, yanchor: "bottom", font: { color: c.red, size: 11 } }],
    };
    return { data, layout };
  }, [local, rows, cur, windowH, c]);

  const hist = useMemo(() => {
    if (!scored.length) return null;
    return [
      { type: "histogram", name: "LightGBM", x: scored.map((r) => r.pred - r.actual!), opacity: 0.7, marker: { color: c.warn }, xbins: { size: 5 } },
      { type: "histogram", name: "Persistence (no change)", x: scored.map((r) => r.current! - r.actual!), opacity: 0.45, marker: { color: "#94a3b8" }, xbins: { size: 5 } },
    ];
  }, [scored, c]);

  const inSample = res?.dataset.detected_type === "ohiot1dm_xml";
  const synthetic = files.length === 1 && files[0].name === "synthetic_01.csv";

  return (
    <div className="container max-w-screen-xl py-10 space-y-8">
      <PageHeader eyebrow="Forecast Lab" title="See the 30-minute forecast on real data">
        Upload CGM data (or use the synthetic demo patient). The backend makes an independent 30-minute forecast at every 5-minute point
        using only data up to that moment. Here you can replay those forecasts and compare them with what really happened.
      </PageHeader>

      {/* --------------------------------------------------- source */}
      <div className="grid lg:grid-cols-[1.4fr_1fr] gap-5">
        <Panel className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="font-semibold">Upload CGM Data</div>
            <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
              backendDown ? "border-destructive/40 text-destructive bg-destructive/10"
                : health.data ? "border-emerald-500/40 text-emerald-700 dark:text-emerald-400 bg-emerald-500/10" : "text-muted-foreground")}>
              <span className={cn("w-1.5 h-1.5 rounded-full", backendDown ? "bg-destructive" : health.data ? "bg-emerald-500" : "bg-muted-foreground")} />
              {backendDown ? "Backend offline" : health.data ? `Backend ready · ${health.data.model_name} ${health.data.model_version}` : "Checking backend…"}
            </span>
          </div>
          {backendDown && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm space-y-1">
              <div className="font-medium text-destructive">The forecasting backend is not reachable at {API_BASE_URL}.</div>
              <div className="text-muted-foreground">Start it from the repository root, then this page reconnects by itself:</div>
              <code className="block text-xs bg-background/70 rounded px-2 py-1.5 border">{START_CMD}</code>
            </div>
          )}
          <div onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); addFiles(Array.from(e.dataTransfer.files)); }}
            className="border-2 border-dashed rounded-xl p-7 text-center cursor-pointer hover:bg-muted/40 hover:border-primary/40 transition-colors">
            <UploadCloud className="w-9 h-9 mx-auto text-muted-foreground" />
            <p className="mt-3 font-medium">Drop files here, or click to add files</p>
            <p className="text-xs text-muted-foreground mt-1">Add 1–4 files of <strong>one</strong> dataset (you can add them one at a time) · .csv or .xml · up to 25 MB in total · processed in memory, never stored</p>
            <input ref={inputRef} type="file" multiple accept=".csv,.xml" className="hidden"
              onChange={(e) => { addFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
          </div>
          {files.length > 0 && (
            <div className="space-y-2">
              {files.map((f) => (
                <div key={f.name} className="flex items-center justify-between gap-2 rounded-lg border bg-muted/40 px-3 py-2 text-sm">
                  <span className="flex items-center gap-2 min-w-0"><FileUp className="w-4 h-4 text-primary shrink-0" />
                    <span className="truncate">{f.name}</span><span className="text-xs text-muted-foreground shrink-0">{(f.size / 1024).toFixed(0)} KB</span></span>
                  <button onClick={() => removeFile(f.name)} disabled={run.isPending} aria-label={`Remove ${f.name}`}
                    className="text-muted-foreground hover:text-foreground disabled:opacity-40"><X className="w-4 h-4" /></button>
                </div>
              ))}
              <p className="text-xs text-muted-foreground">
                HUPA-UCM: add the Libre export <em>and</em> its Preprocessed CSV for insulin and carbs. OhioT1DM: training and testing XML of the same patient can go together.
              </p>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => start(files)} disabled={!files.length || run.isPending || backendDown}>
              <Brain className="w-4 h-4 mr-2" />{run.isPending ? "Forecasting…" : `Run forecast${files.length ? ` (${files.length} file${files.length > 1 ? "s" : ""})` : ""}`}
            </Button>
            <Button variant="secondary" onClick={useDemo} disabled={run.isPending || backendDown}><Sparkles className="w-4 h-4 mr-2" />Use the synthetic demo patient</Button>
            <Button variant="outline" asChild><a href="/samples/synthetic_01.csv" download><Download className="w-4 h-4 mr-2" />CSV template</a></Button>
          </div>
          {run.isPending && (
            <div className="flex items-center gap-2 text-sm text-primary">
              <Activity className="w-4 h-4 animate-spin" />Validating the data and forecasting every 5-minute point… {elapsed}s
              {elapsed >= 5 && <span className="text-xs text-muted-foreground">(real OhioT1DM files take 10–40 s; please wait, do not upload again)</span>}
            </div>
          )}
          {error && <div className="rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm p-3 break-words">{error}</div>}
        </Panel>
        <Panel className="space-y-3 text-sm">
          <div className="font-semibold">Supported formats</div>
          <ul className="space-y-2 text-muted-foreground">
            <li><strong className="text-foreground">OhioT1DM XML</strong> — 1–2 files of one patient</li>
            <li><strong className="text-foreground">FreeStyle Libre / HUPA-UCM export</strong> — plus the optional HUPA-UCM Preprocessed CSV for insulin and carbs (~15-minute data, flagged)</li>
            <li><strong className="text-foreground">CSV template</strong> — <code className="text-xs">timestamp, glucose_mg_dl, bolus_units, carbs_g</code>, local time, empty cell = no event</li>
          </ul>
          <p className="text-xs text-muted-foreground">This research system accepts structured glucose/insulin/carbohydrate data. It does not process medical-record PDFs, images or EHR files.
            At least about an hour of CGM data is needed before a forecast can be made.</p>
        </Panel>
      </div>

      {res && cur && (
        <>
          {res.warnings.length > 0 && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex gap-3 text-sm">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
              <ul className="space-y-1 text-amber-900 dark:text-amber-200">{res.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </div>
          )}
          {synthetic && (
            <Callout title="This is the synthetic demo patient">
              It is a smooth, artificial glucose curve made for demonstrations (not from any real person). On such a smooth curve the
              no-change guess is unusually good, so use it to see <strong>how</strong> forecasting works, not how accurate the model is.
              Real accuracy comes from the research on unseen patients: MARD 10.03 % vs 12.11 % for the no-change baseline.
            </Callout>
          )}
          {inSample && (
            <Callout kind="caution" title="This is OhioT1DM data">
              The deployed model was trained on all 12 OhioT1DM patients, so its scores on OhioT1DM files are <strong>in-sample</strong> and look better than
              the honest unseen-patient result (MARD 10.03 %). Use HUPA-UCM or your own data to see out-of-sample behaviour.
            </Callout>
          )}

          {/* --------------------------------------------------- headline */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <Stat tone="primary" label="Latest forecast" value={latest.pred.toFixed(1)} unit="mg/dL"
              hint={`for ${format(new Date(latest.tFc), "MMM d, HH:mm")} · now ${latest.current != null ? latest.current.toFixed(0) : "—"} mg/dL`} />
            <Stat label="Forecasts in this file" value={res.n_predictions.toLocaleString("en-US")} hint={res.truncated ? "latest 7 days only (cap)" : `${scored.length.toLocaleString("en-US")} with a real value 30 min later`} />
            <Stat label="MARD on this file" value={model ? model.mard.toFixed(2) : "—"} unit="%" hint={persist ? `no-change baseline ${persist.mard.toFixed(2)} %` : undefined} />
            <Stat label="Data" value={res.dataset.sampling_profile} hint={`${res.dataset.detected_type} · ${res.dataset.n_glucose_readings.toLocaleString("en-US")} readings · ${res.dataset.glucose_coverage_percent} % coverage`} />
          </div>

          {/* --------------------------------------------------- replay chart */}
          <Panel className="space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <div className="font-semibold">Real glucose vs the 30-minute forecasts</div>
                <div className="text-xs text-muted-foreground">Each forecast point is drawn at the time it predicts. The red dashed segment is the forecast made at the cursor.</div>
              </div>
              <Segmented options={[{ value: "6", label: "6 h" }, { value: "24", label: "24 h" }, { value: "72", label: "3 days" }, { value: "all", label: "All" }]}
                value={windowH} onChange={setWindowH} />
            </div>
            {chart && <ThemedPlot data={chart.data} layout={chart.layout} height={420} />}
            <div className="rounded-xl border bg-muted/30 p-4 space-y-3">
              <div className="flex items-center gap-3">
                <Button size="sm" variant="outline" onClick={() => { if (cursor >= rows.length - 1) setCursor(0); setPlaying(!playing); }}>
                  {playing ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}{playing ? "Pause" : "Replay"}
                </Button>
                <input type="range" min={0} max={rows.length - 1} value={cursor} onChange={(e) => { setPlaying(false); setCursor(+e.target.value); }}
                  className="flex-1 accent-[hsl(var(--primary))]" aria-label="Replay time" />
              </div>
              <div className="grid sm:grid-cols-4 gap-3 text-sm">
                <div><div className="text-xs text-muted-foreground">Now</div><div className="font-semibold">{format(new Date(cur.tPred), "MMM d, HH:mm")}</div>
                  <div className="text-xs text-muted-foreground">reading {cur.current != null ? `${cur.current.toFixed(0)} mg/dL` : "—"}</div></div>
                <div><div className="text-xs text-muted-foreground">Forecast for {format(new Date(cur.tFc), "HH:mm")}</div>
                  <div className="font-semibold text-amber-600 dark:text-amber-400">{cur.pred.toFixed(1)} mg/dL</div>
                  <div className="text-xs text-muted-foreground">{cur.current != null ? `${cur.pred - cur.current >= 0 ? "▲ rising" : "▼ falling"} ${Math.abs(cur.pred - cur.current).toFixed(0)} mg/dL` : ""}</div></div>
                <div><div className="text-xs text-muted-foreground">What really happened</div>
                  <div className="font-semibold">{cur.actual != null ? `${cur.actual.toFixed(0)} mg/dL` : "not in the file (future)"}</div></div>
                <div><div className="text-xs text-muted-foreground">Error</div>
                  <div className="font-semibold">{cur.actual != null ? `${Math.abs(cur.pred - cur.actual).toFixed(1)} mg/dL (${((100 * Math.abs(cur.pred - cur.actual)) / cur.actual).toFixed(1)} %)` : "—"}</div>
                  {cur.actual != null && cur.current != null && <div className="text-xs text-muted-foreground">no-change guess off by {Math.abs(cur.current - cur.actual).toFixed(0)} mg/dL</div>}</div>
              </div>
            </div>
          </Panel>

          {/* --------------------------------------------------- scores */}
          {model && persist && (
            <div className="grid lg:grid-cols-2 gap-5">
              <Panel className="space-y-4">
                <div>
                  <div className="font-semibold">How good were the forecasts on this file?</div>
                  <div className="text-xs text-muted-foreground">{model.n.toLocaleString("en-US")} forecasts with a real reading 30 minutes later, scored with the research definitions.</div>
                </div>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-muted-foreground border-b"><th className="py-2 font-medium">Measure</th><th className="py-2 font-medium">LightGBM</th><th className="py-2 font-medium">No-change baseline</th></tr></thead>
                  <tbody>
                    {[["MARD (%)", model.mard, persist.mard, "lower"], ["MAE (mg/dL)", model.mae, persist.mae, "lower"], ["RMSE (mg/dL)", model.rmse, persist.rmse, "lower"],
                      ["R²", model.r2, persist.r2, "higher"], ["Clarke zone A (%)", model.zoneA, persist.zoneA, "higher"], ["Clarke A+B (%)", model.zoneAB, persist.zoneAB, "higher"]].map(([label, a, b, better]) => {
                      const win = better === "lower" ? (a as number) < (b as number) : (a as number) > (b as number);
                      return (
                        <tr key={label as string} className="border-b last:border-0">
                          <td className="py-2">{label as string}</td>
                          <td className={cn("py-2 font-semibold tabular-nums", win && "text-emerald-600 dark:text-emerald-400")}>{(a as number).toFixed(label === "R²" ? 3 : 2)}{win && <CheckCircle2 className="inline w-3.5 h-3.5 ml-1" />}</td>
                          <td className="py-2 tabular-nums text-muted-foreground">{(b as number).toFixed(label === "R²" ? 3 : 2)}</td>
                        </tr>);
                    })}
                  </tbody>
                </table>
                <p className="text-xs text-muted-foreground">One file is one person: these numbers vary. The research result across 12 unseen patients is MARD 10.03 % vs 12.11 % for the baseline.</p>
                {hist && <ThemedPlot data={hist} layout={{ barmode: "overlay", xaxis: { title: { text: "Forecast − real (mg/dL)" }, range: [-100, 100] }, yaxis: { title: { text: "Count" } }, margin: { t: 8, r: 8, l: 48, b: 48 } }} height={240} />}
              </Panel>
              <Panel className="space-y-2">
                <div className="font-semibold">Clarke Error Grid of this file</div>
                <div className="text-xs text-muted-foreground">Each dot is one forecast. Zones A and B are clinically acceptable.</div>
                <ClarkeGrid actual={scored.map((r) => r.actual!)} predicted={scored.map((r) => r.pred)} height={460}
                  highlight={cur.actual != null ? { actual: cur.actual, predicted: cur.pred } : undefined} />
              </Panel>
            </div>
          )}

          <div className="grid md:grid-cols-2 gap-4">
            <Expand title={`Validation checks (${res.validation.checks.length} passed)`}>
              <ul className="list-disc pl-5 space-y-1">{res.validation.checks.map((ch) => <li key={ch}>{ch}</li>)}</ul>
            </Expand>
            <Expand title="What was done to the data">
              {res.preprocessing.length ? <ul className="list-disc pl-5 space-y-1">{res.preprocessing.map((p) => <li key={p}>{p}</li>)}</ul> : <p>Read as recorded; no conversion needed.</p>}
              {Object.keys(res.skipped).length > 0 && <p>Times without a forecast: {Object.entries(res.skipped).map(([k, v]) => `${k} ${v}`).join(", ")}.</p>}
              <p>Model {res.model.name} · version {res.model.version} · features {res.model.feature_version}.</p>
            </Expand>
          </div>
          <p className="text-xs text-muted-foreground flex items-center gap-2"><Brain className="w-3.5 h-3.5" />{res.disclaimer || DISCLAIMER}</p>
        </>
      )}
    </div>
  );
}
