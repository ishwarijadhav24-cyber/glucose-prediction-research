"use client";

import { useMemo, useState } from "react";
import { PageHeader, Section, Panel, Callout, Stat, Segmented, FigureCard } from "@/components/blocks";
import { ThemedPlot, usePlotColors } from "@/components/plot";
import { research, MODEL_INFO, fmt, fmtP, ModelName } from "@/lib/research";

const LEVELS = [0, 5, 10, 20, 30, 50];
const summary = research.objective2.summary;
const rows = research.objective2.patients;
const days = research.objective2.days;
const vsP0 = research.objective2.vs_p0;
const PATIENTS = Array.from(new Set(rows.map((r) => r.patient)));
const sum = (model: string, p: number) => summary.find((r) => r.model === model && r.p_level === p)!;
const dayOf = (p: number) => days.find((d) => d.p_level === p)!.median;

export default function PersonalizationPage() {
  const [idx, setIdx] = useState(3);
  const [model, setModel] = useState<"LightGBM" | "Ridge">("LightGBM");
  const [patient, setPatient] = useState<string>("all");
  const c = usePlotColors();
  const p = LEVELS[idx];
  const s = sum(model, p);
  const s0 = sum(model, 0);
  const test = vsP0.find((r) => r.model === model && r.p_level === p);

  const curve = useMemo(() => {
    const series = (m: string) => patient === "all"
      ? LEVELS.map((l) => sum(m, l).MARD_mean)
      : LEVELS.map((l) => rows.find((r) => r.model === m && r.p_level === l && r.patient === patient)!.MARD);
    return (["Persistence", "Ridge", "LightGBM"] as ModelName[]).map((m) => ({
      type: "scatter", mode: "lines+markers", name: MODEL_INFO[m].label,
      x: LEVELS.map(dayOf), y: series(m), customdata: LEVELS,
      line: { color: MODEL_INFO[m].color, width: m === model ? 3.5 : 1.8, dash: m === "Persistence" ? "dot" : "solid" },
      marker: { size: m === model ? 9 : 6 },
      hovertemplate: `${MODEL_INFO[m].label}<br>P = %{customdata} % (≈ %{x:.1f} days)<br>MARD %{y:.2f} %<extra></extra>`,
    }));
  }, [patient, model]);

  const curveLayout = useMemo(() => ({
    xaxis: { title: { text: "Patient's own data used (median days)" } },
    yaxis: { title: { text: "MARD % on the fixed test half" } },
    shapes: [{ type: "line", x0: dayOf(p), x1: dayOf(p), yref: "paper", y0: 0, y1: 1, line: { color: c.warn, width: 2, dash: "dash" } }],
  }), [p, c]);

  const heat = useMemo(() => {
    const z = PATIENTS.map((pt) => LEVELS.slice(1).map((l) => rows.find((r) => r.model === model && r.p_level === l && r.patient === pt)!.Delta_MARD));
    return [{
      type: "heatmap", z, x: LEVELS.slice(1).map((l) => `${l} %`), y: PATIENTS.map((pt) => `#${pt}`),
      colorscale: [[0, "#16a34a"], [0.5, c.dark ? "#1e293b" : "#f8fafc"], [1, "#dc2626"]], zmid: 0,
      colorbar: { title: { text: "Δ MARD" }, thickness: 12 },
      hovertemplate: "Patient %{y} · P = %{x}<br>Δ MARD %{z:.2f} points<extra></extra>",
    }];
  }, [model, c]);

  return (
    <>
      <PageHeader eyebrow="Objective 2 · Personalization" title="How much of a patient's own data does the model need?">
        For each patient: start from the population model (trained on the other 11), adapt it with the first P % of that patient's data
        (continued LightGBM boosting, or prior-regularised Ridge), and always test on the same last 50 % — the personal data never overlaps the test half.
      </PageHeader>

      <Panel className="space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="text-sm font-medium">Personal data used: <span className="text-primary text-lg font-bold">{p} %</span>
            <span className="text-muted-foreground"> · ≈ {fmt(dayOf(p), 1)} days</span></div>
          <Segmented options={[{ value: "LightGBM", label: "LightGBM" }, { value: "Ridge", label: "Ridge" }]} value={model} onChange={setModel} />
        </div>
        <input type="range" min={0} max={LEVELS.length - 1} step={1} value={idx} onChange={(e) => setIdx(+e.target.value)}
          className="w-full accent-[hsl(var(--primary))]" aria-label="Personalization level" />
        <div className="flex justify-between text-xs text-muted-foreground -mt-3">{LEVELS.map((l) => <span key={l}>{l} %</span>)}</div>
        <div className="grid sm:grid-cols-4 gap-3">
          <Stat tone="primary" label={`${model} MARD`} value={fmt(s.MARD_mean)} unit="%" hint={`± ${fmt(s.MARD_std)} across 12 patients`} />
          <Stat label="Change vs no own data" value={p === 0 ? "—" : `${s.MARD_mean - s0.MARD_mean > 0 ? "+" : ""}${fmt(s.MARD_mean - s0.MARD_mean)}`} unit={p === 0 ? "" : "pts"} hint={`P0 = ${fmt(s0.MARD_mean)} %`} />
          <Stat label="Patients improved" value={test ? `${test.n_improved} / ${test.n_patients}` : "—"} hint="vs the same patient at P = 0" />
          <Stat label="Significance" value={test ? `p = ${fmtP(test.p)}` : "—"} hint={test ? (test.p < 0.05 ? "significant (p < 0.05)" : "not significant") : "baseline level"} tone={test && test.p < 0.05 ? "primary" : "default"} />
        </div>
      </Panel>

      <Section title="The personalization curve" subtitle="Average over the 12 patients, or pick one patient to see their own curve.">
        <Panel className="space-y-3">
          <select value={patient} onChange={(e) => setPatient(e.target.value)} className="rounded-md border bg-background px-3 py-1.5 text-sm">
            <option value="all">Average of all 12 patients</option>
            {PATIENTS.map((pt) => <option key={pt} value={pt}>Patient {pt}</option>)}
          </select>
          <ThemedPlot data={curve} layout={curveLayout} height={340} />
        </Panel>
        <div className="grid md:grid-cols-2 gap-4">
          <Callout kind="finding" title="Small but consistent after ~10 days">
            LightGBM: {fmt(sum("LightGBM", 0).MARD_mean)} % → {fmt(sum("LightGBM", 20).MARD_mean)} % at 20 % (p = {fmtP(vsP0.find((r) => r.model === "LightGBM" && r.p_level === 20)!.p)})
            → {fmt(sum("LightGBM", 50).MARD_mean)} % at 50 %. About 0.2 points (~2 %). A few days of data (5–10 %) do not help.
          </Callout>
          <Callout kind="caution" title="Read carefully">
            Ridge never improves significantly. The LightGBM recipe changes at P ≥ 20 % (early stopping instead of fixed rounds), so the curve
            partly reflects that change. P0 here (9.83 %) differs from Objective 1 (10.03 %) because only the last half of each patient is tested.
          </Callout>
        </div>
      </Section>

      <Section title="Who benefits?" subtitle={`Change in MARD for each patient and level (${model}). Green = better than without personal data, red = worse.`}>
        <Panel><ThemedPlot data={heat} layout={{ margin: { t: 16, r: 16, l: 64, b: 48 }, xaxis: { title: { text: "Personal data used" } } }} height={420} /></Panel>
      </Section>

      <Section title="Original research figures" subtitle="Generated by evaluate_personalization.py. Click to enlarge.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <FigureCard src="/figures/o2_curve_mard.png" title="MARD curve" caption="MARD against the personalization level." />
          <FigureCard src="/figures/o2_curve_rmse.png" title="RMSE curve" caption="RMSE against the personalization level." />
          <FigureCard src="/figures/o2_heatmap.png" title="Adaptation heatmap" caption="ΔMARD per patient and level." />
          <FigureCard src="/figures/o2_clarke_p0_p50.png" title="Clarke P0 vs P50" caption="Clarke grid without and with 50 % personal data." />
        </div>
      </Section>
    </>
  );
}
