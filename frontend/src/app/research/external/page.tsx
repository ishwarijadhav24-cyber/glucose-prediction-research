"use client";

import { useMemo, useState } from "react";
import { PageHeader, Section, Panel, Callout, Segmented, Stat, FigureCard } from "@/components/blocks";
import { ThemedPlot, usePlotColors } from "@/components/plot";
import { research, MODELS, MODEL_INFO, fmt, fmtP, ModelName } from "@/lib/research";
import { cn } from "@/lib/utils";

const ohio = research.objective1.summary;
const o3 = research.objective3;

export default function ExternalPage() {
  const [cohort, setCohort] = useState<"main" | "all">("main");
  const [gapModel, setGapModel] = useState<ModelName>("LightGBM");
  const [scatterModel, setScatterModel] = useState<ModelName>("LightGBM");
  const c = usePlotColors();
  const hupa = o3[cohort];
  const hL = o3.main.find((r) => r.model === "LightGBM")!;
  const hP = o3.main.find((r) => r.model === "Persistence")!;
  const vsL = o3.vs_persistence.find((r) => r.model === "LightGBM")!;

  const bars = useMemo(() => [
    { type: "bar", name: "OhioT1DM (LOSO)", x: MODELS.map((m) => MODEL_INFO[m].label),
      y: MODELS.map((m) => ohio.find((r) => r.model === m)!.MARD_mean), marker: { color: c.accent },
      hovertemplate: "%{x} · OhioT1DM<br>MARD %{y:.2f} %<extra></extra>" },
    { type: "bar", name: `HUPA-UCM ${cohort === "main" ? "MAIN (19)" : "ALL (23)"}`, x: MODELS.map((m) => MODEL_INFO[m].label),
      y: MODELS.map((m) => hupa.find((r) => r.model === m)!.MARD_mean), marker: { color: c.warn },
      hovertemplate: "%{x} · HUPA-UCM<br>MARD %{y:.2f} %<extra></extra>" },
  ], [cohort, hupa, c]);

  const g = o3.gap.find((r) => r.model === gapModel)!;
  const waterfall = useMemo(() => [{
    type: "waterfall", orientation: "v",
    x: ["OhioT1DM", "Scoring rule", "15-min sampling", "Population / device", "HUPA-UCM"],
    measure: ["absolute", "relative", "relative", "relative", "total"],
    y: [g.ohio_A, g.eval_rule_effect, g.sampling_effect, g.remaining_effect, 0],
    text: [fmt(g.ohio_A), fmt(g.eval_rule_effect), `+${fmt(g.sampling_effect)}`, `+${fmt(g.remaining_effect)}`, fmt(g.hupa_main)],
    textposition: "outside",
    increasing: { marker: { color: "#f59e0b" } }, decreasing: { marker: { color: "#22c55e" } }, totals: { marker: { color: "#ef4444" } },
    connector: { line: { color: c.muted } },
    hovertemplate: "%{x}: %{y:.2f}<extra></extra>",
  }], [g, c]);

  const scatter = useMemo(() => {
    const pts = Array.from(new Set(o3.patients.filter((r) => r.in_main).map((r) => r.patient)));
    const get = (pt: string, m: string) => o3.patients.find((r) => r.patient === pt && r.model === m)!.MARD;
    const x = pts.map((pt) => get(pt, "Persistence")), y = pts.map((pt) => get(pt, scatterModel));
    const lim = [Math.min(...x, ...y) - 1, Math.max(...x, ...y) + 1];
    return {
      data: [
        { type: "scatter", mode: "lines", x: lim, y: lim, line: { color: c.muted, dash: "dash" }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "markers", x, y, text: pts, showlegend: false,
          marker: { size: 11, color: y.map((v, i) => (v < x[i] ? "#22c55e" : "#ef4444")), line: { width: 1, color: c.text } },
          hovertemplate: `%{text}<br>Persistence %{x:.2f} % · ${MODEL_INFO[scatterModel].label} %{y:.2f} %<extra></extra>` },
      ],
      layout: { xaxis: { title: { text: "Persistence MARD %" }, range: lim }, yaxis: { title: { text: `${MODEL_INFO[scatterModel].label} MARD %` }, range: lim } },
      better: y.filter((v, i) => v < x[i]).length, n: pts.length,
    };
  }, [scatterModel, c]);

  return (
    <>
      <PageHeader eyebrow="Objective 3 · External transfer evaluation" title="Does it still work on a different population and sensor?">
        The OhioT1DM-trained models are applied to HUPA-UCM (Spain, FreeStyle Libre, a reading every ~15 minutes) <strong>without any retraining</strong>.
        Same causal processing; a point is scored only when real Libre readings exist at t and at t + 30 min.
      </PageHeader>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat tone="primary" label="LightGBM on HUPA-UCM" value={fmt(hL.MARD_mean)} unit="% MARD" hint="OhioT1DM: 10.03 %" />
        <Stat label="Persistence on HUPA-UCM" value={fmt(hP.MARD_mean)} unit="% MARD" tone="muted" />
        <Stat label="LightGBM better for" value={`${vsL.n_beats_persistence} / ${vsL.n_patients}`} unit="patients" hint={`p = ${fmtP(vsL.p)} — not significant`} />
        <Stat label="Evaluated points" value="99,855" hint="19 MAIN patients (ALL: 23 · 105,423)" />
      </div>

      <Section title="OhioT1DM vs HUPA-UCM, all models" subtitle="The ranking changes: the simple Ridge model transfers best; flexible models lose more.">
        <Panel className="space-y-3">
          <Segmented options={[{ value: "main", label: "MAIN · 19 patients" }, { value: "all", label: "ALL · 23 patients" }]} value={cohort} onChange={setCohort} />
          <ThemedPlot data={bars} layout={{ barmode: "group", yaxis: { title: { text: "MARD %" }, range: [8, 15] } }} height={340} />
        </Panel>
        <Panel className="p-0 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b text-left text-muted-foreground">
              {["Model", "HUPA MARD", "vs Persistence", "Patients better", "p-value", "Verdict"].map((h) => <th key={h} className="p-3 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>{o3.vs_persistence.map((r) => {
              const verdict = r.p < 0.05 ? (r.mean_diff < 0 ? "Significantly better" : "Significantly worse") : "No significant difference";
              return (
                <tr key={r.model} className={cn("border-b last:border-0", r.model === "LightGBM" && "bg-primary/5")}>
                  <td className="p-3 font-medium">{MODEL_INFO[r.model as ModelName].label}</td>
                  <td className="p-3 tabular-nums">{fmt(o3.main.find((x) => x.model === r.model)!.MARD_mean)} %</td>
                  <td className="p-3 tabular-nums">{r.mean_diff > 0 ? "+" : ""}{fmt(r.mean_diff)}</td>
                  <td className="p-3 tabular-nums">{r.n_beats_persistence} / {r.n_patients}</td>
                  <td className="p-3 tabular-nums">{fmtP(r.p)}</td>
                  <td className={cn("p-3 font-medium", verdict.includes("better") ? "text-emerald-600 dark:text-emerald-400" : verdict.includes("worse") ? "text-red-600 dark:text-red-400" : "text-muted-foreground")}>{verdict}</td>
                </tr>);
            })}</tbody>
          </table>
        </Panel>
      </Section>

      <Section title="Why accuracy drops: a measured decomposition"
        subtitle="We re-ran LOSO on OhioT1DM changing only the test data: first with the HUPA scoring rule, then thinned to one reading per 15 minutes. What remains is due to the different people and device.">
        <Panel className="space-y-3">
          <Segmented options={MODELS.map((m) => ({ value: m, label: MODEL_INFO[m].label }))} value={gapModel} onChange={setGapModel} />
          <ThemedPlot data={waterfall} layout={{ showlegend: false, yaxis: { title: { text: "MARD %" }, range: [Math.min(g.ohio_A, g.hupa_main) - 3, Math.max(g.ohio_A, g.hupa_main) + 1.5] } }} height={340} />
          <p className="text-xs text-muted-foreground">
            {MODEL_INFO[gapModel].label}: sampling effect {fmt(g.sampling_effect)} (95 % CI {fmt(g.sampling_ci_lo)}–{fmt(g.sampling_ci_hi)}),
            remaining {fmt(g.remaining_effect)} (95 % CI {fmt(g.remaining_ci_lo)}–{fmt(g.remaining_ci_hi)}), total gap {fmt(g.total_gap)} MARD points.
          </p>
        </Panel>
      </Section>

      <Section title="Patient by patient" subtitle="Each dot is one HUPA-UCM patient. Below the diagonal (green) = the model beats Persistence for that patient.">
        <Panel className="space-y-3">
          <Segmented options={MODELS.filter((m) => m !== "Persistence").map((m) => ({ value: m, label: MODEL_INFO[m].label }))} value={scatterModel} onChange={setScatterModel} />
          <ThemedPlot data={scatter.data} layout={scatter.layout} height={380} />
          <p className="text-sm text-muted-foreground">{MODEL_INFO[scatterModel].label} is better than Persistence for <strong className="text-foreground">{scatter.better} of {scatter.n}</strong> patients.</p>
        </Panel>
      </Section>

      <div className="grid md:grid-cols-2 gap-4">
        <Callout kind="finding" title="What this shows">
          The pipeline transfers without retraining to a new sensor and population; the accuracy drop is measured and about half of it is
          explained by 15-minute sampling; simpler models transfer better.
        </Callout>
        <Callout kind="caution" title="What it does not show">
          It does not establish clinical generalization, nor that LightGBM is useful on 15-minute Libre data — on HUPA-UCM it did not
          significantly outperform Persistence.
        </Callout>
      </div>

      <Section title="Original research figures" subtitle="Generated by evaluate_hupa_external.py and followup_objective3.py. Click to enlarge.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <FigureCard src="/figures/o3_ohio_vs_hupa.png" title="OhioT1DM vs HUPA-UCM" caption="MARD of each model on both datasets." />
          <FigureCard src="/figures/o3_gap.png" title="Gap decomposition" caption="Sampling vs remaining effect." />
          <FigureCard src="/figures/o3_per_patient_mard.png" title="Per-patient MARD" caption="All models on each HUPA-UCM patient." />
          <FigureCard src="/figures/o3_clarke_grid.png" title="Clarke grid (LightGBM)" caption="Clinical zones on HUPA-UCM." />
          <FigureCard src="/figures/o3_model_comparison.png" title="HUPA model comparison" caption="MARD, RMSE and Clarke A+B on HUPA-UCM." />
          <FigureCard src="/figures/o3_glucose_distribution.png" title="Glucose distribution" caption="Glucose values in HUPA-UCM." />
          <FigureCard src="/figures/o3_coverage.png" title="Sensor coverage" caption="How complete each patient's Libre data is." />
        </div>
      </Section>
    </>
  );
}
