"use client";

import { useMemo, useState } from "react";
import { PageHeader, Section, Panel, Callout, Segmented, Stat, FigureCard } from "@/components/blocks";
import { ThemedPlot } from "@/components/plot";
import { research, MODELS, MODEL_INFO, fmt, fmtP, ModelName } from "@/lib/research";
import { cn } from "@/lib/utils";

type Metric = "MARD" | "RMSE" | "MAE" | "R2" | "EGA_A" | "EGA_AB";
const METRICS: { value: Metric; label: string; unit: string; better: "lower" | "higher"; explain: string }[] = [
  { value: "MARD", label: "MARD", unit: "%", better: "lower", explain: "Average error as a % of the real value — the standard CGM metric." },
  { value: "RMSE", label: "RMSE", unit: "mg/dL", better: "lower", explain: "Average error that punishes big mistakes more." },
  { value: "MAE", label: "MAE", unit: "mg/dL", better: "lower", explain: "Plain average error in mg/dL." },
  { value: "R2", label: "R²", unit: "", better: "higher", explain: "How well the forecast follows the ups and downs (1 = perfect). Persistence already reaches 0.81." },
  { value: "EGA_A", label: "Clarke A", unit: "%", better: "higher", explain: "Share of forecasts in the clinically accurate zone A." },
  { value: "EGA_AB", label: "Clarke A+B", unit: "%", better: "higher", explain: "Share in the safe zones A+B. Persistence scores highest, so A+B alone cannot rank models." },
];

const summary = research.objective1.summary;
const patientRows = research.objective1.patients;
const PATIENTS = Array.from(new Set(patientRows.map((r) => r.patient)));

export default function LosoPage() {
  const [metric, setMetric] = useState<Metric>("MARD");
  const [shown, setShown] = useState<ModelName[]>(["Persistence", "Ridge", "LightGBM"]);
  const m = METRICS.find((x) => x.value === metric)!;
  const lgb = summary.find((r) => r.model === "LightGBM")!;
  const vs = research.objective1.vs_persistence;
  const pairs = research.objective1.lightgbm_pairs;

  const barData = useMemo(() => {
    const key = (metric === "EGA_AB" ? "EGA_AB_mean" : `${metric}_mean`) as keyof (typeof summary)[number];
    const sdKey = metric === "MARD" ? "MARD_std" : metric === "RMSE" ? "RMSE_std" : null;
    return [{
      type: "bar",
      x: summary.map((r) => MODEL_INFO[r.model as ModelName].label),
      y: summary.map((r) => r[key] as number),
      error_y: sdKey ? { type: "data", array: summary.map((r) => r[sdKey as keyof typeof r] as number), visible: true, thickness: 1.2 } : undefined,
      marker: { color: summary.map((r) => MODEL_INFO[r.model as ModelName].color) },
      text: summary.map((r) => fmt(r[key] as number, metric === "R2" ? 3 : 2)),
      textposition: "outside",
      hovertemplate: `%{x}<br>${m.label}: %{y:.3f} ${m.unit}<extra></extra>`,
    }];
  }, [metric, m]);

  const barLayout = useMemo(() => {
    const vals = summary.map((r) => r[(metric === "EGA_AB" ? "EGA_AB_mean" : `${metric}_mean`) as keyof (typeof summary)[number]] as number);
    const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.6 || 1;
    return { showlegend: false, yaxis: { title: { text: `${m.label} ${m.unit}` }, range: [Math.max(0, lo - pad), hi + pad] } };
  }, [metric, m]);

  const patientData = useMemo(() => shown.map((model) => {
    const rows = PATIENTS.map((p) => patientRows.find((r) => r.patient === p && r.model === model)!);
    return {
      type: "scatter", mode: "lines+markers", name: MODEL_INFO[model].label,
      x: PATIENTS.map((p) => `#${p}`), y: rows.map((r) => r.MARD),
      line: { color: MODEL_INFO[model].color, width: model === "LightGBM" ? 3 : 1.8 },
      marker: { size: model === "LightGBM" ? 9 : 6 },
      hovertemplate: `${MODEL_INFO[model].label}<br>Patient %{x}: %{y:.2f} % MARD<extra></extra>`,
    };
  }), [shown]);

  return (
    <>
      <PageHeader eyebrow="Objective 1 · Leave-one-subject-out" title="How accurate is it for a patient it has never seen?">
        12 folds: each trains all models from scratch on 11 OhioT1DM patients and tests on the 12th. Preprocessing is fitted on the 11
        only. Metrics are computed per patient, then reported as mean ± SD across the 12 patients.
      </PageHeader>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat tone="primary" label="LightGBM MARD" value={fmt(lgb.MARD_mean)} unit="%" hint={`± ${fmt(lgb.MARD_std)} across patients`} />
        <Stat label="RMSE" value={fmt(lgb.RMSE_mean)} unit="mg/dL" hint={`± ${fmt(lgb.RMSE_std)}`} />
        <Stat label="Clarke zone A" value={fmt(lgb.EGA_A_mean, 1)} unit="%" hint={`Persistence ${fmt(summary[0].EGA_A_mean, 1)} %`} />
        <Stat label="Rows evaluated" value="167,376" hint="every 5-minute point of all 12 test patients" />
      </div>

      <Section title="All six models" subtitle={m.explain}>
        <Panel className="space-y-3">
          <Segmented options={METRICS.map((x) => ({ value: x.value, label: x.label }))} value={metric} onChange={setMetric} />
          <ThemedPlot data={barData} layout={barLayout} height={340} />
          <p className="text-xs text-muted-foreground">{m.better === "lower" ? "Lower is better." : "Higher is better."} Error bars: SD across the 12 patients.</p>
        </Panel>
        <div className="grid md:grid-cols-2 gap-4">
          <Callout kind="finding" title="Every model beats the baseline for every patient">
            LightGBM improves MARD by <strong>{fmt(-vs.find((r) => r.model === "LightGBM")!.mean_diff)} points</strong> on average
            (95 % CI {fmt(-vs.find((r) => r.model === "LightGBM")!.diff_ci_hi)}–{fmt(-vs.find((r) => r.model === "LightGBM")!.diff_ci_lo)}),
            better for 12 of 12 patients (Wilcoxon p = {fmtP(vs.find((r) => r.model === "LightGBM")!.p)}).
          </Callout>
          <Callout title="Why LightGBM was deployed">
            It is <strong>tied</strong> with Stacking and MLP on MARD (LightGBM vs Stacking: {fmt(pairs[1].mean_diff, 3)} points,
            p = {fmt(pairs[1].p, 2)}) and significantly better than Ridge (p = {fmtP(pairs[0].p)}). It has the best RMSE and R² and is a
            single, fast model. It is <strong>not</strong> the single best by MARD.
          </Callout>
        </div>
      </Section>

      <Section title="Patient by patient" subtitle="Each point is one unseen test patient. Choose which models to compare.">
        <Panel className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {MODELS.map((mm) => (
              <button key={mm} onClick={() => setShown((s) => s.includes(mm) ? s.filter((x) => x !== mm) : [...s, mm])}
                className={cn("rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                  shown.includes(mm) ? "text-white" : "text-muted-foreground bg-transparent")}
                style={shown.includes(mm) ? { background: MODEL_INFO[mm].color, borderColor: MODEL_INFO[mm].color } : {}}>
                {MODEL_INFO[mm].label}
              </button>
            ))}
          </div>
          <ThemedPlot data={patientData} layout={{ yaxis: { title: { text: "MARD %" } }, xaxis: { title: { text: "Test patient" } } }} height={340} />
        </Panel>
      </Section>

      <Section title="Statistical tests vs Persistence" subtitle="Paired Wilcoxon signed-rank test across the 12 patients (each patient compared with itself).">
        <Panel className="p-0 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b text-left text-muted-foreground">
              {["Model", "Patients improved", "Mean MARD change", "95 % CI", "p-value"].map((h) => <th key={h} className="p-3 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>{vs.map((r) => (
              <tr key={r.model} className={cn("border-b last:border-0", r.model === "LightGBM" && "bg-primary/5")}>
                <td className="p-3 font-medium">{MODEL_INFO[r.model as ModelName].label}</td>
                <td className="p-3 tabular-nums">{r.n_beats_persistence} / {r.n_patients}</td>
                <td className="p-3 tabular-nums">{fmt(r.mean_diff)}</td>
                <td className="p-3 tabular-nums">{fmt(r.diff_ci_lo)} to {fmt(r.diff_ci_hi)}</td>
                <td className="p-3 tabular-nums">{fmtP(r.p)}</td>
              </tr>))}</tbody>
          </table>
        </Panel>
      </Section>

      <Section title="Original research figures" subtitle="Generated by evaluate_loso.py from the causal pipeline. Click to enlarge.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <FigureCard src="/figures/o1_model_comparison.png" title="Model comparison" caption="Mean MARD and RMSE of the six models across 12 patients." />
          <FigureCard src="/figures/o1_per_patient_mard.png" title="Per-patient MARD" caption="Every model for each held-out patient." />
          <FigureCard src="/figures/o1_actual_vs_predicted.png" title="Predicted vs real" caption="Best model's forecasts against the real glucose." />
          <FigureCard src="/figures/o1_clarke_grid.png" title="Clarke Error Grid" caption="Clinical risk zones of the forecasts." />
        </div>
      </Section>
    </>
  );
}
