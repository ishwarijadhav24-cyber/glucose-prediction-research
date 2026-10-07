"use client";

import { useMemo, useState } from "react";
import { PageHeader, Section, Panel, Callout, Segmented, Stat } from "@/components/blocks";
import { ThemedPlot } from "@/components/plot";
import { research, MODELS, MODEL_INFO, fmt } from "@/lib/research";

const RANGES = ["<70", "70-180", ">180"];
const RANGE_LABEL: Record<string, string> = { "<70": "Below 70 (low)", "70-180": "70–180 (target)", ">180": "Above 180 (high)" };

export default function AnalysesPage() {
  const [ds, setDs] = useState<"Ohio LOSO" | "HUPA MAIN">("Ohio LOSO");
  const { ranges, hypo, bucketing: b } = research.analyses;

  const rangeData = useMemo(() => MODELS.map((m) => ({
    type: "bar", name: MODEL_INFO[m].label, x: RANGES.map((r) => RANGE_LABEL[r]),
    y: RANGES.map((r) => ranges.find((x) => x.dataset === ds && x.model === m && x.range === r)?.MARD ?? null),
    marker: { color: MODEL_INFO[m].color },
    hovertemplate: `${MODEL_INFO[m].label}<br>%{x}: MARD %{y:.1f} %<extra></extra>`,
  })), [ds, ranges]);

  const hypoRows = hypo.filter((h) => h.dataset === ds);
  const hypoData = useMemo(() => [
    { type: "bar", name: "Low episodes detected (0–60 min before onset)", x: hypoRows.map((h) => MODEL_INFO[h.model as keyof typeof MODEL_INFO].label),
      y: hypoRows.map((h) => h.detection_rate), marker: { color: "#94a3b8" }, hovertemplate: "%{x}: %{y:.1f} %<extra></extra>" },
    { type: "bar", name: "Detected with ≥ 15 min warning", x: hypoRows.map((h) => MODEL_INFO[h.model as keyof typeof MODEL_INFO].label),
      y: hypoRows.map((h) => h.detection_15min_rate), marker: { color: "#3b82f6" }, hovertemplate: "%{x}: %{y:.1f} %<extra></extra>" },
  ], [hypoRows]);

  const low = (m: string) => ranges.find((x) => x.dataset === "Ohio LOSO" && x.model === m && x.range === "<70")!;
  const lgbHypo = hypo.find((h) => h.dataset === "Ohio LOSO" && h.model === "LightGBM")!;

  return (
    <>
      <PageHeader eyebrow="Deeper analyses" title="Where the model works, where it fails, and what we checked">
        Averages can hide danger zones and hidden mistakes. These analyses use only the existing research predictions.
      </PageHeader>

      <Section title="Accuracy by glucose range" subtitle="The same average MARD hides very different behaviour in the low range.">
        <Panel className="space-y-3">
          <Segmented options={[{ value: "Ohio LOSO", label: "OhioT1DM (LOSO)" }, { value: "HUPA MAIN", label: "HUPA-UCM (MAIN)" }]} value={ds} onChange={setDs} />
          <ThemedPlot data={rangeData} layout={{ barmode: "group", yaxis: { title: { text: "MARD %" } } }} height={340} />
        </Panel>
        <Callout kind="caution" title="Weakest below 70 mg/dL">
          On OhioT1DM, LightGBM has MARD <strong>{fmt(low("LightGBM").MARD, 1)} %</strong> below 70 mg/dL against <strong>{fmt(low("Persistence").MARD, 1)} %</strong> for
          Persistence. Trained models pull forecasts towards typical values and under-predict how low glucose goes.
        </Callout>
      </Section>

      <Section title="Could it warn of low glucose?" subtitle="Alert = forecast below 70 mg/dL; a low episode = real glucose below 70 for at least 15 minutes. Persistence 'detects' every episode, but only at onset (0 minutes of warning).">
        <Panel className="space-y-3">
          <ThemedPlot data={hypoData} layout={{ barmode: "group", yaxis: { title: { text: "% of low episodes" }, range: [0, 105] } }} height={320} />
        </Panel>
        <Callout kind="caution" title="Not suitable for hypoglycemia alerts">
          LightGBM caught {fmt(lgbHypo.detection_rate, 1)} % of {lgbHypo.n_low_episodes} low episodes, only {fmt(lgbHypo.detection_15min_rate, 1)} % with at
          least 15 minutes of warning. The system therefore makes no low-glucose alerts.
        </Callout>
      </Section>

      <Section title="Leakage audit: event bucketing" subtitle="We searched our own research pipeline for any hidden future information.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <Stat label="Rows affected" value={`${fmt((100 * b.affected_any) / b.total_eligible, 1)} %`} hint={`${b.affected_any.toLocaleString("en-US")} of ${b.total_eligible.toLocaleString("en-US")} evaluated rows`} />
          <Stat label="By bolus / meal" value={`${b.affected_bolus.toLocaleString("en-US")} / ${b.affected_meal.toLocaleString("en-US")}`} />
          <Stat label="MARD research → causal" value={`${b.MARD_research} → ${b.MARD_causal}`} unit="%" hint="aggregate effect: negligible" tone="primary" />
          <Stat label="RMSE research → causal" value={`${b.RMSE_research} → ${b.RMSE_causal}`} unit="mg/dL" />
        </div>
        <Callout title="What it is and what we did">
          Insulin and meal events are summed into the 5-minute slot <strong>floor(event time)</strong>, so a row can contain an event up to 4:59 after its time.
          Glucose is unaffected. The aggregate effect is negligible, so the conclusions stand; it is disclosed, and the <strong>deployed backend removes every
          event after the prediction time</strong>, so live forecasts are strictly causal.
        </Callout>
      </Section>

      <Section title="Heart rate: intended, but not used">
        <Callout kind="caution">
          The feature list contains heart_rate, but the parser searched for a <code>heart_rate</code> tag while OhioT1DM stores <code>basis_heart_rate</code>
          (2018 cohort only). Heart rate was therefore missing in 100 % of rows, became a constant after imputation, and contributed nothing.
          The project makes no wearable or heart-rate claims; mapping it correctly is future work.
        </Callout>
      </Section>

      <Section title="Limitations and future work">
        <Panel className="p-0 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b text-left text-muted-foreground"><th className="p-3 font-medium">Limitation</th><th className="p-3 font-medium">Future work</th></tr></thead>
            <tbody>{[
              ["Only 12 training patients, one CGM type", "Larger, more diverse datasets"],
              ["Weak below 70 mg/dL; no hypo alerts", "Low-range weighting or a dedicated model"],
              ["No gain over Persistence on HUPA-UCM (15-min data)", "Training with 15-minute data for Libre users"],
              ["Heart rate not used (tag mapping)", "Map basis_heart_rate and test it"],
              ["Insulin feature is a simple proxy; settings not tuned", "Physiological insulin model; nested hyperparameter tuning"],
              ["Only the 30-minute horizon; no uncertainty", "More horizons and prediction intervals"],
            ].map(([l, f]) => <tr key={l} className="border-b last:border-0"><td className="p-3">{l}</td><td className="p-3 text-muted-foreground">{f}</td></tr>)}</tbody>
          </table>
        </Panel>
      </Section>
    </>
  );
}
