"use client";

import { useEffect, useMemo, useState } from "react";
import { Play, Pause } from "lucide-react";
import { PageHeader, Section, Panel, Callout, Stat } from "@/components/blocks";
import { ClarkeGrid } from "@/components/clarke-grid";
import { Button } from "@/components/ui/button";
import { research, MODELS, MODEL_INFO, fmt } from "@/lib/research";
import { clarkeZone, ZONE_TEXT } from "@/lib/metrics";
import { cn } from "@/lib/utils";

const PATIENTS = Array.from(new Set(research.objective1.patients.map((r) => r.patient)));

const DATA_ROWS = [
  ["CGM glucose", "Used", "The main input and the prediction target"],
  ["Bolus insulin", "Used", "Insulin lowers glucose over the next hours"],
  ["Meal carbohydrates", "Used", "Carbs raise glucose over the next hours (HUPA servings × 10 = grams)"],
  ["Heart rate", "Not effective", "The parser looked for 'heart_rate'; OhioT1DM stores 'basis_heart_rate' → always missing"],
  ["Basal insulin", "Not used", "Outside the bolus-driven design; future work"],
  ["Skin temp, GSR/EDA, steps, sleep, acceleration", "Not used", "Only in part of the cohort or on other devices — inputs must be identical on both datasets"],
  ["Exercise, stress, illness, finger-sticks", "Not used", "Self-reported or irregular"],
  ["HUPA 'Preprocessed' glucose", "Excluded", "Interpolated: 17.8 % of values can only be reproduced with a LATER reading (future leak)"],
];

const FEATURE_GROUPS = [
  { name: "Current values", n: 4, match: (f: string) => ["glucose", "bolus", "carbs", "heart_rate"].includes(f), why: "The present state" },
  { name: "Insulin activity", n: 1, match: (f: string) => f === "iob", why: "Decaying sum of past boluses (4 h window) — a proxy, not a pharmacokinetic IOB model" },
  { name: "Trend", n: 2, match: (f: string) => f === "glucose_diff1" || f === "glucose_diff2", why: "Rising or falling, and how fast" },
  { name: "Last 30 minutes", n: 2, match: (f: string) => f.startsWith("glucose_roll"), why: "Local level and variability" },
  { name: "Time of day", n: 2, match: (f: string) => f.startsWith("hour_"), why: "Daily patterns; 23:55 is close to 00:00" },
  { name: "Glucose history", n: 12, match: (f: string) => /^glucose_lag_\d+$/.test(f), why: "Glucose 5 … 60 minutes ago" },
  { name: "Trend history", n: 12, match: (f: string) => f.startsWith("glucose_diff1_lag"), why: "How the trend changed over the hour" },
  { name: "Insulin history", n: 12, match: (f: string) => f.startsWith("iob_lag"), why: "Timing of insulin action" },
];

function LosoPlayer() {
  const [test, setTest] = useState(0);
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTest((t) => (t + 1) % PATIENTS.length), 1800);
    return () => clearInterval(id);
  }, [playing]);
  const pid = PATIENTS[test];
  const row = (m: string) => research.objective1.patients.find((r) => r.patient === pid && r.model === m)!;
  return (
    <Panel className="space-y-5">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-medium">Fold {test + 1} of 12 — test patient <span className="text-primary">{pid}</span></div>
        <Button size="sm" variant="outline" onClick={() => setPlaying(!playing)}>
          {playing ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}{playing ? "Pause" : "Play"}
        </Button>
      </div>
      <div className="grid grid-cols-6 sm:grid-cols-12 gap-2">
        {PATIENTS.map((p, i) => (
          <button key={p} onClick={() => { setTest(i); setPlaying(false); }}
            className={cn("rounded-lg border py-3 text-xs font-semibold transition-all",
              i === test ? "bg-amber-500 text-white border-amber-500 scale-105 shadow" : "bg-primary/10 text-primary border-primary/20 hover:bg-primary/20")}>
            {p}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-primary/30" /> 11 training patients (models + preprocessing fitted only on these)</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-amber-500" /> unseen test patient</span>
      </div>
      <div className="grid sm:grid-cols-3 gap-3">
        <Stat label={`LightGBM on ${pid}`} value={fmt(row("LightGBM").MARD)} unit="% MARD" tone="primary" />
        <Stat label={`Persistence on ${pid}`} value={fmt(row("Persistence").MARD)} unit="% MARD" tone="muted" />
        <Stat label="Rows tested" value={row("LightGBM").n.toLocaleString("en-US")} hint="every 5-minute point of this patient" />
      </div>
    </Panel>
  );
}

function MetricExplainer() {
  const [real, setReal] = useState(100);
  const [pred, setPred] = useState(110);
  const err = Math.abs(real - pred);
  const zone = clarkeZone(real, pred);
  const hl = useMemo(() => ({ actual: real, predicted: pred }), [real, pred]);
  return (
    <div className="grid lg:grid-cols-[1fr_1.1fr] gap-5">
      <Panel className="space-y-5">
        {[["Real glucose", real, setReal], ["Predicted glucose", pred, setPred]].map(([label, v, set]) => (
          <label key={label as string} className="block space-y-2">
            <div className="flex justify-between text-sm"><span className="font-medium">{label as string}</span>
              <span className="tabular-nums font-semibold">{v as number} mg/dL</span></div>
            <input type="range" min={40} max={400} value={v as number} onChange={(e) => (set as (n: number) => void)(+e.target.value)}
              className="w-full accent-[hsl(var(--primary))]" />
          </label>
        ))}
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Error" value={err} unit="mg/dL" hint="contributes to MAE; squared for RMSE" />
          <Stat label="Relative error" value={fmt((100 * err) / real, 1)} unit="%" hint="averaged over all points = MARD" tone="primary" />
        </div>
        <Callout kind={zone === "A" || zone === "B" ? "finding" : "caution"} title={`Clarke zone ${zone}`}>{ZONE_TEXT[zone]}</Callout>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Try real 300 → predicted 260 (harmless, zone B) and real 60 → predicted 100 (a missed low, zone D):
          similar errors, very different clinical risk. That is why we report Clarke zones as well as MARD.
        </p>
      </Panel>
      <Panel><ClarkeGrid actual={[]} predicted={[]} highlight={hl} height={400} /></Panel>
    </div>
  );
}

export default function MethodPage() {
  const features = research.features as string[];
  return (
    <div className="container max-w-screen-xl py-12 space-y-14">
      <PageHeader eyebrow="Method" title="From raw sensor data to a tested forecast">
        Everything here is what the code actually does: which data we used and why, how the 47 features are built,
        the 6 models, how we tested them and what the error measures mean.
      </PageHeader>

      <Section title="1 · Datasets" subtitle="OhioT1DM trains the model and is used for Objectives 1–2. HUPA-UCM is only ever used for testing (Objective 3).">
        <div className="grid md:grid-cols-2 gap-4">
          <Panel className="space-y-2">
            <div className="font-semibold">OhioT1DM (USA)</div>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-5">
              <li>12 people with Type 1 diabetes, about 8 weeks each (2018 + 2020 cohorts)</li>
              <li>CGM glucose every 5 minutes, insulin pump boluses, logged meals</li>
              <li><strong className="text-foreground">169,221 rows × 47 features</strong> after processing</li>
            </ul>
          </Panel>
          <Panel className="space-y-2">
            <div className="font-semibold">HUPA-UCM (Spain)</div>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-5">
              <li>25 people; FreeStyle Libre sensor storing a reading every ~15 minutes</li>
              <li><strong className="text-foreground">19 patients (MAIN)</strong>: 2 have no CGM, 4 miss insulin/meal records (ALL = 23)</li>
              <li>99,855 evaluated points; never used for training</li>
            </ul>
          </Panel>
        </div>
        <Panel className="p-0 overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b text-left text-muted-foreground">
              <th className="p-3 font-medium">Signal</th><th className="p-3 font-medium">Status</th><th className="p-3 font-medium">Why</th></tr></thead>
            <tbody>{DATA_ROWS.map(([s, st, why]) => (
              <tr key={s} className="border-b last:border-0">
                <td className="p-3 font-medium">{s}</td>
                <td className="p-3"><span className={cn("rounded-full px-2 py-0.5 text-xs font-medium",
                  st === "Used" ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400" : "bg-muted text-muted-foreground")}>{st}</span></td>
                <td className="p-3 text-muted-foreground">{why}</td>
              </tr>))}</tbody>
          </table>
        </Panel>
      </Section>

      <Section title="2 · Causal processing" subtitle="Every value used at time t comes from time t or earlier.">
        <div className="grid md:grid-cols-3 gap-4">
          <Callout title="5-minute grid">Glucose at grid time t is the latest reading at most 5 minutes old. Insulin and carbs are summed per 5-minute slot.</Callout>
          <Callout title="Forward fill, never interpolation">Short gaps (≤ 30 min) reuse the last known value. Interpolation would use the <strong>next</strong> reading — the future — so we removed it from our first version.</Callout>
          <Callout title="Target hidden">The real value 30 minutes later is the training label and the test answer key; it is never an input.</Callout>
        </div>
      </Section>

      <Section title="3 · The 47 features" subtitle="All of them look only backwards. Hover a group to see its exact feature names.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {FEATURE_GROUPS.map((g) => (
            <div key={g.name} className="group rounded-xl border bg-card p-4" title={features.filter(g.match).join(", ")}>
              <div className="flex items-baseline justify-between"><span className="font-semibold text-sm">{g.name}</span>
                <span className="text-2xl font-bold text-primary tabular-nums">{features.filter(g.match).length}</span></div>
              <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{g.why}</p>
              <p className="text-[11px] font-mono text-muted-foreground/80 mt-2 line-clamp-2 group-hover:line-clamp-none">{features.filter(g.match).join(", ")}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">Before the model: missing values → mean of the training patients; then standard scaling fitted on the training patients only.</p>
      </Section>

      <Section title="4 · Six models" subtitle="Same 47 features, fixed settings, random seed 42.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {MODELS.map((m) => (
            <div key={m} className="rounded-xl border bg-card p-4" style={{ borderLeft: `4px solid ${MODEL_INFO[m].color}` }}>
              <div className="font-semibold">{MODEL_INFO[m].label}{m === "LightGBM" && <span className="ml-2 text-[10px] uppercase tracking-wider rounded bg-primary/10 text-primary px-1.5 py-0.5">deployed</span>}</div>
              <div className="text-sm mt-1">{MODEL_INFO[m].idea}</div>
              <div className="text-xs text-muted-foreground mt-1 leading-relaxed">{MODEL_INFO[m].detail}</div>
            </div>
          ))}
        </div>
        <Callout>XGBoost appeared in early planning but was never implemented or evaluated; LightGBM is the gradient-boosting model of this project.</Callout>
      </Section>

      <Section title="5 · How we tested: leave-one-subject-out"
        subtitle="Train on 11 patients, test on the 12th who was never seen, and repeat until every patient has been the test patient. A random split would put near-identical neighbouring rows in train and test and let the model memorise people.">
        <LosoPlayer />
      </Section>

      <Section title="6 · What the error measures mean" subtitle="We predict a number, so instead of 'accuracy' we measure how far off each forecast is — and whether the error would be dangerous.">
        <MetricExplainer />
      </Section>
    </div>
  );
}
