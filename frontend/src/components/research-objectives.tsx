/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { useTheme } from "@/components/theme-provider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Users,
  UserPlus,
  Globe,
  ArrowDown,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  Layers,
  BrainCircuit,
  Target,
  AlertTriangle,
  Info,
} from "lucide-react";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <Skeleton className="w-full h-[280px]" />,
});

/* ─────────────────────────────────────────────────────────
   STATIC RESEARCH DATA — all values from glucoce_full.pdf
   ───────────────────────────────────────────────────────── */

const OBJ1 = {
  mard: { mean: 10.03, sd: 1.52 },
  rmse: { mean: 20.33, sd: 2.66 },
  mae: 14.24,
  r2: 0.875,
  clarkeAB: 98.0,
  persistenceMARD: { mean: 12.11, sd: 1.92 },
  mardDiff: -2.08,
  allPatientsBeaten: "12 of 12",
  wilcoxonP: 0.0005,
};

const OBJ2_PERSONALIZATION = [
  { pct: "0%", days: 0, mard: 9.83 },
  { pct: "5%", days: 2.4, mard: 9.92 },
  { pct: "10%", days: 4.8, mard: 9.77 },
  { pct: "20%", days: 9.6, mard: 9.66 },
  { pct: "30%", days: 14.4, mard: 9.67 },
  { pct: "50%", days: 24.0, mard: 9.63 },
];

const OBJ3 = {
  ohioMARD: 10.03,
  hupaMARD: 13.38,
  hupaRMSE: 21.95,
  hupaClarkeAB: 95.7,
  mardGap: 3.35,
  hupaPersistenceMARD: 13.03,
  hupaBetterPatients: "10 of 19",
  hupaWilcoxonP: 0.95,
  samplingGap: 1.59,
  remainingGap: 1.81,
  hupaPatients: 19,
  hupaPoints: "99,855",
};

/* ─────────────────────────────────────────────────────────
   Shared sub-components
   ───────────────────────────────────────────────────────── */

function MetricRow({ label, value, unit, muted }: { label: string; value: string; unit?: string; muted?: boolean }) {
  return (
    <div className={`flex justify-between items-baseline py-1.5 border-b border-border/50 last:border-b-0 ${muted ? "opacity-70" : ""}`}>
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-semibold tabular-nums">
        {value}
        {unit && <span className="text-muted-foreground font-normal ml-0.5">{unit}</span>}
      </span>
    </div>
  );
}

function ExpandableNote({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3 rounded-lg border border-border/60 bg-muted/30">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
        aria-expanded={open}
      >
        <span className="flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5" />
          {title}
        </span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {open && (
        <div className="px-3 pb-3 text-xs text-muted-foreground leading-relaxed border-t border-border/40 pt-2">
          {children}
        </div>
      )}
    </div>
  );
}

const sectionFade = {
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-60px" },
  transition: { duration: 0.45 },
};

/* ─────────────────────────────────────────────────────────
   Objective 1 — Subject-Independent Accuracy
   ───────────────────────────────────────────────────────── */

function Objective1() {
  return (
    <motion.div {...sectionFade} className="rounded-2xl border bg-card text-card-foreground shadow-sm flex flex-col h-full">
      <div className="p-6 pb-4 flex flex-col gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="p-2.5 bg-primary/10 rounded-xl">
            <Users className="h-6 w-6 text-primary" />
          </div>
          <Badge variant="secondary" className="text-[10px] tracking-wide uppercase shrink-0">
            Objective 01 · LOSO
          </Badge>
        </div>
        <h3 className="text-lg font-semibold leading-tight">Subject-Independent Accuracy</h3>
        <p className="text-sm text-muted-foreground leading-relaxed italic">
          &ldquo;How accurately can the model predict glucose for a patient it has never seen before?&rdquo;
        </p>
      </div>

      <div className="px-6 pb-2">
        <p className="text-xs text-muted-foreground leading-relaxed">
          Leave-One-Subject-Out (LOSO) cross-validation on 12 OhioT1DM patients.
          Each fold trains on 11 patients and evaluates on the held-out 12th — the realistic unseen-patient scenario
          that avoids leakage from highly similar neighboring 5-minute rows.
        </p>
      </div>

      <div className="flex-1 px-6 pb-4">
        {/* Primary metric */}
        <div className="mt-3 mb-2 p-3 rounded-xl bg-primary/5 border border-primary/10 text-center">
          <div className="text-3xl font-bold tabular-nums text-primary">
            {OBJ1.mard.mean}<span className="text-base font-normal ml-0.5">%</span>
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">
            MARD ± {OBJ1.mard.sd}% <span className="opacity-60">(mean ± SD across folds)</span>
          </div>
        </div>

        <div className="space-y-0">
          <MetricRow label="RMSE" value={`${OBJ1.rmse.mean} ± ${OBJ1.rmse.sd}`} unit="mg/dL" />
          <MetricRow label="MAE" value={`${OBJ1.mae}`} unit="mg/dL" />
          <MetricRow label="R²" value={`${OBJ1.r2}`} />
          <MetricRow label="Clarke A+B" value={`${OBJ1.clarkeAB}`} unit="%" />
        </div>

        {/* vs Persistence */}
        <div className="mt-4 p-3 rounded-lg border border-border/60 bg-muted/20">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">vs. Persistence Baseline</div>
          <MetricRow label="Persistence MARD" value={`${OBJ1.persistenceMARD.mean} ± ${OBJ1.persistenceMARD.sd}`} unit="%" muted />
          <MetricRow label="LightGBM advantage" value={`${OBJ1.mardDiff}`} unit="MARD pts" />
          <MetricRow label="Patients improved" value={OBJ1.allPatientsBeaten} />
          <MetricRow label="Wilcoxon p-value" value={`${OBJ1.wilcoxonP}`} />
        </div>
      </div>

      <div className="px-6 pb-6">
        <ExpandableNote title="Model selection rationale">
          LightGBM, Stacking, and MLP are statistically tied on MARD (LightGBM vs Stacking difference = 0.004 points, p&nbsp;=&nbsp;0.97).
          LightGBM was selected for deployment because it matched the leading models on MARD while providing the best RMSE/R² and a simpler single-model deployment.
        </ExpandableNote>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────
   Objective 2 — Personalization
   ───────────────────────────────────────────────────────── */

function Objective2() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const chartData = useMemo(() => {
    const days = OBJ2_PERSONALIZATION.map((d) => d.days);
    const mards = OBJ2_PERSONALIZATION.map((d) => d.mard);

    return [
      {
        x: days,
        y: mards,
        type: "scatter" as const,
        mode: "lines+markers" as const,
        name: "LightGBM MARD",
        line: { color: isDark ? "#60a5fa" : "#2563eb", width: 2.5, shape: "spline" as const },
        marker: { size: 7, color: isDark ? "#60a5fa" : "#2563eb" },
        hovertemplate: "%{x:.1f} days<br>MARD: %{y:.2f}%<extra></extra>",
      },
      // Highlight the ~9.6 day inflection
      {
        x: [9.6],
        y: [9.66],
        type: "scatter" as const,
        mode: "markers" as const,
        name: "~10 days threshold",
        marker: { size: 12, color: isDark ? "#f59e0b" : "#d97706", symbol: "circle-open", line: { width: 2.5 } },
        hovertemplate: "~10 days of personal data<br>MARD: 9.66%<extra></extra>",
        showlegend: false,
      },
    ];
  }, [isDark]);

  const chartLayout = useMemo(() => {
    const textColor = isDark ? "#e2e8f0" : "#0f172a";
    const gridColor = isDark ? "#334155" : "#e2e8f0";
    return {
      autosize: true,
      height: 240,
      margin: { t: 20, r: 16, l: 48, b: 44 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: textColor, family: "var(--font-inter)", size: 11 },
      showlegend: false,
      xaxis: {
        title: { text: "Personal data (days)", font: { size: 11 } },
        gridcolor: gridColor,
        dtick: 5,
        range: [-1, 26],
      },
      yaxis: {
        title: { text: "MARD (%)", font: { size: 11 } },
        gridcolor: gridColor,
        range: [9.45, 10.05],
        dtick: 0.1,
      },
      annotations: [
        {
          x: 9.6,
          y: 9.66,
          xref: "x",
          yref: "y",
          text: "~10 days",
          showarrow: true,
          arrowhead: 0,
          arrowcolor: isDark ? "#f59e0b" : "#d97706",
          ax: 30,
          ay: -28,
          font: { size: 10, color: isDark ? "#f59e0b" : "#d97706" },
        },
      ],
    };
  }, [isDark]);

  return (
    <motion.div {...sectionFade} className="rounded-2xl border bg-card text-card-foreground shadow-sm flex flex-col h-full">
      <div className="p-6 pb-4 flex flex-col gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="p-2.5 bg-primary/10 rounded-xl">
            <UserPlus className="h-6 w-6 text-primary" />
          </div>
          <Badge variant="secondary" className="text-[10px] tracking-wide uppercase shrink-0">
            Objective 02 · Patient Adaptation
          </Badge>
        </div>
        <h3 className="text-lg font-semibold leading-tight">Personalization</h3>
        <p className="text-sm text-muted-foreground leading-relaxed italic">
          &ldquo;Does adding a patient&apos;s own earlier data to the population model improve forecasts of that patient&apos;s later data?&rdquo;
        </p>
      </div>

      <div className="px-6 pb-2">
        <p className="text-xs text-muted-foreground leading-relaxed">
          For each patient, the population model is trained on the other 11 patients.
          The last 50% of that patient&apos;s data is the fixed test set; personal data comes only from the earlier portion
          at P&nbsp;=&nbsp;0%, 5%, 10%, 20%, 30%, 50%.
        </p>
      </div>

      {/* Chart */}
      <div className="px-4 pb-2">
        <div className="rounded-xl border bg-background/50 overflow-hidden">
          <Plot
            data={chartData as any}
            layout={chartLayout as any}
            config={{ responsive: true, displayModeBar: false, staticPlot: false }}
            style={{ width: "100%", height: 240 }}
            useResizeHandler
          />
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 px-6 pb-4">
        <div className="overflow-x-auto mt-2">
          <table className="w-full text-xs" role="table" aria-label="Personalization results">
            <thead>
              <tr className="border-b border-border/60">
                <th className="text-left py-1.5 text-muted-foreground font-medium">Personal data</th>
                <th className="text-right py-1.5 text-muted-foreground font-medium">≈ Days</th>
                <th className="text-right py-1.5 text-muted-foreground font-medium">MARD</th>
              </tr>
            </thead>
            <tbody>
              {OBJ2_PERSONALIZATION.map((row) => (
                <tr key={row.pct} className="border-b border-border/30 last:border-b-0">
                  <td className="py-1.5 font-medium">{row.pct}</td>
                  <td className="py-1.5 text-right tabular-nums">{row.days}</td>
                  <td className="py-1.5 text-right tabular-nums font-semibold">{row.mard}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Result */}
        <div className="mt-3 p-3 rounded-lg border border-border/60 bg-muted/20">
          <p className="text-xs text-muted-foreground leading-relaxed">
            Personalization gives a <strong className="text-foreground">small but consistent improvement</strong> of about 2% relative,
            or about 0.2 MARD points, once roughly 10 days of the patient&apos;s data are available.
            A few days of data do not help.
          </p>
        </div>
      </div>

      <div className="px-6 pb-6">
        <ExpandableNote title="Methodology note">
          The LightGBM adaptation recipe changes between P&nbsp;≤&nbsp;10% and P&nbsp;≥&nbsp;20%, so the curve
          partly reflects a change in the adaptation approach rather than a pure data-quantity effect.
          The personalization baseline (P&nbsp;=&nbsp;0%) MARD of 9.83% differs from the Objective 1 LOSO MARD
          of 10.03% because the two experiments use different test splits.
        </ExpandableNote>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────
   Objective 3 — External Transfer Evaluation
   ───────────────────────────────────────────────────────── */

function Objective3() {
  return (
    <motion.div {...sectionFade} className="rounded-2xl border bg-card text-card-foreground shadow-sm flex flex-col h-full">
      <div className="p-6 pb-4 flex flex-col gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="p-2.5 bg-primary/10 rounded-xl">
            <Globe className="h-6 w-6 text-primary" />
          </div>
          <Badge variant="secondary" className="text-[10px] tracking-wide uppercase shrink-0">
            Objective 03 · HUPA-UCM
          </Badge>
        </div>
        <h3 className="text-lg font-semibold leading-tight">External Transfer Evaluation</h3>
        <p className="text-sm text-muted-foreground leading-relaxed italic">
          &ldquo;Can the OhioT1DM-trained models be applied to a different dataset without retraining?&rdquo;
        </p>
      </div>

      <div className="px-6 pb-2">
        <p className="text-xs text-muted-foreground leading-relaxed">
          Models trained on OhioT1DM are applied to HUPA-UCM (FreeStyle Libre, ~15-min readings) <strong className="text-foreground">without retraining</strong>.
          The 5-minute grid is forward-filled, never interpolated. A point is scored only when real Libre readings exist at t and t+30.
          <span className="opacity-70"> {OBJ3.hupaPatients} patients · {OBJ3.hupaPoints} evaluated points.</span>
        </p>
      </div>

      <div className="flex-1 px-6 pb-4">
        {/* Transfer gap visual */}
        <div className="mt-2 flex flex-col items-center gap-0">
          {/* OhioT1DM box */}
          <div className="w-full p-3 rounded-xl bg-primary/5 border border-primary/10 text-center">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-0.5">OhioT1DM · LOSO</div>
            <div className="text-2xl font-bold tabular-nums text-primary">
              {OBJ3.ohioMARD}<span className="text-sm font-normal ml-0.5">% MARD</span>
            </div>
          </div>

          {/* Arrow with "no retraining" */}
          <div className="flex flex-col items-center py-1.5">
            <ArrowDown className="w-4 h-4 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium my-0.5">No retraining</span>
            <ArrowDown className="w-4 h-4 text-muted-foreground" />
          </div>

          {/* HUPA-UCM box */}
          <div className="w-full p-3 rounded-xl bg-muted/40 border border-border text-center">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-0.5">HUPA-UCM · FreeStyle Libre</div>
            <div className="text-2xl font-bold tabular-nums text-foreground">
              {OBJ3.hupaMARD}<span className="text-sm font-normal ml-0.5">% MARD</span>
            </div>
          </div>

          {/* Gap label */}
          <div className="mt-2 px-3 py-1.5 rounded-full bg-muted/50 border border-border/60">
            <span className="text-xs font-semibold tabular-nums text-muted-foreground">+{OBJ3.mardGap} MARD points</span>
          </div>
        </div>

        {/* Additional metrics */}
        <div className="mt-4 space-y-0">
          <MetricRow label="HUPA-UCM RMSE" value={`${OBJ3.hupaRMSE}`} unit="mg/dL" />
          <MetricRow label="HUPA-UCM Clarke A+B" value={`${OBJ3.hupaClarkeAB}`} unit="%" />
        </div>

        {/* vs Persistence on HUPA */}
        <div className="mt-3 p-3 rounded-lg border border-border/60 bg-muted/20">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">vs. Persistence on HUPA-UCM</div>
          <MetricRow label="LightGBM MARD" value={`${OBJ3.hupaMARD}`} unit="%" />
          <MetricRow label="Persistence MARD" value={`${OBJ3.hupaPersistenceMARD}`} unit="%" muted />
          <MetricRow label="Patients improved" value={OBJ3.hupaBetterPatients} />
          <MetricRow label="Wilcoxon p-value" value={`${OBJ3.hupaWilcoxonP}`} />
          <p className="text-[11px] text-muted-foreground mt-2 leading-relaxed">
            LightGBM did <strong className="text-foreground">not</strong> significantly outperform Persistence on HUPA-UCM.
          </p>
        </div>

        {/* Interpretation */}
        <div className="mt-3 p-3 rounded-lg border border-border/60 bg-muted/20">
          <p className="text-xs text-muted-foreground leading-relaxed">
            Performance decreased on the external dataset. The evaluation demonstrates <strong className="text-foreground">transferability of the pipeline</strong>,
            but does not establish clinical generalization or superiority on a new population.
          </p>
          <p className="text-xs text-muted-foreground leading-relaxed mt-2">
            About {OBJ3.samplingGap} MARD points of the gap are associated with the 15-minute sampling difference;
            about {OBJ3.remainingGap} points remain attributable to population/device/dataset differences.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────
   Research Pipeline Visual
   ───────────────────────────────────────────────────────── */

function ResearchPipeline() {
  const steps = [
    { icon: <Layers className="w-5 h-5" />, label: "Past CGM glucose + Insulin boluses + Carbohydrate intake" },
    { icon: <Target className="w-5 h-5" />, label: "5-minute research grid" },
    { icon: <FlaskConical className="w-5 h-5" />, label: "47 causal features" },
    { icon: <BrainCircuit className="w-5 h-5" />, label: "LightGBM" },
    { icon: <Target className="w-5 h-5" />, label: "30-minute glucose forecast" },
  ];

  return (
    <motion.div {...sectionFade} className="w-full">
      <div className="max-w-2xl mx-auto">
        <div className="flex flex-col items-center gap-0">
          {steps.map((step, i) => (
            <React.Fragment key={i}>
              <div className="flex items-center gap-3 p-3 px-5 rounded-xl border bg-card text-card-foreground shadow-sm w-full max-w-md">
                <div className="p-2 bg-primary/10 rounded-lg text-primary shrink-0">{step.icon}</div>
                <span className="text-sm font-medium">{step.label}</span>
              </div>
              {i < steps.length - 1 && (
                <div className="py-1">
                  <ArrowDown className="w-4 h-4 text-muted-foreground" />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
        <p className="text-center text-xs text-muted-foreground mt-4 max-w-md mx-auto leading-relaxed">
          Features are constructed from information available at the prediction time.
          The deployed backend enforces strict causal prediction.
        </p>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────
   Limitations & Research Notes
   ───────────────────────────────────────────────────────── */

function ResearchLimitations() {
  return (
    <motion.div {...sectionFade} className="max-w-3xl mx-auto w-full">
      <div className="rounded-xl border border-border/60 bg-muted/20 p-5">
        <div className="flex items-start gap-3 mb-3">
          <AlertTriangle className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
          <h4 className="text-sm font-semibold text-foreground">Known Limitations &amp; Research Notes</h4>
        </div>
        <ul className="space-y-2 text-xs text-muted-foreground leading-relaxed ml-7">
          <li>
            <strong className="text-foreground">Low-glucose performance:</strong> LightGBM MARD was 33.1% below 70 mg/dL versus 24.0% for Persistence.
            Not suitable for hypoglycemia alerts.
          </li>
          <li>
            <strong className="text-foreground">Heart rate:</strong> The heart-rate signal was intended but was effectively missing and contributed nothing to predictions.
          </li>
          <li>
            <strong className="text-foreground">IOB proxy:</strong> The insulin-on-board feature is an exponentially decaying sum of past boluses — a proxy, not a physiological pharmacokinetic model.
          </li>
          <li>
            <strong className="text-foreground">Evaluation scope:</strong> Only the 30-minute forecast horizon was evaluated. 12 OhioT1DM patients, 47 features, 5-minute research grid.
          </li>
          <li>
            <strong className="text-foreground">Research audit:</strong> A small research event-bucketing look-ahead affected about 2.3% of rows in the research evaluation.
            The deployed backend is strictly causal.
          </li>
        </ul>
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────
   MAIN EXPORT — Research Objectives Section
   ───────────────────────────────────────────────────────── */

export function ResearchObjectives() {
  return (
    <section className="py-24 border-t" id="research">
      <div className="container max-w-screen-xl">

        {/* Section Hero */}
        <motion.div {...sectionFade} className="text-center mb-8">
          <Badge variant="outline" className="mb-4">
            <FlaskConical className="w-3 h-3 mr-1" />
            Research Evidence
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
            Three Questions. One Research Framework.
          </h2>
          <p className="text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Beyond a single forecast, the study evaluates whether glucose forecasting works for unseen patients,
            whether personal history helps, and how the pipeline transfers to an independent dataset.
          </p>
        </motion.div>

        {/* Research contribution statement */}
        <motion.div {...sectionFade} className="max-w-3xl mx-auto mb-16">
          <p className="text-center text-xs text-muted-foreground leading-relaxed italic px-4">
            &ldquo;The strength of this project is the combination of subject-independent evaluation, a causal no-future-information design,
            external-dataset transfer evaluation, personalization analysis, and reproducible deployment —
            not a claim of clinical readiness or state-of-the-art accuracy.&rdquo;
          </p>
        </motion.div>

        {/* Three objective cards */}
        <div className="grid lg:grid-cols-3 md:grid-cols-2 gap-6 mb-20">
          <Objective1 />
          <Objective2 />
          <Objective3 />
        </div>

        {/* Research Pipeline */}
        <motion.div {...sectionFade} className="text-center mb-8">
          <h3 className="text-2xl font-bold tracking-tight mb-3">
            From Observations to a 30-Minute Forecast
          </h3>
          <p className="text-muted-foreground text-sm max-w-xl mx-auto">
            The causal pipeline transforms raw CGM, insulin, and carbohydrate data into a structured prediction.
          </p>
        </motion.div>
        <div className="mb-20">
          <ResearchPipeline />
        </div>

        {/* Limitations */}
        <ResearchLimitations />
      </div>
    </section>
  );
}
