"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Users, UserPlus, Globe, ShieldCheck, Clock, Layers, BrainCircuit, LineChart, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { fadeIn, Stat } from "@/components/blocks";
import { research, fmt } from "@/lib/research";

const o1 = research.objective1.summary;
const lgb = o1.find((m) => m.model === "LightGBM")!;
const per = o1.find((m) => m.model === "Persistence")!;
const vsP = research.objective1.vs_persistence.find((m) => m.model === "LightGBM")!;
const o2 = research.objective2.summary.filter((r) => r.model === "LightGBM");
const p0 = o2.find((r) => r.p_level === 0)!;
const p50 = o2.find((r) => r.p_level === 50)!;
const hupaL = research.objective3.main.find((m) => m.model === "LightGBM")!;
const hupaR = research.objective3.main.find((m) => m.model === "Ridge")!;
const gapL = research.objective3.gap.find((g) => g.model === "LightGBM")!;

const OBJECTIVES = [
  {
    href: "/research/loso", icon: Users, tag: "Objective 1 · LOSO", title: "Does it work for a new patient?",
    text: "Train on 11 OhioT1DM patients, test on the 12th the model has never seen — 12 times.",
    result: `MARD ${fmt(lgb.MARD_mean)} % vs ${fmt(per.MARD_mean)} % for the no-change baseline · better for ${vsP.n_beats_persistence}/${vsP.n_patients} patients`,
  },
  {
    href: "/research/personalization", icon: UserPlus, tag: "Objective 2 · Personalization", title: "Does the patient's own data help?",
    text: "Adapt the population model with 5–50 % of a patient's own earlier data; test on their later data.",
    result: `MARD ${fmt(p0.MARD_mean)} % → ${fmt(p50.MARD_mean)} % — small but consistent after ~10 days`,
  },
  {
    href: "/research/external", icon: Globe, tag: "Objective 3 · HUPA-UCM", title: "Does it transfer to another dataset?",
    text: "Apply the OhioT1DM model to a Spanish cohort with a different sensor, without retraining.",
    result: `MARD ${fmt(hupaL.MARD_mean)} % · ${fmt(gapL.sampling_effect)} of the ${fmt(gapL.total_gap)}-point drop is 15-min sampling · Ridge transfers best (${fmt(hupaR.MARD_mean)} %)`,
  },
];

const PIPELINE = [
  { icon: Layers, title: "Past data only", text: "CGM glucose, insulin boluses and carbs up to time t" },
  { icon: Clock, title: "5-minute grid", text: "Causal alignment; short gaps forward-filled, never interpolated" },
  { icon: FlaskConical, title: "47 features", text: "Level, trend, last hour of history, insulin activity, time of day" },
  { icon: BrainCircuit, title: "LightGBM", text: "150 boosted trees trained on all 12 OhioT1DM patients" },
  { icon: LineChart, title: "Forecast t + 30 min", text: "One direct forecast, never recursive" },
];

const RIGOUR = [
  { title: "Unseen patients", text: "Leave-one-subject-out: every test patient is new to the model, as in real use." },
  { title: "No future data", text: "We found and removed a look-ahead in our first pipeline; tests prove later data cannot change a forecast." },
  { title: "A second dataset", text: "Zero-retraining transfer to HUPA-UCM, with a measured explanation of the accuracy drop." },
  { title: "Honest limits", text: "We report where it fails: below 70 mg/dL, and no gain over the baseline on HUPA-UCM." },
];

export default function Home() {
  return (
    <div className="flex-1">
      <section className="relative overflow-hidden border-b">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/15 via-transparent to-transparent" />
        <div className="container max-w-screen-xl py-20 md:py-28 grid lg:grid-cols-[1.2fr_1fr] gap-12 items-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border bg-background/60 px-3 py-1 text-xs font-medium">
              <ShieldCheck className="w-3.5 h-3.5 text-primary" /> Type 1 diabetes · research project
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold tracking-tight leading-[1.1]">
              Forecasting glucose <span className="text-primary">30 minutes ahead</span> — tested on people the model has never seen
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed max-w-2xl">
              A CGM shows glucose <em>now</em>. We forecast it half an hour ahead from past glucose, insulin and meals,
              and ask the question most studies skip: does it still work for a new patient, and on a different dataset?
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Button size="lg" className="h-11 group" asChild>
                <Link href="/lab">Open the Forecast Lab <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" /></Link>
              </Button>
              <Button size="lg" variant="outline" className="h-11" asChild>
                <Link href="/research/loso">Explore the research</Link>
              </Button>
            </div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
            className="grid grid-cols-2 gap-3">
            <Stat tone="primary" label="Average error (MARD)" value={fmt(lgb.MARD_mean)} unit="%" hint="on unseen patients" />
            <Stat label="vs no-change baseline" value={`−${fmt(-vsP.mean_diff)}`} unit="pts" hint={`better for ${vsP.n_beats_persistence} of ${vsP.n_patients} patients`} />
            <Stat label="Typical error" value={fmt(lgb.MAE_mean, 1)} unit="mg/dL" hint={`RMSE ${fmt(lgb.RMSE_mean)} mg/dL`} />
            <Stat label="Clinically safe" value={fmt(lgb.EGA_AB_mean, 1)} unit="%" hint={`Clarke zones A+B · ${fmt(lgb.EGA_A_mean, 1)} % in zone A`} />
            <Stat label="Models compared" value="6" hint="baseline → linear → trees → neural net → ensemble" />
            <Stat label="Patients" value="12 + 19" hint="OhioT1DM (USA) + HUPA-UCM (Spain)" />
          </motion.div>
        </div>
      </section>

      <section className="container max-w-screen-xl py-16 space-y-8">
        <motion.div {...fadeIn} className="max-w-2xl">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Three research questions</h2>
          <p className="text-muted-foreground mt-2">Each objective has its own interactive page with the full results.</p>
        </motion.div>
        <div className="grid md:grid-cols-3 gap-5">
          {OBJECTIVES.map((o, i) => (
            <motion.div key={o.href} {...fadeIn} transition={{ duration: 0.4, delay: i * 0.08 }}>
              <Link href={o.href} className="group h-full flex flex-col rounded-2xl border bg-card p-6 hover:border-primary/50 hover:shadow-md transition-all">
                <div className="flex items-center justify-between">
                  <span className="grid place-items-center w-10 h-10 rounded-xl bg-primary/10"><o.icon className="w-5 h-5 text-primary" /></span>
                  <span className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">{o.tag}</span>
                </div>
                <h3 className="mt-4 text-lg font-semibold">{o.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed flex-1">{o.text}</p>
                <div className="mt-4 rounded-lg bg-muted/50 p-3 text-xs leading-relaxed"><strong>Result:</strong> {o.result}</div>
                <div className="mt-4 text-sm font-medium text-primary flex items-center gap-1">
                  See the results <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="border-y bg-muted/30">
        <div className="container max-w-screen-xl py-16 space-y-8">
          <motion.div {...fadeIn} className="max-w-2xl">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">How one forecast is made</h2>
            <p className="text-muted-foreground mt-2">The same pipeline is used in the research and in the live backend — the deployed predictions are bit-identical to the research code.</p>
          </motion.div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {PIPELINE.map((s, i) => (
              <motion.div key={s.title} {...fadeIn} transition={{ duration: 0.4, delay: i * 0.06 }}
                className="relative rounded-xl border bg-card p-4">
                <div className="text-[11px] font-semibold text-muted-foreground">STEP {i + 1}</div>
                <s.icon className="w-5 h-5 text-primary mt-2" />
                <div className="font-semibold mt-2 text-sm">{s.title}</div>
                <div className="text-xs text-muted-foreground mt-1 leading-relaxed">{s.text}</div>
              </motion.div>
            ))}
          </div>
          <div className="text-center"><Button variant="outline" asChild><Link href="/method">Read the full method</Link></Button></div>
        </div>
      </section>

      <section className="container max-w-screen-xl py-16 grid lg:grid-cols-[1fr_1.3fr] gap-10 items-start">
        <motion.div {...fadeIn} className="space-y-3">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">What makes it trustworthy</h2>
          <p className="text-muted-foreground leading-relaxed">
            Our contribution is not a record-breaking model. It is a careful, leak-free evaluation on unseen patients and on a
            second dataset, a personalization analysis, and a reproducible deployed system.
          </p>
        </motion.div>
        <div className="grid sm:grid-cols-2 gap-4">
          {RIGOUR.map((r, i) => (
            <motion.div key={r.title} {...fadeIn} transition={{ duration: 0.4, delay: i * 0.06 }} className="rounded-xl border bg-card p-5">
              <div className="font-semibold">{r.title}</div>
              <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{r.text}</p>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
}
