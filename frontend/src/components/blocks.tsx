"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Info, Lightbulb, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export const fadeIn = {
  initial: { opacity: 0, y: 14 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-40px" },
  transition: { duration: 0.4 },
};

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: React.ReactNode }) {
  return (
    <motion.div {...fadeIn} className="space-y-3 max-w-3xl">
      {eyebrow && <div className="text-xs font-semibold uppercase tracking-widest text-primary">{eyebrow}</div>}
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">{title}</h1>
      {children && <div className="text-muted-foreground leading-relaxed">{children}</div>}
    </motion.div>
  );
}

export function Section({ title, subtitle, children, className }: {
  title: string; subtitle?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <motion.section {...fadeIn} className={cn("space-y-4", className)}>
      <div>
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {subtitle && <p className="text-sm text-muted-foreground mt-1 max-w-3xl leading-relaxed">{subtitle}</p>}
      </div>
      {children}
    </motion.section>
  );
}

export function Panel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("rounded-2xl border bg-card text-card-foreground shadow-sm p-5", className)}>{children}</div>;
}

export function Stat({ label, value, unit, hint, tone = "default" }: {
  label: string; value: React.ReactNode; unit?: string; hint?: React.ReactNode; tone?: "default" | "primary" | "muted";
}) {
  return (
    <div className={cn("rounded-xl border p-4", tone === "primary" ? "bg-primary/5 border-primary/20" : "bg-card")}>
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={cn("mt-1 text-2xl font-bold tabular-nums", tone === "primary" && "text-primary",
        tone === "muted" && "text-muted-foreground")}>
        {value}{unit && <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>}
      </div>
      {hint && <div className="text-xs text-muted-foreground mt-1 leading-snug">{hint}</div>}
    </div>
  );
}

export function Callout({ kind = "info", title, children }: {
  kind?: "info" | "finding" | "caution"; title?: string; children: React.ReactNode;
}) {
  const style = {
    info: "bg-muted/40 border-border",
    finding: "bg-primary/5 border-primary/20",
    caution: "bg-amber-500/10 border-amber-500/30",
  }[kind];
  const Icon = kind === "finding" ? Lightbulb : kind === "caution" ? AlertTriangle : Info;
  return (
    <div className={cn("rounded-xl border p-4 text-sm leading-relaxed flex gap-3", style)}>
      <Icon className={cn("w-4 h-4 mt-0.5 shrink-0", kind === "caution" ? "text-amber-600 dark:text-amber-400" : "text-primary")} />
      <div>
        {title && <div className="font-semibold mb-1">{title}</div>}
        <div className="text-muted-foreground [&_strong]:text-foreground">{children}</div>
      </div>
    </div>
  );
}

export function Expand({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border bg-muted/20">
      <button onClick={() => setOpen(!open)} aria-expanded={open}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium hover:text-primary transition-colors">
        {title}
        <ChevronDown className={cn("w-4 h-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="px-4 pb-4 text-sm text-muted-foreground leading-relaxed space-y-2">{children}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange, size = "sm" }: {
  options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-lg border bg-muted/40 p-1">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={cn("rounded-md font-medium transition-colors",
            size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
            value === o.value ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function FigureCard({ src, title, caption }: { src: string; title: string; caption: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}
        className="text-left rounded-xl border bg-card overflow-hidden hover:border-primary/50 transition-colors group">
        <div className="bg-white aspect-[4/3] overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={title} loading="lazy" className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform" />
        </div>
        <div className="p-3">
          <div className="text-sm font-medium">{title}</div>
          <div className="text-xs text-muted-foreground mt-0.5 leading-snug">{caption}</div>
        </div>
      </button>
      {open && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setOpen(false)}>
          <div className="max-w-5xl w-full bg-white rounded-xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={title} className="w-full max-h-[80vh] object-contain" />
            <div className="p-3 text-sm text-slate-700 flex justify-between gap-4">
              <span><strong>{title}.</strong> {caption}</span>
              <button className="text-slate-500 hover:text-slate-900" onClick={() => setOpen(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
