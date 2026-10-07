"use client";
import React, { useState } from "react";
import { AlertTriangle, X } from "lucide-react";

export const DISCLAIMER =
  "Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin dosing. Does not replace a CGM.";

export function Disclaimer() {
  const [isVisible, setIsVisible] = useState(true);
  if (!isVisible) return null;
  return (
    <div className="bg-amber-500/10 text-amber-800 dark:text-amber-300 text-xs sm:text-sm px-4 py-2 flex items-center justify-center gap-2 border-b border-amber-500/20 relative">
      <AlertTriangle className="w-4 h-4 shrink-0" />
      <p className="pr-6">{DISCLAIMER}</p>
      <button onClick={() => setIsVisible(false)} aria-label="Dismiss disclaimer"
        className="absolute right-3 top-1/2 -translate-y-1/2 opacity-70 hover:opacity-100">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

export function Footer() {
  return (
    <footer className="border-t mt-16">
      <div className="container max-w-screen-xl py-8 text-xs text-muted-foreground flex flex-col sm:flex-row justify-between gap-3">
        <p>30-minute blood glucose forecasting · Group 19, PCCoE (AI &amp; ML) · OhioT1DM and HUPA-UCM datasets.</p>
        <p>{DISCLAIMER}</p>
      </div>
    </footer>
  );
}
