"use client";
import React, { useState } from "react";
import { AlertTriangle, X } from "lucide-react";

export function Disclaimer() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-destructive/10 text-destructive text-sm px-4 py-2.5 flex items-start sm:items-center justify-center gap-3 border-b border-destructive/20 w-full z-50 relative">
      <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 sm:mt-0" />
      <p className="leading-tight max-w-[80rem] pr-6">
        <strong>RESEARCH USE ONLY.</strong> This application is for demonstration and research purposes only. It is not a medical device, nor is it intended for clinical diagnosis, treatment, or medical decision making. Always consult a healthcare professional.
      </p>
      <button 
        onClick={() => setIsVisible(false)}
        className="absolute right-4 top-1/2 -translate-y-1/2 text-destructive/70 hover:text-destructive transition-colors"
        aria-label="Dismiss disclaimer"
      >
        <X className="w-5 h-5" />
      </button>
    </div>
  );
}
