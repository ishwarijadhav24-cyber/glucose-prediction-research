/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { useTheme } from "@/components/theme-provider";
import { Skeleton } from "@/components/ui/skeleton";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <Skeleton className="w-full h-[300px]" />,
});

export function usePlotColors() {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  return {
    dark,
    text: dark ? "#e2e8f0" : "#0f172a",
    muted: dark ? "#94a3b8" : "#64748b",
    grid: dark ? "#1e293b" : "#e2e8f0",
    band: dark ? "rgba(34,197,94,0.08)" : "rgba(34,197,94,0.10)",
    accent: dark ? "#60a5fa" : "#2563eb",
    warn: dark ? "#fbbf24" : "#d97706",
    red: dark ? "#f87171" : "#dc2626",
  };
}

/** Plotly chart with the site's theme applied; pass layout overrides only. */
export function ThemedPlot({ data, layout = {}, height = 320, config = {} }: {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
  config?: Record<string, any>;
}) {
  const c = usePlotColors();
  const merged = useMemo(() => ({
    autosize: true,
    height,
    margin: { t: 24, r: 16, l: 56, b: 48 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: c.text, family: "var(--font-inter), sans-serif", size: 12 },
    hoverlabel: { font: { family: "var(--font-inter), sans-serif" } },
    legend: { orientation: "h", y: -0.22, x: 0 },
    ...layout,
    xaxis: { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, ...(layout.xaxis ?? {}) },
    yaxis: { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, ...(layout.yaxis ?? {}) },
  }), [layout, height, c]);
  return (
    <Plot
      data={data}
      layout={merged as any}
      config={{ responsive: true, displayModeBar: false, ...config }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
