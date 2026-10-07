/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useMemo } from "react";
import { ThemedPlot, usePlotColors } from "@/components/plot";
import { clarkeZone, Zone } from "@/lib/metrics";

// Standard Clarke Error Grid boundaries (Clarke et al., 1987), in mg/dL.
const LINES: [number[], number[]][] = [
  [[0, 400], [0, 400]],
  [[0, 175 / 3], [70, 70]],
  [[175 / 3, 400 / 1.2], [70, 400]],
  [[70, 70], [84, 400]],
  [[0, 70], [180, 180]],
  [[70, 290], [180, 400]],
  [[70, 70], [0, 56]],
  [[70, 400], [56, 320]],
  [[180, 180], [0, 70]],
  [[180, 400], [70, 70]],
  [[240, 240], [70, 180]],
  [[240, 400], [180, 180]],
  [[130, 180], [0, 70]],
];
const LABELS: [number, number, Zone][] = [
  [30, 15, "A"], [370, 260, "B"], [280, 370, "B"], [160, 370, "C"], [160, 15, "C"],
  [30, 140, "D"], [370, 120, "D"], [30, 370, "E"], [370, 15, "E"],
];
export const ZONE_COLORS: Record<Zone, string> = { A: "#22c55e", B: "#3b82f6", C: "#f59e0b", D: "#f97316", E: "#ef4444" };

export function ClarkeGrid({ actual, predicted, height = 380, highlight }: {
  actual: number[]; predicted: number[]; height?: number; highlight?: { actual: number; predicted: number };
}) {
  const c = usePlotColors();
  const data = useMemo(() => {
    const traces: any[] = LINES.map(([x, y]) => ({
      x, y, type: "scatter", mode: "lines", hoverinfo: "skip", showlegend: false,
      line: { color: c.muted, width: 1 },
    }));
    const byZone: Record<Zone, { x: number[]; y: number[] }> = { A: { x: [], y: [] }, B: { x: [], y: [] },
      C: { x: [], y: [] }, D: { x: [], y: [] }, E: { x: [], y: [] } };
    actual.forEach((a, i) => { const z = clarkeZone(a, predicted[i]); byZone[z].x.push(a); byZone[z].y.push(predicted[i]); });
    (Object.keys(byZone) as Zone[]).forEach((z) => {
      if (!byZone[z].x.length) return;
      traces.push({
        x: byZone[z].x, y: byZone[z].y, type: actual.length > 3000 ? "scattergl" : "scatter", mode: "markers",
        name: `Zone ${z} (${((100 * byZone[z].x.length) / actual.length).toFixed(1)}%)`,
        marker: { color: ZONE_COLORS[z], size: 5, opacity: 0.55 },
        hovertemplate: `Zone ${z}<br>Real %{x:.0f} · Predicted %{y:.0f} mg/dL<extra></extra>`,
      });
    });
    if (highlight) {
      const z = clarkeZone(highlight.actual, highlight.predicted);
      traces.push({
        x: [highlight.actual], y: [highlight.predicted], type: "scatter", mode: "markers", showlegend: false,
        marker: { color: ZONE_COLORS[z], size: 16, line: { color: c.text, width: 2 } },
        hovertemplate: `Zone ${z}<br>Real %{x:.0f} · Predicted %{y:.0f}<extra></extra>`,
      });
    }
    return traces;
  }, [actual, predicted, highlight, c]);

  const layout = useMemo(() => ({
    margin: { t: 16, r: 16, l: 56, b: 52 },
    xaxis: { title: { text: "Real glucose (mg/dL)" }, range: [0, 400], dtick: 50, fixedrange: true },
    yaxis: { title: { text: "Predicted glucose (mg/dL)" }, range: [0, 400], dtick: 50, fixedrange: true },
    showlegend: actual.length > 0,
    annotations: LABELS.map(([x, y, z]) => ({ x, y, text: `<b>${z}</b>`, showarrow: false,
      font: { size: 15, color: ZONE_COLORS[z] } })),
  }), [actual.length]);

  return <ThemedPlot data={data} layout={layout} height={height} />;
}
