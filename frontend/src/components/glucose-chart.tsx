/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, react-hooks/exhaustive-deps */
"use client";

import React, { useMemo } from "react";
import dynamic from "next/dynamic";
import { useTheme } from "@/components/theme-provider";
import { format } from "date-fns";
import { Skeleton } from "./ui/skeleton";
import { Button } from "./ui/button";
import { useState } from "react";

// Dynamically import Plotly to avoid SSR issues and reduce initial bundle size
const Plot = dynamic(() => import("react-plotly.js"), { 
  ssr: false,
  loading: () => <Skeleton className="w-full h-[400px]" />
});

export interface ChartDataPoint {
  timestamp: string;
  value: number;
}

export interface PredictionDataPoint {
  forecast_time: string;
  prediction_time: string;
  value: number;
}

interface GlucoseChartProps {
  historicalGlucose: ChartDataPoint[];
  predictions: PredictionDataPoint[];
  carbs?: ChartDataPoint[];
  bolus?: ChartDataPoint[];
  currentPredictionTime?: string;
  currentForecastTime?: string;
  height?: number;
}

export function GlucoseChart({
  historicalGlucose,
  predictions,
  carbs = [],
  bolus = [],
  currentPredictionTime,
  currentForecastTime,
  height = 400
}: GlucoseChartProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [timeWindow, setTimeWindow] = useState<"6h" | "12h" | "24h" | "all">("6h");

  const chartData = useMemo(() => {
    const data: any[] = [];

    // 1. Historical Glucose (Blue Line / Markers)
    if (historicalGlucose.length > 0) {
      data.push({
        x: historicalGlucose.map(d => d.timestamp),
        y: historicalGlucose.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Actual Glucose',
        line: { color: isDark ? '#3b82f6' : '#2563eb', width: 2 },
        marker: { size: 4 },
        hovertemplate: 'Time: %{x|%b %d, %H:%M}<br>Actual glucose: %{y} mg/dL<extra></extra>',
      });
    }

    // 2. Predictions (Orange Line / Markers)
    if (predictions.length > 0) {
      // Sort predictions by forecast time
      const sortedPreds = [...predictions].sort((a, b) => new Date(a.forecast_time).getTime() - new Date(b.forecast_time).getTime());
      
      data.push({
        x: sortedPreds.map(d => d.forecast_time),
        y: sortedPreds.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Glucose',
        line: { color: isDark ? '#f97316' : '#ea580c', width: 2, dash: 'dot' },
        marker: { size: 6, symbol: 'diamond' },
        hovertemplate: 'Forecast time: %{x|%b %d, %H:%M}<br>Predicted glucose: %{y:.1f} mg/dL<extra></extra>',
      });
    }

    // 3. Carbs (Green Triangles)
    if (carbs.length > 0) {
      data.push({
        x: carbs.map(d => d.timestamp),
        y: carbs.map(() => 70), // Draw near bottom
        type: 'scatter',
        mode: 'markers',
        name: 'Carbs (g)',
        marker: { color: '#22c55e', size: 10, symbol: 'triangle-up' },
        text: carbs.map(d => `${d.value}g`),
        hovertemplate: '%{text}<br>%{x|%b %d, %H:%M}<extra></extra>',
        yaxis: 'y2', // Use secondary axis if we want them out of the way, or just fix y to bottom
      });
    }

    // 4. Bolus (Purple Triangles)
    if (bolus.length > 0) {
      data.push({
        x: bolus.map(d => d.timestamp),
        y: bolus.map(() => 60), // Draw near bottom
        type: 'scatter',
        mode: 'markers',
        name: 'Bolus (U)',
        marker: { color: '#a855f7', size: 10, symbol: 'triangle-down' },
        text: bolus.map(d => `${d.value}U`),
        hovertemplate: '%{text}<br>%{x|%b %d, %H:%M}<extra></extra>',
        yaxis: 'y2',
      });
    }

    return data;
  }, [historicalGlucose, predictions, carbs, bolus, isDark]);

  const layout = useMemo(() => {
    const textColor = isDark ? '#e2e8f0' : '#0f172a';
    const gridColor = isDark ? '#334155' : '#e2e8f0';
    const bgColor = 'transparent';

    const shapes: any[] = [];
    const annotations: any[] = [];

    // Draw Target Range (70 - 180)
    shapes.push({
      type: 'rect',
      xref: 'paper',
      yref: 'y',
      x0: 0,
      x1: 1,
      y0: 70,
      y1: 180,
      fillcolor: isDark ? 'rgba(34, 197, 94, 0.1)' : 'rgba(34, 197, 94, 0.1)',
      line: { width: 0 },
      layer: 'below'
    });

    // Vertical line for Forecast Time
    if (currentForecastTime) {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: currentForecastTime,
        x1: currentForecastTime,
        y0: 0,
        y1: 1,
        line: {
          color: isDark ? '#f87171' : '#ef4444',
          width: 2,
          dash: 'dash',
        }
      });
      
      annotations.push({
        x: currentForecastTime,
        y: 1,
        xref: 'x',
        yref: 'paper',
        text: 'Forecast +30 min',
        showarrow: false,
        xanchor: 'left',
        yanchor: 'bottom',
        font: { size: 10, color: isDark ? '#f87171' : '#ef4444', family: 'var(--font-inter)' },
        bgcolor: isDark ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        borderpad: 2,
      });
    }

    // Calculate x-axis window (recent history from prediction time, or last data point)
    let xaxisRange: [string, string] | undefined = undefined;
    if (historicalGlucose.length > 0 && timeWindow !== "all") {
      const anchorTimeStr = currentPredictionTime || historicalGlucose[historicalGlucose.length - 1].timestamp;
      const anchorTime = new Date(anchorTimeStr).getTime();
      let hours = 6;
      if (timeWindow === "12h") hours = 12;
      if (timeWindow === "24h") hours = 24;
      
      const startTime = anchorTime - (hours * 60 * 60 * 1000);
      const endTime = currentForecastTime ? new Date(currentForecastTime).getTime() + (30 * 60 * 1000) : anchorTime + (60 * 60 * 1000); // Add a small buffer after forecast
      xaxisRange = [new Date(startTime).toISOString(), new Date(endTime).toISOString()];
    }

    return {
      autosize: true,
      height,
      margin: { t: 40, r: 20, l: 50, b: 40 },
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
      font: { color: textColor, family: 'var(--font-inter)' },
      showlegend: true,
      legend: { orientation: 'h', y: -0.2 },
      xaxis: {
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        type: 'date',
        range: xaxisRange,
      },
      yaxis: {
        title: 'Glucose (mg/dL)',
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        range: [40, 400], // Typical CGMs limit
      },
      yaxis2: {
        overlaying: 'y',
        visible: false,
        range: [0, 400], // Hidden axis just to position event markers at bottom
      },
      shapes,
      annotations,
    };
  }, [isDark, currentPredictionTime, height, timeWindow]);

  return (
    <div className="w-full rounded-xl border bg-card p-2 shadow-sm overflow-hidden relative z-0 flex flex-col">
      <div className="absolute top-4 right-4 z-10 flex gap-1 bg-background/80 backdrop-blur-sm p-1 rounded-md border shadow-sm">
        <Button variant={timeWindow === "6h" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs px-2" onClick={() => setTimeWindow("6h")}>6h</Button>
        <Button variant={timeWindow === "12h" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs px-2" onClick={() => setTimeWindow("12h")}>12h</Button>
        <Button variant={timeWindow === "24h" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs px-2" onClick={() => setTimeWindow("24h")}>24h</Button>
        <Button variant={timeWindow === "all" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs px-2" onClick={() => setTimeWindow("all")}>All</Button>
      </div>
      <Plot
        data={chartData}
        layout={layout as any}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
