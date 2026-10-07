// Error metrics, identical in definition to evaluate_loso.py (calculate_mard / rmse / mae / r2 and
// the Clarke Error Grid zone rules). Used to score forecasts against the real readings of an uploaded file.

export type Zone = "A" | "B" | "C" | "D" | "E";

/** Clarke et al. 1987 zone of one (actual, predicted) pair — same rules, same order as evaluate_loso.py. */
export function clarkeZone(act: number, pred: number): Zone {
  if ((act < 70 && pred < 70) || Math.abs(act - pred) < 0.2 * act) return "A";
  if (act <= 70 && pred >= 180) return "E";
  if (act >= 180 && pred <= 70) return "E";
  if (act >= 240 && pred >= 70 && pred <= 180) return "D";
  if (act <= 70 && pred >= 70 && pred <= 180) return "D";
  if (act >= 70 && act <= 290 && pred >= act + 110) return "C";
  if (act >= 130 && act <= 180 && pred <= (7 / 5) * act - 182) return "C";
  return "B";
}

export const ZONE_TEXT: Record<Zone, string> = {
  A: "Clinically accurate (within 20 %, or both below 70 mg/dL)",
  B: "Benign error — would not lead to a wrong treatment",
  C: "Over-correction — could lead to unnecessary treatment",
  D: "Dangerous miss — fails to detect a real high or low",
  E: "Erroneous — the opposite of the truth (low vs high)",
};

export interface Scores {
  n: number;
  mard: number;
  rmse: number;
  mae: number;
  r2: number;
  zoneA: number;
  zoneAB: number;
}

export function scores(actual: number[], pred: number[]): Scores | null {
  const n = actual.length;
  if (n === 0) return null;
  let rel = 0, sq = 0, abs = 0, a = 0, ab = 0;
  const mean = actual.reduce((s, v) => s + v, 0) / n;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const e = actual[i] - pred[i];
    rel += Math.abs(e) / Math.max(Math.abs(actual[i]), 1e-6);
    sq += e * e;
    abs += Math.abs(e);
    ssTot += (actual[i] - mean) ** 2;
    const z = clarkeZone(actual[i], pred[i]);
    if (z === "A") a++;
    if (z === "A" || z === "B") ab++;
  }
  return {
    n,
    mard: (100 * rel) / n,
    rmse: Math.sqrt(sq / n),
    mae: abs / n,
    r2: ssTot > 0 ? 1 - sq / ssTot : NaN,
    zoneA: (100 * a) / n,
    zoneAB: (100 * ab) / n,
  };
}

export interface ReadingSeries {
  t: number[]; // epoch ms, ascending
  v: number[];
}

/** Latest real reading in (time − 5 min, time] — the rule the backend uses for "current" and "actual" glucose. */
export function readingAt(series: ReadingSeries, time: number): number | null {
  const { t, v } = series;
  let lo = 0, hi = t.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= time) { idx = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (idx < 0 || t[idx] <= time - 5 * 60 * 1000) return null;
  return v[idx];
}
