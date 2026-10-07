// Reads the uploaded files in the browser, only to DRAW the real glucose line and to score the backend's
// forecasts against real readings. The backend remains the source of truth for every prediction.
// Mirrors the backend's reading rules: Libre type-0 (historic) glucose only, HUPA Preprocessed file used
// only for insulin and carbs (carb_input servings x 10 = grams, bolus <= 0 = no event), glucose clipped to 40-400.
import Papa from "papaparse";
import { XMLParser } from "fast-xml-parser";

export interface Point {
  timestamp: string; // local ISO "YYYY-MM-DDTHH:MM:SS"
  t: number; // epoch ms (local time)
  value: number;
}

export interface ParsedData {
  glucose: Point[];
  carbs: Point[];
  bolus: Point[];
  formats: string[];
}

const pad = (n: number) => String(n).padStart(2, "0");

function mk(y: number, mo: number, d: number, h: number, mi: number, s: number): Point | null {
  const date = new Date(y, mo - 1, d, h, mi, s);
  if (Number.isNaN(date.getTime())) return null;
  return {
    timestamp: `${y}-${pad(mo)}-${pad(d)}T${pad(h)}:${pad(mi)}:${pad(s)}`,
    t: date.getTime(),
    value: 0,
  };
}

/** Parses the timestamp layouts used by the supported formats (all local time, no zone). */
export function parseTime(raw: string): Point | null {
  const s = raw.trim();
  let m = s.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/); // 2027-03-01 06:00[:00], 2020/06/13 18:40
  if (m) return mk(+m[1], +m[2], +m[3], +m[4], +m[5], +(m[6] ?? 0));
  m = s.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{4}) (\d{1,2}):(\d{2})(?::(\d{2}))?/); // 13-06-2018 18:40[:00] (day first)
  if (m) return mk(+m[3], +m[2], +m[1], +m[4], +m[5], +(m[6] ?? 0));
  return null;
}

const num = (v: unknown) => {
  if (v == null || v === "") return NaN;
  return Number(String(v).replace(",", "."));
};
const clip = (v: number) => Math.min(400, Math.max(40, v));
const norm = (h: string) => h.toLowerCase().normalize("NFD").replace(/[^a-z0-9 _/()]/g, "").trim();

function push(arr: Point[], ts: string, value: number) {
  const p = parseTime(ts);
  if (p && Number.isFinite(value)) arr.push({ ...p, value });
}

function parseXml(text: string, out: ParsedData) {
  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });
  const root = parser.parse(text)?.patient;
  if (!root) return;
  const events = (node: any) => {
    const e = node?.event ?? [];
    return Array.isArray(e) ? e : [e];
  };
  events(root.glucose_level).forEach((e: any) => push(out.glucose, e["@_ts"] ?? "", clip(num(e["@_value"]))));
  events(root.bolus).forEach((e: any) => push(out.bolus, e["@_ts_begin"] ?? "", num(e["@_dose"])));
  events(root.meal).forEach((e: any) => push(out.carbs, e["@_ts"] ?? "", num(e["@_carbs"])));
  out.formats.push("OhioT1DM XML");
}

function parseCsv(text: string, out: ParsedData) {
  const lines = text.replace(/^ï»¿/, "").split(/\r?\n/);
  // Header = first line naming a time column (Libre exports start with 1-2 patient/report lines).
  const hi = lines.findIndex((l) => {
    const n = norm(l);
    return /(^|[,;\t])\s*"?(timestamp|time)"?\s*([,;\t]|$)/.test(l.toLowerCase()) || n.includes("hora") || n.includes("sello de tiempo");
  });
  if (hi < 0) return;
  const header = lines[hi];
  const delim = [";", ",", "\t"].sort((a, b) => header.split(b).length - header.split(a).length)[0];
  const parsed = Papa.parse(lines.slice(hi).join("\n"), { delimiter: delim, skipEmptyLines: true });
  const [head, ...rows] = parsed.data as string[][];
  const cols = head.map(norm);
  const find = (pred: (c: string) => boolean) => cols.findIndex(pred);

  // HUPA-UCM Preprocessed: insulin and carbs only (its glucose is interpolated and never used).
  const iBolus = find((c) => c === "bolus_volume_delivered");
  const iCarb = find((c) => c === "carb_input");
  if (iBolus >= 0 && iCarb >= 0) {
    const iTime = find((c) => c === "time");
    rows.forEach((r: string[]) => {
      const b = num(r[iBolus]), c = num(r[iCarb]);
      if (b > 0) push(out.bolus, r[iTime] ?? "", b);
      if (c > 0) push(out.carbs, r[iTime] ?? "", c * 10);
    });
    out.formats.push("HUPA-UCM Preprocessed (insulin + carbs)");
    return;
  }

  // Normalized CSV template.
  const iTs = find((c) => c === "timestamp");
  const iG = find((c) => c === "glucose_mg_dl");
  if (iTs >= 0 && iG >= 0) {
    const iB = find((c) => c === "bolus_units"), iC = find((c) => c === "carbs_g");
    rows.forEach((r: string[]) => {
      const ts = r[iTs] ?? "";
      const g = num(r[iG]), b = iB >= 0 ? num(r[iB]) : NaN, c = iC >= 0 ? num(r[iC]) : NaN;
      if (Number.isFinite(g)) push(out.glucose, ts, g);
      if (Number.isFinite(b)) push(out.bolus, ts, b);
      if (Number.isFinite(c)) push(out.carbs, ts, c);
    });
    out.formats.push("Normalized CSV");
    return;
  }

  // FreeStyle Libre export (old 'ID;Hora;...' or new 'Dispositivo;...' layout): type-0 historic glucose only.
  const iTime = find((c) => c === "hora" || c.includes("sello de tiempo"));
  const iType = find((c) => c.includes("tipo de registro"));
  const iHist = find((c) => c.includes("hist") && c.includes("gluc"));
  if (iTime >= 0 && iHist >= 0) {
    rows.forEach((r: string[]) => {
      if (iType >= 0 && String(r[iType]).trim() !== "0") return;
      const g = num(r[iHist]);
      if (Number.isFinite(g)) push(out.glucose, r[iTime] ?? "", clip(g));
    });
    out.formats.push("FreeStyle Libre (historic readings)");
  }
}

function dedupe(arr: Point[], sum: boolean): Point[] {
  const m = new Map<number, Point>();
  for (const p of arr) {
    const prev = m.get(p.t);
    m.set(p.t, prev && sum ? { ...p, value: prev.value + p.value } : p);
  }
  return Array.from(m.values()).sort((a, b) => a.t - b.t);
}

export async function parseFilesLocal(files: File[]): Promise<ParsedData> {
  const out: ParsedData = { glucose: [], carbs: [], bolus: [], formats: [] };
  for (const f of files) {
    const text = await f.text();
    try {
      if (text.trimStart().startsWith("<")) parseXml(text, out);
      else parseCsv(text, out);
    } catch {
      /* the backend reports format problems; local parsing only drives the chart */
    }
  }
  out.glucose = dedupe(out.glucose, false);
  out.bolus = dedupe(out.bolus, true);
  out.carbs = dedupe(out.carbs, true);
  return out;
}
