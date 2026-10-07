/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import Papa from "papaparse";
import { XMLParser } from "fast-xml-parser";

export interface ParsedData {
  glucose: { timestamp: string; value: number }[];
  carbs: { timestamp: string; value: number }[];
  bolus: { timestamp: string; value: number }[];
}

export async function parseFileLocal(file: File): Promise<ParsedData> {
  const text = await file.text();
  const ext = file.name.split('.').pop()?.toLowerCase();

  const data: ParsedData = {
    glucose: [],
    carbs: [],
    bolus: [],
  };

  if (ext === "csv") {
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          results.data.forEach((row: any) => {
            // Flexible column matching based on common CGM formats
            const ts = row.timestamp || row.Time || row.Date_Time || row['Device Timestamp'];
            const gl = row.glucose || row.glucose_mg_dl || row['Historic Glucose mg/dL'] || row.Glucose;
            const cb = row.carbs || row.carbs_g || row['Carbohydrates (grams)'];
            const bl = row.bolus || row.bolus_units || row['Bolus Volume (U)'];

            if (ts && gl && !isNaN(Number(gl))) {
              data.glucose.push({ timestamp: ts, value: Number(gl) });
            }
            if (ts && cb && !isNaN(Number(cb))) {
              data.carbs.push({ timestamp: ts, value: Number(cb) });
            }
            if (ts && bl && !isNaN(Number(bl))) {
              data.bolus.push({ timestamp: ts, value: Number(bl) });
            }
          });
          resolve(data);
        },
        error: (error: any) => {
          reject(error);
        }
      });
    });
  } else if (ext === "xml") {
    // Basic XML Parsing (e.g. Dexcom or generic XML)
    try {
      const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });
      const parsed = parser.parse(text);
      
      // Function to parse OhioT1DM timestamp: dd-mm-yyyy HH:MM:SS
      const parseOhioTime = (ts: string) => {
        if (!ts) return null;
        const match = ts.match(/^(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})/);
        if (match) {
          const [_, d, m, y, h, min, s] = match;
          return `${y}-${m}-${d}T${h}:${min}:${s}`;
        }
        return ts; // Fallback
      };

      // Attempt to find glucose events (generic or OhioT1DM)
      const genericEvents = parsed?.Patient?.GlucoseReadings?.Reading || parsed?.Events?.Event || [];
      const genericEventArray = Array.isArray(genericEvents) ? genericEvents : [genericEvents];

      genericEventArray.forEach((ev: any) => {
        const ts = ev['@_ts'] || ev.timestamp || ev.DisplayTime;
        const gl = ev['@_value'] || ev.value || ev.Value;
        if (ts && gl && !isNaN(Number(gl))) {
          data.glucose.push({ timestamp: parseOhioTime(ts) || ts, value: Number(gl) });
        }
      });

      // OhioT1DM Specific
      if (parsed?.patient) {
        // Glucose
        const gEvents = parsed.patient.glucose_level?.event || [];
        const gArray = Array.isArray(gEvents) ? gEvents : [gEvents];
        gArray.forEach((ev: any) => {
          const ts = ev['@_ts'];
          const val = ev['@_value'];
          if (ts && val && !isNaN(Number(val))) {
            data.glucose.push({ timestamp: parseOhioTime(ts) || ts, value: Number(val) });
          }
        });

        // Bolus
        const bEvents = parsed.patient.bolus?.event || [];
        const bArray = Array.isArray(bEvents) ? bEvents : [bEvents];
        bArray.forEach((ev: any) => {
          const ts = ev['@_ts_begin'];
          const val = ev['@_dose'];
          if (ts && val && !isNaN(Number(val))) {
            data.bolus.push({ timestamp: parseOhioTime(ts) || ts, value: Number(val) });
          }
        });

        // Meal (Carbs)
        const mEvents = parsed.patient.meal?.event || [];
        const mArray = Array.isArray(mEvents) ? mEvents : [mEvents];
        mArray.forEach((ev: any) => {
          const ts = ev['@_ts'];
          const val = ev['@_carbs'];
          if (ts && val && !isNaN(Number(val))) {
            data.carbs.push({ timestamp: parseOhioTime(ts) || ts, value: Number(val) });
          }
        });
      }

      // Sort by timestamp
      data.glucose.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      data.bolus.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      data.carbs.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

      return data;
    } catch (e) {
      console.error("XML Parsing error", e);
      return data;
    }
  }

  return data;
}
