"""
Parser for the HUPA-UCM dataset (Objective 3: external validation).

Produces, for each patient, a DataFrame in exactly the same format as
parse_xml_file() in data_processing.py (OhioT1DM):
    index "ts" (regular 5-minute grid), float columns glucose, bolus, carbs, heart_rate.

Data rules (see CLAUDE.md, fixed before any results):
- Glucose comes ONLY from raw FreeStyle Libre files
  (Raw_Data/<PID>/free_style_sensor/), record type 0 (historic readings).
  The glucose column of Preprocessed/<PID>.csv is never used (it is interpolated).
- Bolus = bolus_volume_delivered, carbs = carb_input x 10 (servings -> grams),
  both from Preprocessed/<PID>.csv (separator ";").
- Causal glucose rule, identical to parse_xml_file:
    Step A: reindex(grid, method="ffill", tolerance=5min)
    Step B: ffill(limit=6)
- Only raw readings inside the study window (first to last "time" of
  Preprocessed/<PID>.csv) are used. Glucose is clipped to [40, 400].
- Bolus values <= 0 are invalid records and are treated as no event.
- heart_rate is NaN. Nothing is imputed: missing bolus/carbs stay 0.0 (no event).

is_real (True where Step A found a raw reading in the last 5 minutes) is returned
separately and is never a column of the DataFrame.
"""

import csv
import glob
import os
import re

import numpy as np
import pandas as pd

DATA_DIR = "data/HUPA-UCM"

EXCLUDED_NO_CGM = frozenset({"HUPA0009P", "HUPA0010P"})
EXCLUDED_MISSING_EVENTS = frozenset({"HUPA0011P", "HUPA0015P", "HUPA0018P", "HUPA0020P"})
ANALYSES = ("main", "all")

GLUCOSE_MIN, GLUCOSE_MAX = 40.0, 400.0
MAX_UNPARSEABLE_FRACTION = 0.005
HEADER_SEARCH_LINES = 10

# Column positions (0-based) of time, record type and historic glucose per layout.
LAYOUTS = {
    "OLD": {"time": 1, "type": 2, "glucose": 3},
    "NEW": {"time": 2, "type": 3, "glucose": 4},
}

# Accepted timestamp formats, all day-first. The shapes are mutually exclusive,
# so each string matches at most one format.
DATE_FORMATS = [
    ("%Y/%m/%d %H:%M", re.compile(r"^\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}$")),  # 2018/06/13 17:15
    ("%d/%m/%y %H:%M", re.compile(r"^\d{1,2}/\d{1,2}/\d{2} \d{1,2}:\d{2}$")),  # 27/3/19 16:22
    ("%d-%m-%Y %H:%M", re.compile(r"^\d{1,2}-\d{1,2}-\d{4} \d{1,2}:\d{2}$")),  # 08-09-2020 14:49
]


class NoCGMDataError(Exception):
    """Raised when a patient has no usable type-0 FreeStyle Libre readings."""


class LibreFormatError(ValueError):
    """Raised when a Libre file cannot be read reliably."""


# ----------------------------------------------------------------------------
# Patient lists
# ----------------------------------------------------------------------------

def list_patients(data_dir=DATA_DIR):
    """All patient IDs (one per Preprocessed/<PID>.csv), sorted."""
    files = glob.glob(os.path.join(data_dir, "Preprocessed", "*.csv"))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def list_patients_for_analysis(analysis="main", data_dir=DATA_DIR):
    """Patients of the MAIN (19) or ALL (23) analysis."""
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis {analysis!r}; expected one of {ANALYSES}")
    patients = [p for p in list_patients(data_dir) if p not in EXCLUDED_NO_CGM]
    if analysis == "main":
        patients = [p for p in patients if p not in EXCLUDED_MISSING_EVENTS]
    return patients


# ----------------------------------------------------------------------------
# Preprocessed file (study window, bolus, carbs)
# ----------------------------------------------------------------------------

def _read_preprocessed(pid, data_dir=DATA_DIR):
    path = os.path.join(data_dir, "Preprocessed", f"{pid}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed file not found: {path}")
    prep = pd.read_csv(path, sep=";")
    prep["time"] = pd.to_datetime(prep["time"], format="%Y-%m-%dT%H:%M:%S")
    return prep


def load_study_window(pid, data_dir=DATA_DIR):
    """(start, end) = first and last 'time' in Preprocessed/<PID>.csv."""
    prep = _read_preprocessed(pid, data_dir)
    return prep["time"].min(), prep["time"].max()


def load_events(pid, data_dir=DATA_DIR):
    """Recorded events: (bolus in U, carbs in g), indexed by Preprocessed 'time'.

    NaN means 'nothing recorded' and is kept as NaN here (never estimated).
    """
    prep = _read_preprocessed(pid, data_dir).set_index("time")
    bolus = prep["bolus_volume_delivered"].astype(float)
    carbs = prep["carb_input"].astype(float) * 10.0
    return bolus, carbs


# ----------------------------------------------------------------------------
# Raw FreeStyle Libre files
# ----------------------------------------------------------------------------

def _libre_files(pid, data_dir=DATA_DIR):
    folder = os.path.join(data_dir, "Raw_Data", pid, "free_style_sensor")
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.txt"))
    return sorted(files)


def _detect_encoding(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", raw.decode("utf-8-sig")
    for enc in ("utf-8", "latin-1"):
        try:
            return enc, raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise LibreFormatError(f"Cannot decode {path} as utf-8, utf-8-sig or latin-1")


def _find_header(lines, path):
    """Return (line index, layout) of the header within the first 10 lines."""
    for i, line in enumerate(lines[:HEADER_SEARCH_LINES]):
        stripped = line.lstrip("﻿").strip()
        if "Dispositivo" in stripped:
            return i, "NEW"
        if stripped.startswith("ID") and "Hora" in stripped:
            return i, "OLD"
    raise LibreFormatError(f"No Libre header (OLD 'ID..Hora' or NEW 'Dispositivo') "
                           f"in the first {HEADER_SEARCH_LINES} lines of {path}")


def _detect_delimiter(header_line, path):
    counts = {d: header_line.count(d) for d in (";", ",", "\t")}
    delim = max(counts, key=counts.get)
    if counts[delim] == 0:
        raise LibreFormatError(f"Cannot detect delimiter in header of {path}")
    return delim


def _validate_header(cols, layout, path):
    pos = LAYOUTS[layout]
    names = [c.strip() for c in cols]
    ok = (len(names) > pos["glucose"]
          and names[pos["type"]] == "Tipo de registro"
          and names[pos["glucose"]].startswith("Hist"))
    if layout == "OLD":
        ok = ok and names[pos["time"]] == "Hora"
    if not ok:
        raise LibreFormatError(f"Header of {path} does not match the {layout} layout: {names[:6]}")


def parse_libre_timestamps(strings):
    """Parse day-first Libre timestamps strictly.

    Returns (parsed datetime Series with NaT for unparseable strings,
             dict {format: number of strings parsed with it}).
    """
    s = pd.Series(strings, dtype=object).astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    used = {}
    for fmt, pattern in DATE_FORMATS:
        mask = s.str.match(pattern) & parsed.isna()
        if mask.any():
            parsed[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            n_ok = int(parsed[mask].notna().sum())
            if n_ok:
                used[fmt] = n_ok
    return parsed, used


def _read_libre_file(path):
    """Read every type-0 row of one Libre file.

    Returns (DataFrame[ts, glucose], file metadata). Raises LibreFormatError if
    more than 0.5 % of the type-0 timestamps cannot be parsed.
    """
    encoding, text = _detect_encoding(path)
    lines = text.splitlines()
    hdr_idx, layout = _find_header(lines, path)
    delim = _detect_delimiter(lines[hdr_idx], path)
    header = next(csv.reader([lines[hdr_idx]], delimiter=delim))
    _validate_header(header, layout, path)
    pos = LAYOUTS[layout]

    ts_str, g_str = [], []
    for row in csv.reader(lines[hdr_idx + 1:], delimiter=delim):
        if len(row) <= pos["type"] or row[pos["type"]].strip() != "0":
            continue
        ts_str.append(row[pos["time"]].strip())
        g_str.append(row[pos["glucose"]].strip() if len(row) > pos["glucose"] else "")

    n_type0 = len(ts_str)
    ts, used_formats = parse_libre_timestamps(ts_str)
    n_bad_ts = int(ts.isna().sum())
    if n_type0 and n_bad_ts / n_type0 > MAX_UNPARSEABLE_FRACTION:
        examples = [ts_str[i] for i in np.flatnonzero(ts.isna().values)[:5]]
        raise LibreFormatError(
            f"{path}: {n_bad_ts}/{n_type0} type-0 timestamps could not be parsed "
            f"(> {MAX_UNPARSEABLE_FRACTION:.1%}). Examples: {examples}")

    g_clean = pd.Series(g_str, dtype=object)
    if delim != ",":
        g_clean = g_clean.str.replace(",", ".", regex=False)
    glucose = pd.to_numeric(g_clean, errors="coerce")
    n_bad_g = int((glucose.isna() & ts.notna()).sum())
    if n_type0 and n_bad_g / n_type0 > MAX_UNPARSEABLE_FRACTION:
        raise LibreFormatError(f"{path}: {n_bad_g}/{n_type0} type-0 rows have no numeric glucose")

    df = pd.DataFrame({"ts": ts.values, "glucose": glucose.values})
    df = df.dropna(subset=["ts", "glucose"])
    meta = {
        "file": os.path.basename(path),
        "encoding": encoding,
        "layout": layout,
        "delimiter": delim,
        "date_format": "+".join(used_formats) if used_formats else "none",
        "n_type0": n_type0,
        "n_unparseable_timestamps": n_bad_ts,
        "n_bad_glucose": n_bad_g,
    }
    return df, meta


def _join(values):
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return "+".join(out)


def load_libre_glucose(pid, data_dir=DATA_DIR, restrict_to_window=True):
    """Raw type-0 Libre readings of one patient (all files combined).

    Returns (Series of UNCLIPPED glucose indexed by reading time, metadata).
    The Series is sorted and de-duplicated (keep='last', as parse_xml_file).
    With restrict_to_window=True (default) only readings inside the study
    window are returned. Raises NoCGMDataError if there are no type-0 readings
    (or none inside the window).
    """
    files = _libre_files(pid, data_dir)
    frames, file_meta = [], []
    for path in files:
        df, meta = _read_libre_file(path)
        frames.append(df)
        file_meta.append(meta)

    all_readings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts", "glucose"])
    if all_readings.empty:
        raise NoCGMDataError(f"{pid}: no type-0 FreeStyle Libre readings "
                             f"({len(files)} Libre file(s))")

    all_readings = all_readings.sort_values("ts", kind="mergesort")
    dup_mask = all_readings["ts"].duplicated(keep="last")
    n_dup = int(dup_mask.sum())
    n_conflicting = int(all_readings.groupby("ts")["glucose"].nunique().gt(1).sum())
    all_readings = all_readings[~dup_mask]
    series = pd.Series(all_readings["glucose"].astype(float).values,
                       index=pd.DatetimeIndex(all_readings["ts"].values, name="ts"))

    start, end = load_study_window(pid, data_dir)
    before = int((series.index < start).sum())
    after = int((series.index > end).sum())
    inside = series[(series.index >= start) & (series.index <= end)]

    metadata = {
        "patient": pid,
        "files": [m["file"] for m in file_meta],
        "n_files": len(files),
        "encoding": _join(m["encoding"] for m in file_meta),
        "layout": _join(m["layout"] for m in file_meta),
        "delimiter": _join(repr(m["delimiter"]) for m in file_meta),
        "date_format": _join(m["date_format"] for m in file_meta if m["date_format"] != "none"),
        "n_type0_total": sum(m["n_type0"] for m in file_meta),
        "n_unparseable_timestamps": sum(m["n_unparseable_timestamps"] for m in file_meta),
        "n_bad_glucose": sum(m["n_bad_glucose"] for m in file_meta),
        "n_duplicate_timestamps": n_dup,
        "n_conflicting_duplicates": n_conflicting,
        "window_start": start,
        "window_end": end,
        "readings_before_window": before,
        "readings_in_window": int(len(inside)),
        "readings_after_window": after,
        "per_file": file_meta,
    }

    if restrict_to_window:
        if inside.empty:
            raise NoCGMDataError(f"{pid}: {len(series)} type-0 readings, none inside the "
                                 f"study window {start} - {end}")
        return inside, metadata
    return series, metadata


# ----------------------------------------------------------------------------
# Full patient parse
# ----------------------------------------------------------------------------

def _bucket_events(events, grid):
    """Sum recorded events per 5-minute bucket on the grid (NaN = no event)."""
    events = events.dropna()
    events = events[events != 0]
    out = pd.Series(0.0, index=grid)
    buckets = events.index.floor("5min")
    on_grid = buckets.isin(grid)
    sums = events[on_grid].groupby(buckets[on_grid]).sum()
    out.loc[sums.index] += sums.values
    return out, int((~on_grid).sum())


def parse_hupa_patient(pid, data_dir=DATA_DIR):
    """Parse one HUPA-UCM patient into the OhioT1DM format.

    Returns (df, is_real, metadata):
      df       index 'ts' (5-minute grid), float columns glucose, bolus, carbs, heart_rate
      is_real  bool Series on the same index: a raw reading in the last 5 minutes (Step A)
      metadata dict (layout, delimiter, date format, reading counts, ...)
    """
    raw, metadata = load_libre_glucose(pid, data_dir)
    n_low = int((raw < GLUCOSE_MIN).sum())
    n_high = int((raw > GLUCOSE_MAX).sum())
    glucose_series = raw.clip(GLUCOSE_MIN, GLUCOSE_MAX)

    # Same grid construction as parse_xml_file (first to last glucose reading).
    grid = pd.date_range(start=glucose_series.index.min().floor("5min"),
                         end=glucose_series.index.max().ceil("5min"), freq="5min")
    df = pd.DataFrame(index=grid)
    df.index.name = "ts"

    # Step A: most recent raw reading at or before t, at most 5 minutes old.
    df["glucose"] = glucose_series.reindex(df.index, method="ffill", tolerance=pd.Timedelta("5min"))
    is_real = df["glucose"].notna().rename("is_real")
    # Step B: causal forward fill of short gaps (max 30 minutes).
    df["glucose"] = df["glucose"].ffill(limit=6)

    bolus, carbs = load_events(pid, data_dir)
    # Bolus values <= 0 are invalid records (negative insulin is impossible) -> no event.
    # Zeros are ordinary "no event" rows; only negative values are counted as invalid.
    n_invalid_bolus = int((bolus < 0).sum())
    bolus = bolus.where(bolus > 0)
    df["bolus"], n_bolus_off = _bucket_events(bolus, df.index)
    df["carbs"], n_carbs_off = _bucket_events(carbs, df.index)
    df["heart_rate"] = np.nan
    df = df[["glucose", "bolus", "carbs", "heart_rate"]].astype(float)

    metadata.update({
        "n_invalid_boluses_removed": n_invalid_bolus,
        "n_clipped_below_40": n_low,
        "n_clipped_above_400": n_high,
        "grid_start": df.index.min(),
        "grid_end": df.index.max(),
        "n_grid_rows": int(len(df)),
        "n_bolus_events_outside_grid": n_bolus_off,
        "n_carb_events_outside_grid": n_carbs_off,
    })
    return df, is_real, metadata
