"""
Diagnose HUPA-UCM raw file formats.
Inspection ONLY — does not modify any existing file.
Creates: results_objective3/hupa_format_diagnosis.txt
"""

import os
import glob
import warnings
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

PATIENTS_TO_DIAGNOSE = [
    "HUPA0001P","HUPA0009P","HUPA0010P","HUPA0011P","HUPA0016P",
    "HUPA0019P","HUPA0022P","HUPA0026P","HUPA0027P","HUPA0028P"
]
ALL_PATIENTS = [
    f"HUPA{str(i).zfill(4)}P"
    for i in range(1, 29) if i not in [8, 12, 13]
]
OUT_FILE = "results_objective3/hupa_format_diagnosis.txt"
RAW_DIR  = "data/HUPA-UCM/Raw_Data"
PREP_DIR = "data/HUPA-UCM/Preprocessed"

# ── helpers ──────────────────────────────────────────────────────────────────

def detect_encoding(path):
    for enc in ["utf-8", "utf-8-sig", "latin-1", "utf-16"]:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(16384)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            pass
    return "utf-8"

def best_delimiter(lines):
    counts = {";": 0, ",": 0, "\t": 0}
    for l in lines[:20]:
        if not l.strip():
            continue
        for d in counts:
            counts[d] += l.count(d)
    return max(counts, key=counts.get)

def _detect_date_fmt(sample):
    """Return a description of the date format in the sample string."""
    sample = sample.strip()
    if not sample:
        return "empty"
    if "-" in sample.split()[0]:
        parts = sample.split()[0].split("-")
        if len(parts[0]) == 4:
            return "YYYY-MM-DD HH:MM"
        else:
            return "DD-MM-YYYY HH:MM"
    if "/" in sample.split()[0]:
        parts = sample.split()[0].split("/")
        if len(parts[0]) == 4:
            return "YYYY/MM/DD HH:MM"
        else:
            return "D/M/YY HH:MM or DD/M/YY HH:MM (day-first ambiguous)"
    return "unknown"

def prep_range(pid):
    path = f"{PREP_DIR}/{pid}.csv"
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path, sep=";")
        df["time"] = pd.to_datetime(df["time"])
        return df["time"].min(), df["time"].max()
    except Exception:
        return None, None

def parse_libre_file(path, pid):
    enc  = detect_encoding(path)
    with open(path, encoding=enc, errors="replace") as f:
        raw_lines = f.readlines()

    size      = os.path.getsize(path)
    n_lines   = len(raw_lines)
    non_empty = [l.rstrip("\n") for l in raw_lines if l.strip()]
    delim     = best_delimiter(non_empty)

    # ── 4. first 15 lines ────────────────────────────────────────────────────
    first15 = [f"  {i+1}: {l[:300]}" for i, l in enumerate(raw_lines[:15])]

    # ── 5. header line ───────────────────────────────────────────────────────
    # New FreeStyle LibreLink exports have 2 extra metadata rows before the real header
    hdr_idx  = -1
    hdr_line = ""
    layout   = "OLD"   # "OLD" = ID;Hora;... / "NEW" = Dispositivo;Número de serial;...
    for i, l in enumerate(non_empty):
        ll = l.lower()
        if "dispositivo" in ll and "serial" in ll:
            hdr_idx  = i
            hdr_line = l
            layout   = "NEW"
            break
        if "id" in ll and ("hora" in ll or "time" in ll):
            hdr_idx  = i
            hdr_line = l
            layout   = "OLD"
            break
    if hdr_idx == -1 and non_empty:
        hdr_idx, hdr_line = 1, non_empty[1] if len(non_empty) > 1 else non_empty[0]

    cols = hdr_line.split(delim)
    n_cols = len(cols)

    # determine column indices from header
    type_col = time_col = gluc_hist_col = gluc_scan_col = -1
    for ci, c in enumerate(cols):
        cl = c.lower()
        if "tipo de registro" in cl or "type of record" in cl:
            type_col = ci
        if layout == "OLD":
            if "hora" in cl or ("time" in cl and "device" not in cl):
                time_col = ci
            if "histórico glucosa" in cl or "historial de glucosa" in cl or "histA3rico glucosa" in cl:
                gluc_hist_col = ci
        else:  # NEW layout: col 2 = timestamp, col 3 = type, col 4 = hist, col 5 = scan
            if "sello de tiempo" in cl or "timestamp" in cl:
                time_col = ci
            if "historial de glucosa" in cl or "histórico glucosa" in cl:
                gluc_hist_col = ci
            if "escaneo de glucosa" in cl:
                gluc_scan_col = ci

    # fallback by position
    if type_col == -1: type_col = 3 if layout == "NEW" else 2
    if time_col == -1: time_col = 2 if layout == "NEW" else 1
    if gluc_hist_col == -1: gluc_hist_col = 4 if layout == "NEW" else 3

    data_lines = non_empty[hdr_idx+1:] if hdr_idx != -1 else []

    # ── 6. record types, glucose cols ────────────────────────────────────────
    rec_types  = []
    gluc_cols  = set()
    time_vals  = []
    for l in data_lines:
        row = l.split(delim)
        if len(row) > type_col:
            rec_types.append(row[type_col].strip())
        if len(row) > time_col:
            tv = row[time_col].strip()
            if tv:
                time_vals.append(tv)
        for ci, c in enumerate(row):
            cv = c.replace(",", ".")
            try:
                v = float(cv)
                if 40 <= v <= 500:
                    gluc_cols.add(ci)
            except ValueError:
                pass

    type_counter = Counter(rec_types)

    # ── 8. date format ───────────────────────────────────────────────────────
    sample_ts = [tv for tv in time_vals[:50] if tv]
    fmt_desc  = _detect_date_fmt(sample_ts[0]) if sample_ts else "no data"
    ambiguous = "ambiguous" in fmt_desc

    # ── 9. earliest/latest ───────────────────────────────────────────────────
    ts_df = ts_mf = pd.Series([], dtype="datetime64[ns]")
    if time_vals:
        ts_df = pd.to_datetime(pd.Series(time_vals), dayfirst=True,  errors="coerce").dropna()
        ts_mf = pd.to_datetime(pd.Series(time_vals), dayfirst=False, errors="coerce").dropna()

    # ── 10. compare with preprocessed ────────────────────────────────────────
    ps, pe = prep_range(pid)

    return dict(
        path=path, size=size, n_lines=n_lines, enc=enc, delim=repr(delim),
        layout=layout, hdr_line=hdr_line[:200], n_cols=n_cols,
        type_counter=type_counter, gluc_cols=sorted(gluc_cols),
        time_vals=time_vals, fmt_desc=fmt_desc, ambiguous=ambiguous,
        ts_df_min=ts_df.min() if len(ts_df) else None,
        ts_df_max=ts_df.max() if len(ts_df) else None,
        ts_mf_min=ts_mf.min() if len(ts_mf) else None,
        ts_mf_max=ts_mf.max() if len(ts_mf) else None,
        prep_start=ps, prep_end=pe,
        first15=first15,
    )


def parse_medtronic_file(path, pid):
    enc  = detect_encoding(path)
    with open(path, encoding=enc, errors="replace") as f:
        raw_lines = f.readlines()

    size    = os.path.getsize(path)
    n_lines = len(raw_lines)
    non_empty = [l.rstrip("\n") for l in raw_lines if l.strip()]
    delim   = best_delimiter(non_empty)

    first15 = [f"  {i+1}: {l[:300]}" for i, l in enumerate(raw_lines[:15])]

    # Find real header (CareLink: row with "Index;Date;Time;...")
    hdr_idx  = -1
    hdr_line = ""
    for i, l in enumerate(non_empty):
        ll = l.lower()
        if "index" in ll and ("date" in ll or "fecha" in ll):
            hdr_idx  = i
            hdr_line = l
            break

    cols = hdr_line.split(delim) if hdr_line else []
    sensor_col = -1
    for ci, c in enumerate(cols):
        if "sensor" in c.lower() and "gluc" in c.lower():
            sensor_col = ci
            break

    # Count non-empty sensor values
    sensor_nz = 0
    date_col = time_col = -1
    for ci, c in enumerate(cols):
        cl = c.lower()
        if cl in ("date", "fecha"): date_col = ci
        if cl in ("time", "hora"):  time_col = ci
    data_lines = non_empty[hdr_idx+1:] if hdr_idx != -1 else []
    time_vals = []
    for l in data_lines:
        row = l.split(delim)
        if sensor_col != -1 and len(row) > sensor_col:
            sv = row[sensor_col].replace(",", ".")
            try:
                if float(sv) > 0: sensor_nz += 1
            except ValueError: pass
        if date_col != -1 and len(row) > date_col:
            dv = row[date_col].strip()
            tv = row[time_col].strip() if time_col != -1 and len(row) > time_col else ""
            if dv:
                time_vals.append((dv + " " + tv).strip())

    fmt_desc = _detect_date_fmt(time_vals[0]) if time_vals else "no data"
    return dict(
        path=path, size=size, n_lines=n_lines, enc=enc, delim=repr(delim),
        hdr_idx=hdr_idx, hdr_line=hdr_line[:200], n_cols=len(cols),
        sensor_col=sensor_col,
        sensor_col_name=cols[sensor_col] if sensor_col != -1 else "",
        sensor_nz=sensor_nz, time_vals=time_vals, fmt_desc=fmt_desc,
        first15=first15,
    )


# ── main diagnosis ────────────────────────────────────────────────────────────

def main():
    os.makedirs("results_objective3", exist_ok=True)
    lines = []
    w = lines.append

    for pid in PATIENTS_TO_DIAGNOSE:
        w(f"\n{'='*70}")
        w(f"PATIENT: {pid}")
        w(f"{'='*70}")

        # --- Libre files ---
        libre_files = sorted(
            glob.glob(f"{RAW_DIR}/{pid}/free_style_sensor/*.csv") +
            glob.glob(f"{RAW_DIR}/{pid}/free_style_sensor/*.txt")
        )
        w(f"\n[FREE STYLE LIBRE] {len(libre_files)} file(s)")
        for fp in libre_files:
            r = parse_libre_file(fp, pid)
            w(f"\n  1. Path: {fp}")
            w(f"     Size: {r['size']} bytes  |  Lines: {r['n_lines']}")
            w(f"  2. Encoding: {r['enc']}")
            w(f"  3. Delimiter: {r['delim']}")
            w(f"  4. First 15 lines:")
            for l in r['first15']:
                w(l)
            w(f"  5. Header (non-empty line {r['n_cols']} cols): {r['hdr_line']}")
            w(f"     Layout: {r['layout']}  |  Columns: {r['n_cols']}")
            w(f"  6. Record types: {dict(r['type_counter'].most_common(8))}")
            w(f"     Glucose-like cols (40-500): {r['gluc_cols']}")
            w(f"  8. Date format detected: {r['fmt_desc']}  |  Ambiguous: {r['ambiguous']}")
            if r['time_vals']:
                w(f"     Sample timestamps: {r['time_vals'][:3]}")
            w(f"  9. Parsed range (day-first):   {r['ts_df_min']} to {r['ts_df_max']}")
            if r['ambiguous']:
                w(f"     Parsed range (month-first): {r['ts_mf_min']} to {r['ts_mf_max']}")
            w(f" 10. Preprocessed range: {r['prep_start']} to {r['prep_end']}")
            if r['ts_df_min'] and r['prep_start']:
                raw_covers = (r['ts_df_min'] <= r['prep_start'] and
                              r['ts_df_max'] >= r['prep_end'])
                w(f"     Raw {'COVERS or EXCEEDS' if raw_covers else 'DOES NOT COVER'} preprocessed period.")

        # --- Medtronic files ---
        med_files = sorted(glob.glob(f"{RAW_DIR}/{pid}/medtronic_insulin_pump/*.csv"))
        w(f"\n[MEDTRONIC] {len(med_files)} file(s)")
        for fp in med_files:
            r = parse_medtronic_file(fp, pid)
            w(f"\n  1. Path: {fp}")
            w(f"     Size: {r['size']} bytes  |  Lines: {r['n_lines']}")
            w(f"  2. Encoding: {r['enc']}")
            w(f"  3. Delimiter: {r['delim']}")
            w(f"  4. First 15 lines:")
            for l in r['first15']:
                w(l)
            w(f"  7. Real data header at row {r['hdr_idx']}: {r['hdr_line']}")
            w(f"     Columns: {r['n_cols']}")
            w(f"     Sensor glucose col: {r['sensor_col']} ({r['sensor_col_name']!r})")
            w(f"     Non-empty sensor values: {r['sensor_nz']}")
            w(f"  8. Date format: {r['fmt_desc']}")
            if r['time_vals']:
                w(f"     Sample timestamps: {r['time_vals'][:3]}")

    # ── Summary table for ALL 25 patients ────────────────────────────────────
    w(f"\n\n{'='*70}")
    w("SUMMARY TABLE — ALL 25 PATIENTS")
    w(f"{'='*70}")

    summary_rows = []
    for pid in ALL_PATIENTS:
        libre_files = sorted(
            glob.glob(f"{RAW_DIR}/{pid}/free_style_sensor/*.csv") +
            glob.glob(f"{RAW_DIR}/{pid}/free_style_sensor/*.txt")
        )
        med_files = sorted(glob.glob(f"{RAW_DIR}/{pid}/medtronic_insulin_pump/*.csv"))

        l_lines = l_delim = l_hdr_lang = l_date = l_layout = ""
        raw_s = raw_e = ""
        n_type0 = 0

        for fp in libre_files:
            r = parse_libre_file(fp, pid)
            l_lines   = r['n_lines']
            l_delim   = r['delim']
            hl        = r['hdr_line'].lower()
            l_hdr_lang = "Spanish" if "hora" in hl or "sello" in hl else ("English" if "time" in hl else "other")
            l_date    = r['fmt_desc']
            l_layout  = r['layout']
            n_type0   = r['type_counter'].get('0', 0)
            if r['ts_df_min']:
                raw_s = str(r['ts_df_min'])[:16]
                raw_e = str(r['ts_df_max'])[:16]
            break  # first file only

        m_has_sg = False
        for fp in med_files:
            r = parse_medtronic_file(fp, pid)
            m_has_sg = r['sensor_col'] != -1
            break

        ps, pe = prep_range(pid)

        summary_rows.append({
            "patient"                    : pid,
            "libre_files"                : len(libre_files),
            "libre_lines"                : l_lines,
            "libre_delim"                : l_delim,
            "libre_lang"                 : l_hdr_lang,
            "libre_layout"               : l_layout,
            "libre_date_fmt"             : l_date,
            "n_type0_readings"           : n_type0,
            "med_files"                  : len(med_files),
            "med_has_sensor_glucose"     : m_has_sg,
            "raw_start"                  : raw_s,
            "raw_end"                    : raw_e,
            "prep_start"                 : str(ps)[:16] if ps else "",
            "prep_end"                   : str(pe)[:16] if pe else "",
        })

    df = pd.DataFrame(summary_rows)
    w(df.to_string(index=False))

    # write to file
    text = "\n".join(lines)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    import sys
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    print(f"\n\n[Saved to {OUT_FILE}]")


if __name__ == "__main__":
    main()
