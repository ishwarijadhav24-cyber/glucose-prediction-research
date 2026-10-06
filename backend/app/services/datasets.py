"""Upload adapters: detect the dataset format and convert it to the canonical event schema.

Canonical schema (one row per event, chronological):
    timestamp (local, naive) | glucose_mg_dl | bolus_units | carbs_g   (NaN = no event)

Supported uploads (multipart field "file", 1-4 files of ONE dataset type):
  ohiot1dm_xml    OhioT1DM XML (<patient> with glucose_level/bolus/meal events); 1-2 files of the
                  same patient (e.g. training + testing). Same events and timestamp format that
                  data_processing.parse_xml_file reads.
  hupa_ucm        HUPA-UCM raw FreeStyle Libre export(s) (OLD or NEW layout), optionally plus the
                  matching Preprocessed/<PID>.csv for bolus/carbs. Same rules as the validated
                  research parser parsers/hupa_ucm.py: type-0 historic glucose only, study window
                  from the Preprocessed file, duplicates keep last, clip [40, 400],
                  bolus = bolus_volume_delivered (<= 0 = no event), carbs = carb_input x 10.
                  The Preprocessed glucose column is never read.
  normalized_csv  timestamp,glucose_mg_dl,bolus_units,carbs_g (see csv_input.py)

No value is interpolated or invented. Sampling is detected from the glucose timestamps and
must be ~5 min (native) or ~15 min (accepted only with explicit external-validation warnings).
"""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath

import numpy as np
import pandas as pd
from lxml import etree

import parsers.hupa_ucm as hupa
from backend.app.errors import BackendError, ErrorCode
from backend.app.services.csv_input import parse_events_csv, validate_ranges
from backend.app.services.features import EVENT_COLUMNS

OHIO, HUPA, NORMALIZED = "ohiot1dm_xml", "hupa_ucm", "normalized_csv"
LIBRE, HUPA_PREP = "libre_csv", "hupa_preprocessed_csv"
MAX_FILES = 4
OHIO_TS = "%d-%m-%Y %H:%M:%S"
HUPA_PREP_COLUMNS = ("time", "bolus_volume_delivered", "carb_input")
SAMPLING = {"5min": (4.0, 6.0), "15min": (13.5, 16.5)}


@dataclass
class UploadedFile:
    data: bytes
    filename: str = ""          # used only for the extension and research file ordering; never logged/returned
    kind: str = ""              # detected file type


@dataclass
class CanonicalDataset:
    dataset_type: str
    events: pd.DataFrame
    file_types: list
    sampling_profile: str = ""
    median_interval_minutes: float = float("nan")
    transformations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _bad(code, msg):
    return BackendError(code, msg)


# ---------------------------------------------------------------- detection
def _decode(data: bytes) -> str:
    """Same encoding rule as parsers/hupa_ucm._detect_encoding (BOM -> utf-8-sig, then utf-8, latin-1)."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def detect_file_type(f: UploadedFile) -> str:
    if not f.data.lstrip(b"\xef\xbb\xbf \t\r\n"):
        raise _bad(ErrorCode.INVALID_FORMAT, "The file is empty.")
    head = f.data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    ext = PurePath(f.filename).suffix.lower()
    if head.startswith(b"<"):
        if ext != ".xml":
            raise _bad(ErrorCode.INVALID_FILE, "XML data must use the .xml extension.")
        return OHIO
    if ext != ".csv":
        raise _bad(ErrorCode.INVALID_FILE, "Only .csv (normalized, HUPA-UCM) or .xml (OhioT1DM) files are accepted.")
    try:
        text_head = _decode(f.data[:65536])
    except UnicodeDecodeError:
        raise _bad(ErrorCode.INVALID_FILE, "The file must be text.") from None
    if "\x00" in text_head[:2048]:
        raise _bad(ErrorCode.INVALID_FILE, "The file must be text.")
    lines = text_head.splitlines()
    first = lines[0].lstrip("﻿").strip() if lines else ""
    if set(c.strip() for c in first.split(",")) == set(EVENT_COLUMNS):
        return NORMALIZED
    cols = [c.strip() for c in first.split(";")]
    if all(c in cols for c in HUPA_PREP_COLUMNS):
        return HUPA_PREP
    try:
        hupa._find_header(lines, "upload")
        return LIBRE
    except hupa.LibreFormatError:
        pass
    if "," in first and any(c.strip() in EVENT_COLUMNS for c in first.split(",")):
        return NORMALIZED            # let the normalized parser report the exact column problem
    raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Unrecognised dataset format. Supported: OhioT1DM XML, HUPA-UCM "
                                             "(FreeStyle Libre export + Preprocessed CSV), normalized CSV.")


# ---------------------------------------------------------------- OhioT1DM
_SAFE_XML = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False,
                            remove_comments=True, remove_pis=True)


def _ohio_events(data: bytes):
    if b"<!DOCTYPE" in data[:4096].upper() or b"<!ENTITY" in data.upper():
        raise _bad(ErrorCode.INVALID_FORMAT, "XML with DOCTYPE/ENTITY declarations is not accepted.")
    try:
        root = etree.fromstring(data, parser=_SAFE_XML)
    except etree.XMLSyntaxError:
        raise _bad(ErrorCode.INVALID_FORMAT, "The XML file is not well-formed.") from None
    if root.tag != "patient" or root.find("glucose_level") is None:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Not an OhioT1DM file (expected <patient> with <glucose_level>).")
    rows = []
    # exactly the tags/attributes parse_xml_file reads
    for tag, ts_attr, val_attr, col in (("glucose_level", "ts", "value", "glucose_mg_dl"),
                                        ("bolus", "ts_begin", "dose", "bolus_units"),
                                        ("meal", "ts", "carbs", "carbs_g")):
        for e in root.findall(f".//{tag}/event"):
            rows.append((e.get(ts_attr), col, e.get(val_attr)))
    if not rows:
        raise _bad(ErrorCode.MISSING_COLUMNS, "The OhioT1DM file has no events.")
    raw = pd.DataFrame(rows, columns=["ts", "col", "val"])
    ts = pd.to_datetime(raw["ts"], format=OHIO_TS, errors="coerce")
    if ts.isna().any():
        raise _bad(ErrorCode.INVALID_TIMESTAMP, f"{int(ts.isna().sum())} event(s) have a missing or invalid "
                                                "timestamp (expected DD-MM-YYYY HH:MM:SS).")
    val = pd.to_numeric(raw["val"], errors="coerce")
    if val.isna().any() or not np.isfinite(val).all():
        raise _bad(ErrorCode.INVALID_NUMERIC_VALUE, f"{int(val.isna().sum())} event(s) have a non-numeric value.")
    ev = pd.DataFrame({"timestamp": ts})
    for c in ("glucose_mg_dl", "bolus_units", "carbs_g"):
        ev[c] = np.where(raw["col"] == c, val, np.nan)
    return root.get("id"), ev


def adapt_ohio(files):
    ids, frames = set(), []
    for f in files:
        pid, ev = _ohio_events(f.data)
        ids.add(pid)
        frames.append(ev)
    if len(ids) > 1:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "All OhioT1DM files must belong to the same patient.")
    ev = pd.concat(frames, ignore_index=True)
    range_warnings = validate_ranges(ev, strict=False)
    ev = ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    tr = ["OhioT1DM events read as recorded: glucose_level value, bolus dose at ts_begin, meal carbs (g)."]
    g = ev.loc[ev["glucose_mg_dl"].notna(), "timestamp"]
    if g.duplicated().any():
        tr.append(f"{int(g.duplicated().sum())} duplicate glucose timestamp(s): last value kept (research rule).")
    if len(files) > 1:
        tr.append("Files combined into one timeline (research parsed each file separately).")
    return CanonicalDataset(OHIO, ev, [OHIO] * len(files), transformations=tr, warnings=range_warnings)


# ---------------------------------------------------------------- HUPA-UCM
def _libre_readings(data: bytes) -> pd.DataFrame:
    """In-memory equivalent of parsers.hupa_ucm._read_libre_file (which needs a file path).

    Uses that module's own helpers; the row loop below mirrors _read_libre_file exactly
    (parity is tested against _read_libre_file on every real HUPA-UCM Libre file).
    """
    try:
        text = _decode(data)
        lines = text.splitlines()
        hdr_idx, layout = hupa._find_header(lines, "upload")
        delim = hupa._detect_delimiter(lines[hdr_idx], "upload")
        header = next(csv.reader([lines[hdr_idx]], delimiter=delim))
        hupa._validate_header(header, layout, "upload")
    except (hupa.LibreFormatError, csv.Error):
        raise _bad(ErrorCode.INVALID_FORMAT, "Unrecognised FreeStyle Libre export layout.") from None
    pos = hupa.LAYOUTS[layout]
    ts_str, g_str = [], []
    try:
        for row in csv.reader(lines[hdr_idx + 1:], delimiter=delim):
            if len(row) <= pos["type"] or row[pos["type"]].strip() != "0":
                continue
            ts_str.append(row[pos["time"]].strip())
            g_str.append(row[pos["glucose"]].strip() if len(row) > pos["glucose"] else "")
    except csv.Error:
        raise _bad(ErrorCode.INVALID_FORMAT, "The Libre export is not valid CSV.") from None
    n0 = len(ts_str)
    ts, _ = hupa.parse_libre_timestamps(ts_str)
    if n0 and ts.isna().sum() / n0 > hupa.MAX_UNPARSEABLE_FRACTION:
        raise _bad(ErrorCode.INVALID_TIMESTAMP, "Too many Libre timestamps could not be parsed (> 0.5 %).")
    g = pd.Series(g_str, dtype=object)
    if delim != ",":
        g = g.str.replace(",", ".", regex=False)
    glucose = pd.to_numeric(g, errors="coerce")
    if n0 and (glucose.isna() & ts.notna()).sum() / n0 > hupa.MAX_UNPARSEABLE_FRACTION:
        raise _bad(ErrorCode.INVALID_NUMERIC_VALUE, "Too many Libre glucose values are not numeric (> 0.5 %).")
    df = pd.DataFrame({"ts": ts.values, "glucose": glucose.values}).dropna(subset=["ts", "glucose"])
    return df


def _hupa_events_from_preprocessed(data: bytes):
    try:
        prep = pd.read_csv(io.BytesIO(data), sep=";", usecols=list(HUPA_PREP_COLUMNS))   # glucose never read
    except (ValueError, pd.errors.ParserError):
        raise _bad(ErrorCode.MISSING_COLUMNS, f"The HUPA-UCM Preprocessed file needs columns {list(HUPA_PREP_COLUMNS)}.") \
            from None
    prep["time"] = pd.to_datetime(prep["time"], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    if prep["time"].isna().any():
        raise _bad(ErrorCode.INVALID_TIMESTAMP, "Preprocessed 'time' must be YYYY-MM-DDTHH:MM:SS.")
    bolus = pd.to_numeric(prep["bolus_volume_delivered"], errors="coerce")
    carbs = pd.to_numeric(prep["carb_input"], errors="coerce") * 10.0
    return prep["time"], bolus, carbs


def adapt_hupa(files):
    libre = sorted((f for f in files if f.kind == LIBRE), key=lambda f: f.filename)   # research: sorted file names
    prep = [f for f in files if f.kind == HUPA_PREP]
    if not libre:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "HUPA-UCM uploads need the raw FreeStyle Libre export. The "
                                                 "Preprocessed glucose is interpolated and is never used.")
    if len(prep) > 1:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Upload at most one HUPA-UCM Preprocessed file.")
    tr, warn = ["Glucose: FreeStyle Libre type-0 (historic) readings only; scans and other record types ignored."], []
    readings = pd.concat([_libre_readings(f.data) for f in libre], ignore_index=True)
    if readings.empty:
        raise _bad(ErrorCode.MISSING_COLUMNS, "No type-0 (historic) glucose readings in the Libre export.")
    readings = readings.sort_values("ts", kind="mergesort")
    dup = readings["ts"].duplicated(keep="last")
    if dup.any():
        tr.append(f"{int(dup.sum())} duplicate reading timestamp(s) across files: last value kept (research rule).")
    readings = readings[~dup]
    ev_parts = []
    if prep:
        t, bolus, carbs = _hupa_events_from_preprocessed(prep[0].data)
        start, end = t.min(), t.max()
        inside = (readings["ts"] >= start) & (readings["ts"] <= end)
        if (~inside).any():
            tr.append(f"{int((~inside).sum())} reading(s) outside the Preprocessed study window removed (research rule).")
        readings = readings[inside]
        n_neg = int((bolus < 0).sum())
        if n_neg:
            tr.append(f"{n_neg} bolus value(s) <= 0 treated as no event (research rule).")
        b = bolus.where(bolus > 0)
        c = carbs.where(carbs > 0)
        ev_parts.append(pd.DataFrame({"timestamp": t, "bolus_units": b}).dropna())
        ev_parts.append(pd.DataFrame({"timestamp": t, "carbs_g": c}).dropna())
        tr.append("Bolus = bolus_volume_delivered; carbs = carb_input x 10 (servings to grams). "
                  "The Preprocessed glucose column is not read.")
    else:
        warn.append("No HUPA-UCM Preprocessed file: no insulin or carbohydrate records are available.")
    if readings.empty:
        raise _bad(ErrorCode.INSUFFICIENT_HISTORY, "No glucose readings inside the study window.")
    n_lo, n_hi = int((readings["glucose"] < 40).sum()), int((readings["glucose"] > 400).sum())
    if n_lo or n_hi:
        tr.append(f"Glucose clipped to [40, 400] mg/dL: {n_lo} low, {n_hi} high value(s) (research rule).")
    ev_parts.insert(0, pd.DataFrame({"timestamp": readings["ts"].values,
                                     "glucose_mg_dl": readings["glucose"].clip(40.0, 400.0).values}))
    ev = pd.concat(ev_parts, ignore_index=True).reindex(columns=list(EVENT_COLUMNS))
    warn += validate_ranges(ev, strict=False)
    ev = ev.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return CanonicalDataset(HUPA, ev, [f.kind for f in files], transformations=tr, warnings=warn)


# ---------------------------------------------------------------- normalized CSV
def adapt_normalized(files):
    if len(files) != 1:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Upload exactly one normalized CSV file.")
    ev, warn = parse_events_csv(files[0].data)
    return CanonicalDataset(NORMALIZED, ev, [NORMALIZED], warnings=list(warn))


# ---------------------------------------------------------------- sampling + entry point
def detect_sampling(ds: CanonicalDataset, external: dict | None) -> None:
    g = ds.events.loc[ds.events["glucose_mg_dl"].notna(), "timestamp"]
    if len(g) < 2:
        ds.sampling_profile, ds.median_interval_minutes = "unknown", float("nan")
        return
    med = float(g.diff().median() / pd.Timedelta("1min"))
    ds.median_interval_minutes = round(med, 2)
    for profile, (lo, hi) in SAMPLING.items():
        if lo <= med <= hi:
            ds.sampling_profile = profile
            break
    else:
        raise _bad(ErrorCode.UNSUPPORTED_SAMPLING,
                   f"Glucose readings are about {med:.1f} minutes apart. Supported: ~5 minutes (native CGM, as in "
                   "training) or ~15 minutes (FreeStyle Libre historic data, external-validation only).")
    if ds.sampling_profile == "15min":
        mard = (external or {}).get("MAIN", {}).get("MARD_percent", {}).get("mean")
        ds.warnings.append(
            "Readings are ~15 minutes apart, but the model was trained on 5-minute CGM data. They are placed on the "
            "5-minute grid with the research forward-fill rule (no interpolation), so trend and lag features "
            "contain repeated values. This is NOT equivalent to native 5-minute CGM."
            + (f" Research accuracy on such data (HUPA-UCM): MARD {mard}%." if mard is not None else ""))
        ds.transformations.append("15-minute readings aligned to the 5-minute grid by forward fill "
                                  "(tolerance 5 min, then up to 30 min); nothing interpolated.")


def ingest_upload(files: list[UploadedFile], external_validation: dict | None = None) -> CanonicalDataset:
    if not files:
        raise _bad(ErrorCode.INVALID_FORMAT, "No file uploaded.")
    if len(files) > MAX_FILES:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, f"At most {MAX_FILES} files per upload.")
    for f in files:
        f.kind = detect_file_type(f)
    kinds = {f.kind for f in files}
    if kinds == {OHIO}:
        if len(files) > 2:
            raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Upload at most 2 OhioT1DM files (training and testing).")
        ds = adapt_ohio(files)
    elif kinds <= {LIBRE, HUPA_PREP}:
        ds = adapt_hupa(files)
    elif kinds == {NORMALIZED}:
        ds = adapt_normalized(files)
    else:
        raise _bad(ErrorCode.UNSUPPORTED_FORMAT, "Files of different dataset types cannot be combined.")
    if ds.events["glucose_mg_dl"].notna().sum() == 0:
        raise _bad(ErrorCode.MISSING_COLUMNS, "The dataset contains no glucose readings.")
    detect_sampling(ds, external_validation)
    if ds.events["bolus_units"].notna().sum() == 0 and ds.events["carbs_g"].notna().sum() == 0:
        ds.warnings.append("No insulin or carbohydrate records: those features are zero, as for research patients "
                           "with missing records.")
    return ds
