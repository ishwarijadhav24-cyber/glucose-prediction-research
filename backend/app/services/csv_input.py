"""Strict parser for the supported event CSV (see BACKEND_ARCHITECTURE.md section 6).

Columns (exactly these, any order): timestamp, glucose_mg_dl, bolus_units, carbs_g
Sampling is checked afterwards for all formats (datasets.detect_sampling).
- timestamp: local time 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS', no time zone
- each row carries at least one of the three values; empty cells mean "no event"
Error messages give row numbers and column names, never the uploaded values.
"""

import csv
import io
import re

import numpy as np
import pandas as pd

from backend.app.errors import BackendError, ErrorCode
from backend.app.services.features import EVENT_COLUMNS

VALUE_COLUMNS = ("glucose_mg_dl", "bolus_units", "carbs_g")
TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:[0-5]\d(:[0-5]\d)?$")   # no leap seconds, no T/zone
# Plausibility ranges for every upload format (OhioT1DM data contains bolus up to 25 U and meals up to 450 g).
RANGES = {"glucose_mg_dl": (40.0, 400.0), "bolus_units": (0.0, 25.0), "carbs_g": (0.0, 500.0)}
MAX_ROWS = 200_000


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BackendError(ErrorCode.INVALID_FILE, "The file must be UTF-8 encoded text.") from None


def _parse_ts(values: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    shape_ok = values.str.match(TS_PATTERN)
    for fmt in TS_FORMATS:
        todo = out.isna() & shape_ok
        if todo.any():
            out[todo] = pd.to_datetime(values[todo], format=fmt, errors="coerce")
    return out


def parse_events_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Bytes -> validated, chronologically sorted events DataFrame (+ warnings)."""
    text = _decode(data)
    if not text.strip():
        raise BackendError(ErrorCode.INVALID_FORMAT, "The file is empty.")
    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        body = list(reader)
    except csv.Error:
        raise BackendError(ErrorCode.INVALID_FORMAT, "The file is not valid CSV.") from None
    header = [h.strip() for h in header]
    missing = [c for c in EVENT_COLUMNS if c not in header]
    if missing:
        raise BackendError(ErrorCode.MISSING_COLUMNS, f"Missing required columns: {missing}.")
    extra = [c for c in header if c not in EVENT_COLUMNS]
    if extra or len(set(header)) != len(header):
        raise BackendError(ErrorCode.INVALID_FORMAT,
                           f"Only the columns {list(EVENT_COLUMNS)} are supported (unexpected or repeated columns).")
    body = [r for r in body if any(c.strip() for c in r)]
    if not body:
        raise BackendError(ErrorCode.INVALID_FORMAT, "The file has no data rows.")
    if len(body) > MAX_ROWS:
        raise BackendError(ErrorCode.INVALID_FORMAT, f"Too many rows (maximum {MAX_ROWS}).")
    bad_len = [i + 2 for i, r in enumerate(body) if len(r) != len(header)]
    if bad_len:
        raise BackendError(ErrorCode.INVALID_FORMAT, f"Wrong number of fields in row {bad_len[0]}.")
    raw = pd.DataFrame(body, columns=header).apply(lambda s: s.str.strip())
    rownum = pd.Series(np.arange(len(raw)) + 2, index=raw.index)   # 1-based, header = row 1

    ts = _parse_ts(raw["timestamp"])
    if ts.isna().any():
        raise BackendError(ErrorCode.INVALID_TIMESTAMP,
                           f"Row {int(rownum[ts.isna()].iloc[0])}: timestamp must be local time "
                           "'YYYY-MM-DD HH:MM[:SS]' without a time zone.")
    out = pd.DataFrame({"timestamp": ts})
    for col in VALUE_COLUMNS:
        cell = raw[col]
        num = pd.to_numeric(cell.where(cell != ""), errors="coerce")
        bad = (cell != "") & (num.isna() | ~np.isfinite(num.fillna(0)))
        if bad.any():
            raise BackendError(ErrorCode.INVALID_NUMERIC_VALUE, f"Row {int(rownum[bad].iloc[0])}: '{col}' is not a number.")
        lo, hi = RANGES[col]
        out_of_range = num.notna() & ((num < lo) | (num > hi))
        if out_of_range.any():
            raise BackendError(ErrorCode.INVALID_NUMERIC_VALUE,
                               f"Row {int(rownum[out_of_range].iloc[0])}: '{col}' must be between {lo:g} and {hi:g}.")
        out[col] = num.astype(float)
    empty = out[list(VALUE_COLUMNS)].isna().all(axis=1)
    if empty.any():
        raise BackendError(ErrorCode.INVALID_FORMAT, f"Row {int(rownum[empty].iloc[0])} has no value.")
    for col in VALUE_COLUMNS:
        dup = out.loc[out[col].notna(), "timestamp"].duplicated(keep=False)
        if dup.any():
            first = int(rownum[dup[dup].index].min())
            raise BackendError(ErrorCode.DUPLICATE_TIMESTAMP, f"Row {first}: duplicate timestamp for '{col}'.")

    warnings = []
    if not out["timestamp"].is_monotonic_increasing:
        warnings.append("Rows were not in chronological order; they were sorted.")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out[list(EVENT_COLUMNS)], warnings


def validate_ranges(events: pd.DataFrame, strict: bool = True) -> list[str]:
    """Canonical value check for every adapter (counts only, never values).

    strict=True (normalized CSV): values outside RANGES are rejected.
    strict=False (OhioT1DM / HUPA-UCM research formats): values are used as recorded, exactly as in
    the research pipeline; non-finite or negative values are still rejected, and values above the
    plausibility range are reported as warnings.
    """
    warnings = []
    for col in VALUE_COLUMNS:
        v = events[col].dropna()
        if not np.isfinite(v).all():
            raise BackendError(ErrorCode.INVALID_NUMERIC_VALUE, f"'{col}' contains non-finite values.")
        lo, hi = RANGES[col]
        n_out = int(((v < lo) | (v > hi)).sum())
        if strict and n_out:
            raise BackendError(ErrorCode.INVALID_NUMERIC_VALUE,
                               f"{n_out} '{col}' value(s) outside the supported range {lo:g}-{hi:g}.")
        if not strict:
            if (v < 0).any():
                raise BackendError(ErrorCode.INVALID_NUMERIC_VALUE, f"'{col}' contains negative values.")
            if n_out:
                warnings.append(f"{n_out} '{col}' value(s) outside the usual range {lo:g}-{hi:g}; used as recorded "
                                "(as in the research pipeline).")
    return warnings
