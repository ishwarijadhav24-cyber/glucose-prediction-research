"""Adapter around the research pipeline (data_processing.py, unchanged).

1. Events -> 5-minute causal grid:
   the events are written into an in-memory OhioT1DM-style XML and passed to
   parse_xml_file. The grid, Step A / Step B glucose rule and bolus/carb
   bucketing are therefore exactly the research code.
2. Grid -> 47 features at prediction time t:
   create_features drops rows whose 30-min target is unknown (the newest rows).
   The adapter keeps only grid rows <= t, appends placeholder rows after t (their
   count is measured from create_features), calls create_features unchanged, and
   returns the row at t restricted to feature_list. Every research feature is
   backward-looking, so the placeholders can only affect the target, which is
   discarded. Tests verify this.
"""

import io
from functools import lru_cache
from xml.sax.saxutils import quoteattr

import numpy as np
import pandas as pd

from data_processing import create_features, parse_xml_file

FIVE_MIN = pd.Timedelta("5min")
# History kept before t. The features look back at most ~60 rows (IOB kernel 48 + 12 lags);
# 24 h is far beyond that. Feature parity with the full-history research pipeline is tested.
CONTEXT = pd.Timedelta("24h")
EVENT_COLUMNS = ("timestamp", "glucose_mg_dl", "bolus_units", "carbs_g")
TS_FORMAT = "%d-%m-%Y %H:%M:%S"     # the OhioT1DM XML timestamp format
PLACEHOLDER_GLUCOSE = 100.0


@lru_cache(maxsize=1)
def pipeline_constants() -> dict:
    """Measure, from create_features itself, how many rows it needs and drops.

    horizon_rows      rows at the end with no target (dropped) = prediction horizon in steps
    min_history_rows  grid rows of glucose needed before every feature (except the
                      always-NaN heart_rate) is populated at t
    """
    n = 120
    idx = pd.date_range("2000-01-01", periods=n, freq="5min", name="ts")
    grid = pd.DataFrame({"glucose": 100.0 + np.arange(n) % 7, "bolus": 0.0, "carbs": 0.0,
                         "heart_rate": np.nan}, index=idx)
    feats = create_features(grid)
    horizon_rows = n - len(feats)
    cols = [c for c in feats.columns if c not in ("target", "ts", "heart_rate")]
    complete = feats[cols].notna().all(axis=1).values
    first = int(np.argmax(complete))
    if not complete[first:].all():
        raise RuntimeError("create_features produced NaN after its first complete row")
    return {"horizon_rows": int(horizon_rows), "min_history_rows": first + 1,
            "horizon_minutes": int(horizon_rows) * 5}


def _fmt(v: float) -> str:
    return repr(float(v))   # exact float round-trip


def events_to_xml(events: pd.DataFrame) -> bytes:
    parts = ['<patient id="upload">']
    for tag, col, ts_attr, val_attr in (("glucose_level", "glucose_mg_dl", "ts", "value"),
                                        ("bolus", "bolus_units", "ts_begin", "dose"),
                                        ("meal", "carbs_g", "ts", "carbs")):
        sel = events[events[col].notna()]
        parts.append(f"<{tag}>")
        parts += [f"<event {ts_attr}={quoteattr(t.strftime(TS_FORMAT))} {val_attr}={quoteattr(_fmt(v))}/>"
                  for t, v in zip(sel["timestamp"], sel[col])]
        parts.append(f"</{tag}>")
    parts.append("</patient>")
    return "".join(parts).encode("utf-8")


def build_grid(events: pd.DataFrame) -> pd.DataFrame:
    """Events -> research 5-minute grid, via parse_xml_file (unchanged)."""
    return parse_xml_file(io.BytesIO(events_to_xml(events)))


def features_at(grid: pd.DataFrame, t: pd.Timestamp, feature_list, placeholder_glucose=PLACEHOLDER_GLUCOSE,
                return_target=False):
    """The 47 research features at grid time t, computed only from grid rows <= t."""
    k = pipeline_constants()["horizon_rows"]
    hist = grid.loc[t - CONTEXT: t]
    if len(hist) == 0 or hist.index[-1] != t:
        raise KeyError("t is not a grid timestamp")
    future_idx = pd.date_range(t + FIVE_MIN, periods=k, freq="5min", name="ts")
    future = pd.DataFrame({"glucose": float(placeholder_glucose), "bolus": 0.0, "carbs": 0.0,
                           "heart_rate": np.nan}, index=future_idx)
    ext = pd.concat([hist[["glucose", "bolus", "carbs", "heart_rate"]], future])
    ext.index.name = "ts"
    feats = create_features(ext)
    row = feats[feats["ts"] == t]
    if len(row) != 1:
        raise KeyError("feature row at t was not produced")
    out = row.iloc[0][list(feature_list)].astype(float)
    if return_target:
        return out, float(row.iloc[0]["target"])
    return out
