"""Test data: synthetic CGM events and a raw-event reader for OhioT1DM XML (tests only)."""

import numpy as np
import pandas as pd
from lxml import etree

COLS = ["timestamp", "glucose_mg_dl", "bolus_units", "carbs_g"]


def synthetic_events(start="2027-03-01 06:00:00", hours=30, seed=0, gaps=(), offset_s=97):
    """5-minute CGM readings (not on the grid: offset by offset_s seconds), meals and boluses.

    gaps: list of (start_offset_minutes, length_minutes) without glucose readings.
    """
    rng = np.random.RandomState(seed)
    t0 = pd.Timestamp(start)
    times = t0 + pd.to_timedelta(np.arange(0, hours * 60, 5), unit="min") + pd.Timedelta(seconds=offset_s)
    minutes = (times - t0) / pd.Timedelta("1min")
    g = 140 + 45 * np.sin(minutes / 180.0) + rng.normal(0, 6, len(times)).cumsum() * 0.3
    g = np.clip(np.round(g), 40, 400)
    keep = np.ones(len(times), bool)
    for gs, gl in gaps:
        keep &= ~((minutes >= gs) & (minutes < gs + gl))
    glucose = pd.DataFrame({"timestamp": times[keep], "glucose_mg_dl": g[keep]})
    ev = []
    for day in range(int(np.ceil(hours / 24)) + 1):
        for h, carbs, dose in ((7.5, 45, 4.5), (12.75, 70, 6.0), (19.25, 60, 5.0)):
            t = t0.normalize() + pd.Timedelta(days=day, hours=h) + pd.Timedelta(seconds=int(rng.randint(0, 299)))
            ev.append({"timestamp": t, "carbs_g": float(carbs)})
            ev.append({"timestamp": t + pd.Timedelta(minutes=int(rng.randint(-10, 10))), "bolus_units": dose})
    other = pd.DataFrame(ev)
    other = other[(other["timestamp"] >= times[0]) & (other["timestamp"] <= times[-1])]
    out = pd.concat([glucose, other], ignore_index=True).reindex(columns=COLS)
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def ohio_events(xml_path):
    """Raw glucose / bolus / meal events of one OhioT1DM XML file."""
    root = etree.parse(str(xml_path)).getroot()
    rows = []
    for tag, ts_attr, val_attr, col in (("glucose_level", "ts", "value", "glucose_mg_dl"),
                                        ("bolus", "ts_begin", "dose", "bolus_units"),
                                        ("meal", "ts", "carbs", "carbs_g")):
        for e in root.findall(f".//{tag}/event"):
            rows.append({"timestamp": e.get(ts_attr), col: float(e.get(val_attr))})
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def corrupt_after(events, t, seed=1):
    """Same events up to and including t; radically different events after t."""
    rng = np.random.RandomState(seed)
    before = events[events["timestamp"] <= t]
    after = events[events["timestamp"] > t].copy()
    after["glucose_mg_dl"] = np.where(after["glucose_mg_dl"].notna(),
                                      rng.choice([40.0, 400.0], len(after)), np.nan)
    after["bolus_units"] = np.where(after["bolus_units"].notna(), 25.0, np.nan)
    after["carbs_g"] = np.where(after["carbs_g"].notna(), 300.0, np.nan)
    extra = []
    for s in (1, 30, 59, 61, 119, 179, 241, 299, 301, 600, 1800):   # seconds after t, incl. inside bucket t
        ts = t + pd.Timedelta(seconds=s)
        extra += [{"timestamp": ts, "glucose_mg_dl": float(rng.choice([40, 400]))},
                  {"timestamp": ts, "bolus_units": 20.0}, {"timestamp": ts, "carbs_g": 250.0}]
    out = pd.concat([before, after, pd.DataFrame(extra)], ignore_index=True).reindex(columns=COLS)
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
