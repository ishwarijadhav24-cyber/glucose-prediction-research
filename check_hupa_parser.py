"""
Validation of the HUPA-UCM parser (parsers/hupa_ucm.py) on the real data.

Runs 13 checks on all 25 patients. Every check is a function that returns a list
of failure messages, so it can fail. Before the real run, a SELF-TEST injects a
known fault for every check (interpolation, look-ahead, wrong units, scrambled
timestamps, ...) and confirms the check reports it; a check that cannot detect its
fault makes the whole run FAIL.

Reference data (study window, bolus, carbs) is read here directly from
Preprocessed/<PID>.csv, independently of the parser.

Outputs:
  data/HUPA-UCM/parsed/<PID>.csv               ts, glucose, bolus, carbs, heart_rate, is_real
  results_objective3/hupa_parser_check.csv     per-patient table
  results_objective3/hupa_parser_check_log.txt
  outputs_objective3/hupa_coverage_by_patient.png
  outputs_objective3/hupa_parser_example.png
  outputs_objective3/hupa_evaluable_points.png
"""

import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing import create_features, parse_xml_file
from parsers.hupa_ucm import (
    EXCLUDED_MISSING_EVENTS, EXCLUDED_NO_CGM, GLUCOSE_MAX, GLUCOSE_MIN,
    NoCGMDataError, list_patients, list_patients_for_analysis,
    load_libre_glucose, parse_hupa_patient,
)

DATA_DIR = "data/HUPA-UCM"
OHIO_REFERENCE = "data/OhioT1DM/2018/train/559-ws-training.xml"
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
RESULTS_DIR = "results_objective3"
OUTPUTS_DIR = "outputs_objective3"
EXAMPLE_PATIENT = "HUPA0002P"

MIN_EVALUABLE = 500
MAX_UNPARSEABLE = 0.005
SANITY_MIN_FRACTION = 0.90
EXPECTED_MAIN, EXPECTED_ALL = 19, 23
FIVE_MIN = pd.Timedelta("5min")

CHECK_NAMES = {
    1: "SCHEMA", 2: "FEATURES", 3: "CAUSALITY", 4: "RAW GLUCOSE ONLY", 5: "CLIPPING",
    6: "UNITS", 7: "EXCLUSIONS", 8: "NO IMPUTATION", 9: "NON-EMPTY", 10: "DATES PARSED",
    11: "DATE SANITY", 12: "WINDOW", 13: "COUNTS",
}

COLORS = {"main": "#2a78d6", "all_only": "#eb6834", "excluded_no_cgm": "#898781"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


# ----------------------------------------------------------------------------
# Independent reference data
# ----------------------------------------------------------------------------

def read_preprocessed(pid):
    prep = pd.read_csv(os.path.join(DATA_DIR, "Preprocessed", f"{pid}.csv"), sep=";")
    prep["time"] = pd.to_datetime(prep["time"], format="%Y-%m-%dT%H:%M:%S")
    return prep


def latest_reading_at_or_before(grid_index, raw):
    """For each grid time t: time and value of the most recent raw reading <= t."""
    left = pd.DataFrame({"t": grid_index})
    right = pd.DataFrame({"raw_ts": raw.index, "raw_val": raw.values}).sort_values("raw_ts")
    merged = pd.merge_asof(left, right, left_on="t", right_on="raw_ts", direction="backward")
    merged.index = grid_index
    return merged


def evaluable_mask(df, is_real):
    """is_real at t AND at t+30 min (shift on the full grid) AND target not NaN."""
    real_t30 = is_real.shift(-6, fill_value=False).astype(bool)
    mask_grid = is_real.astype(bool) & real_t30
    feats = create_features(df)
    target_ok = feats["target"].notna().values
    return mask_grid.reindex(pd.DatetimeIndex(feats["ts"])).fillna(False).values & target_ok


# ----------------------------------------------------------------------------
# The checks. Each returns a list of failure messages (empty list = pass).
# ----------------------------------------------------------------------------

def check_schema(df, is_real, ohio):
    f = []
    if list(df.columns) != list(ohio.columns):
        f.append(f"columns {list(df.columns)} != OhioT1DM {list(ohio.columns)}")
    elif list(df.dtypes) != list(ohio.dtypes):
        f.append(f"dtypes {list(df.dtypes)} != OhioT1DM {list(ohio.dtypes)}")
    if df.index.name != "ts" or df.index.name != ohio.index.name:
        f.append(f"index name {df.index.name!r} (OhioT1DM {ohio.index.name!r})")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.dtype != ohio.index.dtype:
        f.append(f"index dtype {df.index.dtype} != OhioT1DM {ohio.index.dtype}")
    elif len(df) > 1:
        steps = pd.Series(df.index).diff().dropna().unique()
        if len(steps) != 1 or steps[0] != FIVE_MIN:
            f.append(f"index is not a regular 5-minute grid (steps: {steps[:5]})")
        if (df.index != df.index.floor("5min")).any():
            f.append("grid timestamps are not on 5-minute boundaries")
    if "is_real" in df.columns:
        f.append("is_real is a column of df")
    if not isinstance(is_real, pd.Series) or is_real.dtype != bool or not is_real.index.equals(df.index):
        f.append("is_real is not a bool Series on the df index")
    return f


def check_features(df, ohio_feature_cols):
    cols = list(create_features(df).columns)
    if cols != ohio_feature_cols:
        extra = [c for c in cols if c not in ohio_feature_cols]
        missing = [c for c in ohio_feature_cols if c not in cols]
        return [f"create_features columns differ (extra {extra}, missing {missing}, same order: False)"]
    return []


def check_causality(df, is_real, raw):
    f = []
    m = latest_reading_at_or_before(df.index, raw)
    age = m["t"] - m["raw_ts"]
    has_g = df["glucose"].notna()
    no_src = has_g & m["raw_ts"].isna()
    if no_src.any():
        f.append(f"{int(no_src.sum())} glucose rows have no raw reading at or before t "
                 f"(first {no_src.idxmax()})")
    bad_age = has_g & m["raw_ts"].notna() & ((age < pd.Timedelta(0)) | (age > pd.Timedelta("35min")))
    if bad_age.any():
        f.append(f"{int(bad_age.sum())} glucose rows whose latest raw reading is not 0-35 min old "
                 f"(first {bad_age.idxmax()}, age {age[bad_age].iloc[0]})")
    real = is_real.reindex(df.index).fillna(False).astype(bool)
    bad_real = real & ~(m["raw_ts"].notna() & (age >= pd.Timedelta(0)) & (age <= FIVE_MIN))
    if bad_real.any():
        f.append(f"{int(bad_real.sum())} is_real rows without a raw reading in the last 5 min "
                 f"(first {bad_real.idxmax()})")
    real_no_g = real & ~has_g
    if real_no_g.any():
        f.append(f"{int(real_no_g.sum())} is_real rows have NaN glucose")
    missed = ~real & m["raw_ts"].notna() & (age <= FIVE_MIN)
    if missed.any():
        f.append(f"{int(missed.sum())} rows with a raw reading in the last 5 min are not is_real "
                 f"(first {missed.idxmax()})")
    return f


def check_raw_only(df, raw):
    m = latest_reading_at_or_before(df.index, raw)
    has_g = df["glucose"].notna()
    expected = m["raw_val"].clip(GLUCOSE_MIN, GLUCOSE_MAX)
    bad = has_g & ~np.isclose(df["glucose"], expected, rtol=0, atol=1e-9, equal_nan=False)
    if bad.any():
        t = bad.idxmax()
        return [f"{int(bad.sum())} glucose values differ from the latest raw reading "
                f"(first {t}: parsed {df.at[t, 'glucose']}, raw {expected[t]})"]
    return []


def check_clipping(df):
    g = df["glucose"].dropna()
    if ((g < GLUCOSE_MIN) | (g > GLUCOSE_MAX)).any():
        return [f"glucose outside [40, 400]: min {g.min()}, max {g.max()}"]
    return []


def recorded_per_bucket(prep, grid):
    t = prep["time"].dt.floor("5min")
    b = prep["bolus_volume_delivered"].fillna(0)
    bolus = b.where(b > 0, 0.0).groupby(t).sum()  # bolus <= 0 is an invalid record (no event)
    carbs = (prep["carb_input"].fillna(0) * 10.0).groupby(t).sum()
    return bolus.reindex(grid, fill_value=0.0), carbs.reindex(grid, fill_value=0.0)


def check_units(df, prep):
    f = []
    exp_bolus, exp_carbs = recorded_per_bucket(prep, df.index)
    for col, exp in (("bolus", exp_bolus), ("carbs", exp_carbs)):
        bad = ~np.isclose(df[col].values, exp.values, rtol=0, atol=1e-6)
        if bad.any():
            i = int(np.argmax(bad))
            f.append(f"{col}: {int(bad.sum())} buckets differ from the recorded value "
                     f"(first {df.index[i]}: parsed {df[col].iloc[i]}, recorded {exp.iloc[i]})")
    return f


def check_no_imputation(df, prep, n_invalid_reported=None):
    """Nothing may be added: parsed events/totals never exceed the recorded POSITIVE events."""
    f = []
    start, end = prep["time"].min(), prep["time"].max()
    in_win = prep[(prep["time"] >= start) & (prep["time"] <= end)]
    exp_bolus, exp_carbs = recorded_per_bucket(in_win, df.index)
    raw_bolus = in_win["bolus_volume_delivered"].fillna(0)
    n_invalid = int((raw_bolus < 0).sum())
    if n_invalid_reported is not None and n_invalid_reported != n_invalid:
        f.append(f"bolus: parser reports {n_invalid_reported} invalid boluses removed, "
                 f"Preprocessed has {n_invalid} negative records")
    for col, rec, exp in (("bolus", raw_bolus.where(raw_bolus > 0, 0.0), exp_bolus),
                          ("carbs", in_win["carb_input"].fillna(0) * 10.0, exp_carbs)):
        parsed = df[col]
        if (parsed < 0).any():
            f.append(f"{col}: negative values")
        if int((parsed != 0).sum()) > int((rec > 0).sum()):
            f.append(f"{col}: {int((parsed != 0).sum())} parsed events > {int((rec > 0).sum())} recorded positive")
        if parsed.sum() > rec[rec > 0].sum() + 1e-6:
            f.append(f"{col}: parsed total {parsed.sum():.3f} > recorded positive total {rec[rec > 0].sum():.3f}")
        invented = (parsed != 0) & (exp == 0)
        if invented.any():
            f.append(f"{col}: {int(invented.sum())} events in buckets with no recorded positive event "
                     f"(first {invented.idxmax()})")
    if df["heart_rate"].notna().any():
        f.append(f"heart_rate has {int(df['heart_rate'].notna().sum())} non-NaN values")
    return f


def check_non_empty(readings_in_window, n_evaluable):
    f = []
    if readings_in_window <= 0:
        f.append("no raw readings inside the study window")
    if n_evaluable < MIN_EVALUABLE:
        f.append(f"only {n_evaluable} evaluable points (< {MIN_EVALUABLE})")
    return f


def check_dates_parsed(meta):
    f = []
    n0, bad = meta["n_type0_total"], meta["n_unparseable_timestamps"]
    if n0 == 0 or bad / n0 > MAX_UNPARSEABLE:
        f.append(f"{bad}/{n0} unparseable type-0 timestamps (> {MAX_UNPARSEABLE:.1%})")
    accounted = (meta["readings_before_window"] + meta["readings_in_window"]
                 + meta["readings_after_window"] + meta["n_duplicate_timestamps"]
                 + bad + meta["n_bad_glucose"])
    if accounted != n0:
        f.append(f"type-0 rows not accounted for: {n0} read, {accounted} explained "
                 f"(before+inside+after+duplicates+unparseable+bad glucose)")
    return f


def check_date_sanity(raw):
    if len(raw) < 2:
        return ["fewer than 2 raw readings in the window"]
    gaps = pd.Series(raw.index).diff().dropna()
    ok = ((gaps >= pd.Timedelta("10min")) & (gaps <= pd.Timedelta("20min"))).mean()
    if ok < SANITY_MIN_FRACTION:
        return [f"only {ok:.1%} of consecutive readings are 10-20 min apart (< {SANITY_MIN_FRACTION:.0%})"]
    return []


def check_window(df, raw_all, start, end):
    f = []
    if df.index.min() < start or df.index.max() > end:
        f.append(f"grid {df.index.min()} - {df.index.max()} outside window {start} - {end}")
    m = latest_reading_at_or_before(df.index, raw_all)
    has_g = df["glucose"].notna()
    outside = has_g & ((m["raw_ts"] < start) | (m["raw_ts"] > end) | m["raw_ts"].isna())
    if outside.any():
        f.append(f"{int(outside.sum())} glucose values come from a reading outside the window "
                 f"(first {outside.idxmax()})")
    return f


def check_exclusions(main, all_, list_fn):
    f = []
    bad_main = sorted(set(main) & (EXCLUDED_NO_CGM | EXCLUDED_MISSING_EVENTS))
    if bad_main:
        f.append(f"MAIN contains excluded patients {bad_main}")
    bad_all = sorted(set(all_) & EXCLUDED_NO_CGM)
    if bad_all:
        f.append(f"ALL contains no-CGM patients {bad_all}")
    missing = sorted(EXCLUDED_MISSING_EVENTS - set(all_))
    if missing:
        f.append(f"ALL is missing {missing}")
    try:
        list_fn(analysis="mian")
        f.append("invalid analysis name did not raise ValueError")
    except ValueError:
        pass
    return f


def check_counts(main, all_):
    f = []
    if len(main) != EXPECTED_MAIN:
        f.append(f"MAIN has {len(main)} patients (expected {EXPECTED_MAIN})")
    if len(all_) != EXPECTED_ALL:
        f.append(f"ALL has {len(all_)} patients (expected {EXPECTED_ALL})")
    return f


# ----------------------------------------------------------------------------
# Self-test: every check must detect an injected fault
# ----------------------------------------------------------------------------

def self_test(ohio, ohio_feature_cols):
    pid = EXAMPLE_PATIENT
    df, is_real, meta = parse_hupa_patient(pid)
    raw, _ = load_libre_glucose(pid)
    raw_all, _ = load_libre_glucose(pid, restrict_to_window=False)
    prep = read_preprocessed(pid)
    start, end = prep["time"].min(), prep["time"].max()
    main = list_patients_for_analysis("main")
    all_ = list_patients_for_analysis("all")

    step_a = raw.clip(GLUCOSE_MIN, GLUCOSE_MAX).reindex(df.index, method="ffill", tolerance=FIVE_MIN)

    def with_glucose(values):
        d = df.copy()
        d["glucose"] = values
        return d

    lookahead = with_glucose(raw.clip(GLUCOSE_MIN, GLUCOSE_MAX)
                             .reindex(df.index, method="bfill", tolerance=pd.Timedelta("35min")))
    interpolated = with_glucose(step_a.interpolate(limit_area="inside", limit=6))
    out_of_range = with_glucose(df["glucose"].copy())
    out_of_range.iloc[out_of_range["glucose"].notna().values.argmax(), 0] = 450.0
    wrong_units = df.copy()
    wrong_units["carbs"] = wrong_units["carbs"] / 10.0
    invented = df.copy()
    invented.loc[(invented["bolus"] == 0).idxmax(), "bolus"] = 1.0  # bolus where none was recorded
    hr_filled = df.copy()
    hr_filled["heart_rate"] = 80.0
    # Check 11 faults: (a) time fields mixed up (hour <-> minute), (b) a second, offset
    # copy of the stream not de-duplicated. A consistent day/month swap of whole days
    # cannot be detected by the 10-20 min spacing rule (days keep their internal spacing).
    scrambled_idx = [t.replace(hour=t.minute, minute=t.hour) if t.minute < 24 else t for t in raw.index]
    scrambled = pd.Series(raw.values, index=pd.DatetimeIndex(scrambled_idx)).sort_index()
    doubled = pd.concat([raw, raw.set_axis(raw.index + pd.Timedelta("7min"))]).sort_index()
    early_grid = df.reindex(pd.date_range(start - pd.Timedelta("1D"), df.index.max(), freq="5min"))
    early_grid.index.name = "ts"
    early_grid["glucose"] = raw_all.clip(GLUCOSE_MIN, GLUCOSE_MAX).reindex(
        early_grid.index, method="ffill", tolerance=FIVE_MIN).ffill(limit=6)
    bad_meta = dict(meta, n_unparseable_timestamps=10, n_type0_total=1000)
    no_raise = lambda analysis="main": main  # accepts any analysis name
    tests = {
        1: [check_schema(df.assign(is_real=is_real), is_real, ohio),
            check_schema(df.reset_index(drop=True), is_real, ohio)],
        2: [check_features(df.assign(steps=0.0), ohio_feature_cols)],
        3: [check_causality(lookahead, is_real, raw),
            check_causality(df, is_real.shift(-1, fill_value=False), raw)],
        4: [check_raw_only(interpolated, raw), check_raw_only(lookahead, raw)],
        5: [check_clipping(out_of_range)],
        6: [check_units(wrong_units, prep)],
        7: [check_exclusions(main + ["HUPA0011P"], all_, list_patients_for_analysis),
            check_exclusions(main, all_, no_raise)],
        8: [check_no_imputation(invented, prep), check_no_imputation(hr_filled, prep)],
        9: [check_non_empty(meta["readings_in_window"], 100), check_non_empty(0, 1000)],
        10: [check_dates_parsed(bad_meta),
             check_dates_parsed(dict(meta, readings_in_window=meta["readings_in_window"] - 5))],
        11: [check_date_sanity(scrambled), check_date_sanity(doubled)],
        12: [check_window(early_grid, raw_all, start, end)],
        13: [check_counts(main[:-1], all_), check_counts(main, all_ + ["HUPA0009P"])],
    }
    detected = {}
    for k, results in tests.items():
        detected[k] = all(len(r) > 0 for r in results)
    return detected


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _group(row):
    if row["status"] == "excluded_no_cgm":
        return "excluded_no_cgm"
    return "main" if row["in_main"] else "all_only"


LABELS = {"main": "MAIN + ALL", "all_only": "ALL only (missing events)",
          "excluded_no_cgm": "Excluded (no Libre CGM)"}


def plot_bars(table, column, ylabel, title, threshold, threshold_label, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    groups = table.apply(_group, axis=1).values
    x = np.arange(len(table))
    for g in ("main", "all_only", "excluded_no_cgm"):
        sel = groups == g
        if sel.any():
            heights = table[column].fillna(0).values[sel]
            ax.bar(x[sel], heights, color=COLORS[g], label=LABELS[g], width=0.7)
            for xi, h in zip(x[sel], heights):  # label zero bars (no-CGM patients)
                if h == 0:
                    ax.text(xi, 0, "no CGM", rotation=90, ha="center", va="bottom",
                            fontsize=7, color=INK_2)
    ax.axhline(threshold, color=INK_2, linestyle="--", linewidth=1, label=threshold_label)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table["patient"], rotation=90)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, loc="left", pad=28)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_parser_example(pid=EXAMPLE_PATIENT):
    df, is_real, _ = parse_hupa_patient(pid)
    raw, _ = load_libre_glucose(pid)
    prep = read_preprocessed(pid).set_index("time")
    start = df.index[len(df) // 2].floor("1D") + pd.Timedelta("6h")
    end = start + pd.Timedelta("12h")
    sl = slice(start, end)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prep.loc[sl].index, prep.loc[sl, "glucose"], linestyle="--", linewidth=1.5,
            color="#898781", label="Preprocessed glucose (interpolated, NOT used)")
    ax.step(df.loc[sl].index, df.loc[sl, "glucose"], where="post", linewidth=2,
            color=COLORS["main"], label="Parsed glucose (causal, 5-min grid)")
    ax.plot(raw.loc[sl].index, raw.loc[sl].values, "o", markersize=6, color=COLORS["all_only"],
            markeredgecolor="white", markeredgewidth=1, label="Raw Libre type-0 readings", zorder=3)
    ax.set_ylabel("Glucose (mg/dL)", color=INK_2)
    ax.set_title(f"{pid}: raw readings vs parsed vs preprocessed ({start:%Y-%m-%d %H:%M} + 12 h)",
                 color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "hupa_parser_example.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PARSED_DIR, exist_ok=True)
    old_diag = os.path.join(OUTPUTS_DIR, "hupa0009_diagnostic.png")
    if os.path.exists(old_diag):
        os.remove(old_diag)
        log(f"Deleted {old_diag}")

    ohio = parse_xml_file(OHIO_REFERENCE)
    ohio_feature_cols = list(create_features(ohio).columns)

    failures = {k: [] for k in CHECK_NAMES}

    log("=" * 70)
    log("SELF-TEST: each check must detect an injected fault")
    log("=" * 70)
    try:
        detected = self_test(ohio, ohio_feature_cols)
    except Exception:
        detected = {k: False for k in CHECK_NAMES}
        log(traceback.format_exc())
    for k in CHECK_NAMES:
        log(f"  CHECK {k:2d} {CHECK_NAMES[k]:<17} detects injected fault: {'YES' if detected.get(k) else 'NO'}")
        if not detected.get(k):
            failures[k].append("SELF-TEST: check did not detect its injected fault")

    patients = list_patients()
    main_list = list_patients_for_analysis("main")
    all_list = list_patients_for_analysis("all")
    failures[7] += check_exclusions(main_list, all_list, list_patients_for_analysis)
    failures[13] += check_counts(main_list, all_list)

    log("")
    log("=" * 70)
    log(f"PER-PATIENT CHECKS ({len(patients)} patients)")
    log("=" * 70)
    rows = []
    for pid in patients:
        status = ("excluded_no_cgm" if pid in EXCLUDED_NO_CGM else
                  "excluded_missing_events" if pid in EXCLUDED_MISSING_EVENTS else "included")
        prep = read_preprocessed(pid)
        start, end = prep["time"].min(), prep["time"].max()
        window_days = (end - start) / pd.Timedelta("1D")
        row = {"patient": pid, "status": status, "in_main": pid in main_list, "in_all": pid in all_list,
               "window_start": start, "window_end": end, "window_days": round(window_days, 1)}
        pfail = {}

        def add(k, msgs):
            if msgs:
                pfail.setdefault(k, []).extend(msgs)

        if pid in EXCLUDED_NO_CGM:
            for fn in (lambda: parse_hupa_patient(pid), lambda: load_libre_glucose(pid)):
                try:
                    fn()
                    add(9, ["expected NoCGMDataError, but the patient was parsed"])
                except NoCGMDataError as e:
                    row["note"] = str(e)
                except Exception as e:
                    add(9, [f"expected NoCGMDataError, got {type(e).__name__}: {e}"])
            stale = os.path.join(PARSED_DIR, f"{pid}.csv")
            if os.path.exists(stale):
                os.remove(stale)
                log(f"  {pid}: removed stale {stale}")
            row.update({"layout": "", "date_format": "", "readings_in_window": 0,
                        "coverage_pct": 0.0, "n_evaluable_points": 0})
        else:
            try:
                df, is_real, meta = parse_hupa_patient(pid)
                raw, _ = load_libre_glucose(pid)
                raw_all, _ = load_libre_glucose(pid, restrict_to_window=False)
            except Exception as e:
                for k in (1, 3, 4, 9, 10):
                    add(k, [f"parser raised {type(e).__name__}: {e}"])
                row["note"] = f"{type(e).__name__}: {e}"
                df = None
            if df is not None:
                ev = evaluable_mask(df, is_real)
                n_eval = int(ev.sum())
                checks = {
                    1: lambda: check_schema(df, is_real, ohio),
                    2: lambda: check_features(df, ohio_feature_cols),
                    3: lambda: check_causality(df, is_real, raw),
                    4: lambda: check_raw_only(df, raw),
                    5: lambda: check_clipping(df),
                    6: lambda: check_units(df, prep),
                    8: lambda: check_no_imputation(df, prep, meta["n_invalid_boluses_removed"]),
                    9: lambda: check_non_empty(meta["readings_in_window"], n_eval),
                    10: lambda: check_dates_parsed(meta),
                    11: lambda: check_date_sanity(raw),
                    12: lambda: check_window(df, raw_all, start, end),
                }
                for k, fn in checks.items():
                    try:
                        add(k, fn())
                    except Exception as e:
                        add(k, [f"check crashed: {type(e).__name__}: {e}"])

                gaps = pd.Series(raw.index).diff().dropna()
                rec_bolus = prep["bolus_volume_delivered"].fillna(0).clip(lower=0)
                rec_carbs = prep["carb_input"].fillna(0) * 10
                row.update({
                    "layout": meta["layout"], "delimiter": meta["delimiter"],
                    "encoding": meta["encoding"], "date_format": meta["date_format"],
                    "n_libre_files": meta["n_files"], "n_type0_total": meta["n_type0_total"],
                    "n_unparseable_timestamps": meta["n_unparseable_timestamps"],
                    "n_bad_glucose": meta["n_bad_glucose"],
                    "n_duplicate_timestamps": meta["n_duplicate_timestamps"],
                    "n_conflicting_duplicates": meta["n_conflicting_duplicates"],
                    "readings_before_window": meta["readings_before_window"],
                    "readings_in_window": meta["readings_in_window"],
                    "readings_after_window": meta["readings_after_window"],
                    "coverage_pct": round(100 * meta["readings_in_window"] / (window_days * 96), 1),
                    "pct_gaps_10_20min": round(100 * ((gaps >= pd.Timedelta("10min")) &
                                                      (gaps <= pd.Timedelta("20min"))).mean(), 1),
                    "grid_start": df.index.min(), "grid_end": df.index.max(), "n_grid_rows": len(df),
                    "pct_grid_rows_real": round(100 * is_real.mean(), 1),
                    "pct_grid_rows_glucose": round(100 * df["glucose"].notna().mean(), 1),
                    "glucose_min": df["glucose"].min(), "glucose_max": df["glucose"].max(),
                    "n_clipped_below_40": meta["n_clipped_below_40"],
                    "n_clipped_above_400": meta["n_clipped_above_400"],
                    "bolus_total_parsed": round(df["bolus"].sum(), 3),
                    "bolus_total_recorded_window": round(rec_bolus.sum(), 3),
                    "carbs_g_total_parsed": round(df["carbs"].sum(), 1),
                    "carbs_g_total_recorded_window": round(rec_carbs.sum(), 1),
                    "n_invalid_boluses_removed": meta["n_invalid_boluses_removed"],
                    "n_bolus_events_outside_grid": meta["n_bolus_events_outside_grid"],
                    "n_carb_events_outside_grid": meta["n_carb_events_outside_grid"],
                    "n_evaluable_points": n_eval,
                })
                out = df.copy()
                out["is_real"] = is_real
                out.to_csv(os.path.join(PARSED_DIR, f"{pid}.csv"))

        for k, msgs in pfail.items():
            failures[k] += [f"{pid}: {m}" for m in msgs]
        row["failed_checks"] = ",".join(str(k) for k in sorted(pfail))
        rows.append(row)
        log(f"  {pid} [{status}] {'OK' if not pfail else 'FAILED checks ' + row['failed_checks']}"
            + (f" ({row['note']})" if row.get("note") else ""))
        for k in sorted(pfail):
            for m in pfail[k]:
                log(f"      CHECK {k}: {m}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "hupa_parser_check.csv"), index=False)

    plot_bars(table, "coverage_pct", "Libre readings / expected (96 per day), %",
              "FreeStyle Libre coverage inside the study window", 70, "70 % coverage",
              os.path.join(OUTPUTS_DIR, "hupa_coverage_by_patient.png"))
    plot_bars(table, "n_evaluable_points", "Evaluable points (real at t and t+30 min)",
              "Evaluable 30-minute prediction points per patient", MIN_EVALUABLE, f"{MIN_EVALUABLE} points",
              os.path.join(OUTPUTS_DIR, "hupa_evaluable_points.png"))
    plot_parser_example()

    log("")
    log("=" * 70)
    log("CHECKS")
    log("=" * 70)
    for k, name in CHECK_NAMES.items():
        log(f"CHECK {k:2d} {name:<17} {'PASS' if not failures[k] else 'FAIL'}")
        for m in failures[k][:20]:
            log(f"      - {m}")
    overall = all(not v for v in failures.values())
    log(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    log("")
    cols = ["patient", "status", "layout", "date_format", "readings_in_window", "window_days",
            "coverage_pct", "n_evaluable_points"]
    log(table[cols].to_string(index=False))
    tot_main = int(table.loc[table["in_main"], "n_evaluable_points"].sum())
    tot_all = int(table.loc[table["in_all"], "n_evaluable_points"].sum())
    log("")
    log(f"Total evaluable points, MAIN ({int(table['in_main'].sum())} patients): {tot_main}")
    log(f"Total evaluable points, ALL  ({int(table['in_all'].sum())} patients): {tot_all}")
    inv = table[table.get("n_invalid_boluses_removed", 0).fillna(0) > 0]
    log("Invalid boluses (<= 0) removed: " + (", ".join(f"{p} ({int(n)})" for p, n in
        zip(inv["patient"], inv["n_invalid_boluses_removed"])) or "none"))
    low = table[table["in_all"] & (table["coverage_pct"] < 70)]
    log("Patients below 70% coverage: " + (", ".join(f"{p} ({c}%)" for p, c in
                                                     zip(low["patient"], low["coverage_pct"])) or "none"))

    with open(os.path.join(RESULTS_DIR, "hupa_parser_check_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
