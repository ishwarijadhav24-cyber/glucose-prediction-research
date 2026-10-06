"""
Objective 3 follow-up: explain the OhioT1DM -> HUPA-UCM accuracy drop.

Leave-one-subject-out on the 12 OhioT1DM patients with the unchanged Objective 1
pipeline (objective3_common.py). Training always uses the other 11 patients'
ORIGINAL 5-minute data. Only the held-out test data changes:
  A  original 5-min data, Objective 1 rule (target and glucose at t finite)
  B  original 5-min data, real readings only (is_real at t and at t+30 min)
  C  simulated 15-min data (readings >= 14 min apart, start offsets 0/1/2),
     grid rebuilt with the causal rule, evaluated with rule B; mean of 3 offsets
is_real(t) = Step A of the causal rule found a raw reading (at most 5 min old),
the same definition as the HUPA-UCM parser. Only OhioT1DM data is read.
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree

from data_processing import create_features, load_subject_data
from evaluate_loso import (calculate_mae, calculate_mard, calculate_r2, calculate_rmse,
                           evaluate_clarke_ega)
from objective3_common import (MODEL_NAMES, OHIO_DIR, TRAINED_MODELS, apply_preprocessing,
                               build_model, feature_columns, fit_preprocessing, load_ohio_features)

RES = "results_objective3/followup"
OUT = "outputs_objective3/followup"
CKPT_ROWS = os.path.join(RES, "followup_checkpoint_rows.csv")
CKPT_INTERVALS = os.path.join(RES, "followup_checkpoint_intervals.csv")
HUPA_SUMMARY = "results_objective3/hupa_summary_main.csv"   # results file, not data
OBJ1_SUMMARY = "results_causal/loso_summary.csv"
CONDITIONS = ["A", "B", "C"]
OFFSETS = [0, 1, 2]
MIN_GAP = pd.Timedelta("14min")
FIVE, THIRTY = pd.Timedelta("5min"), pd.Timedelta("30min")
METRICS = ["MARD", "RMSE", "MAE", "R2", "EGA_A", "EGA_B", "EGA_A+B"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#898781"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def raw_glucose(pid):
    """Raw OhioT1DM CGM readings (all files of the patient, as load_subject_data)."""
    ts, val = [], []
    for f in sorted(glob.glob(os.path.join(OHIO_DIR, f"{pid}*ws-*.xml"))):
        for e in etree.parse(f).getroot().findall(".//glucose_level/event"):
            ts.append(e.get("ts"))
            val.append(float(e.get("value")))
    idx = pd.to_datetime(pd.Series(ts), format="%d-%m-%Y %H:%M:%S")
    assert idx.notna().all(), f"{pid}: unparseable timestamps"
    s = pd.Series(val, index=pd.DatetimeIndex(idx)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def subsample(raw, offset):
    """Keep reading #offset, then each next reading >= 14 min after the last kept one."""
    keep, last = [], None
    for i, t in enumerate(raw.index[offset:], start=offset):
        if last is None or t - last >= MIN_GAP:
            keep.append(i)
            last = t
    return raw.iloc[keep]


def step_a(raw, index):
    return raw.reindex(index, method="ffill", tolerance=FIVE)


def real_mask(grid_index, is_real, feats):
    """is_real at t and at t+30 min (on the full grid, before create_features drops rows)."""
    real = pd.Series(is_real.values, index=grid_index)
    real_t30 = real.reindex(grid_index + THIRTY, fill_value=False).values
    mask_grid = pd.Series(real.values & real_t30, index=grid_index)
    ts = pd.DatetimeIndex(feats["ts"])
    return mask_grid.reindex(ts).fillna(False).values & feats["target"].notna().values


def metrics(y, p):
    a, b, ab, _ = evaluate_clarke_ega(y, p)
    return {"MARD": calculate_mard(y, p), "RMSE": calculate_rmse(y, p), "MAE": calculate_mae(y, p),
            "R2": calculate_r2(y, p), "EGA_A": a, "EGA_B": b, "EGA_A+B": ab}


def evaluate(pid, cond, feats, mask, grid_glucose, fitted, imputer, scaler, cols, offset=None):
    # rows where glucose at t is not finite cannot be scored (Persistence undefined); counted, not hidden
    finite = np.isfinite(feats["glucose"].values) & np.isfinite(feats["target"].values)
    n_nonfinite = int((mask & ~finite).sum())
    ev = feats[mask & finite]
    y = ev["target"].values
    X = apply_preprocessing(imputer, scaler, ev[cols].values)
    preds = {"Persistence": ev["glucose"].values}
    preds.update({m: fitted[m].predict(X) for m in TRAINED_MODELS})
    g_t = grid_glucose.reindex(pd.DatetimeIndex(ev["ts"])).values
    n_bad = int((~np.isclose(preds["Persistence"], g_t, rtol=0, atol=1e-9)).sum())
    return [{"patient": pid, "condition": cond, "offset": offset, "model": m, **metrics(y, preds[m]),
             "n_samples": int(len(y)), "persistence_mismatches": n_bad,
             "n_removed_nonfinite": n_nonfinite} for m in MODEL_NAMES]


def run_patient(pid, data, cols):
    train = pd.concat([d for p, d in data.items() if p != pid], ignore_index=True)
    imputer, scaler, X = fit_preprocessing(train[cols].values)
    fitted = {}
    for m in TRAINED_MODELS:
        fitted[m] = build_model(m)
        fitted[m].fit(X, train["target"].values)

    df = load_subject_data(pid, data_dir=OHIO_DIR)
    raw = raw_glucose(pid)
    feats = data[pid]
    rows = []
    # A: Objective 1 common mask
    mask_a = np.isfinite(feats["target"].values) & np.isfinite(feats["glucose"].values)
    rows += evaluate(pid, "A", feats, mask_a, df["glucose"], fitted, imputer, scaler, cols)
    # B: real readings only
    mask_b = real_mask(df.index, step_a(raw, df.index).notna(), feats)
    rows += evaluate(pid, "B", feats, mask_b, df["glucose"], fitted, imputer, scaler, cols)
    # C: simulated 15-minute sampling
    intervals = []
    for k in OFFSETS:
        sub = subsample(raw, k)
        intervals.append({"patient": pid, "offset": k, "n_kept": len(sub), "n_raw": len(raw),
                          "median_interval_min": pd.Series(sub.index).diff().median() / pd.Timedelta("1min")})
        dfc = df.copy()
        a = step_a(sub, df.index)
        dfc["glucose"] = a.ffill(limit=6)
        fc = create_features(dfc)
        rows += evaluate(pid, "C", fc, real_mask(df.index, a.notna(), fc), dfc["glucose"],
                         fitted, imputer, scaler, cols, offset=k)
    return rows, intervals


def plot(gap):
    g = gap[gap["metric"] == "MARD"].set_index("model").loc[MODEL_NAMES]
    series = [("ohio_A", "Ohio A: original, Objective 1 rule"), ("ohio_B", "Ohio B: original, real readings only"),
              ("ohio_C", "Ohio C: simulated 15-min"), ("hupa_main", "HUPA-UCM MAIN")]
    x, w = np.arange(len(g)), 0.2
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, (col, label) in enumerate(series):
        ax.bar(x + (i - 1.5) * (w + 0.01), g[col], width=w, color=COLORS[i], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_ylabel("MARD (%), mean across patients", color=INK_2)
    ax.set_title("Where the OhioT1DM -> HUPA-UCM MARD gap comes from", color=INK, loc="left", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "followup_gap_mard.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    data = load_ohio_features(OHIO_DIR)
    cols = feature_columns(data)
    patients = sorted(data)

    done_rows = pd.read_csv(CKPT_ROWS) if os.path.exists(CKPT_ROWS) else pd.DataFrame()
    done_int = pd.read_csv(CKPT_INTERVALS) if os.path.exists(CKPT_INTERVALS) else pd.DataFrame()
    done = set(done_rows["patient"].astype(str)) if len(done_rows) else set()
    for pid in patients:
        if pid in done:
            print(f"{pid}: in checkpoint, skipped")
            continue
        rows, ints = run_patient(pid, data, cols)
        done_rows = pd.concat([done_rows, pd.DataFrame(rows)], ignore_index=True)
        done_int = pd.concat([done_int, pd.DataFrame(ints)], ignore_index=True)
        done_rows.to_csv(CKPT_ROWS, index=False)
        done_int.to_csv(CKPT_INTERVALS, index=False)
        print(f"{pid}: done ({len(done_rows['patient'].unique())}/12)", flush=True)

    raw_rows = done_rows.copy()
    raw_rows["patient"] = raw_rows["patient"].astype(str)
    # condition C: average the 3 offsets per patient and model
    pr = (raw_rows.groupby(["patient", "condition", "model"], as_index=False)
          [METRICS + ["n_samples", "persistence_mismatches", "n_removed_nonfinite"]].mean())
    pr["persistence_mismatches"] = raw_rows.groupby(["patient", "condition", "model"])[
        "persistence_mismatches"].sum().values
    pr.to_csv(os.path.join(RES, "followup_patient_results.csv"), index=False)
    summ = pr.groupby(["model", "condition"])[METRICS + ["n_samples"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(RES, "followup_summary.csv"), index=False)

    hupa = pd.read_csv(HUPA_SUMMARY).set_index("model")
    gap_rows = []
    for metric in ("MARD", "RMSE"):
        s = summ.pivot(index="model", columns="condition", values=f"{metric}_mean")
        for m in MODEL_NAMES:
            a, b, c, h = s.loc[m, "A"], s.loc[m, "B"], s.loc[m, "C"], hupa.loc[m, f"{metric}_mean"]
            gap_rows.append({"metric": metric, "model": m, "ohio_A": a, "ohio_B": b, "ohio_C": c, "hupa_main": h,
                             "eval_rule_effect": b - a, "sampling_effect": c - b,
                             "remaining_effect": h - c, "total_gap": h - a})
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(os.path.join(RES, "gap_decomposition.csv"), index=False)
    plot(gap)

    # ---------------- checks ----------------
    log, ok_all = [], True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        log.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    obj1 = pd.read_csv(OBJ1_SUMMARY).set_index("model")["MARD_mean"]
    a_mean = summ[summ["condition"] == "A"].set_index("model")["MARD_mean"]
    diffs = {m: round(float(a_mean[m] - obj1[m]), 4) for m in MODEL_NAMES}
    check("1 condition A reproduces Objective 1 (mean MARD within 0.1)",
          all(abs(d) <= 0.1 for d in diffs.values()), f"differences {diffs}")
    vals = pr[METRICS + ["n_samples"]].to_numpy(dtype=float)
    check("2 12 patients x 6 models x 3 conditions, no duplicates, no NaN/inf",
          len(pr) == 12 * 6 * 3 and pr["patient"].nunique() == 12
          and not pr.duplicated(["patient", "condition", "model"]).any() and np.isfinite(vals).all()
          and len(raw_rows[raw_rows["condition"] == "C"]) == 12 * 6 * len(OFFSETS),
          f"{len(pr)} rows")
    med = done_int["median_interval_min"]
    check("3 condition C median interval 14-16 min for every patient and offset",
          bool(((med >= 14) & (med <= 16)).all()) and len(done_int) == 12 * len(OFFSETS),
          f"range {med.min():.2f}-{med.max():.2f} min")
    check("4 Persistence equals glucose at t on every evaluated row (A, B, C)",
          int(raw_rows["persistence_mismatches"].sum()) == 0,
          f"{int(raw_rows['persistence_mismatches'].sum())} mismatches")
    src = open(__file__, encoding="utf-8").read()
    pattern = re.compile("HUPA-" + "UCM/|import pars" + "ers|from pars" + "ers", re.IGNORECASE)
    hits = [i for i, line in enumerate(src.splitlines(), 1) if pattern.search(line)]
    check("5 no HUPA-UCM data path or parser import in this script", not hits, f"lines {hits}")
    log.append(f"OVERALL: {'PASS' if ok_all else 'FAIL'}")
    with open(os.path.join(RES, "followup_checks_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
