"""
Research evidence report (read-only).

Uses only existing results files and data; no model is trained and no existing
file is modified. Writes CSVs to results_evidence/ and RESEARCH_EVIDENCE.md.

Statistics: Wilcoxon signed-rank (paired per patient), Mann-Whitney U (unpaired),
two-sided. 95% CIs: bootstrap over patients, 2000 resamples, random_state=42.
"""

import csv
import glob
import inspect
import json
import os
import pickle
import re

import joblib
import numpy as np
import pandas as pd
from lxml import etree
from scipy.stats import mannwhitneyu, wilcoxon

import data_processing
from data_processing import parse_xml_file
from objective3_common import MODEL_SETTINGS
from parsers.hupa_ucm import list_patients_for_analysis, load_libre_glucose

OUT = "results_evidence"
MD = "RESEARCH_EVIDENCE.md"
N_BOOT, SEED = 2000, 42
MODELS = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
RANGES = [("<70", -np.inf, 70), ("70-180", 70, 180), (">180", 180, np.inf)]
os.makedirs(OUT, exist_ok=True)
MDL = []  # markdown lines


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def boot_idx(n):
    return np.random.RandomState(SEED).randint(0, n, size=(N_BOOT, n))


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = x[boot_idx(len(x))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(x):
    x = np.asarray(x, float)
    lo, hi = boot_ci(x)
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "ci_lo": lo, "ci_hi": hi}


def wilcox(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        stat, p = np.nan, np.nan
    lo, hi = boot_ci(d)
    return {"n_patients": len(d), "mean_diff": d.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
            "median_diff": float(np.median(d)), "W": stat, "p": p}


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    return df


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def md_table(df, cols=None):
    df = df if cols is None else df[cols]
    MDL.append("| " + " | ".join(map(str, df.columns)) + " |")
    MDL.append("|" + "---|" * len(df.columns))
    for _, r in df.iterrows():
        MDL.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    MDL.append("")


def h(text, level=2):
    MDL.append(f"{'#' * level} {text}\n")


def say(text):
    MDL.append(f"- {text}")


def stats_table(df, group_cols, value_cols):
    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for v in value_cols:
            d = describe(g[v])
            row["n"] = d["n"]
            row.update({f"{v}_mean": d["mean"], f"{v}_sd": d["sd"], f"{v}_ci_lo": d["ci_lo"], f"{v}_ci_hi": d["ci_hi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def vs_persistence(pr, metric="MARD"):
    piv = pr.pivot(index="patient", columns="model", values=metric)
    rows = []
    for m in MODELS[1:]:
        w = wilcox(piv[m], piv["Persistence"])
        rows.append({"model": m, "n_beats_persistence": int((piv[m] < piv["Persistence"]).sum()), **w})
    return pd.DataFrame(rows)


def ci_txt(r, v):
    return f"{r[f'{v}_mean']:.2f} (95% CI {r[f'{v}_ci_lo']:.2f}-{r[f'{v}_ci_hi']:.2f})"


def range_metrics(pred, label):
    """pred: patient, model, y_true, y_pred. Pooled MARD/RMSE per range, CI by patient (cluster) bootstrap."""
    rows = []
    for m in MODELS:
        pm = pred[pred["model"] == m]
        for name, lo, hi in RANGES:
            sel = pm[(pm["y_true"] >= lo) & (pm["y_true"] < hi)] if name != "70-180" else \
                pm[(pm["y_true"] >= 70) & (pm["y_true"] <= 180)]
            e = sel["y_pred"] - sel["y_true"]
            agg = pd.DataFrame({"patient": sel["patient"], "are": (e.abs() / sel["y_true"]).values,
                                "se": (e ** 2).values}).groupby("patient").agg(
                n=("are", "size"), are=("are", "sum"), se=("se", "sum"))
            n, are, se = agg["n"].values, agg["are"].values, agg["se"].values
            if n.sum() == 0:
                continue
            idx = boot_idx(len(n))
            bn, ba, bs = n[idx].sum(1), are[idx].sum(1), se[idx].sum(1)
            ok = bn > 0
            rows.append({"dataset": label, "model": m, "range": name, "n_points": int(n.sum()),
                         "n_patients_with_points": int((n > 0).sum()),
                         "MARD": 100 * are.sum() / n.sum(),
                         "MARD_ci_lo": np.percentile(100 * ba[ok] / bn[ok], 2.5),
                         "MARD_ci_hi": np.percentile(100 * ba[ok] / bn[ok], 97.5),
                         "RMSE": np.sqrt(se.sum() / n.sum()),
                         "RMSE_ci_lo": np.percentile(np.sqrt(bs[ok] / bn[ok]), 2.5),
                         "RMSE_ci_hi": np.percentile(np.sqrt(bs[ok] / bn[ok]), 97.5)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# A. Objective 1
# ----------------------------------------------------------------------------
MDL += ["# Research evidence report", "",
        "Facts and numbers only, from existing results files and data (no retraining). "
        "Tests are two-sided: Wilcoxon signed-rank for paired per-patient comparisons, Mann-Whitney U "
        "for unpaired ones. 95% CIs are bootstrap over patients (2000 resamples, random_state=42). "
        "All CSVs are in `results_evidence/`.", ""]

o1 = pd.read_csv("results_causal/loso_patient_results.csv")
o1["patient"] = o1["patient"].astype(str)
o1["model"] = pd.Categorical(o1["model"], MODELS, ordered=True)
o1 = o1.sort_values(["model", "patient"])
o1["model"] = o1["model"].astype(str)
A1 = save(stats_table(o1, ["model"], ["MARD", "RMSE", "MAE", "EGA_A+B"]), "A1_obj1_model_stats.csv")
A2 = save(vs_persistence(o1), "A2_obj1_vs_persistence.csv")
piv1 = o1.pivot(index="patient", columns="model", values="MARD")
A3 = save(pd.DataFrame([{"comparison": f"LightGBM vs {b}", **wilcox(piv1["LightGBM"], piv1[b])}
                        for b in ("Ridge", "Stacking")]), "A3_obj1_lightgbm_pairs.csv")

h("A. Objective 1: OhioT1DM leave-one-subject-out (12 patients)")
md_table(A1, ["model", "n", "MARD_mean", "MARD_sd", "MARD_ci_lo", "MARD_ci_hi", "RMSE_mean", "RMSE_ci_lo",
              "RMSE_ci_hi", "MAE_mean", "EGA_A+B_mean"])
md_table(A2, ["model", "n_beats_persistence", "n_patients", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p"])
md_table(A3, ["comparison", "n_patients", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p"])
for _, r in A2.iterrows():
    say(f"{r['model']} has lower MARD than Persistence in {r['n_beats_persistence']}/12 patients "
        f"(mean difference {r['mean_diff']:.2f} points, Wilcoxon p={r['p']:.4f}).")
for _, r in A3.iterrows():
    say(f"{r['comparison']}: mean MARD difference {r['mean_diff']:.3f} points "
        f"(95% CI {r['diff_ci_lo']:.3f} to {r['diff_ci_hi']:.3f}), Wilcoxon p={r['p']:.4f}, n=12.")
MDL.append("")

# ----------------------------------------------------------------------------
# B. Objective 2
# ----------------------------------------------------------------------------
o2 = pd.read_csv("results_objective2/personalization_patient_results.csv")
o2["patient"] = o2["patient"].astype(str)
levels = sorted(o2["p_level"].unique())
B1 = save(stats_table(o2, ["model", "p_level"], ["MARD", "RMSE", "Delta_MARD"]), "B1_obj2_level_stats.csv")
rows, gains = [], []
for m in ("LightGBM", "Ridge"):
    piv = o2[o2["model"] == m].pivot(index="patient", columns="p_level", values="MARD")
    for lv in levels[1:]:
        rows.append({"model": m, "p_level": lv, "n_improved": int((piv[lv] < piv[0]).sum()),
                     **wilcox(piv[lv], piv[0])})
    for a, b in zip(levels[:-1], levels[1:]):
        gains.append({"model": m, "from_level": a, "to_level": b, **wilcox(piv[b], piv[a])})
B2 = save(pd.DataFrame(rows), "B2_obj2_vs_P0.csv")
B3 = save(pd.DataFrame(gains), "B3_obj2_consecutive_gains.csv")
days = o2[o2["model"] == "LightGBM"].assign(days=lambda d: d["n_personal_samples"] * 5 / 1440)
B4 = save(days.groupby("p_level")["days"].agg(median="median", min="min", max="max").reset_index(),
          "B4_obj2_levels_in_days.csv")

h("B. Objective 2: personalization (12 patients)")
say("p_level is the personalization level as stored in `personalization_patient_results.csv`; "
    "days = n_personal_samples x 5 min / 1440.")
MDL.append("")
md_table(B4)
md_table(B1, ["model", "p_level", "n", "MARD_mean", "MARD_sd", "MARD_ci_lo", "MARD_ci_hi", "RMSE_mean",
              "RMSE_ci_lo", "RMSE_ci_hi", "Delta_MARD_mean", "Delta_MARD_ci_lo", "Delta_MARD_ci_hi"])
md_table(B2, ["model", "p_level", "n_improved", "n_patients", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p"])
md_table(B3, ["model", "from_level", "to_level", "n_patients", "mean_diff", "diff_ci_lo", "diff_ci_hi", "p"])
for m in ("LightGBM", "Ridge"):
    sig = B2[(B2["model"] == m) & (B2["p"] < 0.05) & (B2["mean_diff"] < 0)]
    if len(sig):
        lv = sig["p_level"].min()
        r = sig[sig["p_level"] == lv].iloc[0]
        d = B4.set_index("p_level").loc[lv]
        say(f"{m}: the smallest level with a significant MARD improvement is p_level {lv} "
            f"(median {d['median']:.1f} days of data; {r['n_improved']}/12 improved; "
            f"mean change {r['mean_diff']:.2f} points, p={r['p']:.4f}).")
    else:
        say(f"{m}: no level shows a significant MARD improvement over P=0 (p<0.05).")
    g = B3[B3["model"] == m]
    say(f"{m}: mean MARD change between consecutive levels: " + "; ".join(
        f"{int(r.from_level)}->{int(r.to_level)}: {r.mean_diff:.2f} (p={r.p:.3f})" for r in g.itertuples()) + ".")
MDL.append("")

# ----------------------------------------------------------------------------
# C. Objective 3
# ----------------------------------------------------------------------------
o3 = pd.read_csv("results_objective3/hupa_patient_results.csv")
main_set = set(list_patients_for_analysis("main"))
C_stats, C_vs = [], []
for tag, sub in (("MAIN", o3[o3["patient"].isin(main_set)]), ("ALL", o3)):
    s = stats_table(sub, ["model"], ["MARD", "RMSE", "EGA_A+B"])
    s.insert(0, "analysis", tag)
    C_stats.append(s)
    v = vs_persistence(sub)
    v.insert(0, "analysis", tag)
    C_vs.append(v)
C1 = save(pd.concat(C_stats, ignore_index=True), "C1_hupa_model_stats.csv")
C2 = save(pd.concat(C_vs, ignore_index=True), "C2_hupa_vs_persistence.csv")
rows = []
for tag, sub in (("MAIN", o3[o3["patient"].isin(main_set)]), ("ALL", o3)):
    for m in MODELS:
        a = o1.loc[o1["model"] == m, "MARD"].values
        b = sub.loc[sub["model"] == m, "MARD"].values
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"comparison": f"Ohio LOSO vs HUPA {tag}", "model": m, "n_ohio": len(a), "n_hupa": len(b),
                     "ohio_mean": a.mean(), "hupa_mean": b.mean(), "U": U, "p": p})
C3 = save(pd.DataFrame(rows), "C3_ohio_vs_hupa_mannwhitney.csv")

fu = pd.read_csv("results_objective3/followup/followup_patient_results.csv")
fu["patient"] = fu["patient"].astype(str)
hmain = o3[o3["patient"].isin(main_set)]
rows = []
for metric in ("MARD", "RMSE"):
    piv = {c: fu[fu["condition"] == c].pivot(index="patient", columns="model", values=metric) for c in "ABC"}
    for m in MODELS:
        eval_rule = wilcox(piv["B"][m], piv["A"][m])
        sampling = wilcox(piv["C"][m], piv["B"][m])
        hv = hmain.loc[hmain["model"] == m, metric].values
        cv = piv["C"][m].values
        rng = np.random.RandomState(SEED)
        rem_boot = [hv[rng.randint(0, len(hv), len(hv))].mean() - cv[rng.randint(0, len(cv), len(cv))].mean()
                    for _ in range(N_BOOT)]
        rows.append({"metric": metric, "model": m, "ohio_A": piv["A"][m].mean(), "ohio_B": piv["B"][m].mean(),
                     "ohio_C": piv["C"][m].mean(), "hupa_main": hv.mean(),
                     "eval_rule_effect": eval_rule["mean_diff"], "eval_rule_ci_lo": eval_rule["diff_ci_lo"],
                     "eval_rule_ci_hi": eval_rule["diff_ci_hi"], "eval_rule_p": eval_rule["p"],
                     "sampling_effect": sampling["mean_diff"], "sampling_ci_lo": sampling["diff_ci_lo"],
                     "sampling_ci_hi": sampling["diff_ci_hi"], "sampling_p": sampling["p"],
                     "n_ohio_patients": sampling["n_patients"],
                     "remaining_effect": hv.mean() - cv.mean(),
                     "remaining_ci_lo": np.percentile(rem_boot, 2.5), "remaining_ci_hi": np.percentile(rem_boot, 97.5),
                     "remaining_mannwhitney_p": mannwhitneyu(hv, cv, alternative="two-sided")[1],
                     "n_hupa_patients": len(hv), "total_gap": hv.mean() - piv["A"][m].mean()})
C4 = save(pd.DataFrame(rows), "C4_gap_decomposition_ci.csv")

# glucose-range metrics
lp = pd.read_csv("results_causal/loso_predictions.csv")
pers_ok = lp[(lp["model"] == "Persistence") & lp["y_pred"].notna()][["patient", "timestamp"]]
lp = lp.merge(pers_ok, on=["patient", "timestamp"])  # Objective 1 common mask
n_check = lp[lp["model"] == "Persistence"].groupby("patient").size()
ref_n = o1[o1["model"] == "Persistence"].set_index("patient")["n_test_samples"]
mask_matches = bool((n_check.rename(index=str).reindex(ref_n.index) == ref_n).all())
hp = pd.read_csv("results_objective3/hupa_predictions.csv")
hp = hp[hp["in_main"]].melt(id_vars=["patient", "target"], value_vars=MODELS, var_name="model",
                            value_name="y_pred").rename(columns={"target": "y_true"})
C5 = save(pd.concat([range_metrics(lp, "Ohio LOSO"), range_metrics(hp, "HUPA MAIN")], ignore_index=True),
          "C5_metrics_by_glucose_range.csv")

h("C. Objective 3: external validation on HUPA-UCM")
md_table(C1, ["analysis", "model", "n", "MARD_mean", "MARD_sd", "MARD_ci_lo", "MARD_ci_hi", "RMSE_mean",
              "RMSE_ci_lo", "RMSE_ci_hi", "EGA_A+B_mean", "EGA_A+B_ci_lo", "EGA_A+B_ci_hi"])
md_table(C2, ["analysis", "model", "n_beats_persistence", "n_patients", "mean_diff", "diff_ci_lo",
              "diff_ci_hi", "p"])
md_table(C3, ["comparison", "model", "n_ohio", "n_hupa", "ohio_mean", "hupa_mean", "U", "p"])
say("Gap decomposition (from `results_objective3/followup/`). The eval-rule and sampling effects are paired "
    "per OhioT1DM patient (Wilcoxon, n=12). The remaining effect is HUPA MAIN (19) minus Ohio C (12), "
    "unpaired (independent bootstrap CI, Mann-Whitney p).")
MDL.append("")
md_table(C4[C4["metric"] == "MARD"], ["model", "ohio_A", "ohio_B", "ohio_C", "hupa_main", "eval_rule_effect",
                                      "eval_rule_p", "sampling_effect", "sampling_ci_lo", "sampling_ci_hi",
                                      "sampling_p", "remaining_effect", "remaining_ci_lo", "remaining_ci_hi",
                                      "remaining_mannwhitney_p", "total_gap"])
say(f"Glucose-range metrics are pooled over points; CIs come from a patient-level (cluster) bootstrap. "
    f"Ohio predictions are restricted to the Objective 1 common mask "
    f"(row counts equal n_test_samples: {mask_matches}).")
MDL.append("")
md_table(C5, ["dataset", "model", "range", "n_points", "n_patients_with_points", "MARD", "MARD_ci_lo",
              "MARD_ci_hi", "RMSE", "RMSE_ci_lo", "RMSE_ci_hi"])
for tag in ("MAIN", "ALL"):
    for _, r in C2[C2["analysis"] == tag].iterrows():
        say(f"HUPA {tag}: {r['model']} has lower MARD than Persistence in {r['n_beats_persistence']}/"
            f"{r['n_patients']} patients (mean difference {r['mean_diff']:.2f}, Wilcoxon p={r['p']:.4f}).")
for _, r in C3[C3["comparison"].str.endswith("MAIN")].iterrows():
    say(f"{r['model']}: per-patient MARD Ohio {r['ohio_mean']:.2f} vs HUPA MAIN {r['hupa_mean']:.2f} "
        f"(Mann-Whitney p={r['p']:.4f}, n=12 vs {r['n_hupa']}).")
for _, r in C4[C4["metric"] == "MARD"].iterrows():
    say(f"{r['model']}: sampling effect (C-B) {r['sampling_effect']:.2f} MARD points "
        f"(95% CI {r['sampling_ci_lo']:.2f} to {r['sampling_ci_hi']:.2f}, Wilcoxon p={r['sampling_p']:.4f}, n=12).")
MDL.append("")

# ----------------------------------------------------------------------------
# D. HUPA-UCM data quality: preprocessed glucose vs raw Libre
# ----------------------------------------------------------------------------
TOL_T, TOL_G = pd.Timedelta("2min").value, 1.0


def compare(prep_t, prep_g, raw):
    rt = raw.index.values.astype("int64")
    rv = raw.values.astype(float)
    t = prep_t.values.astype("int64")
    g = prep_g.values.astype(float)
    pos = np.searchsorted(rt, t, side="right") - 1
    has_prev, has_next = pos >= 0, pos + 1 < len(rt)
    d_prev = np.where(has_prev, t - rt[np.clip(pos, 0, None)], np.iinfo("int64").max)
    d_next = np.where(has_next, rt[np.clip(pos + 1, None, len(rt) - 1)] - t, np.iinfo("int64").max)
    coincide = np.minimum(d_prev, d_next) <= TOL_T
    inside = (t >= rt[0]) & (t <= rt[-1])
    interp = np.interp(t, rt, rv)
    last = np.where(has_prev, rv[np.clip(pos, 0, None)], np.nan)
    gap = np.where(inside & has_next, (rt[np.clip(pos + 1, None, len(rt) - 1)] - rt[np.clip(pos, 0, None)]) / 6e10,
                   np.nan)
    match_interp = inside & (np.abs(g - interp) <= TOL_G)
    match_last = has_prev & (np.abs(g - last) <= TOL_G)
    future = inside & ~match_last & match_interp
    n, ni = len(t), int(inside.sum())
    return {"n_preprocessed": n, "n_inside_raw_range": ni,
            "pct_timestamp_within_2min_of_raw": 100 * coincide.mean(),
            "pct_within_1_of_linear_interp": 100 * match_interp[inside].mean() if ni else np.nan,
            "pct_within_1_of_last_earlier": 100 * match_last[inside].mean() if ni else np.nan,
            "pct_needs_future_reading": 100 * future[inside].mean() if ni else np.nan,
            "pct_in_gaps_over_30min": 100 * np.nanmean(gap[inside] > 30) if ni else np.nan,
            "median_abs_diff_to_interp": float(np.median(np.abs(g - interp)[inside])) if ni else np.nan}


rows = []
for pid in list_patients_for_analysis("all"):
    prep = pd.read_csv(f"data/HUPA-UCM/Preprocessed/{pid}.csv", sep=";")
    prep["time"] = pd.to_datetime(prep["time"], format="%Y-%m-%dT%H:%M:%S")
    prep = prep.dropna(subset=["glucose"])
    raw, _ = load_libre_glucose(pid)
    rows.append({"patient": pid, "reference": "raw Libre type-0", **compare(prep["time"], prep["glucose"], raw)})


def medtronic_sensor(pid):
    files = glob.glob(f"data/HUPA-UCM/Raw_Data/{pid}/medtronic_insulin_pump/*.csv")
    ts, vals = [], []
    for f in files:
        with open(f, encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.read().splitlines()
        hdr = next(i for i, l in enumerate(lines) if l.startswith("Index;Date;Time"))
        cols = lines[hdr].split(";")
        ci = cols.index("Sensor Glucose (mg/dL)")
        for row in csv.reader(lines[hdr + 1:], delimiter=";"):
            if len(row) > ci and row[ci].strip():
                ts.append(f"{row[1]} {row[2]}")
                vals.append(row[ci].replace(",", "."))
    s = pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").values,
                  index=pd.to_datetime(pd.Series(ts), format="%d/%m/%Y %H:%M:%S", errors="coerce"))
    s = s[s.index.notna() & s.notna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


p1 = pd.read_csv("data/HUPA-UCM/Preprocessed/HUPA0001P.csv", sep=";")
p1["time"] = pd.to_datetime(p1["time"], format="%Y-%m-%dT%H:%M:%S")
p1 = p1.dropna(subset=["glucose"])
med = medtronic_sensor("HUPA0001P")
med = med[(med.index >= p1["time"].min()) & (med.index <= p1["time"].max())]
rows.append({"patient": "HUPA0001P", "reference": "Medtronic sensor glucose", **compare(p1["time"], p1["glucose"], med)})
D1 = save(pd.DataFrame(rows), "D1_preprocessed_vs_raw.csv")
libre = D1[D1["reference"] == "raw Libre type-0"]
others = libre[libre["patient"] != "HUPA0001P"]
r1 = libre[libre["patient"] == "HUPA0001P"].iloc[0]
rm = D1[D1["reference"] == "Medtronic sensor glucose"].iloc[0]

h("D. HUPA-UCM data quality: Preprocessed glucose vs raw Libre readings")
say("For each Preprocessed timestamp: is a raw reading within 2 min; is the value within 1 mg/dL of the linear "
    "interpolation between the surrounding raw readings; does it need a FUTURE reading (it differs by more "
    "than 1 mg/dL from the last earlier reading but matches the interpolation). Percentages are of the "
    "timestamps inside the raw-reading range.")
MDL.append("")
md_table(D1)
d = describe(others["pct_within_1_of_linear_interp"])
f = describe(others["pct_needs_future_reading"])
say(f"For the 22 patients other than HUPA0001P, {d['mean']:.1f}% of Preprocessed glucose values on average "
    f"(range {others['pct_within_1_of_linear_interp'].min():.1f}-{others['pct_within_1_of_linear_interp'].max():.1f}%; "
    f"95% CI {d['ci_lo']:.1f}-{d['ci_hi']:.1f}) equal the linear interpolation of raw Libre readings, and "
    f"{f['mean']:.1f}% (range {others['pct_needs_future_reading'].min():.1f}-{others['pct_needs_future_reading'].max():.1f}%) "
    f"can only be reproduced by using a later raw reading.")
say(f"Only {others['pct_timestamp_within_2min_of_raw'].mean():.1f}% of Preprocessed timestamps lie within "
    f"2 min of a real raw reading (mean over the 22 patients).")
say(f"HUPA0001P: only {r1['pct_within_1_of_linear_interp']:.1f}% of its Preprocessed values match the Libre "
    f"interpolation (median difference {r1['median_abs_diff_to_interp']:.1f} mg/dL), but "
    f"{rm['pct_within_1_of_linear_interp']:.1f}% match the interpolation of its Medtronic pump sensor glucose "
    f"(median difference {rm['median_abs_diff_to_interp']:.1f} mg/dL, {rm['pct_timestamp_within_2min_of_raw']:.1f}% "
    f"of timestamps within 2 min of a Medtronic reading).")
MDL.append("")

# ----------------------------------------------------------------------------
# E. Integrity
# ----------------------------------------------------------------------------
def grab(path, pattern):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [l.rstrip() for l in fh if re.search(pattern, l)]


logs = {
    "Parser check (results_objective3/hupa_parser_check_log.txt)":
        grab("results_objective3/hupa_parser_check_log.txt", r"^CHECK\s+\d+ .*(PASS|FAIL)\s*$|^OVERALL"),
    "Objective 3 audit (results_objective3/audit_objective3_log.txt)":
        grab("results_objective3/audit_objective3_log.txt", r"PASS|FAIL"),
    "Objective 3 follow-up (results_objective3/followup/followup_checks_log.txt)":
        grab("results_objective3/followup/followup_checks_log.txt", r"PASS|FAIL"),
}
o2_logs = [p for p in glob.glob("results_objective2/*") if re.search(r"audit|log", os.path.basename(p), re.I)]
for p in o2_logs:
    logs[f"Objective 2 audit ({p})"] = grab(p, r"PASS|FAIL")

hr = []
ref_xml = "data/OhioT1DM/559-ws-training.xml"
root = etree.parse(ref_xml).getroot()
tags_559 = sorted({c.tag for c in root})
for x in sorted(glob.glob("data/OhioT1DM/*-ws-*.xml")):
    r = etree.parse(x).getroot()
    hr.append({"file": os.path.basename(x), "n_heart_rate_elements": len(r.findall(".//heart_rate")),
               "n_basis_heart_rate_events": len(r.findall(".//basis_heart_rate/event"))})
E1 = save(pd.DataFrame(hr), "E1_heart_rate_tags.csv")
parsed = parse_xml_file(ref_xml)
n_hr_parsed = int(parsed["heart_rate"].notna().sum())
try:
    with open("results_causal/patient_features_causal.pkl", "rb") as fh:
        feats = pd.concat(pickle.load(fh).values(), ignore_index=True)
    feat_src = "results_causal/patient_features_causal.pkl"
except Exception:
    from objective3_common import load_ohio_features
    feats = pd.concat(load_ohio_features().values(), ignore_index=True)
    feat_src = "recomputed with load_ohio_features()"
pct_hr = 100 * feats["heart_rate"].notna().mean()
feature_list = json.load(open("models_objective3/feature_list.json", encoding="utf-8"))
imp = joblib.load("models_objective3/imputer.joblib")
hr_fill = float(imp.statistics_[feature_list.index("heart_rate")]) if "heart_rate" in feature_list else np.nan
parse_src = inspect.getsource(parse_xml_file)
hr_query = re.search(r"findall\('([^']*heart_rate[^']*)'\)", parse_src)
E2 = save(pd.DataFrame([{"xml_tags_559": " ".join(tags_559),
                         "parse_xml_file_heart_rate_query": hr_query.group(1) if hr_query else "",
                         "basis_heart_rate_events_in_559": int(E1.loc[E1["file"] == "559-ws-training.xml",
                                                                      "n_basis_heart_rate_events"].iloc[0]),
                         "heart_rate_elements_all_files": int(E1["n_heart_rate_elements"].sum()),
                         "heart_rate_non_nan_parsed_559": n_hr_parsed,
                         "pct_heart_rate_non_nan_obj1_features": pct_hr, "features_source": feat_src,
                         "imputer_fill_value_heart_rate_obj3": hr_fill}]), "E2_heart_rate_summary.csv")
log_rows = [{"log": k, "line": l} for k, v in logs.items() for l in v]
save(pd.DataFrame(log_rows), "E3_integrity_log_lines.csv")

h("E. Integrity")
for k, v in logs.items():
    MDL.append(f"**{k}**\n")
    MDL.append("```")
    MDL += v if v else ["(no PASS/FAIL lines found)"]
    MDL.append("```\n")
if not o2_logs:
    say("Objective 2: no audit log file exists in results_objective2/ (audit_objective2.py saves no log).")
say(f"Tags in {ref_xml}: {', '.join(tags_559)}.")
say(f"parse_xml_file looks for `{hr_query.group(1) if hr_query else '?'}` elements. Across the "
    f"{len(E1)} OhioT1DM XML files there are {int(E1['n_heart_rate_elements'].sum())} such elements and "
    f"{int(E1['n_basis_heart_rate_events'].sum())} `basis_heart_rate` events "
    f"({int((E1['n_basis_heart_rate_events'] > 0).sum())} files; the 2020 cohort has none).")
say(f"parse_xml_file on 559-ws-training.xml returns {n_hr_parsed} non-NaN heart_rate values; "
    f"{pct_hr:.2f}% of heart_rate values in the Objective 1 features are non-NaN ({feat_src}).")
say(f"Objective 3's saved imputer fills heart_rate with {hr_fill} (keep_empty_features=True on an all-NaN column).")
say("No model in Objectives 1-3 received real heart-rate data: the heart_rate feature is NaN in all rows and becomes a constant after imputation."
    if pct_hr == 0 else f"{pct_hr:.2f}% of rows contained real heart-rate data.")
MDL.append("")

# ----------------------------------------------------------------------------
# F. Methods facts
# ----------------------------------------------------------------------------
cf_src = inspect.getsource(data_processing.create_features)
iob = re.search(r"'dia':\s*(\d+),\s*'t_peak':\s*(\d+)", cf_src)
horizon = int(re.search(r"\['target'\]\s*=\s*df\['glucose'\]\.shift\(-(\d+)\)", cf_src).group(1))
settings = json.load(open("models_objective3/settings.json", encoding="utf-8"))
F1 = save(pd.DataFrame({"n": range(1, len(feature_list) + 1), "feature": feature_list}), "F1_features.csv")
F2 = save(pd.DataFrame([{"model": k, "settings": json.dumps(v) if not isinstance(v, str) else v}
                        for k, v in MODEL_SETTINGS.items()]), "F2_model_settings.csv")
chk = pd.read_csv("results_objective3/hupa_parser_check.csv")
counts = [
    {"objective": "1 LOSO", "patients": o1["patient"].nunique(),
     "rows": int(o1.loc[o1["model"] == "Persistence", "n_test_samples"].sum()),
     "note": "evaluated test rows summed over the 12 folds (common mask)"},
    {"objective": "1/3 training (all 12)", "patients": len(settings["patients"]),
     "rows": settings["n_training_samples"], "note": "feature rows, OhioT1DM all files"},
    {"objective": "2 personalization", "patients": o2["patient"].nunique(),
     "rows": int(o2[(o2["model"] == "LightGBM") & (o2["p_level"] == 0)]["n_eval_samples"].sum()),
     "note": f"evaluation rows per level; levels {levels}"},
    {"objective": "3 HUPA MAIN", "patients": int(chk["in_main"].sum()),
     "rows": int(chk.loc[chk["in_main"], "n_evaluable_points"].sum()), "note": "evaluable points"},
    {"objective": "3 HUPA ALL", "patients": int(chk["in_all"].sum()),
     "rows": int(chk.loc[chk["in_all"], "n_evaluable_points"].sum()), "note": "evaluable points"},
]
F3 = save(pd.DataFrame(counts), "F3_patient_counts.csv")
F4 = save(pd.DataFrame([{"library": k, "version": v} for k, v in settings["versions"].items()]),
          "F4_versions_objective3.csv")

h("F. Methods facts")
say(f"Features: {len(feature_list)} ({', '.join(feature_list)}).")
say(f"IOB: exponential kernel exp(-t/t_peak), normalized, DIA = {iob.group(1)} min, t_peak = {iob.group(2)} min, "
    f"5-min steps (create_features defaults).")
say(f"Prediction horizon: target = glucose shifted by {horizon} steps = {horizon * 5} minutes.")
MDL.append("")
md_table(F2)
md_table(F3)
say("Exclusions (HUPA-UCM): HUPA0009P and HUPA0010P have no raw Libre CGM (excluded from both analyses); "
    "HUPA0011P, HUPA0015P, HUPA0018P and HUPA0020P have missing insulin or meal events (excluded from MAIN only).")
say("HUPA-UCM study window: only raw Libre readings between the first and last `time` of "
    "Preprocessed/<PID>.csv are used. Glucose is clipped to [40, 400]. Bolus <= 0 is treated as no event.")
say("Objective 1 was run in an earlier environment whose LightGBM version was not recorded.")
MDL.append("")
md_table(F4)

with open(MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(MDL) + "\n")
print(f"Wrote {MD} and {len(os.listdir(OUT))} CSVs to {OUT}/")
