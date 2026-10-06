"""
Low-glucose (hypoglycemia) alert reliability, from EXISTING predictions only.

Sources (read only):
  results_causal/loso_predictions.csv     OhioT1DM LOSO (Objective 1), restricted to the
                                          Objective 1 common mask (Persistence finite)
  results_objective3/hupa_predictions.csv HUPA-UCM MAIN (19 patients), evaluable points

An alert is raised at time t when the 30-min prediction is below the threshold
(70 or 80 mg/dL). A real low is reference glucose < 70 mg/dL at t+30.

1. Point level: sensitivity, precision (PPV), specificity; on all rows and on rows
   where glucose at t is still >= 70 ("not already low"), i.e. a real warning.
2. Episode level: a low episode = consecutive reference values < 70 (gaps <= 20 min)
   lasting >= 15 min, with a value >= 70 in the 20 min before onset.
   Detected = an alert issued 0-60 min before onset; lead time = onset - first alert.
   Useful detection = detected with >= 15 min of warning (an alert at onset itself gives 0 min).
   False alarm = an alert episode starting while glucose >= 70 with no reference
   value < 70 in the next 60 min. Rates are per patient-day with data.
95% CIs: patient-level (cluster) bootstrap, 2000 resamples, random_state=42.
"""

import os

import numpy as np
import pandas as pd

OUT = "results_evidence/hypo_alerts"
MODELS = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
THRESHOLDS = [70, 80]
LOW = 70.0
H30 = pd.Timedelta("30min")
GAP = pd.Timedelta("20min")
MIN_DUR = pd.Timedelta("15min")
WARN = pd.Timedelta("60min")
N_BOOT, SEED = 2000, 42


def load():
    lp = pd.read_csv("results_causal/loso_predictions.csv")
    ohio = lp.pivot_table(index=["patient", "timestamp"], columns="model", values="y_pred").reset_index()
    truth = lp[lp["model"] == "Persistence"][["patient", "timestamp", "y_true"]]
    ohio = ohio.merge(truth, on=["patient", "timestamp"]).dropna(subset=["Persistence"])
    ohio = ohio.rename(columns={"timestamp": "t"})
    ohio["glucose_t"] = ohio["Persistence"]
    ohio["t"] = pd.to_datetime(ohio["t"])
    ohio["patient"] = ohio["patient"].astype(str)
    hp = pd.read_csv("results_objective3/hupa_predictions.csv", parse_dates=["ts"])
    hupa = hp[hp["in_main"]].rename(columns={"ts": "t", "target": "y_true"})
    cols = ["patient", "t", "glucose_t", "y_true"] + MODELS
    return {"Ohio LOSO": ohio[cols], "HUPA MAIN": hupa[cols]}


def ratio_ci(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.RandomState(SEED).randint(0, len(num), size=(N_BOOT, len(num)))
    bn, bd = num[idx].sum(1), den[idx].sum(1)
    r = bn[bd > 0] / bd[bd > 0]
    est = num.sum() / den.sum() if den.sum() else np.nan
    return 100 * est, 100 * np.percentile(r, 2.5) if len(r) else np.nan, 100 * np.percentile(r, 97.5) if len(r) else np.nan


def point_level(df, model, thr, subset):
    d = df if subset == "all rows" else df[df["glucose_t"] >= LOW]
    a, e = d[model] < thr, d["y_true"] < LOW
    c = pd.DataFrame({"TP": a & e, "FP": a & ~e, "FN": ~a & e, "TN": ~a & ~e}).groupby(d["patient"]).sum()
    sens = ratio_ci(c.TP, c.TP + c.FN)
    ppv = ratio_ci(c.TP, c.TP + c.FP)
    spec = ratio_ci(c.TN, c.TN + c.FP)
    return {"rows": subset, "n_points": int(c.values.sum()), "n_low_points": int((c.TP + c.FN).sum()),
            "n_alerts": int((c.TP + c.FP).sum()), "TP": int(c.TP.sum()), "FP": int(c.FP.sum()),
            "FN": int(c.FN.sum()), "sensitivity": sens[0], "sens_ci_lo": sens[1], "sens_ci_hi": sens[2],
            "precision": ppv[0], "prec_ci_lo": ppv[1], "prec_ci_hi": ppv[2],
            "specificity": spec[0], "spec_ci_lo": spec[1], "spec_ci_hi": spec[2]}


NS = lambda td: td.value  # Timedelta -> int nanoseconds


def runs(times, gap=NS(GAP)):
    """Split sorted times into runs where consecutive times are <= gap apart."""
    if len(times) == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > gap) + 1
    return np.split(times, breaks)


def episodes_for_patient(p, model, thr):
    p = p.sort_values("t")
    T = (p["t"] + H30).values.astype("int64")      # reference time (ns)
    v = p["y_true"].values
    order = np.argsort(T)
    T, v = T[order], v[order]
    T, keep = np.unique(T, return_index=True)
    v = v[keep]
    low = v < LOW
    # low episodes
    eps, n_skipped = [], 0
    i = 0
    while i < len(T):
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(T) and low[j + 1] and T[j + 1] - T[j] <= NS(GAP):
            j += 1
        prev_ok = i > 0 and not low[i - 1] and T[i] - T[i - 1] <= NS(GAP)
        if T[j] - T[i] >= NS(MIN_DUR):
            if prev_ok:
                eps.append(T[i])
            else:
                n_skipped += 1
        i = j + 1
    alert_rows = p[p[model] < thr]
    at = np.sort(alert_rows["t"].values.astype("int64"))
    detected, leads = 0, []
    for t0 in eps:
        lo, hi = np.searchsorted(at, t0 - NS(WARN), "left"), np.searchsorted(at, t0, "right")
        if hi > lo:
            detected += 1
            leads.append((t0 - at[lo]) / 6e10)
    # alert episodes starting while glucose >= 70
    ant = np.sort(alert_rows.loc[alert_rows["glucose_t"] >= LOW, "t"].values.astype("int64"))
    false, n_alert_eps = 0, 0
    low_times = T[low]
    for r in runs(ant):
        ta = r[0]
        n_alert_eps += 1
        k = np.searchsorted(low_times, ta, "left")
        if not (k < len(low_times) and low_times[k] <= ta + NS(WARN)):
            false += 1
    days = p["t"].dt.normalize().nunique()
    return {"episodes": len(eps), "detected": detected, "detected_15": sum(l >= 15 for l in leads), "leads": leads, "skipped_no_onset": n_skipped,
            "alert_episodes": n_alert_eps, "false_alarms": false, "days": days}


def episode_level(df, model, thr):
    per = {pid: episodes_for_patient(g, model, thr) for pid, g in df.groupby("patient")}
    e = pd.DataFrame(per).T
    det = ratio_ci(e["detected"], e["episodes"])
    ppv = ratio_ci(e["alert_episodes"] - e["false_alarms"], e["alert_episodes"])
    leads = np.concatenate([np.asarray(x, float) for x in e["leads"]]) if e["episodes"].sum() else np.array([])
    fa_day = e["false_alarms"].astype(float) / e["days"].astype(float)
    idx = np.random.RandomState(SEED).randint(0, len(e), size=(N_BOOT, len(e)))
    fa_boot = (e["false_alarms"].values.astype(float)[idx].sum(1) / e["days"].values.astype(float)[idx].sum(1))
    return {"n_patients": len(e), "n_low_episodes": int(e["episodes"].sum()),
            "n_episodes_skipped_no_onset": int(e["skipped_no_onset"].sum()),
            "detected": int(e["detected"].sum()), "detection_rate": det[0], "det_ci_lo": det[1], "det_ci_hi": det[2],
            "detected_15min_warning": int(e["detected_15"].sum()),
            "detection_15min_rate": ratio_ci(e["detected_15"], e["episodes"])[0],
            "det15_ci_lo": ratio_ci(e["detected_15"], e["episodes"])[1],
            "det15_ci_hi": ratio_ci(e["detected_15"], e["episodes"])[2],
            "median_lead_min": float(np.median(leads)) if len(leads) else np.nan,
            "lead_q25_min": float(np.percentile(leads, 25)) if len(leads) else np.nan,
            "lead_q75_min": float(np.percentile(leads, 75)) if len(leads) else np.nan,
            "n_alert_episodes": int(e["alert_episodes"].sum()), "n_false_alarms": int(e["false_alarms"].sum()),
            "alert_precision": ppv[0], "alert_prec_ci_lo": ppv[1], "alert_prec_ci_hi": ppv[2],
            "patient_days": int(e["days"].sum()),
            "false_alarms_per_day": float(e["false_alarms"].sum() / e["days"].sum()),
            "fa_day_ci_lo": float(np.percentile(fa_boot, 2.5)), "fa_day_ci_hi": float(np.percentile(fa_boot, 97.5))}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load()
    pts, eps = [], []
    for name, df in data.items():
        for thr in THRESHOLDS:
            for m in MODELS:
                for subset in ("all rows", "glucose at t >= 70"):
                    pts.append({"dataset": name, "threshold": thr, "model": m, **point_level(df, m, thr, subset)})
                eps.append({"dataset": name, "threshold": thr, "model": m, **episode_level(df, m, thr)})
        print(f"{name}: done")
    pt, ep = pd.DataFrame(pts), pd.DataFrame(eps)
    pt.to_csv(os.path.join(OUT, "hypo_point_level.csv"), index=False)
    ep.to_csv(os.path.join(OUT, "hypo_episode_level.csv"), index=False)
    cols = ["dataset", "threshold", "model", "n_low_episodes", "detection_rate", "detection_15min_rate", "det15_ci_lo", "det15_ci_hi",
            "median_lead_min", "alert_precision", "false_alarms_per_day"]
    print(ep[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
