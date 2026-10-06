"""
Objective 3, Phase 4: external validation on HUPA-UCM.

Loads the OhioT1DM-trained models from models_objective3/ and applies them to
HUPA-UCM WITHOUT retraining or refitting anything (the saved imputer and scaler
are only used with .transform). Evaluation only at evaluable points:
is_real at t AND at t+30 min (shift by 6 on the full grid, before create_features
drops rows) AND target not NaN. Persistence = glucose at t.

PRIMARY metrics: computed per patient, then mean and SD across patients
(each patient counts once).
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing import create_features
from evaluate_loso import (calculate_mae, calculate_mard, calculate_r2, calculate_rmse,
                           evaluate_clarke_ega)
from parsers.hupa_ucm import list_patients_for_analysis, parse_hupa_patient

MODELS_DIR = "models_objective3"
RESULTS_DIR = "results_objective3"
OUTPUTS_DIR = "outputs_objective3"
MODEL_NAMES = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
METRICS = ["MARD", "RMSE", "MAE", "R2", "EGA_A", "EGA_B", "EGA_A+B"]
CLARKE_MAX_PER_PATIENT = 1000
SEED = 42

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def evaluable_mask(df, is_real, feats):
    real_t30 = is_real.shift(-6, fill_value=False).astype(bool)
    mask_grid = is_real.astype(bool) & real_t30
    return mask_grid.reindex(pd.DatetimeIndex(feats["ts"])).fillna(False).values & feats["target"].notna().values


def metrics(y, p):
    a, b, ab, _ = evaluate_clarke_ega(y, p)
    return {"MARD": calculate_mard(y, p), "RMSE": calculate_rmse(y, p), "MAE": calculate_mae(y, p),
            "R2": calculate_r2(y, p), "EGA_A": a, "EGA_B": b, "EGA_A+B": ab}


def summarize(patient_results, patients):
    sub = patient_results[patient_results["patient"].isin(patients)]
    g = sub.groupby("model")
    out = pd.DataFrame({"model": MODEL_NAMES})
    out["n_patients"] = out["model"].map(g["patient"].nunique())
    out["n_samples_total"] = out["model"].map(g["n_samples"].sum())
    for m in METRICS:
        out[f"{m}_mean"] = out["model"].map(g[m].mean())
        out[f"{m}_std"] = out["model"].map(g[m].std())
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(summary_main):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(MODEL_NAMES))
    for ax, m, unit in zip(axes, ("MARD", "RMSE"), ("%", "mg/dL")):
        ax.bar(x, summary_main[f"{m}_mean"], yerr=summary_main[f"{m}_std"], color=SERIES[0], width=0.6,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        for xi, v in zip(x, summary_main[f"{m}_mean"]):
            ax.text(xi, v / 2, f"{v:.1f}", ha="center", color="white", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES, rotation=20)
        ax.set_ylabel(f"{m} ({unit}), mean ± SD across patients", color=INK_2)
        ax.set_title(f"{m}", color=INK, loc="left")
        _style(ax)
    fig.suptitle("HUPA-UCM external validation, MAIN (19 patients), OhioT1DM-trained models",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "hupa_model_comparison.png"), dpi=200)
    plt.close(fig)


def plot_per_patient(patient_results):
    pats = sorted(patient_results["patient"].unique())
    x = np.arange(len(pats))
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(MODEL_NAMES):
        sub = patient_results[patient_results["model"] == m].set_index("patient").reindex(pats)
        ax.plot(x + (i - 2.5) * 0.11, sub["MARD"], "o", markersize=6, color=SERIES[i], label=m,
                markeredgecolor="white", markeredgewidth=0.8)
    main = set(list_patients_for_analysis("main"))
    ax.set_xticks(x)
    ax.set_xticklabels([p if p in main else p + " *" for p in pats], rotation=90)
    ax.set_ylabel("MARD (%)", color=INK_2)
    ax.set_title("Per-patient MARD on HUPA-UCM (* = ALL only, missing insulin/meal events)",
                 color=INK, loc="left", pad=26)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "hupa_per_patient_mard.png"), dpi=200)
    plt.close(fig)


def plot_ohio_vs_hupa(comp):
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    series = [("OhioT1DM_LOSO_MARD", "OhioT1DM LOSO (Objective 1, 12 patients)"),
              ("HUPA_main_MARD", "HUPA-UCM MAIN (19)"), ("HUPA_all_MARD", "HUPA-UCM ALL (23)")]
    w = 0.26
    for i, (col, label) in enumerate(series):
        ax.bar(x + (i - 1) * (w + 0.02), comp[col], width=w, color=SERIES[i], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"])
    ax.set_ylabel("MARD (%), mean across patients", color=INK_2)
    ax.set_title("Internal (OhioT1DM LOSO) vs external (HUPA-UCM) MARD", color=INK, loc="left", pad=26)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "ohio_vs_hupa_mard.png"), dpi=200)
    plt.close(fig)


def plot_clarke(preds, model="LightGBM"):
    main = list_patients_for_analysis("main")
    sub = preds[preds["patient"].isin(main)]
    sample = (sub.groupby("patient", group_keys=False)
              .apply(lambda d: d.sample(n=min(CLARKE_MAX_PER_PATIENT, len(d)), random_state=SEED)))
    y, p = sample["target"].values, sample[model].values
    a, b, ab, _ = evaluate_clarke_ega(y, p)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, p, s=4, color=SERIES[0], alpha=0.35, linewidths=0)
    lines = [([0, 400], [0, 400], ":"), ([0, 175 / 3], [70, 70], "-"), ([175 / 3, 400 / 1.2], [70, 400], "-"),
             ([70, 70], [84, 400], "-"), ([0, 70], [180, 180], "-"), ([70, 290], [180, 400], "-"),
             ([70, 70], [0, 56], "-"), ([70, 400], [56, 320], "-"), ([180, 180], [0, 70], "-"),
             ([180, 400], [70, 70], "-"), ([240, 240], [70, 180], "-"), ([240, 400], [180, 180], "-"),
             ([130, 180], [0, 70], "-")]
    for xs, ys, ls in lines:
        ax.plot(xs, ys, ls, color=INK_2, linewidth=1)
    for txt, (tx, ty) in {"A": (30, 15), "B": (370, 260), "B ": (280, 370), "C": (160, 370), "C ": (160, 15),
                          "D": (30, 140), "D ": (370, 120), "E": (30, 370), "E ": (370, 15)}.items():
        ax.text(tx, ty, txt.strip(), fontsize=14, color=INK)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference glucose at t+30 min (mg/dL)", color=INK_2)
    ax.set_ylabel("Predicted glucose (mg/dL)", color=INK_2)
    ax.set_title(f"{model} Clarke error grid, HUPA-UCM MAIN\n"
                 f"patient-balanced sample: at most {CLARKE_MAX_PER_PATIENT:,} random points per patient "
                 f"(random_state={SEED}), n={len(sample):,}\nA {a:.1f}%  B {b:.1f}%  A+B {ab:.1f}%",
                 color=INK, loc="left", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "hupa_clarke_grid_lightgbm.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "feature_list.json"), encoding="utf-8") as fh:
        cols = json.load(fh)
    imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    models = {m: joblib.load(os.path.join(MODELS_DIR, f"{m}.joblib")) for m in MODEL_NAMES[1:]}

    patients = list_patients_for_analysis("all")
    main_set = set(list_patients_for_analysis("main"))
    result_rows, pred_frames = [], []
    for pid in patients:
        df, is_real, _ = parse_hupa_patient(pid)
        feats = create_features(df)
        mask = evaluable_mask(df, is_real, feats)
        ev = feats[mask]
        # transform only: nothing is refitted on HUPA-UCM
        X = scaler.transform(np.nan_to_num(imputer.transform(ev[cols].values), nan=0.0))
        y = ev["target"].values
        preds = {"Persistence": ev["glucose"].values}
        preds.update({m: mdl.predict(X) for m, mdl in models.items()})
        for m in MODEL_NAMES:
            result_rows.append({"patient": pid, "in_main": pid in main_set, "model": m,
                                **metrics(y, preds[m]), "n_samples": int(len(y))})
        pf = pd.DataFrame({"patient": pid, "in_main": pid in main_set, "ts": ev["ts"].values,
                           "glucose_t": ev["glucose"].values, "target": y})
        for m in MODEL_NAMES:
            pf[m] = preds[m]
        pred_frames.append(pf)
        print(f"{pid}: {len(y)} evaluable points, LightGBM MARD {result_rows[-2]['MARD']:.2f}")

    pr = pd.DataFrame(result_rows)
    pr.to_csv(os.path.join(RESULTS_DIR, "hupa_patient_results.csv"), index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(os.path.join(RESULTS_DIR, "hupa_predictions.csv"), index=False)
    s_main = summarize(pr, sorted(main_set))
    s_all = summarize(pr, patients)
    s_main.to_csv(os.path.join(RESULTS_DIR, "hupa_summary_main.csv"), index=False)
    s_all.to_csv(os.path.join(RESULTS_DIR, "hupa_summary_all.csv"), index=False)

    ohio = pd.read_csv("results_causal/loso_summary.csv").set_index("model")
    comp = pd.DataFrame({"model": MODEL_NAMES})
    comp["OhioT1DM_LOSO_MARD"] = comp["model"].map(ohio["MARD_mean"])
    comp["OhioT1DM_LOSO_RMSE"] = comp["model"].map(ohio["RMSE_mean"])
    for tag, s in (("main", s_main), ("all", s_all)):
        s = s.set_index("model")
        comp[f"HUPA_{tag}_MARD"] = comp["model"].map(s["MARD_mean"])
        comp[f"HUPA_{tag}_RMSE"] = comp["model"].map(s["RMSE_mean"])
    comp["HUPA_main_EGA_A+B"] = comp["model"].map(s_main.set_index("model")["EGA_A+B_mean"])
    comp.to_csv(os.path.join(RESULTS_DIR, "comparison_with_ohio.csv"), index=False)

    plot_model_comparison(s_main)
    plot_per_patient(pr)
    plot_ohio_vs_hupa(comp)
    plot_clarke(preds)
    print(comp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
