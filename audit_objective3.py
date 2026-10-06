"""
Objective 3, Phase 6: audit of the HUPA-UCM external validation outputs.
Each audit prints PASS/FAIL; the log goes to results_objective3/audit_objective3_log.txt.
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from parsers.hupa_ucm import list_patients_for_analysis, parse_hupa_patient

R = "results_objective3"
MODELS_DIR = "models_objective3"
MODEL_NAMES = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"]
METRICS = ["MARD", "RMSE", "MAE", "R2", "EGA_A", "EGA_B", "EGA_A+B"]
LOG = []
RESULTS = []


def audit(name, ok, detail=""):
    line = f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else "")
    print(line)
    LOG.append(line)
    RESULTS.append(ok)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


pr = pd.read_csv(os.path.join(R, "hupa_patient_results.csv"))
preds = pd.read_csv(os.path.join(R, "hupa_predictions.csv"), parse_dates=["ts"])
main = list_patients_for_analysis("main")
all_ = list_patients_for_analysis("all")

# 1. Row counts
audit("23 x 6 patient-model rows", len(pr) == 23 * 6 and set(pr["patient"]) == set(all_)
      and all(pr.groupby("patient")["model"].apply(lambda s: sorted(s) == sorted(MODEL_NAMES))),
      f"{len(pr)} rows, {pr['patient'].nunique()} patients")
sub = pr[pr["patient"].isin(main)]
audit("MAIN subset 19 x 6", len(sub) == 19 * 6 and set(pr.loc[pr["in_main"], "patient"]) == set(main),
      f"{len(sub)} rows")

# 2. Duplicates / NaN / inf
audit("no duplicate patient-model rows", not pr.duplicated(["patient", "model"]).any())
vals = pr[METRICS + ["n_samples"]].to_numpy(dtype=float)
audit("no NaN or infinite metric values", np.isfinite(vals).all(), f"{int((~np.isfinite(vals)).sum())} bad")
audit("no duplicate prediction rows", not preds.duplicated(["patient", "ts"]).any())

# 3. Summaries recomputed
for tag, pats in (("main", main), ("all", all_)):
    s = pd.read_csv(os.path.join(R, f"hupa_summary_{tag}.csv")).set_index("model")
    g = pr[pr["patient"].isin(pats)].groupby("model")
    ok = True
    for m in METRICS:
        ok &= np.allclose(s.loc[MODEL_NAMES, f"{m}_mean"], g[m].mean().loc[MODEL_NAMES], rtol=0, atol=1e-9)
        ok &= np.allclose(s.loc[MODEL_NAMES, f"{m}_std"], g[m].std().loc[MODEL_NAMES], rtol=0, atol=1e-9)
    ok &= (s.loc[MODEL_NAMES, "n_patients"] == len(pats)).all()
    audit(f"summary {tag}: means and SDs equal recomputed patient-level values", bool(ok))

# 4. Persistence = glucose at t (recomputed from the parser, not from the file)
bad = 0
for pid in all_:
    df, _, _ = parse_hupa_patient(pid)
    p = preds[preds["patient"] == pid]
    g_t = df["glucose"].reindex(pd.DatetimeIndex(p["ts"])).values
    bad += int((~np.isclose(p["Persistence"].values, g_t, rtol=0, atol=1e-9)).sum())
    bad += int((~np.isclose(p["glucose_t"].values, g_t, rtol=0, atol=1e-9)).sum())
audit("Persistence predictions equal glucose at t for every evaluated row", bad == 0,
      f"{bad} mismatches over {len(preds):,} rows")

# 5. Model hashes
with open(os.path.join(MODELS_DIR, "model_hashes.json"), encoding="utf-8") as fh:
    hashes = json.load(fh)
mism = [f for f, h in hashes.items() if sha256(os.path.join(MODELS_DIR, f)) != h]
extra = sorted(set(os.listdir(MODELS_DIR)) - set(hashes) - {"model_hashes.json"})
audit("model file hashes equal model_hashes.json", not mism and not extra and len(hashes) > 0,
      f"{len(hashes)} files; mismatched {mism}; unhashed {extra}")

# 6. No HUPA-UCM in training code (training script and the module it imports)
pattern = re.compile(r"hupa|parsers", re.IGNORECASE)
hits = []
for f in ("train_final_ohio.py", "objective3_common.py"):
    with open(f, encoding="utf-8") as fh:
        hits += [f"{f}:{i}" for i, line in enumerate(fh, 1) if pattern.search(line)]
audit("train_final_ohio.py (and objective3_common.py) contain no HUPA-UCM path or import", not hits, str(hits))

# 7. n_samples = n_evaluable_points from the parser check
chk = pd.read_csv(os.path.join(R, "hupa_parser_check.csv")).set_index("patient")["n_evaluable_points"]
ns = pr.groupby("patient")["n_samples"].agg(["min", "max"])
diff = [p for p in ns.index if not (ns.loc[p, "min"] == ns.loc[p, "max"] == chk.loc[p])]
audit("n_samples per patient equals n_evaluable_points in hupa_parser_check.csv", not diff, f"differs: {diff}")

overall = all(RESULTS)
LOG.append(f"OVERALL: {'PASS' if overall else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)})")
print(LOG[-1])
with open(os.path.join(R, "audit_objective3_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
