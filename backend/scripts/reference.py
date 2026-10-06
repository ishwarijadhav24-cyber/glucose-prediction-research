"""Cross-environment reference predictions for the public synthetic demo dataset.

make:    python backend/scripts/reference.py make     (run once, in the validated environment)
verify:  python backend/scripts/reference.py verify   (run in any other environment, e.g. the container)

The reference covers the complete inference path on backend/demo_data/synthetic_01.csv:
canonical events -> 47 features -> imputer/nan_to_num/scaler output -> LightGBM predictions,
for every eligible prediction time (mode=all) and the latest-time prediction, plus the
error codes of five invalid inputs. Floats are stored exactly (float64 in .npz).

Tolerance: the pipeline uses only element-wise arithmetic, a direct convolution, pandas rolling
statistics and tree traversal, so bit-identical results are expected across platforms. Anything
non-identical is reported with its maximum absolute difference and accepted only if
<= 1e-9 (features / scaled values) and <= 1e-9 mg/dL (predictions); otherwise verify fails.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.errors import BackendError                       # noqa: E402
from backend.app.services.artifacts import installed_versions, load_artifacts   # noqa: E402
from backend.app.services.datasets import UploadedFile, ingest_upload           # noqa: E402
from backend.app.services.inference import InferenceService                     # noqa: E402
from objective3_common import apply_preprocessing                               # noqa: E402

DEMO = ROOT / "backend" / "demo_data" / "synthetic_01.csv"
REF_DIR = ROOT / "backend" / "reference"
NPZ, META = REF_DIR / "synthetic_01_reference.npz", REF_DIR / "synthetic_01_reference.json"
TOL = 1e-9


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _error_cases(raw: bytes) -> dict:
    lines = raw.decode("utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    glucose_rows = [r for r in rows if r.split(",")[1]]
    cases = {
        "insufficient_history": "\n".join([header] + glucose_rows[:13]),
        "ten_minute_sampling": "\n".join([header] + glucose_rows[::2]),
        "out_of_range_glucose": "\n".join([header] + glucose_rows[:20] +
                                          [glucose_rows[20].split(",")[0] + ",900,,"]),
        "missing_column": "\n".join(",".join(r.split(",")[:3]) for r in [header] + glucose_rows[:20]),
        "no_glucose": "\n".join([header] + [r for r in rows if not r.split(",")[1]][:5]),
    }
    return {k: v.encode() + b"\n" for k, v in cases.items()}


def compute(model_dir: Path | None = None) -> tuple[dict, dict]:
    art = load_artifacts(model_dir or ROOT / "models_objective3")
    svc = InferenceService(art)
    raw = DEMO.read_bytes()
    ds = ingest_upload([UploadedFile(raw, "synthetic_01.csv")])
    results, skipped, truncated = svc.predict_many(ds.events, 100_000)
    F = np.vstack([r.features.to_numpy(dtype=float) for r in results])
    X = apply_preprocessing(art.imputer, art.scaler, F)
    latest = svc.predict(ds.events)
    errors = {}
    for name, data in _error_cases(raw).items():
        try:
            d = ingest_upload([UploadedFile(data, f"{name}.csv")])
            svc.predict(d.events)
            errors[name] = "OK"
        except BackendError as e:
            errors[name] = e.code.value
    arrays = {
        "prediction_time_ns": np.array([r.prediction_time.value for r in results], dtype=np.int64),
        "latest_observation_ns": np.array([r.latest_observation_time.value for r in results], dtype=np.int64),
        "features": F, "preprocessed": X,
        "predictions": np.array([r.predicted_glucose_mg_dl for r in results], dtype=np.float64),
        "latest_prediction": np.array([latest.predicted_glucose_mg_dl], dtype=np.float64),
        "latest_time_ns": np.array([latest.prediction_time.value], dtype=np.int64),
    }
    meta = {
        "input": str(DEMO.relative_to(ROOT)).replace("\\", "/"), "input_sha256": _sha(raw),
        "model_version": art.model_version, "feature_version": art.feature_version,
        "feature_list": list(art.feature_list), "n_predictions": len(results), "truncated": truncated,
        "skipped": skipped, "dataset_type": ds.dataset_type, "sampling_profile": ds.sampling_profile,
        "error_codes": errors,
        "environment": {"platform": platform.platform(), "machine": platform.machine(), **installed_versions()},
    }
    return arrays, meta


def make() -> None:
    arrays, meta = compute()
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **arrays)
    meta["npz_sha256"] = _sha(NPZ.read_bytes())
    META.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {NPZ.name} ({meta['n_predictions']} predictions) and {META.name}")


def compare(model_dir: Path | None = None) -> dict:
    ref_meta = json.loads(META.read_text(encoding="utf-8"))
    if _sha(NPZ.read_bytes()) != ref_meta["npz_sha256"]:
        raise RuntimeError("reference .npz does not match its recorded hash")
    ref = np.load(NPZ)
    arrays, meta = compute(model_dir)
    report = {"reference_environment": ref_meta["environment"], "this_environment": meta["environment"]}
    checks = {
        "input_identical": meta["input_sha256"] == ref_meta["input_sha256"],
        "model_identical": meta["model_version"] == ref_meta["model_version"],
        "feature_order_identical": meta["feature_list"] == ref_meta["feature_list"],
        "same_prediction_times_and_order": np.array_equal(arrays["prediction_time_ns"], ref["prediction_time_ns"]),
        "same_latest_observation_times": np.array_equal(arrays["latest_observation_ns"], ref["latest_observation_ns"]),
        "same_error_codes": meta["error_codes"] == ref_meta["error_codes"],
        "same_skipped_counts": meta["skipped"] == ref_meta["skipped"],
    }
    for key in ("features", "preprocessed", "predictions", "latest_prediction"):
        a, b = arrays[key], ref[key]
        same_shape = a.shape == b.shape
        report[f"{key}_bit_identical"] = bool(same_shape and np.array_equal(a, b, equal_nan=True))
        diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0)) if same_shape else np.array([np.inf])
        nan_match = bool(same_shape and np.array_equal(np.isnan(a), np.isnan(b)))
        report[f"{key}_max_abs_diff"] = float(diff.max()) if diff.size else 0.0
        checks[f"{key}_within_tolerance"] = bool(same_shape and nan_match and diff.max() <= TOL)
    report["checks"] = checks
    report["n_predictions"] = int(len(arrays["predictions"]))
    report["passed"] = all(checks.values())
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("make", "verify"):
        sys.exit("usage: reference.py make|verify")
    if sys.argv[1] == "make":
        make()
    else:
        rep = compare()
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["passed"] else 1)
