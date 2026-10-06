"""Load and verify the deployed research artifacts (read-only).

The files are the ones written by train_final_ohio.py into models_objective3/:
  LightGBM.joblib, imputer.joblib, scaler.joblib, feature_list.json, settings.json
Their SHA-256 hashes are checked against model_hashes.json BEFORE anything is
unpickled (joblib/pickle can execute code), installed library versions are checked
against settings.json, and every object must expect exactly the 47 features of
feature_list.json. Any failure raises ArtifactError; there is no fallback model.
"""

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm
import numpy
import pandas
import sklearn

from backend.app.config import DEPLOYED_MODEL

HASH_FILE = "model_hashes.json"
SUPPORT_FILES = ("imputer.joblib", "scaler.joblib", "feature_list.json", "settings.json")
INSTALLED_VERSION_GETTERS = {
    "lightgbm": lambda: lightgbm.__version__,
    "scikit-learn": lambda: sklearn.__version__,
    "numpy": lambda: numpy.__version__,
    "pandas": lambda: pandas.__version__,
    "python": platform.python_version,
}


class ArtifactError(RuntimeError):
    """The artifacts are missing, altered or incompatible. The API must not start."""


@dataclass(frozen=True)
class LoadedArtifacts:
    model_name: str
    model: object
    imputer: object
    scaler: object
    feature_list: tuple
    settings: dict
    hashes: dict
    model_version: str      # first 12 hex chars of the model file's SHA-256
    feature_version: str    # first 12 hex chars of feature_list.json's SHA-256

    @property
    def n_features(self) -> int:
        return len(self.feature_list)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_versions() -> dict:
    return {k: get() for k, get in INSTALLED_VERSION_GETTERS.items()}


def check_versions(expected: dict, installed: dict) -> None:
    """Libraries must match exactly; Python must match major.minor."""
    problems = []
    for lib in ("lightgbm", "scikit-learn", "numpy", "pandas"):
        if lib not in expected:
            problems.append(f"settings.json has no version for {lib}")
        elif installed.get(lib) != expected[lib]:
            problems.append(f"{lib}: installed {installed.get(lib)}, artifact built with {expected[lib]}")
    exp_py = ".".join(str(expected.get("python", "")).split(".")[:2])
    inst_py = ".".join(str(installed.get("python", "")).split(".")[:2])
    if exp_py != inst_py:
        problems.append(f"python: installed {installed.get('python')}, artifact built with {expected.get('python')}")
    if problems:
        raise ArtifactError("Library version mismatch: " + "; ".join(problems))


def verify_hashes(model_dir: Path, filenames, hashes: dict) -> None:
    for name in filenames:
        path = model_dir / name
        if not path.is_file():
            raise ArtifactError(f"Missing artifact file: {name}")
        if name not in hashes:
            raise ArtifactError(f"No recorded hash for artifact file: {name}")
        if sha256(path) != hashes[name]:
            raise ArtifactError(f"Hash mismatch for artifact file: {name}")


def load_artifacts(model_dir: Path, installed: dict | None = None) -> LoadedArtifacts:
    model_dir = Path(model_dir)
    hash_path = model_dir / HASH_FILE
    if not hash_path.is_file():
        raise ArtifactError(f"Missing {HASH_FILE} in model directory")
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))

    model_file = f"{DEPLOYED_MODEL}.joblib"
    verify_hashes(model_dir, (model_file,) + SUPPORT_FILES, hashes)   # before unpickling

    settings = json.loads((model_dir / "settings.json").read_text(encoding="utf-8"))
    check_versions(settings.get("versions", {}), installed if installed is not None else installed_versions())

    feature_list = tuple(json.loads((model_dir / "feature_list.json").read_text(encoding="utf-8")))
    if len(feature_list) != len(set(feature_list)) or not feature_list:
        raise ArtifactError("feature_list.json is empty or has duplicate names")
    if settings.get("n_features") != len(feature_list):
        raise ArtifactError("settings.json n_features does not match feature_list.json")

    model = joblib.load(model_dir / model_file)
    imputer = joblib.load(model_dir / "imputer.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    for label, obj in (("model", model), ("imputer", imputer), ("scaler", scaler)):
        n = getattr(obj, "n_features_in_", None)
        if n != len(feature_list):
            raise ArtifactError(f"{label} expects {n} features, feature_list.json has {len(feature_list)}")

    return LoadedArtifacts(
        model_name=DEPLOYED_MODEL, model=model, imputer=imputer, scaler=scaler, feature_list=feature_list,
        settings=settings, hashes=hashes, model_version=hashes[model_file][:12],
        feature_version=hashes["feature_list.json"][:12],
    )
