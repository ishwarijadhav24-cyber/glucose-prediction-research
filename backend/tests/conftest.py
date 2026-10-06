import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = REPO_ROOT / "models_objective3"
HUPA_DATA = REPO_ROOT / "data" / "HUPA-UCM"
HUPA_PREDICTIONS = REPO_ROOT / "results_objective3" / "hupa_predictions.csv"
LOSO_SUMMARY = REPO_ROOT / "results_causal" / "loso_summary.csv"
OHIO_XML = REPO_ROOT / "data" / "OhioT1DM" / "559-ws-training.xml"

# Research data/results are private and absent in the container. Set
# GLUCOSE_TESTS_NO_RESEARCH_DATA=1 to simulate that environment locally.
FORCE_NO_RESEARCH_DATA = os.environ.get("GLUCOSE_TESTS_NO_RESEARCH_DATA") == "1"


def research_available(*paths) -> bool:
    return not FORCE_NO_RESEARCH_DATA and all(p.exists() for p in paths)


def requires(*paths):
    return pytest.mark.skipif(not research_available(*paths),
                              reason="private research data/results not available (e.g. container)")


@pytest.fixture(scope="session")
def artifacts():
    from backend.app.services.artifacts import load_artifacts
    return load_artifacts(MODEL_DIR)


@pytest.fixture
def model_dir_copy(tmp_path):
    """A writable copy of the artifact directory, for tamper tests."""
    dst = tmp_path / "models"
    shutil.copytree(MODEL_DIR, dst)
    return dst
