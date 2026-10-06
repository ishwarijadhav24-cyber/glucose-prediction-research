"""Verify a deployment artifact directory against backend/deploy/artifact_manifest.json.

python backend/scripts/verify_artifacts.py <model_dir>      (exit 0 = OK)

Checks: exactly the required files are present (no legacy model_artifact.pkl, no unused models),
each file's SHA-256 equals the manifest (fixed when the research artifacts were validated), and
model_hashes.json (used again by the loader at start-up) agrees with the manifest.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "backend" / "deploy" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(model_dir: Path, manifest_path: Path = MANIFEST, allow_extra: bool = False) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    problems = []
    present = {p.name for p in Path(model_dir).iterdir() if p.is_file()}
    missing = sorted(set(manifest) - present)
    extra = sorted(present - set(manifest))
    if missing:
        problems.append(f"missing: {missing}")
    if extra and not allow_extra:
        problems.append(f"unexpected files: {extra}")
    for name, digest in manifest.items():
        p = Path(model_dir) / name
        if p.is_file() and sha256(p) != digest:
            problems.append(f"hash mismatch: {name}")
    hp = Path(model_dir) / "model_hashes.json"
    if hp.is_file():
        recorded = json.loads(hp.read_text(encoding="utf-8"))
        for name, digest in manifest.items():
            if name != "model_hashes.json" and recorded.get(name) != digest:
                problems.append(f"model_hashes.json disagrees with manifest for {name}")
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: verify_artifacts.py <model_dir>")
    issues = verify(Path(sys.argv[1]))
    if issues:
        print("ARTIFACT VERIFICATION FAILED:\n  " + "\n  ".join(issues))
        sys.exit(1)
    print("artifact verification OK")
