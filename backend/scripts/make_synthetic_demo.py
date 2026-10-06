"""Generate the optional public demo dataset (synthetic; not from any real person or study).

Writes backend/demo_data/synthetic_01.csv in the normalized upload format: 3 days of 5-minute
glucose (smooth curve + random walk), three meals and boluses per day, one 50-minute sensor gap.
Deterministic (seeded). Run from the repository root: python backend/scripts/make_synthetic_demo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.tests.helpers import COLS, synthetic_events   # noqa: E402

OUT = ROOT / "backend" / "demo_data" / "synthetic_01.csv"

if __name__ == "__main__":
    ev = synthetic_events(start="2027-03-01 06:00:00", hours=72, seed=2027, gaps=((1500, 50),))
    ev = ev.reindex(columns=COLS)
    ev["timestamp"] = ev["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(ev.to_csv(index=False, na_rep="", lineterminator="\n"), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(ev)} rows)")
