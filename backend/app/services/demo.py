"""Demo patients: CSV files (upload schema) in DEMO_DATA_DIR, loaded once and validated.

Patient ids come from file names and must match ^[A-Za-z0-9_-]{1,40}$; requests can only
select an id from this fixed set (no user-controlled paths).
Which data may be published here is decided separately (licence; see BACKEND_ARCHITECTURE.md 9.4).
"""

import re
from pathlib import Path

import pandas as pd

from backend.app.services.csv_input import parse_events_csv

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class DemoStore:
    def __init__(self, directory: Path | None):
        self.patients: dict[str, pd.DataFrame] = {}
        if directory is None or not Path(directory).is_dir():
            return
        for path in sorted(Path(directory).glob("*.csv")):
            pid = path.stem
            if ID_RE.match(pid):
                events, _ = parse_events_csv(path.read_bytes())   # same validation as uploads
                self.patients[pid] = events

    def ids(self) -> list[str]:
        return list(self.patients)

    def get(self, pid: str) -> pd.DataFrame | None:
        if not ID_RE.match(pid or ""):
            return None
        return self.patients.get(pid)
