import gzip
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "collector"), str(ROOT / "model")]
import trim  # noqa: E402
import train  # noqa: E402


def test_trim_gzips_finished_days_and_drops_old_ones(tmp_path):
    for name in ["2025-09-29.csv.gz", "2025-10-01.csv.gz", "2026-09-29.csv", "2026-09-30.csv", "notes.txt"]:
        (tmp_path / name).write_bytes(b"ts,x\n1,2\n")
    gzipped, deleted = trim.trim(tmp_path, date(2026, 9, 30), retain_days=365)
    assert gzipped == ["2026-09-29.csv"]
    assert deleted == ["2025-09-29.csv.gz"]  # 366 days old; 2025-10-01 is inside the window
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["2025-10-01.csv.gz", "2026-09-29.csv.gz", "2026-09-30.csv", "notes.txt"]
    assert gzip.decompress((tmp_path / "2026-09-29.csv.gz").read_bytes()) == b"ts,x\n1,2\n"


def test_train_sample_caps_rows_evenly():
    df = pd.DataFrame({"date": np.repeat(pd.date_range("2026-01-01", periods=100), 50)})
    assert train.train_sample(df, n=10_000) is None  # small enough: use everything
    rows = train.train_sample(df, n=1_000)
    assert len(rows) == 1_000 and len(set(rows)) == 1_000
    per_day = df.loc[rows].date.value_counts()
    assert per_day.index.nunique() > 90  # spread across the window, not just the latest days
