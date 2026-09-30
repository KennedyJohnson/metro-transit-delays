import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "collector"), str(ROOT / "model")]
import features as F  # noqa: E402
import trim  # noqa: E402
import train  # noqa: E402


def day_rows(day: str, n_trips: int = 30) -> pd.DataFrame:
    """Raw snapshots for one service day: each trip seen at 3 stops, twice at the middle one, plus a cancellation."""
    mid = int(pd.Timestamp(day).tz_localize(F.TZ).timestamp())
    rows = []
    for t in range(n_trips):
        start = 6 * 3600 + t * 600
        for k, (idx, lead) in enumerate([(1, 300), (2, 700), (2, 200), (3, 100)]):
            sched = start + idx * 300
            rows.append(dict(ts=mid + sched - lead, route_id="21", direction_id=t % 2, trip_id=f"T{t}",
                             start_date=day.replace("-", ""), stop_sequence=idx + 1, stop_id=f"S{idx}",
                             delay_s=60 * (t % 7) - 60 + k, source="feed", sched_s=sched, stop_idx=idx, n_stops=10,
                             trip_start_s=start))
    rows.append(dict(ts=mid + 7 * 3600, route_id="21", direction_id=0, trip_id="TX", start_date=day.replace("-", ""),
                     stop_sequence=None, stop_id=None, delay_s=None, source="canceled", sched_s=None, stop_idx=None,
                     n_stops=None, trip_start_s=None))
    return pd.DataFrame(rows)


def test_compacted_days_load_identically(tmp_path):
    raw, packed = tmp_path / "raw", tmp_path / "packed"
    raw.mkdir(), packed.mkdir()
    for day in ["2026-09-28", "2026-09-29"]:
        day_rows(day).to_csv(raw / f"{day}.csv", index=False)
        day_rows(day).to_csv(packed / f"{day}.csv.gz", index=False)
    day_rows("2026-09-30").to_csv(packed / "2026-09-30.csv", index=False)  # today: left alone
    compacted, deleted = trim.trim(packed, date(2026, 9, 30))
    assert compacted == ["2026-09-28.parquet", "2026-09-29.parquet"] and deleted == []
    assert sorted(p.name for p in packed.iterdir()) == ["2026-09-28.parquet", "2026-09-29.parquet", "2026-09-30.csv"]
    (packed / "2026-09-30.csv").unlink()
    key = ["start_date", "trip_id", "stop_sequence"]
    a, b = F.load_obs(raw), F.load_obs(packed)
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True),
                                  b[a.columns].sort_values(key).reset_index(drop=True), check_dtype=False)
    stored = pd.read_parquet(packed / "2026-09-29.parquet")
    assert (stored.source == "canceled").sum() == 1  # kept for later use, though the model skips them
    assert len(stored) < len(day_rows("2026-09-29"))  # repeat sightings of a trip at a stop are dropped
    assert trim.trim(packed, date(2026, 9, 30)) == ([], [])  # idempotent


def test_retention_by_age_and_size(tmp_path):
    for day in ["2023-09-01", "2024-01-01", "2026-09-28", "2026-09-29"]:
        (tmp_path / f"{day}.parquet").write_bytes(b"x" * 1000)
    (tmp_path / "2026-09-30.csv").write_bytes(b"x" * 1000)
    (tmp_path / "README.md").write_text("kept")
    _, deleted = trim.trim(tmp_path, date(2026, 9, 30), retain_days=1095, max_bytes=10_000)
    assert deleted == ["2023-09-01.parquet"]  # older than 3 years
    _, deleted = trim.trim(tmp_path, date(2026, 9, 30), retain_days=1095, max_bytes=2_500)
    assert deleted == ["2024-01-01.parquet", "2026-09-28.parquet"]  # oldest first until it fits; today stays
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-09-29.parquet", "2026-09-30.csv", "README.md"]


def test_train_sample_caps_rows_evenly():
    df = pd.DataFrame({"date": np.repeat(pd.date_range("2026-01-01", periods=100), 50)})
    assert train.train_sample(df, n=10_000) is None  # small enough: use everything
    rows = train.train_sample(df, n=1_000)
    assert len(rows) == 1_000 and len(set(rows)) == 1_000
    per_day = df.loc[rows].date.value_counts()
    assert per_day.index.nunique() > 90  # spread across the window, not just the latest days
