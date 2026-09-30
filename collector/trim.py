"""Keeps the raw snapshot folder compact and bounded (run by the daily job on the `data` branch checkout).

1. Compact: each finished day's CSV (or older .csv.gz) becomes a Parquet file holding only the rows the model can
   use (features.usable: the last sighting per trip and stop) plus cancellations, with categorical strings, 32-bit
   integers, rows grouped by trip and zstd compression. That's about a tenth of the live CSV and a third of gzip,
   and loads identically.
2. Retain: delete days older than RETAIN_DAYS, then, if the folder is still over MAX_BYTES, the oldest days until
   it fits. Three years covers each season, holiday and school term several times over.

    RAW_DIR=store python collector/trim.py [--days 1095] [--max-mb 600]
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import features as F  # noqa: E402

RAW = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
RETAIN_DAYS = 3 * 365
MAX_BYTES = 600 * 1024 * 1024  # stays well inside GitHub's repo size guidance
TZ = ZoneInfo("America/Chicago")
STRINGS = ["route_id", "trip_id", "stop_id", "start_date", "source"]


def day_of(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.name[:10])
    except ValueError:
        return None


def compact(src: Path) -> Path:
    """Rewrite one finished day as compact Parquet next to it and remove the original."""
    df = F.read_raw(src)
    keep = pd.concat([F.usable(df), df[df.source == "canceled"]])
    for c in keep.columns:
        keep[c] = keep[c].astype("category") if c in STRINGS else keep[c].astype("Int32")
    keep = keep.sort_values(["route_id", "direction_id", "trip_id", "ts"])
    out = src.with_name(f"{day_of(src)}.parquet")
    keep.to_parquet(out, index=False, compression="zstd", compression_level=19)
    src.unlink()
    return out


def trim(raw: Path, today: date, retain_days: int = RETAIN_DAYS, max_bytes: int = MAX_BYTES) -> tuple[list[str], list[str]]:
    """Compact every finished day, then drop old days by age and total size. Returns (compacted, deleted)."""
    compacted, deleted = [], []
    for f in sorted([*raw.glob("*.csv"), *raw.glob("*.csv.gz")]):
        d = day_of(f)
        if d is not None and d < today:  # today's file is still being appended to
            compacted.append(compact(f).name)
    days = sorted((f for f in F.raw_files(raw) if day_of(f)), key=lambda f: (day_of(f), f.name))
    cutoff = today - timedelta(days=retain_days)
    total = sum(f.stat().st_size for f in days)
    for f in days:
        if day_of(f) < cutoff or (total > max_bytes and day_of(f) < today):
            total -= f.stat().st_size
            f.unlink()
            deleted.append(f.name)
    return compacted, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=RETAIN_DAYS)
    ap.add_argument("--max-mb", type=int, default=MAX_BYTES // (1024 * 1024))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    compacted, deleted = trim(RAW, datetime.now(TZ).date(), args.days, args.max_mb * 1024 * 1024)
    files = F.raw_files(RAW)
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"compacted {len(compacted)}, deleted {len(deleted)}; keeping {len(files)} days, {mb:.1f} MB")


if __name__ == "__main__":
    main()
