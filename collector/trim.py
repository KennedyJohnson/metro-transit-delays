"""Keeps the raw snapshot folder bounded: gzips finished days and deletes days older than the retention window.

A year covers every season, holiday and school term once, which is what the model needs; older days add little
(the recent7 feature tracks drift) but keep growing the repo's working tree and the daily job's run time.

    python collector/trim.py [--days 365]
"""
import argparse
import gzip
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RETAIN_DAYS = 365
TZ = ZoneInfo("America/Chicago")


def day_of(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.name[:10])
    except ValueError:
        return None


def trim(raw: Path, today: date, retain_days: int = RETAIN_DAYS) -> tuple[list[str], list[str]]:
    """Gzip every finished day's CSV, then delete days before today - retain_days. Returns (gzipped, deleted)."""
    gzipped, deleted = [], []
    for f in sorted(raw.glob("*.csv")):
        d = day_of(f)
        if d is None or d >= today:
            continue  # today's file is still being appended to
        with f.open("rb") as src, gzip.open(f.with_suffix(".csv.gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        f.unlink()
        gzipped.append(f.name)
    cutoff = today - timedelta(days=retain_days)
    for f in sorted(raw.glob("*.csv*")):
        d = day_of(f)
        if d is not None and d < cutoff:
            f.unlink()
            deleted.append(f.name)
    return gzipped, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=RETAIN_DAYS)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    gzipped, deleted = trim(RAW, datetime.now(TZ).date(), args.days)
    kept = len(list(RAW.glob("*.csv*")))
    print(f"gzipped {len(gzipped)}, deleted {len(deleted)} older than {args.days} days, keeping {kept} days")


if __name__ == "__main__":
    main()
