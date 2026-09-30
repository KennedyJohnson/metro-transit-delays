"""Static GTFS helpers shared by the collector and the model/export job."""
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
GTFS_DIR = ROOT / "data" / "gtfs"
GTFS_URL = "https://svc.metrotransit.org/mtgtfs/gtfs.zip"


def ensure_zip(max_age_s=86400) -> Path:
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    z = GTFS_DIR / "gtfs.zip"
    if not z.exists() or datetime.now().timestamp() - z.stat().st_mtime > max_age_s:
        r = requests.get(GTFS_URL, timeout=180)
        r.raise_for_status()
        z.write_bytes(r.content)
    return z


def read(z: Path, name: str, usecols=None) -> pd.DataFrame | None:
    with zipfile.ZipFile(z) as zf:
        if name not in zf.namelist():
            return None
        with zf.open(name) as f:
            df = pd.read_csv(f, dtype=str, usecols=usecols, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def to_secs(s: pd.Series) -> np.ndarray:
    """'25:05:00' -> 90300 (GTFS times may pass 24:00)."""
    p = s.str.strip().str.split(":", expand=True).astype(float)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def stop_positions(z: Path) -> pd.DataFrame:
    """One row per (trip_id, stop_sequence): scheduled seconds, index of the stop within the trip, trip length,
    and the trip's first scheduled departure."""
    st = read(z, "stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    st["sched_s"] = to_secs(st.arrival_time.where(st.arrival_time.notna() & (st.arrival_time.str.strip() != ""), st.departure_time))
    st["stop_sequence"] = st.stop_sequence.astype(int)
    st = st.sort_values(["trip_id", "stop_sequence"])
    g = st.groupby("trip_id", sort=False)
    st["stop_idx"] = g.cumcount()
    st["n_stops"] = g.stop_sequence.transform("size")
    st["trip_start_s"] = g.sched_s.transform("first")
    return st[["trip_id", "stop_sequence", "stop_id", "sched_s", "stop_idx", "n_stops", "trip_start_s"]]


def active_services(z: Path, days: list[date]) -> dict[str, list[str]]:
    """service_ids running on each date (calendar.txt + calendar_dates.txt exceptions)."""
    out = {d.isoformat(): set() for d in days}
    cal = read(z, "calendar.txt")
    names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if cal is not None:
        for row in cal.itertuples():
            start, end = (datetime.strptime(getattr(row, k), "%Y%m%d").date() for k in ("start_date", "end_date"))
            for d in days:
                if start <= d <= end and getattr(row, names[d.weekday()]).strip() == "1":
                    out[d.isoformat()].add(row.service_id)
    cd = read(z, "calendar_dates.txt")
    if cd is not None:
        for row in cd.itertuples():
            d = datetime.strptime(row.date, "%Y%m%d").date().isoformat()
            if d in out:
                (out[d].add if row.exception_type.strip() == "1" else out[d].discard)(row.service_id)
    return {d: sorted(s) for d, s in out.items()}


def next_days(n=14, start: date | None = None) -> list[date]:
    start = start or date.today()
    return [start + timedelta(days=i) for i in range(n)]
