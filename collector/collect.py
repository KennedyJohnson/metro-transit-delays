"""Snapshot how late every active Metro Transit trip is running right now.

Reads the GTFS-Realtime TripUpdates feed and keeps each trip's next-stop update: that stop's arrival delay is the
trip's current lateness there. Every row is joined to the static schedule (cached daily in data/gtfs/) so it also
records the scheduled time at that stop, the stop's position in the trip and the trip's start time, which is what
the model needs to learn delays per departure and per stop. Delay comes from the feed's `delay` field when
present, otherwise predicted arrival minus scheduled arrival. Appends to $RAW_DIR/YYYY-MM-DD.csv (local date of
the poll). Run every ~15 minutes.

    python collector/collect.py [--feed tripupdates.pb]
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import gtfs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# raw snapshots live on the repo's `data` branch; the workflows check it out and point RAW_DIR at it
RAW = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
FEED_URL = "https://svc.metrotransit.org/mtgtfs/tripupdates.pb"
TZ = ZoneInfo("America/Chicago")
FIELDS = ["ts", "route_id", "direction_id", "trip_id", "start_date", "stop_sequence", "stop_id", "delay_s", "source",
          "sched_s", "stop_idx", "n_stops", "trip_start_s"]
MAX_ABS_DELAY = 3 * 3600  # beyond +/-3 h is a feed glitch, not lateness
U = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
T = gtfs_realtime_pb2.TripDescriptor


def parse(feed_bytes) -> tuple[int, pd.DataFrame]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(feed_bytes)
    ts = feed.header.timestamp or int(datetime.now().timestamp())
    rows = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        t = tu.trip
        base = dict(route_id=t.route_id, direction_id=t.direction_id, trip_id=t.trip_id, start_date=t.start_date)
        if t.schedule_relationship == T.CANCELED:
            rows.append(base | dict(source="canceled"))
            continue
        nxt = next((u for u in tu.stop_time_update if u.schedule_relationship == U.SCHEDULED
                    and (u.HasField("arrival") or u.HasField("departure"))), None)
        if nxt is None:
            continue
        ev = nxt.arrival if nxt.HasField("arrival") else nxt.departure
        row = base | dict(stop_sequence=nxt.stop_sequence, stop_id=nxt.stop_id)
        if ev.HasField("delay"):
            rows.append(row | dict(delay_s=ev.delay, source="feed"))
        elif ev.time:
            rows.append(row | dict(predicted=ev.time, source="schedule"))
    df = pd.DataFrame(rows, columns=["route_id", "direction_id", "trip_id", "start_date", "stop_sequence", "stop_id",
                                     "delay_s", "predicted", "source"])
    df.insert(0, "ts", ts)
    return ts, df


def service_midnight(start_date: str) -> int:
    """GTFS times count from noon minus 12 h of the service day (local midnight except on DST change days)."""
    noon = datetime.strptime(start_date, "%Y%m%d").replace(hour=12, tzinfo=TZ)
    return int((noon - timedelta(hours=12)).timestamp())


def add_schedule(df: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its scheduled stop time; compute delay where the feed gave only a predicted time."""
    df = df.copy()
    df["stop_sequence"] = pd.to_numeric(df.stop_sequence, errors="coerce").astype("Int64")
    p = pos.drop(columns="stop_id").astype({"stop_sequence": "Int64"})
    df = df.merge(p, on=["trip_id", "stop_sequence"], how="left")
    # service day: from start_date, else the poll's local date (or the day before, whichever fits the schedule)
    local = datetime.fromtimestamp(int(df.ts.iloc[0]), TZ) if len(df) else None
    missing = df.start_date.fillna("").eq("")
    if missing.any() and local is not None:
        today, yday = local.strftime("%Y%m%d"), (local - timedelta(days=1)).strftime("%Y%m%d")
        now_rel = df.ts - service_midnight(today)
        df.loc[missing, "start_date"] = np.where((df.sched_s - now_rel).abs() > 12 * 3600, yday, today)[missing.to_numpy()]
    need = df.source.eq("schedule") & df.sched_s.notna() & df.start_date.fillna("").ne("")
    if need.any():
        mid = df.loc[need, "start_date"].map(service_midnight)
        df.loc[need, "delay_s"] = df.loc[need, "predicted"] - mid - df.loc[need, "sched_s"]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.source.eq("canceled") | (df.delay_s.notna() & (df.delay_s.abs() <= MAX_ABS_DELAY))
    df = df[keep].copy()
    for c in ["delay_s", "sched_s", "stop_idx", "n_stops", "trip_start_s"]:
        df[c] = df[c].round().astype("Int64")
    return df[FIELDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", help="read a saved TripUpdates .pb instead of fetching")
    args = ap.parse_args()
    if args.feed:
        data = Path(args.feed).read_bytes()
    else:
        r = requests.get(FEED_URL, timeout=30)
        r.raise_for_status()
        data = r.content
    ts, df = parse(data)
    df = clean(add_schedule(df, load_positions()))
    if df.empty:
        sys.exit("feed had no usable trip updates")
    day = datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{day}.csv"
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"{len(df)} trips -> {out.name} ({df.sched_s.notna().mean():.0%} matched to the schedule)")


def load_positions() -> pd.DataFrame:
    """Stop positions from the static GTFS, cached as parquet next to the zip."""
    z = gtfs.ensure_zip()
    cache = z.with_suffix(".positions.parquet")
    if cache.exists() and cache.stat().st_mtime >= z.stat().st_mtime:
        return pd.read_parquet(cache)
    pos = gtfs.stop_positions(z)
    pos.to_parquet(cache, index=False)
    return pos


if __name__ == "__main__":
    main()
