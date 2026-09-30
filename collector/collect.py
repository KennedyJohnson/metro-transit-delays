"""Snapshot how late every active Metro Transit trip is running right now.

Reads the GTFS-Realtime TripUpdates feed and keeps each trip's next-stop update: that stop's arrival delay is
the trip's current lateness. Uses the feed's `delay` field when present, otherwise predicted arrival time minus
the scheduled time from the static GTFS (`stop_times.txt`, cached in data/gtfs/). Appends one row per trip to
data/raw/YYYY-MM-DD.csv (local service day). Run every ~15 minutes.

    python collector/collect.py [--feed path/to/tripupdates.pb]
"""
import argparse
import csv
import io
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from google.transit import gtfs_realtime_pb2

ROOT = Path(__file__).resolve().parent.parent
RAW, GTFS = ROOT / "data" / "raw", ROOT / "data" / "gtfs"
FEED_URL = "https://svc.metrotransit.org/mtgtfs/tripupdates.pb"
GTFS_URL = "https://svc.metrotransit.org/mtgtfs/gtfs.zip"
TZ = ZoneInfo("America/Chicago")
FIELDS = ["ts", "route_id", "direction_id", "trip_id", "start_date", "stop_sequence", "stop_id", "delay_s", "source"]
MAX_ABS_DELAY = 3 * 3600  # anything beyond +/-3 h is a feed glitch, not lateness
U = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate
T = gtfs_realtime_pb2.TripDescriptor


def fetch(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


class Schedule:
    """Scheduled arrival (seconds after service-day midnight) for (trip_id, stop_sequence), loaded lazily."""

    def __init__(self):
        self._times = None

    def _load(self):
        GTFS.mkdir(parents=True, exist_ok=True)
        zpath = GTFS / "gtfs.zip"
        stale = not zpath.exists() or datetime.now().timestamp() - zpath.stat().st_mtime > 86400
        if stale:
            zpath.write_bytes(fetch(GTFS_URL, timeout=120))
        self._times = {}
        with zipfile.ZipFile(zpath) as z, z.open("stop_times.txt") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")):
                h, m, s = map(int, (row["arrival_time"] or row["departure_time"]).split(":"))
                self._times[(row["trip_id"], int(row["stop_sequence"]))] = h * 3600 + m * 60 + s

    def delay(self, trip_id, start_date, stop_sequence, predicted_unix):
        if self._times is None:
            self._load()
        sched = self._times.get((trip_id, stop_sequence))
        if sched is None or not start_date:
            return None
        # GTFS times count from "noon minus 12h" of the service day, which is local midnight except on DST days
        noon = datetime.strptime(start_date, "%Y%m%d").replace(hour=12, tzinfo=TZ)
        return predicted_unix - int((noon - timedelta(hours=12)).timestamp()) - sched


def parse(feed_bytes, schedule):
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(feed_bytes)
    ts = feed.header.timestamp or int(datetime.now().timestamp())
    rows = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip
        if trip.schedule_relationship == T.CANCELED:
            rows.append([ts, trip.route_id, trip.direction_id, trip.trip_id, trip.start_date, "", "", "", "canceled"])
            continue
        nxt = next((u for u in tu.stop_time_update if u.schedule_relationship == U.SCHEDULED
                    and (u.HasField("arrival") or u.HasField("departure"))), None)
        if nxt is None:
            continue
        ev = nxt.arrival if nxt.HasField("arrival") else nxt.departure
        if ev.HasField("delay"):
            delay, source = ev.delay, "feed"
        elif ev.time:
            delay, source = schedule.delay(trip.trip_id, trip.start_date, nxt.stop_sequence, ev.time), "schedule"
        else:
            continue
        if delay is None or abs(delay) > MAX_ABS_DELAY:
            continue
        rows.append([ts, trip.route_id, trip.direction_id, trip.trip_id, trip.start_date,
                     nxt.stop_sequence, nxt.stop_id, int(delay), source])
    return ts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", help="read a saved TripUpdates .pb instead of fetching")
    args = ap.parse_args()
    data = Path(args.feed).read_bytes() if args.feed else fetch(FEED_URL)
    ts, rows = parse(data, Schedule())
    if not rows:
        sys.exit("feed had no usable trip updates")
    day = datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{day}.csv"
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerows(rows)
    print(f"{len(rows)} trips -> {out.name}")


if __name__ == "__main__":
    main()
