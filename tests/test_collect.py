import sys
from datetime import datetime
from pathlib import Path

from google.transit import gtfs_realtime_pb2 as rt

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "collector"), str(ROOT / "tests")]
import collect  # noqa: E402
import synth  # noqa: E402
from common import gtfs  # noqa: E402


def local_ts(*a):
    return int(datetime(*a, tzinfo=collect.TZ).timestamp())


def make_feed():
    f = rt.FeedMessage()
    f.header.gtfs_realtime_version = "2.0"
    f.header.timestamp = local_ts(2026, 10, 5, 8, 0)
    # trip A: feed gives the delay; its first update was already passed (SKIPPED) so the next one is used
    a = f.entity.add(id="a").trip_update
    a.trip.trip_id, a.trip.route_id, a.trip.start_date = "21-0-WK-460", "21", "20261005"
    s = a.stop_time_update.add(stop_sequence=3, stop_id="S2")
    s.schedule_relationship = s.SKIPPED
    a.stop_time_update.add(stop_sequence=4, stop_id="S3").arrival.delay = 240
    # trip B: only a predicted time. Scheduled at stop 2 (S1): 7:40 + 4 min = 07:44; predicted 07:47:30
    b = f.entity.add(id="b").trip_update
    b.trip.trip_id, b.trip.route_id, b.trip.start_date, b.trip.direction_id = "902-0-WK-460", "902", "20261005", 0
    b.stop_time_update.add(stop_sequence=2, stop_id="S1").arrival.time = local_ts(2026, 10, 5, 7, 47, 30)
    # trip C: canceled
    c = f.entity.add(id="c").trip_update
    c.trip.trip_id, c.trip.route_id = "21-1-WK-480", "21"
    c.trip.schedule_relationship = c.trip.CANCELED
    # trip D: absurd delay is dropped
    d = f.entity.add(id="d").trip_update
    d.trip.trip_id, d.trip.route_id = "21-1-WK-500", "21"
    d.stop_time_update.add(stop_sequence=1, stop_id="S9").arrival.delay = 5 * 3600
    return f.SerializeToString()


def test_parse_and_schedule(tmp_path):
    pos = gtfs.stop_positions(synth.write_gtfs(tmp_path / "g.zip"))
    _, df = collect.parse(make_feed())
    out = collect.clean(collect.add_schedule(df, pos)).set_index("trip_id")
    assert set(out.index) == {"21-0-WK-460", "902-0-WK-460", "21-1-WK-480"}
    a = out.loc["21-0-WK-460"]
    assert (a.delay_s, a.source, a.stop_idx, a.n_stops, a.trip_start_s) == (240, "feed", 3, 10, 460 * 60)
    assert a.sched_s == (460 + 12) * 60
    b = out.loc["902-0-WK-460"]
    assert (b.delay_s, b.source, b.stop_idx) == (210, "schedule", 1)
    assert out.loc["21-1-WK-480"].source == "canceled"


def test_after_midnight_without_start_date(tmp_path):
    """A 25:05 stop polled at 1:06 am with no start_date belongs to the previous service day."""
    extra = "N,25:01:00,25:01:00,S0,1\nN,25:05:00,25:05:00,S1,2\n"
    pos = gtfs.stop_positions(synth.write_gtfs(tmp_path / "g.zip", extra))
    f = rt.FeedMessage()
    f.header.gtfs_realtime_version = "2.0"
    f.header.timestamp = local_ts(2026, 10, 6, 1, 5)
    e = f.entity.add(id="n").trip_update
    e.trip.trip_id, e.trip.route_id = "N", "21"
    e.stop_time_update.add(stop_sequence=2, stop_id="S1").arrival.time = local_ts(2026, 10, 6, 1, 6)
    _, df = collect.parse(f.SerializeToString())
    out = collect.clean(collect.add_schedule(df, pos))
    assert out.iloc[0].delay_s == 60 and out.iloc[0].start_date == "20261005"


def test_active_services(tmp_path):
    z = synth.write_gtfs(tmp_path / "g.zip")
    from datetime import date
    s = gtfs.active_services(z, [date(2026, 10, 5), date(2026, 10, 10)])
    assert s == {"2026-10-05": ["WK"], "2026-10-10": ["WE"]}
