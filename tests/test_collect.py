import sys
import zipfile
from datetime import datetime
from pathlib import Path

from google.transit import gtfs_realtime_pb2 as rt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collector"))
import collect  # noqa: E402


def make_feed():
    f = rt.FeedMessage()
    f.header.gtfs_realtime_version = "2.0"
    f.header.timestamp = int(datetime(2026, 10, 5, 8, 0, tzinfo=collect.TZ).timestamp())
    # trip A: feed gives delay directly; its first update was already passed (SKIPPED) so the next one is used
    a = f.entity.add(id="a").trip_update
    a.trip.trip_id, a.trip.route_id, a.trip.start_date = "A", "21", "20261005"
    s = a.stop_time_update.add(stop_sequence=3, stop_id="100")
    s.schedule_relationship = s.SKIPPED
    s = a.stop_time_update.add(stop_sequence=4, stop_id="101")
    s.arrival.delay = 240
    # trip B: only a predicted time -> compare with the schedule (08:10:00 local, predicted 08:13:30)
    b = f.entity.add(id="b").trip_update
    b.trip.trip_id, b.trip.route_id, b.trip.start_date, b.trip.direction_id = "B", "902", "20261005", 1
    s = b.stop_time_update.add(stop_sequence=7, stop_id="200")
    s.arrival.time = int(datetime(2026, 10, 5, 8, 13, 30, tzinfo=collect.TZ).timestamp())
    # trip C: canceled
    c = f.entity.add(id="c").trip_update
    c.trip.trip_id, c.trip.route_id = "C", "5"
    c.trip.schedule_relationship = c.trip.CANCELED
    # trip D: absurd delay is dropped
    d = f.entity.add(id="d").trip_update
    d.trip.trip_id, d.trip.route_id = "D", "5"
    d.stop_time_update.add(stop_sequence=1, stop_id="1").arrival.delay = 5 * 3600
    return f.SerializeToString()


def test_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "GTFS", tmp_path)
    with zipfile.ZipFile(tmp_path / "gtfs.zip", "w") as z:
        z.writestr("stop_times.txt", "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nB,08:10:00,08:10:00,200,7\n")
    ts, rows = collect.parse(make_feed(), collect.Schedule())
    by_trip = {r[3]: r for r in rows}
    assert set(by_trip) == {"A", "B", "C"}
    assert by_trip["A"][5:9] == [4, "101", 240, "feed"]
    assert by_trip["B"][7:9] == [210, "schedule"] and by_trip["B"][2] == 1
    assert by_trip["C"][8] == "canceled"


def test_after_midnight_trip(tmp_path, monkeypatch):
    """GTFS times past 24:00 belong to the previous service day."""
    monkeypatch.setattr(collect, "GTFS", tmp_path)
    with zipfile.ZipFile(tmp_path / "gtfs.zip", "w") as z:
        z.writestr("stop_times.txt", "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nN,25:05:00,25:05:00,9,2\n")
    predicted = int(datetime(2026, 10, 6, 1, 6, 0, tzinfo=collect.TZ).timestamp())
    assert collect.Schedule().delay("N", "20261005", 2, predicted) == 60
