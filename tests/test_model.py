"""End-to-end: synthetic schedule + observations with known structure -> train, export, live-score, and check the
browser code (docs/features.js + docs/model.js) reproduces the Python features and predictions exactly."""
import json
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "model"), str(ROOT / "tests"), str(ROOT / "collector")]
import features as F  # noqa: E402
import synth  # noqa: E402
import train  # noqa: E402
import trees  # noqa: E402
from common import gtfs  # noqa: E402

DAYS = 16


def weather_for(days):
    rng = np.random.default_rng(1)
    t = pd.date_range(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=3), freq="h")
    snowy = rng.random(len(t) // 24 + 1) < 0.25
    snow = np.repeat(snowy, 24)[: len(t)] * rng.gamma(2, 0.6, len(t))
    return pd.DataFrame({"time": t, "temperature_2m": rng.normal(0, 8, len(t)), "precipitation": snow,
                         "snowfall": snow, "wind_speed_10m": rng.gamma(2, 5, len(t))})


def observations(z, days, wx):
    pos = gtfs.stop_positions(z)
    trips = gtfs.read(z, "trips.txt").set_index("trip_id")
    pos = pos.join(trips[["route_id", "direction_id", "service_id"]], on="trip_id")
    snow = wx.set_index("time").snowfall
    rng = np.random.default_rng(0)
    trip_effect = {t: rng.normal(0, 2) for t in trips.index}
    out = {}
    for d in days:
        svc = "WK" if d.dayofweek < 5 else "WE"
        p = pos[pos.service_id == svc].sample(frac=0.35, random_state=int(d.dayofyear))
        hour = (p.sched_s // 3600).astype(int)
        rush = np.where(hour.isin([7, 8, 16, 17]), 3.0, 0.0)
        mu = (np.where(p.route_id == "21", 3.0, 0.5) + rush + p.trip_id.map(trip_effect) + 2.5 * p.stop_idx / 9
              + 3 * snow.reindex(d + pd.to_timedelta(hour, unit="h")).to_numpy())
        delay = (60 * (mu + rng.normal(0, 1.5, len(p)))).round().astype(int)
        mid = int(d.tz_localize(F.TZ).timestamp())
        out[d] = pd.DataFrame({"ts": mid + p.sched_s - 120, "route_id": p.route_id, "direction_id": p.direction_id,
                               "trip_id": p.trip_id, "start_date": d.strftime("%Y%m%d"), "stop_sequence": p.stop_sequence,
                               "stop_id": p.stop_id, "delay_s": delay, "source": "feed", "sched_s": p.sched_s,
                               "stop_idx": p.stop_idx, "n_stops": p.n_stops, "trip_start_s": p.trip_start_s})
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    z = synth.write_gtfs(tmp_path / "gtfs.zip")
    monkeypatch.setattr(train, "RAW", tmp_path / "raw")
    monkeypatch.setattr(train, "OUT", tmp_path / "docs")
    monkeypatch.setattr(gtfs, "ensure_zip", lambda *a, **k: z)
    monkeypatch.setattr(train, "STALE_HOURS", 24 * 30)
    today = pd.Timestamp(date.today())
    days = list(pd.date_range(today - pd.Timedelta(days=DAYS), today - pd.Timedelta(days=1)))
    wx = weather_for(days)
    monkeypatch.setattr(train, "get_weather", lambda s, e: wx)
    obs = observations(z, days, wx)
    (tmp_path / "raw").mkdir()
    return tmp_path, days, obs, wx


def put(tmp, obs, days):
    for d in days:
        obs[d].to_csv(tmp / "raw" / f"{d:%Y-%m-%d}.csv", index=False)


def test_collecting_exports_schedules(world):
    tmp, days, obs, _ = world
    put(tmp, obs, days[:3])
    train.main()
    meta = json.loads((tmp / "docs/meta.json").read_text())
    assert meta["model"] is False and meta["days_collected"] == 3
    routes = json.loads((tmp / "docs/routes.json").read_text())["routes"]
    assert {r["id"] for r in routes} == {"21", "902"}
    body = json.loads((tmp / "docs/routes/21.json").read_text())
    east = body["dirs"]["0"]
    assert [s[0] for s in east["stops"]] == synth.STOPS and east["name"] == "East to S9"
    assert all(len(t["t"]) == len(synth.STOPS) for t in east["trips"])
    assert {t["id"] for t in east["trips"]} == {t[0] for t in synth.trips() if t[1] == "21" and t[2] == 0}


def test_train_live_and_parity(world):
    tmp, days, obs, wx = world
    put(tmp, obs, days[:-2])
    train.main()
    m = json.loads((tmp / "docs/metrics.json").read_text())
    assert m["mae"] < m["mae_trip_history"] < m["mae_on_time"]
    assert m["late_auc"] > m["late_auc_trip_history"]
    assert json.loads((tmp / "docs/live.json").read_text()) == []

    # two more days arrive; the next run scores the published model on them before retraining
    put(tmp, obs, days[-2:])
    snapshot = tmp / "published"
    shutil.copytree(tmp / "docs", snapshot)
    train.main()
    live = json.loads((tmp / "docs/live.json").read_text())
    assert [x["date"] for x in live] == [f"{d:%Y-%m-%d}" for d in days[-2:]]
    assert live[0]["mae"] < live[0]["mae_on_time"]

    # parity: browser features + model == Python features + model, on the published snapshot
    monkeypatch_out(snapshot)
    st, rec = train.published_stats()
    meta = json.loads((snapshot / "meta.json").read_text())
    cal = json.loads((snapshot / "calendar.json").read_text())
    routes = {r["id"]: r for r in json.loads((snapshot / "routes.json").read_text())["routes"]}
    g = F.load_obs(tmp / "raw")
    g = g[g.date == days[-1]].sample(40, random_state=0)
    rec_day = pd.Series(rec.to_numpy(), index=pd.MultiIndex.from_arrays([rec.index, [days[-1]] * len(rec)]))
    X = F.build(g, st, rec_day, wx)
    cases = []
    for (_, o), (_, x) in zip(g.iterrows(), X.iterrows()):
        body = json.loads((snapshot / "routes" / f"{routes[o.route_id]['file']}.json").read_text())
        dirb = body["dirs"][str(o.direction_id)]
        col = [s[0] for s in dirb["stops"]].index(o.stop_id)
        trip = next(t for t in dirb["trips"] if t["start"] == o.trip_start_s // 60 and t["t"][col] is not None
                    and body["svc"][t["s"]] == ("WK" if days[-1].dayofweek < 5 else "WE"))
        cases.append({"args": {"dateStr": f"{days[-1]:%Y-%m-%d}", "dir": next(d for d in routes[o.route_id]["dirs"] if d["id"] == o.direction_id),
                               "trip": trip, "col": col, "stopMean": dirb["stops"][col][2], "meta": meta,
                               "holidays": cal["holidays"]},
                      "py": [None if np.isnan(v) else v for v in x.to_numpy()]})
    wx_map = {f"{t:%Y-%m-%dT%H:00}": {c: float(r[c]) for c in F.WX} for t, r in wx.set_index("time").iterrows()}
    (tmp / "cases.json").write_text(json.dumps({"cases": cases, "weather": wx_map, "features": F.FEATURES}))
    out = subprocess.run(["node", str(ROOT / "tests" / "parity.js"), str(tmp / "cases.json"), str(snapshot)],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    for case, row in zip(cases, js["rows"]):
        np.testing.assert_allclose(np.array(row, dtype=float), np.array(case["py"], dtype=float), rtol=1e-9, atol=1e-9, equal_nan=True)
    for k in train.TARGETS:
        py = trees.TreeModel(json.loads((snapshot / f"model_{k}.json").read_text())).predict(X.to_numpy())
        np.testing.assert_allclose(js["preds"][k], py, rtol=1e-9, atol=1e-12)


def monkeypatch_out(path):
    train.OUT = path


def test_load_obs_drops_not_started_placeholders(tmp_path):
    mid = int(pd.Timestamp("2026-10-05").tz_localize(F.TZ).timestamp())
    base = dict(route_id="21", direction_id=0, start_date="20261005", stop_id="S0", source="feed", n_stops=10, trip_start_s=28800)
    rows = [
        base | dict(trip_id="started", stop_sequence=3, stop_idx=2, sched_s=29000, delay_s=120, ts=mid + 29000 + 60),
        base | dict(trip_id="placeholder", stop_sequence=1, stop_idx=0, sched_s=30000, delay_s=0, ts=mid + 30000 - 1500),
        base | dict(trip_id="far_ahead", stop_sequence=5, stop_idx=4, sched_s=32000, delay_s=60, ts=mid + 32000 - 1800),
        base | dict(trip_id="first_stop_late", stop_sequence=1, stop_idx=0, sched_s=31000, delay_s=240, ts=mid + 31000 + 200),
    ]
    (tmp_path / "raw").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "raw" / "2026-10-05.csv", index=False)
    assert sorted(F.load_obs(tmp_path / "raw").trip_id) == ["first_stop_late", "started"]


def test_train_with_row_cap(world, monkeypatch):
    """Past MAX_TRAIN_ROWS the models fit on a sample, but history stats and exports still use every row."""
    tmp, days, obs, _ = world
    put(tmp, obs, days)
    n_all = len(F.load_obs(tmp / "raw"))
    monkeypatch.setattr(train, "MAX_TRAIN_ROWS", n_all // 3)
    train.main()
    meta = json.loads((tmp / "docs/meta.json").read_text())
    assert meta["model"] is True and meta["observations"] == n_all
    for k in ("median", "late", "early"):
        assert json.loads((tmp / f"docs/model_{k}.json").read_text())["trees"]
