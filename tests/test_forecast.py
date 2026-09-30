"""End-to-end run of the forecast on synthetic snapshots with a known structure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
import forecast as F  # noqa: E402

ROUTES = {"21": 3.0, "5": 2.0, "902": 0.5, "6": 1.5}


def synth_weather(days):
    rng = np.random.default_rng(1)
    idx = pd.MultiIndex.from_product([days, range(24)], names=["date", "hour"]).to_frame(index=False)
    snow = (rng.random(len(days)) < 0.2).repeat(24) * rng.gamma(2, 0.5, len(idx))
    return idx.assign(temperature_2m=rng.normal(0, 8, len(idx)), precipitation=snow, snowfall=snow,
                      wind_speed_10m=rng.gamma(2, 5, len(idx)), weather_code=np.where(snow > 0, 71, 1))


@pytest.fixture
def world(tmp_path, monkeypatch):
    for name, sub in [("RAW", "data/raw"), ("FC", "data/forecasts"), ("OUT", "docs/data")]:
        monkeypatch.setattr(F, name, tmp_path / sub)
    monkeypatch.setattr(F, "ROOT", tmp_path)
    today = pd.Timestamp(pd.Timestamp.now(F.TZ).date())
    days = pd.date_range(today - pd.Timedelta(days=24), today - pd.Timedelta(days=1))
    wx = synth_weather(pd.date_range(days[0] - pd.Timedelta(days=1), today + pd.Timedelta(days=1)))
    monkeypatch.setattr(F, "get_weather", lambda s, e: wx)
    snow = wx.set_index(["date", "hour"]).snowfall
    rng = np.random.default_rng(0)
    (tmp_path / "data/raw").mkdir(parents=True)
    for d in days:
        rows = []
        for h in range(5, 24):
            ts = int(d.tz_localize(F.TZ).timestamp()) + h * 3600 + 600
            rush = 2.0 if h in (7, 8, 16, 17) else 0.0
            for r, base in ROUTES.items():
                for k in range(12):
                    mu = base + rush + 4 * snow[(d, h)] + (1 if d.dayofweek == 4 else 0)
                    rows.append([ts, r, 0, f"{r}-{h}-{k}", d.strftime("%Y%m%d"), 3, "1", int(60 * rng.normal(mu, 1.5)), "feed"])
        pd.DataFrame(rows, columns=["ts", "route_id", "direction_id", "trip_id", "start_date", "stop_sequence",
                                    "stop_id", "delay_s", "source"]).to_csv(tmp_path / f"data/raw/{d.date()}.csv", index=False)
    return tmp_path, today


def test_end_to_end(world):
    tmp, today = world
    F.main()
    out = tmp / "docs/data"
    m = pd.read_json(out / "metrics.json", typ="series")
    assert m.model_mae < m.baseline_last_week_mae and m.model_mae < m.baseline_14day_mean_mae
    fc = pd.read_json(out / "forecast.json", typ="series")
    assert set(fc.routes) == set(ROUTES) and len(fc.routes["21"]["pred"]) == len(F.HOURS)
    assert (tmp / f"data/forecasts/{today.date()}.csv").exists()


def test_live_scoring(world):
    tmp, today = world
    yday = today - pd.Timedelta(days=1)
    (tmp / "data/forecasts").mkdir(parents=True)
    pd.DataFrame([(r, yday, h, 1.0, 0.0) for r in ROUTES for h in F.HOURS],
                 columns=["route", "date", "hour", "pred", "naive"]).to_csv(tmp / f"data/forecasts/{yday.date()}.csv", index=False)
    live = F.score_past_forecasts(F.aggregate(F.load_obs()))
    assert live and live[0]["date"] == str(yday.date()) and live[0]["n"] == len(ROUTES) * len(F.HOURS)


def test_collecting_status(world, monkeypatch):
    tmp, _ = world
    for f in sorted((tmp / "data/raw").glob("*.csv"))[:-3]:
        f.unlink()
    F.main()
    s = pd.read_json(tmp / "docs/data/status.json", typ="series")
    assert s.collecting and s.days_collected == 3
