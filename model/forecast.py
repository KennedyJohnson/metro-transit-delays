"""Forecast how late each Metro Transit route will run, by hour, for the rest of today.

1. Aggregate the collector's snapshots to route x service-day x hour: mean lateness (minutes) and share of trips 5+
   minutes late.
2. Features known at forecast time (~4 am): route, hour, day of week, holiday, hourly weather (Open-Meteo:
   archive for the past, forecast for today) and lags from *previous* days only: same route-hour yesterday and a
   week ago, the route-hour's trailing 14-day mean, the route's mean yesterday.
3. Backtest: train on all but the last TEST_DAYS days, score those days against two baselines (same hour last
   week; trailing 14-day route-hour mean).
4. Refit on everything, forecast today, publish docs/data/*.json. Each day's forecast is also saved to
   data/forecasts/ and scored once the day's actuals are in (docs/data/live.json), so the published accuracy
   includes forecasts the model made before seeing the outcome.

    python model/forecast.py
"""
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent.parent
RAW, FC, OUT = ROOT / "data" / "raw", ROOT / "data" / "forecasts", ROOT / "docs" / "data"
TZ = ZoneInfo("America/Chicago")
MSP = dict(latitude=44.98, longitude=-93.27)
WX = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "weather_code"]
MIN_DAYS, TEST_DAYS, MIN_OBS = 10, 7, 5
HOURS = range(5, 24)
FEATURES = ["route", "hour", "dow", "holiday", "lag1d", "lag7d", "roll14", "route_yday", *WX]
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
              verbose=-1, seed=0)
ROUNDS = 400


def load_obs() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv")) + sorted(RAW.glob("*.csv.gz"))
    df = pd.concat([pd.read_csv(f, dtype={"route_id": str, "trip_id": str, "stop_id": str}) for f in files])
    df = df[df.source != "canceled"].dropna(subset=["delay_s"])
    t = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(TZ)
    # service day from GTFS start_date when present (a 1 am trip belongs to the previous day's service)
    df["date"] = pd.to_datetime(df.start_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["date"] = df.date.fillna(t.dt.tz_localize(None).dt.normalize())
    df["hour"] = t.dt.hour
    return df.rename(columns={"route_id": "route"})


def aggregate(obs: pd.DataFrame) -> pd.DataFrame:
    g = obs.groupby(["route", "date", "hour"])
    agg = pd.DataFrame({"delay_min": g.delay_s.mean() / 60, "late5": g.delay_s.apply(lambda s: (s >= 300).mean()),
                        "n": g.size()}).reset_index()
    return agg[agg.n >= MIN_OBS]


def get_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly MSP weather, local time. Archive for older days, forecast API (incl. recent past) for the rest."""
    frames = []
    try:
        cut = min(end, pd.Timestamp.today().normalize() - pd.Timedelta(days=6))
        if start <= cut:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=dict(
                MSP, hourly=",".join(WX), timezone="America/Chicago",
                start_date=start.strftime("%Y-%m-%d"), end_date=cut.strftime("%Y-%m-%d")))
            r.raise_for_status()
            frames.append(pd.DataFrame(r.json()["hourly"]))
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params=dict(
            MSP, hourly=",".join(WX), timezone="America/Chicago", past_days=7, forecast_days=2))
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
    except requests.RequestException as e:
        print("weather unavailable, continuing without it:", e)
    if not frames:
        return pd.DataFrame(columns=["date", "hour", *WX])
    w = pd.concat(frames).drop_duplicates("time", keep="first")
    t = pd.to_datetime(w.pop("time"))
    return w.assign(date=t.dt.normalize(), hour=t.dt.hour)


def add_features(target: pd.DataFrame, history: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """target: rows (route, date, hour) to describe. history: aggregated actuals; only days before each
    target row's date are used."""
    h = history.set_index(["route", "date", "hour"]).delay_min
    X = target[["route", "date", "hour"]].copy()
    for name, days in [("lag1d", 1), ("lag7d", 7)]:
        key = pd.MultiIndex.from_arrays([X.route, X.date - pd.Timedelta(days=days), X.hour])
        X[name] = h.reindex(key).to_numpy()
    # trailing 14-day route-hour mean, excluding the target day
    daily = history.pivot_table(index="date", columns=["route", "hour"], values="delay_min")
    daily = daily.reindex(pd.date_range(daily.index.min(), X.date.max()))
    roll = daily.rolling(14, min_periods=3).mean().shift(1).stack(["route", "hour"], future_stack=True)
    X["roll14"] = roll.reindex(pd.MultiIndex.from_arrays([X.date, X.route, X.hour])).to_numpy()
    yday = history.groupby(["route", "date"]).delay_min.mean()
    X["route_yday"] = yday.reindex(pd.MultiIndex.from_arrays([X.route, X.date - pd.Timedelta(days=1)])).to_numpy()
    X["dow"] = X.date.dt.dayofweek
    hol = USFederalHolidayCalendar().holidays(X.date.min(), X.date.max())
    X["holiday"] = X.date.isin(hol).astype(int)
    X = X.merge(weather, on=["date", "hour"], how="left")
    X["route"] = X.route.astype("category")
    return X


def fit(X, y, categories):
    X = X[FEATURES].copy()
    X["route"] = pd.Categorical(X.route.astype(str), categories=categories)
    return lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS)


def predict(model, X, categories):
    X = X[FEATURES].copy()
    X["route"] = pd.Categorical(X.route.astype(str), categories=categories)
    return model.predict(X)


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return round(float(np.abs(a[m] - b[m]).mean()), 3), int(m.sum())


def backtest(agg, weather, cats):
    days = np.sort(agg.date.unique())
    split = days[-TEST_DAYS]
    X = add_features(agg, agg, weather)
    tr, te = X.date < split, X.date >= split
    model = fit(X[tr], agg.delay_min[tr.values], cats)
    y, p = agg.delay_min[te.values].to_numpy(), predict(model, X[te], cats)
    naive = X.lag7d[te].fillna(X.lag1d[te]).to_numpy()
    res = {"test_days": [str(pd.Timestamp(days[-TEST_DAYS]).date()), str(pd.Timestamp(days[-1]).date())],
           "n": int(te.sum()), "model_mae": mae(p, y)[0],
           "baseline_last_week_mae": mae(naive, y)[0], "baseline_14day_mean_mae": mae(X.roll14[te].to_numpy(), y)[0],
           "mean_abs_delay": round(float(np.abs(y).mean()), 3)}
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
    res["importance"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()
    return res


def score_past_forecasts(agg):
    live = []
    for f in sorted(FC.glob("*.csv")):
        fc = pd.read_csv(f, dtype={"route": str}, parse_dates=["date"])
        j = fc.merge(agg[["route", "date", "hour", "delay_min"]], on=["route", "date", "hour"])
        if len(j) < 50:
            continue
        live.append({"date": f.stem, "n": len(j), "model_mae": mae(j.pred.to_numpy(), j.delay_min.to_numpy())[0],
                     "baseline_last_week_mae": mae(j.naive.to_numpy(), j.delay_min.to_numpy())[0]})
    return live


def route_names():
    z = ROOT / "data" / "gtfs" / "gtfs.zip"
    try:
        with zipfile.ZipFile(z) as zf, zf.open("routes.txt") as f:
            r = pd.read_csv(f, dtype=str)
        return {row.route_id: (row.route_short_name if isinstance(row.route_short_name, str) else "")
                + ((" " + row.route_long_name) if isinstance(row.route_long_name, str) else "") for row in r.itertuples()}
    except (OSError, KeyError, zipfile.BadZipFile):
        return {}


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), default=str))


def main():
    obs = load_obs()
    agg = aggregate(obs)
    days = agg.date.nunique()
    today = pd.Timestamp(datetime.now(TZ).date())
    status = {"updated": datetime.now(TZ).isoformat(timespec="minutes"), "days_collected": int(obs.date.nunique()),
              "observations": int(len(obs)), "first_day": str(obs.date.min().date()), "min_days": MIN_DAYS}
    newest = pd.Timestamp(int(obs.ts.max()), unit="s", tz="UTC")
    if pd.Timestamp.now(tz="UTC") - newest > pd.Timedelta(hours=36):
        write("status.json", status | {"collecting": days < MIN_DAYS, "stale_since": newest.isoformat()})
        raise SystemExit(f"collector hasn't recorded anything since {newest}; check collect.yml")
    if days < MIN_DAYS:
        write("status.json", status | {"collecting": True})
        print(f"only {days} days of data; need {MIN_DAYS} before modeling")
        return

    weather = get_weather(agg.date.min() - pd.Timedelta(days=1), today + pd.Timedelta(days=1))
    cats = sorted(agg.route.astype(str).unique())
    bt = backtest(agg, weather, cats)
    print("backtest", {k: v for k, v in bt.items() if k != "importance"})

    # refit on everything and forecast today for routes that ran in the last week
    X_all = add_features(agg, agg, weather)
    model = fit(X_all, agg.delay_min.to_numpy(), cats)
    recent = agg[agg.date > agg.date.max() - pd.Timedelta(days=7)].route.astype(str).unique()
    grid = pd.DataFrame([(r, today, h) for r in recent for h in HOURS], columns=["route", "date", "hour"])
    Xt = add_features(grid, agg, weather)
    grid["pred"] = predict(model, Xt, cats).round(2)
    grid["naive"] = Xt.lag7d.fillna(Xt.lag1d).round(2).to_numpy()
    grid["typical"] = Xt.roll14.round(2).to_numpy()
    FC.mkdir(parents=True, exist_ok=True)
    grid.drop(columns="typical").to_csv(FC / f"{today.date()}.csv", index=False)

    names = route_names()
    wx_today = weather[weather.date == today].set_index("hour").reindex(list(HOURS))
    write("forecast.json", {
        "date": str(today.date()), "issued": status["updated"],
        "routes": {r: {"name": names.get(r, r), "pred": g.pred.tolist(),
                       "typical": [None if pd.isna(v) else v for v in g.typical]}
                   for r, g in grid.groupby("route")},
        "hours": list(HOURS),
        "weather": {k: [None if pd.isna(v) else round(float(v), 1) for v in wx_today[k]] for k in ["temperature_2m", "precipitation", "snowfall"]},
    })
    write("metrics.json", bt)
    write("live.json", score_past_forecasts(agg))
    write("status.json", status | {"collecting": False})
    print("forecast", len(grid), "route-hours for", today.date())


if __name__ == "__main__":
    main()
