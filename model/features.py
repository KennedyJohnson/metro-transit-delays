"""Observations -> per-departure features. docs/app.js `featureRow()` mirrors `build()` exactly; keep them in sync.

Every feature is known before the day of travel: when the bus is scheduled at the stop, where the stop falls in
the trip, how late this route/direction, this scheduled trip and this stop have run historically (smoothed
means), how late the route ran over the past week, and the weather forecast for that hour.
"""
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TZ = ZoneInfo("America/Chicago")
K = 20  # smoothing strength: a trip/stop needs ~20 observations before its own history outweighs its route's
LATE, EARLY = 300, -60  # 5+ min late; left 1+ min early
MAX_AHEAD_MIN = 10  # an observation is the bus nearing its next stop, not a forecast far ahead
WX = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
FEATURES = ["hour", "dow", "holiday", "progress", "stop_idx", "start_hour", "rd_mean", "rd_late", "rd_early",
            "trip_mean", "trip_late", "trip_early", "trip_n", "stop_mean", "recent7", *WX]


def daytype(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([dates.dt.dayofweek < 5, dates.dt.dayofweek == 5], ["wk", "sat"], "sun"), index=dates.index)


def load_obs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"route_id": str, "trip_id": str, "stop_id": str, "start_date": str})
                    for f in files], ignore_index=True)
    df = df[(df.source != "canceled") & df.delay_s.notna() & df.sched_s.notna()].copy()
    df["date"] = pd.to_datetime(df.start_date, format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    # Keep only real sightings. A trip that hasn't left its first stop is reported with delay 0 until it starts,
    # so drop those placeholders, and anything whose predicted arrival is more than MAX_AHEAD_MIN away.
    midnight = ((df.date + pd.Timedelta(hours=12)).dt.tz_localize(TZ) - pd.Timedelta(hours=12)
                - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)  # unit-safe epoch seconds
    ahead_min = (midnight + df.sched_s + df.delay_s - df.ts) / 60
    not_started = (df.stop_idx == 0) & (df.delay_s == 0) & (midnight + df.sched_s > df.ts)
    df = df[(ahead_min <= MAX_AHEAD_MIN) & ~not_started].copy()
    # a trip seen at the same stop in several polls: keep the last snapshot (closest to arrival)
    df = df.sort_values("ts").drop_duplicates(["date", "trip_id", "stop_sequence"], keep="last")
    df["direction_id"] = df.direction_id.fillna(0).astype(int)
    df["rd"] = df.route_id + "|" + df.direction_id.astype(str)
    df["dt"] = daytype(df.date)
    df["trip_key"] = df.rd + "|" + df.dt + "|" + (df.trip_start_s // 60).astype(int).astype(str)
    df["stop_key"] = df.rd + "|" + df.stop_id
    df["delay_min"] = df.delay_s / 60
    df["late"] = (df.delay_s >= LATE).astype(int)
    df["early"] = (df.delay_s <= EARLY).astype(int)
    return df.reset_index(drop=True)


def _smooth(g, prior):
    return (g["sum"] + K * prior) / (g["n"] + K)


def stats(df: pd.DataFrame) -> dict:
    """Smoothed historical means. Route/direction shrinks toward the overall mean; trips and stops shrink toward
    their route/direction."""
    out = {"global": {t: float(df[t].mean()) for t in ("delay_min", "late", "early")}}
    rd = {}
    for t in ("delay_min", "late", "early"):
        g = df.groupby("rd")[t].agg(sum="sum", n="size")
        rd[t] = _smooth(g, out["global"][t])
    out["rd"] = pd.DataFrame(rd)
    trip = {}
    tg = df.groupby("trip_key")
    rd_of_trip = tg.rd.first()
    for t in ("delay_min", "late", "early"):
        g = tg[t].agg(sum="sum", n="size")
        trip[t] = _smooth(g, out["rd"][t].reindex(rd_of_trip).to_numpy())
    trip["n"] = tg.size()
    out["trip"] = pd.DataFrame(trip)
    sg = df.groupby("stop_key")
    g = sg.delay_min.agg(sum="sum", n="size")
    out["stop"] = _smooth(g, out["rd"].delay_min.reindex(sg.rd.first()).to_numpy()).rename("delay_min").to_frame()
    return out


def recent7(df: pd.DataFrame) -> pd.Series:
    """Mean lateness of each route/direction over the 7 days *before* each date: index (rd, date)."""
    daily = df.groupby(["date", "rd"]).delay_min.agg(["sum", "size"]).unstack("rd")
    days = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(days=1))
    daily = daily.reindex(days).fillna(0)
    roll = daily.rolling(7, min_periods=1).sum().shift(1)
    mean = (roll["sum"] / roll["size"].replace(0, np.nan)).stack().rename("recent7")
    mean.index = mean.index.set_names(["date", "rd"])
    return mean.swaplevel().sort_index()


def holidays(start, end) -> pd.DatetimeIndex:
    return USFederalHolidayCalendar().holidays(start, end)


def build(rows: pd.DataFrame, st: dict, rec: pd.Series, weather: pd.DataFrame | None) -> pd.DataFrame:
    """rows need: date, rd, trip_key, stop_key, sched_s, stop_idx, n_stops, trip_start_s."""
    X = pd.DataFrame(index=rows.index)
    X["hour"] = (rows.sched_s / 3600) % 24
    X["dow"] = rows.date.dt.dayofweek
    X["holiday"] = rows.date.isin(holidays(rows.date.min(), rows.date.max())).astype(int)
    X["progress"] = rows.stop_idx / np.maximum(rows.n_stops - 1, 1)
    X["stop_idx"] = rows.stop_idx
    X["start_hour"] = rows.trip_start_s / 3600
    rd = st["rd"].reindex(rows.rd)
    X["rd_mean"], X["rd_late"], X["rd_early"] = (rd[t].to_numpy() for t in ("delay_min", "late", "early"))
    g = st["global"]
    for c, t in [("rd_mean", "delay_min"), ("rd_late", "late"), ("rd_early", "early")]:
        X[c] = X[c].fillna(g[t])
    tr = st["trip"].reindex(rows.trip_key)
    for c, t, fb in [("trip_mean", "delay_min", "rd_mean"), ("trip_late", "late", "rd_late"), ("trip_early", "early", "rd_early")]:
        X[c] = np.where(np.isnan(tr[t].to_numpy()), X[fb], tr[t].to_numpy())
    X["trip_n"] = np.log1p(tr["n"].fillna(0).to_numpy())
    sm = st["stop"].delay_min.reindex(rows.stop_key).to_numpy()
    X["stop_mean"] = np.where(np.isnan(sm), X.rd_mean, sm)
    X["recent7"] = rec.reindex(pd.MultiIndex.from_arrays([rows.rd, rows.date])).to_numpy()
    if weather is not None and len(weather):
        when = rows.date + pd.to_timedelta((rows.sched_s // 3600).astype(int), unit="h")
        w = weather.set_index("time").reindex(when)
        for c in WX:
            X[c] = w[c].to_numpy()
    else:
        for c in WX:
            X[c] = np.nan
    return X[FEATURES].astype(float)


def oof_stats_features(df: pd.DataFrame, rec, weather, folds=5, seed=0, rows=None) -> pd.DataFrame:
    """Training features with out-of-fold history stats (folds are whole days, so a trip's own outcome that day
    never feeds its own features). rows: build features for only these index labels (a training sample), while
    the history stats still come from all of df."""
    days = df.date.unique()
    fold_of_day = dict(zip(days, np.random.default_rng(seed).integers(0, folds, len(days))))
    fold = df.date.map(fold_of_day).to_numpy()
    target = df if rows is None else df.loc[rows]
    tfold = target.date.map(fold_of_day).to_numpy()
    parts = []
    for k in range(folds):
        m = tfold == k
        if m.any():
            parts.append(build(target[m], stats(df[fold != k]), rec, weather))
    return pd.concat(parts).loc[target.index]
