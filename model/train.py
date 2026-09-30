"""Daily job: score yesterday's published model, retrain, and export everything the site needs.

1. Live score: the model + history tables the site was serving (docs/data/) are scored on observations
   collected since they were published. Those are departures the model never saw -> docs/data/live.json.
2. Backtest: hold out the last TEST_DAYS days; history tables come from the training days only, like serving.
   Compare with baselines (the trip's own history; its route/direction's history) -> docs/data/metrics.json.
3. Refit three models on everything: typical delay (median, L1 loss), P(5+ min late), P(1+ min early).
4. Export: models, route list, per-route schedules with each trip's/stop's history, service calendar for the
   next 14 days, holidays. Schedules are exported even while still collecting, so the site works from day one.

    python model/train.py
"""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import features as F  # noqa: E402
import trees  # noqa: E402
from common import gtfs  # noqa: E402

# raw snapshots live on the repo's `data` branch; the workflows check it out and point RAW_DIR at it
RAW, OUT = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw")), ROOT / "docs" / "data"
MIN_DAYS, TEST_DAYS, HORIZON = 10, 7, 14
STALE_HOURS = 36  # fail if the collector has been silent this long
MSP = dict(latitude=44.98, longitude=-93.27)
BASE = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
            bagging_freq=1, verbose=-1, seed=0)
TARGETS = {"median": ("delay_min", dict(objective="l1"), "regression"),
           "late": ("late", dict(objective="binary"), "binary"),
           "early": ("early", dict(objective="binary"), "binary")}
ROUNDS = 300
WX_BLANK = 0.15  # train some rows without weather, for days the forecast can't reach
# Keeps the daily job bounded as up to 3 years of snapshots pile up (~19k usable observations a day, ~21M at the
# 3-year cap): fit on an even random sample across the whole retained window, so every season is represented.
# 5M rows is ~1 min per model on the 4-core runner. History stats use every row.
MAX_TRAIN_ROWS = 5_000_000


def train_sample(df: pd.DataFrame, n: int | None = None):
    """Index labels to fit on: all rows if there are at most n, else a seeded random n spread over every day."""
    n = MAX_TRAIN_ROWS if n is None else n
    if len(df) <= n:
        return None
    return np.sort(np.random.default_rng(2).choice(df.index.to_numpy(), n, replace=False))


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), default=str))


def read_json(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def get_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Hourly MSP weather in local time: archive for older days, forecast API (with 7 past days) for the rest."""
    frames = []
    try:
        cut = min(end, pd.Timestamp.today().normalize() - pd.Timedelta(days=6))
        if start <= cut:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=MSP | dict(
                hourly=",".join(F.WX), timezone="America/Chicago", start_date=f"{start:%Y-%m-%d}", end_date=f"{cut:%Y-%m-%d}"))
            r.raise_for_status()
            frames.append(pd.DataFrame(r.json()["hourly"]))
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params=MSP | dict(
            hourly=",".join(F.WX), timezone="America/Chicago", past_days=7, forecast_days=2))
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
    except requests.RequestException as e:
        print("weather unavailable, continuing without it:", e)
        return None
    w = pd.concat(frames).drop_duplicates("time", keep="first")
    return w.assign(time=pd.to_datetime(w.time))


def fit(X, y, params):
    X = X.copy()
    X.loc[np.random.default_rng(1).random(len(X)) < WX_BLANK, F.WX] = np.nan
    return lgb.train(BASE | params, lgb.Dataset(X, y), ROUNDS)


def evaluate(df, X, preds) -> dict:
    y, late, early = df.delay_min.to_numpy(), df.late.to_numpy(), df.early.to_numpy()
    mae = lambda p: round(float(np.nanmean(np.abs(p - y))), 3)  # noqa: E731
    out = {"n": int(len(df)), "mae": mae(preds["median"]), "mae_trip_history": mae(X.trip_mean.to_numpy()),
           "mae_route_history": mae(X.rd_mean.to_numpy()), "mae_on_time": mae(np.zeros(len(y))),
           "late_share": round(float(late.mean()), 4), "early_share": round(float(early.mean()), 4)}
    for k, t, base in [("late", late, "trip_late"), ("early", early, "trip_early")]:
        if 0 < t.mean() < 1:
            out[f"{k}_auc"] = round(roc_auc_score(t, preds[k]), 4)
            out[f"{k}_auc_trip_history"] = round(roc_auc_score(t, X[base]), 4)
            out[f"{k}_brier"] = round(brier_score_loss(t, preds[k]), 4)
            out[f"{k}_mean_pred"] = round(float(np.mean(preds[k])), 4)
    bins = pd.cut(preds["late"], np.linspace(0, 1, 11))
    cal = pd.DataFrame({"p": preds["late"], "y": late}).groupby(bins, observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    out["late_calibration"] = cal.round(4).reset_index(drop=True).to_dict("records")
    return out


# ---------- serving tables <-> stats ----------
def published_stats() -> tuple[dict, pd.Series] | None:
    """Rebuild the history tables from the files the site served (so live scoring uses exactly them)."""
    routes, meta = read_json("routes.json"), read_json("meta.json")
    if not routes or not meta or not meta.get("model"):
        return None
    rd, trip, stop, rec = {}, {}, {}, {}
    for r in routes["routes"]:
        for d in r["dirs"]:
            key = f"{r['id']}|{d['id']}"
            rd[key] = dict(delay_min=d["mean"], late=d["late"], early=d["early"])
            if d.get("recent7") is not None:
                rec[key] = d["recent7"]
        f = OUT / "routes" / f"{r['file']}.json"
        if not f.exists():
            continue
        for d, body in json.loads(f.read_text())["dirs"].items():
            key = f"{r['id']}|{d}"
            for s in body["stops"]:
                if s[2] is not None:
                    stop[f"{key}|{s[0]}"] = s[2]
            for t in body["trips"]:
                for dt, v in (t.get("h") or {}).items():
                    trip[f"{key}|{dt}|{t['start']}"] = dict(delay_min=v[0], late=v[1], early=v[2], n=v[3])
    st = {"global": meta["global"], "rd": pd.DataFrame.from_dict(rd, orient="index"),
          "trip": pd.DataFrame.from_dict(trip, orient="index", columns=["delay_min", "late", "early", "n"]),
          "stop": pd.DataFrame.from_dict({"delay_min": stop})}
    return st, pd.Series(rec, dtype=float)


def live_score(df, weather):
    pub = published_stats()
    meta = read_json("meta.json")
    live = read_json("live.json") or []
    if pub is None:
        return live
    st, rec = pub
    models = {k: trees.TreeModel(read_json(f"model_{k}.json")) for k in TARGETS}
    new = df[df.date > pd.Timestamp(meta["trained_through"])]
    scored = {x["date"] for x in live}
    for day, g in new.groupby("date"):
        d = f"{day:%Y-%m-%d}"
        if d in scored or day >= pd.Timestamp(date.today()) or len(g) < 500:
            continue  # today isn't finished yet
        # serving uses one recent7 value per route/direction, frozen at publish time
        rec_day = pd.Series(rec.to_numpy(), index=pd.MultiIndex.from_arrays([rec.index, [day] * len(rec)]))
        X = F.build(g, st, rec_day, weather)
        preds = {k: m.predict(X[m.features].to_numpy()) for k, m in models.items()}
        live.append({"date": d, "model_trained_through": meta["trained_through"],
                     **{k: v for k, v in evaluate(g, X, preds).items() if k != "late_calibration"}})
    return sorted(live, key=lambda x: x["date"])[-120:]


# ---------- schedule export ----------
def export_schedules(z: Path, st: dict | None, rec: pd.Series | None):
    today = date.today()
    days = gtfs.next_days(HORIZON, today)
    services = gtfs.active_services(z, days)
    running = set().union(*map(set, services.values()))
    routes = gtfs.read(z, "routes.txt")
    trips_ = gtfs.read(z, "trips.txt")
    trips_ = trips_[trips_.service_id.isin(running)]
    trips_["direction_id"] = trips_.direction_id.fillna("0").astype(int)
    stops = gtfs.read(z, "stops.txt", ["stop_id", "stop_name"]).set_index("stop_id").stop_name
    pos = gtfs.stop_positions(z)
    pos = pos[pos.trip_id.isin(trips_.trip_id)].merge(trips_[["trip_id", "route_id", "direction_id", "service_id"]], on="trip_id")
    dirs_txt = gtfs.read(z, "directions.txt")
    dir_names = {} if dirs_txt is None else {(r.route_id, int(r.direction_id)): r.direction for r in dirs_txt.itertuples()}
    rd_stats = st["rd"] if st else pd.DataFrame()
    trip_stats = st["trip"] if st else pd.DataFrame()
    stop_stats = st["stop"].delay_min if st else pd.Series(dtype=float)
    r4 = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)  # noqa: E731

    (OUT / "routes").mkdir(parents=True, exist_ok=True)
    listing = []
    for r in routes.itertuples():
        rp = pos[pos.route_id == r.route_id]
        if rp.empty:
            continue
        fname = "".join(c if c.isalnum() else "_" for c in r.route_id)
        body, dir_list = {"id": r.route_id, "svc": [], "dirs": {}}, []
        svc_index = {}
        for d, dp in rp.groupby("direction_id"):
            key = f"{r.route_id}|{d}"
            # stop order: median position along the trip across all patterns
            order = (dp.stop_idx / np.maximum(dp.n_stops - 1, 1)).groupby(dp.stop_id).median().sort_values()
            col = {s: i for i, s in enumerate(order.index)}
            heads = trips_[(trips_.route_id == r.route_id) & (trips_.direction_id == d)].trip_headsign
            name = dir_names.get((r.route_id, int(d))) or (heads.mode().iat[0] if heads.notna().any() else f"Direction {d}")
            stop_rows = [[s, stops.get(s, s), r4(stop_stats.get(f"{key}|{s}"))] for s in order.index]
            trip_rows = []
            for tid, tp in dp.groupby("trip_id"):
                times = [None] * len(col)
                for s, sec in zip(tp.stop_id, tp.sched_s):
                    times[col[s]] = int(sec // 60)
                svc = tp.service_id.iat[0]
                svc_index.setdefault(svc, len(svc_index))
                start = int(tp.trip_start_s.iat[0] // 60)
                hist = {}
                for dt in ("wk", "sat", "sun"):
                    tk = f"{key}|{dt}|{start}"
                    if tk in trip_stats.index:
                        v = trip_stats.loc[tk]
                        hist[dt] = [r4(v.delay_min), r4(v.late), r4(v.early), int(v.n)]
                trip_rows.append({"id": tid, "s": svc_index[svc], "start": start, "t": times, **({"h": hist} if hist else {})})
            trip_rows.sort(key=lambda t: t["start"])
            body["dirs"][str(d)] = {"name": name, "stops": stop_rows, "trips": trip_rows}
            rs = rd_stats.loc[key] if key in rd_stats.index else None
            dir_list.append({"id": int(d), "name": name, "mean": r4(rs.delay_min) if rs is not None else None,
                             "late": r4(rs.late) if rs is not None else None, "early": r4(rs.early) if rs is not None else None,
                             "recent7": r4(rec.get(key)) if rec is not None else None})
        body["svc"] = list(svc_index)
        (OUT / "routes" / f"{fname}.json").write_text(json.dumps(body, separators=(",", ":")))
        short = r.route_short_name if isinstance(r.route_short_name, str) else ""
        long_ = r.route_long_name if isinstance(r.route_long_name, str) else ""
        listing.append({"id": r.route_id, "file": fname, "short": short, "long": long_, "dirs": dir_list})
    write("routes.json", {"routes": listing})
    write("calendar.json", {"services": services,
                            "holidays": [f"{h:%Y-%m-%d}" for h in F.holidays(days[0], days[-1])]})
    return len(listing)


def main():
    df = F.load_obs(RAW)
    now = datetime.now(F.TZ)
    meta_old = read_json("meta.json") or {}
    days = df.date.nunique() if len(df) else 0
    meta = {"updated": now.isoformat(timespec="minutes"), "days_collected": int(days),
            "observations": int(len(df)), "first_day": f"{df.date.min():%Y-%m-%d}" if len(df) else None,
            "min_days": MIN_DAYS}
    if len(df):
        newest = pd.Timestamp(int(df.ts.max()), unit="s", tz="UTC")
        if pd.Timestamp.now(tz="UTC") - newest > pd.Timedelta(hours=STALE_HOURS):
            write("meta.json", meta_old | {"stale_since": newest.isoformat()})
            raise SystemExit(f"collector hasn't recorded anything since {newest}; check collect.yml")

    z = gtfs.ensure_zip()
    if days < MIN_DAYS:
        n = export_schedules(z, None, None)
        write("meta.json", meta | {"model": False})
        print(f"{days} days of data (need {MIN_DAYS}); exported schedules for {n} routes")
        return

    weather = get_weather(df.date.min() - pd.Timedelta(days=1), pd.Timestamp(now.date()) + pd.Timedelta(days=2))
    write("live.json", live_score(df, weather))

    rec = F.recent7(df)
    # backtest: history tables from training days only, like serving
    split = np.sort(df.date.unique())[-TEST_DAYS]
    tr, te = df[df.date < split], df[df.date >= split]
    rows = train_sample(tr)
    Xtr = F.oof_stats_features(tr, rec, weather, rows=rows)
    Xte = F.build(te, F.stats(tr), rec, weather)
    bt_models = {k: fit(Xtr, tr.loc[Xtr.index, col], p) for k, (col, p, _) in TARGETS.items()}
    bt = evaluate(te, Xte, {k: m.predict(Xte) for k, m in bt_models.items()})
    bt["test_days"] = [f"{pd.Timestamp(split):%Y-%m-%d}", f"{df.date.max():%Y-%m-%d}"]
    imp = pd.Series(bt_models["median"].feature_importance("gain"), index=F.FEATURES)
    bt["importance_median"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()
    print("backtest", {k: v for k, v in bt.items() if not isinstance(v, (list, dict))})
    write("metrics.json", bt)

    # final models on everything
    X = F.oof_stats_features(df, rec, weather, rows=train_sample(df))
    for k, (col, p, kind) in TARGETS.items():
        write(f"model_{k}.json", trees.dump(fit(X, df.loc[X.index, col], p), kind))
    st = F.stats(df)
    last = df.date.max()
    rec_now = rec.xs(last + pd.Timedelta(days=1), level="date") if (last + pd.Timedelta(days=1)) in rec.index.get_level_values("date") else pd.Series(dtype=float)
    n = export_schedules(z, st, rec_now)
    write("meta.json", meta | {"model": True, "trained_through": f"{last:%Y-%m-%d}", "global": st["global"],
                               "features": F.FEATURES})
    print(f"trained on {len(X):,} of {len(df):,} observations over {days} days; exported {n} routes")


if __name__ == "__main__":
    main()
