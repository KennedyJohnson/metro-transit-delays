# metro-transit-delays

Static GitHub Pages site (`docs/`, served from main): pick route/direction/stop/day/time → per-departure typical delay, P(5+ min late), P(1+ min early), most/least reliable bus, saved trips (localStorage). Python pipeline, no backend. Companion to Gopher X Metro (separate org repo; this repo never touches it).

## Pieces
- `common/gtfs.py` - static GTFS: download/cache (`data/gtfs/`, gitignored), `stop_positions()` (sched secs, stop_idx, n_stops, trip_start_s per trip/stop_sequence), `active_services()`.
- `collector/collect.py` - GTFS-RT TripUpdates → next-stop delay per trip, joined to schedule → appends `$RAW_DIR/YYYY-MM-DD.csv` (default data/raw; the `data` branch in Actions) (FIELDS). Missing start_date → today/yesterday by schedule fit.
- `model/features.py` - `load_obs` (dedupe trip+stop keep last), smoothed stats (K=20: trip/stop shrink to route-dir, route-dir to global), `recent7`, `build()` → FEATURES. **`docs/features.js featureRow()` mirrors `build()`; change both.**
- `model/trees.py` / `docs/model.js` - JSON tree dump + evaluators (Python vectorized port == JS).
- `model/train.py` - live-score published model (rebuilds stats from `docs/data/routes*.json`), backtest last 7 days, fit 3 models (300 rounds), export `docs/data/{meta,routes,calendar,metrics,live}.json`, `routes/<route>.json` (stops [id,name,stop_mean], trips {id: GTFS trip_id (Gopher X Metro matches NexTrip trip_id on it - keep), s: svc idx, start, t: minutes per stop column, h: {wk|sat|sun: [mean,late,early,n]}}), `model_{median,late,early}.json`. Schedules exported even before MIN_DAYS=10. Fails if newest snapshot > STALE_HOURS=36.
- `tests/` - synthetic GTFS (`synth.py`), end-to-end train/live, Python↔JS parity via `tests/parity.js` (needs node).

## Automation
- `collect.yml` self-chaining: each run polls every 15 min for ~5.5 h (commit+push per poll), then `gh workflow run collect.yml`; 6-hourly cron is only a backup; each poll also starts forecast.yml once per UTC day after 09:00 if it hasn't run (cron hadn't been firing for this repo), `forecast.yml` daily 09:15 UTC (`collector/trim.py`: gzip finished days + delete raw days older than RETAIN_DAYS=365; pytest, train.py, commit; failure → `stale-data` issue), `ci.yml` pytest. Collect and forecast use separate concurrency groups (a 5.5 h collect run must not block training); pushes use pull --rebase retries.
- Size limits: raw data keeps 365 days (`collector/trim.py`); models fit on at most `MAX_TRAIN_ROWS` (2M) rows sampled across the window (`train.train_sample`), while history stats use every row. Raw snapshots live on the `data` branch (workflows check it out to `store/`, `RAW_DIR` points there); collector polls fetch+hard-reset to origin/data, append, push; forecast.yml's last step trims then force-pushes (with lease) an orphan single commit to `data`, so its history never grows. `main` has no raw data.
