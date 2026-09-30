# metro-transit-delays

Static GitHub Pages site (`docs/`, served from main) with a day-ahead, hourly forecast of Metro Transit lateness per route. Python, no backend. Companion to Gopher X Metro (separate org repo).

## Pieces
- `collector/collect.py` - GTFS-RT TripUpdates → next-stop delay per active trip → appends `data/raw/YYYY-MM-DD.csv` (local date of poll). Uses feed `delay`, else predicted time - scheduled (static GTFS `stop_times.txt`, cached in gitignored `data/gtfs/`). Canceled trips kept with `source=canceled`.
- `model/forecast.py` - aggregates to route x service day x hour (`MIN_OBS`=5), features from prior days only + Open-Meteo weather, LightGBM L1. Backtest last `TEST_DAYS`=7 vs last-week and 14-day baselines → `docs/data/metrics.json`; forecast today → `docs/data/forecast.json` + `data/forecasts/<date>.csv`; scores past forecast files → `docs/data/live.json`; `status.json` (collecting until `MIN_DAYS`=10). Exits non-zero if newest snapshot >36 h old.
- `docs/` - `index.html`, `app.js` (vanilla JS, SVG chart), `style.css` (light/dark tokens).
- `tests/` - synthetic feed/schedule and synthetic 24-day world; no network.

## Automation
- `collect.yml` - every 15 min, commits `data/raw` (pull --rebase retry). Shares concurrency group `data` with the forecast job.
- `forecast.yml` - daily 09:15 UTC: gzips finished raw days, runs tests + forecast, commits; failure → `stale-data` issue.
- `ci.yml` - pytest on PRs/pushes.
