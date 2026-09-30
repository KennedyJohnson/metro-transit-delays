# Will my bus be late?

[![CI](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/ci.yml/badge.svg)](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/ci.yml)
[![Collect](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/collect.yml/badge.svg)](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/collect.yml)
[![Daily model](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/forecast.yml/badge.svg)](https://github.com/KennedyJohnson/metro-transit-delays/actions/workflows/forecast.yml)

**Live site:** https://kennedyjohnson.github.io/metro-transit-delays/

Pick a Twin Cities Metro Transit route, direction, stop, day and time. For each scheduled departure, the site shows:
- how late it usually runs;
- the chance it's 5+ minutes late;
- the chance it leaves early.

It also names the most and least reliable buses around your time. Save your regular trips and they're one tap away next time. It's a companion to [Gopher X Metro](https://github.com/Gopher-X-Metro/Gopher-X-Metro), which shows where buses are right now. This site is for planning which bus to catch.

## How it works

1. **Collect** ([`collector/collect.py`](collector/collect.py), every 15 min via GitHub Actions). The collector reads Metro Transit's [GTFS-Realtime TripUpdates](https://svc.metrotransit.org/mtgtfs/tripupdates.pb) and keeps each active trip's next-stop delay.
   - It matches each row to the static schedule: scheduled time at that stop, the stop's position in the trip, and the trip's start time.
   - It uses the feed's `delay` field, or predicted minus scheduled time when that's missing. After-midnight (`25:05:00`) times, DST, and missing `start_date` are handled.
   - Output goes to `$RAW_DIR/YYYY-MM-DD.csv` (default `data/raw/`; in Actions, a checkout of the `data` branch).
2. **Train** ([`model/train.py`](model/train.py), daily at about 4 am Central). It trains three LightGBM models on individual observed departures:
   - typical delay (median, L1 loss);
   - P(5+ min late);
   - P(1+ min early).

   The features ([`model/features.py`](model/features.py)) are all known before the day of travel:
   - scheduled hour, day of week, holiday;
   - how far along the trip the stop is;
   - smoothed history of the route/direction, of that exact scheduled trip, and of that stop;
   - the route's past week;
   - hourly Open-Meteo weather.

   History tables in training are computed out-of-fold by day, so a departure never sees its own outcome.
3. **Serve** in the browser. Models are exported as JSON trees ([`docs/model.js`](docs/model.js)) and features are rebuilt by [`docs/features.js`](docs/features.js). Per-route schedule files carry each trip's and stop's history, and the day's weather comes from Open-Meteo. A test checks that the browser's features and predictions match Python exactly.
4. **Evaluate:**
   - **Backtest:** the last 7 days are held out against baselines: assume on time, the route's average, and the trip's own average (`docs/data/metrics.json`).
   - **Live:** each morning, before retraining, the model the site was serving is scored on the departures it predicted (`docs/data/live.json`).

Schedules are published from day one. Predictions start once 10 days of data exist.

## Repository setup

GitHub Pages serves `docs/` from `main` (Settings → Pages → Deploy from a branch → `main` / `/docs`). The workflows need no secrets:

| Workflow | When | What |
|---|---|---|
| `collect.yml` | continuous | polls the live feed every 15 min for ~5.5 h per run, then starts the next run itself (6-hourly cron restarts the chain if it breaks) → the `data` branch |
| `forecast.yml` | daily 09:15 UTC | compacts finished days to Parquet and drops raw days older than 3 years (`collector/trim.py`), then squashes the `data` branch to one commit; tests, live score, retrain on up to 5M sampled rows, export `docs/data/`; opens a `stale-data` issue on failure |
| `ci.yml` | pushes / PRs | pytest, incl. the Python↔JS parity test |

Raw snapshots (`YYYY-MM-DD.csv[.gz]`, one per day) live on the `data` branch, not `main`: the collector commits there every 15 minutes, and the daily job compacts each finished day to Parquet (only the rows the model uses, plus cancellations; ~100-250 KB/day, a third of gzip) and replaces that branch's history with a single commit of the last 3 years (600 MB cap), so the repo doesn't grow without bound. Locally, put them in `data/raw/` (or set `RAW_DIR`).

Data files the site reads (`docs/data/`): `meta.json` (status), `routes.json`, `routes/<route>.json` (stops, trips, history), `calendar.json` (services for the next 14 days, holidays), `model_{median,late,early}.json`, `metrics.json` (backtest), `live.json` (daily scores). [Gopher X Metro](https://github.com/Gopher-X-Metro/Gopher-X-Metro) reads the same files to flag departures likely to run late.

## Run locally

```bash
pip install -r requirements.txt
python collector/collect.py          # one snapshot (downloads the static GTFS on first run)
python model/train.py                # schedules always; models once data/raw has >= 10 days
python -m pytest -q tests            # synthetic GTFS + observations, no network; needs node for the parity test
python -m http.server -d docs 8000
```

## Limitations

- Delay is measured at a trip's *next* stop when polled, so each trip is sampled at a few stops per run, not every stop.
- GitHub may delay or skip 15-minute scheduled runs.
- Arrival delay stands in for departure delay.
- Predictions are for planning. They can't see crashes, detours or breakdowns on the day.

Data: Metro Transit (not affiliated). Weather: Open-Meteo (CC BY 4.0).

MIT License.
