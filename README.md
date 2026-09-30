# How late will my bus be today?

**Live site:** https://kennedyjohnson.github.io/metro-transit-delays/

An hour-by-hour forecast of how late each Twin Cities Metro Transit route will run today. A model trained on the history of Metro Transit's own live feed makes the forecast each morning, and every forecast is later scored against what actually happened. It's a companion to [Gopher X Metro](https://github.com/Gopher-X-Metro/Gopher-X-Metro), which shows where the buses are right now.

## How it works

1. **Collect** ([`collector/collect.py`](collector/collect.py), every 15 min via GitHub Actions). The collector reads the [GTFS-Realtime TripUpdates](https://svc.metrotransit.org/mtgtfs/tripupdates.pb) feed and records each active trip's lateness at its next stop.
   - Lateness comes from the feed's `delay` field. If that's missing, the collector computes predicted minus scheduled arrival from the static GTFS, which handles after-midnight `25:05:00` times and DST.
   - Cancelled trips are recorded separately.
   - Output goes to `data/raw/YYYY-MM-DD.csv`. The daily job gzips finished days.
2. **Forecast** ([`model/forecast.py`](model/forecast.py), daily at about 4 am Central).
   - **Target:** average lateness per route per hour.
   - **Features:** the model uses only what's known at forecast time:
     - route, hour, day of week, federal holidays;
     - that route-hour's lateness yesterday and a week ago, and its trailing 14-day mean;
     - the route's mean yesterday;
     - hourly [Open-Meteo](https://open-meteo.com/) weather: archive for training, forecast for today.
   - **Model:** LightGBM with L1 loss.
3. **Evaluate twice:**
   - **Backtest:** the last 7 days are held out and compared against two baselines, same hour last week and the trailing 14-day mean.
   - **Live:** each day's forecast is saved to `data/forecasts/` and scored the next day against the actuals (`docs/data/live.json`). The published accuracy therefore includes forecasts made before the outcome was known.

Forecasting starts once 10 days of data exist. Until then the site shows collection progress.

## Run locally

```bash
pip install -r requirements.txt
python collector/collect.py          # one snapshot
python model/forecast.py             # needs >= 10 days in data/raw
python -m pytest -q tests            # synthetic-data tests, no network
python -m http.server -d docs 8000
```

## Limitations

- Lateness is measured at each trip's *next* stop at poll time. A trip that makes up time later isn't credited.
- GitHub may delay or skip 15-minute scheduled runs, so some hours are sampled more than others.
- Hours with fewer than 5 observations for a route are dropped.
- Lateness can't be measured for trips the feed doesn't track.

Data: Metro Transit (not affiliated). Weather: Open-Meteo (CC BY 4.0).
