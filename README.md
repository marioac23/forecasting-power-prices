# DE-LU Day-Ahead Price Forecast

Hourly forecast of the DE-LU (Germany-Luxembourg) day-ahead electricity price for
every hour from **1 Sep 2026 00:00 to 31 Dec 2027 23:00**, built entirely from the
six provided historical CSVs (1 Jan 2025 - 7 Aug 2026) (no forward curves, weather
data, or other external sources). Every assumption about how 2026-2027 fundamentals
(gas, EUA, load, renewables, capacity) evolve is written down and justified rather
than pulled from a live market source.

**Method**: a two-stage hybrid quantile model - a linear quantile
regression on the core fundamentals (residual load, gas/coal short-run marginal
cost, renewable share) that extrapolates safely into 2027's higher fuel-cost range,
plus a gradient-boosted model on the leftover residual that captures nonlinear
seasonal/hourly shape. See `report.pdf`
for the full method, assumptions, backtest results, and limitations.

## Repository structure

```
├── run.sh              <- run this
├── data/                <- the 6 provided input CSVs 
│   ├── de_lu_day_ahead_price_hourly.csv
│   ├── de_lu_load_hourly.csv
│   ├── de_lu_generation_by_type_hourly.csv
│   ├── de_lu_wind_solar_da_forecast_hourly.csv
│   ├── de_lu_installed_capacity_yearly.csv
│   └── fuel_carbon_prices_daily.csv
├── plots/                <- plots of the forecast output
├── output/               <- everything the pipeline writes lands here
│   ├── history_features.pkl
│   ├── future_features.pkl
│   └── forecast.csv      <- the deliverable
└── workdir/              <- all the code
    ├── runpipeline.py   <- orchestrator (run.sh calls this)
    ├── load_data.py      <- merge raw CSVs onto one hourly grid + patch small gaps
    ├── features.py       <- calendar features, residual load, gas/coal SRMC
    ├── project_feats.py  <- climatology + flat-forward projections for 2026-2027
    ├── build_future.py   <- assembles the Sep 2026 - Dec 2027 feature table
    ├── model.py           <- HybridQuantileModel
    ├── generate_forecast.py    <- trains the final model, writes forecast.csv
    ├── coal_discount_sensitivity.py <- checks the impact of different coal/gas price ratios
    ├── forecast_plots.py  <- plotting script for forecast 
    └── backtest.py       <- 6-fold expanding-window backtest

```

## Requirements

Python 3.10+, with:

```
pandas
numpy
scikit-learn
```

No other dependencies — the model is plain scikit-learn (`QuantileRegressor` +
`HistGradientBoostingRegressor`), no external ML frameworks.

## Quick start

From directory:

```bash
chmod +x run.sh        # once, the first time
./run.sh all           # build history, future, forecast, and run the backtest
```

`output/forecast.csv` is the deliverable when this finishes.

### Running individual stages

```bash
./run.sh history              # rebuild output/history_features.pkl only
./run.sh future                # rebuild output/future_features.pkl only
./run.sh forecast              # train the model + write output/forecast.csv
./run.sh backtest               # run the 6-fold backtest, print metrics
./run.sh history backtest      # e.g. after editing a constant in features.py:
                                # rebuild history, then just check the backtest impact
```

Stages run in the order you list them if you combine several. `history` must run
before `future`/`forecast`/`backtest` for the output to reflect any change you've
just made in `features.py` — the pipeline does not check this for you, so if you
change a constant like a heat rate or the coal-gas discount, rebuild history first.

## Output format

`output/forecast.csv` — one row per hour, no gaps or duplicates:

| column | meaning |
|---|---|
| `ts_utc` | timestamp, UTC, ISO-ish format (`2026-08-31T22:00Z`) |
| `ts_local` | timestamp, Europe/Berlin local time |
| `price_eur_mwh` | median (p50) forecast price, EUR/MWh |
| `p10_eur_mwh` | 10th-percentile forecast price |
| `p90_eur_mwh` | 90th-percentile forecast price |

Full detail, sourcing, and numbers for every point above are in `report.pdf`.
