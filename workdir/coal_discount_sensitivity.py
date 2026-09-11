"""
Sensitivity analysis: how much does the final forecast actually change if the
(unsourced) COAL_GAS_DISCOUNT assumption in features.py is varied?

Since coal has no price series in the provided data, its cost is proxied as a fixed
fraction of the gas price (COAL_GAS_DISCOUNT). Real market data shows this ratio is
NOT stable (e.g. it moved from ~42% to ~23% within a year in real coal/gas markets) --
so rather than pretend one fixed value is correct, this script tests a defensible range
of the assumption directly through the pipeline and reports how much forecast.csv
actually moves, instead of guessing.

Runs, for each discount value in SCENARIOS:
    1. python load_data.py --build-features   (rebuild history with the new discount)
    2. python build_future.py                 (rebuild the future scenario)
    3. python generate_forecast.py             (retrain + regenerate forecast.csv)
then saves each run's forecast.csv separately and prints a comparison table.

IMPORTANT: this overwrites src/features.py, output/history_features.pkl,
output/future_features.pkl, and output/forecast.csv while running -- it backs up
whatever was already there first and restores it automatically at the end, even if
something fails partway through.

Usage (run from the project root, i.e. the folder containing src/ and output/):
    python coal_discount_sensitivity.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import HybridQuantileModel

SCENARIOS = [0.25, 0.40, 0.5, 0.6, 0.7, 0.8]   # coal-gas discount values to test
FEATURES_PATH = Path('features.py')
BACKUP_DIR = Path('../temp/coal_sensitivity_backup')
SCENARIO_OUT_DIR = Path('scenario_forecasts')
HISTORY_PRICE_CSV = Path('../data/de_lu_day_ahead_price_hourly.csv')


def backup_current_state():
    BACKUP_DIR.mkdir(exist_ok=True)
    shutil.copy(FEATURES_PATH, BACKUP_DIR / 'features.py')
    for f in ['history_features.pkl', 'future_features.pkl', 'forecast.csv']:
        p = Path('../output') / f
        if p.exists():
            shutil.copy(p, BACKUP_DIR / f)


def restore_original_state():
    shutil.copy(BACKUP_DIR / 'features.py', FEATURES_PATH)
    for f in ['history_features.pkl', 'future_features.pkl', 'forecast.csv']:
        src = BACKUP_DIR / f
        if src.exists():
            shutil.copy(src, Path('../output') / f)
    print('Restored original features.py and output/ files.')


def set_coal_gas_discount(value):
    text = FEATURES_PATH.read_text()
    new_text = re.sub(r'COAL_GAS_DISCOUNT = [0-9.]+', f'COAL_GAS_DISCOUNT = {value}', text)
    if new_text == text:
        raise RuntimeError('COAL_GAS_DISCOUNT line not found -- check features.py has not changed shape')
    FEATURES_PATH.write_text(new_text)


def run_step(args, log_name):
    result = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    Path('../temp', log_name).write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'{" ".join(args)} failed -- see /tmp/{log_name}')

def run_backtest_for_scenario():
    """6-fold expanding-window backtest, using the SAME model class generate_forecast.py
    actually deploys (RecencyWeightedHybrid) -- run fresh against whatever history_features.pkl
    currently exists on disk (i.e. built with whichever discount is active right now)."""
    sys.path.insert(0, 'src')
    import importlib
    import model as model_module
    import features as features_module
    importlib.reload(features_module)
    importlib.reload(model_module)
    from model import HybridQuantileModel
    from features import FEATURE_COLS
    import numpy as np
 
    df = pd.read_pickle('../output/history_features.pkl').sort_values('ts_utc').reset_index(drop=True)
    fold_starts = pd.date_range('2025-09-01', '2026-07-01', freq='2MS', tz='UTC')
    rows = []
    for start in fold_starts:
        end = start + pd.DateOffset(months=2)
        train = df[df.ts_utc < start]
        test = df[(df.ts_utc >= start) & (df.ts_utc < end)]
        if len(test) == 0 or len(train) < 2000:
            continue
        m = HybridQuantileModel(FEATURE_COLS).fit(train[FEATURE_COLS], train['price_eur_mwh'])
        preds = m.predict(test[FEATURE_COLS])
        p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
        y = test['price_eur_mwh'].values
        rows.append(dict(
            mae=np.mean(np.abs(y - p50)),
            rmse=np.sqrt(np.mean((y - p50) ** 2)),
            bias=np.mean(p50 - y),
            coverage=np.mean((y >= p10) & (y <= p90)),
        ))
    return pd.DataFrame(rows)

def run_scenario(discount):
    pct = int(round(discount * 100))
    print(f'--- scenario: COAL_GAS_DISCOUNT = {discount} ({pct}%) ---')
    set_coal_gas_discount(discount)
    run_step(['load_data.py', '--save-features'], f'scenario_{pct}_hist.log')
    run_step(['build_future.py'], f'scenario_{pct}_future.log')
    run_step(['generate_forecast.py'], f'scenario_{pct}_forecast.log')
    SCENARIO_OUT_DIR.mkdir(exist_ok=True)
    shutil.copy('../output/forecast.csv', SCENARIO_OUT_DIR / f'forecast_{pct}pct.csv')
    backtest_res = run_backtest_for_scenario()
    backtest_res.to_csv(SCENARIO_OUT_DIR / f'backtest_{pct}pct.csv', index=False)
    print('  done (forecast + backtest)')
    return backtest_res.mean()


def compare_scenarios():
    scenarios = {}
    for discount in SCENARIOS:
        pct = int(round(discount * 100))
        df = pd.read_csv(SCENARIO_OUT_DIR / f'forecast_{pct}pct.csv')
        df['ts_local'] = pd.to_datetime(df['ts_local'])
        df['ym'] = df.ts_local.dt.to_period('M')
        scenarios[pct] = df

    monthly = pd.DataFrame({pct: s.groupby('ym')['price_eur_mwh'].mean() for pct, s in scenarios.items()})
    pd.set_option('display.width', 120)
    print('\nMonthly average forecast price (EUR/MWh) by scenario:')
    print(monthly.round(1))

    print('\nOverall summary:')
    for pct, d in scenarios.items():
        print(f'{pct}% discount: mean={d.price_eur_mwh.mean():.1f}  median={d.price_eur_mwh.median():.1f}  '
              f'neg_price_share={100*(d.price_eur_mwh<0).mean():.2f}%  '
              f'avg_band_width={(d.p90_eur_mwh - d.p10_eur_mwh).mean():.1f}')

    return monthly


def plot_scenarios(monthly, out_path='../temp/coal_discount_sensitivity.png'):
    fig, ax = plt.subplots(figsize=(11, 5))
    for pct in monthly.columns:
        ax.plot(range(len(monthly)), monthly[pct], marker='o', ms=4,
                label=f'{pct}% discount' + (' (original)' if pct == 55 else ''))
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('EUR/MWh')
    ax.set_title('Monthly forecast price under different coal-gas discount assumptions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'\nPlot written to {out_path}')

def plot_vs_history_by_calendar_month(out_path='../temp/coal_discount_vs_history.png'):
    """For each scenario, average the forecast by CALENDAR month (1-12, pooling e.g. both
    2026-09 and 2027-09 into one 'September' point) and plot alongside the real historical
    average for that same calendar month -- shows whether each discount assumption still
    tracks the real seasonal shape, not just the raw forecast level."""
    if not HISTORY_PRICE_CSV.exists():
        print(f'\nSkipping vs-history plot: {HISTORY_PRICE_CSV} not found '
              f'(edit HISTORY_PRICE_CSV at the top of this script to point at your data folder)')
        return
 
    hist = pd.read_csv(HISTORY_PRICE_CSV, parse_dates=['ts_utc'])
    hist['ts_local'] = pd.to_datetime(hist['ts_local'])
    hist_by_month = hist.groupby(hist.ts_local.dt.month)['price_eur_mwh'].mean()
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 13), hist_by_month.reindex(range(1, 13)), color='black', lw=2.5,
            marker='o', ms=5, label='history (actual)', zorder=10)
 
    for discount in SCENARIOS:
        pct = int(round(discount * 100))
        df = pd.read_csv(SCENARIO_OUT_DIR / f'forecast_{pct}pct.csv')
        df['ts_local'] = pd.to_datetime(df['ts_local'])
        by_month = df.groupby(df.ts_local.dt.month)['price_eur_mwh'].mean()
        ax.plot(range(1, 13), by_month.reindex(range(1, 13)), marker='o', ms=4,
                label=f'{pct}% discount' + (' (original)' if pct == 55 else ''))
 
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('calendar month')
    ax.set_ylabel('EUR/MWh')
    ax.set_title('Forecast (by discount scenario) vs. real history, by calendar month')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Plot written to {out_path}')

if __name__ == '__main__':
    backup_current_state()
    try:
        backtest_summaries = {}
        for discount in SCENARIOS:
            pct = int(round(discount * 100))
            backtest_summaries[pct] = run_scenario(discount)
 
        bt = pd.DataFrame(backtest_summaries).T
        bt.index.name = 'discount_pct'
        pd.set_option('display.width', 120)
        print('\nBacktest results by scenario (mean across all 6 folds):')
        print(bt.round(3))
 
        monthly = compare_scenarios()
        plot_scenarios(monthly)
        plot_vs_history_by_calendar_month()
    finally:
        restore_original_state()