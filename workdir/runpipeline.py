#!/usr/bin/env python3
"""
Single entry point for the DE-LU day-ahead price forecast pipeline. Run any
combination of stages via flags, instead of calling each script by hand in order.

Stages (each depends on the ones before it having been run at some point:
    1. history   -- merge raw CSVs + feature engineering -> output/history_features.pkl
    2. future    -- build the 2026-09 -> 2027-12 fundamentals scenario -> output/future_features.pkl
    3. forecast  -- train the final model on history, predict on future -> output/forecast.csv
    4. backtest  -- 6-fold expanding-window backtest against history_features.pkl (read-only,
                    does not touch forecast.csv)
"""
import argparse
import sys
import time


def run_history():
    print('=== [history] rebuilding output/history_features.pkl ===')
    t0 = time.time()
    from load_data import save_features
    save_features()
    print(f'    done in {time.time()-t0:.1f}s\n')


def run_future():
    print('=== [future] rebuilding output/future_features.pkl ===')
    t0 = time.time()
    from build_future import build_future_df
    fut = build_future_df()
    fut.to_pickle('../output/future_features.pkl')
    print(f'    wrote output/future_features.pkl  shape={fut.shape}')
    print(f'    done in {time.time()-t0:.1f}s\n')


def run_forecast():
    print('=== [forecast] training final model + writing output/forecast.csv ===')
    t0 = time.time()
    import generate_forecast
    generate_forecast.main()
    print(f'    done in {time.time()-t0:.1f}s\n')


def run_backtest():
    print('=== [backtest] 6-fold expanding-window backtest ===')
    t0 = time.time()
    import pandas as pd
    from backtest import run_backtest 
    df = pd.read_pickle('../output/history_features.pkl')
    res = run_backtest(df)
    pd.set_option('display.width', 160)
    print(res.round(2).to_string(index=False))
    print('\nmeans:')
    print(res[['mae', 'rmse', 'bias', 'coverage']].mean().round(3))
    print(f'\n    done in {time.time()-t0:.1f}s\n')
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--history', action='store_true', help='rebuild output/history_features.pkl')
    parser.add_argument('--future', action='store_true', help='rebuild output/future_features.pkl')
    parser.add_argument('--forecast', action='store_true', help='train model + write output/forecast.csv')
    parser.add_argument('--backtest', action='store_true', help='run the 6-fold backtest and print results')
    parser.add_argument('--all', action='store_true', help='run every stage, in order (history -> future -> forecast -> backtest)')
    args = parser.parse_args()

    if args.all:
        args.history = args.future = args.forecast = args.backtest = True

    if not any([args.history, args.future, args.forecast, args.backtest]):
        parser.print_help()
        print('\nNo stage selected -- nothing to do. Pick at least one flag, or use --all.')
        sys.exit(0)

    t_start = time.time()
    if args.history:
        run_history()
    if args.future:
        run_future()
    if args.forecast:
        run_forecast()
    if args.backtest:
        run_backtest()

    print(f'=== all selected stages finished in {time.time()-t_start:.1f}s total ===')


if __name__ == '__main__':
    main()