'''
Runs backtest
'''
import pandas as pd
import numpy as np
from model import HybridQuantileModel
from features import FEATURE_COLS

# define loss
def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q*error, (q-1)*error))

def run_backtest(df, feat_cols=FEATURE_COLS, target='price_eur_mwh'):
    # sort by date
    df = df.sort_values('ts_utc').reset_index(drop=True)
    # generate 6 dates, 2 months apart from Sep 2025 to Jul 2026 for testing
    fold_starts = pd.date_range('2025-09-01', '2026-07-01', freq='2MS', tz='UTC')
    results = []
    for start in fold_starts:
        end = start + pd.DateOffset(months=2)
        train   = df[df.ts_utc < start]
        test    = df[(df.ts_utc >=start) & (df.ts_utc < end)]
        if len(test) == 0 or len(train) < 2000:
            continue

        # get model and train it
        model = HybridQuantileModel(feat_cols).fit(train, train[target])
        # make prediction
        preds = model.predict(test)
        # get real values
        y_real = test[target].values
        # get quantile predictions
        p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
        # compute metrics
        mae = np.mean(np.abs(y_real - p50))
        rmse = np.sqrt(np.mean((y_real-p50)**2))
        cov = np.mean((y_real >= p10) & (y_real <= p90))
        bias = p50.mean() - y_real.mean()
        # compute montly baseload MAE
        test2 = test.copy()
        test2['pred'] = p50
        test2['ym'] = test2.ts_utc.dt.tz_convert(None).dt.to_period('M')
        monthly = test2.groupby('ym').apply(lambda d: abs(d[target].mean() - d.pred.mean()))

        # append results
        results.append(dict(fold_start=start.date(), n=len(test), mae=mae, rmse=rmse,
                            bias=bias, coverage=cov, monthly_baseload_mae=monthly.mean(),
                            pinbal_p10=pinball_loss(y_real,p10,0.1), pinball_p50=pinball_loss(y_real,p50,0.5), pinball_p90=pinball_loss(y_real,p90,0.9)))
        
    return pd.DataFrame(results)

def run_backtest_recency(df, feat_cols=FEATURE_COLS, target='price_eur_mwh'):
    # sort by date
    df = df.sort_values('ts_utc').reset_index(drop=True)
    # generate 6 dates, 2 months apart from Sep 2025 to Jul 2026 for testing
    fold_starts = pd.date_range('2025-09-01', '2026-07-01', freq='2MS', tz='UTC')
    results = []
    for start in fold_starts:
        end = start + pd.DateOffset(months=2)
        train   = df[df.ts_utc < start]
        test    = df[(df.ts_utc >=start) & (df.ts_utc < end)]
        if len(test) == 0 or len(train) < 2000:
            continue

        # get model and train it
        print("Training...")
        model = HybridQuantileModel(feat_cols).fit(train, train[target])
        # make prediction
        print("Training done! Predicting prices...")
        preds = model.predict(test)
        print("Prices predicted!")
        # get real values
        y_real = test[target].values
        # get quantile predictions
        p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
        # compute metrics
        mae = np.mean(np.abs(y_real - p50))
        rmse = np.sqrt(np.mean((y_real-p50)**2))
        cov = np.mean((y_real >= p10) & (y_real <= p90))
        bias = p50.mean() - y_real.mean()
        # compute montly baseload MAE
        test2 = test.copy()
        test2['pred'] = p50
        test2['ym'] = test2.ts_utc.dt.tz_convert(None).dt.to_period('M')
        monthly = test2.groupby('ym').apply(lambda d: abs(d[target].mean() - d.pred.mean()))

        # append results
        results.append(dict(fold_start=start.date(), n=len(test), mae=mae, rmse=rmse,
                            bias=bias, coverage=cov, monthly_baseload_mae=monthly.mean(),
                            pinbal_p10=pinball_loss(y_real,p10,0.1), pinball_p50=pinball_loss(y_real,p50,0.5), pinball_p90=pinball_loss(y_real,p90,0.9)))
        
  
    return pd.DataFrame(results) 
if __name__ == '__main__':
    df = pd.read_pickle('../output/history_features.pkl')
    print("Running model...")
    res = run_backtest_recency(df)
    print("Model run.")
    res.to_csv('../output/backtest.csv', index = False)
    pd.set_option('display.width', 160)
    print(res)
    print('\nmeans:\n', res[['mae','rmse','bias','coverage','monthly_baseload_mae']].mean())