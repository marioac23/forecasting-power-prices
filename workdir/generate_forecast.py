'''
File that trains the model and predicts future prices
'''
import numpy as np
import pandas as pd
from model import HybridQuantileModel
from features import FEATURE_COLS

# since we are creating features for predicting the price, we need to ad uncertainties
def future_uncertainty(ts_utc, end, max_extra=0.9, scale_days=545):
    days_ahed = (ts_utc - end).dt.total_seconds() / 86400.0
    days_ahead = np.clip(days_ahed, 0, scale_days)
    return 1.0 + max_extra * (days_ahead/scale_days)

def main():
    history = pd.read_pickle('../output/history_features.pkl')
    fut = pd.read_pickle('../output/future_features.pkl')
    end = history.ts_utc.max()
    # Train model and predict
    model = HybridQuantileModel(FEATURE_COLS).fit(history[FEATURE_COLS], history['price_eur_mwh'])
    preds = model.predict(fut[FEATURE_COLS])
    p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
    # Add uncertainties
    factor = future_uncertainty(fut['ts_utc'], end)
    p10_w = p50 - (p50 - p10)*factor
    p90_w = p50 + (p90 - p50)*factor
    # Output
    out = pd.DataFrame({'ts_utc': fut['ts_utc'].dt.strftime('%Y-%m-%dT%H:%MZ'),
        'ts_local': fut['ts_local'],
        'price_eur_mwh': np.round(p50, 2),
        'p10_eur_mwh': np.round(p10_w, 2),
        'p90_eur_mwh': np.round(p90_w, 2),
        })
    # Checks
    assert out['ts_utc'].is_unique
    assert len(out) == len(pd.date_range(fut['ts_utc'].min(), fut['ts_utc'].max(), freq='h'))
    out.to_csv('../output/forecast.csv', index=False)
    print(out.shape)
    print(out.head())
    print(out.tail())

    # quick structural checks
    print('\n--- structural checks ---')
    neg_share = (out.price_eur_mwh < 0).mean()
    print(f'negative price hour share: {neg_share*100:.2f}% (history: {(history.price_eur_mwh<0).mean()*100:.2f}%)')
    fut2 = fut.copy()
    fut2['pred'] = p50
    fut2['ym'] = pd.to_datetime(fut2.ts_local).dt.to_period('M')
    print(fut2.groupby('ym')['pred'].mean())

if __name__ == '__main__':
    main()