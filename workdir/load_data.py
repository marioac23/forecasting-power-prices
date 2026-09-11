'''
File that loads all the historic data and builds it into a single dataframe
'''
import numpy as np
import pandas as pd
input = '../data'

def load_data():
    price = pd.read_csv(f'{input}/de_lu_day_ahead_price_hourly.csv', parse_dates=['ts_utc'])
    load  = pd.read_csv(f'{input}/de_lu_load_hourly.csv', parse_dates=['ts_utc'])
    gen   = pd.read_csv(f'{input}/de_lu_generation_by_type_hourly.csv', parse_dates=['ts_utc'])
    ws    = pd.read_csv(f'{input}/de_lu_wind_solar_da_forecast_hourly.csv', parse_dates=['ts_utc'])
    cap   = pd.read_csv(f'{input}/de_lu_installed_capacity_yearly.csv')
    fuel  = pd.read_csv(f'{input}/fuel_carbon_prices_daily.csv', parse_dates=['date'])
    return price, load, gen, ws, cap, fuel

def build_history():
    price, load, gen, ws, cap, fuel = load_data()
    # Merge in a single df
    df = price.merge(load, on=['ts_utc', 'ts_local'], how='left')
    df = df.merge(ws, on=['ts_utc', 'ts_local'], how='left')
    df = df.merge(gen.drop(columns=['ts_local']), on='ts_utc', how='left')
    # Make sure they all end in the same day (price goes until 8th of august while load, ws and gen go until 7th of august)
    common_end = min(load.ts_utc.max(), ws.ts_utc.max(), gen.ts_utc.max())
    dropped = (df.ts_utc > common_end).sum()
    # Drop rows corresponding to missing day
    df = df[df.ts_utc <= common_end].reset_index(drop=True)

    # Interpolate small NaN patches with limit of 6h
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(limit=6, limit_direction='both')

    # convert fuel dates to hourly
    fuel = fuel.set_index('date').sort_index()
    fuel_dates = pd.date_range(df.ts_utc.min().tz_localize(None).floor('D'),
                               df.ts_utc.max().tz_localize(None).floor('D'), freq='D')
    fuel_hour = fuel.reindex(fuel_dates)
    first_valid = fuel_hour.dropna().index.min()
    n_leading = (fuel_hour.index < first_valid).sum()
    if n_leading:
        print(f'load_data.build_history: {n_leading} leading day(s) before the first fuel/carbon '
        f'settlement ({first_valid.date()}) -- backward-filled from the next trading day')

    # back/forward fill fuel values
    fuel_hour = fuel_hour.ffill().bfill()
    df['date'] = df.ts_utc.dt.tz_localize(None).dt.floor('D')
    df = df.merge(fuel_hour.reset_index().rename(columns={'index':'date'}), on='date', how='left')
    df = df.drop(columns=['date'])

    df = df.sort_values('ts_utc').reset_index(drop=True)

    return df, cap

# save a feats file
def save_features():
    from features import calendar_feat, add_feats
    df, cap = build_history()
    df = calendar_feat(df)
    df = add_feats(df)
    df.to_pickle('../output/history_features.pkl')
    print(f'wrote ../output/history_features.pkl  shape={df.shape}  '
          f'range={df.ts_utc.min()} -> {df.ts_utc.max()}')
    return df, cap


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    if '--save-features' in sys.argv:
        save_features()
    else:
        df, cap = build_history()
        print(df.shape)
        print(df.isna().sum()[df.isna().sum() > 0])
        print(df.tail(3).T)
        print("\n(run with --save-features to also write output/history_features.pkl)")
