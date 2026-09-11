'''
File that projects the fundamental features (load, wind, solar, gas,..) up to 31 Dec 2027

'''
import numpy as np
import pandas as pd

# Use the growth of capicity in previous 2 years to extrapolate for 2027
def future_capacity(cap):
    cap = cap.set_index('technology')
    growth = cap['installed_mw_2026']/cap['installed_mw_2025']
    cap['installed_mw_2027'] = cap['installed_mw_2026'] * growth
    return cap

# project solar and wind
def project_cap_factor_series(df, cap_proj, target_index, tech, gen_col, cap2025, cap2026):
    df_copy = df.copy()
    # divide output by that year's cap: 2025 rows divide by 2025 cap, ...
    # capfactor: what faction of installed capacity was actually generating
    year_cap = np.where(df_copy.year == 2025, cap2025, cap2026)
    df_copy['capfactor'] = df_copy[gen_col] / year_cap
    # average capfactor by month/hour/weekend
    prof = df_copy.groupby(['month', 'hour', 'is_weekend'])['capfactor'].mean()
    # apply typical cap factor to the project capacity from future_cap
    out = pd.DataFrame(index=target_index)
    out['month'] = target_index.month
    out['hour'] = target_index.hour
    out['is_weekend'] = (target_index.dayofweek >= 5).astype(int)
    capfactor = out.set_index(['month', 'hour', 'is_weekend']).index.map(prof)
    out['capfactor'] = capfactor.values
    out['year'] = target_index.year
    cap_by_year = {2026: cap_proj.loc[tech, 'installed_mw_2026'], 2027: cap_proj.loc[tech, 'installed_mw_2027']}
    out['cap'] = out['year'].map(cap_by_year)
    out['value'] = out['capfactor'] * out['cap']

    return out['value'].values

# Project load to future using a grow rate of 2% (2025 -> 2026 there was a growth of 2.35%)
def future_load(df, target_index, load_growth_rate=0.02):
    prof = df.groupby(['month', 'hour', 'is_weekend'])['load_da_forecast_mw'].mean()
    out = pd.DataFrame(index=target_index)
    out['month'] = target_index.month
    out['hour'] = target_index.hour
    out['is_weekend'] = (target_index.dayofweek >= 5).astype(int)
    base = out.set_index(['month', 'hour', 'is_weekend']).index.map(prof).values
    # growth starts from 2026 (approx lat full training year)
    frac_years = (target_index - pd.Timestamp('2026-01-01', tz='UTC')).days / 365.25
    growth_factor = (1 + load_growth_rate) ** frac_years

    return base * growth_factor

# future gas and carbon prices 
def future_gas_eua(df, target_index):
    # average the last 90 days of data: we assume that the prices sty near wherever they recently were
    anchor_start = df.ts_utc.max() - pd.Timedelta(days=90)
    anchor_gas = df.loc[df.ts_utc >= anchor_start, 'gas_the_day_ahead_eur_mwh'].mean()
    anchor_eua = df.loc[df.ts_utc >= anchor_start, 'eua_dec_front_settle_eur_t'].mean()
    anchor_month = df.ts_utc.max().month
    # gas depends on the seasons: we compute the typical monthly price relative to the month our anchor sits in. December will get scaled up a bit compared to September
    monthly_seasonal = df.groupby('month')['gas_the_day_ahead_eur_mwh'].mean()
    seasonal_ratio = monthly_seasonal / monthly_seasonal.loc[anchor_month]
    # eua gets no seasonal adjustment
    months = target_index.month
    gas = anchor_gas * seasonal_ratio.reindex(months).values
    eua = np.full(len(target_index), anchor_eua)

    return gas, eua

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from load_data import build_history
    from features import calendar_feat

    df, cap = build_history()
    df = df.dropna(subset=['load_da_forecast_mw']).reset_index(drop=True)
    df = calendar_feat(df)
    cap_proj = future_capacity(cap)
    print(cap_proj)

    target_index = pd.date_range('2026-09-01 00:00', '2027-12-31 23:00', freq='h', tz='UTC')
    print(len(target_index), 'target hours')

    load_proj = future_load(df, target_index)
    print('projected load sample: ', load_proj[:5], load_proj.mean())

    gas_proj, eua_proj = future_gas_eua(df, target_index)
    print('gas sample: ', gas_proj[:5], ' eua sample: ', eua_proj[:5])
