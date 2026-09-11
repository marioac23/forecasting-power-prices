'''
Creates dataframe using the projected data
'''
import numpy as np
import pandas as pd 
from load_data import build_history
from features import calendar_feat, add_feats, FEATURE_COLS 
from project_feats import future_capacity, project_cap_factor_series, future_load, future_gas_eua

# Create time col for the future data
def make_target_grid():
    local_idx = pd.date_range('2026-09-01 00:00', '2027-12-31 23:00', freq='h',
                               tz='Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    utc_idx = local_idx.tz_convert('UTC')
    return utc_idx, local_idx

def build_future_df():
    df, cap = build_history()
    df = df.dropna(subset=['load_da_forecast_mw']).reset_index(drop=True)
    df = calendar_feat(df)

    cap_proj = future_capacity(cap)
    utc_index, local_idx = make_target_grid()
    # Add time
    fut = pd.DataFrame({'ts_utc': utc_index})
    fut['ts_local'] = local_idx.tz_localize(None).astype(str)
    fut['year'] = local_idx.year
    fut['month'] = local_idx.month
    fut['hour'] = local_idx.hour
    fut['dow'] = local_idx.dayofweek
    fut['is_weekend'] = (fut['dow'] >= 5).astype(int)

    # Add projections
    fut['load_da_forecast_mw'] = future_load(df, utc_index.tz_convert('UTC'))
    fut['da_forecast_solar_mw'] = project_cap_factor_series(
        df, cap_proj, utc_index, 'solar', 'da_forecast_solar_mw', cap.set_index('technology').loc['solar','installed_mw_2025'], cap.set_index('technology').loc['solar','installed_mw_2026'])
    fut['da_forecast_wind_onshore_mw'] = project_cap_factor_series(
        df, cap_proj, utc_index, 'wind_onshore', 'da_forecast_wind_onshore_mw', cap.set_index('technology').loc['wind_onshore','installed_mw_2025'], cap.set_index('technology').loc['wind_onshore','installed_mw_2026'])
    fut['da_forecast_wind_offshore_mw'] = project_cap_factor_series(
        df, cap_proj, utc_index, 'wind_offshore', 'da_forecast_wind_offshore_mw', cap.set_index('technology').loc['wind_offshore','installed_mw_2025'], cap.set_index('technology').loc['wind_offshore','installed_mw_2026'])
    
    gas_proj, eua_proj = future_gas_eua(df, utc_index)
    fut['gas_the_day_ahead_eur_mwh'] = gas_proj
    fut['eua_dec_front_settle_eur_t'] = eua_proj

    # Add cyclic time
    fut['sin_hour'] = np.sin(2*np.pi*fut['hour']/24)
    fut['cos_hour'] = np.cos(2*np.pi*fut['hour']/24)
    doy = local_idx.dayofyear
    fut['sin_doy'] = np.sin(2*np.pi*doy/365.25)
    fut['cos_doy'] = np.cos(2*np.pi*doy/365.25)

    fut['eua_price'] = fut['eua_dec_front_settle_eur_t']
    fut = add_feats(fut)

    return fut

if __name__ == '__main__':
    fut = build_future_df()
    print(fut.shape)
    print(fut[['ts_utc','ts_local']].head(3))
    print(fut[['ts_utc','ts_local']].tail(3))
    na = fut[FEATURE_COLS].isna().sum()
    print(na[na>0])
    print(fut[FEATURE_COLS].describe().T[['min','max','mean']])
    fut.to_pickle('../output/future_features.pkl')