'''
File that produces the fundamental features used for training and predicting
'''
import pandas as pd
import numpy as np

def calendar_feat(df, ts_col='ts_utc', ts_local_col='ts_local'):
    local = pd.to_datetime(df[ts_local_col])
    df = df.copy()
    df['hour'] = local.dt.hour
    df['dow'] = local.dt.dayofweek
    df['month'] = local.dt.month
    df['doy'] = local.dt.dayofyear
    df['year'] = local.dt.year
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    # days are ciclycle and the model needs to understand that so we use sin and cos
    df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)
    df['sin_doy'] = np.sin(2*np.pi*df['doy']/365.25)
    df['cos_doy'] = np.cos(2*np.pi*df['doy']/365.25)
    # add holidays
    # df['is_holiday'] = is_holiday(local)
    return df

def add_feats(df):
    df = df.copy()
    # Feats that tell me if a fowwil plant is needed
    df['res_load_forecast'] = (df['load_da_forecast_mw']
                               - df['da_forecast_solar_mw']
                               - df['da_forecast_wind_offshore_mw']
                               - df['da_forecast_wind_onshore_mw'])
    df['wind_total_forecast'] = df['da_forecast_wind_onshore_mw'] + df['da_forecast_wind_offshore_mw']
    # renewables/total demand: what fraction of tonight's electricity is basically free
    df['renew_share_forecast'] = ((df['da_forecast_solar_mw'] + df['wind_total_forecast'])
                                   / df['load_da_forecast_mw'].replace(0, np.nan))
    
    # Feats that tell me what costs to run a fossil plant
    GAS_HEAT_RATE   = 1.77     
    COAL_HEAT_RATE  = 2.27
    GAS_EF  = 0.2028
    COAL_EF = 0.3388

    df['srmc_gas'] = df['gas_the_day_ahead_eur_mwh']*GAS_HEAT_RATE + df['eua_dec_front_settle_eur_t']*GAS_EF*GAS_HEAT_RATE

    # no coal price given: assume 40%
    COAL_GAS_DISCOUNT = 0.4
    implied_coal_fuel_price = df['gas_the_day_ahead_eur_mwh'] * COAL_GAS_DISCOUNT
    df['srmc_coal'] = implied_coal_fuel_price*COAL_HEAT_RATE + df['eua_dec_front_settle_eur_t']*COAL_EF*COAL_HEAT_RATE
    df['gas_coal_spark_spread_proxy'] = df['srmc_gas'] - df['srmc_coal']
    df['eua_price'] = df['eua_dec_front_settle_eur_t']
    
    return df

FEATURE_COLS = ['hour','dow','month','is_weekend', #'is_holiday',
    'sin_hour','cos_hour','sin_doy','cos_doy',
    'load_da_forecast_mw','da_forecast_solar_mw','da_forecast_wind_offshore_mw',
    'da_forecast_wind_onshore_mw','res_load_forecast','wind_total_forecast','renew_share_forecast',
    'gas_the_day_ahead_eur_mwh','eua_price','srmc_gas','srmc_coal','gas_coal_spark_spread_proxy',
]