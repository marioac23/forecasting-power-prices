'''
Hybrid Model: linear quantile trend (extrapolates) + HistGradientBoosting quantile model on residuals (captures non-linear shape)
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# features that are roughly linear to the price
TREND_COLS = ['res_load_forecast', 'srmc_gas', 'srmc_coal', 'renew_share_forecast']

class HybridQuantileModel:
    def __init__(self, feat_cols, quantiles=(0.1,0.5,0.9), trend_cols=TREND_COLS):
        self.feat_cols = feat_cols
        self.trend_cols = trend_cols
        self.quantiles = quantiles
        self.scaler = StandardScaler()
        self.linear_models = {}
        self.resid_models = {}

    def fit(self, X, y):
        Xt = self.scaler.fit_transform(X[self.trend_cols])
        for q in self.quantiles:
            # alpha=0 turns off regularization (only 4 feats used)
            # solver = highs: pinball loss (quantile loss)
            lin = QuantileRegressor(quantile=q, alpha=0.0, solver='highs')
            lin.fit(Xt, y)
            self.linear_models[q] = lin
        # residual GBM trained on the median linear residual
        lin_median_pred = self.linear_models[0.5].predict(Xt)
        # subtract from price what the median linear model predicts
        # residual: whatever the linear trend didnt explain
        resid = y - lin_median_pred
        for q in self.quantiles:
            # init GBM
            gbm = HistGradientBoostingRegressor(loss='quantile',
                                                quantile=q, max_iter=350, max_depth=5,
                                                learning_rate=0.05, l2_regularization=0.2,
                                                random_state=0)
            # fit GBM using the wigly feats and residual
            gbm.fit(X[self.feat_cols], resid)
            self.resid_models[q] = gbm
        
        return self

    # predict prices
    def predict(self, X):
        Xt = self.scaler.transform(X[self.trend_cols])
        preds = {}
        
        for q in self.quantiles:
            lin_q = self.linear_models[q].predict(Xt)
            resid_q = self.resid_models[q].predict(X[self.feat_cols])
            # combine: linear quantile + residual shape quantile (both are centered at the median)
            preds[q] = lin_q + resid_q 
        # enforce that p10 < p50 < p90
        stacked = np.vstack([preds[q] for q in sorted(self.quantiles)])
        stacked = np.sort(stacked, axis=0)
        for i, q in enumerate(sorted(self.quantiles)):
            preds[q] = stacked[i]
        
        return preds