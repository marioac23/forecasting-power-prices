'''
Makes differents plots based on the model prediction
'''
import pandas as pd
import matplotlib.pyplot as plt

FORECAST_PATH = '../output/forecast.csv'
HIST_PRICE_PATH = '../output/history_features.pkl'
OUTPUT_DIR = '../plots/'

def load_forecast():
    fc = pd.read_csv(FORECAST_PATH)
    fc['ts_utc'] = pd.to_datetime(fc['ts_utc'])
    fc['ts_local'] = pd.to_datetime(fc['ts_local'])
    fc['hour'] = fc.ts_local.dt.hour
    fc['month'] = fc.ts_local.dt.month
    fc['ym'] = fc.ts_local.dt.to_period('M')
    return fc

def load_history():
    h = pd.read_pickle(HIST_PRICE_PATH)
    h['ts_local'] = pd.to_datetime(h.ts_local)
    h['hour'] = h.ts_local.dt.hour
    h['month'] = h.ts_local.dt.month
    return h

def plot_price_band(fc):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(fc.ts_local, fc.p10_eur_mwh, fc.p90_eur_mwh,
                     alpha=0.25, color='steelblue', label='p10-p90 band')
    ax.plot(fc.ts_local, fc.price_eur_mwh, lw=0.6, color='steelblue', label='median forecast')
    ax.axhline(0, color='red', lw=0.6, ls='--')
    ax.set_title('Full forecast price: median + p10-p90 band')
    ax.set_ylabel('EUR/MWh')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR+'forecast_price_band.png')
    plt.close(fig)

def plot_montly_compare(fc, hist):
    monthly_fc = fc.groupby('ym')['price_eur_mwh'].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([str(p) for p in monthly_fc.index], monthly_fc.values, color='steelblue', label='forecast')
    if hist is not None:
        hist_monthly_by_calendar_month = hist.groupby('month')['price_eur_mwh'].mean()
        ref = [hist_monthly_by_calendar_month.loc[p.month] for p in monthly_fc.index]
        ax.plot(range(len(monthly_fc)), ref, color='darkorange', marker='o', ms=3,
                label='historical avg for that calendar month (reference)')
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(monthly_fc)))
    ax.set_xticklabels([str(p) for p in monthly_fc.index], rotation=60, ha='right', fontsize=7)
    ax.set_title('Monthly average forecast price')
    ax.set_ylabel('EUR/MWh')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR+'monthly_compare.png')
    plt.close(fig)

def plot_shape_by_hour_month(fc, hist):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    fc.boxplot(column='price_eur_mwh', by='hour', ax=axes[0][0], showfliers=False)
    axes[0][0].set_title('Forecast: price by hour of day')
    axes[0][0].set_xlabel('hour'); axes[0][0].set_ylabel('EUR/MWh')
    hist.boxplot(column='price_eur_mwh', by='hour', ax=axes[1][0], showfliers=False)
    axes[1][0].set_title('History: price by hour of day')
    axes[1][0].set_xlabel('hour'); axes[1][0].set_ylabel('EUR/MWh')
    fc.boxplot(column='price_eur_mwh', by='month', ax=axes[0][1], showfliers=False)
    axes[0][1].set_title('Forecast: price by month')
    axes[0][1].set_xlabel('calendar month')
    hist.boxplot(column='price_eur_mwh', by='month', ax=axes[1][1], showfliers=False)
    axes[1][1].set_title('history: price by month')
    axes[1][1].set_xlabel('calendar month')
    fig.suptitle('Forecast daily/seasonal shape compared to historical data')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR+'hour_month_compare.png')
    plt.close(fig)

def plot_history_and_forecast_combined(fc, hist):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(hist.ts_utc, hist.price_eur_mwh, lw=0.4, color='steelblue', label='history')
    ax.fill_between(fc.ts_utc, fc.p10_eur_mwh, fc.p90_eur_mwh,
                     alpha=0.25, color='darkorange', label='forecast p10-p90 band')
    ax.plot(fc.ts_utc, fc.price_eur_mwh, lw=0.5, color='darkorange', label='forecast (median)')
    ax.axvline(hist.ts_utc.max(), color='black', lw=1, ls='--')
    ax.text(hist.ts_utc.max(), ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0,
            '  history ends / forecast starts', fontsize=8)
    ax.axhline(0, color='red', lw=0.5, ls=':')
    ax.set_ylabel('EUR/MWh')
    ax.set_title('History and forecast price timeline')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR+'price_forecast_history.png')
    plt.close(fig)

if __name__ == '__main__':
    forecast = load_forecast()
    history = load_history()
    plot_price_band(forecast)
    plot_montly_compare(forecast, history)
    plot_shape_by_hour_month(forecast, history)
    plot_history_and_forecast_combined(forecast, history)
    print("Finished plots")