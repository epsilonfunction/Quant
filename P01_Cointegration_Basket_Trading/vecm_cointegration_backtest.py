import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


def load_data(
    tickers=None,
    start_date='2021-01-01',
    end_date='2025-12-31',
    dropna=True,
):
    if tickers is None:
        tickers = ['MU', 'NVDA', 'TSM', 'AMD', 'MRVL', 'ORCL', 'AVGO', 'SMCI', 'MSFT']

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

    if dropna:
        data = data.dropna()

    if data.empty:
        raise ValueError('No price data loaded; check tickers and date range.')

    return data


def run_vecm_backtest(data, coint_rank=1, k_ar_diff=2, deterministic='ci'):
    # 1) Johansen cointegration test
    jres = coint_johansen(data, det_order=0, k_ar_diff=1)
    print('=== Johansen trace statistics ===')
    print(pd.DataFrame(
        {
            'trace_stat': jres.trace_stat,
            'cv_90': jres.trace_stat_crit_vals[:, 0],
            'cv_95': jres.trace_stat_crit_vals[:, 1],
            'cv_99': jres.trace_stat_crit_vals[:, 2],
        },
        index=[f'rank<={i}' for i in range(len(jres.trace_stat))]
    ))

    # 2) Fit VECM
    vecm = VECM(endog=data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
    fit = vecm.fit()
    print('=== VECM summary ===')
    print(fit.summary())

    beta = fit.beta[:, 0]
    print('beta (cointegration vector):', list(beta))

    has_const = hasattr(fit, 'det_coef') and fit.det_coef is not None
    intercept = float(fit.det_coef[0]) if has_const else 0.0

    spread = data.dot(beta) + intercept
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    df_signals = pd.DataFrame(
        {
            'spread': spread,
            'zscore': zscore,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
        }
    )

    df_signals['long_entry'] = df_signals['zscore'] < -2.0
    df_signals['long_exit'] = df_signals['zscore'] > -0.5
    df_signals['short_entry'] = df_signals['zscore'] > 2.0
    df_signals['short_exit'] = df_signals['zscore'] < 0.5

    return fit, df_signals


def run_forward_test(data, vecm_fit, lookahead=50):
    model = VECM(endog=data, k_ar_diff=vecm_fit.k_ar_diff, coint_rank=vecm_fit.coint_rank, deterministic=vecm_fit.deterministic)
    fwd = model.fit()
    forecast = fwd.predict(steps=lookahead)

    return forecast


def plot_spread(df_signals):
    spread_mean = float(df_signals['spread_mean'].iloc[0])
    spread_std = float(df_signals['spread_std'].iloc[0])

    plt.figure(figsize=(14, 5))
    plt.plot(df_signals['spread'], label='Spread', linewidth=1.3)
    plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
    plt.axhline(spread_mean + 2 * spread_std, color='red', linestyle='--', label='+2 Sigma')
    plt.axhline(spread_mean - 2 * spread_std, color='green', linestyle='--', label='-2 Sigma')
    plt.title('VECM Cointegration Spread')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 3))
    plt.plot(df_signals['zscore'], label='zscore', color='magenta')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--')
    plt.title('Spread Z-Score')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = load_data()
    fit, signals = run_vecm_backtest(data)
    plot_spread(signals)

    print(signals[['zscore', 'long_entry', 'short_entry']].tail(10))

    # Optional forward test (using same model settings; use a truly out-of-sample dataset if possible)
    # forecast = run_forward_test(data, fit, lookahead=50)
    # print(f'Forward forecast shape: {forecast.shape}')

