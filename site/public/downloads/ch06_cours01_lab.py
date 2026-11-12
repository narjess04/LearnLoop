import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def dataset():
    rng = np.random.default_rng(42)
    time = np.arange(72)
    sales = 100 + 1.5*time + 20*np.sin(2*np.pi*time/12) + rng.normal(0, 5, 72)
    return pd.DataFrame({'ventes': sales}, index=pd.date_range('2020-01-01', periods=72, freq='MS'))

def difference(values, lag=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not isinstance(lag, (int, np.integer)) or not 0 < lag < len(values):
        raise ValueError('Retard entier positif et inférieur à la longueur requis.')
    return values[lag:] - values[:-lag]

def check_stationarity(series):
    series = series.dropna()
    return {'ADF': adfuller(series, autolag='AIC')[1],
            'KPSS': kpss(series, regression='c', nlags='auto')[1]}

def run():
    df = dataset()
    print('Fréquence :', df.index.freqstr, 'Manquants :', df.isna().sum().sum())
    df.plot(title='Ventes mensuelles synthétiques', figsize=(11, 4))
    STL(df['ventes'], period=12, robust=True).fit().plot()
    candidate = df['ventes'].diff(12).diff().dropna()
    print('Brut :', check_stationarity(df['ventes']))
    print('Transformé :', check_stationarity(candidate))
    candidate.plot(title='Différenciation saisonnière puis simple', figsize=(11, 4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(candidate, lags=18, ax=axes[0])
    plot_pacf(candidate, lags=18, method='ywm', ax=axes[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()
