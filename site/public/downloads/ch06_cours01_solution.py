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


print(difference([100,110,125,130]))
