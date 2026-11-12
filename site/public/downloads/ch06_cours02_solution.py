import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def dataset():
    rng = np.random.default_rng(42)
    t = np.arange(60)
    sales = 100 + 1.2*t + 25*np.sin(2*np.pi*t/12) + rng.normal(0, 5, 60)
    return pd.DataFrame({'sales': sales}, index=pd.date_range('2021-01-01', periods=60, freq='MS'))

def baseline_forecast(history, horizon, period=12, window=3):
    history = np.asarray(history, dtype=float)
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (horizon, period, window)):
        raise ValueError('Horizon, période et fenêtre entiers positifs requis.')
    if history.ndim != 1 or len(history) < max(period, window, 2) or not np.isfinite(history).all():
        raise ValueError('Historique fini et suffisamment long requis.')
    return {'Naive': np.repeat(history[-1], horizon),
            'Seasonal Naive': np.resize(history[-period:], horizon),
            'Moving Average': np.repeat(history[-window:].mean(), horizon),
            'Drift': history[-1]+np.arange(1,horizon+1)*(history[-1]-history[0])/(len(history)-1)}

def forecasting_metrics(y_true, y_pred, y_train, seasonality=1):
    y_true, y_pred, y_train = [np.asarray(v, dtype=float) for v in (y_true, y_pred, y_train)]
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in (y_true, y_pred, y_train)):
        raise ValueError('Vecteurs finis à une dimension requis.')
    if y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError('Cibles et prédictions non vides et alignées requises.')
    if not isinstance(seasonality, (int, np.integer)) or seasonality < 1 or len(y_train) <= seasonality:
        raise ValueError('Période entière positive et train plus long que cette période requis.')
    error = y_true-y_pred
    mae = np.abs(error).mean()
    scale = np.abs(y_train[seasonality:]-y_train[:-seasonality]).mean()
    denominator = np.abs(y_true)+np.abs(y_pred)
    terms = np.divide(2*np.abs(error), denominator, out=np.zeros_like(error), where=denominator!=0)
    return {'MAE': float(mae), 'RMSE': float(np.sqrt(np.mean(error**2))),
            'MASE': float(mae/scale) if scale>0 else np.nan, 'sMAPE': float(100*terms.mean())}


print(baseline_forecast(np.arange(24.),12))
