import numpy as np

def baseline_forecast(history, horizon, period=12, window=3):
    # TODO : clés Naive, Seasonal Naive, Moving Average et Drift.
    raise NotImplementedError('Complète les baselines')

def forecasting_metrics(y_true, y_pred, y_train, seasonality=1):
    # TODO : MAE, RMSE, MASE, sMAPE ; sMAPE 0/0=0, MASE indéfinie si échelle nulle.
    raise NotImplementedError('Complète les métriques')

print(baseline_forecast(np.arange(24.),12))
