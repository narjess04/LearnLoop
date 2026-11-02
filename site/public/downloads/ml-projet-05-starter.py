import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
time = np.arange(120)
sales = 100 + 0.5*time + 10*np.sin(2*np.pi*time/12) + rng.normal(0, 5, 120)
df = pd.DataFrame({"month": time, "sales": sales})
for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)
df = df.dropna()
X, y = df[["lag_1", "lag_2", "lag_3"]], df["sales"]
# TODO : TimeSeriesSplit, fit et MAE par fold.
# TODO : comparer à la baseline y_pred = lag_1.
# TODO : définir l'horizon ; les vrais lags de validation ne sont disponibles
# qu'en prévision un pas avec observation des valeurs passées.
# Pour un horizon multi-pas, utiliser un protocole compatible (récursif/direct).
