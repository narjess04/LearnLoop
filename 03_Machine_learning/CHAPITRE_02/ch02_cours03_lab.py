import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

rng = np.random.default_rng(42)
normal = rng.normal(loc=50, scale=10, size=(500, 2))
anomalies = rng.uniform(low=100, high=150, size=(20, 2))
X = np.vstack([normal, anomalies])
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
truth = np.r_[np.zeros(500, dtype=bool), np.ones(20, dtype=bool)]
q1, q3 = df.quantile(0.25), df.quantile(0.75)
iqr = q3 - q1
median = np.median(X, axis=0)
mad = np.median(np.abs(X - median), axis=0)
assert (mad > 0).all()
scaled = StandardScaler().fit_transform(X)
flags = {
    "IQR": ((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).any(axis=1).to_numpy(),
    "Z-score": (np.abs(zscore(X, axis=0)) > 3).any(axis=1),
    "Robust Z": (np.abs(0.6745 * (X - median) / mad) > 3.5).any(axis=1),
    "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.05,
                                        random_state=42).fit_predict(X) == -1,
    "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(scaled) == -1,
}

if __name__ == "__main__":
    for name, detected in flags.items():
        assert detected.shape == truth.shape
        print(name, "signalées:", detected.sum(),
              "precision:", precision_score(truth, detected, zero_division=0),
              "recall:", recall_score(truth, detected, zero_division=0))
    print("Comparaison descriptive sur les données de fit, sans test indépendant.")
    print("Les anomalies injectées sont des labels de simulation, pas des erreurs métier prouvées.")
