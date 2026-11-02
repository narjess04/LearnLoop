import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_array, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5,
    n_redundant=2, flip_y=0.05, random_state=42
)
X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(10)])
train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
def evaluate(features):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features.iloc[train_idx], y[train_idx])
    proba = model.predict_proba(features.iloc[test_idx])[:, 1]
    return roc_auc_score(y[test_idx], proba)

print("Features originales :", evaluate(X))
leaked = X.copy()
leaked["future_target"] = y  # INFORMATION INTERDITE à la prédiction.
print("Avec fuite volontaire :", evaluate(leaked))
# Le même split ne rend pas une feature issue de la cible légitime.
