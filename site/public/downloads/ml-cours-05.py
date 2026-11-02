import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=8,
    n_redundant=4, weights=[0.9, 0.1], random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
strategies = {
    "KFold": KFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
}
for name, cv in strategies.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(name, scores, scores.mean(), scores.std())
# Aucun des deux protocoles n'est adapté automatiquement au temps ou aux groupes.
