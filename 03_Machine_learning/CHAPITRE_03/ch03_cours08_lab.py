"""Cours 08. Installer xgboost lightgbm catboost ; --plot pour les figures."""

import argparse
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)

def dataset(n=5000, features=20, informative=10, redundant=5, positive=.08):
    X, y = make_classification(n_samples=n, n_features=features,
        n_informative=informative, n_redundant=redundant, weights=[1-positive, positive],
        flip_y=.01, random_state=42)
    return train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def benchmark(models, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
            scoring={"ROC_AUC": "roc_auc", "AP": "average_precision"})
        rows.append({"Model": name, "ROC_AUC_CV": scores["test_ROC_AUC"].mean(),
            "ROC_AUC_std": scores["test_ROC_AUC"].std(), "AP_CV": scores["test_AP"].mean(),
            "fit_seconds_CV_mean": scores["fit_time"].mean()})
    table = pd.DataFrame(rows).sort_values("ROC_AUC_CV", ascending=False)
    print(table.to_string(index=False))
    chosen = table.iloc[0]["Model"]
    print("Choix figé sur la ROC-AUC CV avant de consulter le test :", chosen)
    return chosen, table

def metrics(model, X, y, threshold=None):
    probabilistic = hasattr(model, "predict_proba")
    score = model.predict_proba(X)[:, 1] if probabilistic else model.decision_function(X)
    pred = model.predict(X) if threshold is None else (score >= threshold).astype(int)
    assert np.isfinite(score).all()
    result = {"ROC_AUC": roc_auc_score(y, score), "AP": average_precision_score(y, score),
        "Gini": 2*roc_auc_score(y, score)-1,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if probabilistic:
        assert ((score >= 0) & (score <= 1)).all()
        result.update(Brier=brier_score_loss(y, score), LogLoss=log_loss(y, score))
    else:
        result.update(Brier=np.nan, LogLoss=np.nan)
    return result

def audit(models, X_train, y_train, X_test, y_test):
    rows=[]
    for name, model in models.items():
        start=perf_counter(); model.fit(X_train, y_train); duration=perf_counter()-start
        rows.append({"Model": name, **metrics(model, X_test, y_test), "fit_seconds": duration})
    table=pd.DataFrame(rows)
    print("Audit final des candidats préspécifiés ; aucune nouvelle sélection à partir du test.")
    print(table.to_string(index=False))
    return table

def figures(models, X, y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    for display, title in [(RocCurveDisplay, "ROC"), (PrecisionRecallDisplay, "Precision-Recall")]:
        fig, ax=plt.subplots()
        for name, model in models.items(): display.from_estimator(model, X, y, ax=ax, name=name)
        ax.set_title(title); fig.tight_layout()
    fig, ax=plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            CalibrationDisplay.from_estimator(model, X, y, n_bins=8, strategy="quantile", ax=ax, name=name)
    fig.tight_layout(); plt.show()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=.05, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=.05, max_depth=4, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbosity=-1, random_state=42, n_jobs=1),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=.05, depth=5, verbose=False, random_seed=42, thread_count=1, allow_writing_files=False),
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--plot", action="store_true"); args=parser.parse_args()
    X_train, X_test, y_train, y_test=dataset()
    candidates=models(); chosen, table=benchmark(candidates, X_train, y_train)
    audit(candidates, X_train, y_train, X_test, y_test)
    if args.plot: figures(candidates, X_test, y_test)
