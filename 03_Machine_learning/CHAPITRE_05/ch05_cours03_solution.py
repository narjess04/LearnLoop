"""Explainable Credit Risk Model. Local Python: pip install numpy pandas scikit-learn matplotlib shap lime xgboost.
Données synthétiques, sans valeur opérationnelle. Utiliser --plot pour les figures.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer

def run(plot=False):
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5,
                              n_redundant=2, random_state=42)
    names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42)
    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.05,
                              eval_metric="logloss", random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("AUC test:", roc_auc_score(y_test, proba))
    print("F1 au seuil 0.5 fixé avant le test:", f1_score(y_test, proba >= .5))
    # Population de référence issue uniquement du train.
    background = X_train[np.random.default_rng(42).choice(len(X_train), 50, replace=False)]
    explainer = shap.TreeExplainer(model, data=background,
        feature_names=names, feature_perturbation="interventional",
        model_output="probability")
    values = explainer(X_test[:60])
    reconstructed = values.base_values + values.values.sum(axis=1)
    assert np.allclose(reconstructed, model.predict_proba(X_test[:60])[:, 1], atol=1e-5)
    importance = pd.Series(np.abs(values.values).mean(axis=0), index=names).sort_values(ascending=False)
    print("Importance globale (amplitude, pas direction):\n", importance)
    stability = []
    for seed in (7, 42, 99):
        local = LimeTabularExplainer(X_train, feature_names=names,
            class_names=["non-défaut", "défaut"], mode="classification",
            random_state=seed, discretize_continuous=True)
        for i in range(3):
            explanation = local.explain_instance(X_test[i], model.predict_proba,
                labels=(1,), num_features=6, num_samples=2000)
            stability.append({"client": i, "seed": seed,
                "probability": float(proba[i]), "local_prediction": float(explanation.local_pred[0]),
                "local_R2": float(explanation.score), "terms": explanation.as_list(label=1)})
            if plot and seed == 42:
                explanation.as_pyplot_figure(label=1)
    print(pd.DataFrame(stability).to_string(index=False))
    assert np.isfinite([row["local_prediction"] for row in stability]).all()
    if plot:
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(values, show=False)
        for i in range(3):
            shap.plots.waterfall(values[i], show=False)
        plt.show()
    return importance, stability

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
