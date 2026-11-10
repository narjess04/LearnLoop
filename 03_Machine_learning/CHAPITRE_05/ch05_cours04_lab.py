"""ML Robustness Benchmark. Synthétique ; seuils de décision fixés avant le test.
Le jeu de stress est un audit final : aucun choix de modèle n'est optimisé dessus.
--plot pour les figures. --xgboost ajoute XGBoost en environnement local.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

def perturb(X, scale, level=0.0, missing_rate=0.0, shift=0.0, seed=42):
    X = np.asarray(X, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if X.ndim != 2 or scale.shape != (X.shape[1],) or not np.isfinite(X).all():
        raise ValueError("Matrice X finie et échelles alignées requises.")
    if not np.isfinite(scale).all() or np.any(scale < 0) or not np.isfinite([level, missing_rate, shift]).all():
        raise ValueError("Paramètres finis et échelles positives ou nulles requis.")
    if level < 0 or not 0 <= missing_rate <= 1:
        raise ValueError("Bruit positif ou nul et taux manquant dans [0,1] requis.")
    rng = np.random.default_rng(seed)
    changed = X.copy() + rng.normal(size=X.shape) * scale * level
    changed[:, 0] += shift * scale[0]
    changed[rng.random(X.shape) < missing_rate] = np.nan
    return changed

def metrics(y, p):
    return {"AUC": roc_auc_score(y, p),
            "F1": f1_score(y, p >= .5, zero_division=0),
            "Brier": brier_score_loss(y, p)}

def run(plot=False, include_xgboost=False):
    X, y = make_classification(n_samples=900, n_features=8,
        n_informative=5, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42)
    estimators = {"Logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1)}
    if include_xgboost:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(n_estimators=60, max_depth=3,
            random_state=42, n_jobs=1, eval_metric="logloss")
    # Les échelles et l'imputation viennent du train, puis restent figées.
    scale = X_train.std(axis=0)
    rows = []
    for name, estimator in estimators.items():
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
        model.fit(X_train, y_train)
        reference = metrics(y_test, model.predict_proba(X_test)[:, 1])
        for scenario, args in [
            ("référence", {}),
            ("bruit 5 % écart-type train", {"level": .05}),
            ("bruit 20 % écart-type train", {"level": .20}),
            ("10 % absences", {"missing_rate": .10}),
            ("décalage de mesure +1 écart-type", {"shift": 1.0}),
            ("stress combiné", {"level": .30, "missing_rate": .20, "shift": 1.0}),
        ]:
            for seed in (7, 42, 99):
                changed = perturb(X_test, scale, seed=seed, **args)
                observed = metrics(y_test, model.predict_proba(changed)[:, 1])
                rows.append({"model": name, "scenario": scenario, "seed": seed,
                    **observed, "delta_AUC": observed["AUC"] - reference["AUC"],
                    "delta_Brier": observed["Brier"] - reference["Brier"]})
    report = pd.DataFrame(rows)
    assert np.isfinite(report.select_dtypes("number").to_numpy()).all()
    clean = report[report["scenario"] == "référence"]
    assert np.allclose(clean["delta_AUC"], 0)
    assert np.allclose(clean["delta_Brier"], 0)
    summary = report.groupby(["model", "scenario"])[["AUC", "F1", "Brier", "delta_AUC", "delta_Brier"]].agg(["mean", "std"])
    print(summary.to_string())
    print("Delta AUC < 0 : dégradation ; delta Brier > 0 : dégradation.")
    print("Le décalage simulé est une corruption de mesure avec labels inchangés, pas une preuve de covariate shift réel.")
    print("La variabilité sur les graines ne couvre ni l'incertitude d'échantillonnage ni la stabilité au réentraînement.")
    if plot:
        import matplotlib.pyplot as plt
        for name in report["model"].unique():
            group = report[report["model"] == name].groupby("scenario")["AUC"].mean()
            group.plot(kind="bar", title=name+" : AUC moyenne par stress")
            plt.tight_layout()
            plt.show()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    args = parser.parse_args()
    run(args.plot, args.xgboost)
