"""Split conformal et CQR : données synthétiques hétéroscédastiques.
Aucune recherche de modèle ni de risque alpha sur le test.
--plot pour les figures ; --xgboost ajoute une comparaison locale.
"""
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Un vecteur de scores finis non vide est requis.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement entre 0 et 1.")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        return float("inf")
    return float(np.sort(scores)[k - 1])

def interval_metrics(y, lower, upper):
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    if y.ndim != 1 or not len(y) or y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("Trois vecteurs alignés non vides sont requis.")
    if not np.isfinite(y).all() or np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("Cibles finies et bornes non NaN requises.")
    coverage = np.mean((lower <= y) & (y <= upper))
    # Un intervalle inversé représente un ensemble vide.
    width = np.maximum(upper - lower, 0)
    return {"coverage": float(coverage), "mean_width": float(width.mean())}

def run(plot=False, include_xgboost=False):
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(1000, 4))
    sigma = 4000 + 9000 * np.abs(X[:, 0])
    y = 250000 + 30000 * X[:, 0] + 15000 * X[:, 1] ** 2 + rng.normal(size=len(X)) * sigma
    train_cal, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    train, cal = train_test_split(train_cal, test_size=.25, random_state=7)
    assert not (set(train) & set(cal) or set(cal) & set(test) or set(train) & set(test))
    models = {"Random Forest": RandomForestRegressor(n_estimators=60, min_samples_leaf=5,
                                                    random_state=42, n_jobs=1)}
    if include_xgboost:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=80, max_depth=3, random_state=42, n_jobs=1)
    rows = []
    plot_bounds = None
    for name, model in models.items():
        model.fit(X[train], y[train])
        scores = np.abs(y[cal] - model.predict(X[cal]))
        predicted = model.predict(X[test])
        for alpha in (.2, .1, .05):
            q = conformal_quantile(scores, alpha)
            lower, upper = predicted - q, predicted + q
            metrics = interval_metrics(y[test], lower, upper)
            row = {"model": name, "target": 1-alpha, "q": q, **metrics}
            for group, mask in [("faible_bruit", np.abs(X[test, 0]) < 1.5),
                                ("fort_bruit", np.abs(X[test, 0]) >= 1.5)]:
                row["coverage_"+group] = interval_metrics(y[test][mask], lower[mask], upper[mask])["coverage"]
            rows.append(row)
            if name == "Random Forest" and alpha == .1:
                plot_bounds = (lower, upper, predicted)
    # CQR : modèles quantiles ajustés exclusivement sur train.
    for alpha in (.2, .1, .05):
        low_model = GradientBoostingRegressor(loss="quantile", alpha=alpha/2,
            n_estimators=70, max_depth=2, random_state=42).fit(X[train], y[train])
        high_model = GradientBoostingRegressor(loss="quantile", alpha=1-alpha/2,
            n_estimators=70, max_depth=2, random_state=42).fit(X[train], y[train])
        def quantiles(indices):
            a, b = low_model.predict(X[indices]), high_model.predict(X[indices])
            return np.minimum(a, b), np.maximum(a, b)
        low_cal, high_cal = quantiles(cal)
        # Les scores CQR peuvent être négatifs : ne pas les tronquer.
        scores = np.maximum(low_cal - y[cal], y[cal] - high_cal)
        q = conformal_quantile(scores, alpha)
        low_test, high_test = quantiles(test)
        rows.append({"model": "CQR", "target": 1-alpha, "q": q,
                     **interval_metrics(y[test], low_test-q, high_test+q)})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("Couverture empirique != garantie conditionnelle. Hypothèse : échangeabilité.")
    print("Comparer largeur, couverture et sous-populations ; aucune AUC ne mesure un intervalle.")
    if plot:
        import matplotlib.pyplot as plt
        low, high, predicted = plot_bounds
        order = np.argsort(predicted)[:60]
        plt.fill_between(np.arange(len(order)), low[order], high[order], alpha=.25, label="Split conformal 90 %")
        plt.plot(predicted[order], label="Prédiction")
        plt.scatter(np.arange(len(order)), y[test][order], s=10, label="Prix observé")
        plt.ylabel("Prix synthétique (DT)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    args = parser.parse_args()
    run(args.plot, args.xgboost)
