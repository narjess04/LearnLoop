import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from ch03_cours02_lab import X_train, X_test, y_train, y_test


def tune_models(X_fit, y_fit):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    specifications = [
        ("Ridge", Ridge(), {"model__alpha": np.logspace(-4, 4, 30)}),
        ("Lasso", Lasso(max_iter=50000), {"model__alpha": np.logspace(-2, 2, 20)}),
        ("Elastic Net", ElasticNet(max_iter=50000),
         {"model__alpha": np.logspace(-2, 1, 8), "model__l1_ratio": [.2, .5, .8, 1.]})]
    fitted = {}
    for name, estimator, grid in specifications:
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="neg_root_mean_squared_error")
        search.fit(X_fit, y_fit)
        assert np.isfinite(search.best_score_)
        fitted[name] = search
        print(name, "paramètres:", search.best_params_, "RMSE de sélection CV:", -search.best_score_)
    return fitted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Expérience séparée : 100 observations train")
    parser.add_argument("--plot", action="store_true", help="Tracer les regularization paths sur train")
    args = parser.parse_args()
    if args.small:
        idx = np.random.default_rng(43).choice(len(X_train), 100, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
    else:
        X_fit, y_fit = X_train, y_train
    searches = tune_models(X_fit, y_fit)
    chosen = max(searches, key=lambda name: searches[name].best_score_)
    print("Choix figé par CV avant lecture des résultats test:", chosen)
    rows = []
    for name, search in searches.items():
        coef = search.best_estimator_.named_steps["model"].coef_
        pred = search.predict(X_test)
        assert np.isfinite(pred).all()
        rows.append({"model": name, "RMSE_test_audit": np.sqrt(mean_squared_error(y_test, pred)),
                     "coefficients_non_nuls": np.count_nonzero(coef)})
    print(pd.DataFrame(rows).to_string(index=False))
    # Vérifier la formulation Ridge avec matrice augmentée, sans inversion de X.T@X.
    reference = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2., solver="svd"))]).fit(X_fit, y_fit)
    Z = reference.named_steps["scaler"].transform(X_fit)
    augmented_X = np.vstack([Z, np.sqrt(2.) * np.eye(Z.shape[1])])
    augmented_y = np.r_[y_fit - y_fit.mean(), np.zeros(Z.shape[1])]
    coefficients = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)[0]
    assert np.allclose(coefficients, reference.named_steps["model"].coef_)
    assert np.isclose(reference.named_steps["model"].intercept_, y_fit.mean())
    # Le fit du scaler reste limité au train, y compris pour les nouvelles observations.
    before = reference.named_steps["scaler"].mean_.copy()
    reference.predict(X_test)
    assert np.array_equal(before, reference.named_steps["scaler"].mean_)
    # Forte pénalité : diminution de la norme L2 et possibilité d'annuler les coefficients L1.
    ridge_strong = clone(reference).set_params(model__alpha=1e6).fit(X_fit, y_fit)
    assert np.linalg.norm(ridge_strong.named_steps["model"].coef_) < np.linalg.norm(coefficients)
    lasso_strong = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=1e6, max_iter=50000))]).fit(X_fit, y_fit)
    assert np.count_nonzero(lasso_strong.named_steps["model"].coef_) == 0
    print("Formulation Ridge, intercept non pénalisé, scaler train et forte pénalité : OK")
    if args.plot:
        import matplotlib.pyplot as plt
        scaler = StandardScaler().fit(X_fit)
        Z = scaler.transform(X_fit)
        alphas = np.logspace(-2, 3, 30)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, estimator, title in [(axes[0], Ridge(), "Ridge"),
                                      (axes[1], Lasso(max_iter=50000), "Lasso")]:
            path = np.array([clone(estimator).set_params(alpha=float(a)).fit(Z, y_fit).coef_ for a in alphas])
            ax.semilogx(alphas, path)
            ax.set(title=title, xlabel="alpha (convention propre au modèle)", ylabel="Coefficient standardisé")
        fig.tight_layout()
        plt.show()
    print("Le score best_score_ sélectionne la configuration ; il n'est pas une estimation indépendante.")
    print("Le bonus --small est une expérience descriptive distincte ; ne pas régler la procédure sur test.")
