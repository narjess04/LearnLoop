import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = make_regression(n_samples=500, n_features=30, n_informative=8,
                       noise=25, random_state=42)
rng = np.random.default_rng(42)
X_correlated = X[:, :5] + rng.normal(0, .05, size=(500, 5))
X = np.hstack([X, X_correlated])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


def make_models():
    return {name: Pipeline([("scaler", StandardScaler()), ("model", model)])
            for name, model in [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge(alpha=10)),
                ("Lasso", Lasso(alpha=1, max_iter=50000)),
                ("Elastic Net", ElasticNet(alpha=1, l1_ratio=.5, max_iter=50000))]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les coefficients")
    args = parser.parse_args()
    models = make_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, cv_scores = [], {}
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error", return_estimator=True)
        rmses = -result["test_score"]
        assert np.isfinite(rmses).all()
        cv_scores[name] = rmses.mean()
        # Comparer la stabilité dans les unités originales, car chaque fold a son scaler.
        raw_coefficients = np.array([p.named_steps["model"].coef_ / p.named_steps["scaler"].scale_
                                      for p in result["estimator"]])
        pipeline.fit(X_train, y_train)
        coefficients = pipeline.named_steps["model"].coef_
        rows.append({"model": name, "RMSE_CV": rmses.mean(), "std_RMSE_CV": rmses.std(),
                     "nonzero": np.count_nonzero(coefficients),
                     "mean_coefficient_std_across_folds": raw_coefficients.std(axis=0).mean()})
    print(pd.DataFrame(rows).to_string(index=False))
    chosen = min(cv_scores, key=cv_scores.get)
    print("Modèle choisi sur CV train avant ouverture du test:", chosen)
    # Audit final de configurations fixées ; ne pas changer le choix à partir de ce tableau.
    audit = []
    for name, pipeline in models.items():
        pred = pipeline.predict(X_test)
        audit.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                      "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    print(pd.DataFrame(audit).to_string(index=False))
    if args.plot:
        import matplotlib.pyplot as plt
        for name, pipeline in models.items():
            plt.plot(pipeline.named_steps["model"].coef_, marker="o", label=name)
        plt.xlabel("Indice de la variable")
        plt.ylabel("Coefficient sur variables standardisées")
        plt.legend()
        plt.tight_layout()
        plt.show()
    print("Les alpha des modèles ont des conventions différentes. Le test ne sert pas à les régler.")
