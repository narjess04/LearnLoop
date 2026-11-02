import numpy as np
import pandas as pd
from math import comb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(300, 1))
y = 5 - 2 * X[:, 0] + 1.5 * X[:, 0] ** 2 - .3 * X[:, 0] ** 3 + rng.normal(0, 2, size=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
pipeline = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                     ("scaler", StandardScaler()), ("model", Ridge())])
param_grid = [
    {"poly__degree": [1, 2, 3, 4, 5, 8], "model": [LinearRegression()]},
    {"poly__degree": [1, 2, 3, 4, 5, 8], "model": [Ridge()],
     "model__alpha": np.logspace(-3, 3, 10)},
]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_root_mean_squared_error")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", action="store_true", help="Évaluer la sélection en validation imbriquée sur train")
    parser.add_argument("--plot", action="store_true", help="Tracer le modèle retenu sur l'intervalle train")
    args = parser.parse_args()
    if args.nested:
        outer = KFold(n_splits=3, shuffle=True, random_state=43)
        outer_rmse = -cross_val_score(search, X_train, y_train, cv=outer,
                                      scoring="neg_root_mean_squared_error")
        print("RMSE externe de la procédure entière:", outer_rmse, outer_rmse.mean())
    search.fit(X_train, y_train)
    print("Paramètres figés par CV:", search.best_params_)
    print("RMSE de sélection (pas une estimation indépendante):", -search.best_score_)
    rows = []
    for index, params in enumerate(search.cv_results_["params"]):
        rows.append({"degree": params["poly__degree"], "model": type(params["model"]).__name__,
                     "alpha": params.get("model__alpha", np.nan),
                     "RMSE_CV": -search.cv_results_["mean_test_score"][index]})
    summary = pd.DataFrame(rows).sort_values("RMSE_CV").groupby(["model", "degree"], sort=False).first()
    print(summary.to_string())
    fitted = search.best_estimator_
    prediction = fitted.predict(X_test)
    assert np.isfinite(prediction).all()
    print("Test final : MAE", mean_absolute_error(y_test, prediction),
          "RMSE", np.sqrt(mean_squared_error(y_test, prediction)), "R2", r2_score(y_test, prediction))
    # Vérifier le nombre de termes et le scaler appris sur les seules observations train.
    poly = fitted.named_steps["poly"]
    assert poly.n_output_features_ == fitted.named_steps["poly"].degree
    train_features = poly.transform(X_train)
    assert np.allclose(fitted.named_steps["scaler"].mean_, train_features.mean(axis=0))
    learned_mean = fitted.named_steps["scaler"].mean_.copy()
    fitted.predict(np.array([[4.]]))  # Hors plage : calcul possible, fiabilité non garantie.
    assert np.array_equal(learned_mean, fitted.named_steps["scaler"].mean_)
    for p, d in [(2, 2), (10, 2), (10, 5)]:
        transformer = PolynomialFeatures(degree=d, include_bias=False).fit(np.zeros((2, p)))
        assert transformer.n_output_features_ == comb(p + d, d) - 1
    print("Comptage combinatoire et scaler limité au train : OK")
    if args.plot:
        import matplotlib.pyplot as plt
        grid = np.linspace(-3, 3, 300).reshape(-1, 1)
        plt.scatter(X_train[:, 0], y_train, alpha=.4, label="Train")
        plt.plot(grid[:, 0], fitted.predict(grid), color="orange", label="Modèle retenu par CV")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    print("Le degré générateur 3 n'est pas garanti de gagner en CV sur chaque échantillon bruité.")
