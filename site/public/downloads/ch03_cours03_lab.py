import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(200, 1))
noise = rng.normal(0, 1.5, size=200)
y = 3 + 2 * X[:, 0] + .8 * X[:, 0] ** 2 + noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


def polynomial_model(degree):
    return Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("scaler", StandardScaler()), ("model", LinearRegression())])


def compare_degrees():
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for degree in range(1, 9):
        pipeline = polynomial_model(degree)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error", return_train_score=True)
        rmse_train = -scores["train_score"]
        rmse_valid = -scores["test_score"]
        assert np.isfinite(rmse_valid).all()
        rows.append({"degree": degree, "RMSE_train": rmse_train.mean(),
                     "RMSE_CV": rmse_valid.mean(), "std_RMSE_CV": rmse_valid.std()})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    degree = int(results.loc[results["RMSE_CV"].idxmin(), "degree"])
    model = polynomial_model(degree).fit(X_train, y_train)
    return model, degree, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Tracer les courbes et les erreurs CV")
    args = parser.parse_args()
    model, degree, results = compare_degrees()
    print("Degré choisi sur CV train avant le test:", degree)
    prediction = model.predict(X_test)
    print("Test final : MAE", mean_absolute_error(y_test, prediction),
          "RMSE", np.sqrt(mean_squared_error(y_test, prediction)), "R2", r2_score(y_test, prediction))
    scaler, regressor = model.named_steps["scaler"], model.named_steps["model"]
    original_coefficients = regressor.coef_ / scaler.scale_
    original_intercept = regressor.intercept_ - original_coefficients @ scaler.mean_
    print("Intercept dans la base polynomiale originale:", original_intercept)
    print("Coefficients:", original_coefficients)
    print("Features:", model.named_steps["poly"].get_feature_names_out(["x"]))
    if args.plot:
        import matplotlib.pyplot as plt
        grid = np.linspace(-3, 3, 300).reshape(-1, 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(X_train[:, 0], y_train, alpha=.5, label="Train")
        for d in [1, 2, 3, 8]:
            fitted = polynomial_model(d).fit(X_train, y_train)
            axes[0].plot(grid[:, 0], fitted.predict(grid), label=f"Degré {d}")
        axes[0].set(xlabel="x", ylabel="y", title="Ajustements réels sur train")
        axes[0].legend()
        axes[1].plot(results["degree"], results["RMSE_train"], marker="o", label="Train")
        axes[1].plot(results["degree"], results["RMSE_CV"], marker="o", label="Validation CV")
        axes[1].set(xlabel="Degré", ylabel="RMSE")
        axes[1].legend()
        fig.tight_layout()
        plt.show()
    print("La relation génératrice est quadratique, mais un échantillon fini ne garantit pas degree=2 en CV.")
