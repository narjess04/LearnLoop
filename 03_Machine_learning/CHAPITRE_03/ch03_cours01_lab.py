import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
n = 1000
surface = rng.uniform(40, 200, n)
rooms = rng.integers(1, 6, n)
building_age = rng.uniform(0, 40, n)
noise = rng.normal(0, 20, n)
price = 50 + 2.0 * surface + 10 * rooms - .6 * building_age + noise
df = pd.DataFrame({"surface": surface, "rooms": rooms,
                   "building_age": building_age, "price": price})
X, y = df.drop(columns="price"), df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les diagnostics graphiques")
    args = parser.parse_args()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    for name, estimator in [("Moyenne train", DummyRegressor(strategy="mean")),
                             ("Régression linéaire", pipeline)]:
        scores = -cross_val_score(estimator, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        print(name, "RMSE CV train:", scores, "mean:", scores.mean(), "std:", scores.std())
        assert np.isfinite(scores).all()
    # Modèle fixe, évaluation finale sans réglage à partir du test.
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Intercept:", model.intercept_)
    coefficients = pd.Series(model.coef_, index=X.columns)
    print("Coefficients dans les unités originales:", coefficients)
    mse = mean_squared_error(y_test, y_pred)
    print("Test final : MAE", mean_absolute_error(y_test, y_pred),
          "MSE", mse, "RMSE", np.sqrt(mse), "R2", r2_score(y_test, y_pred))
    # Le scaling ne change pas les prédictions OLS ici (matrice de plein rang).
    pipeline.fit(X_train, y_train)
    assert np.allclose(model.predict(X_test), pipeline.predict(X_test))
    # Vérifier la formulation matricielle avec un solveur stable, sans inversion explicite.
    design = np.column_stack([np.ones(len(X_train)), X_train.to_numpy()])
    beta, _, rank, _ = np.linalg.lstsq(design, y_train.to_numpy(), rcond=None)
    assert rank == design.shape[1]
    assert np.allclose(beta, np.r_[model.intercept_, model.coef_])
    if args.plot:
        import matplotlib.pyplot as plt
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(y_test, y_pred, alpha=.6)
        limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[0].plot(limits, limits, linestyle="--", color="black")
        axes[0].set(xlabel="Prix observé", ylabel="Prix prédit")
        axes[1].scatter(y_pred, residuals, alpha=.6)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(xlabel="Prix prédit", ylabel="Résidu y - y_pred")
        coefficients.sort_values().plot(kind="barh", ax=axes[2])
        axes[2].set_title("Coefficients : unités différentes")
        fig.tight_layout()
        plt.show()
    print("Données synthétiques, prix en milliers de DT. Les coefficients sont des associations.")
