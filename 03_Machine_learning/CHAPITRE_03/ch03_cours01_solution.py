import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def validate_xy(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("Deux vecteurs de même taille, au moins deux observations, sont requis")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Les données doivent être finies")
    return x, y


def fit_ols(x, y):
    x, y = validate_xy(x, y)
    centered_x = x - x.mean()
    denominator = centered_x @ centered_x
    if denominator == 0:
        raise ValueError("Pente non identifiable : toutes les valeurs de x sont identiques")
    slope = (centered_x @ (y - y.mean())) / denominator
    intercept = y.mean() - slope * x.mean()
    return float(intercept), float(slope)


def fit_gradient_descent(x, y, learning_rate=.1, n_steps=2000):
    x, y = validate_xy(x, y)
    scale = x.std(ddof=0)
    if scale == 0:
        raise ValueError("Variable x constante")
    # Normaliser le train pour améliorer le conditionnement de l'optimisation.
    z = (x - x.mean()) / scale
    intercept, slope = 0., 0.
    for _ in range(n_steps):
        residual = y - (intercept + slope * z)
        # Gradients de J = MSE / 2, calculés AVANT les deux mises à jour.
        grad_intercept = -residual.mean()
        grad_slope = -np.mean(z * residual)
        intercept -= learning_rate * grad_intercept
        slope -= learning_rate * grad_slope
    original_slope = slope / scale
    original_intercept = intercept - original_slope * x.mean()
    return float(original_intercept), float(original_slope)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = np.linspace(0, 10, 100)
    y = 3 + 2 * X + rng.normal(0, 1, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    intercept, slope = fit_ols(X_train, y_train)
    reference = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
    assert np.allclose([intercept, slope], [reference.intercept_, reference.coef_[0]])
    gradient_coefficients = fit_gradient_descent(X_train, y_train)
    assert np.allclose([intercept, slope], gradient_coefficients)
    predictions = intercept + slope * X_test
    print("OLS NumPy:", intercept, slope, "gradient:", gradient_coefficients)
    print("MSE test:", mean_squared_error(y_test, predictions))
    assert np.allclose(predictions, reference.predict(X_test.reshape(-1, 1)))
    for x_invalid, y_invalid in [([1., 1.], [2., 3.]), ([np.nan, 1.], [2., 3.]),
                                 ([1., 2.], [1.]), ([1.], [2.])]:
        try:
            fit_ols(x_invalid, y_invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Entrée non exploitable acceptée")
    print("Comparaison OLS / gradient / sklearn, prédictions et cas limites : OK")
