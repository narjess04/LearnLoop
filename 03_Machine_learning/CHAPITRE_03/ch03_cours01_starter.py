import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 100)
y = 3 + 2 * X + rng.normal(0, 1, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


def fit_ols(x, y):
    # TODO : vérifier les dimensions et refuser une variable x constante.
    # TODO : calculer les moyennes train.
    # TODO : pente = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)**2).
    # TODO : intercept = y_mean - pente*x_mean.
    raise NotImplementedError

# TODO : prédire sur test et calculer le MSE.
# TODO : comparer les coefficients avec sklearn.LinearRegression sur le même train.
# BONUS : implémenter la descente de gradient sur J = MSE / 2.
