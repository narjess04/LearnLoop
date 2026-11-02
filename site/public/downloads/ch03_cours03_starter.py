import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(300, 1))
y = 5 - 2 * X[:, 0] + 1.5 * X[:, 0] ** 2 - .3 * X[:, 0] ** 3 + rng.normal(0, 2, size=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# TODO 1 : baseline linéaire, puis PolynomialFeatures aux degrés 2, 3, 4, 5 et 8.
# TODO 2 : comparer les RMSE sur les mêmes folds train.
# TODO 3 : inclure PolynomialFeatures, StandardScaler et Ridge dans le pipeline.
# TODO 4 : optimiser conjointement degree et alpha sur train uniquement.
# TODO 5 : figer le choix, puis évaluer une fois sur test.
# BONUS : comparer la procédure entière avec une validation imbriquée sur train.
