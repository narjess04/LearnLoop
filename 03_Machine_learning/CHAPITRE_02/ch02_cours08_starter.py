import numpy as np


def my_pca(X, n_components=2):
    X = np.asarray(X, dtype=float)
    # TODO 1 : centrer et conserver la moyenne train.
    # TODO 2 : covariance Xc.T @ Xc / (n - 1).
    # TODO 3 : valeurs/vecteurs propres avec np.linalg.eigh.
    # TODO 4 : trier en ordre décroissant, conserver k vecteurs.
    # TODO 5 : projeter, retourner aussi les paramètres pour transformer un test.
    raise NotImplementedError

# BONUS : comparer valeurs propres et sous-espace avec sklearn.PCA.
# Le signe des vecteurs propres n'est pas unique.
