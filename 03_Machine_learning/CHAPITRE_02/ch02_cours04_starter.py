import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MyStandardScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # TODO : moyenne et écart-type par colonne (ddof=0).
        # TODO : remplacer l'échelle des colonnes constantes par 1.
        return self

    def transform(self, X):
        # TODO : vérifier le fit, les dimensions et standardiser.
        raise NotImplementedError

    def inverse_transform(self, X):
        # BONUS : retrouver les valeurs originales.
        raise NotImplementedError

# TODO : comparer à StandardScaler avec np.allclose, puis utiliser un Pipeline.
