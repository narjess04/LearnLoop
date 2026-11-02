import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRCapper(TransformerMixin, BaseEstimator):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        # TODO : vérifier le fit et appliquer np.clip aux bornes apprises.
        raise NotImplementedError

# TODO : Pipeline(SimpleImputer, IQRCapper, StandardScaler, LogisticRegression).
# BONUS : comparer sans capping et avec RobustScaler, sur les mêmes folds train.
