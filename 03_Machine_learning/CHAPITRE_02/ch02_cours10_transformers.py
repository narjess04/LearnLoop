import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileCapper(TransformerMixin, BaseEstimator):
    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Bornes de quantiles invalides")
        X = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        if not np.isfinite([self.lower_bounds_, self.upper_bounds_]).all():
            raise ValueError("Colonne entièrement manquante : imputer ou retirer explicitement")
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        X = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        if len(input_features) != self.n_features_in_:
            raise ValueError("Nombre de noms de features incompatible")
        return np.asarray(input_features, dtype=object)
