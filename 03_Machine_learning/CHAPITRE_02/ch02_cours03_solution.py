import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class IQRCapper(TransformerMixin, BaseEstimator):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        if not np.isfinite(self.factor) or self.factor < 0:
            raise ValueError("factor doit être fini et positif ou nul")
        X = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        if not np.isfinite([q1, q3]).all():
            raise ValueError("Imputer ou retirer les colonnes entièrement manquantes")
        self.lower_ = q1 - self.factor * (q3 - q1)
        self.upper_ = q3 + self.factor * (q3 - q1)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        X = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_, self.upper_)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LogisticRegression
    capper = IQRCapper().fit([[0., 4.], [1., 4.], [2., 4.], [3., 4.]])
    before = capper.upper_.copy()
    transformed = capper.transform([[1000., 9.]])
    assert np.array_equal(before, capper.upper_)  # transform ne réapprend pas les bornes
    assert transformed[0, 0] == before[0] and transformed[0, 1] == 4
    X, y = make_classification(n_samples=800, n_features=8, random_state=42)
    X[::40, 0] *= 30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, cap, scale in [("Sans capping", "passthrough", StandardScaler()),
                             ("IQR", IQRCapper(), StandardScaler()),
                             ("Robust", "passthrough", RobustScaler())]:
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("capper", cap), ("scaler", scale),
                             ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(name, "ROC-AUC:", scores.mean(), "+/-", scores.std())
    print("Test final réservé ; les seuils IQR sont ajustés dans chaque fold train.")
