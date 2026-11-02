import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class MyStandardScaler(TransformerMixin, BaseEstimator):
    """Exercice pour tableaux numériques finis, sans NaN ni matrices creuses."""
    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.scale_ = np.where(self.std_ == 0, 1., self.std_)
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_", "scale_"])
        X = validate_data(self, X, reset=False, dtype=float)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self, ["mean_", "scale_"])
        X = validate_data(self, X, reset=False, dtype=float)
        return X * self.scale_ + self.mean_

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    X = np.array([[1., 7.], [2., 7.], [3., 7.], [4., 7.]])
    custom = MyStandardScaler().fit(X)
    reference = StandardScaler().fit(X)
    Z = custom.transform(X)
    assert np.allclose(Z, reference.transform(X))
    assert np.allclose(custom.inverse_transform(Z), X)
    assert np.all(Z[:, 1] == 0)
    assert np.allclose(custom.transform([[100., 7.]]), reference.transform([[100., 7.]]))
    assert np.allclose(custom.mean_, [2.5, 7.])  # pas de refit sur nouvelles données
    try:
        clone(custom).transform(X)
    except NotFittedError:
        pass
    else:
        raise AssertionError("transform doit refuser un scaler non ajusté")
    try:
        custom.transform([[1., 2., 3.]])
    except ValueError:
        pass
    else:
        raise AssertionError("Une dimension incorrecte doit être refusée")
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = make_pipeline(MyStandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, X_train, y_train,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    print("Comparaison StandardScaler, inverse et cas limites : OK")
    print("Accuracy CV (train uniquement):", scores, scores.mean())
