"""LearnLoop ch04 cours05 : python ch04_cours05_lab.py [--plot]."""
import argparse
import numpy as np

class GaussianMixture1D:
    """EM univarié pédagogique en log-espace ; plancher de variance explicite."""
    def __init__(self, n_components=2, max_iter=200, tol=1e-5, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim not in (1, 2) or (X.ndim == 2 and X.shape[1] != 1):
            raise ValueError("Un vecteur univarié est requis.")
        X = X.ravel()
        if not X.size or not np.isfinite(X).all():
            raise ValueError("Des valeurs finies non vides sont requises.")
        return X

    def _expectation(self, X):
        log_joint = np.log(self.weights_)[None, :] - 0.5 * (
            np.log(2 * np.pi * self.vars_)[None, :]
            + (X[:, None] - self.means_[None, :]) ** 2 / self.vars_[None, :]
        )
        maximum = log_joint.max(axis=1, keepdims=True)
        log_norm = maximum + np.log(np.exp(log_joint - maximum).sum(axis=1, keepdims=True))
        return np.exp(log_joint - log_norm), float(log_norm.sum())

    def fit(self, X):
        X = self._input(X)
        n, K = len(X), self.n_components
        if not 1 <= K <= n or self.max_iter < 1 or self.tol < 0:
            raise ValueError("Vérifier n_components, max_iter et tol.")
        rng = np.random.default_rng(self.random_state)
        self.means_ = rng.choice(X, K, replace=False)
        self.vars_ = np.full(K, max(float(np.var(X)), 1e-6))
        self.weights_ = np.ones(K) / K
        responsibilities, previous_ll = self._expectation(X)
        self.history_ = [previous_ll]
        self.converged_ = False
        for iteration in range(self.max_iter):
            counts = np.maximum(responsibilities.sum(axis=0), np.finfo(float).eps)
            self.weights_ = counts / counts.sum()
            self.means_ = (responsibilities * X[:, None]).sum(axis=0) / counts
            self.vars_ = np.maximum(
                (responsibilities * (X[:, None] - self.means_) ** 2).sum(axis=0) / counts, 1e-6
            )
            responsibilities, ll = self._expectation(X)
            self.history_.append(ll)
            self.n_iter_ = iteration + 1
            if abs(ll - previous_ll) <= self.tol:
                self.converged_ = True
                break
            previous_ll = ll
        return self

    def predict_proba(self, X):
        if not hasattr(self, "means_"):
            raise ValueError("Appeler fit avant predict_proba.")
        return self._expectation(self._input(X))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def run(plot=False):
    rng = np.random.default_rng(42)
    sample = np.concatenate([rng.normal(-2, 0.7, 300), rng.normal(3, 1.2, 200)])
    manual = GaussianMixture1D(n_components=2).fit(sample)
    probabilities = manual.predict_proba(sample)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(np.sort(manual.means_), [-2, 3], atol=0.3)
    assert np.isclose(manual.weights_.sum(), 1)
    assert np.all(np.diff(manual.history_) >= -1e-7)
    print("EM NumPy : moyennes, variances, poids :", manual.means_, manual.vars_, manual.weights_)
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=[0.7, 1.2, 0.9], random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    model = GaussianMixture(n_components=3, covariance_type="full", n_init=10, reg_covar=1e-6, random_state=42).fit(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(X_scaled), probabilities.argmax(axis=1))
    assert model.converged_
    print("GMM sklearn :", model.weights_, "convergence :", model.converged_)
    print("Responsabilités :", probabilities[:5])
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(*X_scaled.T, c=model.predict(X_scaled), cmap="viridis", s=12)
        plt.scatter(*model.means_.T, marker="X", color="red", s=140)
        plt.title("Affectation argmax d'un GMM ; les responsabilités restent disponibles")
        plt.show()
    return manual, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
