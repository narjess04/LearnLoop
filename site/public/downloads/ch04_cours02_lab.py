"""LearnLoop — Chapitre 4, cours 2. python ch04_cours02_lab.py [--plot]"""
import argparse
import numpy as np

class KMeansScratch:
    """Lloyd pédagogique : initialisation simple, centre vide conservé."""
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Une matrice 2D finie est requise.")
        if not 1 <= self.n_clusters <= len(X) or self.max_iter < 1 or self.tol < 0:
            raise ValueError("Vérifier K, max_iter et tol.")
        rng = np.random.default_rng(self.random_state)
        centers = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        def squared_distances(c):
            return ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        self.history_ = [float(squared_distances(centers).min(axis=1).sum())]
        for iteration in range(self.max_iter):
            labels = squared_distances(centers).argmin(axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points):
                    new_centers[k] = points.mean(axis=0)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            self.history_.append(float(squared_distances(centers).min(axis=1).sum()))
            self.n_iter_ = iteration + 1
            if shift <= self.tol:
                break
        self.cluster_centers_ = centers
        self.labels_ = squared_distances(centers).argmin(axis=1)
        self.inertia_ = float(((X - centers[self.labels_]) ** 2).sum())
        # Les labels sont recalculés après la dernière mise à jour.
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Appeler fit avant predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.cluster_centers_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Matrice finie et nombre de features compatibles requis.")
        return ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run(plot=False):
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
    manual = KMeansScratch(n_clusters=4).fit(X_scaled)
    assert np.all(np.diff(manual.history_) <= 1e-8)
    np.testing.assert_array_equal(manual.labels_, manual.predict(X_scaled))
    assert np.isclose(manual.inertia_, ((X_scaled-manual.cluster_centers_[manual.labels_])**2).sum())
    assert len(np.unique(model.labels_)) == 4
    print("Centroïdes standardisés :", model.cluster_centers_)
    print("Centroïdes en unités d'origine :", scaler.inverse_transform(model.cluster_centers_))
    print("Inertie sklearn :", model.inertia_)
    print("Inertie NumPy, un seul démarrage :", manual.inertia_)
    print("Silhouette :", silhouette_score(X_scaled, model.labels_))
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(*X_scaled.T, c=model.labels_, s=15, cmap="viridis")
        plt.scatter(*model.cluster_centers_.T, marker="X", c="red", s=150)
        plt.title("K-Means après standardisation")
        plt.show()
    return model, manual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
