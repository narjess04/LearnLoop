"""LearnLoop ch04 cours04 : python ch04_cours04_lab.py [--plot]. sklearn >= 1.3."""
import argparse
import numpy as np
from collections import deque

class DBSCANScratch:
    """Version pédagogique euclidienne : distances et mémoire O(n²)."""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
            raise ValueError("Une matrice 2D finie non vide est requise.")
        if self.eps <= 0 or not isinstance(self.min_samples, (int, np.integer)) or self.min_samples < 1:
            raise ValueError("eps positif et min_samples entier >= 1 requis.")
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        neighbors = [np.flatnonzero(distances[i] <= self.eps) for i in range(n)]
        self.core_sample_indices_ = np.flatnonzero([len(ids) >= self.min_samples for ids in neighbors])
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if len(neighbors[i]) < self.min_samples:
                continue  # Bruit provisoire, réaffectable plus tard.
            labels[i] = cluster_id
            queue = deque(neighbors[i].tolist())
            scheduled = set(neighbors[i].tolist())
            while queue:
                q = queue.popleft()
                if not visited[q]:
                    visited[q] = True
                    if len(neighbors[q]) >= self.min_samples:
                        for neighbor in neighbors[q]:
                            neighbor = int(neighbor)
                            if neighbor not in scheduled:
                                queue.append(neighbor)
                                scheduled.add(neighbor)
                if labels[q] == -1:
                    labels[q] = cluster_id
            cluster_id += 1
        self.labels_ = labels
        return labels

from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
def density_metrics(X, labels):
    import numpy as np
    from sklearn.metrics import silhouette_score
    mask = labels != -1
    k = len(np.unique(labels[mask]))
    return {"clusters": k, "noise_ratio": float(np.mean(~mask)),
            "coverage": float(np.mean(mask)),
            "silhouette_non_noise": float(silhouette_score(X[mask], labels[mask])) if 2 <= k < int(mask.sum()) else np.nan}

def run(plot=False):
    X, _ = make_moons(n_samples=500, noise=0.07, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    model = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
    manual = DBSCANScratch(eps=0.3, min_samples=5)
    labels_manual = manual.fit_predict(X_scaled)
    assert adjusted_rand_score(model.labels_, labels_manual) == 1.0
    assert np.array_equal(np.flatnonzero(model.labels_ == -1), np.flatnonzero(labels_manual == -1))
    # Ce cas vérifie la réaffectation d'un bruit provisoire à un cluster.
    border_example = np.array([[0.0], [0.1], [0.2], [0.3]])
    np.testing.assert_array_equal(DBSCANScratch(eps=0.11, min_samples=3).fit_predict(border_example), np.zeros(4, dtype=int))
    assert np.all(DBSCANScratch(eps=0.01, min_samples=3).fit_predict(border_example) == -1)
    print("DBSCAN :", density_metrics(X_scaled, model.labels_))
    distances, _ = NearestNeighbors(n_neighbors=5).fit(X_scaled).kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])  # Le point lui-même est compté.
    X_varied, _ = make_blobs(n_samples=[250, 150, 100], centers=[[-4, 0], [0, 0], [4, 0]],
                             cluster_std=[0.3, 0.8, 1.4], random_state=42)
    X_varied = StandardScaler().fit_transform(X_varied)
    hdb = HDBSCAN(min_cluster_size=20, min_samples=10).fit(X_varied)
    print("HDBSCAN :", density_metrics(X_varied, hdb.labels_))
    print("Forces d'appartenance :", hdb.probabilities_[:10])
    assert np.isfinite(hdb.probabilities_).all()
    assert np.all((hdb.probabilities_ >= 0) & (hdb.probabilities_ <= 1))
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(*X_scaled.T, c=model.labels_, cmap="tab10", s=12)
        axes[0].set_title("DBSCAN sur moons")
        axes[1].plot(k_distances)
        axes[1].set(title="K-Distance (point inclus)", ylabel="Distance au 5e voisin")
        axes[2].scatter(*X_varied.T, c=hdb.labels_, cmap="tab10", s=12)
        axes[2].set_title("HDBSCAN, densités variables")
        fig.tight_layout()
        plt.show()
    return model, hdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
