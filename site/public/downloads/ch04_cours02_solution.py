"""Challenge : comparer K=2..10 sur les mêmes données et la même géométrie."""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def compare(X, title, plot=False):
    results, models = [], []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = silhouette_score(X, model.labels_)
        results.append((k, model.inertia_, score))
        models.append(model)
        print(f"{title} | K={k:2d} | inertie={model.inertia_:.3f} | silhouette={score:.3f}")
    best_index = int(np.argmax([r[2] for r in results]))
    best = models[best_index]
    assert len(results) == 9 and np.isfinite(np.array(results)).all()
    print(f"{title} : meilleur K selon la silhouette observée = {best.n_clusters}")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot([r[0] for r in results], [r[1] for r in results], "o-")
        axes[0].set(title="Inertie", xlabel="K")
        axes[1].plot([r[0] for r in results], [r[2] for r in results], "o-")
        axes[1].set(title="Silhouette", xlabel="K")
        axes[2].scatter(*X[:, :2].T, c=best.labels_, cmap="viridis", s=12)
        axes[2].set_title(f"{title} : K={best.n_clusters}")
        fig.tight_layout()
        plt.show()
    return results

def run(plot=False):
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)
    compare(StandardScaler().fit_transform(X), "Standardisé", plot)
    # Bonus : transformer artificiellement l'unité d'une feature.
    X_different_units = X.copy()
    X_different_units[:, 1] *= 1000
    compare(X_different_units, "Sans scaling, unités différentes", plot)
    compare(StandardScaler().fit_transform(X_different_units), "Avec scaling, unités différentes", plot)
    print("Comparer les partitions et les interprétations, pas l'inertie entre espaces différemment mis à l'échelle.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
