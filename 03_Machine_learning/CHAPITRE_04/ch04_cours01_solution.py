"""Comparer K=2, 3 et 4 sans utiliser les labels de génération."""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def run(plot=False):
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    results = []
    models = []
    for k in (2, 3, 4):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        results.append((k, model.inertia_, silhouette_score(X, model.labels_)))
        models.append(model)
        print(f"K={k} | inertie={results[-1][1]:.2f} | silhouette={results[-1][2]:.3f}")
    assert np.isfinite(np.array(results)).all()
    print("Ces critères ne prouvent pas une segmentation métier.")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, model in zip(axes, models):
            ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=15)
            ax.set_title(f"K={model.n_clusters}")
        fig.tight_layout()
        plt.show()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
