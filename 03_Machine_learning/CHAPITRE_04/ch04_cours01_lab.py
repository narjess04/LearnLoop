"""LearnLoop — Chapitre 4, cours 1. python ch04_cours01_lab.py [--plot]"""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def run(plot=False):
    # Les étiquettes de génération sont volontairement ignorées.
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = model.fit_predict(X)
    assert X.shape == (300, 2)
    assert np.unique(clusters).size == 3
    assert model.cluster_centers_.shape == (3, 2)
    assert np.isfinite(model.inertia_)
    print("Effectifs :", np.bincount(clusters))
    print("Inertie :", model.inertia_)
    print("Silhouette :", silhouette_score(X, clusters))
    if plot:
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis", alpha=0.7)
        plt.scatter(*model.cluster_centers_.T, marker="X", s=200, c="red", label="Centroïdes")
        plt.title("K-Means : données synthétiques, groupes découverts sans y")
        plt.legend()
        plt.show()
    return X, model, clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
