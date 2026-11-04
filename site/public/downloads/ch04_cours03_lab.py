"""LearnLoop — Chapitre 4, cours 3. python ch04_cours03_lab.py [--plot]"""
import argparse
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run(plot=False):
    example = np.array([[1.], [2.], [6.], [8.]])
    Z_example = linkage(example, method="single", metric="euclidean")
    np.testing.assert_allclose(Z_example[:, 2], [1, 2, 4])
    assert len(np.unique(fcluster(Z_example, t=1.5, criterion="distance"))) == 3
    assert len(np.unique(fcluster(Z_example, t=2.5, criterion="distance"))) == 2
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    results = []
    for method in ("single", "complete", "average", "ward"):
        labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results.append((method, score))
        assert len(np.unique(labels)) == 3 and np.isfinite(score)
        print(f"{method:10s} | silhouette={score:.3f}")
    # Petit sous-échantillon lisible pour le dendrogramme.
    Z = linkage(X_scaled[:30], method="ward", metric="euclidean")
    assert Z.shape == (29, 4)
    print("Matrice Z, premières fusions :", Z[:5])
    print("Ward : hauteur SciPy = sqrt(2 * augmentation d'inertie).")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        dendrogram(Z_example, labels=list("ABCD"), ax=axes[0])
        axes[0].axhline(2.5, color="red", linestyle="--", label="Coupe à 2,5 : deux groupes")
        axes[0].legend()
        dendrogram(Z, ax=axes[1])
        axes[1].set(title="Ward : 30 observations", ylabel="Hauteur SciPy, pas Delta brute")
        fig.tight_layout()
        plt.show()
    return Z_example, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
