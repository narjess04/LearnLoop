"""Comparer paramètres, bruit, couverture et stabilité sur des clients synthétiques."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
def density_metrics(X, labels):
    import numpy as np
    from sklearn.metrics import silhouette_score
    mask = labels != -1
    k = len(np.unique(labels[mask]))
    return {"clusters": k, "noise_ratio": float(np.mean(~mask)),
            "coverage": float(np.mean(mask)),
            "silhouette_non_noise": float(silhouette_score(X[mask], labels[mask])) if 2 <= k < int(mask.sum()) else np.nan}

def run(plot=False):
    X, _ = make_blobs(n_samples=[160, 100, 60], centers=[[-4]*5, [0]*5, [4]*5],
                      cluster_std=[0.35, 0.8, 1.4], random_state=42)
    X = X * [2, 30, 10, 5, 0.02] + [20, 500, 200, 100, 0.1]
    X_scaled = StandardScaler().fit_transform(X)
    results, fitted = [], {}
    for eps in (0.15, 0.25, 0.4, 0.65):
        for min_samples in (5, 10):
            name = f"DBSCAN eps={eps}, min_samples={min_samples}"
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
            results.append({"model": name, **density_metrics(X_scaled, labels)})
            fitted[name] = labels
    for size in (10, 20, 35):
        name = f"HDBSCAN min_cluster_size={size}"
        labels = HDBSCAN(min_cluster_size=size, min_samples=10).fit_predict(X_scaled)
        results.append({"model": name, **density_metrics(X_scaled, labels)})
        fitted[name] = labels
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
    fitted["K-Means K=3"] = labels
    results.append({"model": "K-Means K=3", **density_metrics(X_scaled, labels)})
    a = DBSCAN(eps=0.4, min_samples=5).fit_predict(X_scaled)
    b = DBSCAN(eps=0.42, min_samples=5).fit_predict(X_scaled)
    print(pd.DataFrame(results).to_string(index=False))
    print("ARI sous légère perturbation de eps, mêmes observations (bruit inclus) :", adjusted_rand_score(a, b))
    print("Comparer aussi les observations regroupées et les changements bruit/cluster ; l'ARI global seul ne suffit pas.")
    if plot:
        import matplotlib.pyplot as plt
        projection = PCA(n_components=2).fit_transform(X_scaled)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, name in zip(axes, ("DBSCAN eps=0.4, min_samples=5", "HDBSCAN min_cluster_size=20", "K-Means K=3")):
            ax.scatter(*projection.T, c=fitted[name], cmap="tab10", s=14)
            ax.set_title(name, fontsize=9)
        fig.suptitle("PCA pour l'affichage ; modèles entraînés sur les cinq features")
        fig.tight_layout()
        plt.show()
    assert len(results) == 12
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
