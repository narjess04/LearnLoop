"""LearnLoop ch04 cours06 : python ch04_cours06_lab.py [--plot]."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
def evaluate_clustering(X, labels, y_true=None):
    """Internes hors bruit ; métriques externes globales et périmètres explicites."""
    import numpy as np
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),) or not len(X) or not np.isfinite(X).all():
        raise ValueError("Matrice finie et labels alignés requis.")
    mask = labels != -1
    k = len(np.unique(labels[mask]))
    result = {"clusters": k, "noise_ratio": float(np.mean(~mask)),
              "coverage": float(np.mean(mask)), "n_internal": int(mask.sum()),
              "silhouette": np.nan, "DBI": np.nan, "CHI": np.nan}
    if 2 <= k < int(mask.sum()):
        result.update(silhouette=float(silhouette_score(X[mask], labels[mask])),
                      DBI=float(davies_bouldin_score(X[mask], labels[mask])),
                      CHI=float(calinski_harabasz_score(X[mask], labels[mask])))
    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.shape != labels.shape:
            raise ValueError("La référence doit porter sur les mêmes observations.")
        result["ARI_all"] = float(adjusted_rand_score(y_true, labels))
        result["NMI_all"] = float(normalized_mutual_info_score(y_true, labels))
        result["ARI_non_noise"] = float(adjusted_rand_score(y_true[mask], labels[mask])) if mask.any() else np.nan
    return result

def initialization_stability(X, k=4):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    labels = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(X) for seed in range(10)]
    scores = [adjusted_rand_score(labels[i], labels[j]) for i in range(10) for j in range(i + 1, 10)]
    return {"ARI_mean": float(np.mean(scores)), "ARI_min": float(np.min(scores)), "comparisons": len(scores)}

def run(plot=False):
    X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=[0.7, 1.0, 0.8, 1.2], random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    models = {"K-Means": KMeans(n_clusters=4, n_init=10, random_state=42),
              "Ward": AgglomerativeClustering(n_clusters=4, linkage="ward"),
              "DBSCAN": DBSCAN(eps=0.35, min_samples=5),
              "GMM": GaussianMixture(n_components=4, n_init=10, reg_covar=1e-6, random_state=42)}
    rows, partitions = [], {}
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        partitions[name] = labels
        rows.append({"model": name, **evaluate_clustering(X_scaled, labels, y_true)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("Stabilité à l'initialisation, mêmes observations :", initialization_stability(X_scaled))
    assert adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0]) == 1
    assert normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0]) == 1
    # Les partitions invalides pour les métriques internes sont rapportées, pas notées artificiellement.
    for labels in (np.zeros(len(X_scaled), dtype=int), -np.ones(len(X_scaled), dtype=int), np.arange(len(X_scaled))):
        result = evaluate_clustering(X_scaled, labels)
        assert np.isnan(result["silhouette"])
    print("Références synthétiques : utilisées pour l'audit externe, pas pour ajuster les modèles.")
    print("Internes hors bruit ; ARI_all et NMI_all sur toutes les observations. Comparer avec la couverture.")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, (name, labels) in zip(axes, partitions.items()):
            ax.scatter(*X_scaled.T, c=labels, cmap="tab10", s=10)
            ax.set_title(name)
        fig.tight_layout()
        plt.show()
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
