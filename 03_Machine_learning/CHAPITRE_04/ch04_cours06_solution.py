"""Benchmark blobs/moons/circles et stabilité sur intersections d'indices."""
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score
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

def subsample_stability(X, model):
    partitions = []
    for seed in range(3):
        ids = np.sort(np.random.default_rng(seed).choice(len(X), int(0.8 * len(X)), replace=False))
        labels = clone(model).fit_predict(X[ids])
        partitions.append((ids, labels))
    scores, coverages = [], []
    for (ids_a, labels_a), (ids_b, labels_b) in itertools.combinations(partitions, 2):
        common, pos_a, pos_b = np.intersect1d(ids_a, ids_b, return_indices=True)
        assert np.array_equal(ids_a[pos_a], ids_b[pos_b])
        a, b = labels_a[pos_a], labels_b[pos_b]
        scores.append(adjusted_rand_score(a, b))
        coverages.append(np.mean((a != -1) & (b != -1)))
    return float(np.mean(scores)), float(np.mean(coverages))

def run(plot=False):
    datasets = {
        "blobs": make_blobs(n_samples=400, centers=3, random_state=42),
        "moons": make_moons(n_samples=400, noise=0.07, random_state=42),
        "circles": make_circles(n_samples=400, noise=0.05, factor=0.45, random_state=42),
    }
    results = []
    for dataset, (raw, reference) in datasets.items():
        # Géométrie fixée pour cette exploration ; les labels ne guident pas le prétraitement.
        X = StandardScaler().fit_transform(raw)
        models = {}
        for k in (2, 3, 4):
            models[f"K-Means K={k}"] = KMeans(n_clusters=k, n_init=10, random_state=42)
            models[f"Ward K={k}"] = AgglomerativeClustering(n_clusters=k)
            models[f"GMM K={k}"] = GaussianMixture(n_components=k, n_init=3, reg_covar=1e-6, random_state=42)
        for eps in (0.15, 0.3, 0.5):
            models[f"DBSCAN eps={eps}"] = DBSCAN(eps=eps, min_samples=5)
        for name, model in models.items():
            labels = model.fit_predict(X)
            ari, common_coverage = subsample_stability(X, model)
            results.append({"dataset": dataset, "model": name, **evaluate_clustering(X, labels, reference),
                            "ARI_subsamples_all": ari, "common_non_noise_coverage": common_coverage})
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            for ax, name in zip(axes, ("K-Means K=3", "Ward K=3", "DBSCAN eps=0.3", "GMM K=3")):
                ax.scatter(*X.T, c=clone(models[name]).fit_predict(X), cmap="tab10", s=10)
                ax.set_title(name)
            fig.suptitle(dataset)
            fig.tight_layout()
            plt.show()
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    print("Pas de vainqueur automatique : géométrie, stabilité, couverture et utilité doivent être considérées ensemble.")
    print("L'ARI de rééchantillonnage compare uniquement les observations communes ; le bruit est inclus et sa couverture est rapportée.")
    assert len(table) == 36
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
