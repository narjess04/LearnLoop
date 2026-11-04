"""Mini-projet du cours 3 : segmentation synthétique, comparaison et profils."""
import argparse
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

def run(plot=False):
    rng = np.random.default_rng(42)
    X, _ = make_blobs(n_samples=240, centers=3, n_features=4, cluster_std=1.3, random_state=42)
    X = X * np.array([1.2, 1500, 0.3, 20]) + np.array([45, 50000, 8, 350])
    columns = ["age", "revenu_annuel", "frequence_achats", "panier_moyen"]
    df = pd.DataFrame(X, columns=columns)
    df.loc[rng.choice(len(df), 12, replace=False), "revenu_annuel"] = np.nan
    preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_scaled = preprocessing.fit_transform(df)
    results, labels_by_config = [], {}
    for method in ("single", "complete", "average", "ward"):
        for k in range(2, 7):
            labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            results.append({"method": method, "k": k, "silhouette": score, "min_cluster": np.bincount(labels).min()})
            labels_by_config[(method, k)] = labels
    summary = pd.DataFrame(results).sort_values("silhouette", ascending=False)
    best = summary.iloc[0]
    labels = labels_by_config[(best["method"], int(best["k"]))]
    print(summary.to_string(index=False))
    print("Profils dans les unités d'origine (NaN ignorés pour les moyennes) :")
    print(df.assign(cluster=labels).groupby("cluster").agg(["mean", "count"]))
    baseline = KMeans(n_clusters=int(best["k"]), n_init=10, random_state=42).fit_predict(X_scaled)
    print("Silhouette K-Means, même K et mêmes données :", silhouette_score(X_scaled, baseline))
    assert len(results) == 20 and np.isfinite(summary["silhouette"]).all()
    assert sum(np.bincount(labels)) == len(df)
    print("Analyse exploratoire uniquement : examiner tailles, stabilité et pertinence métier avant de recommander une segmentation.")
    print("AgglomerativeClustering n'a pas de predict natif pour de nouveaux clients.")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        dendrogram(linkage(X_scaled[:40], method="ward"), ax=axes[0])
        axes[0].set_title("Ward, 40 clients : vue illustrative")
        axes[1].scatter(df["revenu_annuel"], df["panier_moyen"], c=labels, cmap="viridis", s=15)
        axes[1].set(xlabel="Revenu annuel", ylabel="Panier moyen", title="Vue sur deux variables, pas tout l'espace")
        fig.tight_layout()
        plt.show()
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
