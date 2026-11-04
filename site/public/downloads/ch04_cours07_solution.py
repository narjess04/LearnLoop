"""Compression de 30 variables clients synthétiques : seuils et clustering."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

def run(plot=False):
    latent, _ = make_blobs(n_samples=600, centers=4, n_features=5, random_state=42)
    rng = np.random.default_rng(42)
    X = latent @ rng.normal(size=(5, 30)) + rng.normal(scale=0.5, size=(600, 30))
    train, holdout = train_test_split(X, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(train)
    train, holdout = scaler.transform(train), scaler.transform(holdout)
    full = PCA(svd_solver="full").fit(train)
    rows = []
    baseline = KMeans(n_clusters=4, n_init=10, random_state=42).fit(train)
    print("Silhouette baseline, espace standardisé holdout :", silhouette_score(holdout, baseline.predict(holdout)))
    for threshold in (0.90, 0.95, 0.99):
        pca = PCA(threshold, svd_solver="full").fit(train)
        z_train, z_holdout = pca.transform(train), pca.transform(holdout)
        model = KMeans(n_clusters=4, n_init=10, random_state=42).fit(z_train)
        labels = model.predict(z_holdout)
        rows.append({"threshold": threshold, "components": pca.n_components_,
                     "MSE_holdout": np.mean((holdout - pca.inverse_transform(z_holdout)) ** 2),
                     "silhouette_common_space": silhouette_score(holdout, labels),
                     "silhouette_PCA_space": silhouette_score(z_holdout, labels)})
    table = pd.DataFrame(rows)
    assert np.all(np.diff(table.MSE_holdout) <= 1e-12)
    print(table.to_string(index=False))
    print("Coefficients des deux premières directions :")
    print(pd.DataFrame(full.components_[:2].T, columns=["PC1", "PC2"]).to_string())
    print("Comparer les silhouettes dans l'espace commun ; les scores PCA utilisent des géométries différentes.")
    print("Ces seuils sont exploràs sur holdout : ce jeu est une validation, pas un test final indépendant.")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(np.arange(1, 31), full.explained_variance_ratio_, marker=".")
        axes[1].plot(np.arange(1, 31), full.explained_variance_ratio_.cumsum(), marker=".")
        z = full.transform(holdout)
        axes[2].scatter(z[:, 0], z[:, 1], c=baseline.predict(holdout), s=12)
        axes[0].set_title("Variance expliquàe")
        axes[1].set_title("Variance cumulàe")
        axes[2].set_title("Projection holdout et clusters baseline")
        fig.tight_layout()
        plt.show()
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
