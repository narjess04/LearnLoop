"""Comparer K=1..10 et quatre covariances ; examiner l'incertitude."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def run(plot=False):
    X, _ = make_blobs(n_samples=400, centers=3, cluster_std=[0.7, 1.2, 0.9], random_state=42)
    X = X @ np.array([[1.5, 0.7], [0.2, 0.8]])
    X_scaled = StandardScaler().fit_transform(X)
    results, models = [], {}
    for covariance in ("full", "tied", "diag", "spherical"):
        for k in range(1, 11):
            model = GaussianMixture(n_components=k, covariance_type=covariance, n_init=3,
                                    reg_covar=1e-6, max_iter=300, random_state=42).fit(X_scaled)
            results.append({"covariance": covariance, "k": k, "aic": model.aic(X_scaled),
                            "bic": model.bic(X_scaled), "converged": model.converged_})
            models[(covariance, k)] = model
    table = pd.DataFrame(results).sort_values("bic")
    converged = table[table["converged"]]
    if converged.empty:
        raise RuntimeError("Aucune configuration convergée ; revoir le budget et les données.")
    best = converged.iloc[0]
    model = models[(best["covariance"], int(best["k"]))]
    proba = model.predict_proba(X_scaled)
    uncertainty = 1 - proba.max(axis=1)
    labels = model.predict(X_scaled)
    baseline = KMeans(n_clusters=model.n_components, n_init=10, random_state=42).fit_predict(X_scaled)
    print(table.to_string(index=False))
    print("Configuration retenue par BIC :", best.to_dict())
    print("Cinq observations les plus incertaines :", np.argsort(uncertainty)[-5:])
    print("Responsabilités correspondantes :", proba[np.argsort(uncertainty)[-5:]])
    print("Accord avec K-Means à même K, pas une vérité terrain :", adjusted_rand_score(labels, baseline))
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert np.isfinite(table[["aic", "bic"]]).all().all()
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for covariance in ("full", "tied", "diag", "spherical"):
            subset = table[table["covariance"] == covariance].sort_values("k")
            axes[0].plot(subset["k"], subset["bic"], label=covariance)
        axes[0].set(title="BIC, même dataset et même prétraitement", xlabel="K")
        axes[0].legend()
        axes[1].scatter(*X_scaled.T, c=uncertainty, cmap="magma", s=15)
        axes[1].set_title("Incertitude 1 − responsabilité maximale")
        fig.tight_layout()
        plt.show()
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
