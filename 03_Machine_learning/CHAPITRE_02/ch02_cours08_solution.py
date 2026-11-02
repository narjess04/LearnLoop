import numpy as np
from sklearn.decomposition import PCA


def my_pca(X, n_components=2):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or not np.isfinite(X).all():
        raise ValueError("Attendu : matrice finie, au moins deux observations")
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= min(X.shape):
        raise ValueError("Nombre de composantes invalide")
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("Variance totale nulle : ratio de variance indéfini")
    components = eigenvectors[:, order[:n_components]]
    return centered @ components, {"mean": mean, "components": components,
                                    "eigenvalues": eigenvalues[:n_components],
                                    "explained_variance_ratio": eigenvalues[:n_components] / total}


def transform_pca(X, fitted):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != fitted["mean"].shape[0] or not np.isfinite(X).all():
        raise ValueError("Dimensions incompatibles ou valeurs non finies")
    return (X - fitted["mean"]) @ fitted["components"]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5)) @ np.diag([5., 3., 2., 1., .5])
    train, test = X[:150], X[150:]
    Z, fitted = my_pca(train, 2)
    reference = PCA(n_components=2, svd_solver="full").fit(train)
    assert np.allclose(fitted["eigenvalues"], reference.explained_variance_)
    assert np.allclose(fitted["explained_variance_ratio"], reference.explained_variance_ratio_)
    # Comparer les projecteurs de sous-espace, invariants au signe des axes.
    W = fitted["components"]
    assert np.allclose(W @ W.T, reference.components_.T @ reference.components_)
    assert np.allclose(Z @ W.T, reference.transform(train) @ reference.components_)
    Z_test = transform_pca(test, fitted)
    assert Z_test.shape == (50, 2)
    assert np.allclose(Z_test @ W.T, reference.transform(test) @ reference.components_)
    for invalid in [np.zeros((3, 2)), np.array([[np.nan, 1], [2, 3]])]:
        try:
            my_pca(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Une entrée non exploitable doit être refusée")
    print("PCA NumPy : covariance, variance expliquée, sous-espace, test transform et cas limites OK")
