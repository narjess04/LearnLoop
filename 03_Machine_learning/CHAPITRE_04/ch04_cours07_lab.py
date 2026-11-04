"""PCA : NumPy, SVD, reconstruction et pipeline sans fuite. --plot pour les figures."""
import argparse
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class PCAScratch:
    """Version pédagogique par covariance, pour matrices denses finies."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError("Matrice 2D finie, au moins deux observations requises.")
        if not isinstance(self.n_components, (int, np.integer)) or not 1 <= self.n_components <= min(X.shape):
            raise ValueError("Nombre entier de composantes entre 1 et min(n, p).")
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        covariance = centered.T @ centered / (len(X) - 1)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0), vectors[:, order]
        if values.sum() <= 0:
            raise ValueError("La variance totale est nulle.")
        self.components_ = vectors[:, :self.n_components].T
        self.explained_variance_ = values[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / values.sum()
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("Appeler fit avant transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.mean_) or not np.isfinite(X).all():
            raise ValueError("Dimensions ou valeurs incompatibles.")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        if not hasattr(self, "components_"):
            raise ValueError("Appeler fit avant inverse_transform.")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.n_components or not np.isfinite(Z).all():
            raise ValueError("Projection incompatible.")
        return Z @ self.components_ + self.mean_

def run(plot=False):
    iris = load_iris()
    scaled = StandardScaler().fit_transform(iris.data)
    scratch = PCAScratch(2).fit(scaled)
    reference = PCA(2, svd_solver="full").fit(scaled)
    Z = scratch.transform(scaled)
    # Comparer les projecteurs : les signes des vecteurs propres sont arbitraires.
    assert np.allclose(scratch.components_.T @ scratch.components_, reference.components_.T @ reference.components_)
    assert np.allclose(scratch.explained_variance_ratio_, reference.explained_variance_ratio_)
    singular = np.linalg.svd(scaled, compute_uv=False)
    assert np.allclose(scratch.explained_variance_, singular[:2] ** 2 / (len(scaled) - 1))
    errors = []
    for k in range(1, 5):
        model = PCAScratch(k).fit(scaled)
        errors.append(np.mean((scaled - model.inverse_transform(model.transform(scaled))) ** 2))
    assert np.all(np.diff(errors) <= 1e-12) and errors[-1] < 1e-20
    print("Variance expliquàe :", scratch.explained_variance_ratio_)
    print("MSE de reconstruction pour k=1..4 :", errors)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, test_size=0.2, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(0.95, svd_solver="full")), ("model", LogisticRegression(max_iter=2000))])
    pipeline.fit(X_train, y_train)
    assert pipeline.named_steps["scaler"].n_samples_seen_ == len(X_train)
    print("Accuracy test (exemple pédagogique, choix fixà à l'avance) :", accuracy_score(y_test, pipeline.predict(X_test)))
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(*Z.T, c=iris.target, cmap="viridis", s=14)
        axes[0].set_title("Iris : labels utilisàs uniquement pour la couleur")
        axes[1].plot(range(1, 5), errors, marker="o")
        axes[1].set(xlabel="Composantes", ylabel="MSE standardisée")
        fig.tight_layout()
        plt.show()
    return scratch, errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
