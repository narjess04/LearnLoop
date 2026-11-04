"""Pipeline mixte, PCA/K-Means, anomalies, infàrence, PSI et sauvegarde vérifiée.
--plot pour les figures ; --output chemin.joblib pour conserver le bundle.
"""
import argparse
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

NUMERIC = ["age", "revenu", "nombre_achats"]
CATEGORICAL = ["canal_acquisition"]
FEATURES = NUMERIC + CATEGORICAL

def customers(seed=42, n=600):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 4, n)
    df = pd.DataFrame({"age": np.clip(23 + group * 12 + rng.normal(0, 5, n), 18, 90),
                       "revenu": np.maximum(10000, 23000 + group * 22000 + rng.normal(0, 7000, n)),
                       "nombre_achats": rng.poisson(4 + group * 9),
                       "canal_acquisition": rng.choice(["Web", "Mobile", "Magasin"], n)})
    df.loc[rng.choice(n, max(1, n // 30), replace=False), "revenu"] = np.nan
    return df

def preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])

def clustering_pipeline(k=4, seed=42):
    return Pipeline([("preprocessing", preprocessor()), ("pca", PCA(0.95, svd_solver="full")),
                     ("clustering", KMeans(n_clusters=k, n_init=10, random_state=seed))])

def calculate_psi(reference, current, epsilon=1e-6):
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if reference.ndim != 1 or reference.shape != current.shape or not len(reference):
        raise ValueError("Distributions 1D alignàes requises.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all() or np.any(reference < 0) or np.any(current < 0):
        raise ValueError("Valeurs finies et non nàgatives requises.")
    if reference.sum() <= 0 or current.sum() <= 0 or epsilon <= 0:
        raise ValueError("Chaque distribution doit avoir une masse positive ; epsilon > 0.")
    p, q = reference / reference.sum(), current / current.sum()
    p, q = p + epsilon, q + epsilon
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((p - q) * np.log(p / q)))

def proportions(labels, k):
    return np.bincount(labels, minlength=k) / len(labels)

def metrics(X, labels):
    k = len(np.unique(labels))
    if not 2 <= k < len(X):
        return {"silhouette": np.nan, "DBI": np.nan, "CHI": np.nan}
    return {"silhouette": silhouette_score(X, labels), "DBI": davies_bouldin_score(X, labels), "CHI": calinski_harabasz_score(X, labels)}

def save_bundle(bundle, destination, probe):
    joblib.dump(bundle, destination)
    # Recharger uniquement ce fichier que nous venons de créer.
    loaded = joblib.load(destination)
    assert np.array_equal(bundle["clustering"].predict(probe[FEATURES]), loaded["clustering"].predict(probe[FEATURES]))
    assert np.allclose(bundle["anomalies"].decision_function(probe[FEATURES]), loaded["anomalies"].decision_function(probe[FEATURES]))
    return loaded

def run(plot=False, output=None):
    train, incoming = customers(), customers(7, 200)
    pipeline = clustering_pipeline().fit(train[FEATURES])
    anomaly = Pipeline([("preprocessing", preprocessor()), ("model", IsolationForest(n_estimators=200, contamination=0.05, random_state=42))]).fit(train[FEATURES])
    labels, new_labels = pipeline.predict(train[FEATURES]), pipeline.predict(incoming[FEATURES])
    # L'anomalie reste signalàe sans supprimer automatiquement un petit segment légitime.
    scores = -anomaly.decision_function(incoming[FEATURES])
    full_space = pipeline.named_steps["preprocessing"].transform(incoming[FEATURES])
    print("Métriques incoming, espace prétraité complet :", metrics(full_space, new_labels))
    drift = calculate_psi(proportions(labels, 4), proportions(new_labels, 4))
    assert calculate_psi([0.5, 0.5], [0.5, 0.5]) == 0 and np.isfinite(drift)
    print("PSI des proportions, même modàle et mêmes identifiants de segments :", drift)
    print("Taux d'alertes incoming :", np.mean(scores > 0))
    probe = incoming.head(3).copy()
    probe.loc[probe.index[0], "canal_acquisition"] = "Nouveau canal"
    assert pipeline.predict(probe).shape == (3,)
    bundle = {"clustering": pipeline, "anomalies": anomaly, "features": FEATURES,
              "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
              "reference_proportions": proportions(labels, 4), "seed": 42}
    if output:
        save_bundle(bundle, Path(output), probe)
        print("Bundle conservé :", output)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(bundle, Path(temporary) / "segmentation.joblib", probe)
        print("Sauvegarde/rechargement vérifiés dans un répertoire temporaire.")
    print("Dérive = signal à examiner, pas preuve de dégradation. Données synthétiques ; API et dashboard restent à dàvelopper dans le projet final.")
    if plot:
        import matplotlib.pyplot as plt
        z = pipeline.named_steps["pca"].transform(full_space)
        plt.scatter(z[:, 0], z[:, 1], c=new_labels, cmap="tab10", s=15 + 20 * (scores > 0))
        plt.title("Nouveaux clients : segments et alertes agrandies")
        plt.show()
    return bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.plot, args.output)
