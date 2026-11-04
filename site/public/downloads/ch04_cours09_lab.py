"""Détection inductive : 3 modàles, train/validation/test distincts, scores orientés anomalie."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def transactions(seed, normal=500, anomalies=40):
    rng = np.random.default_rng(seed)
    # Six variables abstraites standardisées ; ce gànàrateur ne représente pas une banque ràelle.
    baseline = rng.normal(size=(normal, 6))
    unusual = rng.normal(loc=2.5, scale=1.5, size=(anomalies, 6))
    X = np.vstack([baseline, unusual])
    y = np.r_[np.zeros(normal, dtype=int), np.ones(anomalies, dtype=int)]
    order = rng.permutation(len(X))
    return X[order], y[order]

def models():
    return {
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.06, random_state=42, n_jobs=1),
        "LOF novelty": make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.06)),
        "One-Class SVM": make_pipeline(StandardScaler(), OneClassSVM(nu=0.05, gamma="scale")),
    }

def evaluate(model, X, y, threshold=0.0):
    scores = -model.decision_function(X)
    predictions = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "F1": f1,
            "AP": average_precision_score(y, scores), "alert_rate": predictions.mean()}, scores

def run(plot=False):
    train, _ = transactions(42, normal=500, anomalies=0)
    test, y_test = transactions(7)
    fitted = models()
    rows = []
    for name, model in fitted.items():
        model.fit(train)
        result, scores = evaluate(model, test, y_test)
        assert np.isfinite(scores).all()
        assert set(np.unique(model.predict(test))) <= {-1, 1}
        rows.append({"model": name, **result})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("Benchmark novelty : référence train normale connue par construction ; test neuf. Paramàtres fixàs à l'avance.")
    print("Les labels synthétiques évaluent uniquement les ràsultats ; AP n'est pas l'intégrale trapàzoàdale PR.")
    if plot:
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, model in fitted.items():
            scores = -model.decision_function(test)
            PrecisionRecallDisplay.from_predictions(y_test, scores, name=name, ax=axes[0])
        from sklearn.decomposition import PCA
        scaler = StandardScaler().fit(train)
        pca = PCA(2).fit(scaler.transform(train))
        z = pca.transform(scaler.transform(test))
        axes[1].scatter(*z.T, c=-fitted["Isolation Forest"].decision_function(test), cmap="magma", s=12)
        axes[1].set_title("Projection PCA pour visualiser le score, pas pour l'ajuster")
        fig.tight_layout()
        plt.show()
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
