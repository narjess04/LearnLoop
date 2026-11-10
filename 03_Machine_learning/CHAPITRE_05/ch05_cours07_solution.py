"""Diagnostics sur validation, puis audit final indépendant. --plot et --curves en option.
Les segments et les données sont synthétiques, sans interprétation causale.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def individual_log_loss(y_true, proba):
    y, p = np.asarray(y_true), np.asarray(proba, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Cibles binaires et probabilités alignées non vides requises.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilités finies dans [0,1] requises.")
    p = np.clip(p, 1e-15, 1-1e-15)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

def segment_report(y_true, proba, segments, threshold=.5):
    losses = individual_log_loss(y_true, proba)
    y, p, segments = np.asarray(y_true), np.asarray(proba), np.asarray(segments)
    if segments.shape != y.shape or pd.isna(segments).any() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Segments alignés non manquants et seuil dans [0,1] requis.")
    pred = (p >= threshold).astype(int)
    frame = pd.DataFrame({"segment": segments, "truth": y, "prediction": pred, "loss": losses})
    rows = []
    for name, group in frame.groupby("segment", observed=True):
        truth, predicted = group["truth"].to_numpy(), group["prediction"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        rows.append({"segment": name, "n": len(group), "prevalence": truth.mean(),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "error_rate": float((truth != predicted).mean()), "log_loss": group["loss"].mean(),
            "precision": tp/(tp+fp) if tp+fp else np.nan,
            "recall": tp/(tp+fn) if tp+fn else np.nan})
    return pd.DataFrame(rows).set_index("segment")

def run(plot=False, curves=False):
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    dev, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    train, valid = train_test_split(dev, test_size=.25, stratify=y[dev], random_state=7)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    p = model.predict_proba(X[valid])[:, 1]
    # Bornes apprises sur train. Les doublons sont retirés sans labels qcut incompatibles.
    edges = np.unique(np.quantile(X[train, 0], [1/3, 2/3]))
    segments = pd.cut(X[valid, 0], bins=np.r_[-np.inf, edges, np.inf]).astype(str)
    report = segment_report(y[valid], p, segments)
    print("Diagnostic sur VALIDATION :\n", report.to_string())
    hard = pd.DataFrame({"row_id": valid, "y_true": y[valid], "probability": p,
                         "log_loss": individual_log_loss(y[valid], p)}).sort_values("log_loss", ascending=False).head(10)
    print("Cas difficiles à investiguer, sans suppression automatique :\n", hard.to_string(index=False))
    if curves:
        sizes, train_scores, valid_scores = learning_curve(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X[train], y[train],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            train_sizes=[.3, .6, 1.], scoring="roc_auc", n_jobs=1)
        print(pd.DataFrame({"n": sizes, "train_auc": train_scores.mean(axis=1),
                            "cv_auc": valid_scores.mean(axis=1)}).to_string(index=False))
    # L'audit test n'est pas réutilisé pour choisir seuil ou features.
    test_p = model.predict_proba(X[test])[:, 1]
    print("Audit TEST : AUC =", roc_auc_score(y[test], test_p),
          "accuracy =", accuracy_score(y[test], test_p >= .5))
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(individual_log_loss(y[valid], p), bins=20)
        plt.xlabel("Log loss individuelle sur validation")
        plt.show()
        Xr, yr = make_regression(n_samples=300, n_features=3, noise=20, random_state=42)
        a, b = train_test_split(np.arange(len(yr)), test_size=.3, random_state=42)
        predicted = LinearRegression().fit(Xr[a], yr[a]).predict(Xr[b])
        plt.scatter(predicted, yr[b]-predicted, s=10)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Prédiction"); plt.ylabel("Résidu y − prédiction")
        plt.show()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--curves", action="store_true")
    args = parser.parse_args()
    run(args.plot, args.curves)
