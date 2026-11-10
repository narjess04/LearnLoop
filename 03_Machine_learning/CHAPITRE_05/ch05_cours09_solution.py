"""Bootstrap apparié, McNemar exact, Holm et validation chronologique.
Hypothèse bootstrap : observations i.i.d. Les dépendances demandent d'autres rééchantillonnages.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def _validate(y, *probabilities):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Deux classes binaires et un vecteur non vide sont requis.")
    arrays = []
    for values in probabilities:
        p = np.asarray(values, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilités finies alignées dans [0,1] requises.")
        arrays.append(p)
    return (y, *arrays)

def compare_auc_bootstrap(y_true, proba_a, proba_b, n_bootstrap=300, seed=42):
    y, a, b = _validate(y_true, proba_a, proba_b)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("Au moins deux réplications sont requises.")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        differences.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    if len(differences) < 2:
        raise ValueError("Trop peu de réplications avec deux classes.")
    lower, upper = np.quantile(differences, [.025, .975])
    return {"difference": float(roc_auc_score(y, a)-roc_auc_score(y, b)),
            "lower": float(lower), "upper": float(upper),
            "n_valid": len(differences), "samples": np.asarray(differences)}

def bootstrap_auc(y_true, proba, n_bootstrap=300, seed=42):
    y, p = _validate(y_true, proba)
    # Une prédiction constante a une AUC de 0.5 sur chaque réplication valide.
    result = compare_auc_bootstrap(y, p, np.full(len(y), .5), n_bootstrap, seed)
    return {"auc": float(roc_auc_score(y, p)),
            "lower": result["lower"]+.5, "upper": result["upper"]+.5,
            "n_valid": result["n_valid"]}

def mcnemar_exact(y_true, pred_a, pred_b):
    y, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    if y.ndim != 1 or not len(y) or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("Vecteurs alignés non vides requis.")
    if not all(np.isin(v, [0, 1]).all() for v in (y, a, b)):
        raise ValueError("Labels et prédictions binaires requis.")
    ca, cb = a == y, b == y
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = float(binomtest(n10, n10+n01, .5).pvalue) if n10+n01 else 1.
    return {"n10": n10, "n01": n01, "pvalue": p}

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Un vecteur non vide de p-values dans [0,1] est requis.")
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    result = np.empty(len(p))
    result[order] = adjusted
    return result

def run(plot=False):
    X, y = make_classification(n_samples=750, n_features=8, n_informative=5, random_state=42)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=42)
    a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    b = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1).fit(X[train], y[train])
    pa, pb = a.predict_proba(X[test])[:, 1], b.predict_proba(X[test])[:, 1]
    print("IC AUC logistique :", bootstrap_auc(y[test], pa))
    comparison = compare_auc_bootstrap(y[test], pa, pb)
    print("Différence AUC logistique − forêt :", {k:v for k,v in comparison.items() if k!="samples"})
    print("McNemar :", mcnemar_exact(y[test], pa>=.5, pb>=.5))
    print("Holm :", holm_adjust([.001, .012, .032, .048, .18]))
    # Démonstration chronologique séparée, générée dans l'ordre du temps.
    rng = np.random.default_rng(7)
    Xt = rng.normal(size=(500, 4))
    yt = (Xt[:, 0] + .3*np.sin(np.arange(500)/50) + rng.normal(size=500) > 0).astype(int)
    rows = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=4, gap=5).split(Xt), 1):
        assert tr.max() < te.min()-5
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xt[tr], yt[tr])
        auc = roc_auc_score(yt[te], model.predict_proba(Xt[te])[:, 1]) if len(np.unique(yt[te]))==2 else np.nan
        rows.append({"fold": fold, "train_end": int(tr.max()), "test_start": int(te.min()), "AUC": auc})
    print("Backtesting chronologique synthétique :\n", pd.DataFrame(rows).to_string(index=False))
    print("Un gap doit correspondre à l'horizon réel des labels. L'IC bootstrap n'inclut pas l'incertitude du réentraînement.")
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(comparison["samples"], bins=25)
        plt.axvline(0, color="red")
        plt.xlabel("Différence AUC appariée (logistique − forêt)")
        plt.show()
    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
