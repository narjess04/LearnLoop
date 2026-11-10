"""Audit pédagogique : groupes synthétiques indépendants, pas une preuve de biais réel.
Défaut = classe positive 1 dans tout le lab. --fairlearn active l'audit et l'atténuation locaux.
--xgboost ajoute un deuxième modèle local. Aucune décision n'est choisie sur le test.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def group_metrics(y_true, y_pred, groups):
    y_true, y_pred, groups = (np.asarray(v) for v in (y_true, y_pred, groups))
    if y_true.ndim != 1 or not len(y_true) or y_pred.shape != y_true.shape or groups.shape != y_true.shape:
        raise ValueError("Trois vecteurs alignés non vides sont requis.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all() or pd.isna(groups).any():
        raise ValueError("Cibles et prédictions binaires, groupes non manquants requis.")
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        truth, pred = y_true[mask], y_pred[mask]
        tp = int(((truth == 1) & (pred == 1)).sum())
        fp = int(((truth == 0) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        tn = int(((truth == 0) & (pred == 0)).sum())
        def ratio(a, b):
            return a / b if b else np.nan
        rows.append({"group": group, "n": len(truth), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "prevalence": truth.mean(), "selection_rate": pred.mean(),
            "TPR": ratio(tp, tp+fn), "FPR": ratio(fp, fp+tn),
            "FNR": ratio(fn, tp+fn), "precision": ratio(tp, tp+fp),
            "accuracy": float((truth == pred).mean())})
    return pd.DataFrame(rows).set_index("group")

def disparity(report, metric):
    values = report[metric]
    if len(values) < 2 or values.isna().any():
        return np.nan
    return float(values.max() - values.min())

def run(include_fairlearn=False, include_xgboost=False):
    X, y = make_classification(n_samples=1500, n_features=10,
        n_informative=6, n_redundant=2, random_state=42)
    groups = np.random.default_rng(42).choice(["A", "B"], size=len(y))
    train_valid, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    train, valid = train_test_split(train_valid, test_size=.25,
        stratify=y[train_valid], random_state=7)
    estimators = {"Logistique": LogisticRegression(max_iter=1000)}
    if include_xgboost:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(n_estimators=70, max_depth=3, random_state=42,
                                              eval_metric="logloss", n_jobs=1)
    reports = {}
    for name, estimator in estimators.items():
        model = make_pipeline(StandardScaler(), estimator).fit(X[train], y[train])
        prediction = model.predict(X[test])
        report = group_metrics(y[test], prediction, groups[test])
        reports[name] = report
        print("\n", name, "\n", report.to_string())
        print("DP difference :", disparity(report, "selection_rate"))
        print("Equal opportunity difference :", disparity(report, "TPR"))
        eo = max(disparity(report, "TPR"), disparity(report, "FPR"))
        print("Equalized odds difference :", eo)
        print("Les NaN représentent des taux non mesurables, pas zéro.")
        if include_fairlearn:
            from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
            from fairlearn.postprocessing import ThresholdOptimizer
            frame = MetricFrame(metrics={"accuracy": accuracy_score},
                y_true=y[test], y_pred=prediction, sensitive_features=groups[test])
            print(frame.by_group)
            assert np.isclose(demographic_parity_difference(y[test], prediction,
                sensitive_features=groups[test]), disparity(report, "selection_rate"))
            assert np.isclose(equalized_odds_difference(y[test], prediction,
                sensitive_features=groups[test]), eo)
            # Contraintes ajustées sur validation uniquement ; accès au groupe requis à l'inférence.
            mitigator = ThresholdOptimizer(estimator=model, constraints="equalized_odds",
                objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
            mitigator.fit(X[valid], y[valid], sensitive_features=groups[valid])
            mitigated = mitigator.predict(X[test], sensitive_features=groups[test], random_state=42)
            print("Après post-processing (audit test, sans garantie de parité exacte hors validation) :")
            print(group_metrics(y[test], mitigated, groups[test]).to_string())
            print("Accuracy avant/après :", accuracy_score(y[test], prediction), accuracy_score(y[test], mitigated))
    print("\nMODEL CARD — démonstration synthétique, non destinée à décider de crédits.")
    print("Usage : apprendre un audit. Groupe A/B tiré aléatoirement, absent des features.")
    print("Limites : petits groupes, labels artificiels, pas de validation de discrimination réelle.")
    print("À compléter : contexte, dommages possibles, intersections, incertitude, recours, responsable et monitoring.")
    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairlearn", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    args = parser.parse_args()
    run(args.fairlearn, args.xgboost)
