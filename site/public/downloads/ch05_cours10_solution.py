"""MVP de gouvernance pédagogique : évaluation, règles configurables et Model Card.
Le verdict est un triage technique, jamais une approbation de production.
"""
import math
import json
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, brier_score_loss, log_loss)

class ClassificationEvaluator:
    def __init__(self, threshold=.5):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Seuil dans [0,1] requis.")
        self.threshold = threshold

    def evaluate(self, y_true, y_proba):
        y, p = np.asarray(y_true), np.asarray(y_proba, dtype=float)
        if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("Cibles binaires et probabilités alignées non vides requises.")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilités finies dans [0,1] requises.")
        pred = (p >= self.threshold).astype(int)
        has_two_classes = len(np.unique(y)) == 2
        return {"n": int(len(y)), "prevalence": float(y.mean()), "threshold": self.threshold,
            "auc": float(roc_auc_score(y, p)) if has_two_classes else None,
            "average_precision": float(average_precision_score(y, p)) if has_two_classes else None,
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}

DEFAULT_RULES = {"auc": (.80, .70), "brier": (.05, .08), "psi": (.10, .25)}

def evaluate_model(metrics, psi, rules=None, critical_findings=None):
    rules = DEFAULT_RULES if rules is None else rules
    statuses, findings = {}, []
    for name, value in {"auc": metrics.get("auc"), "brier": metrics.get("brier"), "psi": psi}.items():
        if name not in rules or len(rules[name]) != 2:
            raise ValueError("Deux seuils requis pour " + name)
        green, amber = rules[name]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (green, amber)):
            raise ValueError("Seuils finis requis.")
        if (name == "auc" and green < amber) or (name != "auc" and green > amber):
            raise ValueError("Ordre des seuils incohérent.")
        valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
        valid = valid and math.isfinite(float(value)) and value >= 0 and (name == "psi" or value <= 1)
        if not valid:
            statuses[name] = "RED"
            findings.append(name + " absent, non défini ou invalide")
            continue
        if name == "auc":
            status = "GREEN" if value >= green else "AMBER" if value >= amber else "RED"
        else:
            status = "GREEN" if value <= green else "AMBER" if value <= amber else "RED"
        statuses[name] = status
        if status != "GREEN":
            findings.append(name + " : " + status + " selon les seuils pédagogiques")
    if critical_findings:
        statuses["critical_controls"] = "RED"
        findings.extend(str(v) for v in critical_findings)
    rank = {"GREEN": 0, "AMBER": 1, "RED": 2}
    verdict = max(statuses.values(), key=rank.get)
    return {"status": verdict, "criteria": statuses, "findings": findings, "requires_human_review": True}

def run():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from ch05_cours08_solution import calculate_psi
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5, random_state=42)
    train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    metrics = ClassificationEvaluator(threshold=.4).evaluate(y[test], model.predict_proba(X[test])[:, 1])
    psi = calculate_psi(X[train, 0], X[test, 0])
    verdict = evaluate_model(metrics, psi)
    card = {"model": "logistique-demo-v1", "dataset": "synthetique-seed42",
        "usage": "Démonstration pédagogique, sans usage de décision de crédit",
        "metrics": metrics, "psi_feature_0": psi, "triage": verdict,
        "not_evaluated": ["SHAP", "robustesse", "fairness", "incertitude conforme", "revue indépendante"],
        "approval": "NON APPROUVÉ POUR LA PRODUCTION",
        "next_steps": ["Compléter les contrôles pertinents", "Revue humaine", "Versionner les artefacts et décisions"]}
    print(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False))
    return card

if __name__ == "__main__":
    run()
