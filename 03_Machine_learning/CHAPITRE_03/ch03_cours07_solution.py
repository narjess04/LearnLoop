import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, brier_score_loss)

X, y = make_classification(n_samples=5000, n_features=25, n_informative=12, n_redundant=6,
                           weights=[.95, .05], flip_y=.01, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25, stratify=y_train, random_state=43)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30, help="Budget de recherche aléatoire (défaut 30)")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations doit être positif")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"auc": "roc_auc", "ap": "average_precision"}
    tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    tree_scores = cross_validate(tree, X_fit, y_fit, cv=cv, scoring=scoring)
    distribution = {"n_estimators": [100, 200, 300, 500], "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 5, 10],
                    "max_features": ["sqrt", .5, .8], "class_weight": [None, "balanced"]}
    search = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), distribution,
                                n_iter=args.iterations, scoring=scoring, refit="auc", cv=cv,
                                random_state=42, n_jobs=1)
    search.fit(X_fit, y_fit)
    print("Budget:", args.iterations, "paramètres forêt:", search.best_params_)
    print("ROC-AUC de sélection forêt:", search.best_score_, "arbre fixe CV:", tree_scores["test_auc"].mean())
    tree.fit(X_fit, y_fit)
    forest = search.best_estimator_
    importance = permutation_importance(forest, X_valid, y_valid, scoring="roc_auc",
                                        n_repeats=5, random_state=42, n_jobs=1)
    print(pd.DataFrame({"feature": np.arange(X.shape[1]), "MDI": forest.feature_importances_,
                         "permutation_valid": importance.importances_mean}).sort_values("permutation_valid", ascending=False).head(10))
    chosen = "Forêt" if search.best_score_ >= tree_scores["test_auc"].mean() else "Arbre"
    print("Famille figée par CV avant test:", chosen)
    for name, model in [("Arbre", tree), ("Forêt", forest)]:
        proba, pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)
        print(name, {"ROC_AUC": roc_auc_score(y_test, proba), "AP": average_precision_score(y_test, proba),
                     "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred), "Brier": brier_score_loss(y_test, proba)})
    assert np.isfinite(forest.predict_proba(X_valid)).all()
    assert np.allclose(forest.feature_importances_.sum(), 1.)
    assert len(forest.estimators_) == forest.n_estimators
    print("Forêt, probabilités et importances : OK ; pas de révision à partir des scores test.")
    print("Le meilleur score de recherche est un score de sélection ; une validation externe estime la procédure complète.")
