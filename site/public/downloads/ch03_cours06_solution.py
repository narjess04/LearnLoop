import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss)
from ch03_cours06_lab import X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                                   stratify=y_train, random_state=43)
    grid = {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10, 20],
            "criterion": ["gini", "entropy"], "ccp_alpha": [0., .001, .01]}
    search = GridSearchCV(DecisionTreeClassifier(random_state=42), grid,
                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                          scoring={"auc": "roc_auc", "ap": "average_precision"}, refit="auc")
    search.fit(X_fit, y_fit)
    model = search.best_estimator_
    print("Paramètres figés:", search.best_params_, "AUC de sélection:", search.best_score_)
    print("Profondeur et feuilles réelles:", model.get_depth(), model.get_n_leaves())
    print(export_text(model, feature_names=feature_names, max_depth=3))
    importance = permutation_importance(model, X_valid, y_valid, scoring="roc_auc",
                                        n_repeats=10, random_state=42)
    print(pd.DataFrame({"feature": feature_names, "MDI": model.feature_importances_,
                         "permutation_valid": importance.importances_mean}).sort_values("permutation_valid", ascending=False))
    proba, pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)
    print("Audit test indépendant : AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    # Vérifier que les probabilités de feuilles sont les proportions train non pondérées.
    leaf_fit, leaf_test = model.apply(X_fit), model.apply(X_test)
    for leaf in np.unique(leaf_test):
        expected = np.mean(np.asarray(y_fit)[leaf_fit == leaf])
        assert np.allclose(proba[leaf_test == leaf], expected)
    # ccp_alpha élevé réduit ou conserve le nombre de feuilles pour un même jeu et arbre de base.
    full = DecisionTreeClassifier(random_state=42, ccp_alpha=0).fit(X_fit, y_fit)
    pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=.02).fit(X_fit, y_fit)
    assert pruned.get_n_leaves() <= full.get_n_leaves()
    assert np.isfinite(proba).all()
    print("Proportions par feuille et réduction de complexité par pruning : OK")
    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        plt.figure(figsize=(18, 9))
        plot_tree(model, max_depth=3, feature_names=feature_names,
                  class_names=["Classe 0", "Classe 1"], filled=True, rounded=True)
        plt.tight_layout()
        plt.show()
    print("Les importances restent descriptives ; elles ne prouvent pas une causalité.")
