from ch02_cours10_lab import X_train, X_test, y_train, y_test, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipeline = make_pipeline(capping=True)
grid = {"preprocessor__numeric__imputer__strategy": ["mean", "median"],
        "preprocessor__numeric__capper": ["passthrough"],
        "model__C": [.01, .1, 1, 10]}
# TODO : ajouter QuantileCapper() comme seconde configuration de capper.
# TODO : comparer class_weight=None et balanced dans la recherche sur train.
# TODO : optimiser par AP, examiner aussi ROC-AUC et Brier.
# TODO : figer les choix puis évaluer une fois sur test.
# TODO : sauvegarder le pipeline, les versions, le schéma et les métriques.
# BONUS : appliquer la démarche à un dataset réel et documenter sa provenance.
