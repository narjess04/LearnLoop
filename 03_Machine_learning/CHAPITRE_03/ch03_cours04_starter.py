import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from ch03_cours04_lab import X_train, X_test, y_train, y_test, make_pipeline

# Réserver la validation du seuil avant de régler C et les poids de classes.
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
pipeline = make_pipeline()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, {"model__C": np.r_[np.logspace(-3, 3, 10), 1.],
                                "model__class_weight": [None, "balanced"]},
                      scoring="average_precision", cv=cv)
# TODO : comparer les configurations sur X_fit, y_fit, sans utiliser X_valid pour le tuning.
# TODO : évaluer Accuracy, Precision, Recall, F1 et les métriques probabilistes.
# TODO : choisir un seuil sur la validation distincte avec des coûts fictifs explicités.
# TODO : figer modèle et seuil ; ne pas réentraîner sans revalider le seuil.
# TODO : réaliser l'audit test puis tracer ROC, PR et calibration.
# BONUS : sigmoid et log-loss NumPy stables, y compris pour des scores extrêmes.
