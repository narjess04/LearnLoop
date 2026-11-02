from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from ch02_cours09_lab import X_train, X_test, y_train, y_test

# Réserver la validation du seuil AVANT la recherche des hyperparamètres.
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)),
                     ("model", LogisticRegression(max_iter=1000))])
search = GridSearchCV(pipeline, {"smote__k_neighbors": [3, 5, 7], "model__C": [.01, .1, 1, 10]},
                      scoring="average_precision",
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
# TODO : fit sur X_fit, y_fit ; afficher les paramètres et scores.
# TODO : choisir le seuil minimisant 10*FN + FP sur X_valid, y_valid uniquement.
# TODO : figer modèle et seuil, puis évaluer une seule fois sur le test.
# BONUS : comparer RandomForest(class_weight="balanced") sur le train.
