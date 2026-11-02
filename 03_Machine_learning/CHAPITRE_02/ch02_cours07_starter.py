from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from ch02_cours07_lab import X_train, X_test, y_train, y_test

pipeline = Pipeline([("scale", StandardScaler()), ("selector", SelectKBest(f_classif)),
                     ("model", LogisticRegression(max_iter=1000))])
search = GridSearchCV(pipeline, {"selector__k": [3, 5, 8, 12, 15], "model__C": [.01, .1, 1, 10]},
                      scoring="roc_auc", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
# TODO : fit sur train, afficher best_params_, best_score_ et le masque sélectionné.
# TODO : comparer avec une baseline sous le même protocole de sélection.
# TODO : figer le choix sur la CV, puis ouvrir une seule fois le test final.
# BONUS : calculer la fréquence de sélection des variables entre folds train.
