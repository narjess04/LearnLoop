from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

onehot_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("model", LogisticRegression(max_iter=1000)),
])
target_pipeline = Pipeline([
    ("encoder", TargetEncoder(target_type="binary", cv=5, random_state=42)),
    ("model", LogisticRegression(max_iter=1000)),
])
# TODO 1 : générer une colonne category à forte cardinalité et une cible binaire.
# TODO 2 : réserver un test avant toute comparaison.
# TODO 3 : comparer les pipelines sur les mêmes folds train en ROC-AUC.
# TODO 4 : tester une régularisation plus forte et gérer les catégories inconnues.
# Ne pas remplacer fit_transform(X, y) par fit(X, y).transform(X).
