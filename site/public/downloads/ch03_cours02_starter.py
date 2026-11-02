import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ch03_cours02_lab import X_train, X_test, y_train, y_test

pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
search = GridSearchCV(pipeline, {"model__alpha": np.logspace(-4, 4, 30)},
                      scoring="neg_root_mean_squared_error",
                      cv=KFold(n_splits=5, shuffle=True, random_state=42))
# TODO 1 : ajuster sur X_train, y_train ; afficher alpha et RMSE CV positive.
# TODO 2 : répéter pour Lasso et ElasticNet, avec des grilles adaptées à chaque objectif.
# TODO 3 : figer le choix selon les scores CV, puis réaliser l'audit final sur test.
# TODO 4 : comparer les coefficients et leur nombre de zéros.
# BONUS : refaire la sélection avec 100 observations train seulement.
# BONUS : tracer les coefficients selon alpha, sans utiliser les résultats du test.
