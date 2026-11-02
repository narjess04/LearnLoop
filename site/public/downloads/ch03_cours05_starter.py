from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8, n_redundant=5,
                           weights=[.9, .1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
# TODO : comparer K=5 sans scaling et avec StandardScaler sur les mêmes folds train.
# TODO : optimiser K, weights, p et le scaler avec GridSearchCV.
# TODO : comparer à une régression logistique sous le même protocole.
# TODO : fixer le choix avant le test ; examiner AP, ROC-AUC, recall, F1 et Brier.
# BONUS : inclure PCA dans le pipeline et sélectionner ses composantes sur train.
