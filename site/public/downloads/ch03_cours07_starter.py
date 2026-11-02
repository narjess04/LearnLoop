from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=25, n_informative=12, n_redundant=6,
                           weights=[.95, .05], flip_y=.01, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
# TODO : comparer arbre et forêt sur les mêmes folds train avec ROC-AUC et AP.
# TODO : tester n_estimators, max_features, profondeur et taille de feuilles.
# TODO : RandomizedSearchCV avec budget explicite et sans parallélisation imbriquée.
# TODO : analyser OOB et permutation importance sur validation distincte.
# TODO : figer les choix avant audit test, puis examiner seuil et calibration.
