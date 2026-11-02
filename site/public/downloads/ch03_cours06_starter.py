import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from ch03_cours06_lab import X_train, X_test, y_train, y_test

X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
model = DecisionTreeClassifier(random_state=42)
grid = {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"], "ccp_alpha": [0., .001, .01]}
# TODO : GridSearchCV sur X_fit et StratifiedKFold, refit selon ROC-AUC.
# TODO : examiner AP, règles, MDI et permutation importance sur X_valid.
# TODO : figer les décisions puis auditer le test indépendant.
# BONUS : comparer pre-pruning et post-pruning sans générer de candidats sur validation.
