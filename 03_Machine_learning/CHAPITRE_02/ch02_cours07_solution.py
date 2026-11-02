import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from ch02_cours07_lab import X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pipeline = Pipeline([("scale", StandardScaler()), ("selector", SelectKBest(f_classif)),
                         ("model", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Comparaison de la sélection et de la baseline dans la même recherche train.
    grid = [{"selector__k": [3, 5, 8, 12, 15], "model__C": [.01, .1, 1, 10]},
            {"selector": ["passthrough"], "model__C": [.01, .1, 1, 10]}]
    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv)
    search.fit(X_train, y_train)
    print("Meilleurs paramètres:", search.best_params_)
    print("Score de sélection CV (optimiste pour une performance finale):", search.best_score_)
    selected = search.best_estimator_.named_steps["selector"]
    mask = np.ones(X_train.shape[1], dtype=bool) if isinstance(selected, str) else selected.get_support()
    print("Variables retenues:", np.flatnonzero(mask))
    print("Test final, choix figé:", roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))
    # Stabilité de la configuration sélectionnée, descriptive, sur train uniquement.
    result = cross_validate(search.best_estimator_, X_train, y_train, cv=cv, return_estimator=True)
    masks = [np.ones(X_train.shape[1], dtype=bool) if isinstance(p.named_steps["selector"], str)
             else p.named_steps["selector"].get_support() for p in result["estimator"]]
    frequencies = np.mean(masks, axis=0)
    assert np.all((frequencies >= 0) & (frequencies <= 1))
    print("Fréquences de sélection:", frequencies)
