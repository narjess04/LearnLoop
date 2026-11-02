import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss)

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8, n_redundant=5,
                           weights=[.9, .1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"ap": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", action="store_true", help="Inclure le choix PCA 5/10/15 dans la recherche")
    args = parser.parse_args()
    candidates = {}
    for name, scaler in [("K5 brut", "passthrough"), ("K5 standardisé", StandardScaler())]:
        pipeline = Pipeline([("scaler", scaler), ("knn", KNeighborsClassifier(n_neighbors=5))])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        candidates[name] = (scores["test_ap"].mean(), pipeline)
        print(name, "AP CV:", scores["test_ap"].mean(), "ROC-AUC CV:", scores["test_roc_auc"].mean())
    knn_pipeline = Pipeline([("scaler", StandardScaler()), ("pca", "passthrough"),
                             ("knn", KNeighborsClassifier(metric="minkowski"))])
    grid = {"scaler": [StandardScaler(), MinMaxScaler(), RobustScaler()],
            "knn__n_neighbors": [3, 5, 7, 11, 15, 21],
            "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
    if args.pca:
        grid["pca"] = ["passthrough", PCA(n_components=5), PCA(n_components=10), PCA(n_components=15)]
    knn_search = GridSearchCV(knn_pipeline, grid, cv=cv, scoring=scoring, refit="ap")
    knn_search.fit(X_train, y_train)
    print("KNN paramètres retenus:", knn_search.best_params_, "AP de sélection:", knn_search.best_score_)
    candidates["KNN optimisé"] = (knn_search.best_score_, knn_search.best_estimator_)
    logistic_pipeline = Pipeline([("scaler", StandardScaler()),
                                  ("model", LogisticRegression(max_iter=2000))])
    logistic_search = GridSearchCV(logistic_pipeline, {"model__C": [.01, .1, 1, 10]},
                                   cv=cv, scoring=scoring, refit="ap")
    logistic_search.fit(X_train, y_train)
    print("Logistique paramètres retenus:", logistic_search.best_params_, "AP de sélection:", logistic_search.best_score_)
    candidates["Logistique optimisée"] = (logistic_search.best_score_, logistic_search.best_estimator_)
    chosen = max(candidates, key=lambda name: candidates[name][0])
    print("Choix figé par AP CV avant ouverture du test:", chosen)
    rows = []
    for name, (_, pipeline) in candidates.items():
        pipeline.fit(X_train, y_train)
        start = time.perf_counter()
        proba = pipeline.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - start
        pred = pipeline.predict(X_test)
        assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
        rows.append({"model": name, "AP": average_precision_score(y_test, proba),
                     "ROC_AUC": roc_auc_score(y_test, proba), "Brier": brier_score_loss(y_test, proba),
                     "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
                     "predict_proba_seconds_one_call": elapsed})
    print(pd.DataFrame(rows).to_string(index=False))
    # Vérifier l'effet spécifique des distances nulles sur un vote pondéré.
    duplicate = KNeighborsClassifier(n_neighbors=3, weights="distance").fit([[0.], [0.], [1.]], [0, 1, 1])
    assert np.allclose(duplicate.predict_proba([[0.]]), [[.5, .5]])
    assert duplicate.predict([[0.]])[0] == 0  # Égalité : première classe dans cet exemple sklearn.
    # Vérifier PCA + scaler + KNN sur train, même sans le bonus de recherche étendue.
    with_pca = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=5)),
                         ("knn", KNeighborsClassifier(n_neighbors=7))])
    with_pca.fit(X_train, y_train)
    assert with_pca.predict(X_test[:3]).shape == (3,)
    assert np.allclose(with_pca.named_steps["scaler"].mean_, X_train.mean(axis=0))
    print("Votes avec distances nulles, égalité et pipeline PCA : OK")
    print("Les durées sont celles d’un seul appel, pas une garantie de vitesse générale.")
    print("Les scores de recherche servent à sélectionner ; le tableau test est un audit final, sans réviser le choix.")
