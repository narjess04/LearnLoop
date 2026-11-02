import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
                           weights=[.85, .15], class_sep=1.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def make_classifier():
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2))])


def classification_lab():
    pipeline = make_classifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X_train, y_train, cv=cv,
                             scoring={"ap": "average_precision", "roc_auc": "roc_auc", "recall": "recall"})
    print("Classification CV train :", {key: results["test_" + key].mean() for key in ["ap", "roc_auc", "recall"]})
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
    print("Test final, configuration fixe : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    assert np.allclose(pipeline.named_steps["scaler"].mean_, X_train.mean(axis=0))
    return pipeline, proba


def regression_lab():
    # Noms distincts : ne pas remplacer les données du problème de classification.
    X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=.2, random_state=42)
    regressor = Pipeline([("scaler", StandardScaler()),
                          ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance"))])
    scores = cross_validate(regressor, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    print("Régression RMSE CV train :", -scores["test_score"].mean())
    regressor.fit(X_reg_train, y_reg_train)
    prediction = regressor.predict(X_reg_test)
    assert np.all(prediction >= y_reg_train.min()) and np.all(prediction <= y_reg_train.max())
    print("Régression test fixe : MAE", mean_absolute_error(y_reg_test, prediction),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, prediction)), "R2", r2_score(y_reg_test, prediction))
    # Avec des poids positifs, la prédiction reste entre les cibles des voisins retenus.
    transformed = regressor.named_steps["scaler"].transform(X_reg_test)
    indices = regressor.named_steps["knn"].kneighbors(transformed, return_distance=False)
    neighborhood_targets = y_reg_train[indices]
    assert np.all(prediction >= neighborhood_targets.min(axis=1) - 1e-10)
    assert np.all(prediction <= neighborhood_targets.max(axis=1) + 1e-10)
    return y_reg_test, prediction


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher ROC, PR et prédictions de régression")
    args = parser.parse_args()
    classifier, proba = classification_lab()
    y_reg_test, prediction = regression_lab()
    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
        axes[2].scatter(y_reg_test, prediction, alpha=.5)
        axes[2].set(xlabel="Cible observée", ylabel="Prédiction KNN", title="Régression : test fixe")
        fig.tight_layout()
        plt.show()
    print("KNN : scores de voisinage, sans garantie de calibration. Données synthétiques.")
