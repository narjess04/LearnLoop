import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=4000, n_features=20, n_informative=10, n_redundant=5,
                           weights=[.9, .1], class_sep=1., random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]


def regression_lab():
    X_reg, y_reg = make_regression(n_samples=2500, n_features=10, n_informative=6, noise=25, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=.8,
                                   random_state=42, n_jobs=1)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Forêt régression RMSE CV:", -scores["test_score"].mean(),
          "test : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    average = np.mean([tree.predict(X_reg_test[:5]) for tree in model.estimators_], axis=0)
    assert np.allclose(model.predict(X_reg_test[:5]), average)
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {"Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
                  "Bagging 200": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8),
                                                   n_estimators=200, random_state=42, n_jobs=1)}
    for trees in [1, 5, 10, 25, 50, 100, 200]:
        candidates[f"Random Forest {trees}"] = RandomForestClassifier(n_estimators=trees, max_depth=8,
                                                                       random_state=42, n_jobs=1)
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring={"auc": "roc_auc", "ap": "average_precision"})
        rows.append({"model": name, "ROC_AUC_CV": scores["test_auc"].mean(),
                     "AP_CV": scores["test_ap"].mean()})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    chosen = results.loc[results["ROC_AUC_CV"].idxmax(), "model"]
    model = candidates[chosen].fit(X_train, y_train)
    pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    print("Choix par CV avant test:", chosen, "audit test : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "F1", f1_score(y_test, pred))
    oob = RandomForestClassifier(n_estimators=300, bootstrap=True, oob_score=True, random_state=42, n_jobs=1)
    oob.fit(X_train, y_train)
    oob_matrix = oob.oob_decision_function_
    valid = np.isfinite(oob_matrix).all(axis=1) & (oob_matrix.sum(axis=1) > 0)
    print("OOB accuracy (défaut sklearn):", oob.oob_score_, "couverture OOB:", valid.mean())
    print("OOB ROC-AUC:", roc_auc_score(y_train[valid], oob_matrix[valid, 1]),
          "OOB AP:", average_precision_score(y_train[valid], oob_matrix[valid, 1]))
    # La forêt moyenne les probabilités des arbres, pas nécessairement leurs décisions de classe.
    average = np.mean([tree.predict_proba(X_test[:5]) for tree in oob.estimators_], axis=0)
    assert np.allclose(oob.predict_proba(X_test[:5]), average)
    regression_lab()
    if args.plot:
        import matplotlib.pyplot as plt
        counts = [1, 5, 10, 25, 50, 100, 200]
        aucs = [float(results.loc[results["model"] == f"Random Forest {n}", "ROC_AUC_CV"].iloc[0]) for n in counts]
        plt.plot(counts, aucs, marker="o")
        plt.xlabel("Nombre d’arbres")
        plt.ylabel("ROC-AUC CV train")
        plt.title("Stabilisation empirique, sans choisir sur test")
        plt.show()
    print("OOB et CV ne sont pas des garanties pour le temps ou les données groupées ; test final distinct.")
