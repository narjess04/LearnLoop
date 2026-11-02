import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=3000, n_features=12, n_informative=7, n_redundant=3,
                           weights=[.9, .1], flip_y=.02, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def regression_lab():
    X_reg, y_reg = make_regression(n_samples=1000, n_features=8, noise=20, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, criterion="squared_error", random_state=42)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    print("Arbre régression RMSE CV train:", -scores["test_score"].mean())
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Arbre régression test fixe : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    leaf_ids = model.apply(X_reg_test)
    for leaf in np.unique(leaf_ids):
        assert np.allclose(pred[leaf_ids == leaf], pred[leaf_ids == leaf][0])
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les AUC CV et un arbre de profondeur 3")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, models = [], {}
    for depth in [2, 3, 5, 7, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_validate(model, X_train, y_train, cv=cv, return_train_score=True,
                                scoring={"auc": "roc_auc", "ap": "average_precision"})
        model.fit(X_train, y_train)
        rows.append({"depth": str(depth), "train_auc_CV": scores["train_auc"].mean(),
                     "validation_auc_CV": scores["test_auc"].mean(), "AP_CV": scores["test_ap"].mean(),
                     "leaves_on_train": model.get_n_leaves()})
        models[str(depth)] = model
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    chosen = results.loc[results["validation_auc_CV"].idxmax(), "depth"]
    model = models[chosen]
    print("Profondeur choisie par CV avant le test:", chosen)
    proba, pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)
    print("Audit final : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    print(pd.DataFrame({"feature": feature_names, "MDI": model.feature_importances_}).sort_values("MDI", ascending=False))
    regression_lab()
    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        axes[0].plot(results["depth"], results["train_auc_CV"], marker="o", label="Train CV")
        axes[0].plot(results["depth"], results["validation_auc_CV"], marker="o", label="Validation CV")
        axes[0].set(xlabel="Profondeur", ylabel="ROC-AUC")
        axes[0].legend()
        shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42).fit(X_train, y_train)
        plot_tree(shallow, feature_names=feature_names, class_names=["Classe 0", "Classe 1"],
                  filled=True, rounded=True, fontsize=7, ax=axes[1])
        fig.tight_layout()
        plt.show()
    print("La sélection de profondeur utilise les folds train ; le test ne sert pas à la modifier.")
