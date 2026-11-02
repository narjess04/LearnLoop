import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                           n_redundant=6, n_repeated=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selectors = {"Sans sélection": "passthrough", "SelectKBest": SelectKBest(f_classif, k=6),
             "RFE": RFE(LogisticRegression(max_iter=1000), n_features_to_select=6),
             "L1": SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear",
                                                       C=.1, max_iter=1000, random_state=42))}

if __name__ == "__main__":
    for name, selector in selectors.items():
        pipeline = Pipeline([("scale", StandardScaler()), ("selector", selector),
                             ("model", LogisticRegression(max_iter=1000))])
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", return_estimator=True)
        assert np.isfinite(result["test_score"]).all()
        print(name, "ROC-AUC:", result["test_score"].mean(), "std:", result["test_score"].std(),
              "fit seconds:", result["fit_time"].mean())
        if name != "Sans sélection":
            masks = np.array([p.named_steps["selector"].get_support() for p in result["estimator"]])
            print("Fréquence de sélection par variable:", masks.mean(axis=0))
    print("Configurations fixes ; test final réservé. Stabilité ne signifie ni causalité ni utilité garantie.")
