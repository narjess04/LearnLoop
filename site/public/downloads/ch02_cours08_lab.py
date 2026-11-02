import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, reduction in [("Sans PCA", "passthrough"), ("PCA 5", PCA(n_components=5)),
                             ("PCA 95%", PCA(n_components=.95, svd_solver="full"))]:
        pipeline = Pipeline([("scale", StandardScaler()), ("pca", reduction),
                             ("model", LogisticRegression(max_iter=2000))])
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", return_estimator=True)
        assert np.isfinite(result["test_score"]).all()
        print(name, "ROC-AUC:", result["test_score"].mean(), "std:", result["test_score"].std())
        if name != "Sans PCA":
            print("Composantes par fold:", [p.named_steps["pca"].n_components_ for p in result["estimator"]])
    print("La variance conservée n'est pas une proportion d'information prédictive. Test final réservé.")
