import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=1200, n_features=10, n_informative=6, random_state=42)
X[rng.random(X.shape) < 0.15] = np.nan
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
imputers = {
    "Median": SimpleImputer(strategy="median"),
    "KNN": KNNImputer(n_neighbors=5, weights="distance"),
    "Iterative": IterativeImputer(max_iter=10, random_state=42),
}

if __name__ == "__main__":
    for name, imputer in imputers.items():
        model = Pipeline([
            # StandardScaler ignore les NaN au fit et les conserve au transform.
            # Mise à l'échelle AVANT KNN pour rendre les distances comparables.
            ("distance_scale", StandardScaler()), ("imputer", imputer),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        start = time.perf_counter()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        assert scores.shape == (5,) and np.isfinite(scores).all()
        print(name, "ROC-AUC:", scores, "mean:", scores.mean(),
              "std:", scores.std(), "seconds:", time.perf_counter() - start)
    print("Test final réservé, non utilisé pour comparer les imputations.")
