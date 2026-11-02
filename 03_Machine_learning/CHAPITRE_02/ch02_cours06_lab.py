import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def create_features(X):
    X = X.copy()
    denominator = X["income"].where(X["income"] > 0)
    X["disposable_income"] = X["income"] - X["expenses"]
    X["expense_ratio"] = X["expenses"].div(denominator)
    X["debt_to_income"] = X["debt"].div(denominator)
    return X


rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({"income": rng.uniform(1500, 10000, n),
                   "expenses": rng.uniform(500, 6000, n),
                   "debt": rng.uniform(0, 30000, n), "age": rng.integers(18, 70, n)})
debt_ratio, expense_ratio = df["debt"] / df["income"], df["expenses"] / df["income"]
logits = -3 + .12 * debt_ratio + 1.5 * expense_ratio - .015 * (df["age"] - 40)
y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, features in [("Originales", "passthrough"),
                           ("Avec features", FunctionTransformer(create_features, validate=False))]:
        pipeline = Pipeline([("features", features),
                             ("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler()),
                             ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(name, scores, "mean:", scores.mean(), "std:", scores.std())
    print("La cible dépend des ratios par construction. Ce gain éventuel n'est pas universel.")
    print("Test final réservé.")
