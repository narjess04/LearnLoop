import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
n = 1800
codes = rng.integers(0, 120, n)
effects = rng.normal(0, 1.2, 120)
X = pd.DataFrame({"category": [f"C{v}" for v in codes]})
X.loc[:39, "category"] = [f"rare_{v}" for v in range(40)]
y = rng.binomial(1, 1 / (1 + np.exp(-(-.7 + effects[codes]))))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    models = {}
    for name, encoder in [
        ("One-Hot", OneHotEncoder(handle_unknown="ignore")),
        ("Target auto", TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42)),
        ("Target smooth20", TargetEncoder(target_type="binary", smooth=20., cv=5, random_state=42)),
    ]:
        pipeline = Pipeline([("encoder", encoder), ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(name, "CV ROC-AUC:", scores.mean(), "+/-", scores.std())
        models[name] = (scores.mean(), pipeline)
    name, (_, selected) = max(models.items(), key=lambda item: item[1][0])
    selected.fit(X_train, y_train)
    print("Choix par CV:", name, "test final:", roc_auc_score(y_test, selected.predict_proba(X_test)[:, 1]))
    unseen = pd.DataFrame({"category": ["jamais_vue"]})
    te = TargetEncoder(target_type="binary", cv=5, random_state=42)
    te.fit_transform(X_train, y_train)
    assert np.allclose(te.transform(unseen), np.mean(y_train))
    ohe = OneHotEncoder(handle_unknown="ignore").fit(X_train)
    assert ohe.transform(unseen).nnz == 0
    print("Catégories inconnues : moyenne train pour TargetEncoder, zéros pour One-Hot.")
