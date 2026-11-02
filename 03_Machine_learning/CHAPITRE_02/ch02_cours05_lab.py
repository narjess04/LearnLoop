import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
n = 1200
df = pd.DataFrame({
    "age": rng.integers(18, 70, n), "income": rng.normal(4000, 1200, n),
    "region": rng.choice(["Tunis", "Sfax", "Sousse", "Nabeul"], n),
    "risk_level": rng.choice(["Faible", "Moyen", "Élevé"], n),
})
risk_effect = df["risk_level"].map({"Faible": -1.2, "Moyen": 0., "Élevé": 1.2})
region_effect = df["region"].map({"Tunis": .4, "Sfax": -.5, "Sousse": .8, "Nabeul": -.3})
logits = -1 + risk_effect + region_effect - .0002 * (df["income"] - 4000)
y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for strategy in ["One-Hot", "Ordinal risk"]:
        transformers = [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), ["age", "income"]),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
            ("risk", OneHotEncoder(handle_unknown="ignore") if strategy == "One-Hot"
             else OrdinalEncoder(categories=[["Faible", "Moyen", "Élevé"]],
                                 handle_unknown="use_encoded_value", unknown_value=-1),
             ["risk_level"]),
        ]
        pipeline = Pipeline([("preprocess", ColumnTransformer(transformers)),
                             ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(strategy, scores, "mean:", scores.mean(), "std:", scores.std())
    print("La région reste nominale dans les deux stratégies. Test final réservé.")
