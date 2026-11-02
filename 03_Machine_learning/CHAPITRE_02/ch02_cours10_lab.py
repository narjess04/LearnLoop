import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_credit_data(n=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, size=n).astype(float)
    income = rng.lognormal(mean=8.2, sigma=.65, size=n)
    debt_ratio = rng.beta(a=2, b=5, size=n)
    employment_type = rng.choice(["CDI", "CDD", "Independent"], size=n, p=[.60, .25, .15])
    region = rng.choice(["Tunis", "Sousse", "Sfax"], size=n)
    logits = -3 + 4 * debt_ratio - .0001 * (income - 3500) + .5 * (employment_type == "CDD")
    target = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    df = pd.DataFrame({"age": age, "income": income, "debt_ratio": debt_ratio,
                       "employment_type": employment_type, "region": region, "default": target})
    # Anomalies de mesure ajoutées après la génération du phénomène latent.
    for name in ["age", "income", "employment_type"]:
        idx = rng.choice(n, size=n // 12, replace=False)
        df.loc[idx, name] = np.nan
    df.loc[rng.choice(n, size=max(1, n // 100), replace=False), "income"] *= 12
    return df


df = make_credit_data()
X = df.drop(columns="default")
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from ch02_cours10_transformers import QuantileCapper

numeric_features = ["age", "income", "debt_ratio"]
categorical_features = ["employment_type", "region"]


def make_pipeline(capping=True, class_weight=None):
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("capper", QuantileCapper() if capping else "passthrough"),
                                  ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                                      ("categorical", categorical_pipeline, categorical_features)])
    return Pipeline([("preprocessor", preprocessor),
                     ("model", LogisticRegression(class_weight=class_weight, max_iter=2000))])


if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                                 average_precision_score, brier_score_loss)
    import argparse
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Chemin facultatif du pipeline joblib à sauvegarder")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "ap": "average_precision", "brier": "neg_brier_score"}
    candidates = {}
    for capping in [False, True]:
        for weight in [None, "balanced"]:
            pipeline = make_pipeline(capping, weight)
            result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            assert np.isfinite(result["test_ap"]).all()
            name = f"capping={capping}, class_weight={weight}"
            print(name, "ROC-AUC", result["test_roc_auc"].mean(), "AP", result["test_ap"].mean(),
                  "Brier", -result["test_brier"].mean())
            candidates[name] = (result["test_ap"].mean(), pipeline)
    name, (_, full_pipeline) = max(candidates.items(), key=lambda pair: pair[1][0])
    full_pipeline.fit(X_train, y_train)
    proba = full_pipeline.predict_proba(X_test)[:, 1]
    pred = full_pipeline.predict(X_test)
    print("Choix par AP CV:", name)
    print("Test final : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    print("Features :", full_pipeline.named_steps["preprocessor"].get_feature_names_out())
    new_client = pd.DataFrame([{"age": 35, "income": 2500, "debt_ratio": .65,
                                "employment_type": "CDI", "region": "Bizerte"}])
    print("Sortie synthétique pour un nouveau client:", full_pipeline.predict_proba(new_client)[0, 1])
    if args.save:
        joblib.dump(full_pipeline, args.save)
        print("Pipeline sauvegardé :", args.save)
        # Ne recharger que des fichiers fiables. Garder le module de transformeurs disponible.
        loaded = joblib.load(args.save)
        assert np.allclose(loaded.predict_proba(new_client), full_pipeline.predict_proba(new_client))
    print("Données fictives ; les sorties ne sont pas un score bancaire validé.")
