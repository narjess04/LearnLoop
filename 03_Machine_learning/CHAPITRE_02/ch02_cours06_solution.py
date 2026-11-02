import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def feature_engineering(df):
    result = df.copy(deep=True)
    denominator = result["income"].where(result["income"] > 0)
    result["disposable_income"] = result["income"] - result["expenses"]
    result["expense_ratio"] = result["expenses"].div(denominator)
    result["debt_to_income"] = result["debt"].div(denominator)
    result["income_log"] = np.log1p(result["income"].where(result["income"] >= 0))
    opened = pd.to_datetime(result["account_open_date"], errors="coerce")
    observed = pd.to_datetime(result["observation_date"], errors="coerce")
    days = (observed - opened).dt.days
    result["account_tenure_days"] = days.where(days >= 0)
    result["age_group"] = pd.cut(result["age"], bins=[18, 30, 45, 60, np.inf],
                                  labels=["18-29", "30-44", "45-59", "60+"], right=False)
    result["income_debt_interaction"] = result["income"] * result["debt_to_income"]
    return result


numeric = ["income", "expenses", "debt", "age", "disposable_income", "expense_ratio",
           "debt_to_income", "income_log", "account_tenure_days", "income_debt_interaction"]
pipeline = Pipeline([
    ("features", FunctionTransformer(feature_engineering, validate=False)),
    ("columns", ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), numeric),
        ("age_group", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))]), ["age_group"]),
    ])),
    ("model", LogisticRegression(max_iter=1000)),
])

if __name__ == "__main__":
    from ch02_cours06_lab import df, y
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    example = pd.DataFrame({"income": [0., 100., -2.], "expenses": [20., 30., 2.],
                            "debt": [10., 40., 3.], "age": [18, 45, 70],
                            "account_open_date": ["2025-01-01", "2027-01-01", "invalide"],
                            "observation_date": ["2026-01-01"] * 3})
    original = example.copy(deep=True)
    result = feature_engineering(example)
    pd.testing.assert_frame_equal(example, original)
    assert not np.isinf(result[["expense_ratio", "debt_to_income"]].to_numpy()).any()
    assert result["account_tenure_days"].dropna().ge(0).all()
    assert pd.isna(result.loc[1, "account_tenure_days"])
    assert set(numeric).issubset(result.columns) and "age_group" in result
    assert np.isclose(result.loc[1, "expense_ratio"], .3)
    X = df.assign(account_open_date="2020-01-01", observation_date="2026-01-01")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="roc_auc")
    print("Tests des ratios, dates, colonnes et absence de mutation : OK")
    print("CV train uniquement:", scores.mean(), scores.std())
    print("L'interaction income * debt_to_income est algébriquement redondante avec debt si income > 0.")
