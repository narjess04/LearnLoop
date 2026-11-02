import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({"age": rng.integers(18, 70, n),
                   "income": rng.normal(4000, 1000, n),
                   "debt_ratio": rng.uniform(0.1, 0.9, n)})
df_mcar, df_mar = df.copy(), df.copy()
mask_mcar = rng.random(n) < 0.15
missing_probability = np.where(df["age"] < 30, 0.35, 0.05)
mask_mar = rng.random(n) < missing_probability
df_mcar.loc[mask_mcar, "income"] = np.nan
df_mar.loc[mask_mar, "income"] = np.nan

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    train_idx, val_idx = train_test_split(np.arange(n), test_size=0.25, random_state=42)
    for name, incomplete in [("MCAR", df_mcar), ("MAR", df_mar)]:
        print(name, "missing rates (%):", incomplete.isna().mean() * 100)
        print(incomplete.groupby(incomplete["income"].isna())["age"].describe())
        validation_missing = incomplete.iloc[val_idx]["income"].isna().to_numpy()
        assert validation_missing.any()
        for method, imputer in [
            ("Median", SimpleImputer(strategy="median")),
            ("KNN", KNNImputer(n_neighbors=5)),
            ("Iterative", IterativeImputer(random_state=42, max_iter=10)),
        ]:
            scaler = StandardScaler()
            train = scaler.fit_transform(incomplete.iloc[train_idx])
            validation = scaler.transform(incomplete.iloc[val_idx])
            imputer.fit(train)
            restored = scaler.inverse_transform(imputer.transform(validation))
            expected = df.iloc[val_idx]["income"].to_numpy()[validation_missing]
            predicted = restored[validation_missing, 1]
            assert np.isfinite(predicted).all()
            print(name, method, "MAE sur revenus masqués:", mean_absolute_error(expected, predicted))
    print("Ici le revenu généré est indépendant de l'âge : exploiter l'âge ne garantit pas un gain.")
    print("Le mécanisme MAR est connu par construction, pas prouvé par les seuls indicateurs observés.")
