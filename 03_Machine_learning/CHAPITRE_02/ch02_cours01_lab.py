import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "age": rng.integers(18, 75, n),
    "income": rng.lognormal(10.5, 0.6, n),
    "debt_ratio": rng.uniform(0, 1, n),
    "region": rng.choice(["Tunis", "Sfax", "Sousse", "Nabeul"], n),
    "default": rng.choice([0, 1], n, p=[0.92, 0.08]),
})
missing_idx = rng.choice(n, 100, replace=False)
df.loc[missing_idx, "income"] = np.nan
df.loc[0, "age"] = -5
df.loc[1, "debt_ratio"] = 1.8

print("Shape:", df.shape)
print(df.dtypes)
print(df.describe(include="all"))
print("Valeurs manquantes (%):", df.isna().mean() * 100)
print("Doublons exacts:", df.duplicated().sum())
print("Âges invalides:", (df["age"] < 0).sum())
print("Ratios invalides:", (~df["debt_ratio"].between(0, 1)).sum())
print("Cible (%):", df["default"].value_counts(normalize=True) * 100)
assert df.shape == (1000, 5)
assert df["income"].isna().sum() == 100
assert (df["age"] < 0).sum() == 1
assert (~df["debt_ratio"].between(0, 1)).sum() == 1

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df["income"].hist(bins=30, ax=axes[0])
    axes[0].set(title="Distribution du revenu", xlabel="Revenu", ylabel="Effectif")
    axes[1].boxplot(df["income"].dropna())
    axes[1].set_title("Boxplot du revenu")
    df["region"].value_counts().plot(kind="bar", ax=axes[2])
    axes[2].set_title("Distribution des r?gions")
    fig.tight_layout()
    plt.show()
