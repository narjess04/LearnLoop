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

# TODO 1 : calculer les taux de valeurs manquantes.
# TODO 2 : comparer les distributions d'âge selon la présence du revenu.
# TODO 3 : réserver une validation et comparer médiane / KNN / Iterative.
# TODO 4 : utiliser les valeurs masquées connues pour mesurer la MAE.
# Pourquoi le MAR est-il connu ici et pas identifiable de cette façon en données réelles ?
