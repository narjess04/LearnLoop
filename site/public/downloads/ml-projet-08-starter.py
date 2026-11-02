from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data = load_wine(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# TODO : EDA sur train, baseline et plusieurs familles de modèles.
# TODO : mêmes folds StratifiedKFold, accuracy et F1 macro.
# TODO : tuning avec budget fixé et test tenu à l'écart.
# TODO : refit, une évaluation finale et analyse des erreurs.
# TODO : README, versions, données, protocole et limites.
