import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = make_classification(
    n_samples=1000, n_features=30, n_informative=8, random_state=42
)
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.05] = np.nan
# TODO : pipeline imputation médiane -> scaler -> sélection -> logistique.
# TODO : StratifiedKFold et scores, moyenne, écart-type descriptif.
# TODO : expliquer dans quel périmètre chaque étape apprend ses paramètres.
