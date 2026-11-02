from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = load_wine(return_X_y=True, as_frame=True)
# TODO : split stratifié 80/20 (random_state=42)
# TODO : pipeline StandardScaler + LogisticRegression
# TODO : fit sur train uniquement
# TODO : prédire le test et calculer accuracy et F1 macro
# TODO : afficher les probabilités de cinq observations
# TODO : relier chaque colonne à model.classes_
