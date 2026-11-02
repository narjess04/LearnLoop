import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)
print(X.head())
print(y.value_counts())
print("Valeurs manquantes :", X.isna().sum().sum())
print(X.describe())
print("Classes :", dict(enumerate(dataset.target_names)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("Répartition train :", y_train.value_counts(normalize=True))
print("Répartition test :", y_test.value_counts(normalize=True))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

y_pred = model.predict(X_test)
# Vérifier l'ordre des colonnes plutôt que le supposer.
positive_column = list(model.classes_).index(1)
y_proba = model.predict_proba(X_test)[:, positive_column]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC (classe positive 1) :", roc_auc_score(y_test, y_proba))
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["malignant", "benign"], zero_division=0
))
print("Lignes = réel, colonnes = prédit, ordre [0, 1]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Accuracy de la baseline :",
      accuracy_score(y_test, baseline.predict(X_test)))
# Comparaison pédagogique fixée à l'avance.
# Ne pas régler les hyperparamètres sur le test.
