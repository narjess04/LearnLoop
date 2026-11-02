import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
}
for name, model in models.items():
    results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"auc": "roc_auc", "ap": "average_precision"},
        n_jobs=1
    )
    print(name, "AUC CV :", results["test_auc"].mean(),
          "AP CV :", results["test_ap"].mean())
# Choix didactique annoncé avant l'expérience : approfondir la logistique.
search = GridSearchCV(
    models["Logistic"], {"logisticregression__C": [0.01, 0.1, 1, 10]},
    cv=cv, scoring="roc_auc", refit=True, n_jobs=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
predictions = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, list(best_model.classes_).index(1)]
print("Test final AUC :", roc_auc_score(y_test, proba))
print("Test final AP :", average_precision_score(y_test, proba))
print(classification_report(y_test, predictions, labels=[0, 1],
      target_names=["malignant", "benign"], zero_division=0))
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
# La classe positive est 1, donc benign dans ce dataset pédagogique.
