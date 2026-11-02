import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, average_precision_score,
                             roc_auc_score, balanced_accuracy_score, brier_score_loss)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

X, y = make_classification(n_samples=5000, n_features=20, n_informative=8,
                           n_redundant=5, weights=[.95, .05], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"ap": "average_precision", "roc_auc": "roc_auc",
           "balanced_accuracy": "balanced_accuracy", "recall": "recall"}


def make_models():
    return {name: Pipeline([("scaler", StandardScaler()), ("sampler", sampler),
                             ("model", LogisticRegression(class_weight=weight, max_iter=1000))])
            for name, sampler, weight in [
                ("Baseline", "passthrough", None),
                ("Balanced weights", "passthrough", "balanced"),
                ("Random oversampling", RandomOverSampler(random_state=42), None),
                ("SMOTE", SMOTE(random_state=42), None)]}


if __name__ == "__main__":
    candidates = {}
    for name, pipeline in make_models().items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        assert np.isfinite(scores["test_ap"]).all()
        print(name, {metric: (scores["test_" + metric].mean(), scores["test_" + metric].std())
                     for metric in scoring})
        candidates[name] = (scores["test_ap"].mean(), pipeline)
    name, (_, model) = max(candidates.items(), key=lambda pair: pair[1][0])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    assert len(pred) == len(y_test)  # Le test n'est pas rééchantillonné.
    print("Choix par Average Precision CV :", name)
    print("Test final : AP", average_precision_score(y_test, proba),
          "ROC-AUC", roc_auc_score(y_test, proba), "Balanced Accuracy", balanced_accuracy_score(y_test, pred),
          "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    # Optionnel : python ch02_cours09_lab.py --plot
    import sys
    if "--plot" in sys.argv:
        from sklearn.metrics import PrecisionRecallDisplay
        import matplotlib.pyplot as plt
        PrecisionRecallDisplay.from_predictions(y_test, proba)
        plt.show()
