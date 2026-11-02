import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from ch02_cours09_lab import X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                                   stratify=y_train, random_state=43)
    pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)),
                         ("model", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, {"smote__k_neighbors": [3, 5, 7], "model__C": [.01, .1, 1, 10]},
                          scoring="average_precision", cv=cv)
    search.fit(X_fit, y_fit)
    print("Paramètres:", search.best_params_, "AP de sélection:", search.best_score_)
    valid_proba = search.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0., 1., 101)
    costs = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_valid, valid_proba >= threshold, labels=[0, 1]).ravel()
        costs.append(10 * fn + fp)
    threshold = float(thresholds[np.argmin(costs)])
    print("Seuil choisi sur validation:", threshold, "coût validation:", min(costs))
    # Garder ce modèle ajusté sur X_fit : réentraîner changerait ses probabilités.
    test_proba = search.predict_proba(X_test)[:, 1]
    test_pred = test_proba >= threshold
    assert len(test_pred) == len(y_test)
    print("Test final, modèle et seuil figés : AP", average_precision_score(y_test, test_proba))
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
    print(classification_report(y_test, test_pred, zero_division=0))
    # Comparaison du bonus sur les seuls folds train, sans modifier le choix déjà figé.
    forest = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    scores = cross_val_score(forest, X_fit, y_fit, cv=cv, scoring="average_precision")
    print("Bonus forêt : AP CV train", scores.mean(), scores.std())
