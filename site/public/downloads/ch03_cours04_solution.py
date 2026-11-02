import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, log_loss)
from ch03_cours04_lab import (X_train, X_test, y_train, y_test, make_pipeline,
                              probability_metrics, plot_diagnostics)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    exp_negative_absolute = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_negative_absolute),
                     exp_negative_absolute / (1 + exp_negative_absolute))


def binary_log_loss_from_logits(y_true, logits):
    y_true, logits = np.asarray(y_true, dtype=float), np.asarray(logits, dtype=float)
    if y_true.shape != logits.shape or y_true.size == 0 or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Cibles binaires et scores de mêmes dimensions requis")
    if not np.isfinite(logits).all():
        raise ValueError("Scores finis requis")
    return float(np.mean(np.logaddexp(0., logits) - y_true * logits))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les diagnostics probabilistes")
    args = parser.parse_args()
    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                                   stratify=y_train, random_state=43)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(make_pipeline(),
                          {"model__C": np.r_[np.logspace(-3, 3, 10), 1.],
                           "model__class_weight": [None, "balanced"]},
                          scoring={"ap": "average_precision", "roc_auc": "roc_auc",
                                   "brier": "neg_brier_score", "log_loss": "neg_log_loss"},
                          refit="ap", cv=cv)
    search.fit(X_fit, y_fit)
    print("Paramètres retenus par AP CV:", search.best_params_)
    print("AP de sélection, pas une évaluation indépendante:", search.best_score_)
    rows = []
    for i, parameters in enumerate(search.cv_results_["params"]):
        rows.append({"C": parameters["model__C"],
                     "class_weight": str(parameters["model__class_weight"]),
                     "AP_CV": search.cv_results_["mean_test_ap"][i],
                     "ROC_AUC_CV": search.cv_results_["mean_test_roc_auc"][i],
                     "Brier_CV": -search.cv_results_["mean_test_brier"][i]})
    print(pd.DataFrame(rows).sort_values("AP_CV", ascending=False).groupby("class_weight").head(3).to_string(index=False))
    model = search.best_estimator_
    validation_proba = model.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0., 1., 101)
    cost_fn, cost_fp = 10, 1  # Coûts pédagogiques fictifs.
    costs = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_valid, validation_proba >= threshold, labels=[0, 1]).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
    selected_threshold = float(thresholds[np.argmin(costs)])
    print("Seuil choisi sur validation:", selected_threshold, "coût fictif validation:", min(costs))
    # Garder le modèle de X_fit : un refit pourrait changer les probabilités et le seuil approprié.
    learned_mean = model.named_steps["scaler"].mean_.copy()
    test_proba = model.predict_proba(X_test)[:, 1]
    print("Test final, choix figés:", probability_metrics(y_test, test_proba))
    for threshold in [.5, selected_threshold]:
        pred = test_proba >= threshold
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        print({"threshold_fixed": threshold, "accuracy": accuracy_score(y_test, pred),
               "precision": precision_score(y_test, pred, zero_division=0),
               "recall": recall_score(y_test, pred, zero_division=0),
               "f1": f1_score(y_test, pred, zero_division=0),
               "specificity": tn / (tn + fp), "cost_fixed": cost_fn * fn + cost_fp * fp})
        print(np.array([[tn, fp], [fn, tp]]))
    assert np.array_equal(learned_mean, model.named_steps["scaler"].mean_)
    assert np.allclose(learned_mean, X_fit.to_numpy().mean(axis=0))
    # Bonus : stabilité numérique et comparaison aux définitions de référence.
    z = np.array([-5., -1., 0., 1., 5.])
    truth = np.array([0, 0, 1, 1, 1])
    assert np.isclose(binary_log_loss_from_logits(truth, z), log_loss(truth, sigmoid(z)))
    extreme = sigmoid(np.array([-1000., 0., 1000.]))
    assert np.isfinite(extreme).all() and np.allclose(extreme, [0., .5, 1.])
    assert np.isclose(binary_log_loss_from_logits([1, 0], [-1000., 1000.]), 1000.)
    assert np.allclose(sigmoid(-z), 1 - sigmoid(z))
    print("Bonus sigmoid/log-loss, scores extrêmes et scaler sur X_fit uniquement : OK")
    if args.plot:
        plot_diagnostics(y_test, test_proba)
    print("Discrimination et calibration à examiner séparément ; l’audit test ne sert pas à réviser les choix.")
