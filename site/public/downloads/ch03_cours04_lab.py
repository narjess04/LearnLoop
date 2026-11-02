import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, log_loss, brier_score_loss)

rng = np.random.default_rng(42)
n = 5000
income = rng.lognormal(mean=8.2, sigma=.5, size=n)
debt_ratio = rng.beta(a=2, b=5, size=n)
employment_years = rng.integers(0, 30, size=n)
previous_incidents = rng.binomial(1, .12, size=n)
logit = -3.5 - .00015 * income + 4.0 * debt_ratio - .04 * employment_years + 1.3 * previous_incidents
probability = 1 / (1 + np.exp(-logit))
default = rng.binomial(1, probability)
df = pd.DataFrame({"income": income, "debt_ratio": debt_ratio,
                   "employment_years": employment_years, "previous_incidents": previous_incidents,
                   "default": default})
X, y = df.drop(columns="default"), df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def make_pipeline(C=1., class_weight=None):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, C=C, class_weight=class_weight))])


def probability_metrics(y_true, proba):
    return {"roc_auc": roc_auc_score(y_true, proba),
            "average_precision": average_precision_score(y_true, proba),
            "log_loss": log_loss(y_true, proba, labels=[0, 1]),
            "brier": brier_score_loss(y_true, proba)}


def plot_diagnostics(y_true, proba):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1])
    CalibrationDisplay.from_predictions(y_true, proba, n_bins=10, strategy="quantile", ax=axes[2])
    axes[2].set_title("Calibration : description du test, sans réglage")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher ROC, PR et calibration")
    args = parser.parse_args()
    pipeline = make_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring={"ap": "average_precision", "roc_auc": "roc_auc",
                                     "log_loss": "neg_log_loss", "brier": "neg_brier_score"})
    print("Prévalence train:", y_train.mean())
    print("CV train : AP", scores["test_ap"].mean(), "ROC-AUC", scores["test_roc_auc"].mean(),
          "Log-Loss", -scores["test_log_loss"].mean(), "Brier", -scores["test_brier"].mean())
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = proba >= .5  # Convention explicite pour le seuil de cet exercice.
    assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
    print("Audit final de la baseline fixe:", probability_metrics(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    coefficients = pipeline.named_steps["model"].coef_[0]
    raw_coefficients = coefficients / pipeline.named_steps["scaler"].scale_
    increments = np.array([1000., .1, 5., 1.])
    interpretation = pd.DataFrame({"feature": X.columns, "coefficient_standardise": coefficients,
                                    "OR_par_ecart_type": np.exp(coefficients),
                                    "increment_original": increments,
                                    "OR_par_increment_original": np.exp(raw_coefficients * increments)})
    print(interpretation.to_string(index=False))
    print("Incréments : revenu +1000, ratio +0.1, ancienneté +5 ans, incident de 0 à 1.")
    # Vérifier l'interprétation conditionnelle des odds sur un seul emprunteur fictif.
    client = X_test.iloc[[0]].copy()
    changed = client.copy()
    changed["income"] += 1000.
    p0, p1 = pipeline.predict_proba(client)[0, 1], pipeline.predict_proba(changed)[0, 1]
    assert np.isclose((p1 / (1-p1)) / (p0 / (1-p0)), np.exp(raw_coefficients[0] * 1000.))
    if args.plot:
        plot_diagnostics(y_test, proba)
    print("Données et coefficients générateurs fictifs ; les sorties ne sont pas un score bancaire validé.")
    print("Le Brier mesure une qualité probabiliste globale, pas uniquement la calibration.")
