import json
import platform
import tempfile
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.exceptions import NotFittedError
from ch02_cours10_lab import X_train, X_test, y_train, y_test, make_pipeline
from ch02_cours10_transformers import QuantileCapper

if __name__ == "__main__":
    pipeline = make_pipeline()
    grid = {"preprocessor__numeric__imputer__strategy": ["mean", "median"],
            "preprocessor__numeric__capper": ["passthrough", QuantileCapper()],
            "model__C": [.1, 1], "model__class_weight": [None, "balanced"]}
    search = GridSearchCV(pipeline, grid,
                          scoring={"ap": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"},
                          refit="ap", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    search.fit(X_train, y_train)
    print("Paramètres retenus:", search.best_params_, "AP de sélection:", search.best_score_)
    fitted = search.best_estimator_
    probabilities = fitted.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc": roc_auc_score(y_test, probabilities),
               "ap": average_precision_score(y_test, probabilities),
               "brier": brier_score_loss(y_test, probabilities)}
    print("Test final, choix figé:", metrics)
    # Vérifier la transformation sans réapprendre de statistiques.
    cap = QuantileCapper(lower=0, upper=1).fit([[1., 4.], [2., 4.], [3., 4.]])
    before = cap.upper_bounds_.copy()
    assert np.array_equal(cap.transform([[100., 7.]]), [[3., 4.]])
    assert np.array_equal(before, cap.upper_bounds_)
    try:
        QuantileCapper().transform([[1., 2.]])
    except NotFittedError:
        pass
    else:
        raise AssertionError("Le fit doit être requis")
    try:
        QuantileCapper(lower=.9, upper=.1).fit([[1.], [2.]])
    except ValueError:
        pass
    else:
        raise AssertionError("Bornes invalides non refusées")
    new_data = X_test.iloc[:3].copy()
    new_data["region"] = "nouvelle_region"
    expected = fitted.predict_proba(new_data)
    assert np.isfinite(expected).all()
    assert len(fitted.named_steps["preprocessor"].get_feature_names_out()) == fitted.named_steps["model"].n_features_in_
    metadata = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "schema": X_train.dtypes.astype(str).to_dict(),
                "metrics_test": metrics, "dataset": "simulation pédagogique, random_state=42"}
    # Démonstration sans laisser de modèle dans le dossier de travail.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pipeline.joblib"
        joblib.dump(fitted, path)
        loaded = joblib.load(path)  # Fichier fiable créé ici, jamais un fichier tiers inconnu.
        assert np.allclose(loaded.predict_proba(new_data), expected)
        (Path(directory) / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Tests du capper, catégories inconnues, noms des features et aller-retour joblib : OK")
    print("Pour conserver un artefact : python ch02_cours10_lab.py --save pipeline.joblib")
