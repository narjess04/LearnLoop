"""Monitoring pédagogique : PSI, KS, JS et absences. --evidently pour le rapport HTML local.
Sans labels, aucune AUC de production ne peut être déduite de ces seuls contrôles.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

def calculate_psi(reference, current, bins=10):
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins doit être un entier >= 2.")
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if ref.ndim != 1 or cur.ndim != 1:
        raise ValueError("Deux vecteurs numériques sont requis.")
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        raise ValueError("Échantillon fini vide.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins+1)))
    if len(edges) < 2:
        raise ValueError("Référence constante : définir des bins métier explicites.")
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(ref, bins=edges)
    b, _ = np.histogram(cur, bins=edges)
    # Lissage positif puis normalisation ; les bornes extrêmes conservent toute la masse.
    p = (a+1e-6)/(a.sum()+1e-6*len(a))
    q = (b+1e-6)/(b.sum()+1e-6*len(b))
    return float(np.sum((p-q)*np.log(p/q)))

def monitoring_status(psi, auc_drop, missing_rate):
    values = np.asarray([psi, auc_drop, missing_rate], dtype=float)
    if not np.isfinite(values).all() or psi < 0 or not -1 <= auc_drop <= 1 or not 0 <= missing_rate <= 1:
        raise ValueError("Indicateurs finis ; PSI >= 0, delta AUC dans [-1,1], absences dans [0,1].")
    # auc_drop = AUC référence − AUC courante ; négatif = amélioration.
    statuses = [
        2 if psi > .25 else 1 if psi > .10 else 0,
        2 if auc_drop > .05 else 1 if auc_drop > .02 else 0,
        2 if missing_rate > .10 else 1 if missing_rate > .05 else 0,
    ]
    return ["GREEN", "AMBER", "RED"][max(statuses)]

def run(include_evidently=False):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"revenu": rng.normal(3000, 700, 800),
        "endettement": rng.normal(35, 10, 800), "anciennete": rng.normal(8, 3, 800)})
    current = pd.DataFrame({"revenu": rng.normal(3800, 800, 800),
        "endettement": rng.normal(42, 12, 800), "anciennete": rng.normal(8, 3, 800)})
    current.loc[:31, "revenu"] = np.nan
    rows = []
    for column in reference:
        a, b = reference[column].dropna().to_numpy(), current[column].dropna().to_numpy()
        statistic, pvalue = ks_2samp(a, b)
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(a, edges)[0]; q = np.histogram(b, edges)[0]
        rows.append({"variable": column, "PSI": calculate_psi(a, b),
            "KS": statistic, "p_value": pvalue,
            "p_bonferroni": min(1., pvalue*len(reference.columns)),
            # scipy retourne une DISTANCE : carré pour obtenir la divergence.
            "JS_divergence_base2": jensenshannon(p/p.sum(), q/q.sum(), base=2)**2,
            "missing_reference": reference[column].isna().mean(),
            "missing_current": current[column].isna().mean()})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("KS significatif ≠ impact métier ; PSI heuristique ≠ test de baisse d'AUC.")
    print("Pas de labels ici : performance et concept drift restent à investiguer.")
    if include_evidently:
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset
        definition = DataDefinition(numerical_columns=list(reference.columns))
        snapshot = Report(metrics=[DataDriftPreset()]).run(
            reference_data=Dataset.from_pandas(reference, data_definition=definition),
            current_data=Dataset.from_pandas(current, data_definition=definition))
        snapshot.save_html("data_drift_report.html")
        print("Rapport local : data_drift_report.html")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidently", action="store_true")
    run(parser.parse_args().evidently)
