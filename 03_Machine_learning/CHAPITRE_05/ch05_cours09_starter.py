"""Complète le bootstrap APPARIÉ et la correction de Holm."""
import numpy as np
from sklearn.metrics import roc_auc_score

def compare_auc_bootstrap(y_true, proba_a, proba_b, n_bootstrap=300, seed=42):
    # TODO : valider les entrées, tirer les MÊMES indices pour A et B.
    # TODO : ignorer les réplications mono-classe ; signaler un nombre valide insuffisant.
    # TODO : retourner difference, lower, upper, n_valid et samples (ndarray).
    # lower/upper = quantiles 2.5 % et 97.5 % des différences AUC.
    raise NotImplementedError("Complète compare_auc_bootstrap.")

def holm_adjust(pvalues):
    # TODO : trier les p-values, multiplier par m-rang, cummax et borner à 1.
    # TODO : remettre les valeurs ajustées dans l'ordre initial.
    raise NotImplementedError("Complète holm_adjust.")

def run():
    import ch05_cours09_solution as benchmark
    benchmark.compare_auc_bootstrap = compare_auc_bootstrap
    benchmark.holm_adjust = holm_adjust
    return benchmark.run()

if __name__ == "__main__":
    run()
