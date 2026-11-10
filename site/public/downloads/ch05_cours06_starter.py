"""Calcule les métriques de chaque groupe. Défaut = classe 1.
Une division par zéro donne NaN : un taux non mesurable ne vaut pas zéro.
"""
import numpy as np
import pandas as pd

def group_metrics(y_true, y_pred, groups):
    # TODO : valider les trois vecteurs alignés et les valeurs binaires.
    # TODO : pour chaque groupe, compter TP, FP, FN, TN et n.
    # TODO : calculer prevalence, selection_rate, TPR, FPR, FNR, precision et accuracy.
    # TODO : renvoyer un DataFrame indexé par "group".
    raise NotImplementedError("Complète group_metrics.")

def run():
    import ch05_cours06_solution as benchmark
    benchmark.group_metrics = group_metrics
    return benchmark.run()

if __name__ == "__main__":
    run()
