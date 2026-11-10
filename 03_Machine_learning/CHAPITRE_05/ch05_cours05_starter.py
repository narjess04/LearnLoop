"""Complète le rang conforme et les métriques, puis observe Split Conformal et CQR."""
import math
import numpy as np

def conformal_quantile(scores, alpha):
    # TODO : valider un vecteur fini non vide et 0 < alpha < 1.
    # TODO : k = ceil((n+1) * (1-alpha)).
    # TODO : si k > n, renvoyer +inf ; sinon le k-ième plus petit score.
    # Les scores CQR peuvent être négatifs : ne pas les transformer en valeurs absolues.
    raise NotImplementedError("Complète conformal_quantile.")

def interval_metrics(y, lower, upper):
    # TODO : renvoyer {"coverage": ..., "mean_width": ...}.
    # Couverture : proportion de lower <= y <= upper, bornes incluses.
    # Largeur : mean(maximum(upper-lower, 0)).
    raise NotImplementedError("Complète interval_metrics.")

def run():
    import ch05_cours05_solution as benchmark
    benchmark.conformal_quantile = conformal_quantile
    benchmark.interval_metrics = interval_metrics
    return benchmark.run()

if __name__ == "__main__":
    run()
