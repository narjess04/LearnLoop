"""Complète la fonction de perturbation puis lance le benchmark fourni.
Les échelles doivent être calculées sur le train ; ne réentraîne pas sur les stress tests.
"""
import numpy as np

def perturb(X, scale, level=0.0, missing_rate=0.0, shift=0.0, seed=42):
    # TODO : valider les dimensions, les échelles, level et missing_rate.
    # TODO : créer rng = np.random.default_rng(seed) et copier X en float.
    # TODO : ajouter bruit gaussien * scale * level.
    # TODO : décaler la première variable de shift * scale[0].
    # TODO : remplacer par NaN les cellules tirées avec probabilité missing_rate.
    # TODO : retourner la copie, sans modifier X ni scale.
    raise NotImplementedError("Complète perturb pour tester la robustesse.")

def run():
    import ch05_cours04_solution as benchmark
    benchmark.perturb = perturb
    return benchmark.run()

if __name__ == "__main__":
    run()
