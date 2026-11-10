"""Simulation d'additivité ; pas un calcul de valeurs SHAP."""
import numpy as np

def reconstruct(base_value, contributions):
    return float(base_value + np.asarray(contributions, dtype=float).sum())

def global_importance(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Une matrice non vide (observations, variables) est requise.")
    return np.abs(values).mean(axis=0)

def run():
    phi = np.array([.25, .18, -.10, -.05])
    assert np.isclose(reconstruct(.5, phi), .78)
    print("Probabilité reconstruite :", reconstruct(.5, phi))
    print("Importance globale :", global_importance([[.2, -.1], [-.3, .4], [.1, -.2]]))

if __name__ == "__main__":
    run()
