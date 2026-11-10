"""Atelier NumPy : contributions additives et importance globale.
Simulation pédagogique ; ce code ne calcule pas de valeurs SHAP.
Les scripts du mini-projet complet utilisent réellement SHAP et LIME en local.
"""
import numpy as np

def reconstruct(base_value, contributions):
    # TODO : renvoyer base_value + somme des contributions (sans arrondir ni borner).
    raise NotImplementedError("Complète reconstruct.")

def global_importance(values):
    # TODO : values est une matrice (observations, variables).
    # Retourner la moyenne des valeurs absolues, par variable (axis=0).
    raise NotImplementedError("Complète global_importance.")

def run():
    phi = np.array([.25, .18, -.10, -.05])
    print("Probabilité reconstruite :", reconstruct(.50, phi))
    matrix = np.array([[.2, -.1], [-.3, .4], [.1, -.2]])
    print("Importance globale :", global_importance(matrix))

if __name__ == "__main__":
    run()
