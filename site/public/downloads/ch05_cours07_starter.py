"""Complète la perte individuelle puis analyse les erreurs sur validation."""
import numpy as np

def individual_log_loss(y_true, proba):
    # TODO : valider deux vecteurs alignés, labels binaires, proba finies dans [0,1].
    # TODO : borner une COPIE des probabilités avec eps=1e-15 pour éviter log(0).
    # TODO : renvoyer -(y*log(p)+(1-y)*log(1-p)) pour chaque ligne.
    raise NotImplementedError("Complète individual_log_loss.")

def run():
    import ch05_cours07_solution as benchmark
    benchmark.individual_log_loss = individual_log_loss
    return benchmark.run()

if __name__ == "__main__":
    run()
