"""Complète PSI et les règles d'alerte. Les absences sont suivies séparément."""
import numpy as np

def calculate_psi(reference, current, bins=10):
    # TODO : filtrer les valeurs non finies ; refuser les échantillons vides/constante.
    # TODO : apprendre les bins sur référence, retirer les doublons, étendre à +/- inf.
    # TODO : histogrammes, lissage positif normalisé, somme (p-q)*log(p/q).
    raise NotImplementedError("Complète calculate_psi.")

def monitoring_status(psi, auc_drop, missing_rate):
    # TODO : GREEN si PSI<=.10, baisse AUC<=.02 et absences<=.05.
    # TODO : RED si PSI>.25 OU baisse AUC>.05 OU absences>.10 ; sinon AMBER.
    # Seuils fictifs et configurables, sans réentraînement automatique.
    raise NotImplementedError("Complète monitoring_status.")

def run():
    import ch05_cours08_solution as benchmark
    benchmark.calculate_psi = calculate_psi
    benchmark.monitoring_status = monitoring_status
    print("Alerte fictive :", monitoring_status(.14, .03, .04))
    return benchmark.run()

if __name__ == "__main__":
    run()
