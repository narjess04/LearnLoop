"""Choisir modàle et seuil sur validation synthétique puis auditer un test neuf."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, average_precision_score
from ch04_cours09_lab import transactions, models, evaluate

def select_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # La derniàre paire precision/recall n'a pas de seuil associà.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])

def run(plot=False):
    train, _ = transactions(42, normal=700, anomalies=0)
    validation, y_valid = transactions(13)
    test, y_test = transactions(7)
    fitted = models()
    candidates = []
    for name, model in fitted.items():
        model.fit(train)
        threshold, f1 = select_threshold(y_valid, -model.decision_function(validation))
        candidates.append({"model": name, "threshold": threshold, "F1_validation": f1})
    table = pd.DataFrame(candidates)
    print(table.to_string(index=False))
    winner = max(candidates, key=lambda row: row["F1_validation"])
    scores = -fitted[winner["model"]].decision_function(test)
    # >= cohérent avec precision_recall_curve, y=1 représente une anomalie.
    predictions = (scores >= winner["threshold"]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    result = {"precision": p, "recall": r, "F1": f1, "AP": average_precision_score(y_test, scores), "alert_rate": predictions.mean()}
    print("Configuration figée sur validation :", winner)
    print("Audit final indépendant :", result)
    print("Le seuil utilise des labels de validation : ce réglage est supervisà, malgrà un entraînement sans y.")
    print("Les anomalies synthétiques sont plus simples que des cas ràels ; documenter coàts des alertes et retours màtier.")
    if plot:
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay.from_predictions(y_test, scores)
        plt.show()
    return winner, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
