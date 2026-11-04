"""Sélection sur validation, stabilité sur observations communes, audit final ràservà."""
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ch04_cours10_lab import customers, preprocessor, clustering_pipeline, metrics, calculate_psi, proportions

def run(plot=False):
    train, validation, test = customers(42, 600), customers(13, 200), customers(7, 200)
    # Géométrie commune fixàe sur train : aucune comparaison de silhouettes entre PCA différents.
    common = preprocessor().fit(train)
    X_validation = common.transform(validation)
    rows, fitted = [], {}
    for k in range(2, 11):
        model = clustering_pipeline(k).fit(train)
        fitted[k] = model
        row = metrics(X_validation, model.predict(validation))
        rows.append({"k": k, **row})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    k = int(table.loc[table.silhouette.idxmax(), "k"])
    print("K choisi sur validation (critàre fixà : silhouette dans espace commun) :", k)
    # Réentraîner TOUT le pipeline sur chaque sous-échantillon et prédire les mêmes clients validation.
    partitions = []
    for seed in (1, 2, 3, 4):
        subset = train.sample(frac=0.8, random_state=seed)
        model = clustering_pipeline(k, seed).fit(subset)
        partitions.append(model.predict(validation))
    stability = [adjusted_rand_score(a, b) for a, b in itertools.combinations(partitions, 2)]
    print("Stabilité sur validation commune, ARI moyen/min :", np.mean(stability), np.min(stability))
    # Le modàle et k sont figés avant l'accàs au test.
    model = fitted[k]
    test_labels = model.predict(test)
    audit = metrics(common.transform(test), test_labels)
    print("Audit final, espace prétraité commun :", audit)
    print("PSI train/test, même modàle :", calculate_psi(proportions(model.predict(train), k), proportions(test_labels, k)))
    print("La silhouette ne certifie pas la pertinence màtier ; documenter profils, effectifs, échantillonnage et limites.")
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(table.k, table.silhouette, marker="o")
        plt.xlabel("K")
        plt.ylabel("Silhouette validation, espace commun")
        plt.show()
    return table, stability, audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
