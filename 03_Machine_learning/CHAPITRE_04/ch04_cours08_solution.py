"""Benchmark des paramètres et graines : distances 2D non interprétées comme vàrità."""
import argparse
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ch04_cours08_lab import load_sample, compare, plot_embeddings

def neighbor_agreement(a, b, k=10):
    # Comparaison des voisinages, invariante aux rotations et translations des cartes.
    first = NearestNeighbors(n_neighbors=k).fit(a).kneighbors(return_distance=False)
    second = NearestNeighbors(n_neighbors=k).fit(b).kneighbors(return_distance=False)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(first, second)]))

def run(plot=False, samples=600, iterations=750):
    X, y = load_sample(samples)
    configs = [("PCA", "PCA", 0, 42)]
    configs += [(f"t-SNE p={p} seed={s}", "t-SNE", p, s) for p in (5, 15, 30, 50) for s in (42, 7)]
    configs += [(f"UMAP n={n} seed={s}", "UMAP", n, s) for n in (5, 15, 30, 60) for s in (42, 7)]
    embeddings, table = compare(X, configs, iterations)
    print(table.to_string(index=False))
    for kind, values, label in (("t-SNE", (5, 15, 30, 50), "p"), ("UMAP", (5, 15, 30, 60), "n")):
        for parameter in values:
            a, b = (embeddings[f"{kind} {label}={parameter} seed={seed}"] for seed in (42, 7))
            print(kind, parameter, "accord voisinages k=10 :", neighbor_agreement(a, b))
    print("Stabilité locale entre deux seeds ; pas de comparaison directe des coordonnées ni de preuve de classes naturelles.")
    if plot:
        plot_embeddings(embeddings, y)
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=750)
    args = parser.parse_args()
    run(args.plot, args.samples, args.iterations)
