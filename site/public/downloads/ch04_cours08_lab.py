"""Digits : PCA, t-SNE, UMAP ; mêmes observations et màtrique de voisinage.
Installer umap-learn. --plot pour les figures ; --samples et --iterations règlent le budget.
"""
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split

def load_sample(samples=600):
    data = load_digits()
    if not 60 <= samples <= len(data.data):
        raise ValueError("samples doit àtre compris entre 60 et 1797.")
    ids = np.arange(len(data.data))
    if samples < len(ids):
        ids, _ = train_test_split(ids, train_size=samples, stratify=data.target, random_state=42)
    return StandardScaler().fit_transform(data.data[ids]), data.target[ids]

def compare(X, configurations, iterations=750):
    if iterations < 300:
        raise ValueError("Prévoir au moins 300 itàrations, après early exaggeration.")
    import umap
    embeddings, rows = {}, []
    for name, kind, parameter, seed in configurations:
        if kind == "PCA":
            model = PCA(n_components=2, svd_solver="full")
        elif kind == "t-SNE":
            if not 0 < parameter < len(X):
                raise ValueError("La perplexità doit àtre inférieure à n.")
            model = TSNE(n_components=2, perplexity=parameter, learning_rate="auto", init="pca", max_iter=iterations, random_state=seed)
        else:
            model = umap.UMAP(n_components=2, n_neighbors=int(parameter), min_dist=0.1, random_state=seed, n_jobs=1, n_epochs=200)
        start = time.perf_counter()
        embedding = model.fit_transform(X)
        seconds = time.perf_counter() - start
        assert embedding.shape == (len(X), 2) and np.isfinite(embedding).all()
        score = trustworthiness(X, embedding, n_neighbors=10)
        assert 0 <= score <= 1
        embeddings[name] = embedding
        rows.append({"model": name, "seed": seed, "seconds": seconds, "trustworthiness_k10": score})
    return embeddings, pd.DataFrame(rows)

def plot_embeddings(embeddings, y):
    import matplotlib.pyplot as plt
    count = len(embeddings)
    fig, axes = plt.subplots((count + 2) // 3, min(3, count), figsize=(13, 4 * ((count + 2) // 3)), squeeze=False)
    for ax, (name, embedding) in zip(axes.flat, embeddings.items()):
        ax.scatter(*embedding.T, c=y, cmap="tab10", s=8)
        ax.set_title(name)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()

def run(plot=False, samples=600, iterations=750):
    X, y = load_sample(samples)
    configurations = [("PCA", "PCA", 0, 42), ("t-SNE p=30", "t-SNE", 30, 42), ("UMAP n=15", "UMAP", 15, 42)]
    embeddings, table = compare(X, configurations, iterations)
    print(table.to_string(index=False))
    print("Exploration transductive sur les mêmes points ; y sert uniquement à la couleur.")
    print("Le premier appel UMAP peut inclure une compilation JIT : ces temps ne constituent pas un benchmark de vitesse généralisable.")
    if plot:
        plot_embeddings(embeddings, y)
    return embeddings, table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=750)
    args = parser.parse_args()
    run(args.plot, args.samples, args.iterations)
