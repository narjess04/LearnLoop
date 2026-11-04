# Cours 6 — Évaluation des algorithmes de clustering

Repère de roadmap : 60 min de cours. Le support approfondi fourni annonce 90–120 min ; prévoir ce temps supplémentaire pour les démonstrations et les manipulations.

## 1. Pourquoi évaluer le clustering est-il difficile ?

### 1.1 Le problème de l'absence de labels

Imaginons que nous appliquions K-Means sur un dataset de clients.

Nous obtenons trois groupes :
- Cluster 0 : 120 clients.
- Cluster 1 : 180 clients.
- Cluster 2 : 100 clients.
Mais comment savoir si ces groupes représentent réellement des profils différents ?

Un algorithme peut toujours produire une partition. Cela ne signifie pas nécessairement que cette partition est pertinente.

L'évaluation doit examiner plusieurs aspects :
- Cohésion : les observations d'un même cluster sont-elles proches ?
- Séparation : les clusters sont-ils suffisamment distincts ?
- Stabilité : les résultats restent-ils similaires lorsque les données changent légèrement ?
- Interprétabilité : les groupes ont-ils un sens métier ?
- Robustesse : les résultats sont-ils sensibles aux hyperparamètres et aux valeurs extrêmes ?

### 1.2 Exemple visuel

Deux segmentations possibles
Bonne séparation
Groupes compacts et éloignés

Forte superposition
Groupes moins distincts

Illustrations conceptuelles de deux partitions. La qualité dépend également de la géométrie des données et de l'objectif recherché.

## 2. Les trois grandes familles d'évaluation

### 2.1 Métriques internes

Les métriques internes utilisent uniquement :
\[
X,\quad \hat y
\]

où :
- \(X\) représente les caractéristiques.
- \(\hat y\) représente les clusters prédits.
Elles n'ont pas besoin de véritables labels.

Exemples :
- Silhouette Score.
- Davies-Bouldin Index.
- Calinski-Harabasz Index.

### 2.2 Métriques externes

Les métriques externes comparent les clusters obtenus à une partition de référence.

Elles nécessitent :
\[
y_{\text{true}},\quad \hat y
\]

Exemples :
- Adjusted Rand Index (ARI).
- Normalized Mutual Information (NMI).
- Adjusted Mutual Information (AMI).
Ces métriques sont utiles sur des datasets de benchmark disposant de labels connus.

Attention : les labels de référence ne sont pas nécessairement des groupes naturellement détectables dans l'espace des caractéristiques.

### 2.3 Évaluation relative

L'évaluation relative consiste à comparer plusieurs modèles ou configurations.

Par exemple :
- K-Means avec \(K=3\).
- K-Means avec \(K=4\).
- DBSCAN avec différentes valeurs de eps.
- GMM avec différentes structures de covariance.
Il ne s'agit pas d'une métrique unique, mais d'une démarche de sélection.

Partie I — Les métriques internes

## 3. Silhouette Score

Le Silhouette Score est l'une des métriques les plus utilisées pour évaluer le clustering.

Il mesure simultanément :
- La cohésion intra-cluster.
- La séparation inter-clusters.

### 3.1 Définition mathématique

Pour une observation \(x_i\), on définit :
\[
a(i)=\text{distance moyenne aux autres points de son cluster}
\]

\[
b(i)=\text{plus petite distance moyenne aux points d'un autre cluster}
\]

Le coefficient de silhouette est :
\[
\boxed{s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}}
\]

Il appartient à l'intervalle :
\[
-1\leq s(i)\leq1
\]

### 3.2 Interprétation

Valeur	Signification
Proche de 1	Observation bien intégrée à son cluster
Proche de 0	Observation située près d'une frontière
Négative	Observation potentiellement mieux placée dans un autre cluster

Le score global correspond à la moyenne :
\[
\boxed{S=\frac1n\sum_{i=1}^{n}s(i)}
\]

Pour un cluster contenant une seule observation, la silhouette de cette observation est conventionnellement fixée à 0.

### 3.3 Exemple numérique

Supposons :
\[
a(i)=2,\qquad b(i)=5
\]

Alors :
\[
s(i)=\frac{5-2}{5}=0.6
\]

Le point est relativement bien affecté.

À l'inverse, si :
\[
a(i)=6,\qquad b(i)=3
\]

Alors :
\[
s(i)=\frac{3-6}{6}=-0.5
\]

Cela indique que l'observation est, en moyenne, plus proche d'un autre cluster.

### 3.4 Laboratoire interactif

[[silhouettelab]]

### 3.5 Implémentation Python

~~~python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
print(f"Silhouette Score : {silhouette_score(X, labels):.3f}")
~~~

### 3.6 Limites du Silhouette Score

Le Silhouette Score favorise souvent les groupes compacts et bien séparés.

Il peut donc pénaliser des clusters de formes complexes pourtant correctement identifiés par DBSCAN.

Par exemple, deux croissants bien séparés peuvent produire un score modeste parce que certaines observations d'un même croissant sont géométriquement éloignées.

Autre limitation : le calcul exact peut être coûteux sur de grands datasets, puisqu'il repose sur des distances entre observations.

## 4. Davies-Bouldin Index (DBI)

Le Davies-Bouldin Index mesure la similarité entre clusters en combinant leur dispersion interne et la distance entre leurs centres.

Contrairement au Silhouette Score :

Plus le Davies-Bouldin Index est faible, meilleure est généralement la séparation selon ce critère.

### 4.1 Formulation mathématique

Pour chaque cluster \(C_i\), nous définissons sa dispersion :
\[
S_i=\frac{1}{|C_i|}\sum_{x\in C_i}\|x-\mu_i\|
\]

La distance entre les centres de deux clusters est :
\[
M_{ij}=\|\mu_i-\mu_j\|
\]

Le rapport de similarité est :
\[
R_{ij}=\frac{S_i+S_j}{M_{ij}}
\]

Le Davies-Bouldin Index est :
\[
\boxed{DBI=\frac1K\sum_{i=1}^{K}\max_{j\neq i}R_{ij}}
\]

### 4.2 Intuition

Un bon clustering devrait avoir :
- Une faible dispersion interne \(S_i\).
- Une grande distance entre les centres \(M_{ij}\).
Le rapport \(R_{ij}\) devient alors faible.

### 4.3 Exemple numérique

Considérons deux clusters :
\[
S_1=1,\quad S_2=1.5,\quad M_{12}=5
\]

Alors :
\[
R_{12}=\frac{1+1.5}{5}=0.5
\]

Si la distance entre les centres diminue à 2 :
\[
R_{12}=\frac{2.5}{2}=1.25
\]

Le score augmente, ce qui traduit une séparation moins favorable.

### 4.4 Implémentation Python

~~~python
from sklearn.metrics import davies_bouldin_score
dbi = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index : {dbi:.3f}")
~~~

### 4.5 Limites

Le Davies-Bouldin Index repose sur les centroïdes et la dispersion autour de ces centres.

Il est donc généralement plus adapté aux clusters compacts qu'aux structures très irrégulières.

Il ne constitue pas une mesure universelle de la qualité d'un clustering.

## 5. Calinski-Harabasz Index (CHI)

Le Calinski-Harabasz Index est également appelé Variance Ratio Criterion.

Il compare :
- La dispersion entre les clusters.
- La dispersion à l'intérieur des clusters.
Plus le score est élevé, meilleur est généralement le clustering selon ce critère.

### 5.1 Formulation mathématique

Soit :
- \(n\) : nombre total d'observations.
- \(K\) : nombre de clusters.
- \(\mu\) : moyenne globale.
- \(\mu_k\) : moyenne du cluster \(k\).
La dispersion intra-cluster est :
\[
W=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|^2
\]

La dispersion inter-clusters est :
\[
B=\sum_{k=1}^{K}|C_k|\|\mu_k-\mu\|^2
\]

Le score est :
\[
\boxed{CHI=\frac{B/(K-1)}{W/(n-K)}}
\]

Cette formule s'applique lorsque :
\[
2\leq K\leq n-1
\]

### 5.2 Interprétation

Un bon clustering possède idéalement :
- Une dispersion intra-cluster faible.
- Une dispersion inter-clusters élevée.
Le rapport devient alors important.

### 5.3 Implémentation Python

~~~python
from sklearn.metrics import calinski_harabasz_score
chi = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index : {chi:.2f}")
~~~

### 5.4 Limites

Le CHI peut favoriser certaines partitions avec des groupes compacts.

Sa valeur dépend également de la structure du dataset et du nombre de clusters.

Il est surtout utile pour comparer des configurations sur un même jeu de données, avec un prétraitement cohérent.

## 6. Comparaison des métriques internes

Métrique	Valeur recherchée	Mesure principale
Silhouette	Maximiser	Cohésion et séparation
Davies-Bouldin	Minimiser	Dispersion et proximité des centres
Calinski-Harabasz	Maximiser	Rapport de dispersions inter/intra
Inertie K-Means	Minimiser à K fixé	Compacité des clusters

Attention : l’inertie optimale est non croissante avec K ; des ajustements indépendants peuvent toutefois atteindre des minima locaux différents. Elle ne doit donc pas être utilisée seule pour sélectionner \(K\).

### 6.1 Exemple de comparaison

Résultats fictifs de trois configurations
Modèle	Silhouette ↑	DBI ↓	CHI ↑
K-Means	0.62	0.48	810
Ward	0.59	0.52	760
GMM	0.56	0.61	720

Exemple purement illustratif, calculé sur aucune donnée réelle.
Dans cet exemple, K-Means présente les meilleurs résultats sur les trois métriques.

Mais cela ne suffit pas à conclure qu'il est systématiquement le meilleur choix.

Il faudrait également examiner la stabilité et la pertinence des groupes.

Partie II — Les métriques externes

## 7. Adjusted Rand Index (ARI)

L'Adjusted Rand Index permet de comparer deux partitions d'un même ensemble d'observations.

Il est particulièrement utile lorsque nous disposons de véritables labels.

### 7.1 Intuition du Rand Index

Pour chaque paire d'observations, nous pouvons vérifier si les deux partitions sont d'accord :
- Les deux observations appartiennent au même groupe dans les deux partitions.
- Elles appartiennent à des groupes différents dans les deux partitions.
Le Rand Index mesure la proportion d'accords.

Cependant, il ne corrige pas les accords obtenus par hasard.

### 7.2 Adjusted Rand Index

L'ARI corrige cette limite.
\[
\boxed{ARI=\frac{RI-\mathbb E[RI]}{\max(RI)-\mathbb E[RI]}}
\]

Il existe une formulation plus détaillée basée sur une table de contingence.

Soit \(n_{ij}\) le nombre d'observations appartenant à la classe réelle \(i\) et au cluster prédit \(j\).

On définit :
\[
a_i=\sum_jn_{ij}
\]

\[
b_j=\sum_in_{ij}
\]

Alors :
\[
ARI=
\frac{
\sum_{ij}\binom{n_{ij}}2-
\frac{\sum_i\binom{a_i}2\sum_j\binom{b_j}2}{\binom n2}
}{
\frac12\left[\sum_i\binom{a_i}2+\sum_j\binom{b_j}2\right]-
\frac{\sum_i\binom{a_i}2\sum_j\binom{b_j}2}{\binom n2}
}
\]

### 7.3 Interprétation

ARI	Interprétation
1	Partitions identiques à une permutation des labels près
Proche de 0	Accord proche de celui attendu par hasard
Négatif	Accord inférieur à l'attente sous le modèle de hasard

L'ARI ne dépend pas des noms numériques attribués aux clusters.

Par exemple :

~~~python
y_true = [0, 0, 1, 1]
y_pred = [1, 1, 0, 0]
~~~

Les partitions sont identiques, même si les identifiants sont inversés.

### 7.4 Implémentation

~~~python
from sklearn.metrics import adjusted_rand_score
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [1, 1, 0, 0, 2, 2]
print("ARI :", adjusted_rand_score(y_true, y_pred))
~~~

Résultat :

~~~text
ARI : 1.0
~~~

## 8. Normalized Mutual Information (NMI)

La NMI mesure la quantité d'information partagée entre deux partitions.

Elle repose sur la théorie de l'information.

### 8.1 Entropie

L'entropie d'une variable discrète \(Y\) est :
\[
\boxed{H(Y)=-\sum_yP(y)\log P(y)}
\]

Elle mesure l'incertitude associée à cette variable.

### 8.2 Information mutuelle

L'information mutuelle entre deux variables \(Y\) et \(C\) est :
\[
\boxed{I(Y;C)=\sum_{y,c}P(y,c)\log\frac{P(y,c)}{P(y)P(c)}}
\]

Elle quantifie la réduction d'incertitude sur une partition lorsqu'on connaît l'autre.

### 8.3 Normalisation

Une normalisation courante est :
\[
\boxed{NMI=\frac{2I(Y;C)}{H(Y)+H(C)}}
\]

C'est la normalisation arithmétique utilisée par défaut dans normalized_mutual_info_score de scikit-learn.

Le score varie entre 0 et 1 :
- 0 : aucune information mutuelle.
- 1 : partitions identiques à une permutation des labels près.
Contrairement à l'ARI, la NMI n'est pas ajustée pour les accords dus au hasard.

Pour cela, nous pouvons utiliser l'Adjusted Mutual Information (AMI).

### 8.4 Implémentation

~~~python
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
nmi = normalized_mutual_info_score(y_true, y_pred)
ami = adjusted_mutual_info_score(y_true, y_pred)
print("NMI :", nmi)
print("AMI :", ami)
~~~

### 8.5 ARI vs NMI

Critère	ARI	NMI
Fondement	Accords entre paires	Théorie de l'information
Correction du hasard	Oui	Non
Valeur maximale	1	1
Valeurs négatives	Possibles	Non
Permutation des labels	Invariant	Invariant

Partie III — Évaluation avancée

## 9. Comment évaluer DBSCAN et HDBSCAN ?

DBSCAN et HDBSCAN peuvent produire des observations étiquetées -1, correspondant au bruit.

Cela crée un problème particulier.

Si nous calculons directement un Silhouette Score sur toutes les observations, le label -1 peut être considéré comme un cluster supplémentaire.

Cela peut produire une interprétation trompeuse.

### 9.1 Exclure le bruit pour certaines métriques

~~~python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

labels = DBSCAN(eps=0.4, min_samples=5).fit_predict(X)
mask = labels != -1
n_clusters = len(set(labels[mask]))
print("Clusters :", n_clusters)
print(f"Noise ratio : {np.mean(~mask):.2%}")
if 2 <= n_clusters < np.sum(mask):
    print("Silhouette hors bruit :", silhouette_score(X[mask], labels[mask]))
else:
    print("Silhouette non applicable")
~~~

### 9.2 Pourquoi le taux de bruit est-il important ?

Imaginons deux modèles :
Modèle	Silhouette hors bruit	Bruit
DBSCAN A	0.72	8 %
DBSCAN B	0.89	65 %

Le modèle B possède un meilleur Silhouette Score.

Mais il exclut 65 % des observations.

Peut-on réellement considérer qu'il est meilleur ?

Pas nécessairement.

Le modèle B peut simplement conserver quelques groupes très compacts et rejeter une grande partie du dataset.

Il est donc essentiel d'évaluer simultanément :
- La qualité des clusters conservés.
- Le pourcentage de bruit.
- La couverture des observations.
- La stabilité des résultats.
- La pertinence métier.

### 9.3 DBCV — Density-Based Clustering Validation

Pour les algorithmes basés sur la densité, une métrique spécialisée appelée DBCV (Density-Based Clustering Validation) peut être pertinente.

Elle cherche à mesurer la qualité des regroupements en tenant compte de la densité et de la séparation des clusters.

Elle peut être mieux adaptée aux groupes de formes complexes que les métriques reposant sur les centroïdes.

Cependant, son calcul et sa disponibilité dépendent de l'implémentation utilisée. Elle n'est pas directement fournie comme métrique standard par sklearn.metrics.

## 10. Stabilité des clusters

Une segmentation de qualité devrait idéalement rester relativement cohérente lorsque nous modifions légèrement les données ou les conditions d'entraînement.

### 10.1 Pourquoi mesurer la stabilité ?

Imaginons que nous entraînions K-Means sur un dataset de clients.

Avec random_state=42, nous obtenons trois groupes.

Avec random_state=10, les résultats changent fortement.

Cela peut indiquer que le modèle est sensible à son initialisation.

Cependant, les différences peuvent également provenir de la structure du dataset.

### 10.2 Stabilité face à l'initialisation

Nous pouvons entraîner plusieurs fois K-Means avec des graines aléatoires différentes.

Puis comparer les partitions avec l'ARI.

~~~python
def initialization_stability(X, k=4):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    labels = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(X) for seed in range(10)]
    scores = [adjusted_rand_score(labels[i], labels[j]) for i in range(10) for j in range(i + 1, 10)]
    return {"ARI_mean": float(np.mean(scores)), "ARI_min": float(np.min(scores)), "comparisons": len(scores)}

print("Stabilité sur les mêmes observations :", initialization_stability(X))
~~~

Nous utilisons volontairement n_init=1 pour observer la sensibilité à l'initialisation.

En pratique, plusieurs initialisations sont généralement préférables.

### 10.3 Stabilité par rééchantillonnage

Une autre approche consiste à :
1. Prélever plusieurs échantillons du dataset.
2. Réentraîner le modèle sur chaque échantillon.
3. Comparer les affectations sur un ensemble commun d'observations.
4. Calculer la similarité des partitions.
Il faut faire attention à ne pas comparer directement deux tableaux de labels provenant d'échantillons contenant des observations différentes.

L'ARI nécessite que les deux partitions portent sur les mêmes observations.

## 11. Construire un pipeline complet d'évaluation

Nous allons maintenant comparer plusieurs algorithmes sur un même dataset.

### 11.1 Génération des données

~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=[0.7, 1.0, 0.8, 1.2], random_state=42)
X_scaled = StandardScaler().fit_transform(X)
~~~

### 11.2 Entraînement des modèles

~~~python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

models = {
    "K-Means": KMeans(n_clusters=4, n_init=10, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=4, linkage="ward"),
    "DBSCAN": DBSCAN(eps=0.35, min_samples=5),
    "GMM": GaussianMixture(n_components=4, n_init=10, reg_covar=1e-6, random_state=42),
}
~~~

### 11.3 Calcul des métriques

~~~python
def evaluate_clustering(X, labels, y_true=None):
    """Internes hors bruit ; métriques externes globales et périmètres explicites."""
    import numpy as np
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),) or not len(X) or not np.isfinite(X).all():
        raise ValueError("Matrice finie et labels alignés requis.")
    mask = labels != -1
    k = len(np.unique(labels[mask]))
    result = {"clusters": k, "noise_ratio": float(np.mean(~mask)),
              "coverage": float(np.mean(mask)), "n_internal": int(mask.sum()),
              "silhouette": np.nan, "DBI": np.nan, "CHI": np.nan}
    if 2 <= k < int(mask.sum()):
        result.update(silhouette=float(silhouette_score(X[mask], labels[mask])),
                      DBI=float(davies_bouldin_score(X[mask], labels[mask])),
                      CHI=float(calinski_harabasz_score(X[mask], labels[mask])))
    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.shape != labels.shape:
            raise ValueError("La référence doit porter sur les mêmes observations.")
        result["ARI_all"] = float(adjusted_rand_score(y_true, labels))
        result["NMI_all"] = float(normalized_mutual_info_score(y_true, labels))
        result["ARI_non_noise"] = float(adjusted_rand_score(y_true[mask], labels[mask])) if mask.any() else np.nan
    return result

results = []
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    results.append({"model": name, **evaluate_clustering(X_scaled, labels, y_true)})
print(pd.DataFrame(results).to_string(index=False))
~~~

Point méthodologique important : dans ce code, les métriques internes sont calculées sur les observations non bruitées, tandis que l'ARI et la NMI sont calculés sur toutes les observations.

Ces scores ne mesurent donc pas exactement le même périmètre.

Pour une comparaison rigoureuse, il faut documenter cette différence et tenir compte du taux de bruit.

De plus, les véritables labels y_true sont disponibles ici uniquement parce que nous avons créé un dataset synthétique. Dans un problème réel non supervisé, ils peuvent être totalement absents.

## 12. Comment sélectionner le meilleur modèle ?

Il n'existe pas de métrique universelle permettant de déterminer automatiquement le meilleur clustering.

Une bonne démarche consiste à examiner plusieurs dimensions.
Cadre de décision pour choisir un clustering
Qualité géométrique

Silhouette, Davies-Bouldin, Calinski-Harabasz.

Stabilité

Résistance aux initialisations et aux rééchantillonnages.

Couverture

Proportion d'observations regroupées et taux de bruit.

Interprétabilité

Profils cohérents et exploitables dans le contexte métier.

Coût computationnel

Temps d'entraînement, mémoire et capacité à traiter de nouvelles données.

Un modèle légèrement moins performant sur le Silhouette Score peut être préférable s'il est plus stable, plus interprétable et plus adapté aux contraintes opérationnelles.

## 13. Erreurs fréquentes à éviter

Erreur 1 — Utiliser l'Accuracy pour évaluer directement un clustering. Les identifiants de clusters sont arbitraires. L'Accuracy brute n'est donc pas adaptée sans mise en correspondance des labels.

Erreur 2 — Comparer des Silhouette Scores calculés avec des distances ou des prétraitements différents. Les conditions d'évaluation doivent être cohérentes.

Erreur 3 — Considérer le score le plus élevé comme une preuve absolue. Les métriques internes reposent sur certaines hypothèses géométriques.

Erreur 4 — Ignorer les points de bruit. Pour DBSCAN et HDBSCAN, le taux de bruit peut modifier fortement l'interprétation des résultats.

Erreur 5 — Confondre stabilité et pertinence. Un modèle peut être très stable tout en produisant des groupes sans intérêt métier.

Erreur 6 — Choisir les hyperparamètres uniquement sur les métriques internes. Il faut également considérer la structure des données, la robustesse et l'objectif de l'application.

Erreur 7 — Utiliser des labels de référence pour choisir un modèle tout en présentant l'évaluation comme entièrement non supervisée. Dès que les labels guident la sélection, ils interviennent dans la procédure d'évaluation.

## 14. Mini-projet LearnLoop — Clustering Benchmark Framework

Projet : Clustering Evaluation Benchmark
Intermédiaire → Avancé

Contexte : tu dois développer un système capable de comparer automatiquement plusieurs algorithmes de clustering.

Travail demandé :
1. Générer trois datasets synthétiques : blobs, moons et cercles.
2. Appliquer K-Means, Agglomerative Clustering, DBSCAN et GMM.
3. Tester plusieurs configurations d'hyperparamètres.
4. Calculer Silhouette, DBI et CHI.
5. Calculer ARI et NMI à l'aide des labels synthétiques.
6. Mesurer le taux de bruit.
7. Étudier la stabilité des partitions.
8. Visualiser les résultats de chaque algorithme.
9. Construire un tableau comparatif.
10. Justifier le meilleur choix pour chaque dataset.
[[solution]]

Livrable : un notebook Python et un tableau de bord comparatif présentant les métriques, les visualisations et les conclusions.

## 15. Questions d’entretien technique

[[interview]]

## 16. Quiz de validation

[[quiz]]

## 17. Synthèse du cours

À la fin de ce cours, tu dois maîtriser les éléments suivants :
Concept	À retenir
Métriques internes	Évaluation sans labels de référence
Métriques externes	Comparaison avec une partition de référence
Silhouette Score	Cohésion et séparation ; maximiser
Davies-Bouldin	Dispersion relative ; minimiser
Calinski-Harabasz	Rapport inter/intra ; maximiser
ARI	Accord entre partitions corrigé du hasard
NMI	Information partagée entre partitions
AMI	Information mutuelle ajustée du hasard
Stabilité	Cohérence face aux perturbations
Noise Ratio	Proportion d'observations non regroupées

La distinction essentielle : un clustering de qualité ne se résume pas à un score élevé. Il doit également être stable, interprétable et adapté à la structure des données et à l'objectif métier.

## 18. Ma progression

[[assessment]]

Prochaine étape : Cours 7 — Analyse en composantes principales (PCA) (support à venir).
