# Cours 3 — Clustering hiérarchique

Objectif du cours : comprendre le fonctionnement du clustering hiérarchique, maîtriser les différentes méthodes de liaison (linkage), interpréter un dendrogramme, choisir le nombre de clusters et implémenter l'algorithme en Python.

Dans le cours précédent, nous avons étudié K-Means, un algorithme qui nécessite de fixer le nombre de clusters \(K\) avant l'entraînement.

Mais imaginons que nous ne connaissions pas ce nombre. Et si nous souhaitions comprendre non seulement quels individus se ressemblent, mais aussi comment les groupes se construisent progressivement ?

C'est précisément ce que permet le clustering hiérarchique.

## 1. Introduction et intuition

Le clustering hiérarchique (Hierarchical Clustering) est une famille d'algorithmes d'apprentissage non supervisé permettant de construire une hiérarchie de groupes à partir des similarités entre observations.

Contrairement à K-Means, il ne repose pas nécessairement sur des centroïdes et permet de représenter les relations entre groupes sous forme d'un arbre appelé dendrogramme.

### 1.1 Exemple intuitif

Imaginons six clients, caractérisés par leur comportement d'achat.
Exemple fictif de six clients. Les couleurs servent uniquement à illustrer les groupes visuellement proches.
Nous observons naturellement trois paires : A-B, C-D et E-F.

Le clustering hiérarchique va construire progressivement une organisation des observations :
- Il identifie les observations les plus proches.
- Il les regroupe selon un critère de liaison.
- Il répète l'opération sur les groupes obtenus.
- Il construit une hiérarchie complète.
Cette hiérarchie peut ensuite être découpée à différents niveaux pour obtenir deux, trois ou davantage de clusters.

## 2. Les deux approches du clustering hiérarchique

Il existe deux grandes stratégies.

### 2.1 Clustering agglomératif (bottom-up)

Chaque observation commence dans son propre cluster. À chaque itération, les deux clusters les plus proches sont fusionnés.

C'est l'approche la plus couramment utilisée.

### 2.2 Clustering divisif (top-down)

Toutes les observations commencent dans un seul cluster. L'algorithme divise progressivement les groupes jusqu'à atteindre le niveau souhaité.

Dans ce cours, nous allons nous concentrer sur le clustering hiérarchique agglomératif, notamment celui disponible dans scikit-learn.

## 3. Fonctionnement de l'algorithme agglomératif

Supposons que nous disposions de \(n\) observations.

Initialement :
\[
C_i=\{x_i\},\qquad i=1,\ldots,n
\]

Chaque observation constitue un cluster indépendant.

L'algorithme suit les étapes suivantes :
Initialisation

Créer \(n\) clusters individuels

Calcul des distances

Mesurer les distances entre les clusters

Fusion

Fusionner les deux clusters les plus proches

Mise à jour

Recalculer les distances entre les nouveaux groupes

Condition d'arrêt

Continuer jusqu'au nombre de clusters souhaité ou jusqu'à obtenir un seul cluster

Pour une hiérarchie complète, il faut effectuer exactement :
\[
\boxed{n-1\text{ fusions}}
\]

Par exemple, pour six observations, cinq fusions sont nécessaires pour obtenir un seul groupe.

## 4. Les mesures de distance

Avant de regrouper les observations, nous devons mesurer leur similarité.

### 4.1 Distance euclidienne

Pour deux observations \(x_i,x_j\in\mathbb R^p\) :
\[
d(x_i,x_j)=\sqrt{\sum_{m=1}^{p}(x_{im}-x_{jm})^2}
\]

C'est la distance géométrique classique.

### 4.2 Distance de Manhattan

\[
d(x_i,x_j)=\sum_{m=1}^{p}|x_{im}-x_{jm}|
\]

Elle additionne les écarts absolus sur chaque caractéristique.

### 4.3 Distance cosinus

La distance cosinus peut s'écrire :
\[
d_{\cos}(x_i,x_j)=1-\frac{x_i\cdot x_j}{\|x_i\|\|x_j\|}
\]

Elle mesure une différence d'orientation plutôt qu'une distance géométrique classique.

Elle est notamment utile pour comparer des représentations vectorielles de textes, à condition que les vecteurs ne soient pas nuls.

Point important : le choix de la métrique influence directement les regroupements. Toutes les méthodes de liaison ne sont pas compatibles avec toutes les distances. Par exemple, la méthode de Ward utilise la distance euclidienne.

## 5. Les méthodes de liaison (Linkage Methods)

C'est la partie mathématique centrale du cours.

Une fois que plusieurs observations ont été regroupées, comment calculer la distance entre deux clusters ?

Il existe différentes méthodes.

### 5.1 Single Linkage

La distance entre deux clusters est la plus petite distance entre leurs observations.
\[
\boxed{D(A,B)=\min_{x\in A,y\in B}d(x,y)}
\]

Avantage : peut identifier des groupes de formes allongées ou irrégulières.

Inconvénient : sensible à l'effet de chaînage (chaining effect). Quelques points intermédiaires peuvent relier des groupes pourtant distincts.

### 5.2 Complete Linkage

La distance entre deux clusters est la plus grande distance entre leurs observations.
\[
\boxed{D(A,B)=\max_{x\in A,y\in B}d(x,y)}
\]

Avantage : tend à produire des groupes compacts.

Inconvénient : peut être sensible aux observations extrêmes.

### 5.3 Average Linkage

La distance entre deux clusters correspond à la moyenne des distances entre toutes les paires d'observations appartenant aux deux groupes.
\[
\boxed{D(A,B)=\frac{1}{|A||B|}\sum_{x\in A}\sum_{y\in B}d(x,y)}
\]

Cette méthode représente un compromis entre Single et Complete Linkage.

Elle prend en compte l'ensemble des observations plutôt que seulement la paire la plus proche ou la plus éloignée.

### 5.4 Ward Linkage

La méthode de Ward cherche à minimiser l'augmentation de la somme des carrés intra-cluster provoquée par chaque fusion.

Pour deux clusters \(A\) et \(B\), l'augmentation d'inertie est :
\[
\boxed{\Delta(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2}
\]

où :
- \(|A|\) et \(|B|\) sont les effectifs des clusters.
- \(\mu_A\) et \(\mu_B\) sont leurs centroïdes.
À chaque étape, Ward fusionne la paire qui entraîne la plus faible augmentation d'inertie.

Cette méthode est particulièrement intéressante lorsque nous recherchons des groupes relativement compacts.

### 5.5 Comparaison des méthodes

Méthode	Critère	Caractéristique
Single	Distance minimale	Chaînage possible
Complete	Distance maximale	Groupes compacts
Average	Distance moyenne	Compromis
Ward	Augmentation d'inertie	Groupes souvent compacts

Le choix doit dépendre de la géométrie des données, de la métrique utilisée et de l'objectif métier.

## 6. Exemple mathématique complet

Considérons quatre observations unidimensionnelles :
\[
A=1,\quad B=2,\quad C=6,\quad D=8
\]

Nous utiliserons la distance euclidienne et Single Linkage.

### 6.1 Matrice des distances

\[
D=
\begin{pmatrix}
0&1&5&7\\
1&0&4&6\\
5&4&0&2\\
7&6&2&0
\end{pmatrix}
\]

### 6.2 Première fusion

La distance minimale est :
\[
d(A,B)=1
\]

Nous fusionnons donc A et B.

Les clusters deviennent :
\[
\{A,B\},\{C\},\{D\}
\]

### 6.3 Deuxième fusion

La distance entre C et D est :
\[
d(C,D)=2
\]

Nous fusionnons C et D.

Les clusters deviennent :
\[
\{A,B\},\{C,D\}
\]

### 6.4 Troisième fusion

Avec Single Linkage :
\[
D(\{A,B\},\{C,D\})=\min(5,7,4,6)
\]

\[
\boxed{D=4}
\]

Nous obtenons finalement un seul cluster.
Dendrogramme de notre exemple
Les fusions se produisent aux hauteurs 1, 2 et 4. Une coupe strictement entre 1 et 2 produit trois clusters ; strictement entre 2 et 4, deux clusters. À 1,5, on obtient donc trois groupes ; à 2,5, deux groupes.

[[dendrogram]]
Le dendrogramme nous permet de visualiser l'ordre des fusions et les distances auxquelles elles se produisent.

## 7. Comprendre les dendrogrammes

Un dendrogramme est une représentation arborescente de la hiérarchie construite par l'algorithme.

Il contient trois éléments principaux :
- Les feuilles : les observations individuelles.
- Les branches : les regroupements successifs.
- La hauteur des fusions : la distance ou le critère de liaison associé à chaque regroupement.
Comment déterminer le nombre de clusters ?
Une méthode consiste à tracer une ligne horizontale à une certaine hauteur.

Le nombre de branches verticales coupées correspond au nombre de clusters obtenus.

Attention : un grand saut dans les hauteurs de fusion peut suggérer une séparation intéressante, mais il ne garantit pas que le nombre de clusters soit optimal.

## 8. Implémentation Python avec SciPy

Nous allons générer un dataset puis construire un dendrogramme.

~~~python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.8, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
Z = linkage(X_scaled, method="ward", metric="euclidean")
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=[str(i) for i in range(len(X))])
plt.title("Hierarchical Clustering — Ward")
plt.xlabel("Observations")
plt.ylabel("Hauteur SciPy de fusion")
plt.tight_layout()
plt.show()
~~~

### 8.1 Comprendre la matrice de liaison

La fonction linkage() retourne une matrice \(Z\) de dimension :
\[
(n-1)\times4
\]

Chaque ligne contient quatre informations :
Colonne	Signification
Z[i,0]	Premier groupe fusionné
Z[i,1]	Deuxième groupe fusionné
Z[i,2]	Hauteur de fusion
Z[i,3]	Nombre d'observations dans le nouveau groupe

Les nouveaux clusters reçoivent des identifiants à partir de \(n\).

Avec Ward, les hauteurs fournies par SciPy suivent sa convention de distance de Ward et ne sont pas directement égales à l'augmentation brute d'inertie \(\Delta\).

## 9. Implémentation avec scikit-learn

~~~python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X_scaled)
print("Clusters :", set(labels))
print("Silhouette :", silhouette_score(X_scaled, labels))
~~~

Nous pouvons visualiser les résultats :

~~~python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", alpha=0.7)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
~~~

## 10. Comparaison pratique des méthodes de liaison

Nous pouvons comparer les quatre méthodes sur le même dataset.

~~~python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

for method in ("single", "complete", "average", "ward"):
    model = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"{method:10s} | Silhouette: {score:.3f}")
~~~

Ce test permet de comparer la cohésion et la séparation des groupes obtenus.

Cependant, un score plus élevé ne signifie pas automatiquement que la segmentation est plus pertinente pour une application métier.

## 11. Comparaison : K-Means vs Clustering hiérarchique

Critère	K-Means	Hiérarchique
Nombre de clusters	Fixé avant l'entraînement	Peut être choisi en coupant l'arbre
Principe	Centroïdes	Fusions ou divisions
Résultat	Partition	Hiérarchie de partitions
Dendrogramme	Non	Oui
Grands datasets	Généralement plus adapté	Souvent coûteux
Formes complexes	Limité	Dépend du linkage
Interprétation	Centroïdes	Relations hiérarchiques

Complexité algorithmique
Pour K-Means, une approximation courante est :
\[
O(nKpT)
\]

où \(T\) représente le nombre d'itérations.

Pour le clustering hiérarchique agglomératif, de nombreuses implémentations classiques nécessitent une mémoire de l'ordre de :
\[
O(n^2)
\]

Le temps d'exécution dépend fortement de la méthode de liaison et de l'implémentation. Certaines implémentations optimisées atteignent \(O(n^2)\), tandis que des approches naïves peuvent nécessiter \(O(n^3)\).

Cela explique pourquoi le clustering hiérarchique peut devenir coûteux sur de très grands datasets.

## 12. Bonnes pratiques et erreurs fréquentes

Erreur 1 : oublier la standardisation. Une variable présentant une grande échelle peut dominer les distances.

Erreur 2 : choisir Ward avec une distance incompatible. Ward repose sur la géométrie euclidienne et la minimisation de l'augmentation d'inertie.

Erreur 3 : interpréter toutes les branches du dendrogramme comme des groupes pertinents. La hiérarchie est mathématique, mais les clusters doivent être validés selon le contexte.

Erreur 4 : utiliser systématiquement le clustering hiérarchique sur des millions d'observations. Le coût mémoire et calculatoire peut devenir prohibitif.

Erreur 5 : considérer le Silhouette Score comme une preuve absolue. Il faut également examiner la stabilité, la taille des groupes et leur interprétabilité.

## 13. Mini-projet pratique — LearnLoop

Customer Segmentation — Hierarchical Clustering
Intermédiaire

Contexte : une entreprise souhaite segmenter ses clients en fonction de leurs comportements d'achat.

Dataset suggéré : Mall Customers, ou un dataset synthétique comportant âge, revenu annuel, fréquence d'achat et montant moyen dépensé.

Travail demandé :

1. Explorer les données et identifier les variables pertinentes.

2. Gérer les valeurs manquantes et standardiser les variables numériques.

3. Construire un dendrogramme avec Ward.

4. Tester plusieurs nombres de clusters.

5. Comparer Single, Complete, Average et Ward.

6. Calculer le Silhouette Score.

7. Visualiser les groupes et interpréter leurs caractéristiques.

8. Comparer les résultats avec K-Means.

[[solution]]

Livrable : un notebook Python contenant les visualisations, les métriques, les interprétations et une recommandation argumentée.

## 14. Questions d’entretien technique

[[interview]]

## 15. Quiz — Vérification des connaissances

[[quiz]]

## 16. Synthèse du cours

À la fin de ce troisième cours, tu dois être capable de :
- Expliquer les approches agglomérative et divisive.
- Calculer manuellement les premières fusions d'un clustering hiérarchique.
- Distinguer les méthodes Single, Complete, Average et Ward.
- Construire et interpréter un dendrogramme.
- Implémenter le clustering hiérarchique avec SciPy et scikit-learn.
- Évaluer la qualité des clusters et comparer les résultats avec K-Means.
Prochain cours : Chapitre 4 — Cours 4 : DBSCAN et HDBSCAN.

Nous étudierons le clustering basé sur la densité, les notions de core points, border points et noise, les paramètres eps et min_samples, ainsi que les raisons pour lesquelles DBSCAN peut détecter des formes complexes que K-Means ne parvient pas à identifier.

## 17. Ma progression

[[assessment]]

La version AgglomerativeClustering utilisée ici ne fournit pas de méthode predict native : affecter de nouveaux clients demande une règle séparée à définir et à évaluer.
