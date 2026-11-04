# Cours 4 — DBSCAN et HDBSCAN

Repère de roadmap : 75 min de cours. Le support approfondi fourni annonce 90–120 min ; prévoir ce temps supplémentaire pour les démonstrations et les manipulations.

## 1. Introduction : pourquoi avons-nous besoin de DBSCAN ?

### 1.1 Rappel des limites de K-Means

K-Means regroupe les observations autour de centroïdes en minimisant la somme des distances euclidiennes au carré.

Cette approche fonctionne particulièrement bien lorsque les clusters sont compacts et relativement sphériques.

Mais imaginons un dataset contenant deux groupes en forme de croissants.
K-Means
Découpage selon la proximité aux centroïdes

DBSCAN
Regroupement selon les régions denses

Illustration conceptuelle : les couleurs montrent la différence de regroupement recherchée, et non le résultat d'une exécution réelle.
Problème : K-Means peut séparer incorrectement des groupes allongés ou incurvés, car il attribue chaque observation au centroïde le plus proche.

DBSCAN adopte une approche différente : au lieu de chercher des centres, il recherche des régions de forte densité séparées par des régions de faible densité.

### 1.2 Définition de DBSCAN

DBSCAN signifie :

Density-Based Spatial Clustering of Applications with Noise.

Il s'agit d'un algorithme de clustering basé sur la densité, introduit par Ester et ses collaborateurs en 1996.

Il possède trois caractéristiques majeures :
- Il ne nécessite pas de fixer le nombre de clusters à l'avance.
- Il peut identifier des clusters de formes arbitraires.
- Il peut considérer certaines observations comme du bruit plutôt que de les affecter obligatoirement à un groupe.
C'est particulièrement intéressant dans des applications telles que la détection d'anomalies, l'analyse spatiale ou la segmentation comportementale.

## 2. Les concepts fondamentaux de DBSCAN

Pour comprendre DBSCAN, nous devons maîtriser quatre notions : le voisinage, les points centraux, les points de bordure et les points de bruit.

### 2.1 Le voisinage \(\varepsilon\)

Pour une observation \(x_i\), on définit son voisinage de rayon \(\varepsilon\) :
\[
\boxed{N_\varepsilon(x_i)=\{x_j\in X:d(x_i,x_j)\leq\varepsilon\}}
\]

où :
- \(X\) représente l'ensemble des observations.
- \(d\) est une fonction de distance.
- \(\varepsilon\) est le rayon du voisinage.
Le voisinage inclut le point \(x_i\) lui-même.
[[densitylab]]

### 2.2 Core Point — Point central

Un point est considéré comme central lorsque son voisinage contient au moins min_samples observations.
\[
\boxed{|N_\varepsilon(x_i)|\geq MinPts}
\]

Dans scikit-learn, min_samples inclut le point lui-même.

Par exemple, avec min_samples=5, un point doit posséder au moins quatre autres observations dans son voisinage pour être central.

### 2.3 Border Point — Point de bordure

Un point de bordure :
- Ne possède pas suffisamment de voisins pour être central.
- Appartient au voisinage d'au moins un point central.
Il peut donc être rattaché à un cluster sans permettre à lui seul de prolonger ce cluster.

### 2.4 Noise Point — Point de bruit

Un point de bruit :
- N'est pas central.
- N'appartient au voisinage d'aucun point central.
Dans scikit-learn, les observations considérées comme du bruit reçoivent l'étiquette :
label = -1

Les trois types de points

Core

Border

Noise

Schéma illustratif des rôles des observations dans DBSCAN.

## 3. Fondements mathématiques : connectivité par densité

DBSCAN ne regroupe pas simplement des observations proches. Il utilise des relations de connectivité.

### 3.1 Directement accessible par densité

Un point \(q\) est directement accessible par densité depuis un point \(p\) lorsque :
\[
q\in N_\varepsilon(p)
\]

et :
\[
|N_\varepsilon(p)|\geq MinPts
\]

Autrement dit, \(p\) doit être un point central et \(q\) doit se trouver dans son voisinage.

Attention : cette relation n'est pas nécessairement symétrique. Si \(q\) est un point de bordure, \(q\) peut être directement accessible depuis \(p\), mais l'inverse n'est pas garanti.

### 3.2 Accessible par densité

Un point \(q\) est accessible par densité depuis \(p\) s'il existe une chaîne :
\[
p=p_1,p_2,\ldots,p_m=q
\]

telle que chaque \(p_{i+1}\) soit directement accessible par densité depuis \(p_i\).

Cette propriété permet à DBSCAN de construire des clusters allongés, incurvés ou de formes irrégulières.

### 3.3 Connecté par densité

Deux points \(p\) et \(q\) sont connectés par densité s'il existe un point \(o\) depuis lequel ils sont tous les deux accessibles par densité.

C'est cette notion qui permet de définir les clusters de DBSCAN.

## 4. Fonctionnement de l'algorithme DBSCAN

L'algorithme fonctionne de manière itérative.
Étape 1 — Sélectionner un point non visité

Étape 2 — Calculer son voisinage ε

Voisins insuffisants

Marquer comme bruit provisoire

Voisins suffisants

Créer un cluster

Étape 3 — Étendre le cluster à partir des points centraux

Étape 4 — Répéter jusqu'à visiter toutes les observations

Un point initialement marqué comme bruit peut être réaffecté ultérieurement à un cluster s'il est découvert dans le voisinage d'un point central.

### 4.1 Pseudocode

~~~text
DBSCAN(X, eps, min_samples):

    Pour chaque point P non visité :

        Marquer P comme visité

        Voisins = regionQuery(P, eps)

        Si nombre(Voisins) < min_samples :
            Marquer P comme bruit provisoire

        Sinon :
            Créer un nouveau cluster C
            Ajouter P à C

            Pour chaque point Q dans les voisins :
                Si Q n'est pas visité :
                    Marquer Q comme visité
                    VoisinsQ = regionQuery(Q, eps)

                    Si Q est un point central :
                        Ajouter ses voisins à explorer

                Si Q n'appartient à aucun cluster :
                    Ajouter Q à C

    Retourner les clusters et le bruit
~~~

## 5. Comprendre les paramètres eps et min_samples

Ce sont les deux hyperparamètres fondamentaux de DBSCAN.

### 5.1 Le paramètre eps

Il définit le rayon du voisinage.
- Un eps trop petit peut produire de nombreux points de bruit et fragmenter les clusters.
- Un eps trop grand peut fusionner plusieurs groupes distincts.

### 5.2 Le paramètre min_samples

Il définit le nombre minimal d'observations nécessaires pour qu'un point soit central.
- Une valeur faible permet de détecter de petites régions denses, mais peut rendre l'algorithme sensible au bruit.
- Une valeur élevée exige des régions plus densément peuplées.
Il n'existe pas de valeur universellement optimale.

### 5.3 Explorer l'influence des paramètres

Retrouve le laboratoire de voisinage plus haut pour modifier eps et min_samples et observer les rôles recalculés.

## 6. Comment choisir eps ? La méthode K-Distance

Une méthode courante consiste à analyser la distance au \(k\)-ième plus proche voisin.

On choisit généralement \(k\) en cohérence avec min_samples.

Pour chaque observation :
1. Calculer les distances à ses voisins.
2. Extraire la distance au \(k\)-ième voisin, en comptant l'observation elle-même.
3. Trier ces distances.
4. Rechercher une zone de changement marqué de pente.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=500, noise=0.07, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])
plt.plot(k_distances)
plt.xlabel("Observations triées")
plt.ylabel("Distance au 5e voisin (point lui-même inclus)")
plt.title("K-Distance Graph")
plt.grid(alpha=0.3)
plt.show()
~~~

Exemple illustratif de courbe K-Distance
Valeurs fictives montrant une augmentation marquée en fin de courbe.

Le coude peut suggérer une valeur de eps, mais il n'est pas toujours clairement identifiable, notamment lorsque les données présentent des densités différentes.

## 7. Implémentation de DBSCAN from scratch

Pour maîtriser un algorithme, il est important de comprendre comment l'implémenter sans utiliser directement sa version scikit-learn.

Nous allons construire une version pédagogique avec NumPy.

~~~python
import numpy as np
from collections import deque

class DBSCANScratch:
    """Version pédagogique euclidienne : distances et mémoire O(n²)."""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
            raise ValueError("Une matrice 2D finie non vide est requise.")
        if self.eps <= 0 or not isinstance(self.min_samples, (int, np.integer)) or self.min_samples < 1:
            raise ValueError("eps positif et min_samples entier >= 1 requis.")
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        neighbors = [np.flatnonzero(distances[i] <= self.eps) for i in range(n)]
        self.core_sample_indices_ = np.flatnonzero([len(ids) >= self.min_samples for ids in neighbors])
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if len(neighbors[i]) < self.min_samples:
                continue  # Bruit provisoire, réaffectable plus tard.
            labels[i] = cluster_id
            queue = deque(neighbors[i].tolist())
            scheduled = set(neighbors[i].tolist())
            while queue:
                q = queue.popleft()
                if not visited[q]:
                    visited[q] = True
                    if len(neighbors[q]) >= self.min_samples:
                        for neighbor in neighbors[q]:
                            neighbor = int(neighbor)
                            if neighbor not in scheduled:
                                queue.append(neighbor)
                                scheduled.add(neighbor)
                if labels[q] == -1:
                    labels[q] = cluster_id
            cluster_id += 1
        self.labels_ = labels
        return labels
~~~

Analyse du code
La matrice distances contient les distances entre toutes les paires d'observations.

Sa dimension est :
\[
D\in\mathbb R^{n\times n}
\]

Cela implique une complexité mémoire de :
\[
\boxed{O(n^2)}
\]

pour cette implémentation pédagogique.

La liste queue sert à explorer les observations accessibles depuis les points centraux. Elle permet d'étendre progressivement chaque cluster.

Cette version n'est pas optimisée pour de grands datasets. Les implémentations professionnelles utilisent notamment des structures de recherche de voisins adaptées.

## 8. Implémentation avec scikit-learn

Nous allons utiliser make_moons, un dataset classique composé de deux groupes en forme de croissants.

~~~python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=500, noise=0.07, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
model = DBSCAN(eps=0.3, min_samples=5, metric="euclidean")
labels = model.fit_predict(X_scaled)
print("Nombre de clusters :", len(set(labels) - {-1}))
print("Nombre de points de bruit :", np.sum(labels == -1))
~~~

Les valeurs de eps et min_samples sont ici des points de départ pour l'expérimentation, pas des paramètres optimaux garantis.

## 9. Comment évaluer DBSCAN ?

Contrairement à K-Means, DBSCAN peut produire du bruit et un nombre variable de clusters.

Cela rend son évaluation un peu plus délicate.

### 9.1 Silhouette Score

Le Silhouette Score peut être utilisé lorsqu'une partition valide contient au moins deux groupes.
\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

Mais attention : dans scikit-learn, les observations de bruit possèdent le label -1, qui peut être traité comme un groupe distinct lors du calcul.

Il est donc souvent préférable d'évaluer les clusters non bruités séparément, tout en rapportant la proportion de bruit.

~~~python
from sklearn.metrics import silhouette_score
import numpy as np

mask = labels != -1
n_clusters = len(set(labels[mask]))
print(f"Noise ratio: {np.mean(~mask):.2%}")
if 2 <= n_clusters < np.sum(mask):
    score = silhouette_score(X_scaled[mask], labels[mask])
    print(f"Silhouette hors bruit: {score:.3f}")
else:
    print("Silhouette non applicable")
~~~

Un score calculé après exclusion du bruit doit être interprété conjointement avec le taux de bruit : retirer beaucoup d'observations peut artificiellement faciliter la séparation des groupes restants.

### 9.2 Autres critères

Nous pouvons également analyser :
- Le nombre de clusters obtenus.
- La proportion d'observations considérées comme du bruit.
- La taille des clusters.
- La stabilité des résultats lorsque les paramètres changent.
- La pertinence métier des groupes.

Partie II — HDBSCAN

## 10. Pourquoi HDBSCAN ?

DBSCAN fonctionne bien lorsque les clusters possèdent des densités relativement comparables.

Mais imaginons deux groupes :
- Un groupe très dense avec de nombreuses observations proches.
- Un groupe plus dispersé avec des observations éloignées les unes des autres.
Avec un seul rayon eps, il peut être difficile d'identifier correctement les deux.

HDBSCAN signifie Hierarchical Density-Based Spatial Clustering of Applications with Noise.

Il étend les idées de DBSCAN en construisant une hiérarchie de structures de densité et en sélectionnant les clusters les plus persistants.
DBSCAN
Un seul seuil de voisinage

HDBSCAN
Peut distinguer des densités différentes

Schéma conceptuel. Les résultats effectifs dépendent des données et des hyperparamètres.

## 11. Fondements mathématiques de HDBSCAN

HDBSCAN utilise plusieurs concepts importants.

### 11.1 Core Distance

Pour une observation \(x\), la core distance correspond à la distance nécessaire pour atteindre un nombre défini de voisins.

On peut la représenter par :
\[
core_k(x)=d(x,x_{(k)})
\]

où \(x_{(k)}\) est le \(k\)-ième plus proche voisin, selon la convention de comptage adoptée.

Cette distance reflète la densité locale :
- Faible core distance : région dense.
- Grande core distance : région moins dense.

### 11.2 Mutual Reachability Distance

HDBSCAN définit une distance transformée :
\[
\boxed{d_{mreach}(a,b)=\max\left(core_k(a),core_k(b),d(a,b)\right)}
\]

Cette distance tient compte de la densité locale des deux observations.

Elle réduit notamment l'influence de connexions artificielles créées par des points situés dans des régions peu denses.

### 11.3 Minimum Spanning Tree

HDBSCAN construit une structure de graphe basée sur les distances de mutual reachability.

Un arbre couvrant minimal (Minimum Spanning Tree, MST) relie toutes les observations en minimisant la somme des poids des arêtes.

À partir de cette structure, l'algorithme peut étudier les regroupements à différentes échelles de densité.

### 11.4 Condensed Cluster Tree

HDBSCAN construit ensuite une hiérarchie condensée des clusters.

Les petits groupes peuvent être considérés comme du bruit selon le paramètre min_cluster_size.

L'algorithme peut sélectionner les clusters en fonction de leur persistance à travers les niveaux de densité.

Une méthode courante de sélection est Excess of Mass (EOM), qui privilégie des groupes stables dans la hiérarchie.

## 12. Les hyperparamètres de HDBSCAN

Paramètre	Rôle
min_cluster_size	Taille minimale souhaitée pour un cluster
min_samples	Influence le niveau de conservatisme et la détection du bruit
metric	Métrique utilisée pour les distances
cluster_selection_method	Méthode de sélection des clusters

Dans les implémentations usuelles, min_cluster_size et min_samples ont une influence majeure.

Une valeur élevée de min_samples peut augmenter le nombre d'observations considérées comme du bruit.

Une valeur faible de min_cluster_size permet d'identifier de plus petits groupes, avec un risque accru de fragmentation.

## 13. Implémentation de HDBSCAN en Python

Nous utiliserons l'implémentation disponible dans les versions récentes de scikit-learn.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

X, _ = make_blobs(n_samples=[250, 150, 100], centers=[[-4, 0], [0, 0], [4, 0]],
                  cluster_std=[0.3, 0.8, 1.4], random_state=42)
X_scaled = StandardScaler().fit_transform(X)
model = HDBSCAN(min_cluster_size=20, min_samples=10)
labels = model.fit_predict(X_scaled)
print("Clusters :", len(set(labels) - {-1}))
print(f"Noise ratio : {np.mean(labels == -1):.2%}")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=15)
plt.title("HDBSCAN, densités variables")
plt.show()
~~~

Remarque : l'API sklearn.cluster.HDBSCAN est disponible à partir de scikit-learn 1.3. Il existe également un package tiers hdbscan, dont certaines conventions de paramètres diffèrent. Dans sklearn, min_samples compte le point lui-même ; ce n’est pas la même convention que dans le package tiers.

### 13.1 Les probabilités d'appartenance

HDBSCAN peut fournir des scores décrivant la force d'appartenance d'une observation à son cluster.

Dans scikit-learn :

~~~python
probabilities = model.probabilities_
print(probabilities[:10])
~~~

Ces scores ne doivent pas être interprétés comme des probabilités de classe supervisée classiques. Ils reflètent la force de l'appartenance au cluster selon le modèle.

## 14. Comparaison : K-Means vs DBSCAN vs HDBSCAN

Critère	K-Means	DBSCAN	HDBSCAN
Nombre de clusters fixé	Oui	Non	Non
Formes complexes	Limité	Oui	Oui
Densités variables	Limité	Difficile	Plus adapté
Détection du bruit	Non	Oui	Oui
Centroïdes	Oui	Non	Non
Hyperparamètres principaux	n_clusters	eps, min_samples	min_cluster_size, min_samples
Interprétabilité	Centroïdes	Densité locale	Hiérarchie de densité

Quand utiliser chaque algorithme ?
K-Means convient particulièrement aux données numériques présentant des groupes compacts, lorsqu'une segmentation rapide est recherchée.

DBSCAN est intéressant lorsque les clusters présentent des formes irrégulières, une densité relativement homogène et du bruit.

HDBSCAN devient pertinent lorsque les densités varient ou lorsqu'on souhaite sélectionner les clusters à partir d'une hiérarchie de densité.

Il faut toutefois tester ces hypothèses sur les données réelles.

## 15. Les limites de DBSCAN et HDBSCAN

### 15.1 Sensibilité à la dimensionnalité

Dans les espaces de grande dimension, les distances peuvent devenir moins discriminantes.

C'est une manifestation de la curse of dimensionality.

Pour des données de grande dimension, une réduction de dimension adaptée peut être utile avant le clustering.

Attention toutefois : réduire les dimensions peut aussi modifier la structure des distances.

### 15.2 Sensibilité aux paramètres

DBSCAN est particulièrement sensible au choix de eps.

HDBSCAN réduit certaines difficultés liées au choix d'un rayon global, mais nécessite toujours un réglage et une interprétation rigoureuse de ses paramètres.

### 15.3 Coût algorithmique

Le coût de DBSCAN dépend notamment de la structure de recherche des voisins, de la métrique et de la dimensionnalité.

Une recherche de voisinage efficace peut être proche de \(O(n\log n)\) dans des conditions favorables, tandis que certaines situations peuvent nécessiter un temps ou une mémoire quadratiques.

HDBSCAN ajoute des étapes de construction hiérarchique qui peuvent également être coûteuses.

### 15.4 Détection d'anomalies

Un point étiqueté comme bruit par DBSCAN ou HDBSCAN n'est pas nécessairement une anomalie métier.

Par exemple, dans un dataset bancaire, un client ayant un comportement inhabituel n'est pas automatiquement frauduleux.

Il est essentiel de distinguer observation statistiquement atypique et événement réellement anormal.

## 16. Mini-projet LearnLoop — Density-Based Customer Segmentation

Projet : Customer Behavior Clustering
Intermédiaire

Contexte : une plateforme e-commerce souhaite découvrir des profils de clients et identifier les comportements atypiques.

Les données contiennent :
- Fréquence d'achat.
- Montant moyen des commandes.
- Nombre de visites mensuelles.
- Ancienneté du client.
- Taux de retour des produits.
Travail demandé :
1. Réaliser une analyse exploratoire des données.
2. Standardiser les caractéristiques.
3. Entraîner DBSCAN avec plusieurs combinaisons de paramètres.
4. Utiliser la méthode K-Distance pour guider le choix de eps.
5. Entraîner HDBSCAN.
6. Comparer le nombre de clusters, le taux de bruit et la stabilité.
7. Visualiser les clusters avec une réduction de dimension.
8. Interpréter les profils identifiés.
9. Comparer les résultats avec K-Means.
10. Produire une synthèse argumentée du modèle le plus adapté.
[[solution]]

Livrable : un notebook Python reproductible, accompagné d'une analyse comparative et de recommandations métier.

## 17. Questions d’entretien technique

[[interview]]

## 18. Quiz de validation

[[quiz]]

## 19. Synthèse du cours

À la fin de ce cours, tu dois retenir les distinctions suivantes :
Concept	À retenir
DBSCAN	Clustering fondé sur la densité
eps	Rayon du voisinage
min_samples	Seuil minimal de densité locale
Core Point	Observation suffisamment entourée
Border Point	Observation non centrale reliée à un point central
Noise Point	Observation non rattachée à un cluster
HDBSCAN	Clustering hiérarchique fondé sur la densité
Mutual Reachability	Distance tenant compte des densités locales
K-Distance	Outil d'aide au choix de eps

La distinction essentielle : K-Means cherche des centres, DBSCAN recherche des régions denses, et HDBSCAN analyse la structure de densité à plusieurs niveaux.

## 20. Ma progression

[[assessment]]

Prochaine étape : Cours 5 — Gaussian Mixture Models (GMM).
