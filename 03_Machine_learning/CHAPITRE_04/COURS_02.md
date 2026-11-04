# Cours 2 — K-Means Clustering

## 1. Introduction : pourquoi K-Means ?

Imaginons que nous disposions d'un dataset de 1 000 clients contenant deux variables :
- Le montant moyen des achats.
- Le nombre d'achats par mois.
Nous souhaitons identifier des profils de clients sans disposer de catégories préexistantes.

K-Means est un algorithme de clustering non supervisé qui partitionne les observations en \(K\) groupes, de manière à rapprocher les observations de leur centre respectif.

### 1.1 Intuition géométrique

Chaque observation est représentée par un point dans un espace à \(p\) dimensions.

L'algorithme recherche \(K\) points particuliers appelés centroïdes, qui représentent les centres des groupes.
Trois clusters fictifs. Les croix représentent les centroïdes.
L'objectif est que les observations d'un même groupe soient proches de leur centroïde.

## 2. Formulation mathématique

Soit un ensemble de données :
\[
X=\{x_1,x_2,\ldots,x_n\}
\]

avec \(x_i\in\mathbb R^p\).

Nous souhaitons trouver \(K\) clusters \(C_1,\ldots,C_K\).

La fonction objectif de K-Means est :
\[
\boxed{J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\|x_i-\mu_k\|_2^2}
\]

où :
- \(K\) : nombre de clusters.
- \(C_k\) : ensemble des observations du cluster \(k\).
- \(\mu_k\) : centroïde du cluster \(k\).
- \(\|x_i-\mu_k\|_2^2\) : distance euclidienne au carré.
Cette fonction est appelée Within-Cluster Sum of Squares (WCSS) ou inertie.

K-Means cherche à minimiser \(J\).

### 2.1 Calcul d'un centroïde

Considérons trois observations :
\[
x_1=(2,3),\quad x_2=(4,5),\quad x_3=(6,4)
\]

Le centroïde correspond à la moyenne des coordonnées :
\[
\mu=\frac{1}{3}(x_1+x_2+x_3)
\]

Donc :
\[
\mu=\left(\frac{2+4+6}{3},\frac{3+5+4}{3}\right)
\]

\[
\boxed{\mu=(4,4)}
\]

Cette propriété explique pourquoi K-Means est sensible aux valeurs extrêmes : la moyenne est influencée par les observations éloignées.
[[lloyd]]

## 3. Fonctionnement de l'algorithme

K-Means fonctionne selon un processus itératif.
Étape 1 — Initialisation

Choisir K centroïdes initiaux

Étape 2 — Affectation

Associer chaque observation au centroïde le plus proche

Étape 3 — Mise à jour

Recalculer les centroïdes à partir des observations affectées

Étape 4 — Convergence ?

Sinon, répéter les étapes 2 et 3

### 3.1 Étape d'affectation

Chaque observation \(x_i\) est affectée au centroïde le plus proche :
\[
c_i=\arg\min_{k\in\{1,\ldots,K\}}\|x_i-\mu_k\|_2^2
\]

### 3.2 Étape de mise à jour

Pour chaque cluster, nous recalculons son centroïde :
\[
\mu_k=\frac{1}{|C_k|}\sum_{x_i\in C_k}x_i
\]

Ces deux opérations sont répétées jusqu'à la convergence ou jusqu'à atteindre le nombre maximal d'itérations.

L'inertie ne peut pas augmenter après une étape complète d'affectation et de mise à jour dans l'algorithme classique de Lloyd, mais le résultat peut correspondre à un minimum local.

## 4. Implémentation K-Means from scratch

Avant d'utiliser scikit-learn, nous allons implémenter l'algorithme avec NumPy.

~~~python
import numpy as np

class KMeansScratch:
    """Lloyd pédagogique : initialisation simple, centre vide conservé."""
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Une matrice 2D finie est requise.")
        if not 1 <= self.n_clusters <= len(X) or self.max_iter < 1 or self.tol < 0:
            raise ValueError("Vérifier K, max_iter et tol.")
        rng = np.random.default_rng(self.random_state)
        centers = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        def squared_distances(c):
            return ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        self.history_ = [float(squared_distances(centers).min(axis=1).sum())]
        for iteration in range(self.max_iter):
            labels = squared_distances(centers).argmin(axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points):
                    new_centers[k] = points.mean(axis=0)
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            self.history_.append(float(squared_distances(centers).min(axis=1).sum()))
            self.n_iter_ = iteration + 1
            if shift <= self.tol:
                break
        self.cluster_centers_ = centers
        self.labels_ = squared_distances(centers).argmin(axis=1)
        self.inertia_ = float(((X - centers[self.labels_]) ** 2).sum())
        # Les labels sont recalculés après la dernière mise à jour.
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Appeler fit avant predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.cluster_centers_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Matrice finie et nombre de features compatibles requis.")
        return ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
~~~

Cette version est pédagogique : elle utilise une initialisation aléatoire simple et conserve les centres précédents si un cluster devient vide. Une implémentation industrielle doit gérer plus finement l'initialisation, les clusters vides et les performances.

## 5. Implémentation avec scikit-learn

~~~python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels = model.fit_predict(X_scaled)
print("Centroïdes standardisés :", model.cluster_centers_)
print("Centroïdes en unités initiales :", scaler.inverse_transform(model.cluster_centers_))
print("Inertie :", model.inertia_)
print("Silhouette :", silhouette_score(X_scaled, labels))
~~~

### 5.1 Pourquoi standardiser les données ?

K-Means utilise les distances euclidiennes.

Imaginons deux variables :
- Âge : entre 18 et 80.
- Revenu annuel : entre 10 000 et 150 000.
Sans standardisation, le revenu risque de dominer le calcul des distances en raison de son échelle numérique.

Le StandardScaler transforme une variable selon :
\[
z=\frac{x-\mu}{\sigma}
\]

La standardisation est généralement recommandée lorsque les variables ont des échelles différentes, mais elle doit rester adaptée à leur signification métier.

## 6. Comment choisir le nombre de clusters K ?

K-Means exige de fixer \(K\) avant l'entraînement.

Deux approches courantes permettent de guider ce choix.

### 6.1 Elbow Method

Nous entraînons K-Means avec plusieurs valeurs de \(K\), puis observons l'évolution de l'inertie.
[[elbow]]

Exemple illustratif — Elbow Method
Valeurs fictives d'inertie pour différents nombres de clusters.

Dans cet exemple fictif, le coude se situe approximativement autour de \(K=4\).

L’inertie optimale est non croissante lorsque K augmente. Des ajustements indépendants peuvent toutefois produire des minima locaux moins favorables ; examiner plusieurs initialisations. Le but n'est donc pas simplement de choisir le plus petit score.

### 6.2 Silhouette Score

Le coefficient de silhouette d'une observation est :
\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

où :
- \(a(i)\) : distance moyenne aux autres observations du même cluster.
- \(b(i)\) : plus petite distance moyenne à un autre cluster.
Le score varie entre \(-1\) et \(1\).
Valeur	Interprétation générale
Proche de 1	Observation bien regroupée
Proche de 0	Observation proche d'une frontière
Négative	Observation potentiellement mal affectée

Le score moyen peut aider à comparer plusieurs valeurs de \(K\), sans garantir que les groupes soient pertinents pour le problème métier.

## 7. K-Means++ : pourquoi l'initialisation est importante

Une initialisation aléatoire défavorable peut conduire à un mauvais minimum local.

K-Means++ améliore la sélection des centroïdes initiaux :

1. Choisir un premier centroïde aléatoirement.

2. Calculer la distance de chaque point à son centroïde déjà sélectionné le plus proche.

3. Choisir un nouveau centroïde avec une probabilité proportionnelle au carré de cette distance.

4. Répéter jusqu'à obtenir \(K\) centroïdes.

Cela favorise des centroïdes initiaux éloignés les uns des autres.

L'utilisation de plusieurs initialisations (n_init) permet également de réduire la dépendance à un seul démarrage.

## 8. Limites et pièges fréquents

Limite	Conséquence
Nombre \(K\) fixé	Nécessite une sélection préalable
Sensibilité aux outliers	Centroïdes déplacés
Hypothèse de groupes compacts	Difficultés avec des formes complexes
Sensibilité aux échelles	Variables dominantes
Initialisation	Résultats potentiellement différents
Variables catégorielles brutes	Distance euclidienne souvent inadaptée

K-Means convient particulièrement aux groupes relativement compacts et bien séparés.

Pour les formes non convexes, les méthodes de densité peuvent être utiles. DBSCAN peut lui-même échouer lorsque les densités varient fortement ; HDBSCAN sera étudié avec ses propres limites au cours 4.

## 9. Coding Challenge — LearnLoop

Challenge : trouver le meilleur K
Intermédiaire

Construis un programme qui :

1. Génère 600 observations avec make_blobs.

2. Standardise les caractéristiques.

3. Entraîne K-Means pour \(K=2\) à \(K=10\).

4. Calcule l'inertie et le Silhouette Score pour chaque valeur.

5. Affiche les deux courbes.

6. Sélectionne le meilleur \(K\) selon le Silhouette Score.

7. Visualise les clusters obtenus.

[[solution]]

Bonus : compare les résultats avec et sans standardisation sur un dataset dont les variables possèdent des échelles très différentes.

## 10. Questions d’entretien technique

[[interview]]

## 11. Quiz de validation

[[quiz]]

## 12. Ce qu'il faut retenir

À la fin de ce cours, tu dois être capable d'expliquer la fonction objectif de K-Means, de décrire ses étapes d'affectation et de mise à jour, de coder une version simple avec NumPy, de l'utiliser avec scikit-learn et de justifier le choix du nombre de clusters.

Tu dois également comprendre pourquoi la standardisation, l'initialisation et l'analyse des limites sont essentielles pour obtenir des résultats interprétables.

Prochain cours : Chapitre 4 — Cours 3 : Clustering hiérarchique, avec les dendrogrammes, les différentes méthodes de liaison (single, complete, average, Ward) et une implémentation complète.

## 13. Ma progression

[[assessment]]
