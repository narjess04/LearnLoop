# Cours 1 — Introduction à l’apprentissage non supervisé

## 1. Qu’est-ce que l’apprentissage non supervisé ?

L’apprentissage non supervisé (Unsupervised Learning) analyse des observations sans recevoir les réponses cibles attendues. Contrairement à une tâche supervisée, nous n’utilisons pas de variable cible \(y\) pour ajuster le modèle.

Nous disposons d’une matrice de caractéristiques :

\[X \in \mathbb{R}^{n \times p}\]

- \(n\) : nombre d’observations.
- \(p\) : nombre de caractéristiques, ou features.

L’objectif est de découvrir des groupes, des relations, une représentation plus compacte ou des observations atypiques. L’absence de labels ne signifie ni absence d’hypothèses, ni absence d’évaluation : le choix des variables, des distances et des paramètres influence la structure découverte.

## 2. Exemple intuitif : segmentation des clients

Une entreprise connaît la fréquence d’achat et le montant moyen dépensé par ses clients, mais ne possède pas de catégories préétablies. Un algorithme de clustering peut proposer des groupes de comportements similaires.

[[clustercloud]]

On pourra ensuite interpréter ces groupes : clients peu actifs, réguliers ou très actifs. Ces noms sont attribués après l’analyse. Les identifiants de clusters sont arbitraires : le groupe 0 n’est pas intrinsèquement « meilleur » que le groupe 1.

Une segmentation statistique ne prouve pas l’existence de catégories naturelles. Il faut vérifier sa stabilité et sa pertinence pour la décision envisagée.

## 3. Les grandes familles de méthodes

Famille	Objectif	Exemples
Clustering	Regrouper des observations similaires	K-Means, clustering hiérarchique, DBSCAN, GMM
Réduction de dimension	Représenter avec moins de variables	PCA, t-SNE, UMAP
Détection d’anomalies	Repérer des observations inhabituelles	Isolation Forest, LOF, One-Class SVM

**Clustering.** Les catégories ne sont pas fournies au modèle ; les groupes dépendent de la géométrie choisie.

**Réduction de dimension.** On conserve une partie de l’information ou de la structure selon un critère donné. PCA vise des directions de variance ; t-SNE et UMAP sont souvent utilisés pour explorer visuellement les voisinages. Une carte en deux dimensions ne garantit pas que les distances globales soient préservées.

**Détection d’anomalies.** Un point atypique mérite une investigation ; ce n’est pas automatiquement une erreur ou une fraude. La détection de nouveautés sur un jeu de référence supposé normal demande de préciser le protocole.

## 4. Différence entre supervisé et non supervisé

Critère	Supervisé	Non supervisé
Variable cible y	Utilisée pendant l’apprentissage	Absente de l’ajustement
Objectif	Prédire une cible	Découvrir une structure ou une représentation
Exemples	Régression, classification	Clustering, PCA
Évaluation	Accuracy, F1, RMSE selon la tâche	Silhouette pour certains clusterings ; variance expliquée pour PCA
Application	Prédire un défaut	Segmenter des clients

Il n’existe pas une métrique universelle du non supervisé. La silhouette évalue cohésion et séparation pour une distance donnée. La variance expliquée concerne PCA et ne mesure pas la qualité d’un clustering.

Des labels de référence peuvent parfois servir à une évaluation externe, s’ils existent ; ils ne doivent alors pas être confondus avec les données utilisées pour ajuster un algorithme non supervisé.

## 5. Première implémentation Python

Nous générons 300 observations artificielles et choisissons trois clusters. Le nombre de centres de génération est connu parce que l’exemple est synthétique ; dans un cas réel, il faut justifier K.

~~~python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.8,
    random_state=42,
)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis", alpha=0.7)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X", s=200, c="red", label="Centroïdes",
)
plt.title("K-Means Clustering")
plt.legend()
plt.show()
~~~

Le symbole _ indique que nous n’utilisons pas les étiquettes produites par make_blobs. K-Means reçoit uniquement X. Les noms et numéros des groupes ne portent pas de sens métier intrinsèque.

Ce dataset utilise deux variables synthétiques sur des échelles comparables. Sur des variables métier d’unités différentes, il faudra examiner la mise à l’échelle avant d’utiliser une distance euclidienne.

Nous étudierons l’objectif mathématique et les étapes de K-Means au cours 2. Le laboratoire téléchargeable ajoute les effectifs, l’inertie et la silhouette, calculés lorsque tu exécutes Python.

## 6. Exercice : modifier le nombre de groupes

Exécute le laboratoire, puis compare K=2, K=3 et K=4 sur les mêmes observations. Observe les partitions, les effectifs et la silhouette. Explique pourquoi le nombre de groupes est une décision de modélisation.

[[solution]]

## 7. Vérification des connaissances

[[quiz]]

## 8. Questions pour expliquer ton raisonnement

[[interview]]

## 9. Mini-projet du chapitre

À l’issue des dix cours, le projet **Customer Segmentation & Anomaly Detection System** combinera clustering, PCA, détection d’anomalies et visualisations interactives. Son cahier des charges est disponible dans les projets ; les notions seront introduites progressivement dans les prochains cours.

Le projet demandera de documenter les variables, les transformations, la stabilité des groupes et les limites de l’interprétation. Une visualisation séduisante ne remplace pas cette analyse.

## 10. Ce que je sais maintenant faire

[[assessment]]

**Prochaine étape : Cours 2 — K-Means Clustering.**
