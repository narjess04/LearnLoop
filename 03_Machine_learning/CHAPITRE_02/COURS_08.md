Dans le cours précédent, nous avons étudié le Feature Selection, qui permet de conserver les variables les plus pertinentes.

Aujourd'hui, nous allons découvrir une autre approche : plutôt que de supprimer certaines variables, nous allons apprendre à combiner les variables existantes pour construire une représentation plus compacte des données.

C'est le principe de la réduction de dimension.


## 1. Qu'est-ce que la réduction de dimension ?

Supposons que notre dataset possède 100 variables.

Certaines sont fortement corrélées. D'autres décrivent des aspects similaires d'un même phénomène.

Nous pouvons chercher à représenter les observations dans un espace contenant seulement 10, 5 ou même 2 dimensions.

Mathématiquement, nous recherchons une transformation :
\[
f:\mathbb{R}^{p}\rightarrow\mathbb{R}^{k}
\]

avec :
\[
k<p
\]

Où :

- \(p\) est le nombre initial de dimensions.
- \(k\) est le nombre de dimensions après transformation.

Exemple
Imaginons trois variables :

- Revenu mensuel.
- Revenu annuel.
- Capacité d'épargne.

Elles peuvent partager une information commune liée aux ressources financières.

Une méthode comme PCA peut combiner ces variables pour produire une composante synthétique.

Cette composante ne correspond pas nécessairement à une variable métier directement observable.



## 2. Feature Selection versus Dimensionality Reduction

C'est une distinction fondamentale en entretien.
Feature Selection

On conserve certaines colonnes originales.


Réduction de dimension

On construit de nouvelles dimensions.



Critère	Feature Selection	Réduction de dimension
Principe	Sélectionner des variables	Transformer les variables
Variables finales	Colonnes originales	Nouvelles composantes
Interprétabilité	Souvent meilleure	Souvent plus difficile
Exemples	RFE, SelectKBest, L1	PCA, SVD, UMAP
Objectif	Réduire les variables utilisées	Construire une représentation compacte




## 3. Pourquoi réduire la dimension ?


### 3.1 Réduire la redondance

Des variables fortement corrélées peuvent partager une grande partie de leur information.

### 3.2 Accélérer certains modèles

Travailler avec 20 dimensions au lieu de 1 000 peut réduire les coûts de calcul.

Cela dépend néanmoins de la méthode de réduction et du modèle.

### 3.3 Réduire certains effets du bruit

Certaines dimensions peuvent contenir beaucoup de variations peu informatives.

Une représentation compacte peut aider à atténuer ces variations.

### 3.4 Visualiser les données

Un dataset à 100 dimensions ne peut pas être visualisé directement.

En le projetant en 2D ou 3D, nous pouvons explorer les regroupements, les structures et certaines anomalies.

### 3.5 Limiter certains effets de la malédiction de la dimension

Lorsque le nombre de dimensions augmente, les données peuvent devenir plus dispersées et les distances moins discriminantes.

C'est particulièrement important pour certains algorithmes basés sur les voisins.

Cependant, réduire la dimension ne garantit pas une meilleure performance.

Partie I — PCA : Principal Component Analysis



## 4. Intuition de la PCA

La PCA, ou Analyse en Composantes Principales, est une méthode de réduction de dimension linéaire et non supervisée.

Elle recherche de nouvelles directions dans lesquelles les données présentent une variance maximale.

Ces directions sont appelées composantes principales.
Exemple intuitif
Supposons que nous disposions de deux variables fortement corrélées :

- Taille.
- Poids.

Les observations peuvent former un nuage allongé.

La PCA recherche la direction qui suit au mieux cet allongement.
[[projection]]

La première composante principale, PC1, capture la plus grande variance possible.

La deuxième composante, PC2, capture la plus grande variance restante tout en étant orthogonale à PC1.

Si nous conservons uniquement PC1, nous passons de deux dimensions à une seule.

Nous perdons une partie de l'information, mais nous conservons la direction de variation dominante.



## 5. Mathématiques de la PCA

Considérons une matrice de données :
\[
X\in\mathbb{R}^{n\times p}
\]

Avec :

- \(n\) observations.
- \(p\) variables.

Étape 1 — Centrer les données
Pour chaque variable :
\[
X_c=X-\mu
\]

Où \(\mu\) est le vecteur des moyennes.

Le centrage est essentiel pour la PCA classique.
Étape 2 — Calculer la matrice de covariance
\[
\Sigma=\frac{1}{n-1}X_c^TX_c
\]

La matrice de covariance décrit les variations des variables et leurs relations linéaires.
Étape 3 — Calculer les valeurs propres et vecteurs propres
Nous cherchons :
\[
\Sigma v_i=\lambda_i v_i
\]

Avec :

- \(v_i\) : vecteur propre.
- \(\lambda_i\) : valeur propre.

Les vecteurs propres définissent les directions principales.

Les valeurs propres indiquent la variance capturée dans chaque direction.
Étape 4 — Trier les composantes
Nous ordonnons :
\[
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p
\]

La composante associée à \(\lambda_1\) est PC1.
Étape 5 — Projeter les observations
Si nous conservons \(k\) composantes :
\[
W_k=[v_1,v_2,\ldots,v_k]
\]

La nouvelle représentation est :
\[
Z=X_cW_k
\]

Avec :
\[
Z\in\mathbb{R}^{n\times k}
\]

Nous avons donc réduit le nombre de dimensions de \(p\) à \(k\).



## 6. Explained Variance Ratio

La variance expliquée permet de mesurer la proportion de variance capturée par chaque composante.

Pour la composante \(i\) :
\[
EVR_i=\frac{\lambda_i}{\sum_{j=1}^{p}\lambda_j}
\]

Exemple
Supposons :

Composante	Variance expliquée
PC1	50 %
PC2	25 %
PC3	15 %
PC4	7 %
PC5	3 %

Les deux premières composantes expliquent :
\[
50\%+25\%=75\%
\]

Les trois premières expliquent :
\[
50\%+25\%+15\%=90\%
\]

[[variance]]

Exemple fictif : trois composantes conservent 90 % de la variance totale.

Attention : conserver 90 % de la variance ne signifie pas conserver 90 % de l'information utile pour prédire la cible.

La PCA ne connaît pas la variable cible.

Une direction de faible variance peut parfois être très prédictive.



## 7. PCA avec scikit-learn

Exemple simple
~~~python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("Shape originale :", X_train.shape)
print("Shape réduite :", X_train_pca.shape)
print("Variance expliquée :", pca.explained_variance_ratio_)
~~~




Pourquoi standardiser avant la PCA ?
La PCA maximise la variance.

Si une variable possède une variance beaucoup plus élevée uniquement parce qu'elle est mesurée dans une unité plus grande, elle peut dominer les composantes.

Exemple :

- Âge : entre 18 et 70.
- Revenu : entre 1 000 et 100 000.

Sans standardisation, le revenu risque de dominer la PCA.

StandardScaler est donc souvent utilisé avant PCA, lorsque les variables sont exprimées dans des unités différentes et que nous souhaitons leur donner un poids comparable.

Ce n'est toutefois pas une obligation universelle : dans certains contextes, conserver les variances originales est pertinent.



## 8. Comment choisir le nombre de composantes ?

Nous avons plusieurs stratégies.
Méthode 1 — Nombre fixe
~~~python
PCA(n_components=5)
~~~




Nous conservons exactement cinq composantes, si les dimensions des données le permettent.
Méthode 2 — Seuil de variance
~~~python
PCA(n_components=.95, svd_solver="full")
~~~




Avec le solveur approprié, scikit-learn choisit le nombre minimal de composantes permettant d'expliquer au moins 95 % de la variance.
Méthode 3 — Scree Plot
On trace la variance expliquée en fonction du nombre de composantes.

Nous cherchons éventuellement un « coude » à partir duquel l'ajout de nouvelles composantes apporte peu de variance supplémentaire.
Méthode 4 — Performance du modèle
Dans un problème supervisé, nous pouvons choisir le nombre de composantes en fonction d'une métrique évaluée par cross-validation.

C'est souvent la méthode la plus pertinente lorsque notre objectif principal est la prédiction.



## 9. Explorateur interactif — Nombre de composantes PCA


[[variance]]




## 10. Qu'est-ce que la SVD ?

La SVD est une factorisation matricielle fondamentale en algèbre linéaire.

Pour une matrice :
\[
X\in\mathbb{R}^{n\times p}
\]

Nous pouvons écrire :
\[
X=U\Sigma V^T
\]

Avec :

- \(U\) : vecteurs singuliers gauches.
- \(\Sigma\) : matrice des valeurs singulières.
- \(V^T\) : transposée des vecteurs singuliers droits.

Intuition
La SVD décompose une matrice en directions fondamentales.

En conservant uniquement les plus grandes valeurs singulières et leurs vecteurs associés, nous obtenons une approximation de rang réduit :
\[
X_k=U_k\Sigma_kV_k^T
\]

Cette approximation minimise l'erreur de reconstruction de Frobenius parmi les matrices de rang au plus \(k\).



## 11. PCA versus SVD

La PCA peut être calculée à partir de la SVD de la matrice centrée.

En effet, si :
\[
X_c=U\Sigma V^T
\]

Alors :
\[
X_c^TX_c=V\Sigma^2V^T
\]

Les directions principales correspondent donc aux vecteurs singuliers droits.
Différence pratique importante
PCA de scikit-learn centre les données.

TruncatedSVD ne centre pas automatiquement les données.

Cela rend TruncatedSVD particulièrement utile pour les matrices creuses (sparse matrices), comme celles utilisées en traitement automatique du langage.



## 12. TruncatedSVD et NLP

Imaginons une matrice TF-IDF :

Document	Banque	Crédit	Machine	Learning
D1	0.8	0.6	0	0
D2	0.7	0.7	0	0
D3	0	0	0.8	0.6
D4	0	0	0.6	0.8

Dans un vrai corpus, nous pouvons avoir des dizaines de milliers de colonnes.

La matrice est souvent très creuse : la plupart des valeurs sont nulles.

TruncatedSVD permet de construire une représentation dense de dimension plus faible.

Dans le domaine NLP, cette approche est associée à la Latent Semantic Analysis (LSA).
Exemple Python
~~~python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
documents = ["machine learning classification", "deep learning neural networks",
             "credit risk probability default", "bank credit risk modeling",
             "machine learning risk prediction"]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
print("TF-IDF shape :", X_tfidf.shape)
print("Reduced shape :", X_reduced.shape)
print(X_reduced)
~~~




Pour un projet prédictif, le vectorizer et la SVD doivent être ajustés uniquement sur le train.

Partie III — t-SNE : t-distributed Stochastic Neighbor Embedding



## 13. Pourquoi utiliser t-SNE ?

PCA est une méthode linéaire.

Elle cherche à conserver les directions de variance maximale.

Mais certaines données présentent des structures non linéaires complexes.

t-SNE est une méthode non linéaire principalement utilisée pour la visualisation en deux ou trois dimensions.

Son objectif est de préserver autant que possible certaines relations de voisinage locales.
Intuition
Imaginons un dataset contenant des images de chiffres manuscrits.

Chaque image est représentée par des centaines de pixels.

Deux images du chiffre 3 peuvent avoir des représentations numériques proches, tandis que les images des chiffres 3 et 8 peuvent présenter des ressemblances.

t-SNE cherche à construire une représentation 2D dans laquelle les observations voisines dans l'espace initial restent souvent voisines.
Visualisation conceptuelle après t-SNE
Exemple fictif de projection en deux dimensions. Les positions et séparations ne proviennent pas d'un entraînement réel.



## 14. Comment fonctionne t-SNE ?

t-SNE cherche à représenter les proximités entre observations sous forme de probabilités.

Dans l'espace original, il construit des probabilités de voisinage.

Dans l'espace réduit, il construit d'autres probabilités à partir d'une distribution de Student.

Il minimise ensuite une divergence de Kullback-Leibler :
\[
KL(P\|Q)=\sum_{i\neq j}p_{ij}\log\frac{p_{ij}}{q_{ij}}
\]

Où :

- \(P\) décrit les voisinages dans l'espace original.
- \(Q\) décrit les voisinages dans l'espace réduit.

L'optimisation cherche à rendre ces distributions aussi cohérentes que possible.
Le paramètre perplexity
La perplexité influence l'échelle des voisinages considérés.

Une perplexité faible privilégie des relations plus locales.

Une perplexité élevée prend en compte des voisinages plus étendus.

Elle doit rester inférieure au nombre d'observations.
Implémentation
~~~python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
# Exploration descriptive uniquement ; perplexity < nombre d’observations.
~~~




Cet exemple convient à une exploration descriptive. Pour une évaluation prédictive, il faut respecter la séparation des données et éviter de traiter une projection exploratoire comme un modèle de production.



## 15. Les limites importantes de t-SNE

Il faut absolument les connaître.

1. Les distances globales peuvent être trompeuses. Deux groupes éloignés dans la visualisation ne sont pas nécessairement aussi éloignés dans l'espace original.

2. La taille des groupes peut être trompeuse. Un groupe visuellement plus grand n'est pas forcément plus variable dans l'espace initial.

3. Les résultats dépendent des paramètres. Une autre perplexité ou initialisation peut modifier la représentation.

4. La projection n'est pas directement interprétable. Les axes n'ont généralement pas de signification métier simple.

5. Dans scikit-learn, t-SNE ne fournit pas une méthode transform() standard pour projeter directement de nouvelles observations.

t-SNE est donc principalement utilisé pour explorer et visualiser des données, plutôt que comme étape standard d'un pipeline prédictif.

Partie IV — UMAP : Uniform Manifold Approximation and Projection



## 16. Qu'est-ce que UMAP ?

UMAP est une méthode non linéaire de réduction de dimension fondée sur une représentation des voisinages sous forme de graphe.

Elle cherche à préserver certaines structures locales tout en conservant, selon les données et paramètres, une partie de l'organisation globale.

Elle est souvent utilisée pour :

- Visualiser des embeddings.
- Explorer des clusters.
- Réduire des représentations de documents.
- Analyser des données de grande dimension.

UMAP peut également produire des représentations de plus de trois dimensions.
Deux paramètres essentiels
n_neighbors contrôle l'échelle des voisinages utilisés pour construire la représentation.

min_dist influence la proximité minimale recherchée entre les points dans l'espace réduit.



## 17. Explorateur interactif — Paramètres UMAP

[[umap]]

Implémentation Python
UMAP nécessite généralement l'installation de la bibliothèque umap-learn.
pip install umap-learn


~~~python
import umap
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print(X_umap.shape)
# Exploration descriptive : ce code ne constitue pas un protocole prédictif indépendant.
~~~




La bibliothèque umap-learn permet aussi de projeter de nouvelles observations avec transform() dans des configurations compatibles.

Cela ne garantit pas que la représentation reste optimale lorsque la distribution des nouvelles données évolue.




## 18. PCA versus SVD versus t-SNE versus UMAP

Critère	PCA	TruncatedSVD	t-SNE	UMAP
Type	Linéaire	Linéaire	Non linéaire	Non linéaire
Objectif principal	Préserver la variance	Approximation de rang réduit	Préserver des voisinages locaux	Préserver des structures de voisinage
Visualisation 2D	Oui	Oui	Très utilisé	Très utilisé
Matrices sparse	Limitations pratiques	Très adapté	Souvent prétraitement nécessaire	Selon représentation
Projection de nouvelles données	Oui	Oui	Pas de transform() standard dans sklearn	Oui avec umap-learn
Interprétabilité	Relativement meilleure	Modérée	Faible	Faible
Usage prédictif	Fréquent	Fréquent	Moins courant	Possible, à valider

Quelle méthode choisir ?

- PCA : lorsque nous voulons une représentation linéaire compacte, souvent avant un modèle supervisé.
- TruncatedSVD : lorsque nous travaillons avec de grandes matrices creuses, notamment TF-IDF.
- t-SNE : lorsque l'objectif principal est la visualisation de voisinages locaux.
- UMAP : lorsque nous voulons explorer des structures non linéaires ou construire des embeddings de dimension réduite.

Aucune méthode n'est systématiquement meilleure.




## 19. Réduction de dimension et Data Leakage

La réduction de dimension peut aussi créer une fuite de données.
Mauvaise approche
~~~python
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)
# Mauvaise approche : la PCA a vu la structure du futur test.
~~~




La PCA utilise ici la structure de l'ensemble complet, y compris le futur test.
Bonne approche
~~~python
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
~~~




Encore mieux, utiliser un pipeline :
~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=5)),
                     ("model", LogisticRegression(max_iter=1000))])
~~~




Lorsque nous utilisons ce pipeline en cross-validation, les paramètres de standardisation et les composantes PCA sont appris uniquement sur les folds d'entraînement.

Partie V — Coding Lab



## 20. Comparer un modèle avec et sans PCA

Nous allons utiliser le dataset Breast Cancer de scikit-learn.

Il contient 30 variables numériques.

L'objectif est de comparer :
1. Régression logistique sans PCA.
2. Régression logistique avec PCA à 5 composantes.
3. Régression logistique avec PCA conservant 95 % de la variance.
~~~python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, reduction in [("Sans PCA", "passthrough"), ("PCA 5", PCA(n_components=5)),
                             ("PCA 95%", PCA(n_components=.95, svd_solver="full"))]:
        pipeline = Pipeline([("scale", StandardScaler()), ("pca", reduction),
                             ("model", LogisticRegression(max_iter=2000))])
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", return_estimator=True)
        assert np.isfinite(result["test_score"]).all()
        print(name, "ROC-AUC:", result["test_score"].mean(), "std:", result["test_score"].std())
        if name != "Sans PCA":
            print("Composantes par fold:", [p.named_steps["pca"].n_components_ for p in result["estimator"]])
    print("La variance conservée n'est pas une proportion d'information prédictive. Test final réservé.")
~~~




Questions d'analyse
1. Quel modèle obtient la meilleure ROC-AUC moyenne ?
2. Combien de composantes sont nécessaires pour conserver 95 % de la variance ?
3. La réduction de dimension améliore-t-elle nécessairement la performance ?
4. Quel est le compromis entre interprétabilité et compression ?
5. Pourquoi ne faut-il pas choisir les composantes sur l'ensemble complet avant la cross-validation ?



## 21. Coding Challenge — PCA from Scratch

Challenge 08 — Build PCA from Scratch

+350 XP

Implémente les étapes fondamentales de la PCA avec NumPy, puis compare les résultats avec scikit-learn.
~~~python
import numpy as np


def my_pca(X, n_components=2):
    X = np.asarray(X, dtype=float)
    # TODO 1 : centrer et conserver la moyenne train.
    # TODO 2 : covariance Xc.T @ Xc / (n - 1).
    # TODO 3 : valeurs/vecteurs propres avec np.linalg.eigh.
    # TODO 4 : trier en ordre décroissant, conserver k vecteurs.
    # TODO 5 : projeter, retourner aussi les paramètres pour transformer un test.
    raise NotImplementedError

# BONUS : comparer valeurs propres et sous-espace avec sklearn.PCA.
# Le signe des vecteurs propres n'est pas unique.
~~~




Point technique
Les signes des vecteurs propres ne sont pas uniques.

Deux implémentations peuvent donc produire des composantes opposées en signe tout en représentant exactement le même sous-espace.

Il faut tenir compte de cette propriété lors de la comparaison.



[[solution]]




## 22. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 23. Mini-quiz interactif


[[quiz]]




## 24. Validation du cours


[[assessment]]

À retenir : la réduction de dimension consiste à construire une représentation plus compacte des données. PCA et SVD sont particulièrement utiles pour les transformations linéaires, tandis que t-SNE et UMAP permettent d'explorer des structures non linéaires.

Le prochain cours sera consacré au Class Imbalance — Gestion des classes déséquilibrées, avec SMOTE, pondération des classes, rééchantillonnage et métriques adaptées aux événements rares.