# Cours 5 — Gaussian Mixture Models (GMM)

Repère de roadmap : 75 min de cours. Le support approfondi fourni annonce 90–120 min ; prévoir ce temps supplémentaire pour les démonstrations et les manipulations.

## 1. Introduction : pourquoi Gaussian Mixture Models ?

### 1.1 Les limites du clustering classique

Reprenons un exemple de segmentation des clients.

Nous disposons de deux caractéristiques :
- Le revenu annuel.
- Le montant moyen des dépenses.
K-Means pourrait identifier trois groupes : clients à faible, moyenne et forte dépense.

Cependant, lorsqu'un client se trouve à la frontière de deux groupes, K-Means lui attribue obligatoirement une seule étiquette.

GMM permet de représenter cette incertitude.
Hard clustering vs Soft clustering
Hard Clustering

Une seule étiquette

Soft Clustering

Appartenance probabiliste

Illustration conceptuelle : les proportions sont fictives et servent à expliquer la différence entre les deux approches.
Avec un GMM, nous pourrions obtenir :
\[
P(C_1\mid x)=0.60
\]

\[
P(C_2\mid x)=0.40
\]

Ces valeurs sont des probabilités d'appartenance conditionnelles au modèle estimé. Elles ne garantissent pas que les groupes correspondent à de véritables catégories naturelles.

## 2. Rappel mathématique : la distribution gaussienne

Pour comprendre les GMM, nous devons commencer par la distribution normale.

### 2.1 Distribution gaussienne univariée

Une variable aléatoire \(X\) suit une loi normale lorsque :
\[
X\sim\mathcal N(\mu,\sigma^2)
\]

Sa fonction de densité est :
\[
\boxed{f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
\]

où :
- \(\mu\) représente la moyenne.
- \(\sigma^2\) représente la variance.
- \(\sigma\) représente l'écart-type.
La moyenne contrôle la position de la courbe et la variance contrôle son étalement.

### 2.2 Explorer une distribution gaussienne

[[gaussiandemo]]

### 2.3 Distribution gaussienne multivariée

Dans les applications de Machine Learning, nous travaillons généralement avec plusieurs caractéristiques.

Soit :
\[
x\in\mathbb R^d
\]

La densité gaussienne multivariée est :
\[
\boxed{
\mathcal N(x\mid\mu,\Sigma)=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
}
\]

où :
- \(\mu\in\mathbb R^d\) est le vecteur moyen.
- \(\Sigma\in\mathbb R^{d\times d}\) est la matrice de covariance.
- \(|\Sigma|\) est son déterminant.
- \(\Sigma^{-1}\) est son inverse.
Cette formule suppose une matrice de covariance définie positive.

### 2.4 Pourquoi la matrice de covariance est-elle importante ?

Pour deux caractéristiques, la matrice de covariance est :
\[
\Sigma=
\begin{pmatrix}
\sigma_1^2 & \operatorname{Cov}(X_1,X_2)\\
\operatorname{Cov}(X_1,X_2)&\sigma_2^2
\end{pmatrix}
\]

Elle décrit :
- La dispersion de chaque caractéristique.
- Les relations linéaires entre les caractéristiques.
- L'orientation et la forme des contours de densité.
Une distribution gaussienne multivariée peut ainsi produire des contours elliptiques.
Sphérique

Diagonale

Complète

Exemples de formes de covariance en deux dimensions.

## 3. Qu'est-ce qu'un Gaussian Mixture Model ?

Un GMM suppose que la distribution observée peut être représentée comme une combinaison pondérée de plusieurs distributions gaussiennes.

Pour \(K\) composantes :
\[
\boxed{p(x)=\sum_{k=1}^{K}\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)}
\]

où :
- \(K\) représente le nombre de composantes gaussiennes.
- \(\pi_k\) est le poids de la composante \(k\).
- \(\mu_k\) est son vecteur moyen.
- \(\Sigma_k\) est sa matrice de covariance.
Les poids respectent :
\[
\pi_k\geq0
\]

\[
\sum_{k=1}^{K}\pi_k=1
\]

### 3.1 Exemple avec trois composantes

Imaginons :
\[
\pi_1=0.25,\quad \pi_2=0.45,\quad \pi_3=0.30
\]

La densité globale est :
\[
p(x)=0.25\mathcal N_1(x)+0.45\mathcal N_2(x)+0.30\mathcal N_3(x)
\]

[[mixturedemo]]

Les composantes peuvent se chevaucher. Une observation peut donc avoir une probabilité d'appartenance non nulle à plusieurs composantes.

## 4. La variable latente Z

Le GMM introduit une variable latente :
\[
Z\in\{1,\ldots,K\}
\]

Cette variable indique la composante qui a généré une observation.

Le modèle génératif suit trois étapes :
1. Sélectionner une composante \(Z=k\) avec une probabilité \(\pi_k\).
2. Générer une observation selon \(\mathcal N(\mu_k,\Sigma_k)\).
3. Répéter le processus pour obtenir les données.
Le problème est que nous observons \(X\), mais pas les valeurs de \(Z\).

C'est pourquoi nous devons estimer simultanément les paramètres du mélange et l'appartenance probable des observations aux composantes.

## 5. L'algorithme Expectation-Maximization (EM)

L'algorithme Expectation-Maximization permet d'estimer les paramètres d'un modèle probabiliste comportant des variables latentes.

Pour un GMM, nous cherchons :
\[
\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K}
\]

L'objectif est de maximiser la vraisemblance des observations.

### 5.1 Fonction de vraisemblance

Pour \(n\) observations indépendantes :
\[
L(\theta)=\prod_{i=1}^{n}p(x_i\mid\theta)
\]

En remplaçant la densité du GMM :
\[
L(\theta)=\prod_{i=1}^{n}\left(\sum_{k=1}^{K}\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\right)
\]

Nous travaillons généralement avec la log-vraisemblance :
\[
\boxed{\ell(\theta)=\sum_{i=1}^{n}\log\left(\sum_{k=1}^{K}\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\right)}
\]

Cette expression est difficile à optimiser directement à cause du logarithme d'une somme.

EM propose une optimisation itérative.

### 5.2 Étape E — Expectation

À partir des paramètres courants, nous calculons la responsabilité :
\[
\boxed{\gamma_{ik}=P(Z_i=k\mid x_i,\theta)}
\]

En appliquant le théorème de Bayes :
\[
\boxed{\gamma_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}}
\]

Pour chaque observation :
\[
\sum_{k=1}^{K}\gamma_{ik}=1
\]

La responsabilité mesure le degré d'appartenance probabiliste de l'observation \(i\) à la composante \(k\).

### 5.3 Exemple numérique de l'étape E

Supposons qu'une observation possède les valeurs suivantes :
\[
\pi_1\mathcal N_1(x)=0.12
\]

\[
\pi_2\mathcal N_2(x)=0.08
\]

Les responsabilités sont :
\[
\gamma_{i1}=\frac{0.12}{0.12+0.08}=0.60
\]

\[
\gamma_{i2}=\frac{0.08}{0.12+0.08}=0.40
\]

Probabilités d'appartenance de l'observation
Composante 1

60 %

Composante 2

40 %

Valeurs fictives illustrant le soft clustering.

### 5.4 Étape M — Maximization

À partir des responsabilités calculées, nous mettons à jour les paramètres du modèle.

On définit d'abord :
\[
N_k=\sum_{i=1}^{n}\gamma_{ik}
\]

Il représente l'effectif effectif de la composante \(k\).

Mise à jour des poids :
\[
\boxed{\pi_k^{new}=\frac{N_k}{n}}
\]

Mise à jour des moyennes :
\[
\boxed{\mu_k^{new}=\frac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}x_i}
\]

Mise à jour des covariances :
\[
\boxed{\Sigma_k^{new}=\frac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}(x_i-\mu_k^{new})(x_i-\mu_k^{new})^T}
\]

Ces formules correspondent au cas de matrices de covariance complètes, sans contraintes supplémentaires.

### 5.5 Pourquoi EM fonctionne-t-il ?

L'algorithme alterne :
- E-step : estimer les responsabilités à partir des paramètres actuels.
- M-step : optimiser les paramètres en utilisant ces responsabilités.
Initialisation des paramètres

Poids, moyennes, covariances

E-step

Calcul des responsabilités γ

M-step

Mise à jour de π, μ et Σ

Convergence ?

Sinon, recommencer à l'E-step

EM garantit que la log-vraisemblance ne diminue pas lors d'itérations exactes dans les conditions habituelles.

Cependant, il ne garantit pas de trouver le maximum global.

Une mauvaise initialisation peut conduire à une solution sous-optimale. De plus, sans régularisation ou contraintes appropriées, la vraisemblance d'un mélange gaussien peut devenir non bornée lorsque certaines covariances dégénèrent.

## 6. Implémentation d'un GMM from scratch

Pour comprendre le mécanisme, nous allons implémenter un mélange gaussien univarié.

Nous allons estimer :
- Les poids des composantes.
- Les moyennes.
- Les variances.
- Les responsabilités.

~~~python
import numpy as np

class GaussianMixture1D:
    """EM univarié pédagogique en log-espace ; plancher de variance explicite."""
    def __init__(self, n_components=2, max_iter=200, tol=1e-5, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim not in (1, 2) or (X.ndim == 2 and X.shape[1] != 1):
            raise ValueError("Un vecteur univarié est requis.")
        X = X.ravel()
        if not X.size or not np.isfinite(X).all():
            raise ValueError("Des valeurs finies non vides sont requises.")
        return X

    def _expectation(self, X):
        log_joint = np.log(self.weights_)[None, :] - 0.5 * (
            np.log(2 * np.pi * self.vars_)[None, :]
            + (X[:, None] - self.means_[None, :]) ** 2 / self.vars_[None, :]
        )
        maximum = log_joint.max(axis=1, keepdims=True)
        log_norm = maximum + np.log(np.exp(log_joint - maximum).sum(axis=1, keepdims=True))
        return np.exp(log_joint - log_norm), float(log_norm.sum())

    def fit(self, X):
        X = self._input(X)
        n, K = len(X), self.n_components
        if not 1 <= K <= n or self.max_iter < 1 or self.tol < 0:
            raise ValueError("Vérifier n_components, max_iter et tol.")
        rng = np.random.default_rng(self.random_state)
        self.means_ = rng.choice(X, K, replace=False)
        self.vars_ = np.full(K, max(float(np.var(X)), 1e-6))
        self.weights_ = np.ones(K) / K
        responsibilities, previous_ll = self._expectation(X)
        self.history_ = [previous_ll]
        self.converged_ = False
        for iteration in range(self.max_iter):
            counts = np.maximum(responsibilities.sum(axis=0), np.finfo(float).eps)
            self.weights_ = counts / counts.sum()
            self.means_ = (responsibilities * X[:, None]).sum(axis=0) / counts
            self.vars_ = np.maximum(
                (responsibilities * (X[:, None] - self.means_) ** 2).sum(axis=0) / counts, 1e-6
            )
            responsibilities, ll = self._expectation(X)
            self.history_.append(ll)
            self.n_iter_ = iteration + 1
            if abs(ll - previous_ll) <= self.tol:
                self.converged_ = True
                break
            previous_ll = ll
        return self

    def predict_proba(self, X):
        if not hasattr(self, "means_"):
            raise ValueError("Appeler fit avant predict_proba.")
        return self._expectation(self._input(X))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
~~~

Cette version est volontairement pédagogique. Pour des applications réelles, il est préférable d'utiliser des calculs en espace logarithmique, notamment avec logsumexp, pour améliorer la stabilité numérique.

### 6.1 Tester notre implémentation

~~~python
import numpy as np

rng = np.random.default_rng(42)
X1 = rng.normal(-2, 0.7, 300)
X2 = rng.normal(3, 1.2, 200)
X = np.concatenate([X1, X2])
model = GaussianMixture1D(n_components=2).fit(X)
print("Moyennes :", model.means_)
print("Variances :", model.vars_)
print("Poids :", model.weights_)
print("Itérations :", model.n_iter_)
~~~

Les moyennes estimées devraient se rapprocher de \(-2\) et \(3\), sans que l'ordre des composantes soit garanti.

## 7. Implémentation avec scikit-learn

Nous allons maintenant travailler avec des données bidimensionnelles.

~~~python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=600, centers=3, cluster_std=[0.7, 1.2, 0.9], random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=10,
                      reg_covar=1e-6, random_state=42).fit(X_scaled)
labels = gmm.predict(X_scaled)
proba = gmm.predict_proba(X_scaled)
print("Responsabilités :", proba[:5])
~~~

### 7.1 Les principaux attributs

Attribut	Signification
weights_	Poids estimés des composantes
means_	Moyennes des composantes
covariances_	Matrices de covariance
converged_	Indique si le critère de convergence est satisfait
n_iter_	Nombre d'itérations EM
lower_bound_	Borne inférieure moyenne obtenue par EM

### 7.2 predict() vs predict_proba()

predict() retourne l'indice de la composante ayant la responsabilité la plus élevée.

~~~python
labels = gmm.predict(X_scaled)
~~~

predict_proba() retourne les responsabilités pour toutes les composantes.

~~~python
probabilities = gmm.predict_proba(X_scaled)
~~~

Par exemple, une observation pourrait avoir :
\[
[0.05,\ 0.80,\ 0.15]
\]

Elle serait affectée à la deuxième composante (indice 1 en Python), avec une responsabilité de 80 %.

## 8. Les différents types de covariance

Scikit-learn propose quatre structures de covariance.

### 8.1 Full

Chaque composante possède sa propre matrice de covariance complète.

~~~python
covariance_type="full"
~~~

C'est le choix le plus flexible, mais aussi celui qui nécessite généralement le plus de paramètres.

### 8.2 Tied

Toutes les composantes partagent une même matrice de covariance complète.

~~~python
covariance_type="tied"
~~~

### 8.3 Diag

Chaque composante possède sa propre matrice de covariance diagonale.

~~~python
covariance_type="diag"
~~~

Les covariances entre caractéristiques sont supposées nulles à l'intérieur de chaque composante.

### 8.4 Spherical

Chaque composante possède une covariance de la forme :
\[
\Sigma_k=\sigma_k^2I
\]

~~~python
covariance_type="spherical"
~~~

Les contours de densité sont sphériques dans l'espace des caractéristiques.
Comparaison
Type	Flexibilité	Nombre de paramètres
Full	Très élevée	Élevé
Tied	Moyenne	Réduit
Diag	Moyenne	Modéré
Spherical	Faible	Faible

Le nombre exact de paramètres dépend de la dimensionnalité et du nombre de composantes.

## 9. Comment choisir le nombre de composantes K ?

Contrairement à DBSCAN, GMM nécessite de fixer le nombre de composantes avant l'entraînement.

Cependant, nous pouvons comparer plusieurs valeurs de \(K\) avec des critères de sélection statistique.

### 9.1 Akaike Information Criterion (AIC)

\[
\boxed{AIC=2p-2\ln(\hat L)}
\]

où :
- \(p\) est le nombre de paramètres libres.
- \(\hat L\) est la vraisemblance maximisée.
L'AIC équilibre qualité d'ajustement et complexité.

### 9.2 Bayesian Information Criterion (BIC)

\[
\boxed{BIC=p\ln(n)-2\ln(\hat L)}
\]

où \(n\) représente le nombre d'observations.

Le BIC pénalise davantage la complexité que l'AIC lorsque \(n\) est suffisamment grand.

Pour ces deux critères :

Plus la valeur est faible, meilleur est le compromis selon le critère considéré.

### 9.3 Implémentation Python

~~~python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

aic_scores, bic_scores = [], []
K_values = range(1, 11)
for k in K_values:
    model = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                            reg_covar=1e-6, random_state=42).fit(X_scaled)
    aic_scores.append(model.aic(X_scaled))
    bic_scores.append(model.bic(X_scaled))
plt.plot(list(K_values), aic_scores, marker="o", label="AIC")
plt.plot(list(K_values), bic_scores, marker="s", label="BIC")
plt.xlabel("Nombre de composantes")
plt.ylabel("Critère")
plt.legend()
plt.show()
~~~

Les courbes du laboratoire et du challenge sont calculées à l’exécution. AIC et BIC peuvent privilégier des configurations différentes.

Le nombre de composantes sélectionné n'est pas nécessairement égal au nombre de groupes métier pertinents.

## 10. K-Means vs GMM : quelle différence mathématique ?

K-Means minimise :
\[
J=\sum_{i=1}^{n}\|x_i-\mu_{c_i}\|^2
\]

GMM maximise :
\[
\ell(\theta)=\sum_{i=1}^{n}\log\left(\sum_{k=1}^{K}\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)\right)
\]

Critère	K-Means	GMM
Approche	Géométrique	Probabiliste
Affectation	Hard	Soft
Centroïdes	Oui	Moyennes gaussiennes
Covariances	Non modélisées explicitement	Oui
Formes elliptiques	Limité	Oui
Fonction objectif	Inertie	Log-vraisemblance
Nombre de groupes	Fixé	Nombre de composantes fixé

Relation entre K-Means et GMM
K-Means peut être vu comme une limite particulière d'un mélange gaussien lorsque :
- Les covariances sont sphériques, identiques et deviennent très petites.
- Les poids des composantes sont comparables.
- Les responsabilités deviennent presque binaires.
Cette relation explique pourquoi les deux algorithmes peuvent produire des résultats proches sur des clusters sphériques bien séparés.

## 11. Limites et pièges fréquents

1. Sensibilité à l'initialisation. EM peut converger vers différents maxima locaux. Plusieurs initialisations sont recommandées.

2. Covariances dégénérées. Une composante peut se concentrer excessivement autour de quelques observations. Le paramètre reg_covar permet d'ajouter une régularisation numérique.

3. Mauvais choix de K. Trop de composantes peuvent conduire à un modèle inutilement complexe ; trop peu peuvent masquer la structure des données.

4. Hypothèse gaussienne. Si les données présentent des structures fortement non gaussiennes, le modèle peut nécessiter de nombreuses composantes pour les représenter.

5. Sensibilité aux observations extrêmes. Les estimations de moyennes et de covariances peuvent être influencées par des valeurs atypiques.

6. Interprétation des probabilités. Les responsabilités ne constituent pas une garantie que les catégories découvertes correspondent à des groupes réels.

7. Haute dimensionnalité. Les matrices de covariance complètes nécessitent un nombre de paramètres qui augmente rapidement avec le nombre de caractéristiques.

## 12. Mini-projet LearnLoop — Probabilistic Customer Segmentation

Projet : Soft Customer Segmentation
Intermédiaire → Avancé

Contexte : une entreprise e-commerce souhaite segmenter ses clients tout en quantifiant l'incertitude de leur appartenance aux différents profils.

Dataset : données synthétiques ou dataset de segmentation client contenant des variables numériques.

Travail demandé :
1. Explorer les données et identifier les caractéristiques pertinentes.
2. Standardiser les variables numériques.
3. Entraîner plusieurs GMM avec \(K\) variant de 1 à 10.
4. Comparer les structures de covariance full, tied, diag et spherical.
5. Utiliser AIC et BIC pour sélectionner une configuration.
6. Visualiser les composantes et leurs centres.
7. Étudier les responsabilités des observations.
8. Identifier les observations dont l'affectation est incertaine.
9. Comparer les résultats avec K-Means.
10. Produire une analyse métier des profils identifiés.
[[solution]]

Livrable : un notebook Python avec graphiques, métriques, analyse probabiliste et conclusion argumentée.

## 13. Questions d’entretien technique

[[interview]]

## 14. Quiz de validation

[[quiz]]

## 15. Synthèse du cours

À la fin de ce cours, tu dois maîtriser les concepts suivants :
Concept	À retenir
Gaussian Mixture Model	Mélange pondéré de distributions gaussiennes
Soft Clustering	Appartenance probabiliste à plusieurs composantes
Variable latente \(Z\)	Composante génératrice non observée
Responsabilité \(\gamma_{ik}\)	Probabilité conditionnelle d'appartenance
E-step	Estimation des responsabilités
M-step	Mise à jour des paramètres
Covariance	Dispersion et orientation des composantes
AIC / BIC	Critères de comparaison des modèles
EM	Optimisation itérative de la vraisemblance

La distinction essentielle : K-Means affecte les observations à des centroïdes, tandis que GMM modélise leur distribution probabiliste et quantifie l'incertitude d'appartenance aux composantes.

## 16. Ma progression

[[assessment]]

Prochaine étape : Cours 6 — Évaluation des algorithmes de clustering.
