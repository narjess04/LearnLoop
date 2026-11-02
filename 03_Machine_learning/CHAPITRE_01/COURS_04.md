Dans le cours précédent, nous avons appris à minimiser une fonction de perte grâce à la descente de gradient.

Mais minimiser l'erreur d'entraînement ne garantit pas qu'un modèle sera performant sur de nouvelles données.

C'est précisément le problème de la généralisation, l'un des concepts les plus importants du Machine Learning.

## 1. Qu'est-ce que la généralisation ?

Un modèle généralise lorsqu'il parvient à produire de bonnes prédictions sur des observations qu'il n'a jamais utilisées pour apprendre.

Prenons un exemple : nous entraînons un modèle pour prédire les prix immobiliers à partir de la surface, de la localisation et d'autres caractéristiques.

Un modèle qui prédit parfaitement les prix des maisons de son dataset d'entraînement, mais se trompe fortement sur de nouvelles maisons, n'a pas bien appris à généraliser.

### 1.1 Training Error versus Generalization Error

On distingue deux notions.

Erreur empirique (Training Error) :
\[
\hat R_{\text{train}}(f)=\frac{1}{n}\sum_{i=1}^{n}L(f(x_i),y_i)
\]

Elle mesure la perte moyenne sur les observations d'entraînement.

Risque de généralisation (Expected Risk) :
\[
R(f)=\mathbb{E}_{(X,Y)\sim P_{\text{data}}}[L(f(X),Y)]
\]

Il mesure la perte attendue sur la distribution réelle des données.

Cette seconde quantité n'est généralement pas directement calculable, puisque nous ne connaissons pas toute la distribution des données.

Nous l'estimons donc à partir d'observations indépendantes, notamment avec un ensemble de validation ou de test.

L'objectif du ML n'est pas seulement de minimiser \(\hat R_{\text{train}}\), mais d'obtenir un modèle dont le risque de généralisation est faible.

## 2. Underfitting et Overfitting

Il existe deux problèmes classiques lorsqu'on entraîne un modèle.
Underfitting

Modèle trop simple pour capturer la structure.

Overfitting

Modèle trop ajusté aux observations.

Illustrations conceptuelles : les points représentent les observations et les courbes les fonctions apprises.

### 2.1 Underfitting — Sous-apprentissage

Un modèle est en underfitting lorsqu'il n'arrive pas à capturer suffisamment les relations présentes dans les données.

Les causes possibles sont une complexité insuffisante, des variables peu informatives, une régularisation excessive ou un entraînement insuffisant.

Symptôme fréquent : erreurs élevées sur train et validation.

### 2.2 Overfitting — Surapprentissage

Un modèle est en overfitting lorsqu'il s'ajuste trop aux particularités de l'échantillon d'entraînement, y compris au bruit.

Il peut obtenir une excellente performance sur le train, mais une performance nettement moins bonne sur des données indépendantes.

Les causes possibles sont une complexité excessive par rapport à la quantité de données, des features peu pertinentes ou une régularisation insuffisante.

### 2.3 Tableau de diagnostic

Train error	Validation error	Diagnostic possible
Élevée	Élevée	Underfitting
Faible	Élevée	Overfitting
Faible	Faible	Bonne généralisation apparente
Élevée	Faible	Situation inhabituelle, à examiner

Ces diagnostics sont des indications, pas des preuves absolues. Il faut notamment vérifier que les métriques et les distributions sont comparables.

## 3. Visualisation interactive : la complexité d'un modèle

Dans cet exemple, nous allons faire varier le degré d'un polynôme utilisé pour approximer des observations bruitées.
[[polynomial]]

La leçon est importante : un modèle plus complexe n'est pas automatiquement meilleur.
4. Le compromis biais-variance
Le biais et la variance permettent d'analyser les sources d'erreur d'un modèle.

### 4.1 Le biais

Le biais mesure l'écart entre la prédiction moyenne d'un algorithme, sur différents échantillons d'entraînement, et la vraie fonction que l'on cherche à estimer.

Un biais élevé peut indiquer que le modèle utilise des hypothèses trop restrictives.

### 4.2 La variance

La variance mesure la sensibilité des prédictions aux changements de l'échantillon d'entraînement.

Un modèle à variance élevée peut produire des prédictions très différentes selon les données utilisées pour l'entraîner.

### 4.3 Décomposition mathématique

Dans le cadre de la régression avec perte quadratique, sous les hypothèses classiques :
\[
\mathbb E[(Y-\hat f(x))^2]
=
\operatorname{Bias}[\hat f(x)]^2
+
\operatorname{Var}[\hat f(x)]
+
\sigma^2
\]

Où :
- Le biais au carré représente l'erreur systématique.
- La variance représente la sensibilité à l'échantillon d'entraînement.
- \(\sigma^2\) représente le bruit irréductible.
Ce dernier terme est essentiel : même un excellent modèle ne peut pas éliminer toute l'incertitude présente dans les données.
[[biasvariance]]

Attention : le biais-variance est un cadre d'analyse, et non une règle selon laquelle toute augmentation de complexité doit nécessairement diminuer le biais et augmenter la variance.

## 5. Comment diagnostiquer l'overfitting ?

Nous disposons de plusieurs méthodes.

### 5.1 Comparer train et validation

Il faut calculer la même métrique sur les deux ensembles et examiner leur écart.

Par exemple :
Modèle	Accuracy train	Accuracy validation
A	72 %	70 %
B	91 %	89 %
C	100 %	73 %

Le modèle C présente un écart important qui suggère un problème de généralisation.

Le modèle B semble ici plus intéressant, sous réserve de vérifier la qualité du protocole de validation et les autres métriques pertinentes.

### 5.2 Learning Curves

Les courbes d'apprentissage représentent les performances du modèle selon la quantité de données utilisées pour l'entraînement.
[[learningcurves]]

Les learning curves peuvent nous aider à déterminer si davantage de données sont susceptibles d'améliorer la performance, ou si le modèle semble limité par sa capacité à représenter les relations.

### 5.3 Validation Curves

Les validation curves examinent l'évolution des performances selon un hyperparamètre, par exemple max_depth pour un arbre de décision.

Elles permettent d'identifier les réglages qui conduisent à un modèle trop simple ou trop complexe.

## 6. Comment réduire l'overfitting ?

Méthode	Principe	Exemple
Plus de données pertinentes	Améliorer la couverture de la distribution	Collecter davantage d'observations
Régularisation	Pénaliser certaines formes de complexité	L1, L2
Réduire la complexité	Limiter la capacité du modèle	max_depth
Feature selection	Éliminer les variables peu pertinentes	Sélection dans un pipeline
Early stopping	Arrêter avant un surajustement excessif	Boosting, réseaux de neurones
Data augmentation	Créer des variantes pertinentes	Transformations d'images
Bagging	Réduire certaines sources de variance	Random Forest

La validation croisée est également essentielle, mais il faut bien comprendre son rôle : elle ne réduit pas directement l'overfitting. Elle aide surtout à mieux estimer les performances et à choisir les modèles.

### 6.1 Régularisation L1 et L2

La régularisation consiste à ajouter un terme de pénalité à la fonction objectif.

Pour L2 :
\[
J_{\text{reg}}(w)=J(w)+\lambda\sum_jw_j^2
\]

Pour L1 :
\[
J_{\text{reg}}(w)=J(w)+\lambda\sum_j|w_j|
\]

L1 peut conduire certains coefficients à zéro, ce qui favorise des solutions parcimonieuses.

L2 tend à réduire les coefficients sans nécessairement les annuler.

Nous approfondirons ces mécanismes dans le chapitre sur la régression et la régularisation.

## 7. Coding Lab — Observer le surapprentissage avec Python

Nous allons maintenant entraîner de vrais modèles polynomiaux de complexités différentes.

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(100, 1))
y = np.sin(X[:, 0]) + rng.normal(0, 0.25, 100)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)
degrees = [1, 3, 15]
x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, degree in zip(axes, degrees):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(), LinearRegression()
    )
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    val_mse = mean_squared_error(y_val, model.predict(X_val))
    print(degree, "MSE train:", train_mse, "MSE validation:", val_mse)
    ax.scatter(X_train[:, 0], y_train, label="train", s=14)
    ax.scatter(X_val[:, 0], y_val, label="validation", s=14)
    ax.plot(x_plot[:, 0], model.predict(x_plot), color="olive")
    ax.set_title(f"Degré {degree}")
    ax.set_ylim(-3, 3)
    ax.legend()
fig.tight_layout()
plt.show()
~~~

Analyse attendue
Le modèle de degré 1 risque de ne pas capturer correctement la forme sinusoïdale.

Le modèle de degré 3 peut mieux approximer cette relation sur l'intervalle observé.

Le modèle de degré 15 dispose de beaucoup plus de flexibilité et peut s'ajuster au bruit, particulièrement avec un petit échantillon.

Les résultats exacts dépendent des données et du découpage. Il faut examiner les courbes et les erreurs plutôt que supposer qu'un degré précis sera toujours optimal.

## 8. Coding Challenge — Régularisation et généralisation

Challenge 04 — Polynomial Regression
Projet guidé · validation par livrables
Ton objectif est de comparer une régression polynomiale non régularisée à une régression Ridge.

Tu dois mesurer les erreurs train et validation, puis étudier l'effet du paramètre alpha.

~~~python
# Reprendre X_train, X_val, y_train, y_val du Coding Lab.
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(
        PolynomialFeatures(degree=15, include_bias=False),
        StandardScaler(), Ridge(alpha=alpha)
    )
    # TODO : fit sur train, puis MSE train et validation.
    # TODO : tracer les erreurs en fonction de alpha.
    pass
# Comparer les réglages sur validation, pas sur un test final.
~~~

Question à résoudre : est-ce que la régularisation améliore systématiquement la performance de validation lorsque alpha augmente ?

Indice : une régularisation excessive peut provoquer de l'underfitting.

## 9. Cas pratique — Diagnostic d'un modèle ML

[[quiz]]

## 10. Questions d'entretien — Niveau avancé

[[interview]]

## 11. Validation du cours

[[assessment]]

À retenir : l'objectif du Machine Learning n'est pas d'obtenir le score d'entraînement le plus élevé, mais de construire un modèle capable de généraliser à de nouvelles observations.

Le prochain cours, « Découper les données correctement », approfondira les stratégies de validation, notamment StratifiedKFold, GroupKFold, TimeSeriesSplit et la Nested Cross-Validation.