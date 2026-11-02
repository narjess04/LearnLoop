Dans le cours précédent, nous avons étudié ce qu'est le Machine Learning, ses différentes familles et les concepts fondamentaux.

Aujourd'hui, nous allons construire un véritable modèle de classification supervisée.

Notre objectif n'est pas seulement de faire fonctionner le code. À la fin de ce cours, tu devras pouvoir expliquer chaque étape, justifier chaque décision technique et reconstruire le pipeline sans recopier un tutoriel.

## 1. Comprendre le cycle de vie d'un modèle ML

Un projet Machine Learning ne se limite pas à appeler model.fit().
[[lifecycle]]

Vue pédagogique simplifiée. Dans un projet réel, l'exploration, la validation et l'amélioration du modèle forment un processus itératif.
Nous utiliserons les outils suivants :
Bibliothèque	Rôle
NumPy	Calculs numériques
Pandas	Manipulation des données
Scikit-learn	Entraînement et évaluation
Matplotlib	Visualisation

## 2. Comprendre notre problème métier

Nous allons utiliser le dataset Breast Cancer Wisconsin, disponible directement dans scikit-learn.

Il contient des mesures numériques issues d'images de cellules, associées à deux catégories diagnostiques. Nous l'utilisons exclusivement comme dataset pédagogique pour comprendre la classification, pas comme outil médical.
[[dataset]]

Dataset intégré à scikit-learn. Dans son encodage : 0 = malignant, 1 = benign.

### 2.1 Définir X et y

Nous disposons d'une matrice de caractéristiques :
\[
X \in \mathbb{R}^{569 \times 30}
\]

Et d'un vecteur de classes :
\[
y \in \{0,1\}^{569}
\]

Chaque ligne de \(X\) représente une observation. Chaque colonne correspond à une caractéristique numérique.

L'objectif est d'apprendre une fonction de prédiction :
\[
f_\theta:X\rightarrow \{0,1\}
\]

Plus précisément, notre régression logistique estimera une probabilité, puis appliquera un seuil pour produire une classe.

## 3. Charger et explorer les données

~~~python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)
print(X.head())
print(y.value_counts())
print("Valeurs manquantes :", X.isna().sum().sum())
print(X.describe())
print("Classes :", dict(enumerate(dataset.target_names)))
~~~

Pourquoi explorer les données avant l'entraînement ?
Parce qu'un algorithme ne peut pas corriger automatiquement tous les problèmes présents dans un dataset. Ici l’inspection initiale sert à comprendre le schéma du jeu pédagogique. Dans un projet réel, réserver le test tôt et ne pas exploiter ses distributions pour choisir des transformations ou des variables.

Nous devons notamment examiner la distribution des variables, les valeurs manquantes, les classes, les doublons et la cohérence des données.

Dans ce dataset, les 30 features sont numériques et il n'y a pas de valeurs manquantes.

Nous développerons l'EDA en profondeur dans le chapitre consacré au preprocessing.

## 4. Train, validation et test : pourquoi séparer les données ?

C'est l'un des concepts les plus importants en Machine Learning.

Si nous entraînons et évaluons notre modèle sur exactement les mêmes observations, nous risquons de surestimer sa capacité à généraliser.

Nous devons donc réserver des données qui ne serviront pas à apprendre les paramètres du modèle.
Exemple de découpage 60 % / 20 % / 20 %
Train 60 %

Val 20 %

Test 20 %

Training set — Apprendre les paramètres du modèle.

Validation set — Comparer les modèles et choisir les hyperparamètres.

Test set — Estimer la performance finale sur des données mises de côté.

Dans notre premier exemple, nous utiliserons simplement un split 80 % train / 20 % test, et nous ne ferons pas d'optimisation d'hyperparamètres.

Plus tard, nous utiliserons la cross-validation pour sélectionner les modèles sans exploiter le test.

### 4.1 Implémentation

~~~python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("Répartition train :", y_train.value_counts(normalize=True))
print("Répartition test :", y_test.value_counts(normalize=True))
~~~

Comprendre chaque paramètre
- test_size=0.20 : réserver 20 % des observations pour le test.
- stratify=y : conserver approximativement les proportions des classes dans chaque ensemble.
- random_state=42 : rendre le découpage aléatoire reproductible.
Question fréquente en entretien : pourquoi utiliser stratify ?

Parce qu'un découpage aléatoire peut créer des proportions de classes différentes entre train et test, particulièrement lorsque les classes sont déséquilibrées.

La stratification aide à maintenir une répartition comparable. Elle ne résout cependant pas, à elle seule, le problème du déséquilibre des classes.

## 5. Feature Scaling : pourquoi normaliser les variables ?

Imaginons deux caractéristiques :
- Revenu annuel : 50 000
- Ratio d'endettement : 0,35
Leurs échelles sont très différentes.

Certains algorithmes, comme la régression logistique régularisée, les SVM et KNN, sont sensibles à l'échelle des variables.

Nous allons utiliser StandardScaler.

### 5.1 Standardisation : la formule

\[
z=\frac{x-\mu}{\sigma}
\]

Avec :
- \(x\) : valeur initiale.
- \(\mu\) : moyenne calculée sur le train.
- \(\sigma\) : écart-type calculé sur le train.
- \(z\) : valeur standardisée.
Par exemple, si \(x=120\), \(\mu=100\) et \(\sigma=10\), alors \(z=2\).
[[scaler]]

Pour une variable constante, StandardScaler conserve une échelle égale à 1 : on ne divise pas par zéro. Standardiser ne rend pas automatiquement une distribution normale.

### 5.2 Pourquoi ne pas standardiser avant le split ?

Parce que la moyenne et l'écart-type calculés sur toutes les données contiendraient des informations issues du test set.

Cela crée une fuite d'informations (data leakage).

La bonne approche consiste à apprendre les paramètres de standardisation uniquement sur le train, puis à réutiliser ces paramètres sur les nouvelles observations.

## 6. Pipeline : combiner preprocessing et modèle

Scikit-learn propose un objet très important : Pipeline.

Il permet d'enchaîner plusieurs transformations et un estimateur final.

~~~python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
~~~

Que se passe-t-il lors de model.fit() ?

## 1. Le StandardScaler calcule la moyenne et l'écart-type des features du train.

## 2. Il transforme les observations d'entraînement.

## 3. La LogisticRegression apprend ses coefficients à partir des données transformées.

Lors de model.predict(X_test), le scaler applique les statistiques apprises sur le train, puis le classifieur produit les prédictions.
Point important pour un entretien avancé
Pourquoi utiliser un Pipeline plutôt que standardiser manuellement ?

Parce qu'il facilite la reproductibilité, réduit les risques de data leakage et permet d'intégrer les transformations dans la cross-validation et l'optimisation des hyperparamètres.

## 7. Entraînement : que fait réellement fit() ?

Dans notre exemple, nous utilisons une régression logistique.

Elle estime une probabilité à partir d'une combinaison linéaire des caractéristiques :
\[
P(y=1\mid x)=\sigma(w^Tx+b)
\]

Ici x désigne une observation (vecteur de 30 features), et non la matrice X entière. Avec la fonction sigmoid :
\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

La fonction sigmoid transforme un nombre réel en une valeur comprise entre 0 et 1.
[[sigmoid]]

Nous étudierons la dérivation de la log-loss et l'optimisation de ses coefficients dans le cours consacré à la régression logistique.

Pour l'instant, retiens que fit() apprend les paramètres à partir des données d'entraînement, tandis que predict() utilise les paramètres déjà appris.

## 8. Évaluer notre modèle

Maintenant que le modèle est entraîné, nous allons prédire les classes du test set.

~~~python
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

y_pred = model.predict(X_test)
# Vérifier l'ordre des colonnes plutôt que le supposer.
positive_column = list(model.classes_).index(1)
y_proba = model.predict_proba(X_test)[:, positive_column]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC (classe positive 1) :", roc_auc_score(y_test, y_proba))
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["malignant", "benign"], zero_division=0
))
print("Lignes = réel, colonnes = prédit, ordre [0, 1]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
~~~

Complément : une baseline à comparer

~~~python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Accuracy de la baseline :",
      accuracy_score(y_test, baseline.predict(X_test)))
# Comparaison pédagogique fixée à l'avance.
# Ne pas régler les hyperparamètres sur le test.
~~~

### 8.1 Accuracy

L'accuracy mesure la proportion de prédictions correctes.
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Mais elle peut être trompeuse.

Imagine un dataset où 99 % des observations appartiennent à la classe 0. Un modèle qui prédit toujours 0 atteint 99 % d'accuracy, sans détecter aucune observation de la classe 1.

C'est pourquoi nous devons également examiner la précision, le rappel, le F1-score, la ROC-AUC et parfois la PR-AUC.

### 8.2 Matrice de confusion

[[confusion]]

Exemple pédagogique indépendant des résultats du modèle entraîné.
Ici, la classe positive est la classe 1. Dans notre dataset scikit-learn, il s'agit de la classe benign. Il faut donc toujours préciser quelle classe est considérée comme positive avant d'interpréter precision et recall.

[[verification]]

## 9. Comprendre l'API scikit-learn

Voici les méthodes que tu dois connaître :
Méthode	Rôle
fit(X, y)	Entraîner le modèle
predict(X)	Prédire des classes ou valeurs
predict_proba(X)	Estimer les probabilités des classes, si disponible
transform(X)	Transformer des données
fit_transform(X)	Apprendre puis appliquer une transformation
score(X, y)	Calculer le score par défaut de l'estimateur

Attention : score() ne renvoie pas systématiquement l'accuracy. Pour une régression linéaire scikit-learn, par exemple, il renvoie généralement le coefficient \(R^2\).

## 10. Projet guidé — Wine Classification

Cette fois, je ne te donne pas directement la solution complète.
Challenge 02 — Wine Classification
Projet guidé · validation par livrables

Tu dois entraîner un modèle sur le dataset Wine de scikit-learn, qui contient trois classes.

Objectifs :
- Charger le dataset.
- Faire un split stratifié 80/20.
- Construire un pipeline StandardScaler + LogisticRegression.
- Entraîner le modèle.
- Calculer accuracy et macro F1-score.
- Afficher les probabilités prédites pour cinq observations.
- Expliquer pourquoi chaque ligne du pipeline est nécessaire.

~~~python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = load_wine(return_X_y=True, as_frame=True)
# TODO : split stratifié 80/20 (random_state=42)
# TODO : pipeline StandardScaler + LogisticRegression
# TODO : fit sur train uniquement
# TODO : prédire le test et calculer accuracy et F1 macro
# TODO : afficher les probabilités de cinq observations
# TODO : relier chaque colonne à model.classes_
~~~

## 11. Questions d'entretien

[[interview]]

## 12. Validation du cours

[[assessment]]

Ce que tu dois retenir : construire un modèle ML, ce n'est pas seulement choisir un algorithme. C'est organiser correctement les données, apprendre sans fuite d'informations, évaluer honnêtement les performances et pouvoir reproduire les résultats.

Le prochain cours, « Comment un modèle apprend-il réellement ? », sera plus mathématique. Nous allons ouvrir la boîte noire de fit() pour comprendre exactement comment un modèle apprend ses paramètres, puis implémenter la descente de gradient from scratch avec NumPy.

## Livrables du projet Wine

- Un script relançable avec les versions de Python et scikit-learn.
- Un split stratifié documenté, une baseline et un pipeline sans fuite.
- Un rapport accuracy/F1 macro et une matrice de confusion multiclasses.
- Cinq lignes de probabilités, avec l’ordre des classes expliqué et des sommes proches de 1.
- Une analyse de quelques erreurs et les limites de ce petit dataset.

Ce projet se retrouve aussi dans l’espace Projets. Les étapes cochées sont un suivi personnel, pas une revue automatique du code.