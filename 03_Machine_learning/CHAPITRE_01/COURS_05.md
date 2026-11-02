Dans le cours précédent, nous avons compris pourquoi un modèle peut être excellent sur les données d'entraînement et mauvais sur de nouvelles observations.

Aujourd'hui, nous allons apprendre à mesurer correctement la capacité de généralisation d'un modèle.

C'est un sujet fondamental pour les Data Scientists et ML Engineers, mais également une source fréquente de questions techniques en entretien.

## 1. Pourquoi découper les données ?

Supposons que nous construisions un modèle de prédiction du défaut de crédit sur 10 000 observations.

Nous entraînons le modèle sur les 10 000 observations, puis nous calculons une accuracy de 98 % sur ces mêmes observations.

Pouvons-nous affirmer que notre modèle est performant ?

Non. Ce résultat mesure uniquement sa performance sur les données utilisées pendant l'apprentissage.

Il ne permet pas d'estimer honnêtement comment il se comportera sur de nouveaux clients.

Pour cela, nous devons séparer les données en plusieurs ensembles.

## 2. Training, Validation et Test Set

[[splitbar]]

### 2.1 Training Set

Le training set est utilisé pour apprendre les paramètres du modèle.

Par exemple, dans une régression logistique, il sert à estimer les coefficients \(w\) et le biais \(b\).

### 2.2 Validation Set

Le validation set sert à comparer plusieurs configurations, sélectionner un modèle et régler ses hyperparamètres.

Exemples : choisir max_depth d'un arbre ou la force de régularisation d'une régression logistique.

### 2.3 Test Set

Le test set est réservé à l'évaluation finale, une fois les choix de modèle et d'hyperparamètres terminés.

Si nous modifions plusieurs fois notre modèle en fonction des résultats du test, nous commençons à surajuster nos décisions à ce test.

Principe essentiel : le test set ne doit pas devenir un second validation set.

## 3. Holdout Validation — Le découpage simple

La stratégie la plus simple consiste à réserver une partie des données pour la validation ou le test.

En Python :

~~~python
from sklearn.model_selection import train_test_split
# X et y représentent ici des observations indépendantes.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
~~~

Comprendre les paramètres
Paramètre	Signification
test_size=0.2	Réserve 20 % des données
random_state=42	Rend le découpage reproductible
stratify=y	Préserve approximativement les proportions des classes
shuffle=True	Mélange les observations avant le découpage, par défaut

Ce découpage convient à de nombreux datasets tabulaires dont les observations peuvent raisonnablement être considérées comme indépendantes et échangeables.

Mais il présente une limite : le résultat dépend du découpage choisi.

Un split peut être plus favorable qu'un autre.

C'est pour réduire cette dépendance que nous utilisons la cross-validation.

## 4. K-Fold Cross-Validation

La K-Fold Cross-Validation consiste à diviser les données d'entraînement en \(K\) parties, appelées folds.

À chaque itération :
1. Nous entraînons le modèle sur \(K-1\) folds.
2. Nous évaluons le modèle sur le fold restant.
3. Nous répétons l'opération jusqu'à ce que chaque fold ait servi à la validation.
[[folds]]

### 4.1 Calculer le score moyen

Supposons que nous obtenions les accuracies suivantes :
Fold	Accuracy
1	0,88
2	0,91
3	0,89
4	0,92
5	0,90

La moyenne est :
\[
\overline{S}=\frac{1}{K}\sum_{k=1}^{K}S_k
\]

Dans cet exemple :
\[
\overline{S}=0.90
\]

Nous pouvons aussi calculer l'écart-type des scores :
\[
s=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(S_k-\overline{S})^2}
\]

Ici, cette formule correspond à l'écart-type descriptif avec dénominateur \(K\), soit environ 0,014.

Un écart-type faible signifie que les scores des folds sont proches. Il ne garantit toutefois pas que le modèle sera stable en production, et cet écart-type ne constitue pas à lui seul un intervalle de confiance statistique.

### 4.2 Implémentation avec scikit-learn

~~~python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
print("Scores :", scores)
print("Moyenne :", scores.mean(), "Écart-type descriptif :", scores.std())
~~~

À chaque fold, scikit-learn ajuste un nouveau pipeline sur les données d'entraînement du fold. Le scaler n'est donc pas entraîné sur les observations de validation.

## 5. Stratified K-Fold

Pour les problèmes de classification, notamment lorsque les classes sont déséquilibrées, nous utilisons fréquemment StratifiedKFold.
Exemple
Supposons un dataset contenant 1 000 observations :
Distribution des classes
Classe 0 — 90 %

10 %

900 observations de classe 0 et 100 observations de classe 1.

Avec une K-Fold classique, les proportions peuvent varier d'un fold à l'autre.

La stratification cherche à préserver approximativement les proportions de chaque classe dans les différents folds.

~~~python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
# ROC-AUC binaire ; chaque fold doit contenir les deux classes.
~~~

KFold versus StratifiedKFold
KFold	StratifiedKFold
Découpage en K groupes	Découpage en K groupes stratifiés
Ne préserve pas nécessairement les proportions des classes	Préserve approximativement les proportions
Convient à de nombreux problèmes de régression	Particulièrement adapté à la classification

La stratification ne garantit pas une estimation non biaisée si d'autres dépendances existent entre les observations.

## 6. GroupKFold — Éviter la contamination entre groupes

Imaginons un dataset médical où chaque patient possède plusieurs observations.

Si les mesures d'un même patient apparaissent à la fois dans l'entraînement et dans la validation, le modèle peut exploiter des caractéristiques propres à ce patient.

Cela peut produire une évaluation trop optimiste lorsque l'objectif est de généraliser à de nouveaux patients.

GroupKFold permet d'empêcher qu'un même groupe apparaisse simultanément dans les parties train et validation d'un fold.
Exemple : séparation par patient
Découpage incorrect
Train
A
B
C

Validation
A
D

Le patient A apparaît dans les deux ensembles.

Découpage correct
Train
A
B
C

Validation
D
E

Les patients sont distincts.

Implémentation

~~~python
from sklearn.model_selection import GroupKFold, cross_val_score
# X, y et groups doivent correspondre au même jeu groupé.
# groups contient par exemple un identifiant de sujet.
cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    model, X, y, groups=groups, cv=cv, scoring="accuracy"
)
# Vérifier au moins cinq groupes distincts.
~~~

Les groupes doivent être définis selon la dépendance réelle des données : patient, utilisateur, entreprise, document ou autre unité pertinente.

## 7. TimeSeriesSplit — Validation temporelle

Pour les séries temporelles, un découpage aléatoire peut conduire à entraîner le modèle sur des observations futures pour prédire des événements passés.

Cela ne correspond pas au scénario réel d'utilisation.

Prenons une entreprise qui souhaite prédire les ventes du mois prochain.

Nous devons respecter l'ordre chronologique.
TimeSeriesSplit Explorer

Chaque itération agrandit la fenêtre d'entraînement et avance la période de validation.
Fold 1

T

T

T

V

Fold 2

T

T

T

T

V

Fold 3

T

T

T

T

T

V

Fold 4

T

T

T

T

T

T

V

Train

Validation

Non utilisé

Schéma simplifié d'une validation temporelle à fenêtre d'entraînement croissante.
Implémentation

~~~python
from sklearn.model_selection import TimeSeriesSplit
# X doit être ordonné chronologiquement.
tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, val_idx in tscv.split(X):
    assert train_idx.max() < val_idx.min()
    print("Train :", train_idx, "Validation :", val_idx)
# Adapter gap au délai des labels et au chevauchement des fenêtres.
~~~

Le paramètre gap permet d'exclure un nombre d'observations entre train et validation. Il peut être utile lorsqu'il faut éviter une contamination liée aux fenêtres temporelles.

Attention : TimeSeriesSplit ne garantit pas automatiquement l'absence de fuite temporelle. Les features doivent elles-mêmes être construites uniquement avec les informations disponibles au moment de la prédiction.

## 8. Nested Cross-Validation — Niveau avancé

La Nested Cross-Validation est une technique importante lorsqu'on souhaite évaluer un processus complet de sélection de modèle et d'hyperparamètres.

Elle utilise deux boucles.
- Inner CV : sélectionne les hyperparamètres.
- Outer CV : estime la performance du processus de sélection sur des folds externes.

Exemple avec scikit-learn

~~~python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
search = GridSearchCV(
    pipeline, {"logisticregression__C": [0.01, 0.1, 1, 10]},
    cv=inner_cv, scoring="roc_auc", n_jobs=1
)
# Ajustement et sélection se déroulent à l'intérieur de chaque fold externe.
scores = cross_val_score(search, X_train, y_train, cv=outer_cv, scoring="roc_auc")
print("Évaluation externe :", scores, scores.mean())
~~~

Pourquoi est-ce important ?
Si nous utilisons les mêmes folds pour sélectionner les hyperparamètres et annoncer la performance finale, l'estimation peut être optimiste.

La Nested CV permet de mieux séparer l'optimisation des hyperparamètres de l'évaluation du processus de sélection.

Elle est particulièrement utile pour les petits et moyens datasets et les comparaisons rigoureuses de modèles, mais elle peut être coûteuse en calcul.

## 9. Comment choisir la bonne stratégie ?

Situation	Stratégie généralement adaptée
Dataset tabulaire, observations indépendantes	Holdout ou KFold
Classification déséquilibrée	StratifiedKFold
Plusieurs observations par patient ou client	GroupKFold
Données temporelles	TimeSeriesSplit ou validation chronologique adaptée
Sélection d'hyperparamètres avec estimation rigoureuse	Nested CV
Très peu de données	K-Fold adapté, parfois Leave-One-Out

Il existe aussi des situations combinant plusieurs contraintes, par exemple des données temporelles avec plusieurs observations par client. Dans ce cas, une stratégie de validation personnalisée peut être nécessaire.

## 10. Coding Lab — Comparer les méthodes de validation

Nous allons comparer KFold et StratifiedKFold sur un dataset de classification déséquilibré.

~~~python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=8,
    n_redundant=4, weights=[0.9, 0.1], random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
strategies = {
    "KFold": KFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
}
for name, cv in strategies.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(name, scores, scores.mean(), scores.std())
# Aucun des deux protocoles n'est adapté automatiquement au temps ou aux groupes.
~~~

Analyse attendue
Compare les scores moyens, leur dispersion et les proportions des classes dans chaque fold.

Ne conclus pas automatiquement que StratifiedKFold doit donner une meilleure moyenne. Son intérêt principal est de préserver la distribution des classes et de rendre les folds plus comparables sur cet aspect.

## 11. Coding Challenge — Validation temporelle

Challenge 05 — Time Series Validation
Projet guidé · validation par livrables
Tu disposes d'une série de ventes mensuelles.

Tu dois construire des variables retardées (lag features), appliquer TimeSeriesSplit et comparer les performances d'un modèle avec une baseline naïve.

~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
time = np.arange(120)
sales = 100 + 0.5*time + 10*np.sin(2*np.pi*time/12) + rng.normal(0, 5, 120)
df = pd.DataFrame({"month": time, "sales": sales})
for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)
df = df.dropna()
X, y = df[["lag_1", "lag_2", "lag_3"]], df["sales"]
# TODO : TimeSeriesSplit, fit et MAE par fold.
# TODO : comparer à la baseline y_pred = lag_1.
# TODO : définir l'horizon ; les vrais lags de validation ne sont disponibles
# qu'en prévision un pas avec observation des valeurs passées.
# Pour un horizon multi-pas, utiliser un protocole compatible (récursif/direct).
~~~

Pour chaque fold, la baseline naïve prédit que les ventes du mois courant sont égales à celles du mois précédent.

L'objectif est de vérifier si le modèle ML apporte réellement une amélioration par rapport à cette règle simple.

## 12. Questions d'entretien

[[interview]]

## 13. Validation du cours

[[assessment]]

À retenir : une bonne stratégie de validation doit reproduire aussi fidèlement que possible la manière dont le modèle sera utilisé sur de nouvelles données. Le bon découpage dépend donc du problème, des dépendances entre observations et du contexte métier.

Le prochain cours sera consacré au Data Leakage, l'une des causes les plus trompeuses de performances artificiellement élevées en Machine Learning.

### Attention à l’horizon temporel

Les lags réels de la période de validation simulent une prévision à un pas, si les observations précédentes arrivent entre deux prédictions. Une prévision multi-pas doit construire ses lags selon le scénario de disponibilité, sans utiliser les cibles futures.