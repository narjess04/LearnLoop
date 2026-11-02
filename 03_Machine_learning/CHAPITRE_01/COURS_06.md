Chapitre 1 : Fondements du Machine Learning
MACHINE LEARNING · CHAPITRE 01 · COURS 06

Data Leakage — Comprendre, détecter et éviter les fuites de données
Pourquoi un modèle peut afficher 99 % de performance en validation, puis échouer en production ? Apprendre à identifier les fuites d'informations et construire des pipelines fiables.
 3–4 heures
Théorie + Python
Entretiens ML Engineer

Progression du chapitre

Cours 6 / 8

Dans le cours 5, nous avons appris à séparer les données avec train_test_split, StratifiedKFold, GroupKFold et TimeSeriesSplit.

Mais même avec un bon découpage, une erreur de préparation peut fausser toute l'évaluation.

C'est ce que nous allons étudier aujourd'hui : le Data Leakage.

## 1. Qu'est-ce que le Data Leakage ?

Le Data Leakage, ou fuite de données, se produit lorsque des informations qui ne devraient pas être disponibles au moment de l'entraînement ou de la prédiction influencent le modèle ou son évaluation.

Cela peut créer une performance artificiellement élevée.
Exemple concret
Imaginons que nous souhaitions prédire si un client remboursera son crédit.

Notre dataset contient :
Feature	Description	Disponible au moment de la demande ?
income	Revenu du client	Oui
debt_ratio	Ratio d'endettement	Oui
credit_history	Historique de crédit	Oui
recovery_amount	Montant récupéré après un défaut	Non
default_flag	Variable cible	Non, pour une nouvelle demande

Si nous utilisons recovery_amount pour prédire le défaut au moment de l'octroi, le modèle dispose d'une information future.

Il pourrait apprendre une relation très forte, mais cette variable ne sera pas disponible lors de la prédiction réelle.

Résultat : excellente performance pendant l'évaluation, puis échec au déploiement.

## 2. Les principales catégories de Data Leakage

Target Leakage

Une feature révèle directement ou indirectement la réponse que nous souhaitons prédire.

Exemple : utiliser le statut final d'un dossier pour prédire son statut à l'ouverture.

Train–Test Contamination

Les données de validation ou de test influencent une étape d'apprentissage.

Exemple : effectuer une sélection de features sur tout le dataset avant la cross-validation.

Temporal Leakage

Le modèle exploite des informations provenant du futur par rapport à la date de prédiction.

Exemple : utiliser les ventes de décembre pour prédire les ventes de novembre.

Group / Duplicate Leakage

Des observations liées ou dupliquées se retrouvent dans les deux ensembles, alors que l'objectif est de généraliser à de nouveaux groupes.

Exemple : plusieurs lignes d'un même patient ou client réparties entre train et test.

Ces catégories peuvent se chevaucher. Par exemple, une variable créée après l'événement cible peut être à la fois une fuite temporelle et une fuite de cible.
3. Le piège du preprocessing avant le split
C'est une erreur fréquente chez les débutants.

### 3.1 La mauvaise approche

~~~python
# CONTRE-EXEMPLE : fuite de prétraitement, à ne pas reproduire.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
~~~

Pourquoi est-ce incorrect ?

Parce que fit_transform(X) calcule la moyenne et l'écart-type à partir de toutes les observations, y compris celles du test.

Le preprocessing a donc appris des informations sur la distribution du test.

Cette contamination ne produit pas nécessairement une forte augmentation du score, mais elle invalide la séparation stricte entre entraînement et évaluation.

### 3.2 La bonne approche

~~~python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
~~~

Le pipeline garantit ici que le scaler apprend ses statistiques uniquement à partir du train.
Comprendre la différence
Incorrect

Correct

## 4. Les autres transformations qui peuvent provoquer une fuite

Le scaling n'est pas le seul risque.
Opération	Risque si effectuée avant le split ou hors des folds
Imputation	Statistiques calculées sur des données de validation
Feature selection	Sélection influencée par les cibles de validation
PCA	Composantes apprises sur des observations de validation
Target encoding	Utilisation indirecte de la cible
SMOTE	Création d'exemples synthétiques avant le découpage
Normalisation	Paramètres calculés sur le dataset complet
Hyperparameter tuning	Choix influencés par le test final

La règle générale est la suivante :

Toute transformation qui apprend des paramètres à partir des données doit être ajustée uniquement sur la partie d'entraînement pertinente.

Pour le rééquilibrage des classes avec SMOTE, on utilise par exemple un pipeline compatible avec les méthodes de rééchantillonnage, comme celui d'imbalanced-learn.

## 5. Expérience Python — Une fuite volontaire de la cible

Nous allons construire une expérience contrôlée pour observer comment une feature contenant directement la cible peut fausser les performances.
Expérience : dataset sain versus dataset contaminé

Nous créons un problème de classification, puis nous ajoutons volontairement une colonne future_outcome égale à la cible. C'est une fuite directe, utilisée uniquement pour démontrer le problème.

~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_array, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5,
    n_redundant=2, flip_y=0.05, random_state=42
)
X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(10)])
train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
def evaluate(features):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features.iloc[train_idx], y[train_idx])
    proba = model.predict_proba(features.iloc[test_idx])[:, 1]
    return roc_auc_score(y[test_idx], proba)

print("Features originales :", evaluate(X))
leaked = X.copy()
leaked["future_target"] = y  # INFORMATION INTERDITE à la prédiction.
print("Avec fuite volontaire :", evaluate(leaked))
# Le même split ne rend pas une feature issue de la cible légitime.
~~~

Ce que tu dois observer
Le dataset contaminé peut obtenir une ROC-AUC proche de 1, puisque la feature future_outcome révèle directement la réponse.

Cette performance ne signifie pas que le modèle a découvert une relation prédictive exploitable.

Elle signifie qu'il a reçu une information qu'il ne devrait pas posséder.

## 6. Temporal Leakage — Le problème des informations futures

Prenons un exemple de prévision des ventes.

Nous voulons prédire les ventes du jour \(t\).

Une feature construite avec les ventes du jour \(t+1\) n'est pas disponible au moment de la prédiction.
Exemple incorrect
df["future_sales"] = df["sales"].shift(-1)

Cette colonne contient les ventes du lendemain.

Si elle est utilisée pour prédire les ventes du jour, elle introduit une information future.
Exemple correct
df["sales_lag_1"] = df["sales"].shift(1)df["sales_lag_7"] = df["sales"].shift(7)

Ces features utilisent les ventes passées.

Mais attention : leur validité dépend du moment exact où la prédiction est effectuée. Une feature historiquement disponible n'est pas forcément accessible dans tous les scénarios opérationnels.
Le concept de Prediction Time
Une bonne question à se poser pour chaque feature :
Cette information était-elle réellement disponible au moment où le modèle devait effectuer sa prédiction ?

C'est l'un des contrôles les plus importants pour éviter le leakage.

## 7. Group Leakage et doublons

Supposons que nous possédions 1 000 clients, chacun avec plusieurs observations.

Si les observations d'un même client sont réparties entre train et test, le modèle peut apprendre des particularités propres à ce client.

Cela devient problématique si notre objectif est de généraliser à de nouveaux clients.
Exemple de contrôle
train_clients = set(train_df["client_id"])test_clients = set(test_df["client_id"])overlap = train_clients.intersection(test_clients)print("Clients communs :", len(overlap))

Si le nombre de clients communs est supérieur à zéro, il faut vérifier si ce chevauchement est compatible avec le scénario métier.

Pour une généralisation à de nouveaux clients, on peut utiliser GroupShuffleSplit ou GroupKFold.

Pour une prédiction future sur des clients déjà connus, la séparation temporelle peut être plus pertinente.

## 8. Comment détecter un Data Leakage ?

Il n'existe pas de test unique capable de détecter toutes les fuites.

La détection repose sur une combinaison de contrôles techniques et métier.
Checklist d'audit
Data Leakage Audit

0/10

## 9. Coding Lab — Feature Selection sans Data Leakage

Une erreur subtile consiste à sélectionner les features en utilisant tout le dataset avant la cross-validation.
Mauvaise approche

~~~python
# CONTRE-EXEMPLE : la sélection voit les labels de futurs folds de validation.
from sklearn.feature_selection import SelectKBest, f_classif
X_selected = SelectKBest(f_classif, k=10).fit_transform(X, y)
~~~

La sélection des features utilise ici les cibles de toutes les observations.

Si nous faisons ensuite une cross-validation sur X_selected, les folds de validation ont déjà influencé le choix des features.
Bonne approche

~~~python
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=1000, n_features=100, n_informative=10,
    n_redundant=10, random_state=42
)
pipeline = make_pipeline(
    SelectKBest(f_classif, k=10), StandardScaler(),
    LogisticRegression(max_iter=1000)
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
print(scores, scores.mean(), scores.std())
~~~

Grâce au pipeline, la sélection des features est recalculée uniquement à partir du train de chaque fold.

## 10. Diagnostic interactif — Trouve la fuite

[[quiz]]

## 11. Coding Challenge — Construire un pipeline sans fuite

Challenge 06 — Leakage-Free Pipeline
Projet guidé · validation par livrables
Construis un pipeline de classification intégrant une imputation, une standardisation, une sélection de features et une régression logistique.

Évalue ensuite le modèle avec StratifiedKFold, sans fuite d'informations.

~~~python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = make_classification(
    n_samples=1000, n_features=30, n_informative=8, random_state=42
)
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.05] = np.nan
# TODO : pipeline imputation médiane -> scaler -> sélection -> logistique.
# TODO : StratifiedKFold et scores, moyenne, écart-type descriptif.
# TODO : expliquer dans quel périmètre chaque étape apprend ses paramètres.
~~~

## 12. Questions d'entretien — ML Engineer

[[interview]]

## 13. Validation du cours

[[assessment]]

À retenir : une performance élevée n'a de valeur que si le protocole d'évaluation est fiable. Le Data Leakage peut rendre un modèle artificiellement performant sans qu'il soit réellement capable de généraliser.

Le prochain cours sera consacré au Hyperparameter Tuning : comment choisir intelligemment les hyperparamètres d'un modèle avec GridSearchCV, RandomizedSearchCV et Optuna, sans surajuster les résultats de validation.