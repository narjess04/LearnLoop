Nous arrivons au dernier cours théorique du chapitre 1.

Jusqu'à présent, nous avons étudié les concepts séparément : apprentissage, fonctions de perte, généralisation, validation croisée, data leakage et optimisation des hyperparamètres.

Aujourd'hui, nous allons assembler toutes ces connaissances dans un véritable workflow de Machine Learning.

L'objectif est de comprendre non seulement comment entraîner un modèle, mais aussi comment conduire un projet de manière professionnelle.
1. Le cycle de vie d'un projet Machine Learning
Un projet ML ne commence pas avec model.fit().

Il commence par une question métier, puis progresse vers une solution évaluée et exploitable.
[[workflow]]
Workflow inspiré de CRISP-DM, complété par le déploiement et le monitoring. En pratique, les étapes sont itératives : on peut revenir à la préparation des données après l'évaluation.

## 2. Étape 1 — Business Understanding

Avant de construire un modèle, nous devons comprendre ce que nous souhaitons résoudre.

Prenons un cas de prédiction du churn : une entreprise veut identifier les clients susceptibles de résilier leur abonnement.

Il faut répondre à plusieurs questions :
- Quelle décision sera prise grâce à la prédiction ?
- À quel moment la prédiction sera-t-elle effectuée ?
- Quelles données seront disponibles à cet instant ?
- Quel est le coût des faux positifs et des faux négatifs ?
- Comment mesurera-t-on la réussite du projet ?

### 2.1 Transformer le besoin métier en problème ML

Question métier	Traduction ML
Quels clients risquent de partir ?	Classification binaire
Quel sera le montant des ventes ?	Régression
Quels clients ont des comportements similaires ?	Clustering
Quels produits recommander ?	Recommandation / ranking

### 2.2 Définir les métriques

La métrique doit refléter le besoin métier.

Par exemple, dans une détection de fraude, manquer une fraude peut coûter cher, mais déclencher trop de fausses alertes peut également poser problème.

Nous pouvons alors examiner le rappel, la précision, la courbe précision-rappel et les coûts associés aux erreurs.

Une accuracy élevée ne suffit pas toujours.

## 3. Étape 2 — Data Understanding

Nous devons ensuite comprendre les données.

L'Exploratory Data Analysis, ou EDA, permet d'identifier les caractéristiques du dataset et les problèmes potentiels.

### 3.1 Les principaux contrôles

Contrôle	Question
Dimensions	Combien de lignes et de colonnes ?
Types	Variables numériques ou catégorielles ?
Valeurs manquantes	Quelles variables sont incomplètes ?
Doublons	Plusieurs observations identiques ?
Distribution cible	Les classes sont-elles déséquilibrées ?
Outliers	Certaines valeurs sont-elles inhabituelles ?
Corrélations	Certaines variables sont-elles fortement liées ?
Temporalité	Les données ont-elles une structure chronologique ?

### 3.2 Exemple Python

~~~python
import pandas as pd
# Prévoir ce fichier et un schéma précisant notamment la cible.
df = pd.read_csv("dataset.csv")
print(df.shape)
df.info()
print(df.describe(include="all"))
print(df.isna().sum().sort_values(ascending=False))
print("Doublons :", df.duplicated().sum())
print(df["target"].value_counts(normalize=True))
~~~

Point essentiel : une EDA descriptive est possible avant la séparation, mais toute transformation apprenante ou sélection guidée par la cible doit respecter le protocole de validation.

Pour éviter d'orienter les décisions à partir du test, une bonne pratique consiste à réserver le test final tôt, puis à conduire l'analyse exploratoire approfondie et les choix de modélisation sur les données d'entraînement.

## 4. Étape 3 — Construire une stratégie de validation

Nous devons choisir un découpage cohérent avec la manière dont le modèle sera utilisé.

Pour des observations indépendantes, nous pouvons utiliser un split stratifié.

Pour des données temporelles, nous devons respecter la chronologie.

Pour des données groupées, nous devons examiner les dépendances entre observations.
La séparation recommandée
Train, Cross-Validation et Test final
Train + Cross-Validation

Test final

Le test final reste isolé pendant la sélection du modèle et le réglage des hyperparamètres.

Le train sert à ajuster les modèles.

La cross-validation permet de comparer les configurations.

Le test final sert à estimer la performance après toutes les décisions de sélection.

## 5. Étape 4 — Construire une baseline

Une erreur fréquente consiste à commencer immédiatement avec un modèle complexe.

Avant de tester Random Forest, XGBoost ou un réseau de neurones, nous devons disposer d'un point de comparaison simple.

C'est le rôle de la baseline.

### 5.1 DummyClassifier

DummyClassifier permet de construire une baseline qui ignore les features et applique une stratégie simple.

~~~python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
# Pour comparer avant sélection, utiliser les mêmes folds que les autres modèles.
# L'accès au test ci-dessous est réservé à l'évaluation finale.
predictions = baseline.predict(X_test)
~~~

Cette baseline prédit toujours la classe majoritaire.

Elle permet notamment de vérifier si un modèle plus sophistiqué apporte réellement de la valeur.

Pour une comparaison rigoureuse, il est préférable de comparer les modèles sur les mêmes folds de validation, puis de réserver le test pour la fin.

### 5.2 Pourquoi la baseline est importante

Supposons un dataset contenant 95 % de clients non défaillants.

Un modèle prédisant systématiquement « non-défaillant » atteint déjà 95 % d'accuracy.

Un modèle complexe obtenant 96 % d'accuracy n'est donc pas nécessairement satisfaisant.

Il faut examiner d'autres métriques, comme le rappel, la précision, la ROC-AUC ou la PR-AUC.

## 6. Étape 5 — Construire un Pipeline ML

Un pipeline permet de regrouper les étapes de preprocessing et de modélisation.

Cela réduit les risques de fuite de données et facilite la reproductibilité.
Exemple : données numériques et catégorielles

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

numeric_features = ["age", "income", "debt_ratio"]
categorical_features = ["region", "employment_type"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])
mixed_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])
# Nécessite un DataFrame X_train_mixed contenant exactement ces colonnes.
# mixed_model.fit(X_train_mixed, y_train_mixed)
~~~

Ce pipeline est un exemple pour un DataFrame contenant les colonnes indiquées.

Lorsqu'on appelle fit(), les transformations apprennent leurs paramètres sur les données d'entraînement.

Lorsqu'on appelle predict(), les mêmes transformations sont appliquées aux nouvelles observations.

## 7. Étape 6 — Comparer plusieurs modèles

Une approche professionnelle consiste à comparer plusieurs algorithmes sous le même protocole de validation.

Par exemple :
- Logistic Regression.
- Decision Tree.
- Random Forest.
- Gradient Boosting.
Le but n'est pas de choisir automatiquement le modèle le plus complexe.

Il faut comparer les performances, la stabilité, l'interprétabilité, le coût de calcul et les contraintes opérationnelles.
Exemple de comparaison
[[modelcomparison]]

Scores fictifs. Le meilleur modèle dépend des données et de la métrique choisie.

## 8. Étape 7 — Optimiser les hyperparamètres

Après avoir identifié quelques modèles prometteurs, nous pouvons effectuer une recherche d'hyperparamètres.

Comme nous l'avons vu au cours 7, les principales approches sont :
Méthode	Usage
GridSearchCV	Petite grille bien définie
RandomizedSearchCV	Recherche large avec budget limité
Optuna	Recherche flexible et adaptative

La recherche doit être réalisée uniquement sur le train, en utilisant la cross-validation.

Le test final reste isolé.

## 9. Étape 8 — Évaluation finale et analyse des erreurs

Une fois le meilleur modèle sélectionné, nous le réentraînons sur les données d'entraînement disponibles, puis nous évaluons ses performances sur le test final.

### 9.1 Classification

Les métriques courantes comprennent :
- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC / Average Precision.
- Log Loss.
- Brier Score.

### 9.2 Régression

Les métriques courantes comprennent :
\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

\[
RMSE=\sqrt{MSE}
\]

### 9.3 Analyse des erreurs

Il faut également comprendre où et pourquoi le modèle se trompe.

Exemples :
- Le modèle est-il moins performant sur certains segments ?
- Certaines observations présentent-elles des erreurs très importantes ?
- Le modèle manque-t-il les événements rares ?
- Les probabilités prédites sont-elles bien calibrées ?
Une bonne analyse des erreurs est souvent plus utile qu'une simple augmentation du score global.

## 10. Coding Lab — Un workflow ML complet

Nous allons maintenant appliquer les étapes essentielles sur un dataset réel disponible dans scikit-learn.

Nous utiliserons le dataset Breast Cancer Wisconsin, avec une classification binaire.

L'objectif pédagogique est d'apprendre à organiser une expérience ML complète, et non de construire un système de diagnostic médical.

~~~python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
}
for name, model in models.items():
    results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"auc": "roc_auc", "ap": "average_precision"},
        n_jobs=1
    )
    print(name, "AUC CV :", results["test_auc"].mean(),
          "AP CV :", results["test_ap"].mean())
# Choix didactique annoncé avant l'expérience : approfondir la logistique.
search = GridSearchCV(
    models["Logistic"], {"logisticregression__C": [0.01, 0.1, 1, 10]},
    cv=cv, scoring="roc_auc", refit=True, n_jobs=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
predictions = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, list(best_model.classes_).index(1)]
print("Test final AUC :", roc_auc_score(y_test, proba))
print("Test final AP :", average_precision_score(y_test, proba))
print(classification_report(y_test, predictions, labels=[0, 1],
      target_names=["malignant", "benign"], zero_division=0))
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
# La classe positive est 1, donc benign dans ce dataset pédagogique.
~~~

Remarque méthodologique importante
Ce code compare d'abord plusieurs familles de modèles, puis optimise spécifiquement la régression logistique pour illustrer le tuning.

Dans un projet réel, si la comparaison initiale conduit à sélectionner Random Forest ou un autre algorithme, il faut poursuivre avec le modèle choisi selon le protocole défini.

Il ne faut pas sélectionner un modèle en fonction du test final.

## 11. Reproductibilité — Une exigence professionnelle

Un projet ML doit pouvoir être reproduit.

Cela signifie que quelqu'un d'autre doit pouvoir retrouver les résultats dans des conditions comparables.
Les éléments à enregistrer
Élément	Pourquoi ?
Version du dataset	Identifier les données utilisées
Seed	Reproduire les opérations aléatoires
Version Python	Compatibilité
Versions des bibliothèques	Éviter les différences d'implémentation
Hyperparamètres	Reproduire la configuration
Stratégie de validation	Comprendre les scores
Métriques	Comparer les résultats
Artefact du modèle	Réutiliser le modèle entraîné

Exemple de sauvegarde

~~~python
import joblib
# Crée une archive locale uniquement si tu choisis d'exécuter ces lignes.
joblib.dump(best_model, "model.joblib")
loaded_model = joblib.load("model.joblib")
# Charger uniquement des artefacts fiables : pickle/joblib peut exécuter du code.
~~~

Il faut charger uniquement des artefacts provenant de sources fiables, car les formats basés sur pickle, dont joblib, peuvent exécuter du code lors du chargement.

## 12. Structure recommandée d'un projet ML

~~~text
ml-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
~~~

Un notebook est utile pour l'exploration.

Mais pour un projet professionnel, il est préférable de placer le code réutilisable dans des modules Python structurés.

## 13. Projet final du chapitre — Classification tabulaire de A à Z

CAPSTONE PROJECT · CHAPITRE 01
End-to-End ML Classification

Projet final

Tu vas construire un projet complet de classification sur le dataset Wine de scikit-learn, avec trois classes et 13 variables numériques.

Le projet doit inclure une EDA, une baseline, plusieurs modèles, une validation croisée, une optimisation des hyperparamètres et une évaluation finale.

Livrables attendus
- Un notebook d'exploration.
- Un script d'entraînement reproductible.
- Un tableau comparatif des modèles.
- Une analyse des erreurs.
- Un README expliquant la méthodologie et les résultats.

Code de départ

~~~python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data = load_wine(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# TODO : EDA sur train, baseline et plusieurs familles de modèles.
# TODO : mêmes folds StratifiedKFold, accuracy et F1 macro.
# TODO : tuning avec budget fixé et test tenu à l'écart.
# TODO : refit, une évaluation finale et analyse des erreurs.
# TODO : README, versions, données, protocole et limites.
~~~

Grille d'évaluation du projet
Critère	Points
Compréhension du problème et EDA	15
Stratégie de validation	15
Pipeline sans fuite	20
Baseline et comparaison des modèles	15
Hyperparameter tuning	15
Évaluation et analyse des erreurs	10
Reproductibilité et README	10
Total	100

## 14. Mini-évaluation finale

[[quiz]]

## 15. Questions d'entretien — Synthèse du chapitre

[[interview]]

## 16. Validation du cours

[[assessment]]

Chapitre 1 — Les huit supports sont disponibles
Il reste le projet pratique de synthèse pour valider tes acquis.

Ce que tu as étudié : les fondamentaux du ML, l'entraînement, les fonctions de perte, la généralisation, les stratégies de validation, les fuites de données, l'optimisation des hyperparamètres et la méthodologie complète d'un projet.

La prochaine grande étape sera le Chapitre 2 — Data Preprocessing & Feature Engineering, où nous approfondirons le traitement des données manquantes, les variables catégorielles, les outliers, la sélection de variables et les techniques de création de features.

Le point le plus important de ce chapitre est qu'un bon Machine Learning Engineer ne se contente pas d'entraîner des algorithmes : il construit des expériences fiables, comprend les limites de ses résultats et sait justifier chacune de ses décisions.

[[capstone]]