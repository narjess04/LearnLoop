Nous arrivons au dernier cours du chapitre 2.

Jusqu'ici, nous avons appris à traiter les valeurs manquantes, détecter les outliers, standardiser les variables, encoder les catégories, créer et sélectionner des features, réduire la dimension et gérer les classes déséquilibrées.

Mais une question essentielle reste :

Comment réunir toutes ces opérations dans un système fiable, réutilisable et prêt pour la production ?

La réponse : les Machine Learning Pipelines.


## 1. Pourquoi avons-nous besoin des pipelines ?

Imaginons un projet de prédiction du défaut de paiement.

Notre dataset contient :

Variable	Type	Problème
age	Numérique	Valeurs manquantes
income	Numérique	Outliers
debt_ratio	Numérique	Échelles différentes
employment_type	Catégorielle	Encodage nécessaire
region	Catégorielle	Modalités inconnues possibles
default	Cible	Classe minoritaire

Sans pipeline, nous pourrions écrire :
~~~python
X = imputer.fit_transform(X)
X = scaler.fit_transform(X)
X = encoder.fit_transform(X)
model.fit(X, y)
# Illustration d’un enchaînement manuel : ne pas ajuster sur le futur test.
~~~




Ce code présente plusieurs difficultés :

- Les traitements numériques et catégoriels ne sont pas nécessairement les mêmes.
- Nous risquons d'apprendre des paramètres sur le test.
- Les transformations doivent être reproduites exactement en production.
- Les étapes deviennent difficiles à maintenir.
- La cross-validation peut être incorrecte si le preprocessing est effectué avant les folds.

Un pipeline permet d'organiser ces opérations en une seule chaîne cohérente.



## 2. Architecture d'un pipeline Machine Learning


[[architecture]]

Chaque transformation apprend uniquement sur train puis réutilise ses paramètres pour les nouvelles observations.




## 3. Comprendre fit, transform et predict

Avant de construire un pipeline, il faut comprendre trois opérations.
fit()
Apprend les paramètres à partir des données d'entraînement.

Exemples :

- StandardScaler.fit() apprend les moyennes et écarts-types.
- SimpleImputer.fit() apprend les valeurs d'imputation.
- PCA.fit() apprend les composantes principales.
- LogisticRegression.fit() apprend les coefficients.

transform()
Applique une transformation à des données.

Exemple :
\[
z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
\]

Les statistiques du train sont réutilisées sur le test.
fit_transform()
Combine l'apprentissage et la transformation.

Il est généralement utilisé pendant l'entraînement des transformeurs.
predict()
Produit les prédictions du modèle.
Exemple
~~~python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~




Il ne faut pas faire :
~~~python
# Incorrect pour une évaluation indépendante
X_test_scaled = scaler.fit_transform(X_test)
~~~




Cela apprend de nouvelles statistiques sur le test et rend le preprocessing incohérent avec celui de l'entraînement.



## 4. Premier pipeline simple

~~~python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]
~~~




Dans cet exemple, toutes les variables doivent être numériques ou déjà converties dans un format compatible.
Que fait pipeline.fit() ?
1. Ajuste l'imputer sur X_train.
2. Transforme X_train.
3. Ajuste le scaler sur les données imputées.
4. Standardise les données.



## 5. Entraîne la régression logistique.


[[pipeline]]




## 6. Pourquoi ColumnTransformer ?

Dans un dataset réel, toutes les colonnes ne nécessitent pas le même traitement.

Exemple :

Variables	Traitement
age, income, debt_ratio	Imputation médiane + StandardScaler
region, employment_type	Imputation mode + OneHotEncoder

ColumnTransformer permet d'appliquer plusieurs pipelines en parallèle à différents groupes de colonnes.
Architecture
\[
X=
\begin{bmatrix}
X_{\mathrm{num}} & X_{\mathrm{cat}}
\end{bmatrix}
\]

Après transformation :
\[
X'=
\begin{bmatrix}
T_{\mathrm{num}}(X_{\mathrm{num}}) &
T_{\mathrm{cat}}(X_{\mathrm{cat}})
\end{bmatrix}
\]

Les résultats sont concaténés pour former la matrice finale.



## 7. Construire un ColumnTransformer

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
numeric_features = ["age", "income", "debt_ratio"]
categorical_features = ["region", "employment_type"]
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                                  ("categorical", categorical_pipeline, categorical_features)])
~~~

Pourquoi handle_unknown="ignore" ?
Imaginons que le train contienne :

region = Tunis, Sousse, Sfax

Et que le test contienne :

region = Bizerte

Sans gestion des catégories inconnues, l'encodeur peut produire une erreur.

Avec handle_unknown="ignore", une catégorie inconnue est représentée par des zéros dans les colonnes One-Hot correspondant à cette feature.

Cela évite l'échec technique, mais ne garantit pas que la prédiction soit fiable pour cette nouvelle catégorie.



## 8. Le paramètre remainder

Par défaut :
~~~python
ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_features)], remainder="drop")
~~~




Les colonnes non spécifiées sont supprimées.

Avec :
~~~python
ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_features)], remainder="passthrough")
~~~




Les colonnes non sélectionnées sont conservées.

Attention : conserver automatiquement toutes les colonnes peut introduire des identifiants, des variables non pertinentes ou des données susceptibles de créer une fuite.

Pour un pipeline de production, il est souvent préférable de sélectionner explicitement les features autorisées.



## 9. Retrouver les noms des variables transformées

Après l'entraînement :
~~~python
feature_names = preprocessor.get_feature_names_out()
print(feature_names)
~~~




Exemples de noms :
numeric__age
numeric__income
numeric__debt_ratio
categorical__region_Tunis
categorical__region_Sfax
categorical__employment_type_CDI


Cette fonctionnalité est utile pour interpréter les coefficients ou les importances de variables.

Partie III — FunctionTransformer et transformations personnalisées



## 10. À quoi sert FunctionTransformer ?

Parfois, les transformations disponibles dans scikit-learn ne suffisent pas.

Nous pouvons avoir besoin de :

- Calculer le logarithme d'une variable.
- Convertir des unités.
- Appliquer une transformation mathématique.
- Réaliser un traitement métier déterministe.

FunctionTransformer permet d'intégrer certaines fonctions personnalisées dans un pipeline.
Exemple : transformation logarithmique
Pour une variable positive ou nulle :
\[
x'=\log(1+x)
\]

~~~python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")
~~~




Exemple avec un pipeline
~~~python
income_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("scaler", StandardScaler())])
~~~




Attention : log1p nécessite des valeurs strictement supérieures à -1. Il faut vérifier cette condition avant d'appliquer la transformation.
Limite de FunctionTransformer
Une fonction qui doit apprendre des paramètres pendant fit() ne se prête pas toujours à un simple FunctionTransformer.

Dans ce cas, il est préférable de créer un transformeur personnalisé.



## 11. Créer un transformeur personnalisé

Nous allons construire un transformeur qui plafonne les valeurs extrêmes selon les quantiles du train.
~~~python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileCapper(TransformerMixin, BaseEstimator):
    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Bornes de quantiles invalides")
        X = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        if not np.isfinite([self.lower_bounds_, self.upper_bounds_]).all():
            raise ValueError("Colonne entièrement manquante : imputer ou retirer explicitement")
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        X = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        if len(input_features) != self.n_features_in_:
            raise ValueError("Nombre de noms de features incompatible")
        return np.asarray(input_features, dtype=object)
~~~




Pourquoi apprendre les quantiles pendant fit() ?
Parce que les limites de plafonnement doivent être calculées uniquement sur les observations d'entraînement.

Les mêmes limites seront ensuite appliquées aux observations de validation, de test et de production.
Exemple d'intégration

~~~python
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("capper", QuantileCapper()), ("scaler", StandardScaler())])
~~~




Ici, les quantiles sont calculés après imputation. Si l'objectif est de calculer les quantiles uniquement sur les valeurs observées, l'ordre ou l'implémentation doit être adapté.

Partie IV — FeatureUnion et transformations parallèles



## 12. Qu'est-ce que FeatureUnion ?

FeatureUnion combine les sorties de plusieurs transformeurs appliqués au même ensemble de données.

Contrairement à ColumnTransformer, qui applique souvent des traitements différents à des colonnes différentes, FeatureUnion peut appliquer plusieurs représentations au même jeu de features.
Exemple
Nous voulons utiliser :

- Les variables standardisées originales.
- Une représentation PCA de ces mêmes variables.

Puis concaténer les deux représentations.
~~~python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features = FeatureUnion([("original", StandardScaler()),
                         ("pca", Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=3))]))])
X_train_features = features.fit_transform(X_train_numeric)
~~~




Si X_train_numeric contient 10 colonnes, la sortie contiendra 13 colonnes : les 10 originales standardisées et les 3 composantes PCA.

Cette stratégie peut être utile, mais elle ajoute des features potentiellement redondantes. Il faut donc vérifier qu'elle apporte une amélioration réelle.

Partie V — Pipelines et Cross-Validation



## 13. Pourquoi intégrer le preprocessing dans la validation ?

Imaginons un dataset contenant 1 000 observations.

Nous appliquons une cross-validation à cinq folds.

À chaque itération :

- 800 observations servent à entraîner.
- 200 observations servent à valider.

Si nous standardisons les 1 000 observations avant la cross-validation, les moyennes et écarts-types utilisent également les 200 observations de validation.

Nous avons donc une fuite d'information.
[[folds]]

Implémentation
~~~python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print("ROC-AUC moyenne :", scores.mean())
print("Écart-type :", scores.std())
~~~




La cross-validation évalue alors le processus complet : preprocessing et modèle.



## 14. GridSearchCV sur un pipeline

Un avantage majeur des pipelines est la possibilité d'optimiser les hyperparamètres des différentes étapes.

La syntaxe est :
nom_etape__nom_parametre


Exemple :
~~~python
param_grid = {"model__C": [.01, .1, 1, 10]}
~~~




Optimiser preprocessing et modèle ensemble
~~~python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
param_grid = {"preprocessor__numeric__imputer__strategy": ["mean", "median"],
              "model__C": [.01, .1, 1, 10]}
search = GridSearchCV(full_pipeline, param_grid, scoring="roc_auc",
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
~~~




Ce code suppose que full_pipeline contient une étape nommée preprocessor, avec un sous-transformeur nommé numeric contenant une étape imputer, et une étape finale nommée model.

La recherche choisit la configuration obtenant le meilleur score moyen de validation croisée.

Le test final doit rester indépendant.

Partie VI — Pipeline avec SMOTE



## 15. Intégrer le rééquilibrage dans le pipeline

Dans le cours précédent, nous avons étudié SMOTE.

Il existe une distinction importante :

- Un transformeur applique transform().
- Un sampler applique fit_resample() pendant l'entraînement.

Pour intégrer SMOTE, nous utilisons imblearn.pipeline.Pipeline.
Exemple
~~~python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
pipeline = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)),
                        ("model", LogisticRegression(max_iter=1000))])
~~~




SMOTE intervient pendant l'entraînement du pipeline.

Les observations de validation et de test ne sont pas rééchantillonnées lors de predict().
Attention aux données mixtes
SMOTE classique est principalement adapté aux variables numériques continues.

Avec des données catégorielles, une interpolation directe des colonnes One-Hot peut générer des combinaisons invalides.

Pour des données mixtes, il faut envisager des méthodes adaptées, comme SMOTENC, ou préférer d'abord une pondération des classes.

Partie VII — Sauvegarde et production



## 16. Pourquoi sauvegarder le pipeline complet ?

Supposons que nous entraînions un modèle avec :
1. Imputation.
2. Standardisation.
3. One-Hot Encoding.
4. Régression logistique.
Si nous sauvegardons uniquement la régression logistique, le modèle ne saura pas traiter correctement de nouvelles données brutes.

Il faut sauvegarder le pipeline complet.
Exemple avec joblib
~~~python
import joblib
joblib.dump(full_pipeline, "credit_risk_pipeline.joblib")
# Le module contenant les transformeurs personnalisés doit rester importable.
~~~




Chargement :
~~~python
pipeline_loaded = joblib.load("credit_risk_pipeline.joblib")
predictions = pipeline_loaded.predict(new_data)
~~~




Précaution de sécurité : joblib.load() peut exécuter du code lors de la désérialisation. Il ne faut charger que des fichiers provenant de sources fiables.
Autres précautions
Pour déployer un pipeline, nous devons également conserver :

- Les versions de Python et des bibliothèques.
- Le schéma des colonnes attendues.
- Les catégories et transformations apprises.
- La version du modèle.
- Les métriques de validation.
- Les informations nécessaires au suivi de la qualité des données.

Partie VIII — Projet final : Data Preparation Pipeline



## 17. Objectif du projet

Nous allons construire un pipeline de préparation des données capable de :
1. Gérer les valeurs manquantes.
2. Traiter les valeurs extrêmes.
3. Standardiser les variables numériques.
4. Encoder les variables catégorielles.
5. Entraîner un classifieur.
6. Évaluer les performances sans fuite de données.
7. Sauvegarder le pipeline pour l'inférence.
Nous utiliserons un dataset synthétique de risque de crédit.



## 18. Étape 1 — Générer le dataset

~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_credit_data(n=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, size=n).astype(float)
    income = rng.lognormal(mean=8.2, sigma=.65, size=n)
    debt_ratio = rng.beta(a=2, b=5, size=n)
    employment_type = rng.choice(["CDI", "CDD", "Independent"], size=n, p=[.60, .25, .15])
    region = rng.choice(["Tunis", "Sousse", "Sfax"], size=n)
    logits = -3 + 4 * debt_ratio - .0001 * (income - 3500) + .5 * (employment_type == "CDD")
    target = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    df = pd.DataFrame({"age": age, "income": income, "debt_ratio": debt_ratio,
                       "employment_type": employment_type, "region": region, "default": target})
    # Anomalies de mesure ajoutées après la génération du phénomène latent.
    for name in ["age", "income", "employment_type"]:
        idx = rng.choice(n, size=n // 12, replace=False)
        df.loc[idx, name] = np.nan
    df.loc[rng.choice(n, size=max(1, n // 100), replace=False), "income"] *= 12
    return df


df = make_credit_data()
~~~




Les données sont entièrement fictives. Les relations utilisées pour générer la cible servent uniquement à l'apprentissage technique.



## 19. Étape 2 — Séparer train et test

~~~python
X = df.drop(columns="default")
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
~~~




Le test final représente des données jamais utilisées pour ajuster le preprocessing ou choisir les hyperparamètres.



## 20. Étape 3 — Construire les traitements numériques

~~~python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ch02_cours10_transformers import QuantileCapper
numeric_features = ["age", "income", "debt_ratio"]
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("capper", QuantileCapper(lower=.01, upper=.99)),
                             ("scaler", StandardScaler())])
~~~




Nous utilisons ici la classe QuantileCapper définie dans la section 11.

Le traitement des outliers est un choix expérimental. Il faut comparer cette stratégie à une version sans plafonnement.



## 21. Étape 4 — Construire les traitements catégoriels

~~~python
from sklearn.preprocessing import OneHotEncoder
categorical_features = ["employment_type", "region"]
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore"))])
~~~







## 22. Étape 5 — Construire le pipeline complet

~~~python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
preprocessor = ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                                  ("categorical", categorical_pipeline, categorical_features)])
full_pipeline = Pipeline([("preprocessor", preprocessor),
                          ("model", LogisticRegression(class_weight="balanced", max_iter=2000))])
full_pipeline.fit(X_train, y_train)
print("Pipeline entraîné.")
~~~




Nous avons maintenant un pipeline qui accepte directement un DataFrame contenant les colonnes brutes.



## 23. Étape 6 — Évaluer les performances


À exécuter après l’étape 24 et après avoir figé les choix de preprocessing et de modèle. Le laboratoire complet respecte cet ordre.

~~~python
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss)
# Seulement après validation et choix figés (étape 24).
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Average Precision:", average_precision_score(y_test, y_proba))
print("Brier Score:", brier_score_loss(y_test, y_proba))
~~~




Pourquoi utiliser le Brier Score ?

Parce qu'il mesure la qualité des probabilités prédites :
\[
Brier=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2
\]

Pour un modèle de probabilité de défaut, cette information est essentielle.

L'utilisation de class_weight="balanced" peut affecter la calibration : il faut donc examiner les probabilités avec attention.



## 24. Étape 7 — Cross-validation du pipeline complet

~~~python
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(full_pipeline, X_train, y_train, cv=cv,
                        scoring={"roc_auc": "roc_auc", "ap": "average_precision", "brier": "neg_brier_score"})
print("ROC-AUC :", scores["test_roc_auc"].mean())
print("AP :", scores["test_ap"].mean())
print("Brier :", -scores["test_brier"].mean())
~~~




Ici, l'imputation, le plafonnement, le scaling, l'encodage et le modèle sont réentraînés sur chaque fold d'entraînement.

C'est précisément ce que nous recherchons.



## 25. Étape 8 — Sauvegarder le pipeline

~~~python
import joblib
joblib.dump(full_pipeline, "data_preparation_pipeline.joblib")
~~~




Le fichier contient les transformations ajustées et le modèle entraîné.
Exemple d'inférence
~~~python
new_client = pd.DataFrame([{"age": 35, "income": 2500, "debt_ratio": .65,
                            "employment_type": "CDI", "region": "Tunis"}])
probability = full_pipeline.predict_proba(new_client)[0, 1]
print("Probabilité prédite :", probability)
~~~




Cette probabilité est une sortie du modèle synthétique, et non une estimation de risque validée pour une utilisation bancaire réelle.



## Projet final — Scripts complets et challenge

Le laboratoire rassemble les étapes, compare avec et sans plafonnement et pondération sur les mêmes folds train, puis ouvre le test une seule fois après sélection. Les données sont synthétiques ; le prolongement du programme consiste à appliquer la méthode à un dataset réel documenté.

~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_credit_data(n=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, size=n).astype(float)
    income = rng.lognormal(mean=8.2, sigma=.65, size=n)
    debt_ratio = rng.beta(a=2, b=5, size=n)
    employment_type = rng.choice(["CDI", "CDD", "Independent"], size=n, p=[.60, .25, .15])
    region = rng.choice(["Tunis", "Sousse", "Sfax"], size=n)
    logits = -3 + 4 * debt_ratio - .0001 * (income - 3500) + .5 * (employment_type == "CDD")
    target = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    df = pd.DataFrame({"age": age, "income": income, "debt_ratio": debt_ratio,
                       "employment_type": employment_type, "region": region, "default": target})
    # Anomalies de mesure ajoutées après la génération du phénomène latent.
    for name in ["age", "income", "employment_type"]:
        idx = rng.choice(n, size=n // 12, replace=False)
        df.loc[idx, name] = np.nan
    df.loc[rng.choice(n, size=max(1, n // 100), replace=False), "income"] *= 12
    return df


df = make_credit_data()
X = df.drop(columns="default")
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from ch02_cours10_transformers import QuantileCapper

numeric_features = ["age", "income", "debt_ratio"]
categorical_features = ["employment_type", "region"]


def make_pipeline(capping=True, class_weight=None):
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("capper", QuantileCapper() if capping else "passthrough"),
                                  ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                                      ("categorical", categorical_pipeline, categorical_features)])
    return Pipeline([("preprocessor", preprocessor),
                     ("model", LogisticRegression(class_weight=class_weight, max_iter=2000))])


if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                                 average_precision_score, brier_score_loss)
    import argparse
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Chemin facultatif du pipeline joblib à sauvegarder")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "ap": "average_precision", "brier": "neg_brier_score"}
    candidates = {}
    for capping in [False, True]:
        for weight in [None, "balanced"]:
            pipeline = make_pipeline(capping, weight)
            result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            assert np.isfinite(result["test_ap"]).all()
            name = f"capping={capping}, class_weight={weight}"
            print(name, "ROC-AUC", result["test_roc_auc"].mean(), "AP", result["test_ap"].mean(),
                  "Brier", -result["test_brier"].mean())
            candidates[name] = (result["test_ap"].mean(), pipeline)
    name, (_, full_pipeline) = max(candidates.items(), key=lambda pair: pair[1][0])
    full_pipeline.fit(X_train, y_train)
    proba = full_pipeline.predict_proba(X_test)[:, 1]
    pred = full_pipeline.predict(X_test)
    print("Choix par AP CV:", name)
    print("Test final : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    print("Features :", full_pipeline.named_steps["preprocessor"].get_feature_names_out())
    new_client = pd.DataFrame([{"age": 35, "income": 2500, "debt_ratio": .65,
                                "employment_type": "CDI", "region": "Bizerte"}])
    print("Sortie synthétique pour un nouveau client:", full_pipeline.predict_proba(new_client)[0, 1])
    if args.save:
        joblib.dump(full_pipeline, args.save)
        print("Pipeline sauvegardé :", args.save)
        # Ne recharger que des fichiers fiables. Garder le module de transformeurs disponible.
        loaded = joblib.load(args.save)
        assert np.allclose(loaded.predict_proba(new_client), full_pipeline.predict_proba(new_client))
    print("Données fictives ; les sorties ne sont pas un score bancaire validé.")
~~~


### À toi de construire le pipeline de recherche

~~~python
from ch02_cours10_lab import X_train, X_test, y_train, y_test, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipeline = make_pipeline(capping=True)
grid = {"preprocessor__numeric__imputer__strategy": ["mean", "median"],
        "preprocessor__numeric__capper": ["passthrough"],
        "model__C": [.01, .1, 1, 10]}
# TODO : ajouter QuantileCapper() comme seconde configuration de capper.
# TODO : comparer class_weight=None et balanced dans la recherche sur train.
# TODO : optimiser par AP, examiner aussi ROC-AUC et Brier.
# TODO : figer les choix puis évaluer une fois sur test.
# TODO : sauvegarder le pipeline, les versions, le schéma et les métriques.
# BONUS : appliquer la démarche à un dataset réel et documenter sa provenance.
~~~


[[solution]]




## 26. Les erreurs à éviter absolument

Erreur	Conséquence	Bonne pratique
Imputer avant le split	Fuite d'information	Imputation dans le pipeline
Standardiser tout le dataset	Validation biaisée	Fit sur train uniquement
SMOTE avant cross-validation	Fuite et validation irréaliste	Sampler dans imblearn.Pipeline
Encoder train et test séparément	Colonnes incohérentes	Réutiliser le même encodeur
Optimiser sur le test	Surestimation de performance	Validation distincte
Sauvegarder uniquement le modèle	Inférence incohérente	Sauvegarder le pipeline complet
Ignorer les catégories inconnues	Erreurs en production	Stratégie explicite
Ignorer la calibration	Probabilités trompeuses	Vérifier Brier et calibration




## Questions d’entretien — Pipelines

[[interview]]




## 27. Quiz de fin de chapitre


[[quiz]]




## 28. Bilan du chapitre 2


Coche les compétences que tu maîtrises réellement.

[[assessment]]




## 29. Ce que tu dois maintenant savoir faire

À l'issue de ce chapitre, tu dois être capable de recevoir un dataset brut et de construire un système de préparation des données qui :

- Analyse la qualité des données.
- Traite les valeurs manquantes et les outliers.
- Encode et standardise les variables.
- Crée et sélectionne des features.
- Gère les éventuels déséquilibres de classes.
- Évite les fuites de données.
- Permet une validation reproductible.
- Peut être sauvegardé et réutilisé en production.

C'est une compétence centrale pour les postes de Data Scientist et Machine Learning Engineer.
Fin du chapitre 2 — Data Preprocessing & Feature Engineering

Tu as parcouru les dix cours. La prochaine étape est d'approfondir les algorithmes de Machine Learning supervisé.


Consulter la roadmap pour poursuivre avec le chapitre 3. 
À retenir : un pipeline n'est pas seulement un moyen de simplifier le code. C'est un élément essentiel pour rendre les expériences de Machine Learning cohérentes, reproductibles et fiables, de l'entraînement jusqu'à l'inférence.