Dans le cours précédent, nous avons appris à créer de nouvelles variables avec le Feature Engineering.

Mais imaginons que nous commencions avec 30 variables et que nous en créions 150 supplémentaires. Avons-nous réellement besoin des 180 variables pour entraîner notre modèle ?

Pas nécessairement.

Aujourd'hui, nous allons apprendre à identifier les variables utiles, redondantes ou peu informatives, afin de construire des modèles plus simples, plus rapides et parfois plus performants.


## 1. Qu'est-ce que le Feature Selection ?

Le Feature Selection consiste à sélectionner un sous-ensemble des variables disponibles sans modifier leur définition.

Supposons un dataset bancaire contenant :

Variable	Description
income	Revenu
annual_income	Revenu annuel
debt_ratio	Ratio d'endettement
age	Âge
customer_id	Identifiant client
risk_score	Score de risque
random_feature	Variable aléatoire

Certaines variables peuvent être redondantes. Par exemple, income et annual_income contiennent potentiellement la même information si l'une est exactement égale à douze fois l'autre.

D'autres peuvent ne pas avoir de valeur prédictive réelle, comme une variable aléatoire.
Pourquoi sélectionner les variables ?

- Réduire le risque de surapprentissage.
- Diminuer le temps d'entraînement.
- Améliorer l'interprétabilité.
- Réduire les coûts de collecte et de stockage.
- Supprimer certaines redondances.

Attention : supprimer des variables ne garantit pas une amélioration des performances. Une variable apparemment faible peut devenir utile en interaction avec d'autres.



## 2. Les trois grandes familles de Feature Selection

Filter

Critères statistiques


Wrapper

Évaluations répétées du modèle


Embedded

Sélection intégrée à l'apprentissage




Famille	Principe	Exemples
Filter	Sélection selon des statistiques	Variance, ANOVA, Mutual Information
Wrapper	Tester des sous-ensembles avec un modèle	RFE, RFECV
Embedded	Exploiter les mécanismes du modèle entraîné	Lasso, L1, SelectFromModel

Nous allons maintenant étudier chaque famille.




## 3. Filter Methods — Méthodes statistiques

Les Filter Methods évaluent les variables selon un critère statistique, sans entraîner un modèle prédictif complet pour chaque sous-ensemble.

Elles sont généralement rapides et constituent une bonne première approche.

### 3.1 VarianceThreshold

Une variable constante ne permet pas de distinguer les observations.

Exemple :

Client	Age	Pays
A	25	Tunisie
B	35	Tunisie
C	40	Tunisie
D	50	Tunisie

La colonne Pays est constante.

Sa variance est nulle :
\[
Var(X)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar x)^2=0
\]

Elle n'apporte donc aucune information discriminante dans cet échantillon.
Implémentation
~~~python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)
~~~




threshold=0.0 supprime les variables constantes.

Pour les variables non constantes mais de faible variance, la décision dépend de l'échelle. Une faible variance numérique n'implique pas nécessairement une faible utilité prédictive.

### 3.2 SelectKBest

SelectKBest sélectionne les \(k\) variables ayant les meilleurs scores selon un test choisi.
~~~python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
~~~




Ici, f_classif utilise un score F d'ANOVA pour mesurer l'association entre chaque variable numérique et une cible catégorielle.
Comprendre le score F
L'idée est de comparer la variation entre les groupes à la variation à l'intérieur des groupes.
\[
F=\frac{MS_{\text{between}}}{MS_{\text{within}}}
\]

Un score élevé indique que les moyennes des groupes sont relativement différentes par rapport à leur dispersion interne.

Mais ANOVA ne détecte pas toutes les relations non linéaires.

### 3.3 Mutual Information

La Mutual Information mesure la dépendance statistique entre deux variables.

Sa formule pour des variables discrètes est :
\[
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
\]

Si \(X\) et \(Y\) sont indépendantes :
\[
I(X;Y)=0
\]

Contrairement à une corrélation linéaire, la Mutual Information peut détecter certaines relations non linéaires.
Exemple conceptuel
Imaginons :
\[
Y=X^2
\]

Lorsque \(X\) est réparti symétriquement autour de zéro, la corrélation linéaire entre \(X\) et \(Y\) peut être nulle, malgré une relation déterministe.
Une relation non linéaire peut échapper à la corrélation
Illustration de Y = X². Une faible corrélation de Pearson ne signifie pas une absence de dépendance.
Implémentation
~~~python
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
~~~




L'estimation de Mutual Information peut être sensible à la taille de l'échantillon et à certains paramètres. Il faut aussi indiquer correctement quelles features sont discrètes lorsque cela est pertinent.




## 4. Explorer la sélection des variables


[[selection]]

Scores fictifs : un score individuel ne garantit pas un sous-ensemble optimal.




## 5. Le problème de la multicolinéarité

La multicolinéarité apparaît lorsque plusieurs variables explicatives présentent de fortes relations linéaires entre elles.

Exemple :
\[
AnnualIncome=12\times MonthlyIncome
\]

Ces deux variables sont parfaitement redondantes si cette relation est exacte.
Pourquoi est-ce problématique ?
Pour certains modèles linéaires, une forte multicolinéarité peut :

- Rendre les coefficients instables.
- Augmenter leur incertitude.
- Rendre leur interprétation difficile.

Elle ne détériore pas nécessairement les prédictions.

### 5.1 Corrélation de Pearson

\[
r_{XY}=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
\]

Le coefficient varie entre -1 et 1.

- Proche de 1 : forte relation linéaire positive.
- Proche de -1 : forte relation linéaire négative.
- Proche de 0 : faible relation linéaire.
~~~python
corr = X_train.corr(numeric_only=True)
print(corr)
~~~




On peut repérer les paires fortement corrélées, mais il n'existe pas de seuil universel imposant la suppression d'une variable.

### 5.2 VIF — Variance Inflation Factor

Le VIF mesure à quel point une variable peut être expliquée linéairement par les autres.
\[
VIF_j=\frac{1}{1-R_j^2}
\]

Où \(R_j^2\) est le coefficient de détermination obtenu en régressant la variable \(X_j\) sur les autres variables explicatives.
Exemple
Si :
\[
R_j^2=0.90
\]

Alors :
\[
VIF_j=\frac{1}{1-0.90}=10
\]

Un VIF élevé indique une forte dépendance linéaire avec les autres variables.

Les seuils 5 et 10 sont parfois utilisés comme repères, mais leur interprétation dépend du contexte.
Python
~~~python
import pandas as pd
import numpy as np
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Retirer les colonnes constantes ; imputer préalablement les absences.
X_vif = X_numeric.loc[:, X_numeric.nunique() > 1]
if not np.isfinite(X_vif.to_numpy()).all():
    raise ValueError("Le VIF exige des données numériques finies")
design = add_constant(X_vif, has_constant="add")
vif = pd.DataFrame({"feature": X_vif.columns,
                    "VIF": [variance_inflation_factor(design.to_numpy(), i + 1)
                            for i in range(X_vif.shape[1])]})
print(vif)
~~~




Pour une analyse rigoureuse, il faut gérer les valeurs manquantes, les variables constantes et la présence d'une constante dans le modèle auxiliaire.




## 6. Wrapper Methods — Sélection basée sur le modèle

Les Wrapper Methods évaluent des sous-ensembles de variables en utilisant un algorithme prédictif.

Elles sont souvent plus coûteuses que les Filter Methods.

### 6.1 RFE — Recursive Feature Elimination

RFE signifie élimination récursive de variables.

Son fonctionnement :
1. Entraîner un modèle sur toutes les variables.
2. Calculer une importance des variables.
3. Supprimer la ou les variables les moins importantes.
4. Réentraîner le modèle.
5. Répéter jusqu'à atteindre le nombre de variables souhaité.
[[rfe]]

Implémentation
~~~python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5, step=1)
selector.fit(X_train, y_train)
print("Variables retenues :", selector.support_)
print("Classement :", selector.ranking_)
~~~




L'estimateur doit fournir des coefficients ou des importances de variables exploitables.
Limite
RFE peut être coûteux, car il entraîne plusieurs fois le modèle.



## 7. RFECV — RFE avec Cross-Validation

Avec RFE, nous devons choisir le nombre de variables à conserver.

Mais comment déterminer ce nombre ?

RFECV combine l'élimination récursive et la cross-validation.

Il évalue différents nombres de variables et retient celui qui optimise la métrique choisie dans le cadre de cette validation.
Implémentation
~~~python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
selector = RFECV(LogisticRegression(max_iter=1000), step=1,
                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                 scoring="roc_auc", min_features_to_select=2)
selector.fit(X_train, y_train)
print("Nombre optimal :", selector.n_features_)
print("Variables retenues :", selector.support_)
~~~




Attention au Data Leakage
Si nous utilisons RFECV pour sélectionner les variables, puis évaluons le modèle sur les mêmes folds ayant servi à choisir ces variables, nous risquons d'obtenir une estimation optimiste.

Pour estimer correctement la performance finale du processus de sélection, nous pouvons utiliser une validation imbriquée (nested cross-validation) ou conserver un test indépendant.




## 8. Embedded Methods — Sélection intégrée au modèle

Ces méthodes utilisent les mécanismes internes d'apprentissage du modèle.

### 8.1 Régularisation L1 — Lasso

Nous avons étudié la régularisation au chapitre 1.

La régularisation L1 ajoute une pénalité basée sur la somme des valeurs absolues des coefficients :
\[
L(\beta)=L_{\mathrm{prediction}}(\beta)+\lambda\sum_{j=1}^{p}|\beta_j|
\]

Cette pénalité peut conduire certains coefficients à devenir exactement nuls.

Une variable dont le coefficient est nul n'intervient plus directement dans la prédiction du modèle linéaire.
Exemple avec LogisticRegression
~~~python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty="l1", solver="liblinear", C=.1, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print(model.coef_)
~~~




Dans scikit-learn, C contrôle l'inverse de la force de régularisation.

- C faible : régularisation plus forte.
- C élevé : régularisation plus faible.

[[lone]]


### 8.2 SelectFromModel

SelectFromModel permet de sélectionner les variables selon les importances ou les coefficients d'un modèle.
~~~python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42), threshold="median")
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
~~~




Ici, nous conservons les variables dont l'importance est au moins égale à la médiane des importances calculées.

Le choix du seuil doit être validé.



## 9. Feature Importance avec Random Forest

Les modèles à base d'arbres peuvent fournir des importances de variables.

Dans scikit-learn :
~~~python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
~~~




Ces importances reposent généralement sur la réduction moyenne d'impureté apportée par les variables.
Limites importantes

- Elles peuvent favoriser certaines variables ayant de nombreuses possibilités de séparation.
- Les variables corrélées peuvent se partager l'importance.
- Une forte importance ne démontre pas une relation causale.

Il est donc utile de compléter cette analyse par d'autres approches.



## 10. Permutation Importance

La Permutation Importance mesure l'effet de la perturbation d'une variable sur les performances d'un modèle déjà entraîné.
Principe
1. Évaluer le modèle sur un jeu de validation.
2. Mélanger aléatoirement les valeurs d'une variable.
3. Réévaluer le modèle.
4. Mesurer la différence de score.
La formule est :
\[
PI_j=S_{\mathrm{baseline}}-S_{\mathrm{permuted},j}
\]

Si le score diminue fortement après permutation, la variable était utile au modèle sur cet ensemble d'évaluation.
[[permutation]]

Implémentation
~~~python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=42, scoring="roc_auc")
importance_df = pd.DataFrame({"feature": X_valid.columns,
                              "importance_mean": result.importances_mean,
                              "importance_std": result.importances_std})
print(importance_df.sort_values("importance_mean", ascending=False))
~~~




Attention aux variables corrélées
Si deux variables portent presque la même information, mélanger l'une d'elles peut avoir peu d'effet, car le modèle continue d'utiliser l'autre.

Une faible Permutation Importance ne signifie donc pas toujours que la variable est inutile.




## 11. Comparaison générale des méthodes

Méthode	Famille	Coût	Détecte les interactions ?
VarianceThreshold	Filter	Très faible	Non
ANOVA F-test	Filter	Faible	Non, score univarié
Mutual Information	Filter	Modéré	Dépendances individuelles non linéaires, pas les interactions pures
RFE	Wrapper	Élevé	Selon le modèle
RFECV	Wrapper	Très élevé	Selon le modèle
L1	Embedded	Modéré	Pas sans features d'interaction
Random Forest Importance	Embedded	Modéré à élevé	Peut refléter des interactions
Permutation Importance	Analyse post-entraînement	Modéré	Selon le modèle entraîné

Il faut aussi distinguer Feature Selection et Feature Importance : calculer une importance ne revient pas automatiquement à sélectionner un ensemble optimal de variables.



## 12. Feature Selection et Data Leakage

La sélection de variables doit être effectuée à l'intérieur du pipeline de validation.
Mauvaise approche
~~~python
selector.fit(X, y)
X_selected = selector.transform(X)
# Mauvaise approche : la sélection a déjà vu les cibles des futurs folds de validation.
~~~




Ici, la sélection utilise toutes les cibles, y compris celles des futurs folds de validation.
Bonne approche
~~~python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([("scaler", StandardScaler()), ("selection", SelectKBest(f_classif, k=5)),
                     ("model", LogisticRegression(max_iter=1000))])
~~~




Lorsque ce pipeline est utilisé dans cross_val_score, le scaler et le sélecteur sont ajustés uniquement sur le train de chaque fold.

C'est une règle fondamentale.



## 13. Coding Lab — Comparer plusieurs méthodes de sélection

Nous allons créer un dataset avec 20 variables, dont seulement certaines sont informatives.

Nous comparerons :

- Sans Feature Selection.
- SelectKBest.
- RFE.
- Sélection par régularisation L1.
~~~python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                           n_redundant=6, n_repeated=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selectors = {"Sans sélection": "passthrough", "SelectKBest": SelectKBest(f_classif, k=6),
             "RFE": RFE(LogisticRegression(max_iter=1000), n_features_to_select=6),
             "L1": SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear",
                                                       C=.1, max_iter=1000, random_state=42))}

if __name__ == "__main__":
    for name, selector in selectors.items():
        pipeline = Pipeline([("scale", StandardScaler()), ("selector", selector),
                             ("model", LogisticRegression(max_iter=1000))])
        result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", return_estimator=True)
        assert np.isfinite(result["test_score"]).all()
        print(name, "ROC-AUC:", result["test_score"].mean(), "std:", result["test_score"].std(),
              "fit seconds:", result["fit_time"].mean())
        if name != "Sans sélection":
            masks = np.array([p.named_steps["selector"].get_support() for p in result["estimator"]])
            print("Fréquence de sélection par variable:", masks.mean(axis=0))
    print("Configurations fixes ; test final réservé. Stabilité ne signifie ni causalité ni utilité garantie.")
~~~




Questions d'analyse
1. Quelle méthode obtient la meilleure ROC-AUC moyenne ?
2. Est-ce que la sélection améliore systématiquement la performance ?
3. Pourquoi RFE est-il généralement plus coûteux que SelectKBest ?
4. Pourquoi certaines variables redondantes peuvent-elles rester sélectionnées ?
5. Comment vérifier que les variables retenues sont stables entre les folds ?
Ce laboratoire compare des configurations fixes. Pour optimiser le nombre de variables ou les hyperparamètres, il faut effectuer la recherche dans le protocole de validation, sans utiliser le test final.



## 14. Coding Challenge — Construire un pipeline de sélection

Challenge 07 — Feature Selection Pipeline

+300 XP

Construis un pipeline complet qui sélectionne les variables et optimise le nombre de features retenues.
~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from ch02_cours07_lab import X_train, X_test, y_train, y_test

pipeline = Pipeline([("scale", StandardScaler()), ("selector", SelectKBest(f_classif)),
                     ("model", LogisticRegression(max_iter=1000))])
search = GridSearchCV(pipeline, {"selector__k": [3, 5, 8, 12, 15], "model__C": [.01, .1, 1, 10]},
                      scoring="roc_auc", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
# TODO : fit sur train, afficher best_params_, best_score_ et le masque sélectionné.
# TODO : comparer avec une baseline sous le même protocole de sélection.
# TODO : figer le choix sur la CV, puis ouvrir une seule fois le test final.
# BONUS : calculer la fréquence de sélection des variables entre folds train.
~~~




Bonus — Stabilité des variables
Répète la sélection sur plusieurs folds et calcule combien de fois chaque variable est retenue.

Une variable sélectionnée dans 100 % des folds est généralement plus stable qu'une variable sélectionnée dans 20 % des folds, même si la stabilité seule ne prouve pas son utilité.



[[solution]]




## 15. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 16. Mini-quiz interactif


[[quiz]]




## 17. Validation du cours


[[assessment]]

À retenir : sélectionner des variables consiste à trouver un équilibre entre information prédictive, simplicité, stabilité et coût. La meilleure sélection n'est pas forcément celle qui conserve le moins de variables, mais celle qui généralise correctement.

Prochain cours : Réduction de dimension — PCA, SVD, t-SNE et UMAP. Nous verrons comment réduire le nombre de dimensions en construisant de nouvelles représentations, plutôt qu'en supprimant simplement des colonnes.