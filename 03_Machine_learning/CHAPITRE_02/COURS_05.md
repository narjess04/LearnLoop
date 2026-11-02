Nous avons déjà étudié les valeurs manquantes, les outliers et le feature scaling.

Aujourd'hui, nous allons répondre à une question essentielle : comment permettre à un modèle de Machine Learning d'utiliser des informations comme une ville, un secteur d'activité ou un niveau de risque ?

Ces informations sont souvent représentées par du texte, alors que la majorité des algorithmes classiques attendent des variables numériques.

Mais attention : transformer des catégories en nombres ne suffit pas. Le type d'encodage choisi peut introduire des relations artificielles ou provoquer du Data Leakage.


## 1. Comprendre les variables catégorielles

Prenons un dataset contenant les informations suivantes :

Client	Région	Niveau de risque	Revenu
C001	Tunis	Faible	3 500
C002	Sfax	Élevé	2 200
C003	Sousse	Moyen	4 100
C004	Tunis	Moyen	3 000

Nous distinguons deux types de variables catégorielles.
Variables nominales
Elles n'ont aucun ordre naturel.

Exemples : région, couleur, secteur d'activité, type de contrat.

Il serait incorrect d'affirmer que Tunis est supérieur à Sfax ou que Sousse est inférieur à Tunis.
Variables ordinales
Elles possèdent un ordre naturel.

Par exemple :
\[
Faible < Moyen < Élevé
\]

Cet ordre peut être représenté numériquement.

En revanche, cela ne signifie pas nécessairement que la distance entre « faible » et « moyen » est identique à celle entre « moyen » et « élevé ».



## 2. Label Encoding

Le Label Encoding consiste à attribuer un entier à chaque catégorie.

Par exemple :

Région	Encodage
Sfax	0
Sousse	1
Tunis	2

Implémentation
~~~python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(["Défaut", "Non-défaut", "Défaut"])
print(y_encoded)
print(encoder.classes_)
~~~




LabelEncoder est principalement destiné à encoder la variable cible y.

Pour les features catégorielles X, on préfère généralement OneHotEncoder ou OrdinalEncoder.
Pourquoi éviter les codes entiers arbitraires ?
Supposons :
\[
Sfax=0,\quad Sousse=1,\quad Tunis=2
\]

Un modèle pourrait interpréter une relation numérique entre ces régions.

Pour une régression linéaire, par exemple :
\[
\hat y=\beta_0+\beta_1 \times Région
\]

Cela impose une structure linéaire artificielle entre les trois catégories.

Conclusion : ne pas confondre une catégorie nominale avec une quantité numérique.



## 3. One-Hot Encoding

Le One-Hot Encoding est l'une des méthodes les plus importantes en Machine Learning.

Il consiste à créer une colonne binaire pour chaque catégorie.
Exemple
Transformation One-Hot

Région	Sfax	Sousse	Tunis
Tunis	0	0	1
Sfax	1	0	0
Sousse	0	1	0
Tunis	0	0	1


Chaque région devient une variable indicatrice.

Aucun ordre artificiel n'est imposé entre les régions.
Implémentation avec scikit-learn
~~~python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.DataFrame({"region": ["Tunis", "Sfax", "Sousse", "Tunis"]})
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(df[["region"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
print(encoded_df)
~~~




Le paramètre handle_unknown
Imaginons que le train contienne :

- Tunis
- Sfax
- Sousse

Mais que le test contienne une nouvelle catégorie :

- Nabeul

Sans configuration adaptée, l'encodeur peut générer une erreur.

Avec :
~~~python
OneHotEncoder(handle_unknown="ignore")
~~~




La catégorie inconnue est représentée par des zéros sur les colonnes apprises pour cette feature.

Cela permet au pipeline de fonctionner, mais il faut comprendre que le modèle ne dispose d'aucune information spécifique à cette nouvelle catégorie.



## 4. Le problème de la cardinalité

La cardinalité correspond au nombre de catégories distinctes d'une variable.

Par exemple :

Variable	Cardinalité
État du produit : neuf/occasion	2
Région	24
Secteur d'activité	100
Code produit	50 000

Si nous appliquons One-Hot Encoding à une colonne contenant 50 000 catégories, nous pouvons générer un très grand nombre de features.

Cela entraîne potentiellement :

- Une augmentation de la mémoire utilisée.
- Des calculs plus coûteux.
- Des catégories très rares.
- Un risque de surapprentissage pour certaines configurations.

Comment gérer une forte cardinalité ?
Plusieurs possibilités existent : regrouper les catégories rares, limiter les catégories retenues, utiliser un encodage par fréquence ou employer un Target Encoding avec précautions.

Scikit-learn permet aussi de regrouper des catégories peu fréquentes :
~~~python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
~~~




Ici, les catégories présentes moins de dix fois dans les données d'ajustement peuvent être regroupées.



## 5. Ordinal Encoding

L'Ordinal Encoding est adapté aux variables présentant un ordre naturel.

Exemple :

Risque	Encodage
Faible	0
Moyen	1
Élevé	2

Implémentation
~~~python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
df = pd.DataFrame({"risk_level": ["Faible", "Moyen", "Élevé", "Moyen"]})
encoder = OrdinalEncoder(categories=[["Faible", "Moyen", "Élevé"]],
                         handle_unknown="use_encoded_value", unknown_value=-1)
encoded = encoder.fit_transform(df[["risk_level"]])
print(encoded)
~~~




Le paramètre categories permet de définir explicitement l'ordre métier.
Attention à l'interprétation
Avec l'encodage :
\[
Faible=0,\quad Moyen=1,\quad Élevé=2
\]

Un modèle linéaire traite les écarts numériques comme réguliers.

Cela peut être acceptable dans certains cas, mais il faut vérifier que cette hypothèse convient.

Pour les modèles à base d'arbres, l'encodage ordinal peut être pratique, à condition que l'ordre soit pertinent.



## 6. Comparaison interactive — One-Hot versus Ordinal


[[encoding]]

L’encodage ordinal conserve l’ordre dans une colonne ; le One-Hot représente chaque catégorie indépendamment.




## 7. Target Encoding — Utiliser la variable cible

Nous arrivons à une méthode plus avancée.

Le Target Encoding consiste à remplacer chaque catégorie par une statistique calculée à partir de la cible.

En classification binaire, on utilise souvent la moyenne de y pour chaque catégorie.
Exemple
Supposons que nous souhaitions prédire le défaut de paiement.

Secteur	Clients	Défauts	Taux de défaut
Industrie	100	5	5 %
Commerce	100	12	12 %
Services	100	8	8 %

Le Target Encoding pourrait produire :
\[
Industrie \rightarrow 0.05
\]

\[
Commerce \rightarrow 0.12
\]

\[
Services \rightarrow 0.08
\]

L'avantage est que nous obtenons une seule colonne numérique, même si la variable contient de nombreuses catégories.
Formule simple
Pour une catégorie \(c\) :
\[
TE(c)=\frac{\sum_{i:X_i=c}y_i}{N_c}
\]

Où \(N_c\) représente le nombre d'observations de cette catégorie.
Le problème : surapprentissage
Imaginons une catégorie qui n'apparaît qu'une seule fois.

Si l'unique client est en défaut, son Target Encoding brut vaut :
\[
TE(c)=1
\]

Le modèle pourrait apprendre une relation trop spécifique à cette observation.

C'est pourquoi on utilise souvent un mécanisme de régularisation appelé smoothing.



## 8. Target Encoding avec smoothing

Une formule courante est :
\[
TE(c)=
\frac{N_c\bar y_c+m\bar y}{N_c+m}
\]

Avec :

- \(N_c\) : nombre d'observations de la catégorie.
- \(\bar y_c\) : moyenne de la cible dans la catégorie.
- \(\bar y\) : moyenne globale de la cible.
- \(m\) : force de régularisation.

Lorsque la catégorie est rare, son encodage est rapproché de la moyenne globale.

Lorsque la catégorie contient beaucoup d'observations, sa propre moyenne a davantage de poids.
[[smoothing]]

Les taux de catégorie (30 %) et global (8 %) sont fixés pour cette simulation mathématique.




## 9. Le danger majeur du Target Encoding : Data Leakage

Le Target Encoding est une méthode supervisée.

Contrairement au One-Hot Encoding, il utilise directement la variable cible y.

Il faut donc être particulièrement vigilant.
Mauvaise approche
~~~python
# Ne pas calculer le Target Encoding sur tout le dataset avant le split.
~~~




Cela transmettrait des informations du test au train.

Mais il existe également un risque plus subtil : la fuite à l'intérieur du train.

Si nous calculons le taux de défaut d'une catégorie en utilisant la cible de l'observation que nous encodons, nous pouvons créer une dépendance artificielle.
La solution : Cross-Fitting
Le principe consiste à diviser les données d'entraînement en plusieurs folds.

Pour chaque fold, nous calculons les statistiques d'encodage sur les autres folds.

Ensuite, nous appliquons ces statistiques au fold exclu.
[[crossfitting]]

Avec scikit-learn
~~~python
import pandas as pd
from sklearn.preprocessing import TargetEncoder
X = pd.DataFrame({"sector": ["Industrie", "Commerce", "Services"] * 4})
y = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
encoder = TargetEncoder(target_type="binary", smooth="auto", cv=3, random_state=42)
# fit_transform utilise le cross-fitting
X_encoded = encoder.fit_transform(X, y)
print(X_encoded)
~~~




Avec TargetEncoder de scikit-learn, fit_transform(X, y) utilise un mécanisme de cross-fitting.

Il ne faut pas supposer que fit(X, y).transform(X) est équivalent : cette seconde approche utilise des statistiques apprises sur l'ensemble de X et peut introduire une fuite interne.

Dans un Pipeline, l'appel à fit() du pipeline déclenche normalement fit_transform() sur les transformeurs intermédiaires, ce qui permet de bénéficier de ce comportement.

Le Target Encoding doit malgré tout être intégré à un protocole de validation rigoureux.



## 10. Frequency Encoding

Le Frequency Encoding consiste à remplacer une catégorie par sa fréquence d'apparition.
Exemple

Région	Nombre de clients	Fréquence
Tunis	500	0,50
Sfax	300	0,30
Sousse	200	0,20

La formule est :
\[
FE(c)=\frac{N_c}{N}
\]

Implémentation
~~~python
frequencies = X_train["region"].value_counts(normalize=True)
X_train_encoded = X_train["region"].map(frequencies)
X_test_encoded = X_test["region"].map(frequencies).fillna(0)
~~~




Avantages

- Une seule colonne numérique.
- Adapté aux variables de forte cardinalité.
- N'utilise pas directement la cible.

Limites
Deux catégories ayant la même fréquence obtiennent le même encodage.

La fréquence ne représente pas nécessairement une relation avec la cible.

Elle doit être calculée sur le train uniquement, et les catégories inconnues doivent être gérées explicitement.



## 11. Comment choisir la bonne méthode ?

Situation	Méthode souvent pertinente
Variable nominale, faible cardinalité	One-Hot Encoding
Variable ordinale	Ordinal Encoding
Cible catégorielle	LabelEncoder, si nécessaire
Variable nominale, forte cardinalité	One-Hot regroupé, Frequency ou Target Encoding
Catégories rares	Regroupement des modalités
Modèle linéaire	Souvent One-Hot pour les catégories nominales
Modèle à base d'arbres	Ordinal selon le sens des catégories, One-Hot ou autres méthodes adaptées

Il n'existe pas de méthode universellement meilleure.

La sélection doit tenir compte de la cardinalité, de l'algorithme, du risque de fuite, du coût de calcul et des performances en validation.



## 12. Pipeline complet avec variables numériques et catégorielles

Nous allons combiner plusieurs notions des cours précédents :

- Imputation des valeurs manquantes.
- Standardisation des variables numériques.
- One-Hot Encoding des catégories nominales.
- Ordinal Encoding des catégories ordonnées.
- Entraînement d'une régression logistique.
~~~python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
numeric_features = ["age", "income", "debt_ratio"]
nominal_features = ["region", "employment_type"]
ordinal_features = ["risk_level"]
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
nominal_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                             ("encoder", OneHotEncoder(handle_unknown="ignore"))])
ordinal_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", add_indicator=False)),
                             ("encoder", OrdinalEncoder(categories=[["Faible", "Moyen", "Élevé"]],
                                                        handle_unknown="use_encoded_value", unknown_value=-1))])
preprocess = ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                                ("nominal", nominal_pipeline, nominal_features),
                                ("ordinal", ordinal_pipeline, ordinal_features)])
pipeline = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
~~~




Cet exemple suppose que X_train et X_test sont des DataFrames contenant les colonnes indiquées.

Remarque : ici les absences ordinales sont imputées par la modalité la plus fréquente. Ajouter un niveau « Unknown » dans un ordre métier peut créer une relation artificielle ; un indicateur d’absence séparé est une autre option.



## 13. Coding Lab — Comparer One-Hot et Ordinal Encoding

Nous allons créer un dataset synthétique de classification contenant des variables numériques et catégorielles.

Puis nous comparerons deux stratégies d'encodage sur le même modèle.
~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
n = 1200
df = pd.DataFrame({
    "age": rng.integers(18, 70, n), "income": rng.normal(4000, 1200, n),
    "region": rng.choice(["Tunis", "Sfax", "Sousse", "Nabeul"], n),
    "risk_level": rng.choice(["Faible", "Moyen", "Élevé"], n),
})
risk_effect = df["risk_level"].map({"Faible": -1.2, "Moyen": 0., "Élevé": 1.2})
region_effect = df["region"].map({"Tunis": .4, "Sfax": -.5, "Sousse": .8, "Nabeul": -.3})
logits = -1 + risk_effect + region_effect - .0002 * (df["income"] - 4000)
y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for strategy in ["One-Hot", "Ordinal risk"]:
        transformers = [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), ["age", "income"]),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
            ("risk", OneHotEncoder(handle_unknown="ignore") if strategy == "One-Hot"
             else OrdinalEncoder(categories=[["Faible", "Moyen", "Élevé"]],
                                 handle_unknown="use_encoded_value", unknown_value=-1),
             ["risk_level"]),
        ]
        pipeline = Pipeline([("preprocess", ColumnTransformer(transformers)),
                             ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(strategy, scores, "mean:", scores.mean(), "std:", scores.std())
    print("La région reste nominale dans les deux stratégies. Test final réservé.")
~~~




Analyse attendue
Compare les performances des deux stratégies.

Observe notamment que le One-Hot Encoding n'impose aucune relation numérique entre les niveaux de risque, alors que l'Ordinal Encoding suppose une progression régulière des codes.

Les résultats peuvent être proches ou différents selon la structure des données.



## 14. Coding Challenge — Target Encoding sans fuite

Challenge 05 — Leakage-Free Encoding
+300 XP

Construis un pipeline permettant de comparer One-Hot Encoding et Target Encoding sur une variable catégorielle de forte cardinalité.
~~~python
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

onehot_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("model", LogisticRegression(max_iter=1000)),
])
target_pipeline = Pipeline([
    ("encoder", TargetEncoder(target_type="binary", cv=5, random_state=42)),
    ("model", LogisticRegression(max_iter=1000)),
])
# TODO 1 : générer une colonne category à forte cardinalité et une cible binaire.
# TODO 2 : réserver un test avant toute comparaison.
# TODO 3 : comparer les pipelines sur les mêmes folds train en ROC-AUC.
# TODO 4 : tester une régularisation plus forte et gérer les catégories inconnues.
# Ne pas remplacer fit_transform(X, y) par fit(X, y).transform(X).
~~~




Bonus
Ajoute des catégories très rares et observe leur impact sur le Target Encoding.

Teste également une régularisation plus forte et compare les résultats.


[[solution]]




## 15. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 16. Mini-quiz interactif


[[quiz]]




## 17. Validation du cours


[[assessment]]

À retenir : l'encodage est une décision de modélisation. Une mauvaise représentation des catégories peut introduire un ordre artificiel, augmenter inutilement la dimension ou provoquer une fuite de données.

Le prochain cours sera consacré au Feature Engineering — Créer des variables pertinentes à partir des données existantes, avec des ratios, des interactions, des variables temporelles et des transformations avancées.