Dans le cours précédent, nous avons appris à identifier les types de variables, analyser les distributions et détecter les problèmes de qualité des données.

Aujourd'hui, nous allons approfondir un problème rencontré dans presque tous les projets de Machine Learning : les valeurs manquantes.

L'objectif n'est pas simplement de remplacer les NaN par une moyenne. Il faut comprendre pourquoi les données sont manquantes, quelles informations cette absence peut contenir et comment choisir une méthode de traitement adaptée.


## 1. Pourquoi les valeurs manquantes sont-elles importantes ?

Imaginons un dataset bancaire contenant les informations suivantes :

Client	Revenu	Ratio d'endettement	Ancienneté	Défaut
C001	3 200	0,35	5	0
C002	—	0,61	2	1
C003	5 100	0,22	—	0
C004	2 800	—	1	1
C005	4 200	0,40	7	0

Les tirets représentent des valeurs manquantes.

Plusieurs questions apparaissent :
- Pourquoi le revenu de C002 est-il absent ?
- Est-ce une erreur technique ou une information que le client n'a pas déclarée ?
- Les clients sans revenu renseigné ont-ils un risque différent ?
- Quelle méthode d'imputation préservera le mieux les informations utiles ?
Ces questions sont essentielles, car les données manquantes peuvent modifier les distributions, les relations entre variables et les performances des modèles.



## 2. Les trois mécanismes de données manquantes : MCAR, MAR et MNAR

C'est une distinction théorique fondamentale, souvent demandée en entretien Data Scientist.

Notons :
- \(X\) : les données complètes, y compris les valeurs non observées.
- \(X_{\mathrm{obs}}\) : les valeurs observées.
- \(X_{\mathrm{mis}}\) : les valeurs manquantes.
- \(R\) : un indicateur de présence, avec \(R=1\) lorsque la valeur est observée.

### 2.1 MCAR — Missing Completely At Random

Les données sont manquantes complètement au hasard.

La probabilité qu'une valeur soit absente ne dépend ni des valeurs observées ni des valeurs manquantes.
\[
P(R\mid X)=P(R)
\]

Exemple : un problème technique aléatoire fait perdre 5 % des mesures d'un capteur, indépendamment de leur valeur.

### 2.2 MAR — Missing At Random

Le mécanisme de données manquantes peut dépendre des variables observées, mais pas des valeurs manquantes elles-mêmes une fois les variables observées prises en compte.
\[
P(R\mid X_{\mathrm{obs}},X_{\mathrm{mis}})
=
P(R\mid X_{\mathrm{obs}})
\]

Exemple : le revenu est moins souvent renseigné par certaines catégories professionnelles, et cette catégorie est connue dans le dataset.

Le revenu manquant dépend alors potentiellement d'une variable observée : la catégorie professionnelle.

### 2.3 MNAR — Missing Not At Random

Le mécanisme de données manquantes dépend encore de la valeur non observée, même après prise en compte des informations disponibles.

Exemple : les personnes ayant des revenus très élevés choisissent plus fréquemment de ne pas déclarer leur revenu, même à caractéristiques observées comparables.

Dans ce cas, l'absence du revenu est directement liée à sa valeur réelle.
[[mechanisms]]

Comment les distinguer en pratique ?

Mécanisme	Dépendance	Difficulté
MCAR	Indépendante des données	Hypothèse forte
MAR	Dépend des variables observées	Plausible selon le contexte
MNAR	Dépend aussi des valeurs non observées	Difficile à identifier

Il est important de comprendre qu'on ne peut généralement pas prouver MAR ou exclure MNAR uniquement à partir des données observées.

Le contexte métier, la manière dont les données ont été collectées et les analyses de sensibilité jouent un rôle essentiel.



## 3. Détecter et quantifier les valeurs manquantes

Avant toute imputation, nous devons effectuer un diagnostic.

### 3.1 Nombre de valeurs manquantes


~~~python
df.isna().sum()
~~~





### 3.2 Pourcentage de valeurs manquantes


~~~python
missing_rate = df.isna().mean() * 100
print(missing_rate.sort_values(ascending=False))
~~~




La formule est :
\[
MissingRate_j =
\frac{N_{\mathrm{missing},j}}{N}\times100
\]


### 3.3 Analyser les lignes incomplètes


~~~python
# Nombre de lignes ayant au moins une valeur manquante
print(df.isna().any(axis=1).sum())
# Nombre de valeurs manquantes par ligne
print(df.isna().sum(axis=1).value_counts())
~~~





### 3.4 Étudier les patterns de missingness


~~~python
missing_pattern = df.isna().astype(int)
print(missing_pattern.corr())
~~~




Cela permet d'explorer si les absences de certaines colonnes sont associées.

Une corrélation entre indicateurs de missingness ne prouve toutefois pas un mécanisme MAR ou MNAR.



## 4. Première stratégie — Supprimer les valeurs manquantes

Deux opérations courantes existent.
Supprimer les lignes incomplètes

~~~python
df_clean = df.dropna()
~~~




Supprimer certaines colonnes

~~~python
df_clean = df.drop(columns=["variable_trop_incomplete"])
~~~




Avantages
La suppression est simple et ne nécessite pas de prédire les valeurs absentes.
Limites
Elle peut réduire fortement la taille du dataset, modifier la distribution des données et introduire un biais de sélection.

En particulier, la suppression complète des lignes n'est pas automatiquement appropriée lorsque les absences sont MAR ou MNAR.

Important : il n'existe pas de seuil universel, comme 30 % ou 50 % de valeurs manquantes, imposant la suppression d'une colonne.

La décision dépend du volume de données, de l'importance de la variable et du mécanisme de missingness.



## 5. Deuxième stratégie — Imputation simple

L'imputation consiste à remplacer une valeur manquante par une valeur estimée ou définie.

### 5.1 Mean Imputation

On remplace les valeurs manquantes par la moyenne des valeurs observées.
\[
\hat x_{\mathrm{missing}}=\bar{x}_{\mathrm{observed}}
\]


### 5.2 Median Imputation

On remplace les valeurs manquantes par la médiane.

Cette approche est souvent plus robuste lorsque la distribution est asymétrique ou contient des valeurs extrêmes.

### 5.3 Most Frequent Imputation

On utilise la modalité la plus fréquente.

Elle convient notamment à certaines variables catégorielles.

### 5.4 Constant Imputation

On utilise une valeur constante, par exemple "Unknown" pour une catégorie manquante.

Mais attention : une valeur absente ne signifie pas nécessairement une nouvelle catégorie métier. Le choix doit être justifié.



## 6. Explorateur interactif — Quelle imputation choisir ?

[[imputation]]

Implémentation avec SimpleImputer

~~~python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
~~~




Pourquoi fit_transform() sur le train et seulement transform() sur le test ?

Parce que la médiane doit être calculée uniquement sur le train. Sinon, nous introduisons une contamination train-test.



## 7. Les limites de l'imputation simple

L'imputation par moyenne ou médiane est rapide et souvent utile comme baseline.

Mais elle possède plusieurs limites.
Réduction artificielle de la dispersion
Supposons une variable contenant :
\[
X=[10,12,14,16,18]
\]

Si nous remplaçons plusieurs valeurs manquantes par la moyenne 14, nous augmentons artificiellement le nombre d'observations proches du centre.

La variance empirique peut alors diminuer.
Modification des relations entre variables
Une imputation indépendante colonne par colonne ne tient pas compte des relations entre les variables.

Par exemple, le revenu peut être lié à l'ancienneté professionnelle, au métier et à l'âge.

Une imputation par moyenne ignore ces relations.
Incertitude non représentée
L'imputation simple attribue une valeur unique aux observations manquantes.

Elle ne reflète pas correctement l'incertitude liée à ces valeurs, notamment lorsque l'objectif est une analyse statistique inférentielle.

C'est pourquoi nous allons étudier des méthodes plus élaborées.



## 8. KNNImputer — Imputation par les plus proches voisins

KNN signifie K-Nearest Neighbors.

L'idée consiste à estimer une valeur manquante à partir d'observations similaires.
Exemple
Imaginons un client dont le revenu est manquant.

Nous recherchons d'autres clients présentant des caractéristiques proches : âge, ancienneté et ratio d'endettement.

Le revenu manquant est alors estimé à partir des revenus disponibles chez les voisins retenus.
Distance euclidienne
Pour deux observations complètes :
\[
d(x_i,x_j)=
\sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}
\]

Avec des valeurs manquantes, scikit-learn utilise par défaut une distance adaptée, appelée nan_euclidean, qui tient compte des dimensions observées conjointement.
Les distances exploitent les dimensions observées conjointement ; un schéma en deux axes reste une illustration.

Implémentation

~~~python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
# Mettre les variables sur des échelles comparables avant de calculer les distances.
# Apprendre cette mise à l’échelle sur train uniquement, dans le pipeline.
~~~




weights="distance" donne davantage de poids aux voisins les plus proches.
Avantages
KNNImputer peut exploiter les relations locales entre observations et produire des estimations plus adaptées qu'une moyenne globale.
Limites
Il peut être coûteux sur de grands datasets, sensible aux échelles des variables et moins fiable lorsque les voisins comparables sont rares.

Le choix de n_neighbors est un hyperparamètre à valider.



## 9. IterativeImputer — Imputation multivariée

IterativeImputer utilise les autres variables pour estimer les valeurs manquantes d'une colonne.

Le principe est itératif.
1. Initialiser les valeurs manquantes.
2. Choisir une variable incomplète.
3. Entraîner un modèle pour prédire cette variable à partir des autres.
4. Remplacer ses valeurs manquantes par les prédictions.
5. Répéter le processus sur les variables incomplètes.
Exemple conceptuel
Nous voulons estimer le revenu :
\[
Income=f(Age,\ Seniority,\ DebtRatio,\ldots)
\]

Puis le ratio d'endettement :
\[
DebtRatio=g(Age,\ Income,\ Seniority,\ldots)
\]

Les estimations sont actualisées au fil des itérations.
Implémentation

~~~python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
~~~




L'import enable_iterative_imputer est nécessaire dans les versions de scikit-learn où cette fonctionnalité est encore expérimentale.
Points importants
- Cette méthode exploite les relations entre variables.
- Elle peut être plus coûteuse que SimpleImputer.
- Elle dépend des hypothèses et de la qualité des modèles d'imputation.
- Elle ne garantit pas de meilleures performances prédictives.
Par défaut, IterativeImputer produit une imputation déterministe. Avec une configuration adaptée, notamment sample_posterior=True et un estimateur compatible, il peut contribuer à générer plusieurs imputations.

Il faut distinguer cette approche d'une véritable analyse statistique par imputation multiple, qui implique également de combiner correctement les résultats et leur incertitude.



## 10. Missing Indicators — Quand l'absence devient une information

Parfois, le fait qu'une valeur soit manquante est lui-même prédictif.

Prenons un exemple bancaire : l'absence d'un bilan financier peut être liée au type d'entreprise, à sa taille ou à son mode de déclaration.

Nous pouvons créer un indicateur binaire :
\[
M_j=
\begin{cases}
1 & \text{si }X_j\text{ est manquant}\\
0 & \text{sinon}
\end{cases}
\]

Exemple

Revenu initial	Revenu imputé	income_missing
3 200	3 200	0
NaN	3 800	1
4 500	4 500	0
NaN	3 800	1

L'indicateur permet au modèle de distinguer une valeur réellement observée d'une valeur imputée.
Avec SimpleImputer

~~~python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median", add_indicator=True)
~~~




Les indicateurs sont ajoutés pour les variables qui présentaient des valeurs manquantes lors du fit().

Cela peut améliorer la prédiction lorsque le mécanisme d'absence apporte de l'information, mais ce n'est pas systématique.



## 11. Comparaison des méthodes


Méthode	Complexité	Exploite les autres variables ?	Limite principale
Suppression	Faible	Non	Perte de données et biais possible
Mean / Median	Faible	Non	Ignore les relations multivariées
Most Frequent	Faible	Non	Peut surreprésenter une modalité
KNNImputer	Moyenne à élevée	Oui	Coût et sensibilité aux distances
IterativeImputer	Élevée	Oui	Hypothèses et coût de calcul
Missing Indicators	Faible	Signale l'absence	N'est pas une imputation à lui seul

Il n'existe pas de meilleure méthode universelle. La bonne approche consiste à comparer plusieurs stratégies avec une validation adaptée.



## 12. Construire un pipeline sans Data Leakage

Nous avons vu au chapitre 1 que les transformations doivent être ajustées uniquement sur les données d'entraînement.

C'est particulièrement important pour l'imputation.
Exemple complet

~~~python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print("Mean ROC-AUC:", scores.mean())
print("Std ROC-AUC:", scores.std())
~~~




Le pipeline garantit que l'imputation est recalculée sur le train de chaque fold.

Cela évite que les observations de validation influencent les statistiques d'imputation.



## 13. Coding Lab — Comparer trois méthodes d'imputation

Nous allons construire un dataset synthétique contenant des valeurs manquantes, puis comparer :
- SimpleImputer avec médiane.
- KNNImputer.
- IterativeImputer.
L'objectif est de mesurer les performances d'un même classifieur selon la méthode utilisée.

~~~python
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=1200, n_features=10, n_informative=6, random_state=42)
X[rng.random(X.shape) < 0.15] = np.nan
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
imputers = {
    "Median": SimpleImputer(strategy="median"),
    "KNN": KNNImputer(n_neighbors=5, weights="distance"),
    "Iterative": IterativeImputer(max_iter=10, random_state=42),
}

if __name__ == "__main__":
    for name, imputer in imputers.items():
        model = Pipeline([
            # StandardScaler ignore les NaN au fit et les conserve au transform.
            # Mise à l'échelle AVANT KNN pour rendre les distances comparables.
            ("distance_scale", StandardScaler()), ("imputer", imputer),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        start = time.perf_counter()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        assert scores.shape == (5,) and np.isfinite(scores).all()
        print(name, "ROC-AUC:", scores, "mean:", scores.mean(),
              "std:", scores.std(), "seconds:", time.perf_counter() - start)
    print("Test final réservé, non utilisé pour comparer les imputations.")
~~~




Analyse attendue
Après exécution, réponds aux questions suivantes :
1. Quelle méthode obtient la meilleure ROC-AUC moyenne ?
2. Les différences sont-elles importantes ou très faibles ?
3. Quelle méthode est la plus coûteuse en calcul ?
4. La méthode la plus complexe est-elle nécessairement la meilleure ?
5. Quel serait l'intérêt d'ajouter des indicateurs de valeurs manquantes ?
Les résultats dépendent du dataset et de la simulation de missingness. Il ne faut pas généraliser une seule expérience.



## 14. Coding Challenge — Simuler MCAR et MAR

Challenge 02 — Missingness Simulator
+250 XP

Tu vas créer deux versions d'un dataset : une avec des valeurs manquantes complètement aléatoires, et une où la probabilité d'absence dépend d'une variable observée.

Ensuite, tu compareras les performances de plusieurs méthodes d'imputation.


~~~python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({"age": rng.integers(18, 70, n),
                   "income": rng.normal(4000, 1000, n),
                   "debt_ratio": rng.uniform(0.1, 0.9, n)})
df_mcar, df_mar = df.copy(), df.copy()
mask_mcar = rng.random(n) < 0.15
missing_probability = np.where(df["age"] < 30, 0.35, 0.05)
mask_mar = rng.random(n) < missing_probability
df_mcar.loc[mask_mcar, "income"] = np.nan
df_mar.loc[mask_mar, "income"] = np.nan

# TODO 1 : calculer les taux de valeurs manquantes.
# TODO 2 : comparer les distributions d'âge selon la présence du revenu.
# TODO 3 : réserver une validation et comparer médiane / KNN / Iterative.
# TODO 4 : utiliser les valeurs masquées connues pour mesurer la MAE.
# Pourquoi le MAR est-il connu ici et pas identifiable de cette façon en données réelles ?
~~~




Question de réflexion
Dans cette simulation, la probabilité d'absence dépend de l'âge, qui reste observé.

Pourquoi peut-on considérer ce mécanisme comme MAR dans le cadre du processus de génération défini ?

Et pourquoi une imputation exploitant l'âge pourrait-elle être plus adaptée qu'une médiane globale ?


[[solution]]




## 15. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 16. Mini-quiz interactif


[[quiz]]




## 17. Validation du cours


[[assessment]]

À retenir : traiter les valeurs manquantes ne consiste pas à choisir automatiquement entre moyenne, médiane et KNN. Il faut d'abord comprendre le mécanisme d'absence, préserver l'information pertinente et valider les méthodes sans fuite de données.

Le prochain cours sera consacré aux Outliers — Détection et traitement des valeurs atypiques, avec IQR, Z-score, Isolation Forest, Local Outlier Factor et les stratégies de traitement adaptées aux modèles ML.