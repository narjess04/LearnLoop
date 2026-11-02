Dans le cours précédent, nous avons appris à transformer les variables catégorielles grâce au One-Hot, Ordinal, Frequency et Target Encoding.

Aujourd'hui, nous passons à une compétence essentielle du Data Scientist : créer de nouvelles variables à partir des données existantes.

Un bon modèle ne dépend pas uniquement de l'algorithme choisi. La manière dont les informations sont représentées peut avoir un impact considérable sur ses performances.


## 1. Qu'est-ce que le Feature Engineering ?

Le Feature Engineering consiste à construire, transformer ou combiner des variables afin de mieux représenter un phénomène que nous souhaitons prédire.

Supposons que nous cherchions à prédire le défaut de paiement d'un client.

Nous disposons de :

- Son revenu mensuel.
- Ses charges mensuelles.
- Le montant de ses dettes.
- Son ancienneté professionnelle.

Ces variables sont utiles, mais elles ne représentent pas directement toutes les relations métier.

Nous pourrions créer :
\[
DebtToIncome=\frac{Dette}{Revenu}
\]

\[
DisposableIncome=Revenu-Charges
\]

\[
ExpenseRatio=\frac{Charges}{Revenu}
\]

Ces nouvelles variables peuvent aider le modèle à identifier des situations financières différentes.
Exemple

Client	Revenu	Charges	Taux de charges
A	3 000	1 000	33,3 %
B	6 000	4 000	66,7 %
C	2 500	1 000	40 %

Un revenu élevé ne signifie pas automatiquement une situation financière favorable. Le rapport entre les revenus et les charges apporte une information supplémentaire.



## 2. Les principales familles de Feature Engineering


1. Transformations mathématiques

Logarithmes, puissances, racines, ratios et différences.



2. Variables d'interaction

Combiner plusieurs features pour représenter des relations entre elles.



3. Variables temporelles

Extraire des informations de dates et construire des historiques.



4. Agrégations et regroupements

Moyennes, sommes, fréquences, statistiques par client ou groupe.



5. Discrétisation (Binning)

Transformer une variable continue en intervalles ou catégories.






## 3. Transformations mathématiques


### 3.1 Ratios

Un ratio permet de comparer deux quantités.

Exemple :
\[
DebtRatio=\frac{TotalDebt}{TotalAssets}
\]

En Python :
~~~python
df["debt_ratio"] = df["total_debt"] / df["total_assets"]
~~~




Mais attention : si le dénominateur vaut zéro, le résultat peut devenir infini ou indéfini.

Une version plus prudente :
~~~python
import numpy as np
# Masquer le dénominateur avant la division évite les divisions par zéro.
denominator = df["total_assets"].where(df["total_assets"] > 0)
df["debt_ratio"] = df["total_debt"].div(denominator)
~~~




Il faut définir les règles selon le sens métier. Un actif nul n'est pas nécessairement équivalent à une donnée manquante.

### 3.2 Différences

Les différences permettent de représenter des écarts.
\[
IncomeChange=Income_t-Income_{t-1}
\]

Exemple :
~~~python
df["income_change"] = df["income_current"] - df["income_previous"]
~~~





### 3.3 Taux de variation

\[
GrowthRate=
\frac{X_t-X_{t-1}}{X_{t-1}}
\]

Exemple :
~~~python
denominator = df["revenue_previous"].where(df["revenue_previous"] != 0)
df["revenue_growth"] = (df["revenue_current"] - df["revenue_previous"]).div(denominator)
~~~





### 3.4 Transformations logarithmiques

Pour certaines variables non négatives et très asymétriques :
\[
X'=\log(1+X)
\]

~~~python
df["income_log"] = np.log1p(df["income"])
~~~




Cette transformation réduit l'influence numérique des grandes valeurs, mais elle ne garantit pas une amélioration de performance.



## 4. Variables d'interaction

Une variable d'interaction représente l'effet combiné de plusieurs variables.

Prenons un modèle linéaire :
\[
\hat y=\beta_0+\beta_1x_1+\beta_2x_2
\]

Dans cette formulation, l'effet de \(x_1\) ne dépend pas de \(x_2\).

Nous pouvons ajouter une interaction :
\[
\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2
\]

Le terme \(x_1x_2\) permet au modèle de représenter une relation supplémentaire.
Exemple métier
Nous pouvons supposer que l'effet d'un ratio d'endettement dépend du revenu.
\[
Interaction=Income\times DebtRatio
\]

Exemple Python
~~~python
df["income_debt_interaction"] = df["income"] * df["debt_ratio"]
~~~




Avec PolynomialFeatures
Scikit-learn permet de créer automatiquement des interactions.
~~~python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
~~~




Pour deux variables \(x_1\) et \(x_2\), cette configuration produit :
\[
[x_1,\ x_2,\ x_1x_2]
\]

Avec interaction_only=False, elle peut également produire les carrés :
\[
[x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2]
\]

Attention à l'explosion dimensionnelle
Avec \(p\) variables et des polynômes de degré 2 incluant les carrés, le nombre de features générées, sans le biais, est :
\[
p+\frac{p(p+1)}{2}
\]

Pour 100 variables :
\[
100+\frac{100\times101}{2}=5150
\]

Cette augmentation peut entraîner un surapprentissage et un coût de calcul important.



## 5. Explorateur interactif — Comprendre les interactions


[[interaction]]

Simulation avec coefficients fictifs, sans lien avec une probabilité de défaut réelle.




## 6. Feature Engineering sur les dates

Les variables temporelles contiennent souvent beaucoup d'informations.

Supposons une colonne :
~~~python
df["transaction_date"]
~~~




Nous pouvons extraire :

- Le mois.
- Le jour de la semaine.
- L'heure.
- Le trimestre.
- Le caractère week-end ou non.

Exemple Python
~~~python
import pandas as pd
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["month"] = df["transaction_date"].dt.month
df["day_of_week"] = df["transaction_date"].dt.dayofweek
df["hour"] = df["transaction_date"].dt.hour
df["quarter"] = df["transaction_date"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
~~~




Variables de durée
Nous pouvons aussi calculer l'ancienneté d'un client.
\[
Tenure=Date_{\mathrm{observation}}-Date_{\mathrm{ouverture}}
\]

~~~python
df["tenure_days"] = (pd.to_datetime(df["observation_date"]) - pd.to_datetime(df["account_open_date"])).dt.days
~~~




La date d'observation doit correspondre au moment où la prédiction est censée être effectuée.

Utiliser une date future créerait une fuite temporelle.



## 7. Encodage cyclique — Sinus et cosinus

Certaines variables temporelles sont cycliques.

Par exemple :

- Décembre et janvier sont proches dans le calendrier.
- Dimanche et lundi sont voisins dans un cycle hebdomadaire.
- 23 h et 0 h sont proches dans une journée.

Si nous représentons les mois par des entiers de 1 à 12, un modèle basé sur la distance considère janvier et décembre comme éloignés de 11 unités.

Pour corriger cela, nous pouvons utiliser deux transformations :
\[
Month_{\sin}=\sin\left(\frac{2\pi Month}{12}\right)
\]

\[
Month_{\cos}=\cos\left(\frac{2\pi Month}{12}\right)
\]

Visualisation
[[cyclic]]
Implémentation
~~~python
import numpy as np
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
~~~




Cette technique est particulièrement utile pour les modèles qui ne découvrent pas naturellement les relations cycliques.



## 8. Agrégations par groupe

Les agrégations permettent de créer des variables décrivant le comportement d'une entité.

Prenons un dataset de transactions :

Client	Transaction	Montant
A	T001	100
A	T002	200
A	T003	300
B	T004	50
B	T005	150

Nous pouvons créer :

- Le montant total des transactions par client.
- Le montant moyen.
- Le nombre de transactions.
- Le montant maximal.

Exemple Python
~~~python
aggregations = df.groupby("client_id")["amount"].agg(["mean", "sum", "count", "max"]).reset_index()
print(aggregations)
# Description globale uniquement : pour prédire à une date, exclure les transactions futures.
~~~




Attention au contexte temporel
Si nous prédisons un événement au 1er juin, nous ne devons pas utiliser des transactions du mois de juillet.

Les agrégations doivent être calculées sur les informations disponibles avant le moment de prédiction.

C'est une source fréquente de Data Leakage.



## 9. Rolling Features — Variables historiques

Dans les problèmes temporels, les statistiques glissantes sont très utiles.

Par exemple :
\[
RollingMean_t=
\frac{1}{k}\sum_{i=1}^{k}X_{t-i}
\]

Cette formule calcule la moyenne des \(k\) valeurs précédentes, sans utiliser la valeur courante.
Exemple
~~~python
df = df.sort_values(["client_id", "transaction_date"])
df["rolling_mean_3"] = df.groupby("client_id")["amount"].transform(
    lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
)
# Pour des transactions simultanées, définir aussi leur ordre/disponibilité métier.
~~~




Le shift(1) est important : il exclut la transaction courante du calcul.

Cela évite d'utiliser une information qui pourrait être indisponible au moment de la prédiction.

Il faut toutefois distinguer une fenêtre de trois observations d'une fenêtre de trois jours : ce ne sont pas les mêmes notions.



## 10. Binning — Transformer des valeurs continues en catégories

Le binning consiste à découper une variable numérique en intervalles.

Par exemple, nous pouvons transformer l'âge en groupes.

Âge	Groupe
18–29	Jeune
30–44	Adulte
45–59	Senior
60+	Senior+

Avec pandas
~~~python
df["age_group"] = pd.cut(df["age"], bins=[18, 30, 45, 60, np.inf],
                         labels=["18-29", "30-44", "45-59", "60+"], right=False)
~~~




Avec KBinsDiscretizer
~~~python
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")
X_binned = discretizer.fit_transform(X_train[["age"]])
X_test_binned = discretizer.transform(X_test[["age"]])
~~~




Trois stratégies courantes

Stratégie	Principe
Uniform	Intervalles de même largeur
Quantile	Intervalles contenant approximativement le même nombre d'observations
KMeans	Intervalles définis à partir de regroupements

Le binning peut être utile pour représenter certaines non-linéarités, mais il entraîne une perte de précision.

Il doit être évalué plutôt qu'appliqué systématiquement.



## 11. Feature Engineering et Data Leakage

C'est l'une des parties les plus importantes du cours.

Une feature peut être très prédictive parce qu'elle contient une information qui n'aurait jamais été disponible au moment de la décision.
Exemple
Nous souhaitons prédire le défaut d'un client au 1er janvier.

Nous construisons :
~~~python
df["total_late_payments_2026"]
# Interdit si cette colonne inclut des retards après la date de prédiction.
~~~




Si cette variable inclut les retards de paiement survenus après le 1er janvier, elle contient de l'information future.

Le modèle pourrait obtenir d'excellents scores en validation, mais ces scores seraient trompeurs.
Trois règles fondamentales
1. Disponibilité temporelle : chaque feature doit être disponible au moment de la prédiction.
2. Séparation train/test : les transformations apprenant des paramètres doivent être ajustées uniquement sur le train.
3. Absence d'information cible : les features ne doivent pas incorporer directement ou indirectement la cible future.



## 12. Feature Engineering avec Pipeline

Nous pouvons automatiser les transformations afin d'assurer leur reproductibilité.
Exemple avec FunctionTransformer
~~~python
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def create_features(X):
    X = X.copy()
    denominator = X["income"].where(X["income"] > 0)
    X["disposable_income"] = X["income"] - X["expenses"]
    X["expense_ratio"] = X["expenses"].div(denominator)
    X["debt_to_income"] = X["debt"].div(denominator)
    return X

pipeline = Pipeline([("features", FunctionTransformer(create_features, validate=False)),
                     ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000))])
~~~




Cette approche permet de centraliser le Feature Engineering dans un pipeline.

Les transformations déterministes comme les ratios peuvent être appliquées directement, mais toute transformation qui apprend des paramètres doit être ajustée uniquement sur le train.



## 13. Coding Lab — Feature Engineering sur un dataset bancaire

Nous allons créer un dataset synthétique et comparer deux modèles :

- Modèle A : variables originales.
- Modèle B : variables originales + features créées.

L'objectif est de vérifier si le Feature Engineering améliore la performance.
~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def create_features(X):
    X = X.copy()
    denominator = X["income"].where(X["income"] > 0)
    X["disposable_income"] = X["income"] - X["expenses"]
    X["expense_ratio"] = X["expenses"].div(denominator)
    X["debt_to_income"] = X["debt"].div(denominator)
    return X


rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({"income": rng.uniform(1500, 10000, n),
                   "expenses": rng.uniform(500, 6000, n),
                   "debt": rng.uniform(0, 30000, n), "age": rng.integers(18, 70, n)})
debt_ratio, expense_ratio = df["debt"] / df["income"], df["expenses"] / df["income"]
logits = -3 + .12 * debt_ratio + 1.5 * expense_ratio - .015 * (df["age"] - 40)
y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, features in [("Originales", "passthrough"),
                           ("Avec features", FunctionTransformer(create_features, validate=False))]:
        pipeline = Pipeline([("features", features),
                             ("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler()),
                             ("model", LogisticRegression(max_iter=1000))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        assert np.isfinite(scores).all()
        print(name, scores, "mean:", scores.mean(), "std:", scores.std())
    print("La cible dépend des ratios par construction. Ce gain éventuel n'est pas universel.")
    print("Test final réservé.")
~~~




Analyse attendue
Compare les résultats.

Dans cette simulation, la cible dépend directement de ratios que nous avons créés.

Il est donc plausible que le modèle utilisant le Feature Engineering soit avantagé.

Mais les résultats restent dépendants de l'échantillon et de la validation.

Important : dans un projet réel, il ne faut pas créer des variables uniquement parce qu'elles améliorent un score sur un unique jeu de validation. Il faut aussi vérifier leur stabilité, leur interprétabilité et leur disponibilité en production.



## 14. Coding Challenge — Construire un Feature Engineering Pipeline

Challenge 06 — Feature Engineering Pipeline
+300 XP

Construis une fonction capable de générer automatiquement plusieurs nouvelles variables à partir d'un dataset bancaire.
~~~python
import numpy as np
import pandas as pd


def feature_engineering(df):
    df = df.copy()
    # Colonnes : income, expenses, debt, age, account_open_date, observation_date.
    # TODO : disposable_income, expense_ratio, debt_to_income, income_log.
    # TODO : account_tenure_days, age_group, income_debt_interaction.
    # TODO : traiter les dénominateurs nuls et les dates incohérentes.
    return df

# BONUS : vérifier absence de ratios infinis, durées cohérentes et absence de mutation.
~~~




Bonus
Ajoute des tests automatiques :

- Vérifier qu'aucun ratio n'est infini.
- Vérifier que les durées sont cohérentes.
- Vérifier que les nouvelles colonnes sont présentes.
- Vérifier que la fonction ne modifie pas le DataFrame original.


[[solution]]




## 15. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 16. Mini-quiz interactif


[[quiz]]




## 17. Validation du cours


[[assessment]]

À retenir : le Feature Engineering permet de transformer des données brutes en représentations potentiellement plus utiles. Les meilleures features sont celles qui apportent une information pertinente, disponible au moment de la prédiction et validée expérimentalement.

Le prochain cours sera consacré au Feature Selection, où nous apprendrons à sélectionner les variables les plus pertinentes et à réduire les features inutiles ou redondantes.