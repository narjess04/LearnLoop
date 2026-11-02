
## 1. Pourquoi comprendre les données avant de les transformer ?

Supposons que tu reçoives un dataset de 50 000 clients avec des informations financières, démographiques et comportementales.

Tu souhaites entraîner un modèle pour prédire leur probabilité de défaut.

Avant de choisir un algorithme, il faut savoir :
- Quelles variables sont numériques, catégorielles ou temporelles ?
- Quelles colonnes contiennent des valeurs manquantes ?
- Certaines valeurs sont-elles incohérentes ?
- Les distributions sont-elles déséquilibrées ?
- Certaines variables contiennent-elles des informations indisponibles au moment de la prédiction ?
Une mauvaise compréhension des données peut conduire à un modèle performant sur le papier, mais inutilisable en pratique.

Principe : la qualité du modèle dépend en partie de la qualité et de la pertinence des données utilisées.

## 2. Les différents types de variables

C'est la première distinction à maîtriser.

Variables numériques

Représentent des quantités mesurables.

Exemples : âge, revenu, montant de crédit, taux d'intérêt.



Variables catégorielles

Représentent des modalités ou des groupes.

Exemples : secteur d'activité, région, type de contrat.



Variables temporelles

Représentent une date, une heure ou une durée.

Exemples : date d'ouverture d'un compte, date de transaction.



Variables binaires

Prennent généralement deux valeurs.

Exemples : défaut oui/non, présence d'un bilan, abonnement actif/inactif.




### 2.1 Variables numériques : discrètes et continues

Une variable discrète prend des valeurs dénombrables, souvent entières.

Exemple : nombre de retards de paiement.

Une variable continue représente une quantité mesurable.

Exemple : revenu annuel.

Attention : le fait qu'une colonne contienne des nombres ne signifie pas qu'elle doit être traitée comme une variable quantitative.

Un code postal ou un identifiant client est numérique en apparence, mais n'exprime pas nécessairement une quantité.

### 2.2 Variables catégorielles : nominales et ordinales


Type	Exemple	Ordre naturel ?
Nominale	Région : Tunis, Sfax, Sousse	Non
Ordinale	Risque : faible, moyen, élevé	Oui
Binaire	Oui / Non	Deux modalités

Cette distinction est essentielle pour choisir une technique d'encodage.

Par exemple, attribuer les valeurs 1, 2 et 3 à des régions pourrait introduire un ordre artificiel.

## 3. Comprendre les distributions

La distribution d'une variable décrit comment ses valeurs sont réparties.

Deux variables peuvent avoir la même moyenne et pourtant présenter des distributions très différentes.

### 3.1 Distribution symétrique

Exemple : distribution normale

Une distribution normale est symétrique autour de sa moyenne. Pour cette distribution, moyenne, médiane et mode coïncident.


### 3.2 Distribution asymétrique

Exemple : distribution asymétrique à droite

La majorité des observations est concentrée vers les petites valeurs, avec une longue queue vers les grandes valeurs. Les revenus ou montants de transaction présentent parfois cette forme.

Une distribution fortement asymétrique peut influencer certaines statistiques, notamment la moyenne et l'écart-type.

Cela ne signifie pas qu'il faut automatiquement la transformer : la décision dépend du modèle et de l'objectif.

## 4. Les statistiques descriptives essentielles


### 4.1 Moyenne

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

La moyenne résume le niveau moyen d'une variable.

Mais elle est sensible aux valeurs extrêmes.

### 4.2 Médiane

La médiane partage les observations ordonnées en deux moitiés.

Elle est généralement plus robuste que la moyenne face aux valeurs extrêmes.

### 4.3 Variance

\[
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

Cette formule correspond à la variance descriptive de la population observée.

Pour estimer la variance d'une population à partir d'un échantillon, on utilise souvent \(n-1\) au dénominateur.

### 4.4 Écart-type

\[
\sigma=\sqrt{\sigma^2}
\]

L'écart-type mesure la dispersion autour de la moyenne.

### 4.5 Quartiles et IQR

L'écart interquartile est :
\[
IQR=Q_3-Q_1
\]

Il mesure l'étendue de la moitié centrale des données.

On l'utilise notamment dans l'analyse des valeurs atypiques.

## 5. Explorateur interactif — Moyenne versus médiane


[[distribution]]


## 6. Comprendre la qualité des données

La qualité des données ne se résume pas au nombre de valeurs manquantes.

Nous pouvons distinguer plusieurs dimensions.

Dimension	Question	Exemple
Complétude	Les informations sont-elles présentes ?	Revenu manquant
Validité	Les valeurs respectent-elles les règles ?	Âge négatif
Unicité	Les enregistrements sont-ils dupliqués ?	Même transaction répétée
Cohérence	Les informations se contredisent-elles ?	Date de clôture antérieure à l'ouverture
Exactitude	Les valeurs correspondent-elles à la réalité ?	Montant incorrect
Actualité	Les informations sont-elles suffisamment récentes ?	Ancien statut client

Une variable peut être complète à 100 % et pourtant être de mauvaise qualité.

Par exemple, une colonne age sans valeur manquante mais contenant des valeurs négatives n'est pas valide.

## 7. Les valeurs manquantes : première analyse

Nous étudierons les techniques d'imputation en détail au cours 2.

Pour le moment, il faut savoir détecter et mesurer les valeurs manquantes.
Calcul du taux de valeurs manquantes
\[
MissingRate_j=
\frac{\text{Nombre de valeurs manquantes de }j}
{\text{Nombre total d'observations}}
\times100
\]

Exemple :

Si une colonne possède 200 valeurs manquantes sur 1 000 lignes :
\[
MissingRate=20\%
\]

Exemple Python
~~~python
missing_count = df.isna().sum()
missing_rate = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing_rate)
~~~



Une variable très incomplète n'est pas automatiquement inutile. La décision de la conserver dépend de sa valeur prédictive, du mécanisme de données manquantes et du contexte métier.

## 8. Analyse des distributions et visualisations

Les visualisations permettent de détecter rapidement des anomalies et de comprendre les variables.
Histogramme
Un histogramme représente la distribution d'une variable numérique.
~~~python
import matplotlib.pyplot as plt
df["income"].hist(bins=30)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income distribution")
plt.show()
~~~



Boxplot
Un boxplot résume la médiane, les quartiles et les valeurs situées au-delà des moustaches.
~~~python
import matplotlib.pyplot as plt
plt.boxplot(df["income"].dropna())
plt.title("Income boxplot")
plt.show()
~~~



Countplot ou bar chart
Pour une variable catégorielle :
~~~python
df["region"].value_counts().plot(kind="bar")
plt.title("Distribution des régions")
plt.show()
~~~



Le choix de la visualisation dépend du type de variable et de la question analysée.

## 9. Analyse de la variable cible


Avant de construire un modèle de classification, examiner les proportions de chaque classe.

[[target]]

Exemple illustratif : 94 % de non-défauts et 6 % de défauts. Un classifieur prédisant toujours « non-défaut » aurait 94 % d’accuracy et un rappel nul sur les défauts. Examiner des métriques adaptées et la stratégie de validation.

StratifiedKFold préserve approximativement les proportions dans les folds lorsque le protocole autorise un découpage stratifié. La chronologie et les groupes restent prioritaires. Le déséquilibre sera étudié au cours 9.


## 10. Coding Lab — Première EDA complète


Dataset synthétique de clients. Objectif : produire un premier diagnostic de qualité et de distribution. À exécuter dans ton environnement Python.

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "age": rng.integers(18, 75, n),
    "income": rng.lognormal(10.5, 0.6, n),
    "debt_ratio": rng.uniform(0, 1, n),
    "region": rng.choice(["Tunis", "Sfax", "Sousse", "Nabeul"], n),
    "default": rng.choice([0, 1], n, p=[0.92, 0.08]),
})
missing_idx = rng.choice(n, 100, replace=False)
df.loc[missing_idx, "income"] = np.nan
df.loc[0, "age"] = -5
df.loc[1, "debt_ratio"] = 1.8

print("Shape:", df.shape)
print(df.dtypes)
print(df.describe(include="all"))
print("Valeurs manquantes (%):", df.isna().mean() * 100)
print("Doublons exacts:", df.duplicated().sum())
print("Âges invalides:", (df["age"] < 0).sum())
print("Ratios invalides:", (~df["debt_ratio"].between(0, 1)).sum())
print("Cible (%):", df["default"].value_counts(normalize=True) * 100)
assert df.shape == (1000, 5)
assert df["income"].isna().sum() == 100
assert (df["age"] < 0).sum() == 1
assert (~df["debt_ratio"].between(0, 1)).sum() == 1

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df["income"].hist(bins=30, ax=axes[0])
    axes[0].set(title="Distribution du revenu", xlabel="Revenu", ylabel="Effectif")
    axes[1].boxplot(df["income"].dropna())
    axes[1].set_title("Boxplot du revenu")
    df["region"].value_counts().plot(kind="bar", ax=axes[2])
    axes[2].set_title("Distribution des r?gions")
    fig.tight_layout()
    plt.show()
~~~

Travail demandé : détecter les âges invalides, les ratios d’endettement hors de [0, 1], les absences et le déséquilibre de la cible. Proposer les contrôles avant modélisation. Réserver le test final et réaliser les choix exploratoires sur train.


## 11. Coding Challenge — Construire un Data Quality Report


Challenge 01 — Data Quality Profiler. Compléter le squelette avec moyenne, médiane, écart-type, minimum, maximum et tri décroissant des taux d’absence.

~~~python
import pandas as pd
from ch02_cours01_lab import df


def data_quality_report(df):
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean() * 100,
        "unique_values": df.nunique(dropna=True),
    })
    # TODO : moyenne, médiane, écart-type, minimum, maximum numériques.
    # TODO : trier par missing_rate décroissant.
    # BONUS : compter séparément valeurs absentes et violations métier.
    return report


if __name__ == "__main__":
    print(data_quality_report(df))
~~~

Bonus : compter les violations des règles age ≥ 0, debt_ratio dans [0, 1], default dans {0, 1}. Distinguer absences et valeurs invalides. Pour les colonnes non numériques, laisser les statistiques numériques absentes. Pandas calcule std() avec n−1 par défaut ; std(ddof=0) utilise n.

[[solution]]


## 12. Mini-quiz interactif


[[quiz]]


## 13. Questions d’entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]


## 14. Validation du cours


[[assessment]]

À retenir : comprendre la signification, les distributions et les problèmes de qualité avant de transformer. Justifier chaque transformation.

Prochain cours : Missing Values — MCAR, MAR, MNAR et techniques d’imputation. Support à venir.
