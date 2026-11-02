Dans le cours précédent, nous avons étudié les valeurs manquantes : MCAR, MAR, MNAR et les différentes techniques d'imputation.

Aujourd'hui, nous abordons un autre problème majeur de qualité des données : les outliers, ou valeurs atypiques.

L'objectif n'est pas simplement de les détecter, mais surtout de comprendre pourquoi certaines valeurs doivent être corrigées, tandis que d'autres doivent absolument être conservées.


## 1. Qu'est-ce qu'un outlier ?

Un outlier est une observation qui s'écarte sensiblement du comportement habituel des données selon un critère statistique, algorithmique ou métier.

Imaginons les revenus mensuels suivants :
\[
X=[2000,2200,2400,2500,2700,3000,25000]
\]

La valeur 25 000 est beaucoup plus élevée que les autres.

Mais est-ce une erreur ?

Pas nécessairement. Elle peut correspondre à un client ayant un revenu réellement élevé.
Trois situations à distinguer

Situation	Exemple	Action possible
Erreur de saisie	Âge = 450 ans	Corriger ou invalider
Valeur rare mais réelle	Revenu = 25 000	Conserver ou transformer
Observation anormale	Transaction inhabituelle	Investiguer, éventuellement signaler

Une anomalie statistique n'est pas nécessairement une erreur de données.



## 2. Pourquoi les outliers posent-ils problème ?

Les valeurs extrêmes peuvent influencer certains modèles et certaines statistiques.

### 2.1 Influence sur la moyenne

Prenons :
\[
X=[10,12,13,15,100]
\]

La moyenne est :
\[
\bar x=\frac{10+12+13+15+100}{5}=30
\]

Alors que la médiane est :
\[
Median(X)=13
\]

La moyenne ne représente plus très bien la majorité des observations.

### 2.2 Influence sur certains algorithmes


Algorithme	Sensibilité possible
Régression linéaire avec MSE	Élevée, surtout pour les observations à fort résidu ou levier
KNN	Sensible aux distances et aux échelles
K-Means	Sensible aux points extrêmes
Régression logistique	Peut être sensible aux observations à fort levier
Arbres de décision	Souvent plus robustes aux valeurs extrêmes
Random Forest	Généralement robuste, sans être immunisé

La sensibilité dépend du type d'outlier, du preprocessing et du dataset.



## 3. Méthode 1 — IQR : Interquartile Range

L'IQR est une méthode statistique simple et très utilisée.

On définit :
\[
IQR=Q_3-Q_1
\]

Où :
- \(Q_1\) est le premier quartile.
- \(Q_3\) est le troisième quartile.
Les bornes classiques de Tukey sont :
\[
LowerBound=Q_1-1.5\times IQR
\]

\[
UpperBound=Q_3+1.5\times IQR
\]

Les observations situées au-delà de ces bornes sont signalées comme atypiques.
Exemple
Supposons :
\[
Q_1=20,\quad Q_3=40
\]

Alors :
\[
IQR=40-20=20
\]

\[
LowerBound=20-1.5(20)=-10
\]

\[
UpperBound=40+1.5(20)=70
\]

Une valeur de 90 sera signalée comme outlier.
[[iqr]]

Implémentation Python

~~~python
Q1 = df["income"].quantile(0.25)
Q3 = df["income"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df["income"] < lower) | (df["income"] > upper)]
print("Nombre d’outliers :", len(outliers))
~~~




Limites de l'IQR
Cette méthode est univariée : elle analyse une variable à la fois.

Elle peut signaler de nombreuses valeurs légitimes lorsque la distribution est fortement asymétrique.

Elle ne détecte pas directement les anomalies qui n'apparaissent que dans les relations entre plusieurs variables.



## 4. Méthode 2 — Z-score

Le Z-score mesure la distance d'une observation à la moyenne, en unités d'écart-type.
\[
Z_i=\frac{x_i-\mu}{\sigma}
\]

Une règle fréquemment utilisée consiste à signaler les observations pour lesquelles :
\[
|Z_i|>3
\]

Mais ce seuil est une convention, pas une loi universelle.
Exemple
Supposons :
\[
\mu=50,\quad \sigma=10,\quad x=85
\]

Alors :
\[
Z=\frac{85-50}{10}=3.5
\]

La valeur 85 est située à 3,5 écarts-types au-dessus de la moyenne.
Implémentation

~~~python
from scipy.stats import zscore
z = zscore(df["income"], nan_policy="omit")
outliers = df[abs(z) > 3]
~~~




Limite importante
Le Z-score classique utilise la moyenne et l'écart-type, qui peuvent eux-mêmes être influencés par les outliers.

Il est donc moins adapté à certaines distributions asymétriques ou fortement contaminées.



## 5. Méthode 3 — Robust Z-score avec MAD

Une alternative plus robuste repose sur la médiane et le Median Absolute Deviation.
\[
MAD=Median(|x_i-Median(X)|)
\]

Le Z-score modifié est souvent défini par :
\[
Z_i^{robust}=
\frac{0.6745(x_i-Median(X))}{MAD}
\]

Le facteur 0,6745 permet une mise à l'échelle comparable au Z-score classique sous une distribution normale.

Un seuil couramment rencontré est :
\[
|Z_i^{robust}|>3.5
\]

Exemple Python

~~~python
import numpy as np
x = df["income"].dropna()
median = np.median(x)
mad = np.median(np.abs(x - median))
if mad > 0:
    robust_z = 0.6745 * (x - median) / mad
    outliers = x[np.abs(robust_z) > 3.5]
else:
    print("MAD nul : utiliser une autre stratégie.")
~~~




Lorsque le MAD est nul, la formule ne peut pas être appliquée directement.



## 6. Comparaison IQR, Z-score et Robust Z-score


Critère	IQR	Z-score	Robust Z-score
Statistiques	Quartiles	Moyenne, écart-type	Médiane, MAD
Robustesse aux extrêmes	Bonne	Faible	Bonne
Distribution normale requise ?	Non	Non pour calculer, mais utile pour interpréter les seuils	Non
Détection multivariée	Non	Non	Non
Complexité	Faible	Faible	Faible

Ces trois méthodes sont des approches statistiques univariées.

Pour détecter des anomalies qui dépendent de plusieurs variables simultanément, nous pouvons utiliser des algorithmes spécialisés.



## 7. Méthode 4 — Isolation Forest

Isolation Forest est un algorithme non supervisé de détection d'anomalies.

Son principe est différent des méthodes statistiques précédentes.

Au lieu de mesurer uniquement la distance à une moyenne ou à une médiane, il cherche à isoler les observations à l'aide de partitions aléatoires.
Intuition
Une observation atypique est souvent plus facile à isoler qu'une observation située dans une région dense.

L'algorithme construit plusieurs arbres d'isolation et mesure la longueur moyenne des chemins nécessaires pour isoler chaque observation.

Les points isolés rapidement sont considérés comme plus suspects.
Isolation Forest — Intuition
Schéma conceptuel : un point isolé peut être séparé par peu de partitions.
Implémentation Python

~~~python
from sklearn.ensemble import IsolationForest
features = ["income", "debt_ratio", "age"]
X = df[features].dropna()
model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
predictions = model.fit_predict(X)
# 1 = normal, -1 = anomalie
outliers = X[predictions == -1]
print("Anomalies détectées :", len(outliers))
~~~




Comprendre contamination
Le paramètre contamination sert à définir le seuil de décision selon une proportion supposée d'anomalies dans les données d'entraînement.

Par exemple :

~~~python
contamination=0.05
~~~




indique une proportion attendue de 5 % d'anomalies.

Cela ne signifie pas que les données contiennent réellement 5 % d'erreurs.
Avantages
Isolation Forest peut traiter plusieurs variables et détecter des anomalies qui ne sont pas nécessairement extrêmes sur une seule colonne.
Limites
Il dépend des features utilisées, de leur représentation et du choix des paramètres.

Les anomalies détectées doivent être interprétées dans le contexte métier.



## 8. Méthode 5 — Local Outlier Factor (LOF)

Local Outlier Factor détecte les anomalies en comparant la densité locale d'une observation à celle de ses voisins.
Intuition
Une observation peut être considérée comme atypique lorsqu'elle se trouve dans une zone beaucoup moins dense que les zones occupées par ses voisins.

Cette approche est intéressante lorsque les données présentent plusieurs groupes de densités différentes.
Exemple
LOF — Comparaison de densités locales
Illustration d'un point isolé par rapport aux régions voisines.
Implémentation

~~~python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X_scaled)
outliers = X[labels == -1]
~~~




Par défaut, LocalOutlierFactor est principalement destiné à détecter les anomalies dans les données utilisées lors de fit_predict().

Pour détecter des anomalies sur de nouvelles observations, il faut utiliser le mode novelty=True et respecter la séparation entre entraînement et évaluation.



## 9. Isolation Forest versus LOF


Critère	Isolation Forest	LOF
Principe	Isolation par partitions aléatoires	Densité locale
Multivarié	Oui	Oui
Dépend des distances	Pas directement	Oui
Sensible au scaling	Peut l'être selon la représentation	Généralement oui
Données volumineuses	Souvent pratique	Peut devenir coûteux
Nouvelles observations	predict() disponible	novelty=True nécessaire

Les deux méthodes peuvent être utiles, mais elles ne répondent pas exactement à la même logique.



## 10. Comment traiter les outliers ?

Détecter une valeur atypique ne signifie pas qu'il faut la supprimer.

Il existe plusieurs stratégies.

### 10.1 Conserver

C'est souvent le bon choix lorsque la valeur est réelle et importante pour le problème.

Par exemple, dans la détection de fraude, les transactions inhabituelles peuvent justement être celles que nous cherchons à identifier.

### 10.2 Corriger

Si la valeur provient d'une erreur connue, elle peut être corrigée à partir d'une source fiable.

Exemple : un montant enregistré en millimes au lieu de dinars.

### 10.3 Supprimer

La suppression peut être justifiée lorsqu'une observation est clairement invalide ou hors du périmètre d'analyse.

Mais elle peut également introduire un biais si les valeurs extrêmes sont légitimes.

### 10.4 Capping / Winsorization

Cette méthode consiste à limiter les valeurs à des bornes définies.

Par exemple :
\[
x'=
\begin{cases}
L & x<L\\
x & L\le x\le U\\
U & x>U
\end{cases}
\]

Avec \(L\) et \(U\) les bornes inférieure et supérieure.

### 10.5 Transformation logarithmique

Pour certaines variables positives et fortement asymétriques, nous pouvons utiliser :
\[
x'=\log(1+x)
\]

Cette transformation réduit l'échelle des grandes valeurs.

Elle nécessite des valeurs compatibles avec son domaine de définition et ne convient pas automatiquement à toutes les variables.



## 11. Explorateur interactif — Traitement des valeurs extrêmes


[[outliertreatment]]




## 12. Outliers et Data Leakage

C'est un point essentiel.

Supposons que nous calculions les bornes IQR sur tout le dataset avant de séparer train et test.

Les données de test influencent alors le choix des bornes.

Cela crée une contamination.
Mauvaise approche

~~~python
Q1 = X["income"].quantile(0.25)
Q3 = X["income"].quantile(0.75)
# Mauvaise approche : bornes apprises avant le split.
~~~




Bonne approche

~~~python
Q1 = X_train["income"].quantile(0.25)
Q3 = X_train["income"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
X_train_capped, X_test_capped = X_train.copy(), X_test.copy()
X_train_capped["income"] = X_train["income"].clip(lower, upper)
X_test_capped["income"] = X_test["income"].clip(lower, upper)
~~~




Les bornes sont apprises sur le train, puis réutilisées sur le test.

Pendant la cross-validation, elles doivent également être recalculées à l'intérieur de chaque fold d'entraînement.



## 13. Coding Lab — Comparer les méthodes de détection

Nous allons générer un dataset contenant des observations normales et des anomalies artificielles.

Puis nous comparerons :
- IQR.
- Z-score.
- Isolation Forest.
- Local Outlier Factor.

~~~python
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

rng = np.random.default_rng(42)
normal = rng.normal(loc=50, scale=10, size=(500, 2))
anomalies = rng.uniform(low=100, high=150, size=(20, 2))
X = np.vstack([normal, anomalies])
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
truth = np.r_[np.zeros(500, dtype=bool), np.ones(20, dtype=bool)]
q1, q3 = df.quantile(0.25), df.quantile(0.75)
iqr = q3 - q1
median = np.median(X, axis=0)
mad = np.median(np.abs(X - median), axis=0)
assert (mad > 0).all()
scaled = StandardScaler().fit_transform(X)
flags = {
    "IQR": ((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).any(axis=1).to_numpy(),
    "Z-score": (np.abs(zscore(X, axis=0)) > 3).any(axis=1),
    "Robust Z": (np.abs(0.6745 * (X - median) / mad) > 3.5).any(axis=1),
    "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.05,
                                        random_state=42).fit_predict(X) == -1,
    "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(scaled) == -1,
}

if __name__ == "__main__":
    for name, detected in flags.items():
        assert detected.shape == truth.shape
        print(name, "signalées:", detected.sum(),
              "precision:", precision_score(truth, detected, zero_division=0),
              "recall:", recall_score(truth, detected, zero_division=0))
    print("Comparaison descriptive sur les données de fit, sans test indépendant.")
    print("Les anomalies injectées sont des labels de simulation, pas des erreurs métier prouvées.")
~~~




Analyse attendue
Observe que les méthodes ne détectent pas nécessairement le même nombre d'anomalies.

Dans cet exemple, les anomalies artificielles sont très éloignées des observations normales.

Mais dans des données réelles, les anomalies peuvent être beaucoup plus difficiles à distinguer.

Attention : ce laboratoire utilise l'ensemble des observations pour comparer les détecteurs sur un même dataset. Il ne constitue pas un protocole d'évaluation prédictive sur un test indépendant.



## 14. Coding Challenge — Outlier Detection Pipeline

Challenge 03 — Robust Outlier Processing
+250 XP

Construis une classe scikit-learn permettant de limiter les valeurs extrêmes selon les bornes IQR apprises sur les données d'entraînement.

~~~python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRCapper(TransformerMixin, BaseEstimator):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        # TODO : vérifier le fit et appliquer np.clip aux bornes apprises.
        raise NotImplementedError

# TODO : Pipeline(SimpleImputer, IQRCapper, StandardScaler, LogisticRegression).
# BONUS : comparer sans capping et avec RobustScaler, sur les mêmes folds train.
~~~




Bonus
Compare trois pipelines :
1. Sans traitement des outliers.
2. Avec IQRCapper.
3. Avec RobustScaler.
Analyse les différences de performance et explique pourquoi un traitement plus agressif n'est pas nécessairement meilleur.


[[solution]]




## 15. Mini-quiz interactif


[[quiz]]




## 16. Questions d'entretien — Data Scientist / ML Engineer


Réponds oralement avant de révéler la correction.

[[interview]]




## 17. Validation du cours


[[assessment]]

À retenir : la détection des outliers est une étape d'investigation, pas une instruction automatique de suppression. Le bon traitement dépend de la nature des données, de leur validité et du modèle utilisé.

Le prochain cours sera consacré au Feature Scaling, avec StandardScaler, MinMaxScaler, RobustScaler, Normalizer et les transformations de distribution.