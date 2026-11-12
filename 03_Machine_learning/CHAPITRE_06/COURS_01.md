Cours 01 — Fondamentaux des séries temporelles
Partie A · 01/07
Niveau : Fondamental → Intermédiaire
Durée : 2–3 h


## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Identifier les composantes d'une série temporelle : tendance, saisonnalité, cycles et résidus.
- Comprendre la décomposition additive, multiplicative et STL.
- Expliquer la stationnarité et son importance.
- Appliquer les tests ADF et KPSS.
- Utiliser la différenciation pour stabiliser une série.
- Interpréter les fonctions ACF et PACF.
- Réaliser une première analyse exploratoire avec Python.

## 2. Qu'est-ce qu'une série temporelle ?

Une série temporelle (Time Series) est une succession d'observations enregistrées dans un ordre chronologique.

Contrairement à un dataset classique, l'ordre des observations possède une signification.

Par exemple, les ventes d'aujourd'hui peuvent dépendre des ventes d'hier, de la période de l'année et de la tendance générale.
Exemple : ventes mensuelles d'un magasin

Données fictives à des fins pédagogiques

Les ventes augmentent globalement, avec des pics récurrents en fin d'année.
Mathématiquement, on représente une série par :
\[
Y_t,\quad t=1,2,\ldots,T
\]

où \(Y_t\) représente la valeur observée à l'instant \(t\).
Différence avec le Machine Learning classique
ML classique	Séries temporelles
Observations souvent considérées indépendantes	Dépendances temporelles fréquentes
Ordre des lignes souvent secondaire	Ordre chronologique essentiel
Split aléatoire parfois adapté	Validation chronologique nécessaire
Relations entre variables	Relations entre variables et instants passés
Prédiction d'une cible	Prévision d'une valeur future

Point essentiel : une série temporelle n'est pas simplement un tableau contenant une colonne date. C'est une structure dans laquelle le temps influence potentiellement le phénomène observé.

## 3. Les quatre composantes d'une série temporelle

Une série peut contenir plusieurs structures simultanément.
Tendance
Évolution de long terme

Saisonnalité
Motif répété à période fixe

Cycle
Fluctuation sans période fixe

Résidu
Variation non expliquée


## 3.1 La tendance (Trend)

La tendance correspond à l'évolution générale d'une série sur une longue période.

Exemples : augmentation du nombre d'utilisateurs d'une application, baisse progressive du coût du stockage informatique.

Une tendance linéaire simple s'écrit :
\[
T_t=a+bt
\]

- \(a\) : niveau initial.
- \(b\) : vitesse d'évolution.
- \(t\) : instant temporel.
Lorsque \(b>0\), la tendance est croissante.

## 3.2 La saisonnalité (Seasonality)

La saisonnalité représente un comportement qui se répète à intervalles réguliers.

Exemples : ventes plus élevées en décembre, consommation électrique différente entre le jour et la nuit, trafic web variable selon le jour de la semaine.

La période saisonnière \(m\) dépend de la fréquence des observations.
Fréquence	Exemple de saisonnalité	Période \(m\)
Mensuelle	Annuelle	12
Trimestrielle	Annuelle	4
Journalière	Hebdomadaire	7
Horaire	Journalière	24

Une série peut posséder plusieurs saisonnalités, par exemple une saisonnalité journalière et hebdomadaire.

## 3.3 Le cycle (Cycle)

Un cycle correspond à une fluctuation de plus longue durée, sans périodicité nécessairement fixe.

Exemple : alternance entre périodes d'expansion et de ralentissement économique.

Différence : la saisonnalité revient à une fréquence relativement prévisible, tandis que la durée d'un cycle peut varier.

## 3.4 Le résidu (Residual)

Le résidu représente la partie qui reste après avoir retiré les composantes modélisées.

Il peut contenir du bruit, des événements exceptionnels ou des structures que notre décomposition n'a pas capturées.

Un résidu n'est donc pas nécessairement un bruit blanc.

## 4. Décomposition additive, multiplicative et STL


## 4.1 Décomposition additive

\[
Y_t=T_t+S_t+R_t
\]

On privilégie une décomposition additive lorsque l'amplitude saisonnière reste approximativement constante.

Exemple : un magasin vend chaque décembre environ 50 unités supplémentaires, indépendamment de son volume moyen.

## 4.2 Décomposition multiplicative

\[
Y_t=T_t\times S_t\times R_t
\]

Elle convient lorsque l'amplitude des variations saisonnières augmente avec le niveau de la série.

Exemple : les ventes augmentent de 20 % chaque décembre plutôt que d'un nombre fixe d'unités.

Pour des données strictement positives :
\[
\log(Y_t)=\log(T_t)+\log(S_t)+\log(R_t)
\]

Le logarithme permet ainsi de transformer une structure multiplicative en structure additive.

## 4.3 Décomposition STL

STL signifie Seasonal-Trend decomposition using LOESS.

Elle sépare une série en trois composantes :
\[
Y_t=T_t+S_t+R_t
\]

STL utilise un lissage local pour estimer la tendance et la saisonnalité.

Ses avantages :
- Tendance pouvant être non linéaire.
- Saisonnalité pouvant évoluer progressivement.
- Possibilité de réduire l'influence des valeurs aberrantes avec une estimation robuste.
Implémentation Python

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

period=12 signifie que nous recherchons une saisonnalité annuelle dans des données mensuelles.

Attention : les dates manquantes et les valeurs manquantes doivent être traitées avant l'application de STL.

## 5. La stationnarité

La stationnarité est une notion fondamentale pour comprendre les modèles statistiques temporels.

Une série est faiblement stationnaire lorsque :

## 1. Son espérance est constante.


## 2. Sa variance est constante et finie.


## 3. Sa covariance entre deux instants dépend uniquement de leur décalage temporel.

Mathématiquement :
\[
E[Y_t]=\mu
\]

\[
Var(Y_t)=\sigma^2
\]

\[
Cov(Y_t,Y_{t-k})=\gamma(k)
\]

Une série présentant une forte tendance déterministe ou une saisonnalité marquée n'est généralement pas stationnaire au sens faible.
[[timeseries]]

Comparer deux comportements temporels
Stationnaire (illustration)
Oscillation autour d'un niveau constant.

Non stationnaire (illustration)
Le niveau moyen évolue avec le temps.

Pourquoi est-ce important ?
Les modèles classiques tels qu'ARMA et certaines formulations d'ARIMA reposent sur des hypothèses de stationnarité pour leur composante autorégressive.

En revanche, tous les modèles de prévision ne nécessitent pas que la série brute soit stationnaire.

Par exemple, les modèles basés sur les arbres peuvent exploiter des variables temporelles et des lags sans imposer cette hypothèse de la même façon.

## 6. Les tests de stationnarité : ADF et KPSS

Il ne suffit pas d'observer un graphique pour confirmer la stationnarité.

Deux tests statistiques complémentaires sont couramment utilisés.

## 6.1 Test ADF — Augmented Dickey-Fuller

L'ADF recherche des preuves contre l'hypothèse d'une racine unitaire.
- \(H_0\) : la série possède une racine unitaire.
- \(H_1\) : absence de racine unitaire dans la spécification testée.

## 6.2 Test KPSS

Le KPSS teste la stationnarité autour d'un niveau ou d'une tendance déterministe, selon sa configuration.
- \(H_0\) : stationnarité selon la spécification choisie.
- \(H_1\) : non-stationnarité.
Interprétation combinée
Pour un seuil de 5 %, en utilisant un KPSS de stationnarité autour d'un niveau :
ADF	KPSS	Interprétation
p < 0,05	p ≥ 0,05	Compatible avec une série stationnaire
p ≥ 0,05	p < 0,05	Indices de non-stationnarité
p < 0,05	p < 0,05	Résultats contradictoires
p ≥ 0,05	p ≥ 0,05	Résultats non concluants

Ces tests ne prouvent pas définitivement la stationnarité. Le choix de la constante, de la tendance et des retards peut influencer les conclusions.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## 7. La différenciation (Differencing)

Lorsqu'une série présente une tendance stochastique, la différenciation peut aider à la rendre stationnaire.

## 7.1 Différenciation d'ordre 1

\[
\Delta Y_t=Y_t-Y_{t-1}
\]

Exemple :
Mois	Ventes	Différence
Janvier	100	—
Février	110	+10
Mars	125	+15
Avril	130	+5

Nous n'analysons plus directement les ventes, mais leur variation d'un mois à l'autre.

## 7.2 Différenciation saisonnière

\[
\Delta_m Y_t=Y_t-Y_{t-m}
\]

Pour une série mensuelle avec saisonnalité annuelle :
\[
\Delta_{12}Y_t=Y_t-Y_{t-12}
\]

Nous comparons chaque mois au même mois de l'année précédente.

## 7.3 Différenciation d'ordre 2

\[
\Delta^2Y_t=Y_t-2Y_{t-1}+Y_{t-2}
\]

Elle peut être utile lorsque la première différenciation ne suffit pas, mais une différenciation excessive risque de dégrader la structure utile du signal.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

La différenciation ne garantit pas toujours la stationnarité : il faut examiner la série transformée et ses propriétés.

## 8. ACF et PACF : comprendre la dépendance temporelle


## 8.1 ACF — Autocorrelation Function

L'ACF mesure la corrélation entre une série et ses valeurs décalées.
\[
\rho(k)=Corr(Y_t,Y_{t-k})
\]

Par exemple, une autocorrélation élevée au lag 7 dans des données journalières peut indiquer un comportement hebdomadaire.

## 8.2 PACF — Partial Autocorrelation Function

La PACF mesure la relation entre \(Y_t\) et \(Y_{t-k}\), après avoir contrôlé l'effet linéaire des lags intermédiaires.

Elle aide à distinguer les relations directes des relations transmises par les observations intermédiaires.
Comment les interpréter ?
Exemple pédagogique : ACF vs PACF d'un processus AR(1)
ACF
Décroissance progressive

PACF
Pic principal au lag 1

Illustration schématique, non issue d'un ajustement réel. Les lignes pointillées ne représentent pas des intervalles de confiance calculés.
Quelques comportements typiques :
Modèle théorique	ACF	PACF
AR(p) stationnaire	Décroissance progressive ou oscillante	Coupure après le lag p
MA(q)	Coupure après le lag q	Décroissance progressive
ARMA(p,q)	Décroissance progressive	Décroissance progressive

Ces règles sont des indications théoriques, pas des garanties sur des données réelles.
Implémentation Python

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## 9. Mini-projet pratique : analyser une série temporelle

Nous allons créer une série mensuelle synthétique contenant une tendance, une saisonnalité et du bruit, puis appliquer les concepts étudiés.


## Étape 1 — Générer les données


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 2 — Décomposer la série


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 3 — Tester la stationnarité


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 4 — Différencier et réexaminer


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 5 — Analyser les corrélations temporelles


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Dans ce mini-projet, les transformations sont utilisées pour comprendre la série. Elles ne constituent pas encore un pipeline complet de forecasting.

## 10. Bonnes pratiques et erreurs fréquentes

Erreur	Bonne pratique
Mélanger aléatoirement les dates	Respecter la chronologie
Supposer que toutes les séries sont stationnaires	Vérifier les propriétés et les hypothèses
Différencier systématiquement	Transformer uniquement lorsque c'est pertinent
Confondre cycle et saisonnalité	Examiner la régularité de la période
Choisir ARIMA uniquement avec ACF/PACF	Compléter par les critères et diagnostics
Ignorer les dates manquantes	Vérifier la fréquence et les trous temporels
Interpréter tout résidu comme du bruit blanc	Vérifier les autocorrélations résiduelles

Un point particulièrement important pour les projets professionnels : une transformation calculée avec les données futures peut introduire une fuite temporelle. Toute préparation destinée à un modèle prédictif doit respecter les informations disponibles au moment de la prévision.

[[quiz]]


## 12. Synthèse du cours

À retenir :
- Une série temporelle contient des dépendances chronologiques qu'il faut respecter.
- La tendance décrit l'évolution générale ; la saisonnalité décrit les motifs périodiques.
- STL permet d'extraire tendance, saisonnalité et résidus.
- La stationnarité est essentielle pour certaines familles de modèles statistiques.
- ADF et KPSS fournissent des diagnostics complémentaires.
- La différenciation peut aider à éliminer certaines formes de non-stationnarité.
- ACF et PACF permettent d'examiner la structure des dépendances temporelles.
Prochaine étape — Cours 02 : Baselines et validation temporelle.

Nous verrons comment construire des premières prévisions naïves et saisonnières, appliquer le walk-forward validation, éviter les fuites temporelles et mesurer les performances avec MAE, RMSE, MASE et sMAPE.


## Coding Lab complet

Les blocs aplatis de la pièce jointe sont regroupés ici dans un script autonome. La fonction de métriques tronquée du cours 02 est complétée.

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def dataset():
    rng = np.random.default_rng(42)
    time = np.arange(72)
    sales = 100 + 1.5*time + 20*np.sin(2*np.pi*time/12) + rng.normal(0, 5, 72)
    return pd.DataFrame({'ventes': sales}, index=pd.date_range('2020-01-01', periods=72, freq='MS'))

def difference(values, lag=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not isinstance(lag, (int, np.integer)) or not 0 < lag < len(values):
        raise ValueError('Retard entier positif et inférieur à la longueur requis.')
    return values[lag:] - values[:-lag]

def check_stationarity(series):
    series = series.dropna()
    return {'ADF': adfuller(series, autolag='AIC')[1],
            'KPSS': kpss(series, regression='c', nlags='auto')[1]}

def run():
    df = dataset()
    print('Fréquence :', df.index.freqstr, 'Manquants :', df.isna().sum().sum())
    df.plot(title='Ventes mensuelles synthétiques', figsize=(11, 4))
    STL(df['ventes'], period=12, robust=True).fit().plot()
    candidate = df['ventes'].diff(12).diff().dropna()
    print('Brut :', check_stationarity(df['ventes']))
    print('Transformé :', check_stationarity(candidate))
    candidate.plot(title='Différenciation saisonnière puis simple', figsize=(11, 4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(candidate, lags=18, ax=axes[0])
    plot_pacf(candidate, lags=18, method='ywm', ax=axes[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()
~~~

## Challenge Python

Implémenter difference(values, lag=1) sans modifier les entrées. Générer 72 mois, décomposer avec STL, comparer ADF/KPSS avant et après transformation et commenter ACF/PACF.

[[assessment]]

[[interview]]


## Interpréter le lab

Avec cette graine, les diagnostics après différenciation peuvent être contradictoires : ADF rejette la racine unitaire tandis que KPSS rejette aussi la stationnarité autour d’un niveau. Ne conclus pas automatiquement que la série transformée est stationnaire. Examine les spécifications, la longueur et les transformations.
