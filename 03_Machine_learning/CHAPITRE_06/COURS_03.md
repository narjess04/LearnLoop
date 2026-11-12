Cours 03 — Modèles statistiques classiques : ETS, ARIMA, SARIMA et ARIMAX
Partie A · 03/07
Niveau : Intermédiaire → Avancé
Durée : 3–4 h

Dans les deux cours précédents, nous avons appris à analyser une série temporelle, construire des baselines et évaluer correctement des prévisions avec une validation chronologique.

Nous allons maintenant découvrir les modèles statistiques de forecasting qui constituent encore aujourd'hui des références importantes dans les applications industrielles.

L'objectif n'est pas seulement de savoir utiliser statsmodels, mais de comprendre comment fonctionnent les modèles, comment choisir leurs paramètres et comment vérifier qu'ils sont fiables.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Comprendre le lissage exponentiel et les modèles ETS.
- Expliquer les composantes AR, MA et I d'ARIMA.
- Construire des modèles ARIMA et SARIMA.
- Ajouter des variables explicatives avec ARIMAX et SARIMAX.
- Sélectionner les ordres \(p,d,q\) et \(P,D,Q,m\).
- Comparer des modèles avec AIC, BIC et validation temporelle.
- Analyser les résidus avec ACF, Ljung-Box et diagnostics statistiques.
- Produire des prévisions accompagnées d'intervalles d'incertitude.


## 2. Vue d'ensemble des modèles statistiques

Deux grandes familles de modèles

Modèle	Idée principale	Cas d'utilisation
SES	Lisser le niveau	Série sans tendance ni saisonnalité
Holt	Lisser niveau et tendance	Série avec tendance
Holt-Winters	Ajouter une saisonnalité	Série avec tendance et saisonnalité
ARIMA	Modéliser les dépendances temporelles	Série non saisonnière
SARIMA	Ajouter des dépendances saisonnières	Série saisonnière
ARIMAX / SARIMAX	Exploiter des variables externes	Prévision avec facteurs explicatifs


## 3. Le lissage exponentiel et les modèles ETS

ETS signifie :
- E — Error : erreur.
- T — Trend : tendance.
- S — Seasonality : saisonnalité.
Les modèles ETS reposent sur l'idée que les observations récentes peuvent avoir davantage d'influence sur les prévisions que les observations anciennes.

## 3.1 Simple Exponential Smoothing (SES)

Le lissage exponentiel simple estime un niveau évolutif :
\[
\boxed{l_t=\alpha y_t+(1-\alpha)l_{t-1}}
\]

Où :
- \(l_t\) : niveau estimé à l'instant \(t\).
- \(y_t\) : observation actuelle.
- \(\alpha\in[0,1]\) : paramètre de lissage.
La prévision à partir de \(t\) est :
\[
\hat y_{t+h|t}=l_t
\]

pour \(h\geq1\).

Intuition :
- \(\alpha\) élevé : le niveau réagit rapidement aux observations récentes.
- \(\alpha\) faible : le niveau évolue plus lentement.
[[smoothing]]

Observations

Niveau lissé

Le lissage réagit lentement et atténue fortement les fluctuations.

Implémentation Python

~~~python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
fit=SimpleExpSmoothing(train,initialization_method='estimated').fit()
forecast=fit.forecast(12)
~~~

Ici, train doit être une série univariée, par exemple df["sales"] sur la période d'entraînement.

## 3.2 Holt's Linear Trend

Le modèle de Holt ajoute une tendance au lissage exponentiel.

Il utilise deux composantes :
\[
l_t=\alpha y_t+(1-\alpha)(l_{t-1}+b_{t-1})
\]

\[
b_t=\beta(l_t-l_{t-1})+(1-\beta)b_{t-1}
\]

La prévision est :
\[
\boxed{\hat y_{t+h|t}=l_t+hb_t}
\]

Où \(b_t\) représente la tendance estimée.

Ce modèle est intéressant lorsqu'une série présente une croissance ou une décroissance relativement régulière.

Il existe également une version à tendance amortie (damped trend) pour éviter de prolonger indéfiniment une croissance linéaire trop importante.

## 3.3 Holt-Winters

Holt-Winters ajoute une composante saisonnière.

Pour une saisonnalité additive :
\[
\hat y_{t+h|t}
=
l_t+hb_t+s_{t-m+((h-1)\bmod m)+1}
\]

Où \(s\) représente les indices saisonniers.

Le modèle peut être :
- Additif lorsque l'amplitude saisonnière est relativement constante.
- Multiplicatif lorsque l'amplitude saisonnière dépend du niveau de la série.

~~~python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
fit=ExponentialSmoothing(train,trend='add',seasonal='add',seasonal_periods=12,initialization_method='estimated').fit()
forecast=fit.forecast(12)
~~~

Dans cet exemple, nous supposons une saisonnalité annuelle avec des observations mensuelles.

Attention : Holt-Winters est une méthode de lissage exponentiel. Le cadre ETS est plus général et spécifie explicitement la structure de l'erreur, de la tendance et de la saisonnalité. Dans statsmodels, ETSModel permet de travailler directement avec cette formulation.


## 4. Comprendre ARIMA

ARIMA signifie AutoRegressive Integrated Moving Average.

Il combine trois mécanismes :
AR
Autorégression
Valeurs passées

I
Intégration
Différenciation

MA
Moyenne mobile
Erreurs passées

\[
ARIMA(p,d,q)
\]


## 4.1 Composante AR(p)

Un modèle autorégressif utilise les valeurs précédentes pour expliquer la valeur actuelle.

Pour AR(1) :
\[
\boxed{Y_t=c+\phi_1Y_{t-1}+\varepsilon_t}
\]

Pour AR(p) :
\[
Y_t=c+\sum_{i=1}^{p}\phi_iY_{t-i}+\varepsilon_t
\]

Où :
- \(p\) : nombre de retards.
- \(\phi_i\) : coefficients autorégressifs.
- \(\varepsilon_t\) : innovation aléatoire.
Exemple :
\[
Y_t=10+0.7Y_{t-1}+\varepsilon_t
\]

Si \(Y_{t-1}=100\), la prévision conditionnelle est :
\[
\hat Y_t=10+0.7(100)=80
\]

Dans cet exemple, on suppose que l'espérance conditionnelle de l'innovation est nulle.

## 4.2 Composante MA(q)

Attention : la composante MA d'ARIMA ne correspond pas à la moyenne mobile utilisée comme baseline au cours précédent.

Ici, elle utilise les innovations passées.

Pour MA(1) :
\[
\boxed{Y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}}
\]

Pour MA(q) :
\[
Y_t=\mu+\varepsilon_t+
\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}
\]

L'idée est que les erreurs ou chocs précédents peuvent influencer les observations actuelles.

## 4.3 Composante I(d)

La composante intégrée correspond au nombre de différenciations non saisonnières.

Pour \(d=1\) :
\[
\Delta Y_t=Y_t-Y_{t-1}
\]

Pour \(d=2\) :
\[
\Delta^2Y_t=Y_t-2Y_{t-1}+Y_{t-2}
\]

Le but est de rendre la série différenciée compatible avec une modélisation ARMA stationnaire.

## 4.4 Formulation générale

En utilisant l'opérateur de retard \(B\), défini par \(BY_t=Y_{t-1}\) :
\[
\boxed{
\phi(B)(1-B)^dY_t=c+\theta(B)\varepsilon_t
}
\]

Avec :
\[
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p
\]

\[
\theta(B)=1+\theta_1B+\cdots+\theta_qB^q
\]

La forme exacte des constantes et tendances dépend de la paramétrisation du modèle.
Exemple : ARIMA(1,1,1)
Cela signifie :
- \(p=1\) : un terme autorégressif.
- \(d=1\) : une différenciation.
- \(q=1\) : un terme de moyenne mobile.
On modélise donc la série différenciée avec un ARMA(1,1).

~~~python
from statsmodels.tsa.arima.model import ARIMA
fit=ARIMA(train,order=(1,1,1)).fit()
print(fit.summary())
forecast=fit.forecast(steps=12)
~~~


## 5. Comment choisir p, d et q ?

C'est une étape essentielle de la modélisation ARIMA.

## 5.1 Choisir d

Le paramètre \(d\) correspond à l'ordre de différenciation.

On peut utiliser :
- L'analyse graphique.
- Les tests ADF et KPSS.
- L'examen de l'ACF.
- La comparaison de modèles.
Une stratégie raisonnable consiste à commencer avec \(d=0\), puis tester \(d=1\) si la série semble posséder une racine unitaire.

La différenciation excessive peut introduire des autocorrélations artificielles et dégrader les prévisions.

## 5.2 Choisir p et q avec ACF/PACF

Pour une série stationnaire :
Comportement	Modèle candidat
PACF coupée après p, ACF décroissante	AR(p)
ACF coupée après q, PACF décroissante	MA(q)
ACF et PACF décroissantes	ARMA(p,q)

Ces signatures sont surtout des propriétés théoriques de modèles idéaux. Sur des données réelles, elles peuvent être difficiles à identifier.

~~~python
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
stationary=train.diff().dropna()
fig,axes=plt.subplots(1,2,figsize=(12,4))
plot_acf(stationary,lags=20,ax=axes[0])
plot_pacf(stationary,lags=20,method='ywm',ax=axes[1])
plt.tight_layout()
plt.show()
~~~


## 5.3 AIC et BIC

Pour comparer des modèles statistiques, on utilise fréquemment :
\[
\boxed{AIC=2k-2\ln(\hat L)}
\]

\[
\boxed{BIC=k\ln(n)-2\ln(\hat L)}
\]

Où :
- \(k\) : nombre de paramètres estimés.
- \(\hat L\) : vraisemblance maximisée.
- \(n\) : nombre d'observations.
Les deux critères cherchent un compromis entre qualité d'ajustement et complexité.

Plus faible est généralement préférable, lorsqu'on compare des modèles ajustés sur des données et des vraisemblances comparables.

Le BIC pénalise plus fortement la complexité lorsque \(n\) est suffisamment grand.

Mais le meilleur AIC ne garantit pas la meilleure performance de prévision sur des données futures.

## 5.4 Recherche des ordres


~~~python
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
results=[]
for p,q in product(range(4),range(4)):
    # d fixé ici pour comparer des vraisemblances sur le même problème.
    try:
        fit=ARIMA(train,order=(p,1,q)).fit()
        if fit.mle_retvals.get('converged',False):
            results.append({'p':p,'d':1,'q':q,'AIC':fit.aic,'BIC':fit.bic})
    except (ValueError,ArithmeticError):
        continue
results_df=pd.DataFrame(results)
if results_df.empty:raise ValueError('Aucun ajustement convergé.')
print(results_df.sort_values('AIC').head(10))
~~~

En pratique, on vérifie aussi la convergence de l'optimisation et la validité des paramètres estimés.

On peut utiliser une recherche automatique comme auto_arima de la bibliothèque pmdarima, mais elle ne remplace ni la validation temporelle ni le diagnostic des résidus.


## 6. SARIMA : ARIMA avec saisonnalité

ARIMA ne représente pas explicitement les dépendances saisonnières dans sa forme standard.

SARIMA ajoute une structure saisonnière.
\[
\boxed{
SARIMA(p,d,q)(P,D,Q)_m
}
\]

Les paramètres sont :
Paramètre	Signification
p	Ordre AR non saisonnier
d	Différenciation non saisonnière
q	Ordre MA non saisonnier
P	Ordre AR saisonnier
D	Différenciation saisonnière
Q	Ordre MA saisonnier
m	Période saisonnière

Exemple
\[
SARIMA(1,1,1)(1,1,1)_{12}
\]

Ce modèle combine des composantes non saisonnières et saisonnières, avec une période de 12 mois.

Sa formulation générale est :
\[
\Phi(B^m)\phi(B)(1-B)^d(1-B^m)^D Y_t
=
\Theta(B^m)\theta(B)\varepsilon_t
\]

La différenciation saisonnière est :
\[
\Delta_mY_t=Y_t-Y_{t-m}
\]

Implémentation Python

~~~python
from statsmodels.tsa.statespace.sarimax import SARIMAX
fit=SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=True,enforce_invertibility=True).fit(disp=False)
forecast=fit.get_forecast(steps=12)
predicted_mean=forecast.predicted_mean
confidence_interval=forecast.conf_int()
~~~

Visualisation des prévisions

~~~python
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(train.index,train,label='Historique')
plt.plot(predicted_mean.index,predicted_mean,label='SARIMA')
plt.fill_between(predicted_mean.index,confidence_interval.iloc[:,0],confidence_interval.iloc[:,1],alpha=.2,label='Intervalle prédictif nominal 95 %')
plt.legend()
plt.show()
~~~

Ces intervalles reposent sur les hypothèses du modèle. Ils ne garantissent pas automatiquement une couverture réelle de 95 %.


## 7. ARIMAX et SARIMAX : ajouter des variables explicatives

Jusqu'ici, les modèles utilisent principalement l'historique de la variable cible.

Mais dans de nombreux problèmes, d'autres variables influencent les prévisions.

Exemples :
- Température pour prévoir la consommation électrique.
- Promotions pour prévoir les ventes.
- Jours fériés pour prévoir le trafic.
- Indicateurs économiques pour prévoir une demande.

## 7.1 Principe d'ARIMAX

Une formulation courante est une régression avec erreurs ARIMA :
\[
\boxed{
Y_t=\beta_0+\beta^\top X_t+u_t
}
\]

où \(u_t\) suit un processus ARIMA.

La distinction est importante : les variables externes expliquent une partie du niveau de \(Y_t\), tandis que le processus ARIMA modélise les dépendances restantes.

## 7.2 SARIMAX

SARIMAX combine les variables explicatives et une structure saisonnière.

~~~python
from statsmodels.tsa.statespace.sarimax import SARIMAX
# train_X et future_X : variables connues à leur origine de prévision.
fit=SARIMAX(train_y,exog=train_X,order=(1,1,1),seasonal_order=(1,1,1,12)).fit(disp=False)
forecast=fit.get_forecast(steps=len(future_X),exog=future_X)
predictions=forecast.predicted_mean
~~~

Point critique : les valeurs de future_X doivent être connues à l'avance ou issues de prévisions construites sans fuite temporelle.

Par exemple, les jours fériés futurs sont connus, mais les ventes futures d'un concurrent ne le sont généralement pas.


## 8. Diagnostic des résidus

Construire un modèle et obtenir une prévision ne suffit pas.

Il faut vérifier que les résidus ne contiennent pas de structure temporelle importante non modélisée.

Pour une prévision à un pas :
\[
e_t=Y_t-\hat Y_{t|t-1}
\]

Dans un modèle correctement spécifié, on recherche notamment des innovations non autocorrélées.

## 8.1 Propriétés recherchées

- Moyenne des résidus proche de zéro.
- Absence d'autocorrélation significative.
- Variance relativement stable, si le modèle la suppose.
- Distribution compatible avec les hypothèses utilisées pour l'inférence et les intervalles.
La normalité n'est pas indispensable pour toutes les méthodes de prévision, mais elle est souvent utilisée pour construire des intervalles paramétriques.

## 8.2 Visualisation des résidus


~~~python
import matplotlib.pyplot as plt
residuals=fit.resid
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(residuals)
axes[0].axhline(0,linestyle='--')
axes[1].hist(residuals,bins=20)
plt.tight_layout()
plt.show()
~~~

Les premiers résidus d'un modèle avec différenciation saisonnière peuvent être influencés par l'initialisation. Il faut en tenir compte dans l'interprétation.

## 8.3 Test de Ljung-Box

Le test de Ljung-Box examine si plusieurs autocorrélations des résidus sont conjointement nulles.

Hypothèses :
\[
H_0:\rho_1=\rho_2=\cdots=\rho_h=0
\]

\[
H_1:\text{au moins une autocorrélation est non nulle}
\]

Sa statistique est :
\[
\boxed{
Q=n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k}
}
\]

Interprétation habituelle :
- p-value < 0,05 : rejet de l'absence d'autocorrélation jusqu'aux retards testés.
- p-value ≥ 0,05 : absence de preuve suffisante pour rejeter cette hypothèse.
Ne pas rejeter \(H_0\) ne prouve pas que les résidus sont indépendants.

~~~python
from statsmodels.stats.diagnostic import acorr_ljungbox
# Adapter model_df aux paramètres AR et MA estimés, y compris saisonniers.
lb=acorr_ljungbox(residuals.dropna(),lags=[6,12,18],model_df=4,return_df=True)
print(lb)
~~~

Pour les résidus d'un ARIMA estimé, le nombre de paramètres AR et MA peut être pris en compte dans les degrés de liberté, par exemple avec model_df=p+q. Le choix doit rester cohérent avec le modèle et le nombre de retards testés.

## 8.4 Autres diagnostics

Diagnostic	Objectif
ACF des résidus	Détecter les dépendances restantes
Ljung-Box	Tester conjointement plusieurs autocorrélations
Histogramme / Q-Q plot	Examiner la distribution
ARCH-LM	Détecter une hétéroscédasticité conditionnelle
Résidus dans le temps	Identifier ruptures et changements de variance

Si les résidus présentent encore une forte saisonnalité, cela peut indiquer que la structure saisonnière du modèle est insuffisante.


## 9. Comparer ETS, ARIMA et SARIMA

Voici un exemple illustratif de comparaison sur un même jeu de validation.
Exemple fictif : MAE sur une période de validation



Valeurs fictives pour illustrer la comparaison, non issues d'un benchmark réel

Une MAE plus faible indique de meilleures performances sur cet exemple. Le classement peut changer selon la série, la période et l'horizon.
Un modèle SARIMAX plus complexe n'est pas nécessairement meilleur qu'un ETS simple.

La sélection doit tenir compte :
1. De la performance hors échantillon.
2. De la stabilité sur plusieurs folds.
3. De la qualité des résidus.
4. De la complexité et du coût de maintenance.
5. De la disponibilité réelle des variables externes.


## 10. Mini-projet : forecasting complet avec SARIMA

Nous allons construire un pipeline simplifié de prévision sur une série synthétique mensuelle.

## Étape 1 — Générer les données


~~~python
import numpy as np
import pandas as pd
from ch06_cours03_lab import dataset,fit_candidate,select_best,residual_model_df,CANDIDATES
series,exog=dataset()
~~~


## Étape 2 — Découpage chronologique


~~~python
train,validation,test=series.iloc[:72],series.iloc[72:84],series.iloc[84:]
print(len(train),len(validation),len(test))
~~~


## Étape 3 — Sélectionner le modèle sur validation

Nous comparons quelques configurations candidates, sans utiliser le test.

~~~python
results=[]
for order,seasonal_order in CANDIDATES:
    try:
        fit=fit_candidate(train,order,seasonal_order)
        prediction=fit.forecast(len(validation))
        results.append({'order':order,'seasonal_order':seasonal_order,
                        'MAE':float(np.abs(validation-prediction).mean()),'converged':True})
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
        print('Configuration écartée :',error)
results_df=pd.DataFrame(results).sort_values('MAE')
~~~


## Étape 4 — Réentraîner sur Train + Validation

Après avoir sélectionné les hyperparamètres, nous utilisons toutes les données précédant le test.

~~~python
best=select_best(results)
train_final=pd.concat([train,validation])
final_fit=fit_candidate(train_final,best['order'],best['seasonal_order'])
forecast_result=final_fit.get_forecast(len(test))
predictions=forecast_result.predicted_mean
intervals=forecast_result.conf_int()
~~~


## Étape 5 — Évaluer sur test


~~~python
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(test,predictions)
rmse=np.sqrt(mean_squared_error(test,predictions))
seasonal_baseline=np.resize(train_final.iloc[-12:].to_numpy(),len(test))
print('SARIMA :',mae,rmse)
print('Baseline MAE :',mean_absolute_error(test,seasonal_baseline))
~~~


## Étape 6 — Visualiser les résultats


~~~python
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(train_final.index,train_final,label='Historique')
plt.plot(test.index,test,label='Réel')
plt.plot(predictions.index,predictions,'--',label='SARIMA')
plt.fill_between(predictions.index,intervals.iloc[:,0],intervals.iloc[:,1],alpha=.2)
plt.legend()
plt.tight_layout()
plt.show()
~~~


## Étape 7 — Diagnostic du modèle final


~~~python
from statsmodels.stats.diagnostic import acorr_ljungbox
burn=max(final_fit.loglikelihood_burn,12)
residuals=final_fit.resid.iloc[burn:].dropna()
model_df=residual_model_df(best['order'],best['seasonal_order'])
print(acorr_ljungbox(residuals,lags=[12,18],model_df=model_df,return_df=True))
final_fit.plot_diagnostics(figsize=(12,8))
plt.tight_layout()
plt.show()
~~~

Le nombre de résidus écartés dépend de l’initialisation du modèle. Le code utilise au moins 12 observations et le burn-in signalé par statsmodels. L’ajustement model_df = p + q + P + Q tient compte des termes saisonniers de la configuration choisie ; les retards testés doivent être supérieurs à cette valeur.

Limite de ce mini-projet : nous avons utilisé une seule période de validation pour sélectionner les hyperparamètres. Pour une évaluation plus robuste, nous pourrions remplacer cette étape par une validation walk-forward.


## 11. Les erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Choisir ARIMA sans analyser la série	Mauvaise spécification	Examiner tendance, saisonnalité et stationnarité
Différencier excessivement	Perte de structure utile	Tester des ordres simples
Choisir uniquement avec AIC	Mauvaise généralisation possible	Valider hors échantillon
Ignorer les résidus	Dépendances non modélisées	ACF et Ljung-Box
Utiliser des variables externes futures inconnues	Fuite temporelle	Vérifier leur disponibilité
Utiliser le test pour sélectionner le modèle	Évaluation biaisée	Garder un test final intact
Supposer que les intervalles sont toujours fiables	Incertitude mal calibrée	Vérifier leur couverture empirique
Utiliser un SARIMA complexe sur très peu de données	Estimation instable	Limiter la complexité et comparer aux baselines

[[quiz]]


## 13. Synthèse du cours

Les points essentiels :
- ETS modélise le niveau, la tendance et la saisonnalité à travers le lissage exponentiel.
- ARIMA combine autorégression, différenciation et innovations passées.
- SARIMA étend ARIMA aux dépendances saisonnières.
- SARIMAX ajoute des variables explicatives externes.
- ACF/PACF, AIC et BIC aident à sélectionner les configurations, mais ne remplacent pas la validation temporelle.
- Le diagnostic des résidus est indispensable pour vérifier les structures non expliquées.
- Les baselines restent nécessaires, même lorsqu'on utilise des modèles statistiques avancés.
Prochain cours
Cours 04 — Forecasting avec le Machine Learning

Partie A · 04/07

Nous verrons comment transformer une série temporelle en dataset supervisé grâce aux lags, rolling features et variables calendaires, puis construire des modèles de prévision avec LightGBM.

Nous comparerons également les stratégies de forecasting récursive, directe et multi-horizon, avec un pipeline Python complet respectant la chronologie des données.


## Corrigés apportés au code fourni

La sélection retient le minimum de MAE parmi les modèles convergés, avant le réentraînement sur train + validation. Les comparaisons d’AIC/BIC ne remplacent pas cette validation. Les intervalles sont prédictifs et leur couverture nominale dépend des hypothèses du modèle.


## Challenge Python

Compléter ses_forecast(y, alpha), select_best(results) et residual_model_df(order, seasonal_order). Exclure les candidats non convergés ou aux scores non finis. Choisir le minimum de MAE sur validation sans dépendre de l’ordre de la liste. Vérifier le lissage et compter les paramètres AR et MA saisonniers.

[[assessment]]

[[interview]]
