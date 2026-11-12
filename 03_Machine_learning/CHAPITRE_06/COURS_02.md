Cours 02 — Baselines et validation temporelle
Partie A · 02/07
Niveau : Fondamental → Avancé
Durée estimée : 2–3 h

Dans le cours précédent, nous avons étudié la structure des séries temporelles : tendance, saisonnalité, stationnarité, différenciation, ACF et PACF.

Maintenant, nous allons aborder une question essentielle :

Comment savoir si un modèle de prévision est réellement performant ?

Avant d'utiliser ARIMA, LightGBM ou des modèles de Deep Learning, il faut construire des modèles de référence (baselines) et définir une stratégie de validation adaptée au temps.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Construire des baselines naïves, saisonnières et par moyenne mobile.
- Comprendre les prévisions à un pas et à plusieurs horizons.
- Appliquer un découpage chronologique Train / Validation / Test.
- Implémenter une validation walk-forward avec fenêtre croissante ou glissante.
- Identifier les principales sources de fuite temporelle (temporal data leakage).
- Évaluer des prévisions avec MAE, RMSE, MASE et sMAPE.
- Comparer correctement plusieurs modèles de forecasting.


## 2. Comprendre le problème de forecasting

Le forecasting consiste à prédire une ou plusieurs valeurs futures à partir d'informations disponibles jusqu'à un instant donné.

Mathématiquement :
\[
\hat{Y}_{t+h}=f(Y_t,Y_{t-1},\ldots,Y_{t-p},X_t)
\]

Où :
- \(\hat{Y}_{t+h}\) : valeur prédite à l'horizon \(h\).
- \(Y_t\) : dernière valeur observée.
- \(p\) : nombre de retards utilisés.
- \(X_t\) : variables explicatives connues à l'instant de prévision.
Attention : une variable explicative future ne peut être utilisée que si elle est réellement connue ou prévue au moment où la prédiction est produite.

## 2.1 Prévision à un pas (One-step ahead)

On prédit uniquement l'observation suivante :
\[
\hat{Y}_{t+1}=f(Y_t,Y_{t-1},\ldots)
\]

Exemple : nous sommes lundi et nous prédisons les ventes de mardi.

## 2.2 Prévision multi-horizon (Multi-step ahead)

On prédit plusieurs instants futurs :
\[
\hat{Y}_{t+1},\hat{Y}_{t+2},\ldots,\hat{Y}_{t+H}
\]

Exemple : nous sommes dimanche et nous prédisons les ventes de lundi jusqu'à dimanche prochain.
Illustration : prévision sur quatre périodes
Les valeurs grises sont observées. Les valeurs vertes sont prédites sans utiliser les observations futures.
Cette distinction est importante : une baseline à un pas ne doit pas être évaluée comme une baseline multi-horizon.


## 3. Les modèles de référence (Baselines)

Une baseline est un modèle simple qui sert de point de comparaison.

Elle permet de répondre à une question :

Notre modèle complexe apporte-t-il une réelle amélioration par rapport à une méthode simple ?

Un modèle LightGBM obtenant une MAE de 15 n'est pas nécessairement intéressant si une baseline naïve obtient une MAE de 10.

## 3.1 Baseline naïve (Naive Forecast)

Le principe est simple : la prochaine valeur sera identique à la dernière valeur observée.
\[
\boxed{\hat{Y}_{t+h}=Y_t}
\]

Pour tous les horizons \(h\geq1\), nous conservons la dernière observation connue à l'origine \(t\).

Exemple :
Jour	Ventes
Lundi	100
Mardi	110
Mercredi	105
Jeudi	120
Vendredi	115
Samedi (prévision)	115

Cette baseline est particulièrement utile pour les séries présentant une forte persistance.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## 3.2 Baseline naïve saisonnière (Seasonal Naive)

La méthode naïve saisonnière suppose que le comportement futur reproduit celui de la dernière saison observée.

Pour une prévision à un pas :
\[
\hat{Y}_{t+1}=Y_{t+1-m}
\]

Où \(m\) est la période saisonnière.

Pour une prévision multi-horizon, on répète les dernières \(m\) observations disponibles.

Exemple : si les ventes suivent un cycle hebdomadaire, on peut prédire les ventes du prochain lundi avec les ventes du lundi précédent.
Exemple : saisonnalité hebdomadaire

Semaine observée

Prévision semaine suivante


## 0 unités55 unités110 unités165 unités220 unitésLunMarMerJeuVenSamDim


Données fictives

La prévision reproduit les valeurs de la semaine précédente.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Pour un horizon supérieur à la période saisonnière, les mêmes valeurs sont répétées.

## 3.3 Baseline moyenne mobile (Moving Average)

La moyenne mobile consiste à calculer la moyenne des \(k\) dernières observations.
\[
\hat{Y}_{t+1}=\frac{1}{k}\sum_{i=0}^{k-1}Y_{t-i}
\]

Exemple avec \(k=3\) :
\[
Y=[100,110,120,130,140]
\]

\[
\hat{Y}_{t+1}=\frac{120+130+140}{3}=130
\]

sales = pd.Series([100, 110, 120, 130, 140])window = 3forecast = sales.tail(window).mean()print("Prévision :", forecast)# 130.0

Pour plusieurs horizons, deux possibilités existent : conserver la même moyenne pour tous les horizons ou recalculer récursivement la moyenne à partir des prévisions produites.

Ces deux méthodes ne donnent pas nécessairement les mêmes résultats. Il faut donc préciser la stratégie retenue.

## 3.4 Baseline avec dérive (Drift Method)

Une autre baseline utile prolonge la tendance moyenne entre la première et la dernière observation.
\[
\hat{Y}_{t+h}
=
Y_t+h\frac{Y_t-Y_1}{t-1}
\]

Elle est pertinente comme référence supplémentaire lorsque la série présente une tendance approximativement linéaire.
Comparaison des baselines
Baseline	Principe	Situation adaptée
Naïve	Répéter la dernière valeur	Forte persistance
Saisonnière	Répéter la dernière saison	Saisonnalité régulière
Moyenne mobile	Moyenne des dernières valeurs	Série relativement stable
Drift	Prolonger la tendance moyenne	Tendance approximativement linéaire


## 4. Pourquoi le découpage aléatoire est problématique

Dans le Machine Learning classique, nous utilisons souvent :

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Par défaut, cette fonction mélange les observations.

Dans les séries temporelles, ce mélange peut entraîner une évaluation irréaliste : le modèle risque d'être entraîné sur des observations postérieures à celles qu'il doit prédire.

## 4.1 Train / Validation / Test chronologique

Exemple : découpage 70 % / 15 % / 15 %
Train 70 %

Val 15 %

Test 15 %

PasséFutur 

- Train : apprentissage des paramètres du modèle.
- Validation : sélection du modèle et réglage des hyperparamètres.
- Test : évaluation finale sur des données non utilisées pour les choix précédents.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Le découpage doit aussi tenir compte du nombre de cycles saisonniers et de l'horizon de prévision.

Par exemple, un test couvrant seulement deux semaines est insuffisant pour évaluer correctement une saisonnalité annuelle.


## 5. Walk-forward validation

Une seule séparation Train / Test peut donner une estimation trop dépendante d'une période particulière.

La walk-forward validation, également appelée rolling-origin evaluation, consiste à évaluer le modèle sur plusieurs origines de prévision successives.

À chaque étape :

## 1. On entraîne ou met à jour le modèle avec les données autorisées.


## 2. On produit une prévision pour une période future.


## 3. On compare les prédictions aux observations réellement obtenues.


## 4. On avance l'origine de prévision.


## 5.1 Fenêtre croissante (Expanding Window)

Dans cette stratégie, le volume des données d'entraînement augmente progressivement.
[[timefolds]]

Visualisation des folds
Fold 1

Fold 2

Fold 3

Fold 4

Train

Validation

Futur non utilisé

Avantages : davantage de données historiques sont disponibles à mesure que le temps avance.

Inconvénients : coût d'entraînement croissant et risque d'accorder trop d'importance à des observations anciennes devenues moins représentatives.

## 5.2 Fenêtre glissante (Sliding Window)

Ici, la taille de l'ensemble d'entraînement reste constante.
Fenêtre fixe de quatre périodes
Fold 1

Fold 2

Fold 3

Fold 4

À chaque fold, la fenêtre d'entraînement se décale vers les observations les plus récentes.
Cette stratégie peut être utile lorsque les comportements évoluent rapidement.

Elle permet d'oublier les observations anciennes, mais réduit la quantité de données disponibles pour chaque entraînement.

## 5.3 Implémentation avec TimeSeriesSplit

TimeSeriesSplit de Scikit-learn facilite la validation chronologique.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Par défaut, cette approche utilise une fenêtre d'entraînement croissante.

Pour limiter sa taille :

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Cela permet de construire une fenêtre glissante de 24 observations au maximum.

Attention : TimeSeriesSplit découpe les observations selon leurs positions. Il suppose notamment que les observations sont régulièrement espacées pour rendre les durées des folds comparables.

Pour les prévisions multi-horizon, il faut également vérifier que chaque fold respecte exactement l'horizon et les informations disponibles à son origine.


## 6. Temporal Data Leakage — La fuite temporelle

La fuite temporelle se produit lorsqu'un modèle utilise, pendant l'entraînement ou la prédiction, des informations qui n'auraient pas été disponibles au moment réel de la prévision.

C'est l'une des erreurs les plus importantes en forecasting.

## 6.1 Exemple : moyenne mobile incorrecte

Supposons que nous voulons prédire les ventes du jour \(t\).

Ce code est problématique :

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

La moyenne inclut les ventes du jour \(t\), qui constituent précisément la cible à prédire.

La bonne approche pour une prévision à un pas est :

~~~python
df["rolling_mean_7"] = df["ventes"].shift(1).rolling(7).mean()
~~~


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Le shift(1) garantit que la moyenne ne contient que les observations antérieures à \(t\).

Cette correction est valable pour une prédiction à un pas avec observations passées disponibles. Pour une prévision multi-horizon, les features doivent être construites en fonction de l'origine de prévision.

## 6.2 Normalisation incorrecte

Incorrect

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Le scaler a appris la moyenne et l'écart-type sur toute la série, y compris les périodes futures.

Correct

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Pour chaque fold de validation, le scaler doit être ajusté uniquement sur son ensemble d'entraînement.

## 6.3 Autres sources de fuite temporelle

Source	Risque	Solution
Split aléatoire	Mélange passé et futur	Split chronologique
Rolling sans décalage	Utilise la cible actuelle	Shift adapté à l'horizon
Scaler global	Statistiques futures	Fit sur train uniquement
Variable future inconnue	Information indisponible	Vérifier la date de disponibilité
Feature calculée après événement	Fuite indirecte de la cible	Auditer les timestamps
Tuning sur le test	Suradaptation à la période finale	Validation distincte

Une règle importante :
Pour chaque feature, il faut pouvoir répondre à la question : « Cette information était-elle réellement disponible au moment de la prévision ? »


## 7. Les métriques de forecasting

Nous avons besoin de métriques permettant de comparer les prédictions aux observations réelles.

Soit :
- \(y_t\) : valeur réelle.
- \(\hat y_t\) : valeur prédite.
- \(n\) : nombre de prédictions évaluées.

## 7.1 MAE — Mean Absolute Error

\[
\boxed{MAE=\frac{1}{n}\sum_{t=1}^{n}|y_t-\hat y_t|}
\]

La MAE mesure l'erreur absolue moyenne.

Exemple :
Réel	Prédit	Erreur absolue
100	110	10
120	115	5
140	130	10

\[
MAE=\frac{10+5+10}{3}=8.33
\]

Avantages : interprétation facile, mêmes unités que la cible, sensibilité moins forte aux grandes erreurs que la RMSE.

## 7.2 RMSE — Root Mean Squared Error

\[
\boxed{RMSE=\sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat y_t)^2}}
\]

La RMSE pénalise davantage les grandes erreurs grâce à l'élévation au carré.

Dans l'exemple précédent :
\[
RMSE=\sqrt{\frac{100+25+100}{3}}
=\sqrt{75}\approx8.66
\]

On utilise souvent la RMSE lorsque les grandes erreurs sont particulièrement coûteuses.

## 7.3 MASE — Mean Absolute Scaled Error

La MASE est particulièrement intéressante pour comparer des séries de différentes échelles.

Elle rapporte la MAE du modèle à l'erreur moyenne d'une baseline naïve calculée sur l'ensemble d'entraînement.

Pour une série non saisonnière :
\[
\boxed{
MASE=
\frac{
\frac{1}{H}\sum_{h=1}^{H}|y_{T+h}-\hat y_{T+h}|
}{
\frac{1}{T-1}\sum_{t=2}^{T}|y_t-y_{t-1}|
}
}
\]

Pour une série saisonnière de période \(m\), le dénominateur peut être remplacé par :
\[
\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|
\]

Interprétation :
- MASE < 1 : meilleure MAE que l'échelle d'erreur de la baseline naïve d'entraînement.
- MASE = 1 : erreur égale à cette référence.
- MASE > 1 : erreur supérieure à cette référence.
Nuance importante : MASE < 1 ne garantit pas que le modèle surpasse la baseline naïve sur la période de test. La référence du dénominateur provient du train.

Si cette référence est nulle, la MASE est indéfinie.

## 7.4 sMAPE — Symmetric Mean Absolute Percentage Error

Une définition courante est :
\[
\boxed{
sMAPE=
\frac{100}{n}\sum_{t=1}^{n}
\frac{2|y_t-\hat y_t|}
{|y_t|+|\hat y_t|}
}
\]

La sMAPE exprime une erreur relative en pourcentage.

Avec cette convention, elle est comprise entre 0 % et 200 % pour les termes définis.

Cependant, elle peut devenir instable lorsque les valeurs réelles et prédites sont proches de zéro. Le cas où les deux sont nuls nécessite une convention explicite.

Le terme symmetric ne signifie pas qu'elle est exempte de biais.

## 7.5 Comparaison des métriques

Métrique	Unité	Point fort	Limite
MAE	Unité cible	Simple et interprétable	Pénalise linéairement les erreurs
RMSE	Unité cible	Sensible aux grandes erreurs	Sensible aux outliers
MASE	Sans unité	Comparaison entre séries	Dénominateur parfois nul
sMAPE	%	Erreur relative	Instable près de zéro


## 8. Implémentation Python des métriques


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Cette fonction suppose des vecteurs alignés, de même longueur, et un ensemble d'entraînement contenant plus de \(m\) observations.


## 9. Mini-projet : comparer les baselines sur des ventes mensuelles

Nous allons réaliser une expérience complète.

Objectif : prévoir les ventes des 12 prochains mois à partir de 48 mois d'historique.

Nous allons comparer trois baselines :

## 1. Naïve.


## 2. Naïve saisonnière.


## 3. Moyenne mobile.


## Étape 1 — Générer les données


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 2 — Découpage chronologique


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 3 — Construire les baselines


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Il s'agit bien d'une prévision produite une seule fois à la fin du train pour les 12 mois futurs. Aucune observation du test n'est utilisée pour actualiser les baselines.

## Étape 4 — Évaluer les résultats


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 5 — Visualiser les prévisions


Code correspondant : voir le Coding Lab complet et exécutable en fin de support.


## Étape 6 — Interpréter

Questions d'analyse
- Quelle baseline obtient la MAE la plus faible ?
- La baseline saisonnière capture-t-elle correctement les fluctuations annuelles ?
- Pourquoi une moyenne mobile constante peut-elle mal représenter une tendance croissante ?
- Quelle baseline choisirais-tu comme référence pour un futur modèle SARIMA ou LightGBM ?
Les résultats exacts seront calculés à l'exécution du code. Il ne faut pas supposer à l'avance que la baseline saisonnière sera toujours la meilleure.


## 10. Aller plus loin : évaluer avec un véritable walk-forward

Dans le mini-projet précédent, nous avons évalué une seule origine de prévision.

Pour obtenir une estimation plus robuste, nous pouvons répéter l'expérience sur plusieurs origines.

Code correspondant : voir le Coding Lab complet et exécutable en fin de support.

Chaque fold correspond ici à une prévision de six mois produite à une origine donnée.

Les performances peuvent varier selon la période évaluée. C'est précisément l'intérêt du walk-forward.

Pour comparer plusieurs modèles, il faut conserver les mêmes folds, horizons et règles de disponibilité des données.


## 11. Bonnes pratiques professionnelles

Voici les principes à appliquer dans un véritable projet de forecasting.
Pratique	Pourquoi ?
Définir l'horizon avant la modélisation	Une prévision à J+1 diffère d'une prévision à J+30
Commencer par des baselines	Vérifier la valeur ajoutée du modèle
Respecter les dates	Éviter la fuite temporelle
Utiliser plusieurs origines de validation	Réduire la dépendance à une seule période
Ajuster les transformations sur train	Éviter l'utilisation d'informations futures
Évaluer chaque horizon si nécessaire	Les erreurs peuvent augmenter avec l'horizon
Conserver un test final intact	Obtenir une évaluation indépendante
Comparer les métriques adaptées au métier	Une erreur coûteuse n'est pas toujours bien représentée par la MAE

Exemple métier
Imaginons un système de prévision de la demande pour un supermarché.

Une sous-estimation importante peut provoquer une rupture de stock, tandis qu'une surestimation peut générer des invendus.

Deux modèles avec la même MAE peuvent donc avoir des conséquences économiques très différentes.

C'est pourquoi l'évaluation statistique doit parfois être complétée par une analyse des coûts métier.

[[quiz]]


## 13. Synthèse du cours

Les cinq idées essentielles à retenir :

## 1. Une baseline est indispensable. Un modèle complexe doit démontrer sa valeur ajoutée.


## 2. Le temps doit être respecté. Les découpages aléatoires sont généralement inadaptés à une évaluation de forecasting.


## 3. Le walk-forward reproduit mieux les conditions réelles. Il permet de tester plusieurs origines de prévision.


## 4. La fuite temporelle peut invalider toute une expérimentation. Les features et transformations doivent respecter leur date de disponibilité.


## 5. Les métriques sont complémentaires. MAE, RMSE, MASE et sMAPE ne mesurent pas exactement les mêmes aspects de la performance.

Prochain cours
Cours 03 — Modèles statistiques classiques

Partie A · 03/07

Nous étudierons les modèles ETS, ARIMA, SARIMA et ARIMAX, la sélection des ordres \(p,d,q\), les composantes saisonnières, ainsi que les diagnostics des résidus avec notamment le test de Ljung-Box.


## Coding Lab complet

Les blocs aplatis de la pièce jointe sont regroupés ici dans un script autonome. La fonction de métriques tronquée du cours 02 est complétée.

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def dataset():
    rng = np.random.default_rng(42)
    t = np.arange(60)
    sales = 100 + 1.2*t + 25*np.sin(2*np.pi*t/12) + rng.normal(0, 5, 60)
    return pd.DataFrame({'sales': sales}, index=pd.date_range('2021-01-01', periods=60, freq='MS'))

def baseline_forecast(history, horizon, period=12, window=3):
    history = np.asarray(history, dtype=float)
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (horizon, period, window)):
        raise ValueError('Horizon, période et fenêtre entiers positifs requis.')
    if history.ndim != 1 or len(history) < max(period, window, 2) or not np.isfinite(history).all():
        raise ValueError('Historique fini et suffisamment long requis.')
    return {'Naive': np.repeat(history[-1], horizon),
            'Seasonal Naive': np.resize(history[-period:], horizon),
            'Moving Average': np.repeat(history[-window:].mean(), horizon),
            'Drift': history[-1]+np.arange(1,horizon+1)*(history[-1]-history[0])/(len(history)-1)}

def forecasting_metrics(y_true, y_pred, y_train, seasonality=1):
    y_true, y_pred, y_train = [np.asarray(v, dtype=float) for v in (y_true, y_pred, y_train)]
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in (y_true, y_pred, y_train)):
        raise ValueError('Vecteurs finis à une dimension requis.')
    if y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError('Cibles et prédictions non vides et alignées requises.')
    if not isinstance(seasonality, (int, np.integer)) or seasonality < 1 or len(y_train) <= seasonality:
        raise ValueError('Période entière positive et train plus long que cette période requis.')
    error = y_true-y_pred
    mae = np.abs(error).mean()
    scale = np.abs(y_train[seasonality:]-y_train[:-seasonality]).mean()
    denominator = np.abs(y_true)+np.abs(y_pred)
    terms = np.divide(2*np.abs(error), denominator, out=np.zeros_like(error), where=denominator!=0)
    return {'MAE': float(mae), 'RMSE': float(np.sqrt(np.mean(error**2))),
            'MASE': float(mae/scale) if scale>0 else np.nan, 'sMAPE': float(100*terms.mean())}

def run():
    df = dataset()
    train, test = df.iloc[:48], df.iloc[48:]
    predictions = baseline_forecast(train['sales'], len(test))
    rows = [{'Model': name, **forecasting_metrics(test['sales'], pred, train['sales'], 12)}
            for name, pred in predictions.items()]
    print('Audit à origine fixe, 12 mois :')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train.index, train['sales'], label='Train')
    ax.plot(test.index, test['sales'], label='Réel', linewidth=2)
    for name, pred in predictions.items():
        ax.plot(test.index, pred, '--', label=name)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Développement : aucun fold ne touche les 12 mois du test final.
    rows = []
    for strategy, size in [('Expanding', None), ('Sliding', 24)]:
        cv = TimeSeriesSplit(n_splits=4, test_size=6, max_train_size=size)
        for fold, (tr, va) in enumerate(cv.split(train), 1):
            assert tr[-1]<va[0]
            history, actual = train['sales'].iloc[tr], train['sales'].iloc[va]
            for name, pred in baseline_forecast(history, len(actual)).items():
                rows.append({'strategy': strategy, 'fold': fold, 'Model': name,
                             **forecasting_metrics(actual, pred, history, 12)})
    print('Walk-forward sur développement uniquement :')
    print(pd.DataFrame(rows).groupby(['strategy','Model'])['MAE'].agg(['mean','std']).round(3))

if __name__ == '__main__':
    run()
~~~

## Challenge Python

Implémenter baseline_forecast(history, horizon, period=12, window=3) et forecasting_metrics(y_true, y_pred, y_train, seasonality=1). Comparer quatre baselines à origine fixe, puis sur les mêmes folds chronologiques.

[[assessment]]

[[interview]]
