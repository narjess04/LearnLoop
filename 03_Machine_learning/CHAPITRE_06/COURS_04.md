Cours 04 — Forecasting avec le Machine Learning
Partie A · 04/07
Niveau : Intermédiaire → Avancé
Durée : 3–4 h

Dans le cours précédent, nous avons étudié les modèles statistiques ETS, ARIMA, SARIMA et SARIMAX.

Nous allons maintenant découvrir une autre approche : transformer une série temporelle en un problème de Machine Learning supervisé, afin d'utiliser des algorithmes comme Random Forest, XGBoost, LightGBM et CatBoost pour prédire des valeurs futures.

L'enjeu principal n'est pas simplement d'entraîner un modèle, mais de construire correctement les variables temporelles, choisir une stratégie de prévision et éviter toute fuite d'information future.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Transformer une série temporelle en dataset supervisé.
- Construire des variables retardées (lag features).
- Créer des statistiques glissantes (rolling features).
- Exploiter les variables calendaires et les variables externes.
- Comprendre les stratégies récursive, directe et multi-output.
- Entraîner un modèle LightGBM pour le forecasting.
- Mettre en place une validation temporelle sans fuite de données.
- Comparer un modèle ML à des baselines et à des modèles statistiques.


## 2. Pourquoi utiliser le Machine Learning pour les séries temporelles ?

Les modèles statistiques classiques reposent généralement sur une structure mathématique explicite des dépendances temporelles.

Le Machine Learning permet d'apprendre des relations potentiellement non linéaires entre la cible, son historique et des variables explicatives.

Exemple : pour prédire les ventes d'un magasin, nous pouvons utiliser simultanément :
- Les ventes des jours précédents.
- Les moyennes des sept et trente derniers jours.
- Le jour de la semaine.
- Le mois de l'année.
- Les promotions planifiées.
- Les jours fériés.
- Les prix connus au moment de la prévision.
Le modèle apprend une fonction :
\[
\boxed{
\hat y_t=f(X_t)
}
\]

où \(X_t\) contient des informations disponibles avant la réalisation de \(y_t\).
Comparaison des approches
Modèles statistiques	Machine Learning
ARIMA, SARIMA, ETS	LightGBM, XGBoost, Random Forest
Structure temporelle explicite	Structure temporelle représentée par les features
Hypothèses statistiques spécifiques	Relations non linéaires possibles
Souvent adaptés aux petites séries	Peut bénéficier de davantage de données
Interprétation des paramètres	Importance des variables et SHAP
Intervalles possibles selon le modèle	Incertitude à construire explicitement

Important : le Machine Learning n'est pas automatiquement supérieur aux modèles statistiques. Sur certaines séries simples ou courtes, une baseline saisonnière ou un modèle ETS peut être plus performant.

## 3. Transformer une série temporelle en dataset supervisé

Un algorithme de régression classique attend une matrice de variables explicatives \(X\) et une cible \(y\).

Or, une série temporelle est initialement une succession d'observations.

Nous devons donc construire des exemples d'entraînement à partir de l'historique.

## 3.1 Principe des fenêtres temporelles

Supposons les ventes suivantes :
\[
Y=[100,110,105,120,125,130,140]
\]

Nous voulons prédire chaque valeur à partir des trois observations précédentes.
Lag 3	Lag 2	Lag 1	Cible
100	110	105	120
110	105	120	125
105	120	125	130
120	125	130	140

Chaque ligne devient un exemple supervisé.
[[lagwindow]]

Mathématiquement :
\[
X_t=[Y_{t-1},Y_{t-2},\ldots,Y_{t-p}]
\]

\[
\hat Y_t=f(X_t)
\]

Le nombre \(p\) de retards est un hyperparamètre à sélectionner sur les données de validation.
Implémentation Python

~~~python
import pandas as pd
df=pd.DataFrame({'sales':[100,110,105,120,125,130,140]})
for lag in [1,2,3]:
    df[f'lag_{lag}']=df['sales'].shift(lag)
print(df.dropna())
~~~

Cette transformation constitue la base du forecasting supervisé.


## 4. Feature Engineering temporel

La qualité des variables temporelles peut avoir une influence considérable sur les performances d'un modèle ML.

Nous allons distinguer quatre grandes familles de features.

## 4.1 Lag features

Une variable retardée correspond à une valeur passée de la cible.
\[
Lag_k(t)=Y_{t-k}
\]

Pour une série journalière, nous pouvons construire :
Feature	Signification
lag_1	Valeur de la veille
lag_7	Valeur du même jour la semaine précédente
lag_14	Valeur deux semaines auparavant
lag_28	Valeur quatre semaines auparavant

~~~python
lags=[1,7,14,28]
for lag in lags:
    df[f'lag_{lag}']=df['sales'].shift(lag)
~~~

Il faut sélectionner les lags en fonction de la fréquence, de la saisonnalité et de l'horizon de prévision.

Par exemple, pour une série mensuelle, lag_12 peut capturer une relation annuelle.

## 4.2 Rolling features

Les statistiques glissantes résument le comportement récent de la série.

Exemples :
- Moyenne mobile.
- Écart-type mobile.
- Minimum et maximum récents.
- Médiane glissante.
Pour une moyenne mobile de taille \(w\) :
\[
\boxed{
RM_w(t)=\frac{1}{w}\sum_{i=1}^{w}Y_{t-i}
}
\]

Nous utilisons uniquement les observations antérieures à \(t\).

~~~python
past=df['sales'].shift(1)
df['rolling_mean_7']=past.rolling(7).mean()
df['rolling_std_7']=past.rolling(7).std()
df['rolling_max_14']=past.rolling(14).max()
~~~

Le shift(1) est essentiel pour une prévision à un pas : sans lui, la statistique inclurait la cible actuelle.

Pour une prévision multi-horizon produite à l'instant \(T\), il faut aller plus loin : aucune feature ne doit utiliser une observation réelle postérieure à \(T\).

## 4.3 Variables calendaires

Les variables calendaires permettent au modèle de reconnaître des comportements récurrents.

~~~python
df['day_of_week']=df.index.dayofweek
df['day_of_month']=df.index.day
df['month']=df.index.month
df['quarter']=df.index.quarter
df['is_weekend']=(df.index.dayofweek>=5).astype(int)
~~~

Ces variables sont particulièrement utiles lorsque le comportement dépend du calendrier.

Exemples :
- Les ventes peuvent être plus élevées le samedi.
- La consommation électrique peut varier selon le jour.
- Le trafic peut augmenter pendant certaines périodes de vacances.

## 4.4 Encodage cyclique

Les variables comme le mois ou le jour de la semaine sont cycliques.

Par exemple, décembre et janvier sont proches dans le calendrier, même si leurs valeurs numériques sont 12 et 1.

On peut représenter cette circularité avec sinus et cosinus :
\[
Month_{\sin}=\sin\left(\frac{2\pi m}{12}\right)
\]

\[
Month_{\cos}=\cos\left(\frac{2\pi m}{12}\right)
\]

~~~python
import numpy as np
df['month_sin']=np.sin(2*np.pi*df.index.month/12)
df['month_cos']=np.cos(2*np.pi*df.index.month/12)
~~~

Cet encodage est souvent utile pour les modèles sensibles aux distances entre valeurs. Pour les arbres de décision, il peut également être testé, mais n'est pas systématiquement nécessaire.
Représentation cyclique des 12 mois
La représentation circulaire conserve la proximité entre décembre et janvier.

## 4.5 Variables externes (Exogenous Features)

Les variables externes peuvent enrichir le modèle.

Exemple pour les ventes :
\[
\hat Y_t=f(
Lag_1,Lag_7,RM_7,
Promotion_t,Prix_t,Jour_t
)
\]

Mais leur disponibilité doit être vérifiée.
Variable	Utilisable pour le futur ?
Jour de la semaine	Oui
Jour férié officiel	Oui, si calendrier connu
Promotion planifiée	Oui, si planification disponible
Température future réelle	Non
Prévision météo disponible à l'origine	Oui
Ventes futures réelles	Non


## 5. Les stratégies de forecasting multi-horizon

Nous voulons maintenant prédire plusieurs valeurs futures.

Soit un horizon de prévision \(H=3\).

Il existe différentes stratégies.

## 5.1 Stratégie récursive (Recursive Forecasting)

On entraîne un seul modèle pour prédire une période à l'avance.

Puis, on réutilise ses propres prédictions pour construire les features des périodes suivantes.
Principe récursif

Mathématiquement :
\[
\hat Y_{T+1}=f(X_{T+1})
\]

\[
\hat Y_{T+2}=f(\hat Y_{T+1},Y_T,\ldots)
\]

\[
\hat Y_{T+3}=f(\hat Y_{T+2},\hat Y_{T+1},\ldots)
\]

Avantages : un seul modèle, entraînement simple, horizon flexible.

Inconvénient : les erreurs peuvent se propager et s'amplifier à chaque étape.

## 5.2 Stratégie directe (Direct Forecasting)

On entraîne un modèle différent pour chaque horizon.
\[
\hat Y_{T+1}=f_1(X_T)
\]

\[
\hat Y_{T+2}=f_2(X_T)
\]

\[
\hat Y_{T+3}=f_3(X_T)
\]

Chaque modèle apprend à prédire directement son horizon, sans réutiliser les prédictions précédentes.

Avantages : pas de propagation récursive des erreurs, spécialisation par horizon.

Inconvénients : davantage de modèles à entraîner et à maintenir.

## 5.3 Stratégie multi-output

Un seul modèle prédit simultanément plusieurs horizons.
\[
\boxed{
f(X_T)=
[\hat Y_{T+1},\hat Y_{T+2},\ldots,\hat Y_{T+H}]
}
\]

Exemple :

~~~python
predictions=model.predict(X_future)
# Avec un estimateur multi-sortie : une ligne peut contenir H prévisions.
~~~

Certains modèles prennent en charge directement plusieurs sorties, tandis que d'autres nécessitent une adaptation, par exemple MultiOutputRegressor.

## 5.4 Comparaison des stratégies

Stratégie	Nombre de modèles	Principal avantage	Limite
Récursive	1	Simple et flexible	Propagation des erreurs
Directe	H	Spécifique à chaque horizon	Coût d'entraînement
Multi-output	1 modèle multi-sortie	Prédit tous les horizons ensemble	Complexité et support algorithmique variables

Le choix dépend de l'horizon, de la quantité de données et du comportement de la série.


## 6. LightGBM pour le forecasting

LightGBM est un algorithme de Gradient Boosting basé sur des arbres de décision.

Il est particulièrement adapté aux données tabulaires et peut apprendre des interactions non linéaires entre les variables.

Pour le forecasting, nous ne lui transmettons pas directement une série temporelle brute.

Nous construisons d'abord un dataset supervisé contenant les features temporelles.

## 6.1 Architecture du pipeline



## 6.2 Installation


~~~shell
python -m pip install lightgbm pandas numpy scikit-learn matplotlib
~~~


## 6.3 Création des features


~~~python
import numpy as np
import pandas as pd
from ch06_cours04_lab import create_features,create_features_for_future,FEATURES
# Le fichier lab téléchargeable définit les deux fonctions complètes.
~~~

Cette fonction convient à la construction d'exemples historiques pour un modèle à un pas.

Elle ne doit pas être appliquée naïvement à un dataset contenant des dates futures dont les ventes sont inconnues.


## 7. Mini-projet complet : prévision des ventes avec LightGBM

Nous allons construire un modèle capable de prédire les ventes journalières.

## Étape 1 — Générer un dataset


~~~python
from ch06_cours04_lab import dataset
df=dataset()
~~~


## Étape 2 — Construire les features


~~~python
data=create_features(df)
feature_cols=FEATURES
X,y=data[feature_cols],data['sales']
~~~


## Étape 3 — Split chronologique


~~~python
train_end,val_end=int(len(data)*.70),int(len(data)*.85)
X_train,X_val,X_test=X.iloc[:train_end],X.iloc[train_end:val_end],X.iloc[val_end:]
y_train,y_val,y_test=y.iloc[:train_end],y.iloc[train_end:val_end],y.iloc[val_end:]
~~~


## Étape 4 — Entraîner LightGBM


~~~python
import lightgbm as lgb
model=lgb.LGBMRegressor(n_estimators=500,learning_rate=.03,num_leaves=31,random_state=42,verbosity=-1,n_jobs=1)
model.fit(X_train,y_train,eval_set=[(X_val,y_val)],eval_metric='l1',callbacks=[lgb.early_stopping(30,verbose=False)])
~~~

L'early stopping utilise les performances de validation pour limiter le nombre d'itérations.

Il est important de ne pas utiliser le jeu de test pour choisir les hyperparamètres ou arrêter l'entraînement.

## Étape 5 — Évaluer le modèle


~~~python
from sklearn.metrics import mean_absolute_error,mean_squared_error
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('MAE / RMSE :',mae,rmse)
~~~

Attention à l'interprétation : ce test correspond à des prévisions à un pas avec mise à disposition des observations passées au fil du temps. Les features du test contiennent des valeurs réelles antérieures à chaque date prédite.

Il ne mesure donc pas la performance d'une prévision de toute la période de test produite en une seule fois à sa date de début.

## Étape 6 — Comparer à la baseline saisonnière


~~~python
# Historique brut : éviter la perte artificielle de 28 jours liée au dropna.
baseline_pred=df['sales'].shift(7).loc[y_test.index]
print('ML MAE :',mae,'Baseline MAE :',mean_absolute_error(y_test,baseline_pred))
~~~

Les deux méthodes utilisent ici le même protocole d'évaluation à un pas.

## Étape 7 — Visualiser les prédictions


~~~python
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(y_test.index,y_test,label='Réel')
plt.plot(y_test.index,y_pred,label='LightGBM',alpha=.8)
plt.legend()
plt.tight_layout()
plt.show()
~~~


## 8. Implémenter une prévision récursive sur plusieurs jours

Nous allons maintenant produire des prévisions sur plusieurs jours sans utiliser les valeurs futures réelles.

L'idée est de reconstruire les features à chaque étape, puis d'ajouter la prédiction à l'historique.

~~~python
def recursive_forecast(model,history,horizon,feature_cols=FEATURES):
    if not isinstance(horizon,(int,np.integer)) or horizon<1:
        raise ValueError('Horizon entier positif requis.')
    history=history[['sales']].copy()
    if len(history)<28 or not np.isfinite(history['sales']).all():
        raise ValueError('Au moins 28 observations finies requises.')
    predictions=[]
    for step in range(horizon):
        next_date=history.index[-1]+pd.Timedelta(days=1)
        future=pd.DataFrame({'sales':[np.nan]},index=[next_date])
        data=create_features_for_future(pd.concat([history,future]))
        X_next=data.loc[[next_date],feature_cols]
        if not np.isfinite(X_next.to_numpy(dtype=float)).all():
            raise ValueError('Features futures incomplètes.')
        pred=float(model.predict(X_next)[0])
        if not np.isfinite(pred):
            raise ValueError('Prévision non finie.')
        history.loc[next_date,'sales']=pred
        predictions.append(pred)
    return pd.Series(predictions,index=pd.date_range(history.index[-horizon],periods=horizon,freq='D'),name='prediction')
~~~

La fonction create_features_for_future doit calculer les lags et les statistiques glissantes à partir des valeurs antérieures, tout en conservant la ligne future même si sa cible est inconnue.

Voici une implémentation compatible avec notre exemple :

~~~python
def create_features_for_future(df):
    if not isinstance(df.index,pd.DatetimeIndex) or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError('Dates uniques et triées requises.')
    if len(df)>1 and not (df.index.to_series().diff().iloc[1:]==pd.Timedelta(days=1)).all():
        raise ValueError('Calendrier journalier régulier requis ; traiter les dates absentes.')
    data=df.copy()
    for lag in [1,7,14,28]:
        data[f'lag_{lag}']=data['sales'].shift(lag)
    past=data['sales'].shift(1)
    data['rolling_mean_7']=past.rolling(7).mean()
    data['rolling_std_7']=past.rolling(7).std()
    data['day_of_week']=data.index.dayofweek
    data['month']=data.index.month
    data['is_weekend']=(data.index.dayofweek>=5).astype(int)
    return data
~~~

Utilisation :

~~~python
# Réentraîner avec uniquement les dates antérieures à l'origine.
from ch06_cours04_lab import make_model
history=df.iloc[:850].copy()
origin_data=create_features(history)
origin_model=make_model(iterations=200)
origin_model.fit(origin_data[FEATURES],origin_data['sales'])
future_predictions=recursive_forecast(origin_model,history,14,FEATURES)
print(future_predictions)
~~~

Pour une évaluation rigoureuse, le modèle utilisé ici doit avoir été entraîné uniquement sur les données disponibles avant l'origine choisie. Le modèle entraîné dans la section précédente peut avoir vu des observations postérieures à l'indice 850 : il faut donc le réentraîner avec un découpage approprié avant d'utiliser cet exemple comme backtest.


## 9. Validation temporelle avec LightGBM

Une validation robuste consiste à répéter l'entraînement sur plusieurs fenêtres chronologiques.

~~~python
from sklearn.model_selection import TimeSeriesSplit
# Réserver le test final : folds sur le développement uniquement.
X_dev,y_dev=X.iloc[:val_end],y.iloc[:val_end]
fold_mae=[]
for train_idx,val_idx in TimeSeriesSplit(n_splits=4,test_size=100).split(X_dev):
    fold_model=make_model(iterations=200)
    fold_model.fit(X_dev.iloc[train_idx],y_dev.iloc[train_idx])
    pred=fold_model.predict(X_dev.iloc[val_idx])
    fold_mae.append(mean_absolute_error(y_dev.iloc[val_idx],pred))
print(fold_mae)
~~~

Ce protocole est adapté à l'évaluation de prédictions à un pas lorsque les observations passées deviennent disponibles avant chaque nouvelle prédiction.

Pour un horizon fixe de 7 ou 30 jours, il faudrait adapter la construction des cibles et des features à chaque origine.


## 10. Feature importance et interprétabilité

Les modèles de boosting permettent d'analyser l'influence des variables sur les prédictions.
Importance des variables

~~~python
importance=pd.DataFrame({'feature':feature_cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
print(importance.head(10))
~~~

L'importance native de LightGBM peut être fondée sur le nombre de splits ou sur le gain, selon la configuration.

Elle ne mesure pas directement une relation causale.
SHAP
SHAP permet d'expliquer la contribution des variables aux prédictions d'un modèle.

~~~python
# Extension optionnelle locale : python -m pip install shap
import shap
explainer=shap.TreeExplainer(model)
shap_values=explainer(X_test)
shap.summary_plot(shap_values,X_test)
~~~

Exemple d'interprétation :
- lag_7 peut contribuer fortement à une prévision lorsque le comportement hebdomadaire est régulier.
- rolling_mean_7 peut représenter le niveau récent des ventes.
- day_of_week peut aider à distinguer les jours ouvrés et les week-ends.
Les valeurs SHAP décrivent les contributions selon le modèle et le jeu de référence, pas des effets causaux.


## 11. ML vs ARIMA vs Deep Learning

Critère	ARIMA / SARIMA	LightGBM	LSTM / Transformers
Petites séries	Souvent adapté	Possible	Généralement plus difficile
Non-linéarité	Limitée dans la forme standard	Bonne capacité	Bonne capacité
Variables externes	Avec ARIMAX/SARIMAX	Faciles à intégrer	Possibles
Feature engineering	Relativement limité	Très important	Variable selon architecture
Interprétabilité	Paramètres statistiques	Importance / SHAP	Souvent plus complexe
Coût de calcul	Généralement modéré	Généralement modéré	Potentiellement élevé
Multi-séries	Souvent modèles locaux	Modèle global possible	Modèles globaux possibles

Il n'existe pas de famille universellement meilleure.

Le choix doit être fondé sur les résultats de validation, la quantité de données, les contraintes métier et le coût de maintenance.


## 12. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Construire des lags sans tenir compte de l'horizon	Fuite temporelle	Définir l'origine de prévision
Utiliser une rolling mean contenant la cible	Évaluation artificiellement optimiste	Décaler les observations
Faire un split aléatoire	Mélange temporel	Validation chronologique
Évaluer une prévision à un pas comme une prévision à 30 jours	Résultats trompeurs	Adapter le backtest à l'horizon
Oublier les baselines	Valeur ajoutée inconnue	Comparer naïve et saisonnière
Utiliser les futures variables externes réelles	Information indisponible	Vérifier la disponibilité
Trop de lags sur une petite série	Surapprentissage possible	Sélectionner les features
Ne pas analyser les erreurs par horizon	Faiblesses masquées	Calculer MAE(h), RMSE(h)
Interpréter SHAP comme une causalité	Conclusions incorrectes	Distinguer association et causalité

[[quiz]]


## 14. Synthèse du cours

Les concepts essentiels :
- Le forecasting ML est un problème supervisé construit à partir de l'historique temporel.
- Les lag features permettent d'exploiter les valeurs passées.
- Les rolling features résument les comportements récents.
- Les variables calendaires aident à capturer les cycles connus.
- LightGBM peut apprendre des relations non linéaires à partir de ces features.
- Les stratégies récursive, directe et multi-output répondent à des besoins différents de prévision multi-horizon.
- La validation doit respecter l'origine et l'horizon de prévision.
- Les baselines restent indispensables pour démontrer la valeur ajoutée d'un modèle avancé.
Prochain cours
Cours 05 — Multi-séries, forecasting hiérarchique et demande intermittente

Partie A · 05/07

Nous découvrirons comment prévoir simultanément des centaines ou des milliers de séries, comparer un modèle global à des modèles locaux, réconcilier des prévisions hiérarchiques et traiter les séries avec de nombreux zéros grâce à la méthode de Croston.


## Atelier navigateur et lab LightGBM

L’exercice intégré utilise NumPy et pandas pour vérifier les features et la récursion. Le lab complet utilise LightGBM localement. L’option --engine browser lance une variante Random Forest : elle illustre le protocole et ne reproduit pas les résultats LightGBM. SHAP reste une extension locale optionnelle. Pour des modèles directs ou multi-output, toutes les cibles d’entraînement doivent être disponibles avant la frontière de validation ; purger les derniers exemples si nécessaire.


## Challenge Python

Compléter create_features_for_future(df) et recursive_forecast(model, history, horizon, feature_cols). Conserver la ligne future sans cible, décaler les rolling features et réinjecter les prédictions. Vérifier que les features passées ne changent pas quand le futur change. Le lab LightGBM complet se lance localement.

[[assessment]]

[[interview]]
