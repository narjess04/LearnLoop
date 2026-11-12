Cours 05 — Multi-séries, prévision hiérarchique et demande intermittente
Partie A · 05/07
Niveau : Intermédiaire → Avancé
Durée estimée : 3–4 h

Dans le cours précédent, nous avons appris à utiliser le Machine Learning pour prévoir une série temporelle avec des lags, des rolling features et LightGBM.

Mais dans une entreprise, on ne prévoit généralement pas une seule série. Un supermarché peut avoir des milliers de produits, plusieurs magasins et différentes catégories.

Cela soulève trois problèmes :
- Multi-séries : faut-il entraîner un modèle pour chaque série ou un modèle commun ?
- Prévision hiérarchique : comment garantir que les prévisions par produit correspondent aux prévisions totales ?
- Demande intermittente : comment prévoir des produits qui ne se vendent pas tous les jours ?
Ce cours répond à ces trois questions.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Distinguer les modèles locaux, globaux et hybrides.
- Structurer un dataset contenant plusieurs séries temporelles.
- Construire des features temporelles sans mélanger les séries.
- Entraîner un modèle global avec LightGBM.
- Comprendre les prévisions hiérarchiques et leur réconciliation.
- Appliquer les méthodes Bottom-Up, Top-Down et MinT.
- Comprendre la demande intermittente et la méthode de Croston.
- Évaluer correctement un système de forecasting multi-séries.


## 2. Comprendre le forecasting multi-séries


## 2.1 D'une série unique à plusieurs séries

Dans les cours précédents, nous avions une seule variable :
\[
Y_t=\text{ventes au temps }t
\]

Dans un problème multi-séries, nous avons :
\[
\boxed{Y_{i,t}}
\]

où :
- \(i\) représente une série : produit, magasin, région ou client.
- \(t\) représente l'instant temporel.
- \(Y_{i,t}\) représente la valeur de la série \(i\) à l'instant \(t\).
Par exemple :
Date	Magasin	Produit	Ventes
01/01	Tunis	A	120
01/01	Tunis	B	35
01/01	Sousse	A	95
02/01	Tunis	A	130
02/01	Tunis	B	0
02/01	Sousse	A	102

Chaque combinaison (magasin, produit) correspond à une série distincte.

## 2.2 Pourquoi est-ce plus difficile ?

Les séries peuvent présenter des comportements très différents :
- Le produit A se vend quotidiennement.
- Le produit B ne se vend que quelques fois par mois.
- Le produit C possède une forte saisonnalité.
- Le produit D vient d'être lancé et possède peu d'historique.
Un système de forecasting doit être capable de gérer cette hétérogénéité.
Trois comportements de demande
Produit A — Demande régulière

Produit B — Demande saisonnière

Produit C — Demande intermittente

Données illustratives. Les trois séries n'ont pas les mêmes propriétés statistiques.


## 3. Modèles locaux vs modèles globaux

Deux stratégies principales existent pour modéliser plusieurs séries.

## 3.1 Modèles locaux (Local Models)

Un modèle local est entraîné indépendamment pour chaque série.
\[
\boxed{\hat Y_{i,t+h}=f_i(Y_{i,1:t})}
\]

Par exemple :
- SARIMA pour le produit A.
- ETS pour le produit B.
- Croston pour le produit C.
Chaque série possède son propre modèle et ses propres paramètres.
Chaque série possède un modèle indépendant.
Avantages : personnalisation, interprétation plus simple, possibilité d'utiliser des modèles statistiques spécifiques.

Limites : coût de maintenance élevé lorsque le nombre de séries augmente et difficulté à apprendre avec très peu d'historique.

## 3.2 Modèles globaux (Global Models)

Un modèle global est entraîné sur plusieurs séries simultanément.
\[
\boxed{\hat Y_{i,t+h}=f(X_{i,t},S_i)}
\]

où \(S_i\) représente des caractéristiques propres à la série, comme son identifiant, sa catégorie ou son magasin.

Le modèle apprend des comportements communs à plusieurs produits.

Par exemple, il peut apprendre que les produits alimentaires d'une même catégorie présentent des variations hebdomadaires similaires.

Avantages : mutualisation des données, maintenance simplifiée, possibilité de transférer des régularités entre séries.

Limites : les séries très différentes peuvent être difficiles à représenter avec un modèle commun.

## 3.3 Comparaison

Critère	Local	Global
Nombre de modèles	Un par série	Un ou quelques modèles
Partage d'information	Limité	Oui
Petits historiques	Souvent difficile	Peut être avantageux
Personnalisation	Forte	Via features et identifiants
Maintenance à grande échelle	Plus coûteuse	Généralement plus simple
Risque principal	Surapprentissage local	Mauvaise représentation des différences


## 3.4 Approche hybride

Une approche hybride combine les deux stratégies.

Exemple :
- Modèle global LightGBM pour la majorité des produits.
- SARIMA local pour certains produits saisonniers.
- Croston pour les produits à demande intermittente.
Cette stratégie peut être intéressante dans un système industriel, à condition que les règles de sélection soient définies à partir des données d'entraînement et de validation.


## 4. Préparer un dataset multi-séries

Dans le cours 04, nous avons construit des lags pour une seule série.

Avec plusieurs séries, il faut impérativement calculer les lags indépendamment pour chaque identifiant.

## 4.1 Structure recommandée

Le format long (long format) est généralement pratique.
date	series_id	sales
2025-01-01	A	100
2025-01-02	A	110
2025-01-01	B	30
2025-01-02	B	0


## 4.2 Erreur fréquente


~~~python
# Incorrect : peut faire passer une valeur d'une série à l'autre.
df['lag_1']=df['sales'].shift(1)
~~~

Ce code peut mélanger les séries.

Par exemple, le lag du produit B pourrait récupérer la dernière valeur du produit A.

## 4.3 Bonne implémentation


~~~python
import pandas as pd
df=df.sort_values(['series_id','date']).copy()
for lag in [1,7,14]:
    df[f'lag_{lag}']=df.groupby('series_id',observed=True)['sales'].shift(lag)
~~~

Chaque série utilise uniquement son propre historique.

## 4.4 Rolling features par série


~~~python
grouped=df.groupby('series_id',observed=True)['sales']
df['rolling_mean_7']=grouped.transform(lambda s:s.shift(1).rolling(7).mean())
df['rolling_std_7']=grouped.transform(lambda s:s.shift(1).rolling(7).std())
~~~

Important : avant ces transformations, il faut vérifier que les dates sont régulières. Si certaines dates sont absentes, shift(7) signifie sept lignes précédentes, pas nécessairement sept jours précédents.

Pour des séries journalières, on peut compléter le calendrier de chaque série avant de créer les features, en distinguant les véritables zéros de vente des données manquantes.


## 5. Entraîner un modèle global avec LightGBM

Nous allons entraîner un modèle commun à plusieurs produits.

## 5.1 Générer un dataset synthétique


~~~python
import numpy as np
import pandas as pd
from ch06_cours05_lab import dataset
df=dataset()
~~~


## 5.2 Construire les features


~~~python
def create_panel_features(df):
    data=df.sort_values(['series_id','date']).copy()
    data['date']=pd.to_datetime(data['date'])
    if data.duplicated(['series_id','date']).any():raise ValueError('Dates dupliquées dans une série.')
    gaps=data.groupby('series_id',observed=True)['date'].diff().dropna()
    if not (gaps==pd.Timedelta(days=1)).all():raise ValueError('Compléter les calendriers sans confondre zéro et manquant.')
    grouped=data.groupby('series_id',observed=True)['sales']
    for lag in [1,7,14,28]:data[f'lag_{lag}']=grouped.shift(lag)
    data['rolling_mean_7']=grouped.transform(lambda s:s.shift(1).rolling(7).mean())
    data['rolling_std_7']=grouped.transform(lambda s:s.shift(1).rolling(7).std())
    data['day_of_week']=data['date'].dt.dayofweek;data['month']=data['date'].dt.month
    data['series_id']=data['series_id'].astype('category')
    return data.dropna(subset=['sales','lag_28','rolling_mean_7','rolling_std_7'])

data=create_panel_features(df)
~~~


## 5.3 Découpage chronologique global

Il faut découper selon les dates, pas selon les positions des lignes.

~~~python
train=data[data['date']<'2025-01-01'].copy()
test=data[data['date']>='2025-01-01'].copy()
features=[col for col in data if col not in ['date','sales']]
X_train,y_train=train[features],train['sales']
X_test,y_test=test[features],test['sales']
~~~


## 5.4 Entraîner LightGBM


~~~python
import lightgbm as lgb
model=lgb.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=31,random_state=42,verbosity=-1,n_jobs=1)
model.fit(X_train,y_train,categorical_feature=['series_id'])
predictions=model.predict(X_test)
~~~


## 5.5 Évaluer par série


~~~python
results=test[['date','series_id','sales']].copy()
results['prediction']=predictions
results['absolute_error']=np.abs(results['sales']-results['prediction'])
mae_by_series=results.groupby('series_id',observed=True)['absolute_error'].mean()
print(mae_by_series)
~~~

L'évaluation par série permet d'identifier les produits sur lesquels le modèle est performant ou non.

Attention : ce protocole utilise les observations réelles antérieures à chaque date du test pour calculer les lags. Il représente donc une évaluation à un pas avec actualisation de l'historique, et non une prévision de toute la période de test produite à une origine unique.

Pour une prévision multi-horizon, il faudrait utiliser une stratégie directe ou récursive sans accéder aux observations futures.

## Section 2 du cours — Forecasting hiérarchique

## 6. Comprendre les séries hiérarchiques

Dans une entreprise, les données possèdent souvent plusieurs niveaux d'agrégation.

Exemple :
[[reconciliation]]

Dans cette hiérarchie :
\[
Y_{\text{Total},t}
=
Y_{\text{Tunis},t}
+
Y_{\text{Sousse},t}
\]

Et :
\[
Y_{\text{Tunis},t}
=
Y_{\text{Tunis,A},t}
+
Y_{\text{Tunis,B},t}
\]

Les prévisions doivent idéalement respecter ces relations.

## 7. Le problème de cohérence hiérarchique

Supposons que nous entraînons un modèle indépendant pour chaque niveau.

Nous obtenons :
Niveau	Prévision
Total entreprise	1 000
Magasin Tunis	600
Magasin Sousse	450

Problème :
\[
600+450=1050\neq1000
\]

Les prévisions sont incohérentes.

Cette incohérence peut poser des problèmes pour la planification des stocks, des budgets et des ressources.

La réconciliation hiérarchique consiste à ajuster les prévisions pour qu'elles respectent les contraintes d'agrégation.

## 8. Bottom-Up

La méthode Bottom-Up consiste à prévoir les séries les plus détaillées, puis à agréger leurs prévisions.
\[
\boxed{
\hat Y_{\text{Total}}
=
\sum_{i=1}^{N}\hat Y_i
}
\]

Exemple :
Produit	Prévision
Tunis A	250
Tunis B	350
Sousse A	200
Sousse B	250
Total	1 050

Avantages :
- Cohérence automatique.
- Conservation des prévisions détaillées.
- Méthode simple.
Limite : les séries de bas niveau peuvent être très bruitées ou posséder peu d'observations.

## 9. Top-Down

La méthode Top-Down consiste à prévoir le total, puis à répartir la prévision entre les niveaux inférieurs.

Si le total prévu est 1 000 et que les proportions historiques sont :
- Tunis : 60 %.
- Sousse : 40 %.
Alors :
\[
\hat Y_{\text{Tunis}}=1000\times0.6=600
\]

\[
\hat Y_{\text{Sousse}}=1000\times0.4=400
\]

Cette approche peut être utile lorsque le niveau agrégé est plus stable.

Sa principale limite est que les proportions historiques peuvent évoluer.

Une approche Top-Down ne conserve pas nécessairement toutes les propriétés de cohérence dans les hiérarchies complexes si les répartitions ne sont pas définies de manière cohérente à chaque niveau.

## 10. MinT — Minimum Trace Reconciliation

MinT est une méthode de réconciliation plus avancée.

Elle utilise la structure de covariance des erreurs de prévision pour ajuster les prévisions de différents niveaux.

La formulation générale est :
\[
\boxed{
\tilde{\mathbf y}
=
\mathbf S
(\mathbf S^\top\mathbf W^{-1}\mathbf S)^{-1}
\mathbf S^\top\mathbf W^{-1}
\hat{\mathbf y}
}
\]

Où :
- \(\hat{\mathbf y}\) : vecteur des prévisions initiales.
- \(\tilde{\mathbf y}\) : vecteur des prévisions réconciliées.
- \(\mathbf S\) : matrice d'agrégation hiérarchique.
- \(\mathbf W\) : matrice de covariance des erreurs des prévisions initiales.
Cette expression suppose notamment que les matrices nécessaires sont inversibles. Des variantes utilisent des estimations régularisées ou des pseudo-inverses.

L'idée est de produire des prévisions cohérentes tout en tenant compte de la précision relative des différents niveaux.
Exemple de matrice d'agrégation
Pour deux produits A et B :
\[
\mathbf S=
\begin{bmatrix}
1&1\\
1&0\\
0&1
\end{bmatrix}
\]

Le vecteur de base est :
\[
\mathbf b=
\begin{bmatrix}
Y_A\\
Y_B
\end{bmatrix}
\]

Et :
\[
\mathbf y=\mathbf S\mathbf b
=
\begin{bmatrix}
Y_A+Y_B\\
Y_A\\
Y_B
\end{bmatrix}
\]

Cette représentation permet de généraliser les contraintes de cohérence à des hiérarchies complexes.
Comparaison des méthodes
Méthode	Principe	Avantage	Limite
Bottom-Up	Prévoir puis sommer	Simple et cohérente	Séries détaillées bruitées
Top-Down	Prévoir le total puis répartir	Total souvent plus stable	Proportions parfois inadaptées
MinT	Réconciliation par covariance	Exploite la précision relative	Estimation plus complexe

## Section 3 du cours — Demande intermittente

## 11. Qu'est-ce que la demande intermittente ?

Une demande intermittente correspond à une série contenant de nombreuses périodes sans demande, séparées par des demandes positives.

Exemple :
\[
Y=[0,0,5,0,0,0,8,0,0,3,0,0]
\]

Ce comportement est fréquent pour :
- Les pièces de rechange.
- Les équipements industriels.
- Certains médicaments spécialisés.
- Les produits à faible rotation.
- Les composants de maintenance.
Les méthodes classiques peuvent rencontrer des difficultés lorsque la série contient beaucoup de zéros.

## 11.1 Distinguer zéro et donnée manquante

Un zéro peut signifier :
- Aucune demande.
- Aucune vente à cause d'une rupture de stock.
- Un magasin fermé.
- Une donnée absente codée incorrectement.
Ces situations ne sont pas équivalentes.

En particulier, les ventes observées ne représentent pas toujours la demande réelle lorsqu'un produit est indisponible.


## 12. La méthode de Croston

Croston est une méthode classique de prévision de la demande intermittente.

Son idée est de séparer deux phénomènes :

## 1. La taille des demandes positives.


## 2. Le temps entre les demandes positives.


## 12.1 Exemple

Considérons :
\[
Y=[0,0,5,0,0,0,8,0,0,3,0,0]
\]

Les demandes positives sont :
\[
Z=[5,8,3]
\]

Leurs positions sont :
\[
[3,7,10]
\]

Les intervalles entre les demandes positives sont :
\[
P=[4,3]
\]

Croston applique un lissage exponentiel aux tailles des demandes et aux intervalles.

## 12.2 Formulation mathématique

Lorsqu'une nouvelle demande positive est observée :
\[
\hat Z_t=
\alpha Z_t+(1-\alpha)\hat Z_{t-1}
\]

\[
\hat P_t=
\alpha P_t+(1-\alpha)\hat P_{t-1}
\]

La prévision de demande moyenne par période est :
\[
\boxed{
\hat Y_t=\frac{\hat Z_t}{\hat P_t}
}
\]

Où :
- \(\hat Z_t\) : taille moyenne lissée des demandes positives.
- \(\hat P_t\) : intervalle moyen lissé entre demandes.
- \(\alpha\) : coefficient de lissage.
Entre deux demandes positives, les estimations restent constantes dans la méthode classique.

## 12.3 Exemple numérique

Supposons :
\[
\hat Z=10,\quad \hat P=5
\]

Alors :
\[
\hat Y=\frac{10}{5}=2
\]

Cela signifie une demande moyenne estimée de deux unités par période.

Il ne faut pas interpréter cette prévision comme l'affirmation que deux unités seront demandées à chaque période.

## 12.4 Implémentation Python


~~~python
def demand_vector(y):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError('Demande non négative, finie, non vide et unidimensionnelle requise.')
    return y

def croston_forecast(y,alpha=.1):
    y=demand_vector(y)
    if not 0<alpha<=1:
        raise ValueError('alpha dans ]0,1] requis.')
    positive=np.flatnonzero(y>0)
    if not len(positive):return 0.
    first=positive[0];z_hat=y[first];p_hat=float(first+1);last=first
    for idx in positive[1:]:
        z_hat=alpha*y[idx]+(1-alpha)*z_hat
        p_hat=alpha*(idx-last)+(1-alpha)*p_hat
        last=idx
    return float(z_hat/p_hat)
~~~

Cette implémentation est volontairement pédagogique. Les bibliothèques spécialisées peuvent utiliser d'autres conventions d'initialisation.


## 13. Les limites de Croston et ses variantes

La méthode classique de Croston présente un biais dans certaines conditions.

Deux variantes importantes sont :

## 13.1 SBA — Syntetos-Boylan Approximation

SBA applique une correction à la prévision de Croston :
\[
\boxed{
\hat Y_{\text{SBA}}
=
\left(1-\frac{\alpha}{2}\right)
\frac{\hat Z}{\hat P}
}
\]

Cette correction vise à réduire le biais de la méthode classique dans le cadre théorique étudié.

## 13.2 TSB — Teunter-Syntetos-Babai

TSB modélise séparément :
- La probabilité qu'une demande positive survienne.
- La taille moyenne d'une demande positive.
La prévision devient :
\[
\boxed{
\hat Y_t=\hat p_t\hat z_t
}
\]

La probabilité est actualisée à chaque période, y compris lorsqu'aucune demande n'est observée.

Cette propriété permet à TSB de réagir à une succession de zéros, ce qui est particulièrement utile lorsqu'un produit devient obsolète.
Comparaison
Méthode	Principe	Particularité
Croston	Taille / intervalle	Référence historique
SBA	Croston corrigé	Réduction du biais
TSB	Probabilité × taille	Réagit aux périodes sans demande


## 14. Comment évaluer les prévisions intermittentes ?

Les séries intermittentes nécessitent une attention particulière dans le choix des métriques.

La MAPE est généralement problématique lorsque les valeurs réelles contiennent des zéros.

La sMAPE peut également devenir instable.

On peut utiliser :
- MAE et RMSE pour mesurer les erreurs par période.
- MASE lorsque son dénominateur est non nul et pertinent.
- Erreurs sur la demande cumulée sur un horizon.
- Mesures de biais et indicateurs métier liés aux stocks.
Exemple : demande cumulée
Supposons que la demande réelle sur sept jours soit :
\[
[0,0,5,0,0,3,0]
\]

La demande totale est :
\[
D_{\text{réel}}=8
\]

Si un modèle prédit deux unités par jour :
\[
D_{\text{prédit}}=14
\]

L'erreur sur la demande cumulée est :
\[
|14-8|=6
\]

Cette mesure est importante pour les décisions de réapprovisionnement.

Cependant, une bonne prévision cumulée ne garantit pas que le modèle prédit correctement le moment des demandes.


## 15. Mini-projet : choisir un modèle selon le type de série

Nous allons créer trois séries synthétiques, puis définir une première stratégie de modélisation.

## Étape 1 — Générer les séries


~~~python
from ch06_cours05_lab import intermittent_dataset
df=intermittent_dataset()
~~~


## Étape 2 — Calculer les caractéristiques

Deux indicateurs classiques permettent de caractériser la demande intermittente.

ADI — Average Demand Interval
\[
ADI=\frac{\text{Nombre de périodes}}{\text{Nombre de demandes positives}}
\]

CV² — Squared Coefficient of Variation

Il mesure la variabilité relative des tailles de demandes positives :
\[
CV^2=
\left(
\frac{\sigma_Z}{\mu_Z}
\right)^2
\]

~~~python
def demand_characteristics(y):
    y=demand_vector(y);positive=y[y>0]
    if not len(positive):return {'ADI':np.inf,'CV2':np.nan,'zero_rate':1.}
    return {'ADI':len(y)/len(positive),'CV2':float(np.var(positive)/np.mean(positive)**2),
            'zero_rate':float(np.mean(y==0))}

# Caractériser uniquement l'historique disponible pour choisir une stratégie.
for col in df:
    print(col,demand_characteristics(df[col].iloc[:140]))
~~~


## Étape 3 — Classifier les séries

Une classification descriptive courante utilise les seuils historiques :
\[
ADI=1.32,\quad CV^2=0.49
\]

Catégorie	ADI	CV²
Smooth	< 1,32	< 0,49
Intermittent	≥ 1,32	< 0,49
Erratic	< 1,32	≥ 0,49
Lumpy	≥ 1,32	≥ 0,49

Ces seuils constituent une classification heuristique issue de la littérature, pas des règles universelles.

~~~python
def classify_demand(adi,cv2):
    if np.isinf(adi):return 'All-zero'
    if not np.isfinite(adi) or not np.isfinite(cv2) or adi<1 or cv2<0:
        raise ValueError('ADI >= 1 et CV2 >= 0 finis requis.')
    if adi<1.32:return 'Smooth' if cv2<.49 else 'Erratic'
    return 'Intermittent' if cv2<.49 else 'Lumpy'

for col in df:
    stats=demand_characteristics(df[col].iloc[:140])
    print(col,classify_demand(stats['ADI'],stats['CV2']))
~~~


## Étape 4 — Définir une stratégie initiale

Une première stratégie expérimentale pourrait être :
Type de demande	Modèles à tester
Smooth	ETS, LightGBM, baseline
Intermittent	Croston, SBA, TSB
Erratic	ETS, modèles robustes, ML
Lumpy	Croston/SBA/TSB, modèles probabilistes

Cette table ne signifie pas qu'une méthode est systématiquement optimale pour une catégorie.

Le choix final doit être effectué à partir d'un backtest temporel et des métriques adaptées au problème métier.


## 16. Bonnes pratiques en production

Dans un système de forecasting industriel, plusieurs points sont essentiels.
Problème	Bonne pratique
Des milliers de séries	Envisager un modèle global
Séries de tailles très différentes	Évaluer par série et par segment
Peu d'historique	Tester le partage d'information
Demande intermittente	Comparer Croston, SBA, TSB et baselines
Prévisions hiérarchiques incohérentes	Appliquer une réconciliation
Données manquantes	Distinguer absence de demande et absence d'observation
Ruptures de stock	Identifier les ventes censurées
Nouveaux produits	Exploiter les caractéristiques des produits similaires
Changement de comportement	Réévaluer régulièrement les modèles
Coûts de stockage	Ajouter des métriques métier

Architecture simplifiée d'un système industriel

[[quiz]]


## 18. Synthèse du cours

Les concepts essentiels à retenir :
- Modèles locaux : un modèle indépendant pour chaque série.
- Modèles globaux : un modèle entraîné sur plusieurs séries, avec partage d'information.
- Feature engineering multi-séries : calcul des lags et rolling features par identifiant.
- Forecasting hiérarchique : cohérence entre prévisions détaillées et agrégées.
- Bottom-Up : prévisions détaillées puis agrégation.
- Top-Down : prévision agrégée puis répartition.
- MinT : réconciliation tenant compte des erreurs de prévision.
- Demande intermittente : présence de nombreuses périodes sans demande.
- Croston : lissage séparé des tailles et intervalles de demande.
- SBA et TSB : variantes adaptées à certaines limites de Croston.
Prochain cours
Cours 06 — Prévision probabiliste

Partie A · 06/07

Nous étudierons comment passer d'une prévision ponctuelle à une distribution de prévisions, construire des intervalles de prédiction et quantifier l'incertitude avec la régression quantile, la Pinball Loss, le CRPS, la couverture des intervalles et les méthodes conformales.


## Compléments exécutables

Le lab complète Croston, SBA et TSB et évalue une prévision de 40 jours à origine fixe. Les caractéristiques ADI/CV² sont calculées sur le train. La matrice W de l’exemple de réconciliation est illustrative ; en pratique, l’estimer à partir d’erreurs hors échantillon sur le développement et la régulariser si nécessaire. W = I donne une réconciliation OLS. La réconciliation seule n’impose pas la positivité des prévisions.


## Challenge Python

Compléter croston_forecast(y, alpha) et mint_reconcile(base, S, W). Initialiser l’intervalle à l’indice de la première demande + 1 ; actualiser taille et intervalle seulement sur les demandes positives. Réconcilier les prévisions avec une covariance définie positive. Comparer les totaux et observer TSB après une suite de zéros.

[[assessment]]

[[interview]]
