Partie A · Séries temporelles · Cours 07/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[anomalies]]

Cours 07 — Détection d'anomalies, ruptures temporelles et drift
LearnLoop · Partie A — Séries temporelles · Cours 07/07

Niveau : Intermédiaire → Avancé · Durée : 3–4 h

Dans le cours précédent, nous avons appris à quantifier l'incertitude des prévisions avec les quantiles, les intervalles de prédiction et la Conformal Prediction.

Mais un modèle performant aujourd'hui peut devenir moins fiable demain. Une panne, un changement de comportement des clients ou une modification du marché peuvent transformer la structure d'une série temporelle.

Ce dernier cours de la partie Séries temporelles aborde trois questions essentielles :
- Comment détecter une observation anormale ?
- Comment identifier un changement durable dans le comportement d'une série ?
- Comment surveiller un modèle de forecasting lorsque les données évoluent ?

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Distinguer anomalies ponctuelles, contextuelles et collectives.
- Détecter des anomalies avec Z-score, IQR, STL et Isolation Forest.
- Comprendre les méthodes CUSUM et Change Point Detection.
- Utiliser la bibliothèque ruptures pour identifier des changements de régime.
- Distinguer data drift, concept drift et dégradation des performances.
- Concevoir un système de surveillance temporelle.
- Évaluer les détecteurs d'anomalies sans fuite d'information.


## 2. Qu'est-ce qu'une anomalie temporelle ?

Une anomalie est une observation ou un ensemble d'observations qui s'écarte du comportement attendu.

Dans une série temporelle, une valeur inhabituelle n'est pas nécessairement une anomalie.

Par exemple, 500 commandes peuvent être normales pendant les soldes, mais très inhabituelles un jour ordinaire.

C'est pourquoi le contexte temporel est essentiel.
Les trois grandes catégories
A. Anomalie ponctuelle

Une observation isolée s'écarte fortement des valeurs habituelles.

Exemple : pic inhabituel de transactions sur une seule journée.

B. Anomalie contextuelle

Une valeur devient anormale uniquement dans un contexte particulier.

Un pic peut être normal le week-end, mais inhabituel un jour ouvré.

C. Anomalie collective

Une séquence d'observations présente un comportement anormal, même si ses valeurs individuelles ne sont pas extrêmes.

Exemple : niveau anormalement élevé pendant plusieurs jours consécutifs.

Anomalie vs rupture
Il est important de distinguer ces deux notions.
Anomalie	Rupture temporelle
Événement inhabituel	Modification du régime statistique
Peut être isolée	Souvent persistante
Exemple : pic soudain	Exemple : augmentation durable du niveau
Détection locale	Détection d'un changement structurel

Une rupture peut provoquer des anomalies, mais les deux concepts ne sont pas équivalents.


## 3. Détection statistique avec Z-score

Le Z-score mesure l'écart entre une observation et la moyenne, exprimé en nombre d'écarts-types.
\[
\boxed{Z_t=\frac{Y_t-\mu}{\sigma}}
\]

où :
- \(Y_t\) : observation.
- \(\mu\) : moyenne de référence.
- \(\sigma\) : écart-type de référence.
Une règle heuristique fréquente consiste à considérer :
\[
|Z_t|>3
\]

comme un signal d'anomalie potentielle.
Exemple
Supposons :
\[
\mu=100,\quad \sigma=10,\quad Y_t=145
\]

Alors :
\[
Z_t=\frac{145-100}{10}=4.5
\]

Cette observation est éloignée de la moyenne de référence.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Limites
Le Z-score global suppose que la moyenne et la dispersion de référence sont pertinentes sur toute la période.

Il peut être peu adapté lorsque la série présente une tendance, une saisonnalité ou une distribution fortement asymétrique.

De plus, calculer la moyenne et l'écart-type sur tout le dataset, y compris le futur, introduit une fuite d'information dans un système de détection en ligne.


## 4. Détection robuste avec IQR et MAD


## 4.1 IQR — Interquartile Range

L'écart interquartile est défini par :
\[
IQR=Q_3-Q_1
\]

Une règle classique de détection est :
\[
Y_t<Q_1-1.5IQR
\]

ou :
\[
Y_t>Q_3+1.5IQR
\]

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

L'IQR est moins sensible aux valeurs extrêmes que la moyenne et l'écart-type.

## 4.2 MAD — Median Absolute Deviation

La MAD mesure la dispersion autour de la médiane :
\[
\boxed{
MAD=\operatorname{median}(|Y_i-\operatorname{median}(Y)|)
}
\]

On peut construire un score robuste :
\[
Z_t^{robuste}
=
\frac{0.6745(Y_t-\operatorname{median}(Y))}
{MAD}
\]

Le facteur 0,6745 permet une mise à l'échelle approximative sous une distribution normale.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le seuil 3,5 est une convention heuristique, pas une règle universelle.

Pour une série temporelle, il est souvent préférable de calculer ces statistiques sur un historique de référence ou une fenêtre glissante utilisant uniquement les observations disponibles.


## 5. Détection avec Rolling Statistics

Une approche temporelle consiste à comparer l'observation actuelle au comportement récent.

On peut définir :
\[
\mu_t^{(w)}
=
\frac{1}{w}\sum_{i=1}^{w}Y_{t-i}
\]

et :
\[
Z_t^{rolling}
=
\frac{Y_t-\mu_t^{(w)}}{\sigma_t^{(w)}}
\]

La moyenne et l'écart-type sont calculés uniquement à partir du passé.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Cette méthode est plus adaptative qu'un Z-score global, mais elle peut réagir difficilement aux changements brutaux de régime ou à une saisonnalité complexe.


## 6. Détection d'anomalies avec STL

Nous avons découvert la décomposition STL dans le cours 01.

Rappel :
\[
\boxed{Y_t=T_t+S_t+R_t}
\]

où :
- \(T_t\) : tendance.
- \(S_t\) : saisonnalité.
- \(R_t\) : résidu.
L'idée est de détecter les anomalies sur les résidus plutôt que sur la série brute.
Pourquoi ?
Une valeur élevée pendant une période saisonnière peut être parfaitement normale.

En supprimant la tendance et la saisonnalité estimées, nous pouvons identifier les observations qui s'écartent du comportement attendu.
Décomposition temporelle et détection sur les résidus
Schéma illustratif : une anomalie peut être plus visible après retrait des composantes régulières.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Attention : une décomposition STL calculée sur toute la série peut exploiter des observations futures pour estimer les composantes d'une date passée.

Elle convient à l'analyse rétrospective, mais pas directement à une détection en temps réel sans adaptation.

Pour un système en ligne, il faut estimer le comportement attendu à partir du seul historique disponible.


## 7. Isolation Forest pour les anomalies temporelles

Isolation Forest est un algorithme non supervisé fondé sur une idée simple :

Les observations atypiques sont souvent plus faciles à isoler par des partitions aléatoires.

Il construit plusieurs arbres d'isolation.

Une observation isolée en peu de partitions tend à recevoir un score d'anomalie plus élevé.
Utilisation avec des séries temporelles
Isolation Forest n'intègre pas automatiquement la chronologie.

Il faut donc construire des variables temporelles.

Exemples :
\[
X_t=[
Y_t,
Y_{t-1},
RM_7(t),
STD_7(t),
DayOfWeek_t
]
\]

Si la détection intervient après l'observation de \(Y_t\), on peut utiliser cette valeur courante. Les autres features doivent être calculées sans information future.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

La valeur contamination=0.03 exprime ici une hypothèse de proportion d'anomalies utilisée pour déterminer le seuil de décision.

Elle ne signifie pas que 3 % des observations sont réellement anormales.
Limites
Isolation Forest peut être utile lorsque plusieurs variables décrivent le comportement temporel, mais il ne modélise pas explicitement les dépendances séquentielles.

Il faut également éviter de considérer toute nouveauté comme une anomalie métier.

## Section B du cours — Change Point Detection

## 8. Qu'est-ce qu'une rupture temporelle ?

Une rupture temporelle, ou change point, est un instant où les propriétés statistiques du processus changent.

Cela peut concerner :
- La moyenne.
- La variance.
- La tendance.
- La saisonnalité.
- Les relations entre variables.
Exemple :
\[
Y_t\sim
\begin{cases}
\mathcal N(100,10^2),&t<\tau\\
\mathcal N(140,10^2),&t\geq\tau
\end{cases}
\]

Le point \(\tau\) représente le changement de régime.
Rupture durable du niveau moyen

La différence fondamentale avec une anomalie ponctuelle est que le changement persiste.


## 9. CUSUM — Cumulative Sum

CUSUM est une méthode statistique conçue pour détecter des changements de niveau, notamment de petites variations persistantes.

## 9.1 Principe

On cumule les écarts entre les observations et une valeur de référence.

Une version unilatérale supérieure s'écrit :
\[
\boxed{
S_t^+=
\max(0,S_{t-1}^++Y_t-\mu_0-k)
}
\]

Pour détecter les baisses :
\[
\boxed{
S_t^-=
\max(0,S_{t-1}^-+\mu_0-Y_t-k)
}
\]

où :
- \(\mu_0\) : moyenne de référence.
- \(k\) : paramètre de tolérance.
- \(S_t^+\) et \(S_t^-\) : sommes cumulées.
Une alerte est déclenchée si une statistique dépasse un seuil \(h\).
Intuition
Une seule petite variation peut être ignorée.

Mais si plusieurs observations successives sont légèrement supérieures à la référence, leurs écarts s'accumulent.

CUSUM peut donc détecter une dérive persistante.
Implémentation Python

~~~python
from ch06_cours07_lab import rolling_zscore, robust_zscore, cusum_detector, population_stability_index

def cusum_detector(values,target_mean,k,h):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite([target_mean,k,h]).all() or k<0 or h<=0:
        raise ValueError('Données finies, k>=0 et h>0 requis.')
    positive=negative=0.;alerts=[]
    for t,value in enumerate(values):
        positive=max(0.,positive+value-target_mean-k)
        negative=max(0.,negative+target_mean-value-k)
        if positive>h or negative>h:
            alerts.append(t);positive=negative=0.
    return alerts
~~~

Les paramètres \(k\) et \(h\) doivent être calibrés en fonction de l'amplitude de changement recherchée et du taux de fausses alertes acceptable.


## 10. Détection de ruptures avec PELT

PELT signifie Pruned Exact Linear Time.

C'est un algorithme de segmentation qui cherche à identifier plusieurs points de rupture.
Formulation
On cherche une segmentation minimisant :
\[
\boxed{
\sum_{j=0}^{K}
C(Y_{\tau_j:\tau_{j+1}})
+\beta K
}
\]

où :
- \(C\) : coût associé à un segment.
- \(K\) : nombre de ruptures.
- \(\beta\) : pénalité par rupture.
La pénalité contrôle la complexité de la segmentation.

Une pénalité élevée tend à produire moins de ruptures.

Une pénalité faible peut produire davantage de segments.

PELT utilise une stratégie de programmation dynamique avec élagage. Sa complexité peut être proche du linéaire sous certaines conditions, mais ce n'est pas une garantie universelle.
Implémentation avec ruptures
Installation :

~~~shell
python -m pip install ruptures
~~~

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

La sortie contient les fins de segments, y compris la longueur totale de la série.

Par exemple, une sortie proche de [150, 300, 450] indiquerait deux ruptures internes et la fin du signal.
Visualisation

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## 11. Binary Segmentation et autres méthodes

PELT n'est pas la seule méthode de détection de ruptures.
Binary Segmentation
L'algorithme recherche une rupture, puis applique récursivement le même processus aux sous-segments.

Il est relativement simple et souvent rapide.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Comparaison
Méthode	Principe	Utilisation
CUSUM	Accumulation des écarts	Surveillance séquentielle
Binary Segmentation	Découpage récursif	Segmentation rétrospective
PELT	Optimisation pénalisée	Plusieurs ruptures
Window-based	Comparaison de fenêtres	Changements locaux
Bayesian Change Point	Modélisation probabiliste	Incertitude sur les ruptures

Attention : une méthode appliquée à une série complète détecte généralement les ruptures rétrospectivement. Pour un système temps réel, il faut utiliser une procédure séquentielle ou réexécuter la détection sur les données disponibles à chaque date.

## Section C du cours — Drift et surveillance des modèles

## 12. Comprendre le drift

Le drift correspond à une évolution de la distribution des données ou des relations apprises.

Dans le Machine Learning, plusieurs types de drift sont importants.

## 12.1 Data Drift

La distribution des variables explicatives évolue :
\[
\boxed{
P_t(X)\neq P_{t+\Delta}(X)
}
\]

Exemple : les températures observées en production deviennent très différentes de celles du jeu d'entraînement.

## 12.2 Concept Drift

La relation entre les variables et la cible évolue :
\[
\boxed{
P_t(Y\mid X)\neq
P_{t+\Delta}(Y\mid X)
}
\]

Exemple : auparavant, une promotion de 10 % augmentait fortement les ventes. Après un changement du marché, son effet devient plus faible.

## 12.3 Label Drift

La distribution de la cible évolue :
\[
P_t(Y)\neq P_{t+\Delta}(Y)
\]

Ces catégories peuvent coexister. Une évolution de \(P(Y)\) ne prouve pas, à elle seule, que la relation \(P(Y\mid X)\) a changé.
Comparaison
Type	Ce qui change	Exemple
Data drift	\(P(X)\)	Nouvelle répartition des prix
Concept drift	\(P(Y\mid X)\)	Relation prix-demande modifiée
Label drift	\(P(Y)\)	Distribution des ventes différente


## 13. Les principaux indicateurs de drift


## 13.1 PSI — Population Stability Index

Le PSI mesure une différence entre deux distributions discrétisées en classes.
\[
\boxed{
PSI=\sum_{i=1}^{K}
(p_i-q_i)\ln\left(\frac{p_i}{q_i}\right)
}
\]

où \(p_i\) et \(q_i\) représentent les proportions dans les classes de deux périodes.

Les proportions nulles nécessitent un traitement numérique approprié.

Des seuils heuristiques souvent rencontrés sont :
PSI	Interprétation indicative
< 0,10	Faible changement
0,10–0,25	Changement à examiner
> 0,25	Changement important potentiel

Ces seuils ne sont pas universels. Ils dépendent notamment du contexte, de la taille d'échantillon et du découpage en classes.

## 13.2 Kolmogorov-Smirnov

Le test KS à deux échantillons compare les distributions cumulées :
\[
\boxed{
D_{n,m}=
\sup_x|F_n(x)-G_m(x)|
}
\]

Il peut détecter des différences de distribution sur une variable numérique.

## 13.3 Wasserstein Distance

La distance de Wasserstein mesure, intuitivement, le coût minimal pour transformer une distribution en une autre.

Elle tient compte des distances entre les valeurs.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Une p-value faible signale une incompatibilité statistique avec l'hypothèse d'égalité des distributions, sous les hypothèses du test. Elle ne mesure pas directement l'importance métier du changement.


## 14. Drift vs dégradation de performance

Un point important : détecter un drift ne signifie pas nécessairement que le modèle est devenu moins performant.

Par exemple, une variable peut changer de distribution sans affecter les prévisions.

Inversement, les performances peuvent se dégrader alors qu'aucun changement important n'est détecté sur les distributions marginales des variables.

Il est donc utile de surveiller simultanément :
- Les distributions des features.
- Les distributions des prédictions.
- Les erreurs de forecasting lorsque les observations réelles deviennent disponibles.
- La couverture des intervalles probabilistes.
- Les changements de régime.
Surveillance d'un modèle de forecasting


## 15. Mini-projet : détecter anomalies et changements de régime

Nous allons construire une série synthétique contenant :
- Une tendance légère.
- Une saisonnalité hebdomadaire.
- Deux anomalies ponctuelles.
- Une rupture durable du niveau moyen.

## Étape 1 — Générer la série


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 2 — Détecter les anomalies avec rolling Z-score


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 3 — Détecter les ruptures avec PELT

Comme notre série présente une tendance et une saisonnalité, nous allons commencer par les retirer approximativement.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le résultat dépend de la pénalité et des propriétés du signal. La décomposition STL est ici utilisée pour une analyse rétrospective.

## Étape 4 — Visualiser


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 5 — Interpréter les résultats

Le détecteur d'anomalies peut signaler les pics isolés, mais également les premières observations du nouveau régime.

Le détecteur de ruptures cherche plutôt à identifier la modification persistante autour de l'indice 250.

Cette distinction est importante : un changement de régime ne doit pas nécessairement produire une alerte d'anomalie à chaque nouvelle observation.


## 16. Évaluer un détecteur d'anomalies

Lorsque des labels d'anomalie sont disponibles, nous pouvons utiliser des métriques de classification.
Precision
\[
Precision=\frac{TP}{TP+FP}
\]

Elle mesure la proportion des alertes qui correspondent réellement à des anomalies.
Recall
\[
Recall=\frac{TP}{TP+FN}
\]

Elle mesure la proportion des anomalies détectées.
F1-score
\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]

Mais pour les séries temporelles, ces métriques ne suffisent pas toujours.

Il faut également considérer :
- Le délai de détection.
- Le nombre de fausses alertes par période.
- La capacité à détecter un événement complet.
- Le coût opérationnel des alertes.
Exemple
Un détecteur qui signale une rupture cinq jours après son apparition peut être moins utile qu'un détecteur qui l'identifie dès le premier jour, même si leurs scores de classification sont proches.

Il faut donc définir une tolérance temporelle cohérente avec le métier.


## 17. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Utiliser un Z-score global sur une série saisonnière	Fausses alertes	Modéliser la saisonnalité
Calculer les seuils avec le futur	Fuite temporelle	Utiliser l'historique disponible
Confondre anomalie et rupture	Alertes répétitives	Distinguer événement et régime
Utiliser une pénalité PELT arbitraire	Sur-segmentation ou sous-segmentation	Calibrer la pénalité
Assimiler drift et baisse de performance	Diagnostic incorrect	Surveiller les deux
Ignorer les retards des labels	Évaluation irréaliste	Respecter les dates de disponibilité
Réentraîner automatiquement à chaque alerte	Risque de dégradation	Prévoir une investigation
Évaluer uniquement avec F1	Délai ignoré	Ajouter les métriques temporelles
Ignorer les événements métier	Alertes peu pertinentes	Ajouter du contexte

[[quiz]]


## 19. Synthèse du cours

Les concepts essentiels à retenir :
- Anomalie ponctuelle : observation isolée inhabituelle.
- Anomalie contextuelle : valeur anormale dans un contexte particulier.
- Anomalie collective : séquence présentant un comportement inhabituel.
- Z-score, IQR et MAD : méthodes statistiques de détection.
- Rolling Statistics : adaptation à l'historique récent.
- STL : décomposition permettant d'analyser les résidus.
- Isolation Forest : méthode non supervisée basée sur l'isolation des observations.
- CUSUM : surveillance séquentielle des changements persistants.
- PELT : segmentation permettant d'identifier plusieurs ruptures.
- Data Drift : évolution de la distribution des variables.
- Concept Drift : évolution de la relation entre variables et cible.
- Monitoring : surveillance conjointe des données, performances et incertitudes.
## Section A du cours — Séries temporelles terminée !


## 7 cours sur 7 : fondamentaux, baselines, modèles statistiques, ML forecasting, multi-séries, prévision probabiliste, anomalies et drift.


Prochain cours — Partie B : Machine Learning spécialisé
Cours 08 — Ensemble Learning avancé : Voting, Bagging, Stacking et Blending

Partie B · 01/05

Nous allons apprendre à combiner plusieurs modèles pour améliorer la généralisation, comprendre le rôle des prédictions Out-of-Fold (OOF), construire un StackingClassifier et éviter les fuites de données lors de l'entraînement du méta-modèle.



## Notes sur l’implémentation et le protocole

Rolling Z-score et CUSUM utilisent l’historique autorisé. STL et PELT sont présentés comme analyses rétrospectives. Les seuils de démonstration sont fixés à l’avance ; leur calibration opérationnelle exige une période dédiée. Une p-value KS usuelle ne tient pas automatiquement compte de la dépendance sérielle. Une alerte ne déclenche pas un réentraînement automatique.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np
import pandas as pd

def rolling_zscore(values,window=28):
    series=pd.Series(values,dtype=float).copy()
    if not isinstance(window,int) or window<2:raise ValueError('Fenêtre entière >= 2 requise.')
    past=series.shift(1);mean=past.rolling(window).mean();std=past.rolling(window).std()
    return (series-mean)/std.replace(0,np.nan)

def robust_zscore(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Vecteur fini non vide requis.')
    median=np.median(values);mad=np.median(np.abs(values-median))
    if mad==0:return np.where(values==median,0.,np.copysign(np.inf,values-median))
    return .6745*(values-median)/mad

def cusum_detector(values,target_mean,k,h):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite([target_mean,k,h]).all() or k<0 or h<=0:
        raise ValueError('Données finies, k>=0 et h>0 requis.')
    positive=negative=0.;alerts=[]
    for t,value in enumerate(values):
        positive=max(0.,positive+value-target_mean-k)
        negative=max(0.,negative+target_mean-value-k)
        if positive>h or negative>h:
            alerts.append(t);positive=negative=0.
    return alerts

def population_stability_index(reference,current,bins=10,epsilon=1e-6):
    reference,current=[np.asarray(v,dtype=float) for v in (reference,current)]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in (reference,current)) or bins<2 or epsilon<=0:
        raise ValueError('Distributions finies non vides et paramètres valides requis.')
    # Frontières apprises sur la référence ; bornes infinies pour conserver les nouvelles valeurs extrêmes.
    inner=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)[1:-1]))
    edges=np.r_[-np.inf,inner,np.inf]
    p=np.histogram(reference,edges)[0].astype(float);q=np.histogram(current,edges)[0].astype(float)
    p=np.maximum(p/p.sum(),epsilon);q=np.maximum(q/q.sum(),epsilon);p/=p.sum();q/=q.sum()
    return float(np.sum((p-q)*np.log(p/q)))

def run(plot=False):
    from statsmodels.tsa.seasonal import STL
    from sklearn.ensemble import IsolationForest
    from scipy.stats import ks_2samp,wasserstein_distance
    import ruptures as rpt
    rng=np.random.default_rng(42);n=400;t=np.arange(n)
    values=100+.02*t+8*np.sin(2*np.pi*t/7)+rng.normal(0,3,n)
    values[250:]+=25;values[80]+=45;values[320]-=40
    df=pd.DataFrame({'value':values},index=pd.date_range('2025-01-01',periods=n,freq='D'))
    df['z']=rolling_zscore(df['value'],28);df['anomaly']=df['z'].abs()>3
    reference=df['value'].iloc[:60]
    alerts=cusum_detector(df['value'],reference.mean(),k=4,h=35)
    print('Alertes CUSUM, seuils fixés à l’avance :',alerts)
    # Rétrospectif : STL et PELT exploitent la série complète.
    stl=STL(df['value'],period=7,robust=True).fit();signal=df['value'].to_numpy()-stl.seasonal.to_numpy()
    breakpoints=rpt.Pelt(model='l2',min_size=30).fit(signal).predict(pen=300)
    binary=rpt.Binseg(model='l2',min_size=30).fit(signal).predict(n_bkps=1)
    print('PELT rétrospectif (fin du signal incluse) :',breakpoints,'Binary Segmentation :',binary)
    df['rolling_mean']=df['value'].shift(1).rolling(28).mean();df['rolling_std']=df['value'].shift(1).rolling(28).std()
    features=['value','rolling_mean','rolling_std'];clean=df.dropna(subset=features)
    train=clean.loc[clean.index<df.index[70]];test=clean.loc[clean.index>=df.index[70]]
    model=IsolationForest(n_estimators=100,contamination=.03,random_state=42).fit(train[features])
    print('Isolation Forest :',int((model.predict(test[features])==-1).sum()),'alertes après observation courante.')
    current=df['value'].iloc[300:]
    print('Drift descriptif : PSI',population_stability_index(reference,current),'KS',ks_2samp(reference,current).statistic,
          'Wasserstein',wasserstein_distance(reference,current))
    print('Tests KS usuels : dépendance sérielle et tests multiples à considérer avant interprétation des p-values.')
    actual=np.zeros(n,dtype=bool);actual[[80,320]]=True;detected=df['anomaly'].to_numpy()
    tp=np.sum(actual&detected);fp=np.sum(~actual&detected);fn=np.sum(actual&~detected)
    print('Pics injectés : TP/FP/FN',tp,fp,fn,'; changement durable évalué séparément autour de 250.')
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5));plt.plot(df.index,values,label='Signal')
        plt.scatter(df.index[detected],values[detected],color='red',label='Rolling Z')
        for bp in breakpoints[:-1]:plt.axvline(df.index[bp],color='orange',linestyle='--')
        plt.legend();plt.tight_layout();plt.show()
    return {'breakpoints':breakpoints,'alerts':alerts}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
~~~

## Challenge Python

Compléter rolling_zscore, cusum_detector et population_stability_index. Calculer les seuils glissants sur le passé ; réinitialiser CUSUM après alerte ; apprendre les classes PSI sur la référence et conserver les valeurs nouvelles extrêmes.

[[assessment]]

[[interview]]
