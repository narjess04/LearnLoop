Partie B · Machine Learning spécialisé · Cours 09/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[costdecision]]

Cours 09 — Cost-Sensitive Learning et prise de décision optimale
LearnLoop · Partie B — Machine Learning spécialisé · Cours 09/12

Niveau : Intermédiaire → Avancé · Durée : 3–4 h

Dans le cours précédent, nous avons appris à combiner plusieurs modèles avec le Voting, le Stacking et le Blending pour améliorer leurs prédictions.

Mais un modèle avec une excellente AUC ou une accuracy de 95 % ne prend pas nécessairement les meilleures décisions.

Pourquoi ? Parce que toutes les erreurs de prédiction n'ont pas le même coût.

Dans un système de détection de fraude, laisser passer une transaction frauduleuse peut être beaucoup plus coûteux que signaler une transaction légitime. Dans un système de maintenance prédictive, manquer une panne peut entraîner un arrêt de production important.

Le Cost-Sensitive Learning consiste à intégrer ces conséquences dans l'apprentissage, l'évaluation et surtout la prise de décision.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Distinguer performance statistique et performance économique.
- Construire et interpréter une matrice de coûts.
- Comprendre les coûts associés aux faux positifs et faux négatifs.
- Déterminer un seuil de décision optimal sous des hypothèses explicites.
- Optimiser les décisions à partir des probabilités prédites.
- Comprendre le rôle de class_weight et sample_weight.
- Utiliser les courbes coût-seuil et les courbes de décision.
- Calculer une valeur économique attendue.
- Comparer des modèles selon leur impact métier.
- Concevoir un pipeline de décision sans fuite de données.

## Section A du cours — Comprendre les coûts des erreurs

## 2. Pourquoi l'accuracy peut être trompeuse

Imaginons un dataset contenant 10 000 transactions.

Parmi elles :
- 9 900 transactions légitimes.
- 100 transactions frauduleuses.
Un modèle qui prédit systématiquement « légitime » obtient :
\[
Accuracy=\frac{9900}{10000}=99\%
\]

Distribution des transactions


## 99 % légitimes



## 1 % frauduleuses


Malgré ses 99 % d'accuracy, ce modèle ne détecte aucune fraude.

Il possède donc :
\[
Recall_{\text{fraude}}=0
\]

Si chaque fraude manquée entraîne une perte de 500 €, le coût total des erreurs serait :
\[
100\times500=50\,000\text{ €}
\]

L'accuracy seule ne permet pas de comprendre cet impact.

Le problème n'est pas uniquement de prédire correctement. Il faut également prendre en compte les conséquences des décisions.


## 3. Rappel : la matrice de confusion

En classification binaire, nous avons quatre résultats possibles.
	Réalité : négatif	Réalité : positif
Prédiction : négatif	TN	FN
Prédiction : positif	FP	TP

Avec :
- TP — True Positive : un positif correctement détecté.
- TN — True Negative : un négatif correctement identifié.
- FP — False Positive : un négatif incorrectement signalé.
- FN — False Negative : un positif manqué.
Dans une classification classique, nous utilisons ces quantités pour calculer Precision, Recall, F1-score et Accuracy.

Dans le Cost-Sensitive Learning, nous allons leur associer des coûts.

## 4. Construire une matrice de coûts

Supposons un système de détection de fraude.

Nous définissons les coûts métier suivants, purement illustratifs :
Décision	Transaction légitime	Transaction frauduleuse
Ne pas signaler	0 €	500 €
Signaler	10 €	20 €

Interprétation :
- Un vrai négatif ne génère aucun coût additionnel.
- Un faux négatif entraîne une perte de 500 €.
- Un faux positif entraîne 10 € de coût opérationnel ou de friction.
- Un vrai positif entraîne 20 € de coût de traitement.
Cette matrice est plus informative qu'une simple classification correcte ou incorrecte.
Formulation générale
On définit :
\[
C(a,y)
\]

où :
- \(a\) est l'action prise.
- \(y\) est l'état réel.
- \(C(a,y)\) est le coût associé.
Le coût total observé est :
\[
\boxed{
C_{\text{total}}=
\sum_{i=1}^{n}C(a_i,y_i)
}
\]

Si les coûts sont constants pour chaque case :
\[
\boxed{
C_{\text{total}}=
C_{TN}TN+C_{FP}FP+C_{FN}FN+C_{TP}TP
}
\]

Exemple
Supposons :
\[
TN=940,\quad FP=40,\quad FN=5,\quad TP=15
\]

Avec notre matrice de coûts :
\[
C_{\text{total}}
=
940(0)+40(10)+5(500)+15(20)
\]

\[
\boxed{C_{\text{total}}=3200\text{ €}}
\]

La matrice de confusion permet donc de traduire les prédictions en conséquences économiques.

## Section B du cours — Optimisation du seuil de décision

## 5. Pourquoi le seuil 0,5 n'est pas toujours optimal

Un classifieur probabiliste produit une estimation :
\[
\hat p(x)=P(Y=1\mid X=x)
\]

Par défaut, nous utilisons souvent :
\[
\hat y=
\begin{cases}
1,&\hat p\geq0.5\\
0,&\hat p<0.5
\end{cases}
\]

Mais le seuil 0,5 n'est optimal que sous certaines hypothèses de coûts.

Lorsque les faux négatifs sont beaucoup plus coûteux que les faux positifs, un seuil plus faible peut être préférable.
L’interaction au début de ce support permet d’explorer cet exemple.

Tu peux constater qu'un seuil plus faible augmente généralement le nombre de transactions signalées.

Cela peut réduire les faux négatifs, mais augmenter les faux positifs.

Le meilleur seuil dépend donc de l'équilibre entre ces coûts.


## 6. Dériver le seuil optimal mathématiquement

Nous allons maintenant comprendre comment déterminer un seuil optimal à partir des coûts.

Supposons deux actions :
- \(a=0\) : ne pas signaler.
- \(a=1\) : signaler.
Et une probabilité calibrée :
\[
p=P(Y=1\mid X=x)
\]

Coût attendu de l'action 0
\[
R(0\mid x)
=
C_{TN}(1-p)+C_{FN}p
\]

Coût attendu de l'action 1
\[
R(1\mid x)
=
C_{FP}(1-p)+C_{TP}p
\]

Nous choisissons l'action 1 si :
\[
R(1\mid x)<R(0\mid x)
\]

Après réarrangement, si le dénominateur est positif :
\[
\boxed{
p>
\frac{C_{FP}-C_{TN}}
{(C_{FP}-C_{TN})+(C_{FN}-C_{TP})}
}
\]

C'est le seuil de décision bayésien pour ces deux actions et ces coûts constants.
Cas simplifié
Si :
\[
C_{TN}=C_{TP}=0
\]

Alors :
\[
\boxed{
t^*=
\frac{C_{FP}}{C_{FP}+C_{FN}}
}
\]

Application à notre exemple
Avec :
\[
C_{FP}=10,\quad C_{FN}=500
\]

et :
\[
C_{TN}=0,\quad C_{TP}=20
\]

Nous obtenons :
\[
t^*=
\frac{10}{10+(500-20)}
\]

\[
\boxed{t^*\approx0.0204}
\]

Le seuil théorique est donc d'environ 2,04 %.

Cela peut sembler très faible, mais c'est cohérent avec le fait que manquer une fraude est extrêmement coûteux dans notre exemple.

Condition essentielle : cette formule suppose que les probabilités sont correctement calibrées et que les coûts reflètent bien les conséquences réelles.

Si les coûts dépendent de chaque transaction, une règle de décision individuelle est généralement préférable à un seuil global.


## 7. Optimiser le seuil empiriquement

Dans la pratique, nous pouvons tester plusieurs seuils sur un ensemble de validation.

L'objectif est de sélectionner :
\[
\boxed{
t^*=\arg\min_t C_{\text{validation}}(t)
}
\]

Implémentation Python

~~~python
from ch06_cours09_lab import binary_inputs, compute_cost, bayes_threshold, optimal_actions, select_threshold, net_benefit

def compute_cost(y_true,y_proba,threshold,cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    y,probabilities=binary_inputs(y_true,y_proba)
    if not np.isfinite([threshold,cost_fp,cost_fn,cost_tp,cost_tn]).all():raise ValueError('Seuil et coûts finis requis.')
    action=(probabilities>=threshold).astype(int)
    costs=np.array([[cost_tn,cost_fn],[cost_fp,cost_tp]],dtype=float)
    return float(costs[action,y.astype(int)].sum())
~~~

Règle importante
Le seuil doit être choisi sur la validation, jamais sur le test final.

Sinon, nous utilisons indirectement les résultats du test pour optimiser notre système.

Cela introduit un biais d'évaluation.


## 8. Courbe coût-seuil

La courbe coût-seuil montre comment le coût total varie selon le seuil de classification.

Exemple illustratif, non issu d'un entraînement réel.

Dans cet exemple fictif, le coût minimal est atteint autour de 0,20.

Ce seuil empirique peut différer du seuil théorique en raison d'une calibration imparfaite, de la taille de l'échantillon ou de la structure des coûts.

## Section C du cours — Cost-Sensitive Learning pendant l'entraînement

## 9. Agir sur le seuil ou sur l'apprentissage ?

Il existe deux grandes stratégies.
Stratégie 1 — Post-processing
Nous entraînons un modèle normalement, puis nous ajustons le seuil de décision.
\[
X\rightarrow \hat p(X)\rightarrow \text{Décision optimale}
\]

Cette approche est souvent intéressante lorsque les probabilités sont correctement calibrées.
Stratégie 2 — Cost-sensitive training
Nous modifions la fonction de perte pour accorder davantage d'importance à certaines observations ou classes.
\[
\boxed{
L_{\text{weighted}}
=
\sum_{i=1}^{n}w_iL(y_i,\hat y_i)
}
\]

où \(w_i\) représente le poids de l'observation.

L'objectif est d'influencer directement le comportement du modèle pendant l'entraînement.


## 10. Class Weights

Dans Scikit-learn, plusieurs algorithmes acceptent le paramètre class_weight.
Exemple avec Logistic Regression

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Ici, les observations de la classe positive reçoivent un poids dix fois supérieur à celui de la classe négative.
Pourquoi ?
Le modèle est davantage pénalisé lorsqu'il se trompe sur une observation positive.

Cela peut être utile lorsque les erreurs sur la classe positive sont particulièrement importantes.
Attention
class_weight ne correspond pas automatiquement à la matrice complète des coûts métier.

De plus, modifier les poids peut affecter la calibration des probabilités.

Il faut donc évaluer les probabilités produites et, si nécessaire, les recalibrer sur des données indépendantes.


## 11. Sample Weights

Les coûts peuvent varier d'une observation à l'autre.

Par exemple, dans un système de fraude, le montant des transactions peut être très différent.

Une fraude de 20 € et une fraude de 5 000 € n'ont pas nécessairement les mêmes conséquences.

Nous pouvons attribuer des poids individuels.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Dans un problème réel, les poids peuvent être construits à partir des coûts métier, avec une attention particulière à leur échelle et à leur variabilité.
Différence importante
Approche	Ce qu'elle modifie
Threshold tuning	Règle de décision
Class weights	Importance des classes pendant l'entraînement
Sample weights	Importance des observations
Calibration	Fiabilité des probabilités
Cost-sensitive evaluation	Mesure de la valeur métier

Ces approches peuvent être complémentaires.

## Section D du cours — Decision Curves et valeur économique

## 12. Decision Curve Analysis

La Decision Curve Analysis (DCA) permet d'évaluer l'utilité d'un modèle à différents seuils de décision.

Elle est particulièrement utilisée dans certains contextes de décision clinique, mais son principe peut être adapté à d'autres domaines lorsque le seuil possède une interprétation cohérente en termes de bénéfices et de risques.

Une formulation classique du bénéfice net est :
\[
\boxed{
NB(t)=
\frac{TP(t)}{N}
-
\frac{FP(t)}{N}
\frac{t}{1-t}
}
\]

où :
- \(NB(t)\) : bénéfice net.
- \(t\) : seuil de probabilité.
- \(TP(t)\) : nombre de vrais positifs.
- \(FP(t)\) : nombre de faux positifs.
- \(N\) : nombre d'observations.
Le terme :
\[
\frac{t}{1-t}
\]

représente un rapport implicite entre les conséquences des faux positifs et les bénéfices des vrais positifs, dans le cadre décisionnel de la DCA.
Trois stratégies à comparer
- Model strategy : utiliser les prédictions du modèle.
- Treat all : appliquer l'action positive à toutes les observations.
- Treat none : ne jamais appliquer l'action positive.
Un modèle est utile, dans ce cadre, lorsque son bénéfice net est supérieur à celui des stratégies de référence pour les seuils pertinents.
Implémentation Python

~~~python
from ch06_cours09_lab import binary_inputs, compute_cost, bayes_threshold, optimal_actions, select_threshold, net_benefit

def net_benefit(y_true,y_proba,threshold):
    y,proba=binary_inputs(y_true,y_proba)
    if not 0<threshold<1:raise ValueError('Seuil strictement entre 0 et 1 requis.')
    action=proba>=threshold
    tp=np.sum(action&(y==1));fp=np.sum(action&(y==0))
    return float(tp/len(y)-fp/len(y)*threshold/(1-threshold))
~~~

Les trois courbes peuvent ensuite être tracées avec Matplotlib.

Il faut éviter d'interpréter directement le bénéfice net comme un montant monétaire. Pour une analyse financière, une matrice explicite de coûts et bénéfices est généralement plus adaptée.


## 13. Expected Value — Valeur économique attendue

Jusqu'ici, nous avons minimisé les coûts.

Nous pouvons également chercher à maximiser une valeur économique attendue.

Pour une action \(a\) :
\[
\boxed{
EV(a\mid x)
=
\sum_y P(Y=y\mid X=x)V(a,y)
}
\]

où \(V(a,y)\) représente la valeur associée à la décision.

La décision optimale devient :
\[
\boxed{
a^*(x)=\arg\max_a EV(a\mid x)
}
\]

Exemple : campagne de rétention
Une entreprise souhaite contacter les clients susceptibles de partir.

Supposons :
- Coût d'une offre : 10 €.
- Valeur générée si un client à risque est effectivement retenu : 100 €.
- Probabilité de départ estimée : \(p\).
- Probabilité que l'offre évite le départ d'un client qui serait parti : 30 %.
Sous l'hypothèse simplificatrice que l'offre n'a pas d'autre effet, la valeur attendue de l'action est :
\[
EV(\text{offre})
=
p\times0.30\times100-10
\]

Donc :
\[
EV(\text{offre})=30p-10
\]

L'action devient rentable lorsque :
\[
30p>10
\]

\[
\boxed{p>\frac13}
\]

Le seuil de décision serait donc d'environ 33,3 %.
Limite essentielle
Cette analyse suppose que la probabilité de départ et l'effet de l'offre sont correctement estimés.

Dans un véritable système de rétention, il faut idéalement estimer l'effet causal de l'intervention plutôt que supposer que tous les clients à risque réagissent de la même façon.

Un client qui aurait conservé son abonnement sans offre ne génère pas le même bénéfice qu'un client réellement sauvé par l'intervention.

## Section E du cours — Mini-projet complet

## 14. Objectif du mini-projet

Nous allons construire un système de classification binaire qui :

## 1. Entraîne un modèle probabiliste.


## 2. Évalue sa discrimination.


## 3. Vérifie sa calibration.


## 4. Compare plusieurs seuils de décision.


## 5. Sélectionne un seuil minimisant le coût métier.


## 6. Évalue le résultat sur un test indépendant.

Nous utiliserons un dataset synthétique déséquilibré.

## Étape 1 — Générer les données


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Nous obtenons trois ensembles distincts :
- Train : entraînement du modèle.
- Validation : sélection du seuil.
- Test : évaluation finale.

## Étape 2 — Entraîner un modèle


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 3 — Produire les probabilités


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 4 — Vérifier la qualité des probabilités


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

L'AUC mesure la capacité de discrimination.

L'Average Precision est particulièrement informative pour certaines classifications déséquilibrées.

Le Brier Score et la Log Loss évaluent la qualité des probabilités, mais ne remplacent pas une inspection de la calibration.

## Étape 5 — Chercher le meilleur seuil


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 6 — Comparer le seuil 0,5 au seuil optimisé


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 7 — Comparer les coûts sur test


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le seuil optimisé sur validation n'est pas garanti d'être meilleur sur chaque échantillon de test.

L'objectif est d'estimer si la stratégie se généralise à de nouvelles données.


## 15. Comment choisir entre deux modèles ?

Imaginons deux modèles évalués sur les mêmes données.
Métrique	Modèle A	Modèle B
ROC-AUC	0,94	0,91
Recall	0,70	0,85
Precision	0,80	0,60
Coût total	12 000 €	8 000 €

Ces chiffres sont fictifs.

Le modèle A possède une meilleure AUC.

Mais le modèle B entraîne un coût métier inférieur.

Si la matrice de coûts est pertinente et l'évaluation fiable, le modèle B peut donc être préférable.

Cela ne signifie pas que l'AUC est inutile. Elle reste importante pour évaluer la capacité de discrimination indépendamment d'un seuil.

En pratique, il est pertinent de comparer les modèles à la fois sur leur performance statistique, leur calibration, leur coût économique et leurs contraintes opérationnelles.


## 16. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Utiliser systématiquement 0,5	Décisions sous-optimales	Ajuster le seuil
Optimiser uniquement l'accuracy	Coûts ignorés	Évaluer la matrice de coûts
Choisir le seuil sur test	Data leakage	Utiliser validation
Utiliser des probabilités mal calibrées	Mauvaises décisions	Vérifier la calibration
Confondre class weights et coûts métier	Objectif mal spécifié	Séparer entraînement et décision
Ignorer les coûts des TP	Coût total sous-estimé	Modéliser toutes les actions
Utiliser des coûts arbitraires	Décisions peu pertinentes	Travailler avec le métier
Ignorer les coûts variables	Seuil global inadapté	Décisions individuelles
Ignorer le drift	Seuil devenu obsolète	Monitoring périodique
Confondre risque et effet d'une action	Mauvaise estimation du ROI	Évaluer l'impact causal

[[quiz]]


## 18. Synthèse du cours

Les notions essentielles à retenir :
- Cost-Sensitive Learning : intégrer les conséquences des erreurs dans l'apprentissage et la décision.
- Matrice de coûts : associer un coût à chaque combinaison action-réalité.
- Threshold tuning : sélectionner un seuil adapté à l'objectif métier.
- Seuil bayésien : déterminer une règle optimale à partir de probabilités calibrées et de coûts connus.
- Class weights : pondérer les classes pendant l'entraînement.
- Sample weights : pondérer les observations individuellement.
- Decision Curve Analysis : comparer des stratégies selon leur bénéfice net.
- Expected Value : choisir l'action maximisant la valeur économique attendue.
- Calibration : condition importante pour utiliser correctement les probabilités dans la décision.
- Data leakage : ne jamais optimiser le seuil sur le test final.
Prochain cours
Cours 10 — Survival Analysis : Kaplan-Meier, Cox et modèles de survie

Partie B · 03/05

Nous allons étudier une famille de modèles très utile pour prédire le temps avant un événement : churn, panne, défaut ou autre événement défini. Nous aborderons les données censurées, les fonctions de survie, Kaplan-Meier, le modèle de Cox, les modèles de survie ML et le C-index.


## Notes sur l’implémentation et le protocole

La recherche du seuil inclut les politiques tout signaler et ne rien signaler. La calibration croisée est effectuée dans le train ; la validation reste disponible pour le seuil. Les coûts sont illustratifs et ne constituent pas une recommandation financière. Le bénéfice net DCA n’est pas un montant monétaire.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np

def binary_inputs(y,probabilities):
    y=np.asarray(y);probabilities=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or not len(y) or probabilities.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Cible binaire et probabilités alignées requises.')
    return y,probabilities

def compute_cost(y_true,y_proba,threshold,cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    y,probabilities=binary_inputs(y_true,y_proba)
    if not np.isfinite([threshold,cost_fp,cost_fn,cost_tp,cost_tn]).all():raise ValueError('Seuil et coûts finis requis.')
    action=(probabilities>=threshold).astype(int)
    costs=np.array([[cost_tn,cost_fn],[cost_fp,cost_tp]],dtype=float)
    return float(costs[action,y.astype(int)].sum())

def bayes_threshold(cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    denominator=(cost_fp-cost_tn)+(cost_fn-cost_tp)
    if denominator<=0:raise ValueError('Dénominateur positif requis ; comparer directement les risques sinon.')
    return (cost_fp-cost_tn)/denominator

def optimal_actions(probabilities,costs):
    probabilities=np.asarray(probabilities,dtype=float);costs=np.asarray(costs,dtype=float)
    if probabilities.ndim!=1 or costs.shape!=(2,2) or not np.isfinite(costs).all() or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Probabilités et matrice action x état 2x2 requises.')
    risks=(1-probabilities[:,None])*costs[:,0]+probabilities[:,None]*costs[:,1]
    return risks.argmin(axis=1)

def select_threshold(y_val,proba_val,**costs):
    y_val,proba_val=binary_inputs(y_val,proba_val)
    # Inclure toutes les politiques distinctes, notamment signaler tout / rien.
    thresholds=np.unique(np.r_[0.,proba_val,np.nextafter(1.,np.inf)])
    values=np.array([compute_cost(y_val,proba_val,t,**costs) for t in thresholds])
    index=int(np.argmin(values))
    return float(thresholds[index]),float(values[index])

def net_benefit(y_true,y_proba,threshold):
    y,proba=binary_inputs(y_true,y_proba)
    if not 0<threshold<1:raise ValueError('Seuil strictement entre 0 et 1 requis.')
    action=proba>=threshold
    tp=np.sum(action&(y==1));fp=np.sum(action&(y==0))
    return float(tp/len(y)-fp/len(y)*threshold/(1-threshold))

def run(plot=False):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV,calibration_curve
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix
    X,y=make_classification(n_samples=4500,n_features=20,n_informative=12,n_redundant=4,weights=[.95,.05],random_state=42)
    dev,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
    # Calibration croisée dans le train ; validation indépendante pour le seuil.
    model=CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,random_state=42),method='sigmoid',cv=3)
    model.fit(X[train],y[train]);pv=model.predict_proba(X[val])[:,1]
    print('Validation AUC/AP/Brier/Log Loss :',roc_auc_score(y[val],pv),average_precision_score(y[val],pv),brier_score_loss(y[val],pv),log_loss(y[val],pv))
    threshold,cost=select_threshold(y[val],pv)
    print('Seuil choisi sur validation :',threshold,'Coût validation :',cost,'Seuil théorique :',bayes_threshold())
    pt=model.predict_proba(X[test])[:,1]
    # Audit final des règles fixées : aucune sélection sur le test.
    result={}
    for name,t in [('Défaut .5',.5),('Seuil validation',threshold),('Théorique',bayes_threshold())]:
        result[name]=compute_cost(y[test],pt,t);print(name,'coût test',result[name],confusion_matrix(y[test],pt>=t,labels=[0,1]).tolist())
    print('Ne rien signaler / tout signaler :',compute_cost(y[test],pt,np.nextafter(1.,np.inf)),compute_cost(y[test],pt,0))
    print('Bénéfice net à .1 :',net_benefit(y[test],pt,.1),'(sans unité monétaire).')
    if plot:
        import matplotlib.pyplot as plt
        thresholds=np.linspace(0,1,101);plt.plot(thresholds,[compute_cost(y[val],pv,t) for t in thresholds])
        plt.axvline(threshold,linestyle='--');plt.xlabel('Seuil');plt.ylabel('Coût sur validation');plt.tight_layout();plt.show()
        observed,predicted=calibration_curve(y[val],pv,n_bins=8,strategy='quantile')
        plt.plot(predicted,observed,'o-');plt.plot([0,1],[0,1],'--');plt.title('Calibration sur validation');plt.show()
    return {'threshold':threshold,'test_costs':result}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
~~~

## Challenge Python

Compléter compute_cost, optimal_actions et select_threshold. Utiliser une matrice action x état, comparer les risques attendus et inclure les décisions tout signaler / ne rien signaler dans la recherche sur validation.

[[assessment]]

[[interview]]
