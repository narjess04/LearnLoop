Partie B · Machine Learning spécialisé · Cours 08/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[voting]]

Cours 08 — Ensemble Learning avancé : Voting, Stacking et Blending
LearnLoop · Partie B — Machine Learning spécialisé · Cours 08/12

Niveau : Intermédiaire → Avancé · Durée estimée : 3–4 h

Dans les chapitres précédents, nous avons étudié plusieurs algorithmes de Machine Learning : régression logistique, Random Forest, XGBoost, LightGBM, SVM et bien d'autres.

Nous avons également découvert le Bagging et le Boosting.

Mais une question importante se pose :

Et si, au lieu de sélectionner un seul modèle, nous combinions plusieurs modèles pour obtenir de meilleures prédictions ?

C'est le principe de l'Ensemble Learning.

Dans ce cours, nous allons approfondir trois approches particulièrement importantes : Voting, Stacking et Blending, avec une attention particulière aux prédictions Out-of-Fold et aux risques de fuite de données.


## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Comprendre pourquoi combiner plusieurs modèles peut améliorer la généralisation.
- Distinguer Voting, Bagging, Boosting, Stacking et Blending.
- Implémenter le Hard Voting et le Soft Voting.
- Construire un ensemble de modèles diversifiés.
- Comprendre le fonctionnement d'un méta-modèle.
- Générer correctement des prédictions Out-of-Fold (OOF).
- Implémenter un Stacking manuel et avec Scikit-learn.
- Comprendre les différences entre Stacking et Blending.
- Évaluer un ensemble sans introduire de data leakage.
- Identifier les situations où un ensemble n'apporte aucun avantage.

## 2. Pourquoi utiliser plusieurs modèles ?

Imaginons un problème de classification binaire : prédire si un client risque de quitter un service.

Nous entraînons trois modèles :
- Une régression logistique.
- Un Random Forest.
- Un LightGBM.
Chaque modèle possède ses forces et ses faiblesses.

La régression logistique apprend des relations linéaires dans l'espace des variables transformées.

Le Random Forest capture des interactions non linéaires.

LightGBM construit une succession d'arbres capables de modéliser des relations complexes.

Leurs prédictions peuvent donc être différentes.
Exemple
Client	Régression logistique	Random Forest	LightGBM
A	0,20	0,15	0,25
B	0,75	0,80	0,70
C	0,40	0,65	0,55
D	0,90	0,85	0,95

Les valeurs représentent les probabilités prédites pour la classe positive.

Pour le client C, les modèles ne sont pas entièrement d'accord.

L'Ensemble Learning cherche à exploiter ces différences.
L'idée fondamentale : la diversité
Combiner plusieurs modèles très similaires n'apporte pas nécessairement un gain.

Les ensembles sont particulièrement intéressants lorsque les modèles sont individuellement compétents et que leurs erreurs ne sont pas parfaitement corrélées.

Pour comprendre cette intuition, considérons la moyenne des erreurs de \(M\) modèles de même variance \(\sigma^2\), avec une corrélation commune \(\rho\) entre leurs erreurs.

Sous ces hypothèses :
\[
\boxed{
\operatorname{Var}(\bar e)
=
\frac{\sigma^2}{M}
\left[1+(M-1)\rho\right]
}
\]

Si les erreurs sont indépendantes :
\[
\rho=0
\]

Alors :
\[
\operatorname{Var}(\bar e)=\frac{\sigma^2}{M}
\]

La variance diminue lorsque nous combinons davantage de modèles.

Mais si les erreurs sont parfaitement corrélées :
\[
\rho=1
\]

Alors :
\[
\operatorname{Var}(\bar e)=\sigma^2
\]

La combinaison ne réduit pas la variance.

Cette analyse explique l'intérêt de la diversité, mais elle ne garantit pas que la moyenne des prédictions améliore toutes les métriques : les biais et les autres caractéristiques des modèles comptent également.


## 3. Les grandes familles d'Ensemble Learning

Vue d'ensemble des principales stratégies

Méthode	Principe	Exemple
Bagging	Entraîner des modèles sur différents échantillons	Random Forest
Boosting	Construire des modèles successifs pour corriger les erreurs	XGBoost, LightGBM
Voting	Combiner directement les prédictions	VotingClassifier
Stacking	Apprendre comment combiner les modèles	StackingClassifier
Blending	Apprendre la combinaison sur un holdout	Meta-model sur validation

Nous avons déjà étudié le Bagging et le Boosting dans le chapitre sur l'apprentissage supervisé.

Nous allons donc nous concentrer principalement sur Voting, Stacking et Blending.

## Section A du cours — Voting

## 4. Hard Voting

Le Hard Voting combine les classes prédites par plusieurs modèles.

Chaque modèle vote pour une classe, et la classe majoritaire est sélectionnée.
Exemple
Pour une observation :
Modèle	Prédiction
Logistic Regression	1
Random Forest	0
SVM	1

La prédiction finale est :
\[
\boxed{\hat y=1}
\]

Deux modèles sur trois ont voté pour la classe 1.
Formulation mathématique
Pour \(M\) classifieurs :
\[
\boxed{
\hat y=
\operatorname{mode}
\left(
\hat y^{(1)},\ldots,\hat y^{(M)}
\right)
}
\]

La fonction mode représente ici la classe la plus fréquemment prédite.
Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Avantages
- Simple à comprendre.
- Facile à implémenter.
- Ne nécessite pas de probabilités.
- Peut réduire certaines erreurs individuelles.
Limites
Tous les modèles ont le même poids par défaut.

Un modèle très performant peut être mis en minorité par deux modèles moins performants.

Les égalités doivent également être gérées selon la convention de l'implémentation.


## 5. Soft Voting

Le Soft Voting combine les probabilités prédites plutôt que les classes finales.

Pour une classification binaire :
\[
\boxed{
\hat p(x)=
\frac{1}{M}
\sum_{m=1}^{M}p_m(x)
}
\]

La classe prédite est ensuite déterminée à partir de la probabilité agrégée.
Exemple
Modèle	Probabilité classe 1
Logistic Regression	0,80
Random Forest	0,60
LightGBM	0,70

Alors :
\[
\hat p=\frac{0.80+0.60+0.70}{3}=0.70
\]

Avec un seuil de 0,5 :
\[
\boxed{\hat y=1}
\]

L’interaction au début de ce support permet d’explorer cet exemple.

Implémentation Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Important : Soft Voting fonctionne mieux lorsque les probabilités sont suffisamment fiables. Des modèles mal calibrés peuvent produire des probabilités excessivement confiantes et dégrader l'agrégation.

Il peut être utile d'évaluer leur calibration avec le Brier Score, la Log Loss et des courbes de calibration.


## 6. Weighted Voting

Nous pouvons donner davantage de poids à certains modèles.

La formule devient :
\[
\boxed{
\hat p(x)=
\frac{\sum_{m=1}^{M}w_mp_m(x)}
{\sum_{m=1}^{M}w_m}
}
\]

où \(w_m\geq0\).
Exemple
Supposons :
\[
w_1=1,\quad w_2=2,\quad w_3=3
\]

Et :
\[
p_1=0.8,\quad p_2=0.6,\quad p_3=0.7
\]

Alors :
\[
\hat p=
\frac{1(0.8)+2(0.6)+3(0.7)}{6}
\]

\[
\boxed{\hat p\approx0.683}
\]

Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Les poids doivent être déterminés à partir de données de validation ou d'une procédure de validation croisée.

Il ne faut pas optimiser les poids sur le jeu de test.

## Section B du cours — Stacking

## 7. Comprendre le Stacking

Le Stacking, ou Stacked Generalization, est une approche plus avancée.

Au lieu de calculer une moyenne des prédictions, nous entraînons un nouveau modèle pour apprendre comment les combiner.

Ce nouveau modèle est appelé :

Meta-model ou Meta-learner.
Architecture du Stacking

Le Stacking possède généralement deux niveaux.
Niveau 0 — Base Learners
Plusieurs modèles apprennent à partir des features initiales.
\[
f_1(X),f_2(X),\ldots,f_M(X)
\]

Niveau 1 — Meta-Learner
Le méta-modèle apprend à partir des prédictions des modèles de base :
\[
\boxed{
\hat y=
g(f_1(X),f_2(X),\ldots,f_M(X))
}
\]

Le méta-modèle peut être une régression logistique, une régression Ridge ou un autre algorithme.
Pourquoi est-ce différent du Voting ?
Le Voting utilise une règle de combinaison définie à l'avance.

Le Stacking apprend cette combinaison à partir des données.

Il peut donc apprendre qu'un modèle est plus utile dans certaines situations qu'un autre.


## 8. Le problème fondamental : Data Leakage

Le Stacking peut introduire une fuite de données si nous construisons incorrectement les données du méta-modèle.
Mauvaise approche
Imaginons que nous :

## 1. Entraînons un Random Forest sur tout le train.


## 2. Calculons ses prédictions sur ce même train.


## 3. Utilisons ces prédictions pour entraîner le méta-modèle.

Problème : les prédictions du Random Forest sont calculées sur des observations qu'il a déjà utilisées pour apprendre.

Le méta-modèle reçoit donc des prédictions potentiellement trop optimistes.

Cela peut provoquer un surapprentissage important.
Solution : Out-of-Fold Predictions
Nous devons générer des prédictions sur des observations que le modèle de base n'a pas utilisées pendant son entraînement.

C'est le principe des prédictions Out-of-Fold (OOF).


## 9. Comprendre les prédictions OOF

Supposons que nous utilisons une validation croisée à cinq folds.

À chaque itération :

## 1. Nous entraînons le modèle sur quatre folds.


## 2. Nous prédisons sur le fold restant.


## 3. Nous enregistrons ces prédictions.


## 4. Nous répétons jusqu'à couvrir toutes les observations.

L’interaction au début de ce support permet d’explorer cet exemple.

Chaque observation reçoit une prédiction provenant d'un modèle qui ne l'a pas vue pendant son entraînement.
Formulation mathématique
Pour une observation \(i\) appartenant au fold \(k\) :
\[
\boxed{
z_i^{(m)}=
f_m^{(-k)}(x_i)
}
\]

où :
- \(f_m^{(-k)}\) est le modèle \(m\) entraîné sans le fold \(k\).
- \(z_i^{(m)}\) est sa prédiction OOF.
La matrice des prédictions OOF est :
\[
Z=
\begin{bmatrix}
z_1^{(1)}&z_1^{(2)}&\cdots&z_1^{(M)}\\
z_2^{(1)}&z_2^{(2)}&\cdots&z_2^{(M)}\\
\vdots&\vdots&\ddots&\vdots\\
z_n^{(1)}&z_n^{(2)}&\cdots&z_n^{(M)}
\end{bmatrix}
\]

Cette matrice devient le dataset d'entraînement du méta-modèle.


## 10. Implémentation manuelle du Stacking

Nous allons construire un StackingClassifier pour un problème de classification binaire.

## Étape 1 — Préparer les données


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 2 — Définir les modèles de base


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Nous utilisons ici trois familles de modèles différentes pour favoriser la diversité.

## Étape 3 — Générer les prédictions OOF


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

La matrice obtenue possède trois colonnes : une probabilité OOF pour chaque modèle.

Le prétraitement de la régression logistique est inclus dans un pipeline, afin qu'il soit ajusté uniquement sur les folds d'entraînement.

## Étape 4 — Entraîner le méta-modèle


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 5 — Réentraîner les modèles de base

Après avoir généré les prédictions OOF, nous réentraînons chaque modèle de base sur l'ensemble d'entraînement complet.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 6 — Produire les prédictions finales


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 7 — Évaluer


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le résultat doit être comparé aux modèles individuels et à un Soft Voting simple.

Un Stacking plus complexe n'est pas automatiquement meilleur.


## 11. Stacking avec Scikit-learn

Scikit-learn fournit une implémentation prête à l'emploi : StackingClassifier.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le paramètre passthrough
Lorsque :

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Le méta-modèle reçoit uniquement les prédictions des modèles de base.

Lorsque :

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Il reçoit également les features originales.

Cela peut être utile si les variables initiales contiennent des informations que les prédictions des modèles de base ne capturent pas entièrement.

Mais cela augmente la dimension des données et peut accroître le risque de surapprentissage.
Attention aux séries temporelles
Le cv=5 utilisé ici convient à notre exemple de classification tabulaire indépendante.

Pour des données temporelles, il faut une validation respectant la chronologie, et éventuellement un gap entre entraînement et validation.

Le Stacking temporel peut nécessiter une implémentation OOF personnalisée, notamment lorsque certaines observations initiales ne peuvent pas recevoir de prédiction de validation chronologique.

## Section C du cours — Blending

## 12. Comprendre le Blending

Le Blending est proche du Stacking.

La différence principale concerne la manière dont nous construisons le dataset du méta-modèle.

Dans le Stacking classique, nous utilisons les prédictions OOF obtenues par validation croisée.

Dans le Blending, nous réservons un ensemble de validation distinct.
Architecture

Étapes

## 1. Diviser le train en deux parties.


## 2. Entraîner les modèles de base sur la première partie.


## 3. Générer leurs prédictions sur le holdout.


## 4. Entraîner le méta-modèle sur ces prédictions.


## 5. Évaluer sur un test totalement indépendant.



## 13. Implémentation du Blending


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Dans cet exemple, les modèles de base restent entraînés sur X_base, ce qui garantit la cohérence avec les prédictions utilisées pour entraîner le méta-modèle.

Il existe des variantes qui réentraînent les modèles sur davantage de données, mais cela modifie la distribution des prédictions reçues par le méta-modèle et doit être validé.
Stacking vs Blending
Critère	Stacking	Blending
Entraînement du méta-modèle	Prédictions OOF	Prédictions sur holdout
Utilisation des données	Généralement plus efficace	Une partie réservée
Coût de calcul	Plus élevé	Souvent plus faible
Complexité	Plus importante	Plus simple
Risque de leakage	Si OOF mal construit	Si holdout réutilisé incorrectement
Usage typique	Ensembles avancés	Prototype rapide ou contraintes de calcul


## 14. Comment mesurer la diversité des modèles ?

La diversité est un élément important de l'Ensemble Learning.

Une approche simple consiste à comparer les erreurs ou les prédictions des modèles sur un ensemble de validation indépendant.
Corrélation des erreurs
Pour deux modèles de régression :
\[
e_i^{(1)}=y_i-\hat y_i^{(1)}
\]

\[
e_i^{(2)}=y_i-\hat y_i^{(2)}
\]

Nous pouvons calculer :
\[
\boxed{
\rho=
Corr(e^{(1)},e^{(2)})
}
\]

Une forte corrélation positive indique que les erreurs ont tendance à évoluer ensemble.

Une corrélation plus faible peut signaler une complémentarité intéressante.
Python

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Pour la classification, nous pouvons également examiner :
- Le taux de désaccord entre modèles.
- La corrélation des probabilités prédites.
- La fréquence des erreurs communes.
- Les performances par segment.
Important : une faible corrélation n'est pas suffisante. Deux modèles très mauvais peuvent être diversifiés sans produire un bon ensemble.


## 15. Mini-projet : comparer Voting, Stacking et modèles individuels

Nous allons maintenant construire un benchmark.

## Étape 1 — Définir les modèles


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 2 — Entraîner les modèles individuels


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 3 — Soft Voting


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 4 — Stacking


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 5 — Comparer les résultats


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Interprétation
Il est possible que :
- Stacking soit le meilleur.
- Soft Voting obtienne des performances similaires.
- Un modèle individuel surpasse tous les ensembles.
Ces trois résultats sont parfaitement plausibles.

Pour une comparaison fiable, il est préférable d'utiliser plusieurs folds externes ou plusieurs périodes de validation, plutôt qu'un seul découpage.

Le test final doit rester réservé à l'évaluation.


## 16. Ensemble Learning et coût de calcul

Un ensemble de modèles peut améliorer les performances, mais il augmente aussi la complexité opérationnelle.

Prenons trois modèles :
Modèle	Latence illustrative
Logistic Regression	2 ms
Random Forest	15 ms
LightGBM	5 ms

Si les modèles sont exécutés séquentiellement, la latence totale peut approcher la somme de leurs latences, plus les coûts d'agrégation.

Si les prédictions sont exécutées en parallèle, la latence peut être plus proche de celle du modèle le plus lent, avec des coûts supplémentaires.

Les valeurs ci-dessus sont purement illustratives.
En production, il faut considérer
- La latence d'inférence.
- La consommation mémoire.
- Le coût d'entraînement.
- La complexité du déploiement.
- La surveillance de chaque modèle.
- La reproductibilité.
- La gestion des versions.
- La calibration des probabilités.
Un ensemble légèrement meilleur, mais beaucoup plus coûteux, n'est pas nécessairement le meilleur choix métier.


## 17. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Combiner des modèles presque identiques	Gain limité	Rechercher la complémentarité
Entraîner le méta-modèle sur des prédictions in-sample	Data leakage	Utiliser OOF ou holdout
Optimiser les poids sur le test	Évaluation biaisée	Utiliser une validation indépendante
Ignorer la calibration	Soft Voting peu fiable	Évaluer Brier et Log Loss
Utiliser une CV aléatoire sur des séries temporelles	Fuite temporelle	Validation chronologique
Utiliser un méta-modèle trop complexe	Surapprentissage	Commencer par un modèle simple
Comparer uniquement l'accuracy	Analyse incomplète	Choisir des métriques adaptées
Ignorer la latence	Déploiement coûteux	Mesurer le coût d'inférence
Supposer que Stacking est toujours meilleur	Mauvais choix	Comparer aux baselines

[[quiz]]


## 19. Synthèse du cours

Les concepts essentiels à retenir :
- Ensemble Learning : combinaison de plusieurs modèles pour exploiter leur complémentarité.
- Hard Voting : vote majoritaire des classes.
- Soft Voting : moyenne des probabilités.
- Weighted Voting : agrégation pondérée des prédictions.
- Stacking : méta-modèle apprenant à combiner les prédictions.
- OOF : prédictions générées sur des observations non vues par le modèle de base.
- Blending : variante du Stacking utilisant un holdout.
- Diversité : des erreurs complémentaires peuvent améliorer la généralisation.
- Data leakage : risque majeur lors de la construction du méta-modèle.
- Production : arbitrage entre performance, latence et complexité.
Prochain cours
Cours 09 — Cost-Sensitive Learning et Decision Making

Partie B · 02/05

Nous allons découvrir pourquoi une bonne métrique ML ne signifie pas nécessairement une bonne décision métier, et comment utiliser les matrices de coûts, les seuils optimaux, les courbes de décision et la valeur économique attendue pour choisir les actions les plus pertinentes.


## Notes sur l’implémentation et le protocole

Le méta-modèle manuel apprend sur des probabilités OOF. Les prétraitements restent dans les pipelines de chaque fold. Les bases du Blending restent entraînées sur leur sous-train. Les candidats sont comparés sur validation ; seul le choix fixé est audité sur le test. Cette CV stratifiée concerne des observations tabulaires indépendantes, pas un Stacking temporel.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np

def soft_vote(probabilities,weights=None):
    probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.ndim!=2 or not probabilities.size or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Matrice observations x modèles de probabilités requise.')
    weights=np.ones(probabilities.shape[1]) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(probabilities.shape[1],) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Poids positifs ou nuls, somme strictement positive requise.')
    return probabilities@weights/weights.sum()

def hard_vote(predictions):
    predictions=np.asarray(predictions)
    if predictions.ndim!=2 or not predictions.size or not np.isin(predictions,[0,1]).all():raise ValueError('Prédictions binaires requises.')
    # Convention explicite : égalité en faveur de la classe 0.
    return (predictions.mean(axis=1)>.5).astype(int)

def generate_oof(estimator,X,y,splits):
    from sklearn.base import clone
    X,y=np.asarray(X),np.asarray(y)
    if len(X)!=len(y) or not np.isin(y,[0,1]).all():raise ValueError('Cible binaire alignée requise.')
    predictions=np.full(len(y),np.nan);visits=np.zeros(len(y),dtype=int)
    for train_idx,val_idx in splits:
        train_idx,val_idx=np.asarray(train_idx),np.asarray(val_idx)
        if np.intersect1d(train_idx,val_idx).size or not len(train_idx) or not len(val_idx):raise ValueError('Folds disjoints non vides requis.')
        if np.any(train_idx<0) or np.any(val_idx<0) or np.any(train_idx>=len(y)) or np.any(val_idx>=len(y)) or np.any(visits[val_idx]):
            raise ValueError('Indices valides et une seule visite par observation requis.')
        fitted=clone(estimator).fit(X[train_idx],y[train_idx])
        column=np.flatnonzero(fitted.classes_==1)
        if len(column)!=1:raise ValueError('Chaque train doit contenir la classe positive.')
        predictions[val_idx]=fitted.predict_proba(X[val_idx])[:,column[0]];visits[val_idx]+=1
    if not np.all(visits==1):raise ValueError('Toutes les observations doivent recevoir une prédiction OOF.')
    return predictions

def run(plot=False):
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split,StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,VotingClassifier,StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
    X,y=make_classification(n_samples=1800,n_features=20,n_informative=12,n_redundant=4,random_state=42)
    dev,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
    Xtr,ytr=X[train],y[train]
    estimators=[('lr',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))),
                ('rf',RandomForestClassifier(n_estimators=70,random_state=42,n_jobs=1)),
                ('hgb',HistGradientBoostingClassifier(max_iter=80,random_state=42))]
    cv=StratifiedKFold(3,shuffle=True,random_state=42);splits=list(cv.split(Xtr,ytr))
    Z=np.column_stack([generate_oof(model,Xtr,ytr,splits) for name,model in estimators])
    meta=LogisticRegression(max_iter=1000).fit(Z,ytr)
    bases=[clone(model).fit(Xtr,ytr) for name,model in estimators]
    candidates={name:model for (name,_),model in zip(estimators,bases)}
    candidates['Soft Voting']=VotingClassifier(estimators=estimators,voting='soft').fit(Xtr,ytr)
    candidates['Hard Voting']=VotingClassifier(estimators=estimators,voting='hard').fit(Xtr,ytr)
    candidates['Stacking sklearn']=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(max_iter=1000),cv=splits,stack_method='predict_proba').fit(Xtr,ytr)
    base_idx,blend_idx=train_test_split(np.arange(len(ytr)),test_size=.25,stratify=ytr,random_state=42)
    blend_bases=[clone(model).fit(Xtr[base_idx],ytr[base_idx]) for name,model in estimators]
    blend_meta=LogisticRegression(max_iter=1000).fit(np.column_stack([m.predict_proba(Xtr[blend_idx])[:,1] for m in blend_bases]),ytr[blend_idx])
    def proba(name,indices):
        if name=='Stacking manuel':return meta.predict_proba(np.column_stack([m.predict_proba(X[indices])[:,1] for m in bases]))[:,1]
        if name=='Blending':return blend_meta.predict_proba(np.column_stack([m.predict_proba(X[indices])[:,1] for m in blend_bases]))[:,1]
        return candidates[name].predict_proba(X[indices])[:,1]
    names=[name for name in candidates if name!='Hard Voting']+['Stacking manuel','Blending'];rows=[]
    for name in names:rows.append({'model':name,'AUC_validation':roc_auc_score(y[val],proba(name,val))})
    comparison=pd.DataFrame(rows).sort_values('AUC_validation',ascending=False);winner=comparison.iloc[0]['model']
    print(comparison.round(4).to_string(index=False));print('Hard Voting, accuracy validation :',accuracy_score(y[val],candidates['Hard Voting'].predict(X[val])))
    # Décision fixée avant de consulter le test. Les bases de blending restent sur leur sous-train.
    probabilities=proba(winner,test)
    print('Audit final du modèle choisi :',winner,'AUC',roc_auc_score(y[test],probabilities),'Log Loss',log_loss(y[test],probabilities))
    print('Corrélation des erreurs probabilistes sur validation :')
    print(np.corrcoef(np.column_stack([y[val]-m.predict_proba(X[val])[:,1] for m in bases]),rowvar=False))
    return {'winner':winner,'comparison':comparison}

if __name__=='__main__':run()
~~~

## Challenge Python

Compléter soft_vote et generate_oof. Cloner le modèle dans chaque fold, exclure toutes les observations de validation de son entraînement et obtenir une unique probabilité OOF par ligne. Les poids doivent être non négatifs et avoir une somme positive.

[[assessment]]

[[interview]]
