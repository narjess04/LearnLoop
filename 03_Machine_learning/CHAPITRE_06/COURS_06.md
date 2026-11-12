Partie A · Séries temporelles · Cours 06/12

Les exemples Python aplatis ou tronqués du support sont regroupés dans un lab autonome. Les originaux sont conservés dans supports/. Les fonctions essentielles de l’exercice sont également présentées ci-dessous.

[[probabilistic]]

Cours 06 — Prévision probabiliste : quantiles, incertitude et Conformal Prediction
Partie A · 06/07
Niveau : Intermédiaire → Avancé
Durée estimée : 3–4 h

Dans les cinq cours précédents, nous avons appris à analyser des séries temporelles, construire des baselines, utiliser ARIMA et LightGBM, puis gérer plusieurs séries simultanément.

Jusqu'à présent, nos modèles produisaient principalement des prévisions ponctuelles.

Mais imaginons qu'un modèle annonce 1 000 ventes pour le mois prochain. Cette information ne nous indique pas si les ventes réelles seront probablement proches de 1 000 ou si elles pourraient varier fortement.

C'est précisément le rôle du forecasting probabiliste : quantifier l'incertitude associée aux prévisions pour prendre de meilleures décisions.

## 1. Objectifs pédagogiques

À la fin de ce cours, tu seras capable de :
- Distinguer prévision ponctuelle, intervalle de prédiction et distribution prédictive.
- Comprendre les quantiles conditionnels.
- Entraîner des modèles de régression quantile avec LightGBM.
- Évaluer les quantiles avec la Pinball Loss.
- Comprendre le CRPS et les probabilistic scoring rules.
- Mesurer la couverture et la largeur des intervalles.
- Construire des intervalles avec Conformal Prediction.
- Identifier les limites des garanties conformales sur les séries temporelles.
- Intégrer l'incertitude dans des décisions métier.


## 2. Prévision ponctuelle vs prévision probabiliste


## 2.1 Prévision ponctuelle

Une prévision ponctuelle produit une seule valeur :
\[
\boxed{\hat Y_{t+h}=f(X_t)}
\]

Exemple :
\[
\hat Y_{t+1}=100
\]

Le modèle annonce 100 ventes pour demain.

Mais cette valeur ne renseigne pas directement sur l'incertitude.

## 2.2 Prévision par intervalle

Une prévision par intervalle produit une borne inférieure et une borne supérieure.

Par exemple :
\[
\boxed{Y_{t+1}\in[85,120]}
\]

Cet intervalle peut être construit pour viser une couverture de 90 %.

Cela signifie que la procédure cherche à contenir environ 90 % des observations futures sur des évaluations répétées comparables. Ce n'est pas une garantie absolue pour chaque prévision individuelle.

## 2.3 Prévision par distribution

Une prévision probabiliste complète cherche à estimer :
\[
\boxed{P(Y_{t+h}\mid\mathcal F_t)}
\]

où \(\mathcal F_t\) représente les informations disponibles à l'instant \(t\).

Elle permet de répondre à des questions telles que :
- Quelle est la probabilité que les ventes dépassent 120 unités ?
- Quel niveau de stock permet de couvrir 95 % de la demande prévue ?
- Quelle est l'incertitude à sept jours par rapport à un jour ?
Exemple : une prévision et son intervalle d'incertitude

Observations

Prévision centrale

Intervalle prédictif

Illustration schématique : l'incertitude peut augmenter avec l'horizon, mais ce comportement n'est pas universel.

## 3. Les sources d'incertitude

On distingue notamment deux grandes catégories.

## 3.1 Incertitude aléatoire — Aleatoric Uncertainty

Elle provient de la variabilité inhérente au phénomène.

Exemples : variations imprévisibles de la demande, événements météorologiques, comportements individuels.

Même avec beaucoup de données, cette incertitude ne disparaît pas nécessairement.

## 3.2 Incertitude épistémique — Epistemic Uncertainty

Elle provient d'un manque de connaissance du modèle.

Exemples : peu de données historiques, paramètres mal estimés, absence de certaines variables explicatives.

Elle peut parfois être réduite en collectant davantage de données ou en améliorant la modélisation.
Type	Origine	Réduction possible
Aléatoire	Variabilité du phénomène	Généralement limitée
Épistémique	Connaissance insuffisante	Souvent possible
Changement de régime	Évolution du processus	Nécessite adaptation et surveillance

En pratique, les intervalles prédictifs peuvent combiner plusieurs sources d'incertitude sans les séparer explicitement.


## 4. Comprendre les quantiles

Un quantile représente un seuil associé à une proportion de la distribution.

Le quantile d'ordre \(\tau\) est défini par :
\[
\boxed{
Q_\tau(Y)=\inf\{y:F_Y(y)\geq\tau\}
}
\]

où \(F_Y\) est la fonction de répartition.

Quelques quantiles importants :
Quantile	Signification
Q0.05	5e percentile
Q0.10	10e percentile
Q0.50	Médiane
Q0.90	90e percentile
Q0.95	95e percentile

Un intervalle central à 90 % peut être construit avec :
\[
\boxed{[Q_{0.05},Q_{0.95}]}
\]

L’interaction au début de ce support permet d’explorer cet exemple.

Quantiles conditionnels
En Machine Learning, nous cherchons généralement à estimer :
\[
\boxed{Q_\tau(Y\mid X=x)}
\]

Cela signifie que le quantile dépend des caractéristiques de l'observation.

Par exemple, la distribution des ventes peut être différente un lundi et un samedi.


## 5. Régression quantile

La régression classique avec une perte quadratique cherche généralement à estimer une moyenne conditionnelle.

La régression quantile cherche à estimer un quantile conditionnel.

Pour différents niveaux \(\tau\), nous entraînons des modèles produisant :
\[
\hat Q_{0.05}(X),\quad
\hat Q_{0.50}(X),\quad
\hat Q_{0.95}(X)
\]

Nous obtenons ainsi une prévision centrale et des bornes d'incertitude.

## 5.1 Pinball Loss

La fonction de perte utilisée pour la régression quantile est :
\[
\boxed{
L_\tau(y,\hat q)=
\begin{cases}
\tau(y-\hat q), & y\geq\hat q\\
(1-\tau)(\hat q-y), & y<\hat q
\end{cases}
}
\]

On peut également l'écrire :
\[
L_\tau(y,\hat q)
=
(\tau-\mathbf 1_{y<\hat q})(y-\hat q)
\]

Intuition
Pour \(\tau=0.9\), sous-estimer la cible est davantage pénalisé que la surestimer d'une même quantité.

Supposons :
\[
y=120,\quad\hat q=100
\]

Alors :
\[
L_{0.9}=0.9(120-100)=18
\]

À l'inverse, si :
\[
y=80,\quad\hat q=100
\]

Alors :
\[
L_{0.9}=0.1(100-80)=2
\]

C'est ce comportement asymétrique qui permet au modèle d'apprendre un quantile supérieur.

## 5.2 Implémentation Python


~~~python
from ch06_cours06_lab import finite_vector, pinball_loss, conformal_quantile, conformal_interval, interval_metrics, crps_ensemble, cqr_correction

def pinball_loss(y_true,y_pred,tau):
    y_true,y_pred=finite_vector(y_true),finite_vector(y_pred)
    if y_true.shape!=y_pred.shape or not 0<tau<1:raise ValueError('Formes alignées et tau dans ]0,1[ requis.')
    error=y_true-y_pred
    return float(np.maximum(tau*error,(tau-1)*error).mean())
~~~


## 6. Forecasting quantile avec LightGBM

LightGBM permet d'entraîner des modèles de régression quantile en utilisant objective="quantile".

Nous allons entraîner trois modèles :
- Q0.05 : borne inférieure.
- Q0.50 : médiane.
- Q0.95 : borne supérieure.

## 6.1 Entraîner les modèles


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Ici, X_train et y_train correspondent à un dataset supervisé temporel correctement construit, comme dans le cours 04.

## 6.2 Produire les prévisions


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Les trois modèles étant entraînés séparément, leurs prédictions peuvent parfois se croiser.

Par exemple :
\[
\hat Q_{0.05}(x)>\hat Q_{0.95}(x)
\]

Ce phénomène est appelé quantile crossing.

On peut appliquer un réarrangement monotone ou utiliser une méthode qui impose l'ordre des quantiles.


## 7. Évaluer les prévisions probabilistes

Contrairement à une prévision ponctuelle, une prévision probabiliste doit être évaluée selon plusieurs dimensions.

Deux notions essentielles :
- Calibration : les probabilités annoncées correspondent-elles aux fréquences observées ?
- Sharpness : les distributions ou intervalles sont-ils suffisamment concentrés ?
Un bon modèle doit être bien calibré tout en produisant des prévisions aussi précises que possible.

## 7.1 Couverture empirique — Coverage

Pour un intervalle \([L_t,U_t]\), la couverture empirique est :
\[
\boxed{
Coverage=
\frac{1}{n}\sum_{t=1}^{n}
\mathbf 1(L_t\leq Y_t\leq U_t)
}
\]

Si nous construisons un intervalle nominal à 90 %, nous souhaitons une couverture empirique proche de 90 % sur un ensemble de test suffisamment représentatif.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## 7.2 Largeur moyenne — Average Interval Width

\[
\boxed{
AIW=\frac{1}{n}\sum_{t=1}^{n}(U_t-L_t)
}
\]

Un intervalle très large peut obtenir une bonne couverture sans être utile.

Par exemple :
- Intervalle A : [0, 10 000].
- Intervalle B : [90, 115].
Le premier couvre potentiellement presque toutes les observations, mais fournit peu d'information.

La qualité d'un intervalle ne peut donc pas être évaluée uniquement par sa couverture.

## 7.3 Interval Score

Pour un intervalle nominal de couverture \(1-\alpha\) :
\[
\boxed{
IS_\alpha(L,U;y)
=
(U-L)
+\frac{2}{\alpha}(L-y)\mathbf1(y<L)
+\frac{2}{\alpha}(y-U)\mathbf1(y>U)
}
\]

Ce score pénalise simultanément :
- Les intervalles trop larges.
- Les observations situées en dehors de l'intervalle.
Une valeur plus faible est préférable.


## 8. CRPS — Continuous Ranked Probability Score

Le CRPS permet d'évaluer une distribution prédictive complète.

Il compare la fonction de répartition prédictive \(F\) à l'observation réelle \(y\).
\[
\boxed{
CRPS(F,y)=
\int_{-\infty}^{+\infty}
\left(
F(z)-\mathbf1(y\leq z)
\right)^2 dz
}
\]

Intuition
Le CRPS mesure l'écart entre la distribution prédite et l'observation.

Il récompense les distributions qui attribuent une forte probabilité aux valeurs proches de la réalité.

Le CRPS est une proper scoring rule : en espérance, annoncer la vraie distribution minimise le score.
Lien avec la Pinball Loss
Le CRPS peut également s'exprimer comme une intégrale des pertes quantiles :
\[
\boxed{
CRPS(F,y)=
2\int_0^1
L_\tau(y,Q_\tau(F))\,d\tau
}
\]

Cette relation explique pourquoi l'évaluation de plusieurs quantiles permet d'approcher la qualité d'une distribution prédictive.
Implémentation Python avec des échantillons

~~~python
from ch06_cours06_lab import finite_vector, pinball_loss, conformal_quantile, conformal_interval, interval_metrics, crps_ensemble, cqr_correction

def crps_ensemble(y,samples):
    samples=np.sort(finite_vector(samples));m=len(samples)
    pair_half=np.dot(2*np.arange(1,m+1)-m-1,samples)/(m*m)
    return float(np.abs(samples-y).mean()-pair_half)
~~~

Cette formule évalue la distribution empirique représentée par les échantillons. Pour des distributions paramétriques, on peut utiliser des formules analytiques ou des bibliothèques spécialisées.

## Section B du cours — Conformal Prediction

## 9. Pourquoi utiliser Conformal Prediction ?

Les modèles de régression quantile produisent des intervalles, mais leur couverture réelle peut être différente de la couverture nominale.

Par exemple, un intervalle annoncé à 90 % peut ne couvrir que 75 % des observations.

La Conformal Prediction propose des méthodes permettant de calibrer les intervalles à partir des erreurs observées sur un ensemble de calibration.

Son intérêt principal est qu'elle peut être appliquée à de nombreux modèles prédictifs.

Cependant, les garanties classiques de couverture marginale reposent notamment sur une hypothèse d'échangeabilité des observations et des scores.

Cette hypothèse n'est généralement pas satisfaite automatiquement dans les séries temporelles.

## 9.1 Split Conformal Prediction

La procédure classique comporte trois étapes :

## 1. Entraîner un modèle sur un ensemble d'entraînement.


## 2. Calculer des scores de non-conformité sur un ensemble de calibration distinct.


## 3. Utiliser un quantile de ces scores pour construire les intervalles.

Pour une régression ponctuelle, un score fréquent est :
\[
\boxed{
s_i=|y_i-\hat y_i|
}
\]

Nous calculons ensuite un quantile calibré de ces scores.

Pour une nouvelle observation :
\[
\boxed{
[\hat y(x)-q,\hat y(x)+q]
}
\]

où \(q\) est le seuil de calibration.

## 9.2 Exemple numérique

Supposons les erreurs absolues suivantes sur l'ensemble de calibration :
\[
[2,3,4,5,6,7,8,9,10,12]
\]

Pour une couverture cible de 90 %, le quantile conforme fini-échantillon est déterminé à partir du rang :
\[
k=\lceil(n+1)(1-\alpha)\rceil
\]

Avec \(n=10\) et \(\alpha=0.1\) :
\[
k=\lceil11\times0.9\rceil=10
\]

Le dixième score ordonné est :
\[
q=12
\]

Si la prévision ponctuelle est 100 :
\[
\boxed{[88,112]}
\]

Sous les hypothèses conformales classiques, cette procédure possède une garantie marginale de couverture.

## 9.3 Implémentation Python


~~~python
from ch06_cours06_lab import finite_vector, pinball_loss, conformal_quantile, conformal_interval, interval_metrics, crps_ensemble, cqr_correction

def conformal_interval(y_cal,pred_cal,pred_test,alpha=.1):
    y_cal,pred_cal=finite_vector(y_cal),finite_vector(pred_cal)
    if y_cal.shape!=pred_cal.shape:raise ValueError('Calibration alignée requise.')
    q=conformal_quantile(np.abs(y_cal-pred_cal),alpha)
    pred_test=finite_vector(pred_test)
    return pred_test-q,pred_test+q
~~~

Le cas k > n produit un intervalle non borné, ce qui reflète le manque de données de calibration pour obtenir le niveau demandé avec la garantie classique.


## 10. Conformalized Quantile Regression — CQR

La méthode CQR combine régression quantile et calibration conforme.

Nous commençons avec deux modèles :
\[
\hat Q_{\alpha/2}(X)
\]

\[
\hat Q_{1-\alpha/2}(X)
\]

Puis nous calculons les scores :
\[
\boxed{
s_i=
\max\left(
\hat Q_{\alpha/2}(X_i)-Y_i,
Y_i-\hat Q_{1-\alpha/2}(X_i)
\right)
}
\]

Le score peut être négatif lorsque l'observation est située suffisamment à l'intérieur de l'intervalle initial.

Après calibration, l'intervalle devient :
\[
\boxed{
[\hat Q_{\alpha/2}(X)-q,\;
\hat Q_{1-\alpha/2}(X)+q]
}
\]

Pourquoi CQR est intéressant ?
Contrairement à un intervalle symétrique de largeur constante, CQR peut adapter sa largeur aux caractéristiques de l'observation.

Exemple :
- Période stable : intervalle étroit.
- Période très variable : intervalle large.
Intervalles adaptatifs selon la volatilité
Illustration conceptuelle : les intervalles peuvent être plus larges dans les régions où l'incertitude est élevée.
Implémentation CQR

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Les modèles quantiles doivent être entraînés uniquement sur l'ensemble d'entraînement. L'ensemble de calibration doit rester séparé de celui utilisé pour ajuster ces modèles.


## 11. Le problème particulier des séries temporelles

Les méthodes conformales classiques supposent souvent l'échangeabilité.

Or, dans une série temporelle :
\[
Y_t\not\perp Y_{t-1}
\]

Les observations peuvent dépendre les unes des autres.

Il peut également exister :
- Des tendances.
- Des saisonnalités.
- Des changements de régime.
- Du drift.
- Une variance évolutive.
Par conséquent, il ne faut pas affirmer qu'une méthode split conformal classique garantit automatiquement 90 % de couverture sur une série temporelle arbitraire.

## 11.1 Rolling Conformal

Une approche consiste à recalibrer régulièrement les intervalles avec les erreurs les plus récentes.

Par exemple, nous pouvons conserver les 100 derniers scores de non-conformité disponibles.

À chaque nouvelle période :

## 1. Produire la prévision.


## 2. Construire l'intervalle à partir des scores disponibles.


## 3. Attendre l'observation réelle.


## 4. Calculer le nouveau score.


## 5. Mettre à jour la fenêtre de calibration.

Cette approche peut mieux s'adapter à certains changements temporels, mais elle n'offre pas à elle seule une garantie universelle de couverture.

## 11.2 Calibration par horizon

Pour un forecasting multi-horizon, les erreurs à un jour et à trente jours peuvent être très différentes.

Il peut être pertinent de calculer :
\[
q_1,q_2,\ldots,q_H
\]

Un seuil de calibration distinct pour chaque horizon permet d'adapter les intervalles aux erreurs observées à cet horizon.

## 11.3 Évaluer la couverture dans le temps


Couverture observéeCouverture cible

Données fictives illustrant un changement de calibration

Une baisse persistante de la couverture peut signaler que les intervalles ne représentent plus correctement l'incertitude.


## 12. Mini-projet complet : forecasting probabiliste avec LightGBM

Nous allons créer un dataset synthétique, entraîner trois modèles quantiles et évaluer les intervalles sur une période future.

## Étape 1 — Générer les données


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 2 — Construire les features


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 3 — Train / Calibration / Test

Nous allons réserver une période spécifique à la calibration.

Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 4 — Entraîner les modèles quantiles


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 5 — Calibration CQR


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 6 — Construire les intervalles sur test


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 7 — Évaluer la couverture et la largeur


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.


## Étape 8 — Visualiser les prévisions


Le code de cette étape est regroupé dans le Coding Lab complet en fin de cours.

Comment interpréter ce mini-projet ?
Ce pipeline évalue des prédictions à un pas, en supposant que les observations passées deviennent disponibles avant chaque nouvelle prévision.

Les scores de calibration sont donc chronologiquement ordonnés et potentiellement dépendants. La couverture mesurée sur le test est empirique : elle ne bénéficie pas automatiquement de la garantie théorique classique du split conformal.

Pour un véritable projet industriel, il faudrait compléter cette expérience par un backtest sur plusieurs origines, une calibration adaptée à l'horizon et une surveillance de la couverture dans le temps.


## 13. Application métier : gestion des stocks

Supposons qu'un magasin souhaite prévoir la demande de la semaine prochaine.

Un modèle ponctuel annonce :
\[
\hat Y=100
\]

Mais le modèle probabiliste produit :
Quantile	Demande prévue
Q0.10	80
Q0.50	100
Q0.90	130
Q0.95	145

Le choix du niveau de stock dépend du coût de rupture et du coût de surstockage.
Newsvendor Problem
Dans le problème classique du marchand de journaux, le quantile optimal de commande est :
\[
\boxed{
\tau^*=\frac{C_u}{C_u+C_o}
}
\]

où :
- \(C_u\) : coût marginal d'une unité manquante.
- \(C_o\) : coût marginal d'une unité excédentaire.
Exemple :
\[
C_u=9,\quad C_o=1
\]

Alors :
\[
\tau^*=\frac{9}{10}=0.9
\]

Le quantile Q0.90 peut donc être pertinent pour déterminer une quantité de commande, sous les hypothèses du modèle Newsvendor.

Cela montre une idée essentielle :

La meilleure prévision n'est pas toujours celle qui minimise simplement la MAE. C'est parfois celle qui permet de prendre la meilleure décision.


## 14. Bonnes pratiques et erreurs fréquentes

Erreur	Conséquence	Bonne pratique
Fournir seulement une moyenne	Incertitude ignorée	Ajouter quantiles ou intervalles
Évaluer uniquement la couverture	Intervalles excessivement larges	Mesurer aussi la largeur
Utiliser la MAE pour juger un quantile extrême	Mauvaise métrique	Utiliser Pinball Loss
Supposer les quantiles toujours ordonnés	Quantile crossing	Vérifier et corriger
Calibrer sur le test	Évaluation biaisée	Séparer calibration et test
Supposer l'échangeabilité temporelle	Garantie injustifiée	Adapter et vérifier les hypothèses
Utiliser un seul seuil pour tous les horizons	Mauvaise calibration possible	Tester une calibration par horizon
Ignorer le drift	Couverture dégradée	Surveiller la calibration
Confondre intervalle de confiance et intervalle prédictif	Mauvaise interprétation	Identifier la quantité estimée

[[quiz]]


## 16. Synthèse du cours

Les notions essentielles :
- Prévision ponctuelle : une valeur future estimée.
- Prévision probabiliste : une distribution, des quantiles ou des intervalles.
- Régression quantile : estimation de quantiles conditionnels.
- Pinball Loss : fonction de perte adaptée aux quantiles.
- CRPS : score permettant d'évaluer une distribution prédictive.
- Coverage : proportion des observations couvertes par les intervalles.
- Sharpness : concentration des distributions ou largeur des intervalles.
- Conformal Prediction : calibration des intervalles à partir de scores de non-conformité.
- CQR : combinaison de régression quantile et de calibration conforme.
- Forecasting temporel : les dépendances et changements de régime exigent une attention particulière aux garanties de couverture.
Prochain cours
Cours 07 — Détection d'anomalies et de ruptures temporelles

Partie A · 07/07

Nous terminerons la partie Séries temporelles en étudiant les anomalies ponctuelles et contextuelles, les changements de régime (change point detection), la non-stationnarité, le drift et les méthodes de surveillance temporelle.


## Notes sur l’implémentation et le protocole

Les quantiles croisés sont réarrangés de la même manière sur calibration et test. Le lab utilise une correction CQR bornée inférieurement à zéro : cette variante conservatrice ne rétrécit pas les intervalles lorsque le score conforme est négatif. La calibration glissante n’ajoute une erreur qu’après observation de la cible. La garantie classique sous échangeabilité ne se transpose pas automatiquement à cette série temporelle.

## Coding Lab complet

Exécuter le fichier téléchargé dans un environnement Python local avec les dépendances du README. Les commandes d’installation se lancent dans un terminal. L’exercice intégré utilise les fonctions NumPy/pandas du corrigé et peut être exécuté dans le navigateur.

~~~python
import numpy as np

def finite_vector(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Vecteur fini non vide requis.')
    return values

def pinball_loss(y_true,y_pred,tau):
    y_true,y_pred=finite_vector(y_true),finite_vector(y_pred)
    if y_true.shape!=y_pred.shape or not 0<tau<1:raise ValueError('Formes alignées et tau dans ]0,1[ requis.')
    error=y_true-y_pred
    return float(np.maximum(tau*error,(tau-1)*error).mean())

def conformal_quantile(scores,alpha=.1):
    scores=finite_vector(scores)
    if not 0<alpha<1:raise ValueError('alpha dans ]0,1[ requis.')
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=len(scores) else np.inf

def conformal_interval(y_cal,pred_cal,pred_test,alpha=.1):
    y_cal,pred_cal=finite_vector(y_cal),finite_vector(pred_cal)
    if y_cal.shape!=pred_cal.shape:raise ValueError('Calibration alignée requise.')
    q=conformal_quantile(np.abs(y_cal-pred_cal),alpha)
    pred_test=finite_vector(pred_test)
    return pred_test-q,pred_test+q

def interval_metrics(y,lower,upper,alpha=.1):
    y=finite_vector(y);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if y.shape!=lower.shape or y.shape!=upper.shape or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>upper) or not 0<alpha<1:
        raise ValueError('Intervalles ordonnés alignés et alpha dans ]0,1[ requis.')
    width=upper-lower
    # Masquer les pénalités pour éviter inf*0 avec un intervalle non borné.
    penalty=np.where(y<lower,lower-y,0)+np.where(y>upper,y-upper,0)
    return {'coverage':float(((lower<=y)&(y<=upper)).mean()),'width':float(width.mean()),
            'interval_score':float(np.mean(width+2/alpha*penalty))}

def crps_ensemble(y,samples):
    samples=np.sort(finite_vector(samples));m=len(samples)
    pair_half=np.dot(2*np.arange(1,m+1)-m-1,samples)/(m*m)
    return float(np.abs(samples-y).mean()-pair_half)

def cqr_correction(y,lower,upper,alpha=.1):
    y,lower,upper=map(finite_vector,(y,lower,upper))
    if y.shape!=lower.shape or y.shape!=upper.shape or np.any(lower>upper):raise ValueError('Quantiles ordonnés alignés requis.')
    # Variante conservatrice : ne rétrécit pas l'intervalle si le quantile est négatif.
    return max(0.,conformal_quantile(np.maximum(lower-y,y-upper),alpha))

def run(plot=False):
    import pandas as pd
    from lightgbm import LGBMRegressor
    rng=np.random.default_rng(42);t=np.arange(1200)
    sales=100+.02*t+15*np.sin(2*np.pi*t/7)+rng.normal(0,4+.008*t)
    df=pd.DataFrame({'sales':sales},index=pd.date_range('2022-01-01',periods=len(t),freq='D'))
    for lag in [1,7,14,28]:df[f'lag_{lag}']=df['sales'].shift(lag)
    df['rolling_mean_7']=df['sales'].shift(1).rolling(7).mean()
    df['rolling_std_7']=df['sales'].shift(1).rolling(7).std()
    df['day_of_week']=df.index.dayofweek;df['month']=df.index.month;data=df.dropna()
    a,b=int(.65*len(data)),int(.8*len(data));train,cal,test=data.iloc[:a],data.iloc[a:b],data.iloc[b:]
    features=[c for c in data if c!='sales'];models={}
    for tau in [.05,.5,.95]:
        model=LGBMRegressor(objective='quantile',alpha=tau,n_estimators=180,learning_rate=.05,num_leaves=15,random_state=42,verbosity=-1,n_jobs=1)
        model.fit(train[features],train['sales']);models[tau]=model
    # Même règle de réarrangement sur calibration et test avant toute évaluation.
    cal_q=np.sort(np.column_stack([models[q].predict(cal[features]) for q in [.05,.5,.95]]),axis=1)
    test_q=np.sort(np.column_stack([models[q].predict(test[features]) for q in [.05,.5,.95]]),axis=1)
    qhat=cqr_correction(cal['sales'],cal_q[:,0],cal_q[:,2],.1)
    lower,median,upper=test_q[:,0]-qhat,test_q[:,1],test_q[:,2]+qhat
    print('Correction CQR conservatrice :',qhat)
    print('Quantiles bruts :',interval_metrics(test['sales'],test_q[:,0],test_q[:,2]))
    print('CQR :',interval_metrics(test['sales'],lower,upper))
    print('Pinball :',{q:pinball_loss(test['sales'],test_q[:,j],q) for j,q in enumerate([.05,.5,.95])})
    # Mise à jour après observation, jamais avant : protocole à un pas uniquement.
    scores=list(np.abs(cal['sales'].to_numpy()-cal_q[:,1]));rolling_lower=[];rolling_upper=[]
    for actual,pred in zip(test['sales'],median):
        radius=conformal_quantile(scores[-100:],.1)
        rolling_lower.append(pred-radius);rolling_upper.append(pred+radius)
        scores.append(abs(actual-pred))
    print('Rolling conformal empirique :',interval_metrics(test['sales'],rolling_lower,rolling_upper))
    print('Dépendance temporelle : ces résultats empiriques ne garantissent pas une couverture universelle.')
    print('CRPS illustratif :',crps_ensemble(108,[95,100,105,110,115]))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5));plt.plot(test.index,test['sales'],label='Réel')
        plt.plot(test.index,median,label='Médiane');plt.fill_between(test.index,lower,upper,alpha=.2,label='CQR nominal 90 %')
        plt.legend();plt.tight_layout();plt.show()
    return {'qhat':qhat,'metrics':interval_metrics(test['sales'],lower,upper)}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
~~~

## Challenge Python

Compléter pinball_loss, conformal_quantile et crps_ensemble. Respecter le rang fini-échantillon ceil((n+1)(1-alpha)) et renvoyer l’infini si ce rang dépasse n. Comparer couverture, largeur et Interval Score sans ajuster la calibration sur le test.

[[assessment]]

[[interview]]
