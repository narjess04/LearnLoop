# Chapitre 5 — Cours 9 : Model Validation, Backtesting & Statistical Testing



## 1. Introduction — Pourquoi valider un modèle après son entraînement ?

Dans les chapitres précédents, nous avons appris à entraîner des modèles, à mesurer leurs performances, à expliquer leurs décisions et à surveiller leur comportement.

Mais une question fondamentale demeure :

Comment démontrer qu'un modèle est suffisamment fiable pour être utilisé dans un environnement réel ?

Imaginons que deux modèles XGBoost et Random Forest soient évalués sur le même jeu de test.
Comparaison de deux modèles — exemple fictif
Modèle A — XGBoost
0,89
AUC-ROC

Modèle B — Random Forest
0,86
AUC-ROC

Différence observée : +0,03 point d'AUC en faveur du modèle A.

Le modèle A semble meilleur.

Mais cette différence de 0,03 est-elle suffisamment importante pour conclure qu'il est réellement plus performant ?

Pour répondre, il faut considérer plusieurs facteurs :
- La taille et la représentativité du jeu de test.
- L'incertitude statistique des métriques.
- La variabilité des performances selon les échantillons.
- Les éventuelles fuites de données.
- La stabilité temporelle.
- Les exigences métier et les coûts des erreurs.
C'est l'objectif de la validation des modèles.
Définition
La validation d'un modèle ML est un processus structuré visant à vérifier que le modèle répond à ses objectifs, qu'il est statistiquement fiable et qu'il peut être utilisé dans les conditions prévues.

Elle va au-delà du simple calcul d'une accuracy ou d'une AUC.

## 2. Model Evaluation vs Model Validation vs Backtesting

Ces notions sont proches, mais elles ont des objectifs différents.
Concept	Objectif
Model Evaluation	Mesurer les performances du modèle
Model Validation	Vérifier sa fiabilité, ses limites et son adéquation à l'usage prévu
Backtesting	Comparer les prédictions historiques à des résultats réellement observés

Exemple
Une banque développe un modèle de probabilité de défaut.

Évaluation : le modèle obtient une AUC de 0,85 et un Brier Score de 0,04.

Validation : on vérifie la qualité des données, la discrimination, la calibration, la stabilité, les hypothèses et les limites.

Backtesting : on compare les probabilités de défaut prévues avec les défauts réellement constatés sur une période d'observation définie.

Un modèle peut présenter une bonne discrimination tout en étant mal calibré.

C'est pourquoi une validation professionnelle doit couvrir plusieurs dimensions.

## 3. Les grandes dimensions de la validation

## 1. Data Validation

Vérifier la qualité, la représentativité, la cohérence et l'intégrité des données utilisées.

## 2. Performance Validation

Évaluer la discrimination, les erreurs, la calibration et les performances par segment.

## 3. Robustness & Stability Validation

Tester la sensibilité aux perturbations, aux variations de population et aux changements temporels.

## 4. Model Governance

Documenter les hypothèses, les limites, les résultats, les décisions et les responsabilités.

Ces dimensions se complètent. La validation n'est pas un test unique, mais un ensemble cohérent de contrôles.

## 4. L'incertitude statistique des métriques

Lorsque nous calculons une métrique sur un échantillon, nous obtenons une estimation.

Cette estimation peut varier si nous évaluons le même modèle sur un autre échantillon représentatif.

Par exemple :
\[
\widehat{AUC}=0.87
\]

Cela ne signifie pas que l'AUC sur toute la population future sera exactement 0,87.

Nous devons donc quantifier l'incertitude de cette estimation.

## 4.1. Intervalle de confiance

Un intervalle de confiance à 95 % pourrait être :
\[
IC_{95\%}(AUC)=[0.83,0.91]
\]

Cela signifie que la procédure utilisée pour construire l'intervalle possède, sous ses hypothèses, une couverture de 95 % de la vraie quantité estimée lors de répétitions de l'échantillonnage.

Il ne faut pas interpréter directement cet intervalle fréquentiste comme une probabilité de 95 % que le paramètre fixe appartienne à l'intervalle observé.

## 4.2. Pourquoi les intervalles de confiance sont importants ?

Imaginons deux modèles :
AUC et intervalles de confiance fictifs
Modèle A : 0,89 [0,84 ; 0,92]. Modèle B : 0,86 [0,81 ; 0,91].
Les intervalles se chevauchent.

Cela ne suffit pas à conclure que la différence n'est pas statistiquement significative. Pour comparer correctement les modèles, nous devons étudier directement l'incertitude de leur différence, idéalement en tenant compte du fait qu'ils ont été évalués sur les mêmes observations.

## 5. Bootstrap : mesurer l'incertitude sans formule analytique complexe

Le Bootstrap est une méthode statistique de rééchantillonnage.

Il permet d'estimer la variabilité d'une statistique à partir des données disponibles.

## 5.1. Principe

Supposons que nous disposons d'un jeu de test contenant 1 000 observations.

Nous procédons ainsi :

## 1. Nous tirons aléatoirement 1 000 observations avec remise à partir du jeu de test.

## 2. Nous calculons l'AUC sur ce nouvel échantillon.

## 3. Nous répétons cette opération, par exemple 2 000 fois.

## 4. Nous obtenons une distribution empirique des AUC.

## 5. Nous utilisons les quantiles de cette distribution pour construire un intervalle de confiance.

## 5.2. Visualisation du Bootstrap

[[pairedbootstrap]]

Distribution fictive des AUC bootstrap

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Histogramme illustratif, non issu d'une exécution réelle.

## 5.3. Implémentation Python

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Cette implémentation suppose des observations indépendantes et identiquement distribuées, ou suffisamment proches de ce cadre.

Pour des données temporelles, des observations regroupées par client ou des données dépendantes, il faut adapter la stratégie de rééchantillonnage.

Le Bootstrap percentile est simple, mais peut présenter des limites sur de petits échantillons ou des distributions fortement asymétriques.

## 6. Comparaison statistique de deux modèles

Supposons que nous avons deux modèles :
\[
AUC_A=0.89
\]

\[
AUC_B=0.86
\]

La différence observée est :
\[
\Delta AUC=0.03
\]

Mais cette différence peut être due à la variabilité de l'échantillon.

## 6.1. Paired Bootstrap

Lorsque les deux modèles sont évalués sur les mêmes observations, nous pouvons utiliser un Bootstrap apparié.

À chaque itération, nous rééchantillonnons les mêmes indices pour les deux modèles.

Nous calculons :
\[
\Delta AUC_b=AUC_{A,b}-AUC_{B,b}
\]

Puis nous analysons la distribution des différences.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Si l'intervalle de confiance de la différence exclut zéro, cela constitue un indice statistique en faveur d'une différence entre les modèles, sous les hypothèses de la procédure.

## 6.2. Test de DeLong

Le test de DeLong est une méthode statistique classique pour comparer des AUC corrélées, notamment lorsque les modèles sont évalués sur les mêmes observations.

Il tient compte de la structure de dépendance entre les AUC.

Il est particulièrement pertinent pour comparer le pouvoir discriminant de deux modèles de classification binaire.

## 6.3. Test de McNemar

Le test de McNemar permet de comparer deux classifieurs binaires évalués sur les mêmes observations.

Il s'intéresse aux cas où les modèles ne sont pas d'accord sur la correction de leurs prédictions.

On construit un tableau :
Modèle B correct	Modèle B incorrect
Modèle A correct	\(n_{11}\)	\(n_{10}\)
Modèle A incorrect	\(n_{01}\)	\(n_{00}\)

La statistique asymptotique classique est :
\[
\chi^2=\frac{(n_{10}-n_{01})^2}{n_{10}+n_{01}}
\]

Sous l'hypothèse nulle, les deux types de désaccord sont supposés aussi fréquents.

Pour de petits effectifs de désaccord, un test exact est préférable.
Exemple Python

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Le test de McNemar compare les erreurs de classification au seuil choisi. Il ne compare pas directement les probabilités ni les AUC.

## 7. Backtesting : vérifier les prédictions sur des résultats historiques

Le Backtesting est particulièrement important pour les modèles utilisés dans le temps.

L'objectif est de confronter les prédictions produites à une date donnée avec les résultats réellement observés ensuite.
Exemple : Probability of Default
Un modèle prédit une probabilité de défaut pour chaque client.

Après la période d'observation, nous calculons le taux de défaut observé :
\[
ODR=\frac{\text{Nombre de défauts observés}}{\text{Nombre de clients observés}}
\]

Nous comparons ensuite :
\[
PD_{\text{prédite}}\quad \text{vs}\quad ODR
\]

Exemple fictif
Groupe de risque	PD moyenne	ODR
1	0,5 %	0,6 %
2	1,0 %	1,2 %
3	2,5 %	2,8 %
4	5,0 %	6,5 %
5	10,0 %	14,0 %

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Données fictives : PD prédite et taux de défaut observé par groupe.

Nous observons une sous-estimation du risque dans les groupes les plus risqués.

Mais pour conclure, il faut considérer la taille des groupes, le nombre de défauts et l'incertitude statistique.

## 8. Tests statistiques de calibration

Plusieurs tests peuvent être utilisés pour vérifier la cohérence entre probabilités prédites et résultats observés.

## 8.1. Test binomial

Supposons qu'un groupe contient \(n\) clients, avec une probabilité de défaut commune \(p\), sous une hypothèse simplificatrice.

Le nombre de défauts suit :
\[
D\sim Binomial(n,p)
\]

Nous pouvons tester :
\[
H_0:p_{\text{réel}}\leq p_{\text{prédit}}
\]

contre :
\[
H_1:p_{\text{réel}}>p_{\text{prédit}}
\]

Ce test unilatéral cherche à détecter une sous-estimation du risque.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Cette formulation suppose une probabilité commune et des observations indépendantes.

Lorsque les PD individuelles diffèrent, une loi de Poisson-binômiale ou une méthode adaptée peut être plus appropriée.

## 8.2. Test de Hosmer-Lemeshow

Le test de Hosmer-Lemeshow compare les nombres d'événements observés et attendus dans plusieurs groupes de risque.

Une statistique courante est :
\[
HL=\sum_{g=1}^{G}
\frac{(O_g-E_g)^2}
{E_g(1-E_g/n_g)}
\]

Où :
- \(O_g\) : événements observés dans le groupe \(g\).
- \(E_g\) : événements attendus.
- \(n_g\) : taille du groupe.
Sous certaines conditions, la statistique est comparée à une distribution du chi-deux avec approximativement \(G-2\) degrés de liberté.

Ce test est sensible au regroupement choisi et à la taille de l'échantillon. Il doit être complété par des graphiques et d'autres métriques de calibration.

## 8.3. Brier Score

Le Brier Score mesure l'erreur quadratique moyenne des probabilités prédites :
\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

Plus il est faible, meilleure est la qualité globale des probabilités selon cette métrique.

Il combine des aspects de calibration et de discrimination ; il ne mesure pas uniquement la calibration.

## 9. Correction des tests multiples

Lorsque nous effectuons un grand nombre de tests statistiques, le risque d'obtenir au moins un faux positif augmente.

Par exemple, si nous réalisons 20 tests indépendants avec un seuil de 5 %, la probabilité d'obtenir au moins un faux positif sous les 20 hypothèses nulles est :
\[
1-(1-0.05)^{20}\approx64.2\%
\]

Il faut donc envisager des méthodes de correction.
Holm-Bonferroni
La méthode de Holm-Bonferroni contrôle le risque d'au moins un faux rejet parmi une famille d'hypothèses.

Elle est généralement moins conservatrice que la correction de Bonferroni classique.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Cette méthode est particulièrement utile lorsque nous effectuons des tests de calibration sur plusieurs groupes ou segments.

## 10. Validation temporelle et Backtesting

Pour les données temporelles, un split aléatoire peut produire une évaluation trop optimiste.

Nous devons respecter la chronologie.
Exemple de validation temporelle
Fold 1

Fold 2

Fold 3

Train

Test

Illustration d'une validation à fenêtre d'entraînement croissante.
TimeSeriesSplit

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Ce code suppose que X et y sont triés chronologiquement et que model est un classifieur compatible avec scikit-learn.

Pour certaines données, il faut ajouter un intervalle de séparation (gap) afin d'éviter les fuites temporelles, notamment lorsque les labels portent sur des horizons futurs qui se chevauchent.

## 11. Construire un verdict de validation

Dans un contexte professionnel, les résultats doivent être traduits en une décision compréhensible.

Un exemple de système est le Traffic Light Framework.
Exemple de verdict multi-dimensionnel
Discrimination

AUC et KS satisfaisants

Validé

Calibration

Sous-estimation sur certains segments

Réserves

Stabilité

Changements de population limités

Validé

Robustesse

Stress tests satisfaisants

Validé

Verdict global
Validé avec réserves

Exemple fictif. Les règles d'agrégation et les seuils doivent être définis avant l'évaluation.
Il ne faut pas nécessairement agréger toutes les métriques en un score unique.

Certains critères critiques peuvent justifier une réserve ou un rejet indépendamment des autres résultats.

## 12. Mini-projet — Automated Model Validation Framework

Projet : ML Validation & Backtesting Engine

Objectif : développer un moteur de validation capable d'évaluer automatiquement plusieurs modèles de classification.

Technologies : Python, scikit-learn, SciPy, statsmodels, pandas et Plotly.

## 13. À retenir

- La validation d'un modèle va au-delà du calcul de métriques globales.
- Les intervalles de confiance permettent de quantifier l'incertitude statistique.
- Le Bootstrap permet d'estimer la variabilité de nombreuses métriques.
- Les comparaisons de modèles doivent tenir compte de l'appariement des observations.
- Le Backtesting confronte les prédictions historiques aux résultats réellement observés.
- Les tests de calibration doivent être choisis selon leurs hypothèses et leurs limites.
- Les tests multiples nécessitent parfois une correction statistique.
- La validation temporelle doit respecter la chronologie et éviter les fuites d'information.
- Un verdict de validation doit être traçable, justifié et adapté aux exigences métier.

## 14. Ressources pour approfondir

- Scikit-learn — Model Evaluation
- SciPy — Statistical Tests
- Statsmodels — Statistical Modeling
- Scikit-learn — Cross-validation

Ce dernier cours réunira toutes les compétences du chapitre : évaluation avancée, calibration, XAI, robustesse, incertitude, fairness, diagnostic, monitoring et validation statistique dans un workflow professionnel complet.

## Coding Lab — implémentation complète

~~~python
"""Bootstrap apparié, McNemar exact, Holm et validation chronologique.
Hypothèse bootstrap : observations i.i.d. Les dépendances demandent d'autres rééchantillonnages.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def _validate(y, *probabilities):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Deux classes binaires et un vecteur non vide sont requis.")
    arrays = []
    for values in probabilities:
        p = np.asarray(values, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilités finies alignées dans [0,1] requises.")
        arrays.append(p)
    return (y, *arrays)

def compare_auc_bootstrap(y_true, proba_a, proba_b, n_bootstrap=300, seed=42):
    y, a, b = _validate(y_true, proba_a, proba_b)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("Au moins deux réplications sont requises.")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        differences.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    if len(differences) < 2:
        raise ValueError("Trop peu de réplications avec deux classes.")
    lower, upper = np.quantile(differences, [.025, .975])
    return {"difference": float(roc_auc_score(y, a)-roc_auc_score(y, b)),
            "lower": float(lower), "upper": float(upper),
            "n_valid": len(differences), "samples": np.asarray(differences)}

def bootstrap_auc(y_true, proba, n_bootstrap=300, seed=42):
    y, p = _validate(y_true, proba)
    # Une prédiction constante a une AUC de 0.5 sur chaque réplication valide.
    result = compare_auc_bootstrap(y, p, np.full(len(y), .5), n_bootstrap, seed)
    return {"auc": float(roc_auc_score(y, p)),
            "lower": result["lower"]+.5, "upper": result["upper"]+.5,
            "n_valid": result["n_valid"]}

def mcnemar_exact(y_true, pred_a, pred_b):
    y, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    if y.ndim != 1 or not len(y) or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("Vecteurs alignés non vides requis.")
    if not all(np.isin(v, [0, 1]).all() for v in (y, a, b)):
        raise ValueError("Labels et prédictions binaires requis.")
    ca, cb = a == y, b == y
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = float(binomtest(n10, n10+n01, .5).pvalue) if n10+n01 else 1.
    return {"n10": n10, "n01": n01, "pvalue": p}

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Un vecteur non vide de p-values dans [0,1] est requis.")
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    result = np.empty(len(p))
    result[order] = adjusted
    return result

def run(plot=False):
    X, y = make_classification(n_samples=750, n_features=8, n_informative=5, random_state=42)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=42)
    a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    b = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1).fit(X[train], y[train])
    pa, pb = a.predict_proba(X[test])[:, 1], b.predict_proba(X[test])[:, 1]
    print("IC AUC logistique :", bootstrap_auc(y[test], pa))
    comparison = compare_auc_bootstrap(y[test], pa, pb)
    print("Différence AUC logistique − forêt :", {k:v for k,v in comparison.items() if k!="samples"})
    print("McNemar :", mcnemar_exact(y[test], pa>=.5, pb>=.5))
    print("Holm :", holm_adjust([.001, .012, .032, .048, .18]))
    # Démonstration chronologique séparée, générée dans l'ordre du temps.
    rng = np.random.default_rng(7)
    Xt = rng.normal(size=(500, 4))
    yt = (Xt[:, 0] + .3*np.sin(np.arange(500)/50) + rng.normal(size=500) > 0).astype(int)
    rows = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=4, gap=5).split(Xt), 1):
        assert tr.max() < te.min()-5
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xt[tr], yt[tr])
        auc = roc_auc_score(yt[te], model.predict_proba(Xt[te])[:, 1]) if len(np.unique(yt[te]))==2 else np.nan
        rows.append({"fold": fold, "train_end": int(tr.max()), "test_start": int(te.min()), "AUC": auc})
    print("Backtesting chronologique synthétique :\n", pd.DataFrame(rows).to_string(index=False))
    print("Un gap doit correspondre à l'horizon réel des labels. L'IC bootstrap n'inclut pas l'incertitude du réentraînement.")
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(comparison["samples"], bins=25)
        plt.axvline(0, color="red")
        plt.xlabel("Différence AUC appariée (logistique − forêt)")
        plt.show()
    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)

~~~

## Hypothèses et limites du lab

Le lab présente un bootstrap percentile i.i.d. et ne prétend pas couvrir l'incertitude du réentraînement. Des clients répétés ou des séries temporelles nécessitent un bootstrap par groupes/blocs adapté. Les réplications mono-classe sont comptées et ignorées, avec erreur explicite s'il en reste trop peu. Les mêmes indices sont utilisés pour les deux modèles. Le backtesting temporel est une démonstration séparée, triée et munie d'un gap. Le test binomial simplifié du support suppose une probabilité commune et l'indépendance ; des probabilités hétérogènes conduisent plutôt à un cadre Poisson-binomial. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du projet

[[assessment]]