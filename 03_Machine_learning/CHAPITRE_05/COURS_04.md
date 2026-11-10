# Chapitre 5 — Cours 4 : Robustesse des modèles Machine Learning



## 1. Introduction : un modèle performant est-il forcément robuste ?

Imaginons que nous avons entraîné un modèle XGBoost pour prédire le risque de défaut de crédit.

Sur notre jeu de test, nous obtenons :
AUC-ROC
0,92

F1-score
0,87

Accuracy
94 %

Valeurs fictives pour illustrer le cours.

À première vue, le modèle semble excellent.

Mais imaginons que, quelques semaines après son déploiement :
- Certaines données contiennent des erreurs de saisie.
- Plusieurs variables présentent des valeurs manquantes.
- La distribution des revenus évolue.
- Des clients similaires obtiennent des prédictions très différentes après de petites modifications de leurs caractéristiques.
Le modèle peut alors perdre une partie de ses performances.

C'est précisément le problème étudié par la robustesse des modèles de Machine Learning.

Définition : la robustesse est la capacité d'un modèle à maintenir un comportement et des performances acceptables face à des perturbations, des imperfections des données ou des changements raisonnablement anticipés de son environnement.

La robustesse ne signifie pas qu'une prédiction ne doit jamais changer : certaines modifications de données sont réellement importantes et doivent influencer la décision.

## 2. Robustesse, généralisation et stabilité : quelles différences ?

Ces trois notions sont liées, mais elles ne sont pas interchangeables.
Concept	Question principale
Généralisation	Le modèle fonctionne-t-il sur de nouvelles observations provenant d'une distribution similaire ?
Robustesse	Résiste-t-il à des perturbations ou à des changements définis ?
Stabilité	Ses résultats changent-ils fortement lorsque les données ou les conditions d'entraînement varient légèrement ?

Prenons un exemple.

Un modèle de risque de crédit peut très bien généraliser sur les données de test de 2025, mais devenir moins fiable en 2026 si les conditions économiques évoluent.

Il peut aussi présenter une bonne performance globale tout en étant très sensible aux erreurs de mesure d'une variable.

La robustesse doit donc être évaluée indépendamment de la performance classique.
3. Les principales dimensions de la robustesse

Robustesse aux perturbations des données

Résistance aux erreurs de mesure, au bruit, aux valeurs manquantes et aux variations réalistes des variables d'entrée.

Robustesse aux changements de distribution

Capacité à maintenir des performances acceptables lorsque les données de production diffèrent des données d'entraînement.

Robustesse à l'échantillonnage et à l'entraînement

Capacité à obtenir des résultats comparables lorsque l'on modifie les échantillons d'entraînement, les graines aléatoires ou certains hyperparamètres.

Robustesse face aux perturbations adversariales

Résistance à des modifications intentionnelles des entrées conçues pour perturber les prédictions. C'est une dimension particulière, qui demande des tests adaptés au domaine.

## 4. Robustesse au bruit

## 4.1. Principe

Supposons qu'un modèle utilise le revenu mensuel d'un client.

Le revenu réel est de 3 000 DT, mais une erreur de mesure conduit à enregistrer 3 050 DT.

Une petite variation de cette nature ne devrait pas provoquer une modification disproportionnée de la prédiction, sauf si elle traverse un seuil réellement déterminant pour le modèle.

Mathématiquement, nous pouvons représenter une observation perturbée par :
\[
x'=x+\epsilon
\]

Avec :
- \(x\) : observation originale.
- \(\epsilon\) : perturbation.
- \(x'\) : observation perturbée.
On mesure ensuite la sensibilité de la prédiction :
\[
\Delta f = |f(x')-f(x)|
\]

Plus \(\Delta f\) est élevé pour une petite perturbation, plus le modèle est localement sensible dans cette situation.

## 4.2. Simulation interactive

[[robustnessincome]]

Voici une simulation pédagogique d'un modèle de risque de crédit dont la probabilité prédite varie selon le revenu.

Référence : revenu de 3 000 DT, probabilité de 26.9 %. Modifie le revenu pour observer la sensibilité de la prédiction.
Modèle mathématique fictif, sans données bancaires réelles.
Interprétation : si une modification relativement faible d'une variable entraîne un changement important de la prédiction, il faut examiner si cette sensibilité est justifiée par la logique métier et les données.

## 4.3. Tester le bruit en Python

Nous allons comparer les performances d'un modèle sur des données propres et sur des données légèrement perturbées.
import numpy as npfrom sklearn.datasets import make_classificationfrom sklearn.model_selection import train_test_splitfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import roc_auc_score# Génération des donnéesX, y = make_classification(    n_samples=2000,    n_features=10,    n_informative=6,    n_redundant=2,    random_state=42)X_train, X_test, y_train, y_test = train_test_split(    X, y,    test_size=0.25,    stratify=y,    random_state=42)# Entraînementmodel = RandomForestClassifier(    n_estimators=200,    random_state=42)model.fit(X_train, y_train)# Performance initialeproba_clean = model.predict_proba(X_test)[:, 1]auc_clean = roc_auc_score(y_test, proba_clean)

Dans cet exemple, nous utilisons un bruit gaussien. Dans un projet réel, les perturbations doivent respecter les types de variables, les contraintes métier et les erreurs plausibles.

## 5. Mesurer la dégradation des performances

Il ne suffit pas de constater qu'une métrique a diminué. Nous devons quantifier cette diminution.

Une mesure simple est :
\[
\Delta M = M_{\text{perturbé}}-M_{\text{référence}}
\]

Pour une métrique où une valeur élevée est préférable, comme l'AUC, une variation négative traduit une dégradation.
Exemple

Bruit	AUC fictive
0 %	0,92
5 %	0,90
10 %	0,86
15 %	0,81
20 %	0,76
30 %	0,70

Résultats fictifs : modèle évalué avec plusieurs intensités de bruit.

Dans cette simulation, l'AUC passe de 0,92 à 0,76 lorsque le niveau de bruit atteint 20 %.

La dégradation absolue est :
\[
\Delta AUC=0.76-0.92=-0.16
\]

Une perte de 0,16 point d'AUC est importante dans de nombreux contextes, mais le seuil d'acceptation doit être défini selon les exigences du projet.

Il faut également mesurer la variabilité des résultats en répétant les perturbations avec plusieurs graines aléatoires.

## 6. Robustesse aux valeurs manquantes

Les valeurs manquantes sont fréquentes en production.

Un modèle peut recevoir une observation pour laquelle certaines informations ne sont pas disponibles.

Trois stratégies courantes permettent de gérer cette situation :
- Imputation des valeurs manquantes.
- Utilisation de modèles capables de traiter les valeurs manquantes nativement.
- Ajout d'indicateurs de valeurs manquantes lorsque leur absence contient une information utile.
L'important est de reproduire en production la stratégie de préparation définie pendant l'entraînement.
Exemple avec un Pipeline

L'implémentation Python complète, remise en forme et exécutable, figure dans la section Coding Lab à la fin du support. Les exemples de cette section sont regroupés dans ce script pour conserver un protocole reproductible.

Ici, l'imputation est apprise uniquement sur les données d'entraînement, ce qui évite une fuite d'information depuis le jeu de test.

## 7. Robustesse aux changements de distribution

Une autre source majeure de fragilité est le Data Drift.

Le Data Drift apparaît lorsque la distribution des données évolue entre l'entraînement et la production.

Il faut distinguer plusieurs phénomènes :
Phénomène	Définition
Covariate Shift	La distribution \(P(X)\) change, tandis que \(P(Y\mid X)\) reste supposée stable.
Label Shift	La distribution \(P(Y)\) change, tandis que \(P(X\mid Y)\) reste supposée stable.
Concept Drift	La relation \(P(Y\mid X)\) change.

Par exemple, dans un modèle de crédit, une évolution économique peut modifier la répartition des revenus ou la relation entre l'endettement et le défaut.
Comment détecter ces changements ?
On peut utiliser plusieurs méthodes :
- PSI (Population Stability Index).
- KS test pour comparer des distributions numériques.
- Tests du chi-deux pour certaines variables catégorielles.
- Méthodes multivariées de détection de drift.
- Suivi des métriques de performance lorsque les véritables labels deviennent disponibles.
Un point essentiel : détecter du drift ne signifie pas automatiquement que les performances se sont dégradées. Il faut analyser son impact réel.

Le monitoring du drift sera approfondi dans la partie consacrée au suivi des modèles en production.

## 8. Robustesse à l'entraînement

Un modèle peut être sensible à la manière dont les données d'entraînement sont échantillonnées.

Par exemple, deux Random Forest entraînées sur des échantillons légèrement différents peuvent produire des résultats différents.

Pour étudier cette stabilité, nous pouvons répéter l'entraînement sur plusieurs partitions.
import numpy as npfrom sklearn.model_selection import StratifiedKFold, cross_val_scorefrom sklearn.ensemble import RandomForestClassifiermodel = RandomForestClassifier(    n_estimators=200,    random_state=42)cv = StratifiedKFold(    n_splits=5,    shuffle=True,    random_state=42)scores = cross_val_score(    model,    X_train,    y_train,    cv=cv,    scoring="roc_auc")print("AUC par fold :", scores)print("AUC moyenne :", scores.mean())print("Écart-type :", scores.std())

Un écart-type faible indique une variabilité limitée entre les folds de cette validation croisée.

Mais cela ne garantit pas une robustesse complète : il faut aussi tester d'autres échantillonnages, périodes et conditions de production.

## 9. Stress Testing des modèles ML

Le stress testing consiste à évaluer un modèle dans des conditions dégradées ou inhabituelles, mais pertinentes pour son utilisation.

Cette approche est particulièrement utile pour les systèmes critiques.
Exemple : modèle de risque de crédit
Scénarios de stress — exemple fictif
Scénario	AUC	Variation
Référence	0,89	—
Bruit modéré	0,86	−0,03
10 % de valeurs manquantes	0,84	−0,05
Changement de population	0,79	−0,10
Stress sévère	0,71	−0,18

Ces résultats montrent que la robustesse doit être étudiée sous plusieurs scénarios.

Il est également important de suivre d'autres métriques que l'AUC, notamment la calibration, le rappel des classes importantes et les performances par sous-population.

## 10. Construire un protocole de robustesse professionnel

Dans un projet réel, un protocole solide comprend généralement les étapes suivantes :
1. Définir le comportement attendu. Identifier les erreurs, perturbations et changements de population plausibles.
2. Établir une référence. Mesurer les performances sur un jeu de test représentatif, indépendant de l'entraînement.
3. Créer des scénarios. Simuler différents niveaux de bruit, valeurs manquantes et changements de distribution.
4. Répéter les expériences. Utiliser plusieurs graines et, lorsque nécessaire, plusieurs échantillons pour estimer l'incertitude.

## 5. Mesurer la dégradation. Comparer AUC, F1, Brier Score et d'autres métriques pertinentes.

6. Définir des seuils métier. Fixer des critères d'acceptation avant d'interpréter les résultats.
7. Documenter les limites. Identifier les scénarios problématiques et proposer des actions correctives.

## 11. Mini-projet pratique

Projet : ML Robustness Benchmark

Objectif : construire un outil Python qui évalue automatiquement la robustesse de plusieurs modèles de classification.

## 12. À retenir

- Un modèle performant n'est pas nécessairement robuste.
- La robustesse doit être testée face aux perturbations réalistes et aux changements de distribution.
- Le bruit, les valeurs manquantes, les changements de population et la variabilité de l'entraînement constituent des dimensions importantes.
- Le stress testing permet d'identifier les conditions dans lesquelles les performances deviennent insuffisantes.
- Une évaluation professionnelle doit inclure plusieurs scénarios, des répétitions et des critères d'acceptation définis à l'avance.



## Coding Lab — implémentation complète

~~~python
"""ML Robustness Benchmark. Synthétique ; seuils de décision fixés avant le test.
Le jeu de stress est un audit final : aucun choix de modèle n'est optimisé dessus.
--plot pour les figures. --xgboost ajoute XGBoost en environnement local.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

def perturb(X, scale, level=0.0, missing_rate=0.0, shift=0.0, seed=42):
    X = np.asarray(X, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if X.ndim != 2 or scale.shape != (X.shape[1],) or not np.isfinite(X).all():
        raise ValueError("Matrice X finie et échelles alignées requises.")
    if not np.isfinite(scale).all() or np.any(scale < 0) or not np.isfinite([level, missing_rate, shift]).all():
        raise ValueError("Paramètres finis et échelles positives ou nulles requis.")
    if level < 0 or not 0 <= missing_rate <= 1:
        raise ValueError("Bruit positif ou nul et taux manquant dans [0,1] requis.")
    rng = np.random.default_rng(seed)
    changed = X.copy() + rng.normal(size=X.shape) * scale * level
    changed[:, 0] += shift * scale[0]
    changed[rng.random(X.shape) < missing_rate] = np.nan
    return changed

def metrics(y, p):
    return {"AUC": roc_auc_score(y, p),
            "F1": f1_score(y, p >= .5, zero_division=0),
            "Brier": brier_score_loss(y, p)}

def run(plot=False, include_xgboost=False):
    X, y = make_classification(n_samples=900, n_features=8,
        n_informative=5, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42)
    estimators = {"Logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1)}
    if include_xgboost:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(n_estimators=60, max_depth=3,
            random_state=42, n_jobs=1, eval_metric="logloss")
    # Les échelles et l'imputation viennent du train, puis restent figées.
    scale = X_train.std(axis=0)
    rows = []
    for name, estimator in estimators.items():
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
        model.fit(X_train, y_train)
        reference = metrics(y_test, model.predict_proba(X_test)[:, 1])
        for scenario, args in [
            ("référence", {}),
            ("bruit 5 % écart-type train", {"level": .05}),
            ("bruit 20 % écart-type train", {"level": .20}),
            ("10 % absences", {"missing_rate": .10}),
            ("décalage de mesure +1 écart-type", {"shift": 1.0}),
            ("stress combiné", {"level": .30, "missing_rate": .20, "shift": 1.0}),
        ]:
            for seed in (7, 42, 99):
                changed = perturb(X_test, scale, seed=seed, **args)
                observed = metrics(y_test, model.predict_proba(changed)[:, 1])
                rows.append({"model": name, "scenario": scenario, "seed": seed,
                    **observed, "delta_AUC": observed["AUC"] - reference["AUC"],
                    "delta_Brier": observed["Brier"] - reference["Brier"]})
    report = pd.DataFrame(rows)
    assert np.isfinite(report.select_dtypes("number").to_numpy()).all()
    clean = report[report["scenario"] == "référence"]
    assert np.allclose(clean["delta_AUC"], 0)
    assert np.allclose(clean["delta_Brier"], 0)
    summary = report.groupby(["model", "scenario"])[["AUC", "F1", "Brier", "delta_AUC", "delta_Brier"]].agg(["mean", "std"])
    print(summary.to_string())
    print("Delta AUC < 0 : dégradation ; delta Brier > 0 : dégradation.")
    print("Le décalage simulé est une corruption de mesure avec labels inchangés, pas une preuve de covariate shift réel.")
    print("La variabilité sur les graines ne couvre ni l'incertitude d'échantillonnage ni la stabilité au réentraînement.")
    if plot:
        import matplotlib.pyplot as plt
        for name in report["model"].unique():
            group = report[report["model"] == name].groupby("scenario")["AUC"].mean()
            group.plot(kind="bar", title=name+" : AUC moyenne par stress")
            plt.tight_layout()
            plt.show()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    args = parser.parse_args()
    run(args.plot, args.xgboost)
~~~

## Limites du protocole et prolongements

Le benchmark compare logistique et forêt dans le navigateur. En local, ajoute --xgboost pour le troisième modèle et --plot pour les figures. Les graines décrivent la variabilité des perturbations sur le même test, pas l'incertitude d'échantillonnage ou de réentraînement. Ajoute des splits temporels et une analyse par sous-population pour le mini-projet complet.

Le décalage du script modifie une mesure et conserve les labels : c'est une corruption d'entrée. Il ne démontre pas un covariate shift réel avec P(Y|X) stable. Pour un véritable changement de population, utilise une population cible ou un générateur dont la relation cible est contrôlée.

Les critères d'acceptation doivent être fixés avant l'audit. Delta AUC négatif et delta Brier positif indiquent une dégradation, car ces métriques n'ont pas le même sens d'optimisation.

Sources : https://scikit-learn.org/stable/modules/compose.html et https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

## Vérifier ma compréhension

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du mini-projet

[[assessment]]