# Chapitre 5 — Cours 6 : Fairness, Bias & Responsible Machine Learning



## 1. Introduction : un modèle performant est-il forcément équitable ?

Imaginons qu'une banque développe un modèle de Machine Learning pour prédire si un client remboursera son crédit.

Le modèle affiche une accuracy de 95 % et une AUC-ROC de 0,93.

Ces résultats semblent excellents.

Mais en analysant les prédictions, nous découvrons que le modèle commet beaucoup plus d'erreurs sur un groupe de clients que sur un autre.
Exemple fictif : taux de faux négatifs par groupe

Les graphiques du support utilisent des valeurs fictives. Le simulateur ci-dessous recalcule les taux à partir des effectifs choisis.

Données pédagogiques fictives. Un faux négatif correspond ici à un défaut réel que le modèle n'a pas détecté.

Dans cet exemple, le modèle manque 10 % des défauts réels du groupe A, contre 32 % pour le groupe B.

Cela soulève une question essentielle :

Un modèle peut-il être considéré comme fiable si ses performances sont très différentes selon les populations ?

C'est précisément le sujet de la Machine Learning Fairness, ou équité algorithmique.

L'objectif de ce cours est d'apprendre à détecter les biais, mesurer les disparités entre groupes et concevoir des modèles plus responsables.

## 2. Qu'est-ce que la Fairness en Machine Learning ?

La Fairness désigne l'ensemble des principes et méthodes visant à éviter que les systèmes algorithmiques produisent des différences de traitement injustifiées ou discriminatoires.

Elle ne signifie pas nécessairement que tous les individus doivent obtenir la même prédiction.

L'équité dépend du contexte, de l'objectif du modèle, des conséquences des décisions et des exigences juridiques et éthiques.

Nous distinguons notamment :

Group Fairness — Équité entre groupes

Évalue si les résultats, les erreurs ou les opportunités diffèrent entre plusieurs populations.

Individual Fairness — Équité individuelle

Cherche à garantir que des individus similaires selon une mesure de similarité pertinente soient traités de manière similaire.

Procedural Fairness — Équité procédurale

S'intéresse à la transparence, à la justification des décisions, à la possibilité de contestation et à la responsabilité des acteurs.

## 3. D'où viennent les biais algorithmiques ?

Un biais algorithmique n'est pas nécessairement une erreur de programmation.

Il peut provenir des données, de la définition de la cible, de l'entraînement ou de l'utilisation du modèle.

## 3.1. Sampling Bias

Le biais d'échantillonnage apparaît lorsque les données d'entraînement ne représentent pas correctement la population cible.

Exemple : un modèle de crédit entraîné principalement sur des clients ayant déjà obtenu un prêt peut mal généraliser aux personnes dont les demandes ont historiquement été refusées.

## 3.2. Historical Bias

Les données peuvent refléter des inégalités ou des décisions historiques.

Un modèle peut apprendre et reproduire ces tendances, même si les données sont techniquement correctes.

## 3.3. Measurement Bias

Les variables ou les labels peuvent être mesurés différemment selon les groupes.

Exemple : une variable utilisée comme approximation de la stabilité financière peut ne pas représenter la même réalité pour tous les profils.

## 3.4. Label Bias

La variable cible peut elle-même contenir des biais.

Par exemple, entraîner un modèle sur les décisions d'approbation passées ne signifie pas nécessairement apprendre la capacité réelle de remboursement.

## 3.5. Proxy Bias

Même lorsqu'une caractéristique sensible est retirée du dataset, d'autres variables peuvent indirectement en contenir l'information.

Par exemple, certaines variables géographiques peuvent être corrélées à des caractéristiques démographiques.

À retenir : supprimer une variable sensible ne suffit pas à garantir qu'un modèle est équitable.

## 4. Les principales métriques de Fairness

Il existe plusieurs définitions mathématiques de l'équité. Elles ne mesurent pas toutes la même chose.

Nous allons étudier les quatre plus importantes.

## 4.1. Demographic Parity

La Demographic Parity cherche à obtenir des taux de prédictions positives comparables entre groupes.
\[
P(\hat{Y}=1\mid A=0)
=
P(\hat{Y}=1\mid A=1)
\]

Avec :
- \(\hat{Y}\) : prédiction du modèle.
- \(A\) : variable représentant le groupe étudié.
Dans un contexte de crédit, si \(\hat{Y}=1\) représente l'approbation, cette métrique compare les taux d'approbation.

Exemple :
Groupe	Taux d'approbation
A	70 %
B	45 %

La différence de Demographic Parity est :
\[
DPD=0.70-0.45=0.25
\]

Soit un écart de 25 points de pourcentage.

Cet écart mérite une investigation, mais ne démontre pas à lui seul une discrimination : les distributions des caractéristiques pertinentes peuvent différer.

## 4.2. Equal Opportunity

L'Equal Opportunity compare les taux de vrais positifs entre groupes.
\[
P(\hat{Y}=1\mid Y=1,A=0)
=
P(\hat{Y}=1\mid Y=1,A=1)
\]

Autrement dit, le modèle devrait identifier les véritables cas positifs avec une fréquence comparable.

Pour un modèle détectant les défauts, cela signifie comparer la capacité à détecter les défauts réels.

## 4.3. Equalized Odds

L'Equalized Odds exige que les groupes présentent à la fois :
\[
TPR_{A=0}=TPR_{A=1}
\]

et
\[
FPR_{A=0}=FPR_{A=1}
\]

C'est donc une contrainte plus forte que l'Equal Opportunity.

Elle compare à la fois les vrais positifs et les faux positifs.

## 4.4. Predictive Parity

La Predictive Parity compare la précision des prédictions positives entre groupes :
\[
P(Y=1\mid \hat{Y}=1,A=0)
=
P(Y=1\mid \hat{Y}=1,A=1)
\]

Elle demande que la proportion de véritables positifs parmi les prédictions positives soit comparable.

Il faut faire attention : lorsque les taux de base diffèrent entre groupes, plusieurs définitions de fairness peuvent être mathématiquement incompatibles, sauf dans des cas particuliers.

## 5. Comprendre les métriques avec un exemple interactif

[[fairnessmetrics]]

Imaginons un modèle qui prédit si un client fera défaut.

Nous considérons la classe positive comme le défaut.

Pour chaque groupe, nous avons 100 clients réellement en défaut et 100 clients réellement non défaillants.

Modifie les résultats pour observer les disparités.

Métrique	Groupe A	Groupe B
TPR	90.0 %	68.0 %
FPR	15.0 %	25.0 %
Taux de prédictions positives	52.5 %	46.5 %

Les groupes ont volontairement des taux de base identiques pour simplifier la simulation. Les valeurs ne représentent pas une population réelle.

## 6. Implémentation Python : audit de Fairness

Nous allons entraîner un modèle de classification, puis comparer ses performances entre deux groupes synthétiques.

L'objectif est d'apprendre à construire un audit simple.

Le Coding Lab complet et remis en forme est disponible à la fin du cours, avec les mêmes étapes et les calculs manquants du collage.

Attention : les groupes de cet exemple sont générés aléatoirement et indépendamment des autres variables. Il n'est donc pas conçu pour démontrer un biais réel. Il sert à comprendre le calcul des métriques.

Dans un véritable audit, les groupes doivent correspondre à des catégories pertinentes et être traités conformément aux exigences de confidentialité et de protection des données.

## 7. Utiliser Fairlearn

Bibliothèque Python spécialisée dans l'évaluation et l'atténuation de certaines disparités entre groupes.

~~~bash
pip install fairlearn
~~~

Fairlearn permet notamment de calculer des métriques par groupe.

Le Coding Lab complet et remis en forme est disponible à la fin du cours, avec les mêmes étapes et les calculs manquants du collage.

Il est également possible d'utiliser la bibliothèque AIF360 pour réaliser certains audits et expérimenter des méthodes d'atténuation des biais.

## 8. Comment réduire les biais algorithmiques ?

Les approches d'atténuation sont généralement regroupées en trois familles.
Pre-processing

Agir sur les données avant l'entraînement.

Exemples : améliorer la représentativité des groupes, corriger certains biais de mesure, utiliser des pondérations adaptées.

L'objectif est de réduire les problèmes provenant des données elles-mêmes.

In-processing

Modifier la procédure d'apprentissage du modèle.

Exemples : ajouter des contraintes d'équité, modifier la fonction objectif ou utiliser des méthodes de réduction sous contraintes.

L'objectif est de prendre en compte certains critères de fairness pendant l'entraînement.

Post-processing

Ajuster les décisions après l'entraînement.

Exemples : calibrer les seuils de décision ou appliquer certaines méthodes de correction des disparités.

Ces méthodes doivent être évaluées au regard des contraintes légales et opérationnelles, notamment lorsqu'elles utilisent des caractéristiques sensibles.

## 9. Le compromis entre performance et fairness

Réduire une disparité peut modifier les performances globales du modèle.

Mais il ne faut pas supposer que l'amélioration de la fairness entraîne toujours une baisse de performance.

Dans certains cas, améliorer la qualité des données ou corriger un biais d'échantillonnage peut améliorer simultanément les deux.
Exemple fictif

Les graphiques du support utilisent des valeurs fictives. Le simulateur ci-dessous recalcule les taux à partir des effectifs choisis.

Illustration fictive du compromis entre accuracy et écart de TPR. Chaque point représente un modèle candidat.

L'objectif n'est pas nécessairement de sélectionner le modèle avec la meilleure accuracy.

Il s'agit de choisir une solution qui respecte les exigences de performance, de fairness, de robustesse et de conformité du projet.

## 10. Fairness et Explainable AI : quelle relation ?

Dans le cours 3, nous avons étudié SHAP et LIME.

Ces méthodes peuvent aider à comprendre les variables influentes et à identifier des comportements qui méritent une investigation.

Mais SHAP et LIME ne constituent pas des preuves d'équité.

Par exemple, une faible contribution SHAP moyenne d'une variable sensible ne garantit pas l'absence de discrimination.

Pour réaliser un audit complet, il faut combiner :
- Des métriques de fairness.
- Une analyse des erreurs par groupe.
- Une étude des variables et de leurs corrélations.
- Des méthodes d'explicabilité.
- Une analyse métier, éthique et juridique.
Il faut également analyser les intersections entre plusieurs caractéristiques lorsque cela est pertinent et autorisé.

## 11. Responsible Machine Learning

La fairness fait partie d'un domaine plus large : le Responsible AI.

Un système de Machine Learning responsable doit prendre en compte plusieurs dimensions.
Fairness

Équité et non-discrimination

Explainability

Compréhension des décisions

Robustness

Fiabilité face aux perturbations

Privacy

Protection des données

Accountability

Responsabilité et traçabilité

Transparency

Documentation et communication

Model Cards
Une pratique utile consiste à créer une Model Card pour chaque modèle.

Ce document décrit notamment :
- L'objectif et les usages prévus du modèle.
- Les données utilisées.
- Les performances globales et par sous-population.
- Les limites connues.
- Les risques identifiés.
- Les conditions d'utilisation et de surveillance.
Les Model Cards contribuent à améliorer la documentation et la gouvernance des modèles.

## 12. Mini-projet pratique — Fairness Audit

Projet : Responsible Credit Scoring

Objectif : développer un système d'audit permettant d'évaluer les disparités entre groupes dans un modèle de scoring de crédit.

## 13. À retenir

- Un modèle performant n'est pas nécessairement équitable.
- Les biais peuvent provenir des données, des labels, de l'entraînement ou des décisions historiques.
- Demographic Parity, Equal Opportunity, Equalized Odds et Predictive Parity mesurent différentes notions d'équité.
- Il n'existe pas une métrique de fairness universellement adaptée à tous les problèmes.
- Les méthodes de réduction des disparités peuvent intervenir avant, pendant ou après l'entraînement.
- Une démarche Responsible ML doit combiner performance, robustesse, explicabilité, confidentialité et gouvernance.

## 14. Ressources pour approfondir

- Fairlearn — Documentation officielle
- AI Fairness 360 — IBM
- Model Cards for Model Reporting — Article de recherche

Nous apprendrons à analyser systématiquement les erreurs d'un modèle, identifier les segments problématiques et utiliser ces diagnostics pour améliorer sa fiabilité.

## Coding Lab — implémentation complète

~~~python
"""Audit pédagogique : groupes synthétiques indépendants, pas une preuve de biais réel.
Défaut = classe positive 1 dans tout le lab. --fairlearn active l'audit et l'atténuation locaux.
--xgboost ajoute un deuxième modèle local. Aucune décision n'est choisie sur le test.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def group_metrics(y_true, y_pred, groups):
    y_true, y_pred, groups = (np.asarray(v) for v in (y_true, y_pred, groups))
    if y_true.ndim != 1 or not len(y_true) or y_pred.shape != y_true.shape or groups.shape != y_true.shape:
        raise ValueError("Trois vecteurs alignés non vides sont requis.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all() or pd.isna(groups).any():
        raise ValueError("Cibles et prédictions binaires, groupes non manquants requis.")
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        truth, pred = y_true[mask], y_pred[mask]
        tp = int(((truth == 1) & (pred == 1)).sum())
        fp = int(((truth == 0) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        tn = int(((truth == 0) & (pred == 0)).sum())
        def ratio(a, b):
            return a / b if b else np.nan
        rows.append({"group": group, "n": len(truth), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "prevalence": truth.mean(), "selection_rate": pred.mean(),
            "TPR": ratio(tp, tp+fn), "FPR": ratio(fp, fp+tn),
            "FNR": ratio(fn, tp+fn), "precision": ratio(tp, tp+fp),
            "accuracy": float((truth == pred).mean())})
    return pd.DataFrame(rows).set_index("group")

def disparity(report, metric):
    values = report[metric]
    if len(values) < 2 or values.isna().any():
        return np.nan
    return float(values.max() - values.min())

def run(include_fairlearn=False, include_xgboost=False):
    X, y = make_classification(n_samples=1500, n_features=10,
        n_informative=6, n_redundant=2, random_state=42)
    groups = np.random.default_rng(42).choice(["A", "B"], size=len(y))
    train_valid, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    train, valid = train_test_split(train_valid, test_size=.25,
        stratify=y[train_valid], random_state=7)
    estimators = {"Logistique": LogisticRegression(max_iter=1000)}
    if include_xgboost:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(n_estimators=70, max_depth=3, random_state=42,
                                              eval_metric="logloss", n_jobs=1)
    reports = {}
    for name, estimator in estimators.items():
        model = make_pipeline(StandardScaler(), estimator).fit(X[train], y[train])
        prediction = model.predict(X[test])
        report = group_metrics(y[test], prediction, groups[test])
        reports[name] = report
        print("\n", name, "\n", report.to_string())
        print("DP difference :", disparity(report, "selection_rate"))
        print("Equal opportunity difference :", disparity(report, "TPR"))
        eo = max(disparity(report, "TPR"), disparity(report, "FPR"))
        print("Equalized odds difference :", eo)
        print("Les NaN représentent des taux non mesurables, pas zéro.")
        if include_fairlearn:
            from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
            from fairlearn.postprocessing import ThresholdOptimizer
            frame = MetricFrame(metrics={"accuracy": accuracy_score},
                y_true=y[test], y_pred=prediction, sensitive_features=groups[test])
            print(frame.by_group)
            assert np.isclose(demographic_parity_difference(y[test], prediction,
                sensitive_features=groups[test]), disparity(report, "selection_rate"))
            assert np.isclose(equalized_odds_difference(y[test], prediction,
                sensitive_features=groups[test]), eo)
            # Contraintes ajustées sur validation uniquement ; accès au groupe requis à l'inférence.
            mitigator = ThresholdOptimizer(estimator=model, constraints="equalized_odds",
                objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
            mitigator.fit(X[valid], y[valid], sensitive_features=groups[valid])
            mitigated = mitigator.predict(X[test], sensitive_features=groups[test], random_state=42)
            print("Après post-processing (audit test, sans garantie de parité exacte hors validation) :")
            print(group_metrics(y[test], mitigated, groups[test]).to_string())
            print("Accuracy avant/après :", accuracy_score(y[test], prediction), accuracy_score(y[test], mitigated))
    print("\nMODEL CARD — démonstration synthétique, non destinée à décider de crédits.")
    print("Usage : apprendre un audit. Groupe A/B tiré aléatoirement, absent des features.")
    print("Limites : petits groupes, labels artificiels, pas de validation de discrimination réelle.")
    print("À compléter : contexte, dommages possibles, intersections, incertitude, recours, responsable et monitoring.")
    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairlearn", action="store_true")
    parser.add_argument("--xgboost", action="store_true")
    args = parser.parse_args()
    run(args.fairlearn, args.xgboost)

~~~

## Hypothèses et prolongements du lab

Sources : https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html et https://fairlearn.org/main/user_guide/mitigation/postprocessing.html.

Dans les paragraphes sur l'approbation, la classe positive est définie pour cet exemple seulement. Le simulateur et tous les scripts ci-dessous retiennent défaut = 1. Un taux d'alertes n'est donc pas un taux d'approbation.

Le lab génère les groupes indépendamment des données : il enseigne les calculs, sans démontrer une discrimination réelle. Les taux à dénominateur nul sont affichés NaN. Les écarts utilisent max − min et nécessitent des taux mesurables dans tous les groupes.

En local, --fairlearn vérifie les écarts et illustre ThresholdOptimizer ajusté sur validation, puis audité sur test. L'attribut de groupe est requis à l'inférence pour cette méthode ; son utilisation dépend du contexte et des règles applicables. L'atténuation n'est pas censée garantir la parité exacte sur de nouvelles données. La comparaison XGBoost est disponible avec --xgboost. Le mini-projet conserve l'analyse SHAP et la rédaction complète de la Model Card comme travail à réaliser.

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du mini-projet

[[assessment]]