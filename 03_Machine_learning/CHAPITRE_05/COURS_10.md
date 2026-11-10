# Chapitre 5 — Cours 10 : End-to-End ML Evaluation & Governance Framework



## 1. Introduction — Du modèle performant au modèle fiable

Au cours de ce chapitre, nous avons étudié plusieurs dimensions essentielles de l'évaluation avancée des modèles de Machine Learning.

Nous savons désormais qu'un modèle ne peut pas être considéré comme fiable uniquement parce qu'il obtient une bonne accuracy ou une bonne AUC.

Imaginons un modèle de scoring de crédit présentant les résultats suivants :
Model Evaluation Scorecard — Exemple fictif
AUC-ROC
0,89
Bonne discrimination

Brier Score
0,042
À contextualiser

PSI
0,18
Changement modéré selon des seuils usuels

Écart de TPR
12 pts
Disparité à investiguer

Le modèle possède une bonne capacité de discrimination, mais plusieurs points nécessitent une investigation.

La question n'est donc plus seulement : « Le modèle est-il performant ? »

Elle devient :

« Le modèle est-il suffisamment fiable, robuste, explicable, équitable, stable et gouverné pour être utilisé dans son contexte métier ? »

C'est précisément l'objectif d'un End-to-End ML Evaluation & Governance Framework.

Dans ce dernier cours, nous allons réunir toutes les compétences du chapitre pour construire un processus professionnel de validation et de gouvernance.

## 2. Qu'est-ce qu'un ML Evaluation & Governance Framework ?

Un ML Evaluation & Governance Framework est un ensemble structuré de méthodes, de règles, de contrôles et de processus permettant d'évaluer et de superviser les modèles tout au long de leur cycle de vie.

Il doit répondre à six questions fondamentales.

Le modèle est-il performant ?

Mesurer sa capacité prédictive, ses erreurs et sa calibration.

Le modèle est-il robuste ?

Vérifier sa résistance aux perturbations et aux changements raisonnablement prévisibles.

Peut-on comprendre ses décisions ?

Analyser les variables influentes et expliquer les prédictions importantes.

Présente-t-il des disparités injustifiées ?

Auditer les résultats et les erreurs entre les populations pertinentes.

Reste-t-il fiable dans le temps ?

Surveiller le drift, les performances et les changements de contexte.

Son utilisation est-elle maîtrisée ?

Documenter les limites, les responsabilités, les validations et les décisions.

Ces dimensions ne sont pas indépendantes.

Par exemple, un modèle peut être globalement bien calibré, mais présenter une mauvaise calibration sur certains groupes.

Un framework efficace doit donc permettre une analyse transversale.

## 3. Architecture d'un framework de validation ML

Nous pouvons organiser le système autour de plusieurs modules.
Data & Model Registry

Datasets · Versions · Métadonnées · Artefacts

Data Quality & Validation Protocol

Intégrité · Cohérence · Splits · Prévention du leakage

Performance

AUC · F1 · Brier

Calibration

Reliability · ECE

Explainability

SHAP · LIME

Robustness

Stress tests

Uncertainty

Conformal Prediction

Fairness

Group metrics

Statistical Validation & Backtesting

Bootstrap · Tests statistiques · Validation temporelle

Governance & Decision Engine

Findings · Verdict · Approbations · Rapport

Continuous Monitoring

Drift · Performance · Alertes · Revalidation

Cette architecture est conceptuelle. Elle peut être implémentée dans un monolithe modulaire ou dans plusieurs services selon la taille du projet.

## 4. Étape 1 — Définir le protocole de validation

Avant de calculer les métriques, il faut définir un protocole.

Il doit préciser :
Élément	Questions à documenter
Objectif	Que prédit le modèle et pourquoi ?
Population	Sur quelles observations sera-t-il utilisé ?
Cible	Comment le label est-il défini ?
Horizon	Quelle période couvre la prédiction ?
Données	Quelles sources et versions sont utilisées ?
Splits	Comment les données sont-elles séparées ?
Métriques	Quels indicateurs sont pertinents ?
Seuils	Quels critères déclenchent une réserve ?
Risques	Quelles erreurs ont les conséquences les plus importantes ?
Validation	Qui examine et approuve les résultats ?

Pourquoi définir le protocole avant l'évaluation ?
Si nous choisissons les métriques et les seuils après avoir observé les résultats, nous risquons de favoriser involontairement le modèle évalué.

Un protocole défini à l'avance améliore la reproductibilité et limite ce risque.

## 5. Étape 2 — Organiser les tests de validation

Nous pouvons construire une matrice de validation.
Dimension	Tests possibles	Résultat attendu
Data Quality	Missing, doublons, types, règles métier	Données exploitables
Discrimination	AUC, Gini, KS, PR-AUC	Séparation satisfaisante
Calibration	Brier, courbe de calibration, tests adaptés	Probabilités fiables
Robustesse	Bruit, perturbations, données manquantes	Dégradation maîtrisée
Explainability	SHAP, analyse des variables	Comportement compréhensible
Fairness	TPR, FPR, disparités	Écarts évalués et justifiés
Incertitude	Couverture conforme, largeur	Incertitude utile
Stabilité	PSI, tests de drift	Changements surveillés
Validation statistique	Bootstrap, tests comparatifs	Conclusions étayées

Cette matrice n'est pas une checklist universelle.

Certains tests peuvent être inutiles pour un projet donné, tandis que d'autres contrôles spécifiques seront indispensables.

Par exemple, la calibration des probabilités est essentielle si un modèle est utilisé pour estimer des risques, mais elle peut être moins centrale dans une application qui utilise uniquement un classement.

## 6. Étape 3 — Implémenter un moteur d'évaluation

Pour LearnLoop, nous pouvons imaginer un module Python réutilisable.

## 6.1. Structure du projet

~~~text
ml_governance/
├── data_quality/
│   └── validator.py
├── evaluation/
│   ├── performance.py
│   ├── calibration.py
│   └── uncertainty.py
├── explainability/
│   └── shap_analysis.py
├── robustness/
│   └── stress_tests.py
├── fairness/
│   └── fairness_metrics.py
├── monitoring/
│   └── drift_detector.py
├── governance/
│   ├── rules.py
│   └── report.py
└── main.py
~~~

## 6.2. Créer un évaluateur de classification

Nous allons commencer par une version simple et extensible.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Ce module calcule plusieurs métriques de base.

Dans une application professionnelle, il faudra également enregistrer les effectifs, les distributions des classes, les intervalles de confiance et les métadonnées d'évaluation.

## 6.3. Exemple d'utilisation

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

## 7. Étape 4 — Définir un système de verdict

[[governanceverdict]]

Un framework de gouvernance doit transformer les résultats techniques en constats exploitables.

Cependant, toutes les métriques ne doivent pas nécessairement avoir le même poids.
Exemple de règles
Supposons que nous définissions les règles fictives suivantes :
Critère	Vert	Orange	Rouge
AUC	≥ 0,80	0,70–0,80	< 0,70
Brier Score	≤ 0,05	0,05–0,08	> 0,08
PSI	≤ 0,10	0,10–0,25	> 0,25

Les seuils sont uniquement pédagogiques. En particulier, le Brier Score dépend de la prévalence de la cible et doit être comparé à des références pertinentes.
Validé avec réserves

AUC-ROC

0.86

Brier Score

0.060

PSI

0.17

Discrimination — AUC

Vert

Calibration / erreur — Brier

Orange

Stabilité — PSI

Orange

Implémentation Python du Decision Engine

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Un moteur réel doit également gérer les métriques absentes, les résultats non concluants et les contrôles critiques bloquants.

## 8. Étape 5 — Documenter les résultats avec une Model Card

Une Model Card est un document synthétique présentant les caractéristiques, les performances et les limites d'un modèle.

Elle facilite la communication entre Data Scientists, ingénieurs ML, responsables métier, validateurs et auditeurs.
Exemple de structure
Model Card
Credit Default Prediction · Version 1.2

Description

Modèle de classification binaire estimant la probabilité de défaut sur un horizon défini.

Modèle

XGBoost · Classification supervisée

Performances

AUC : 0,89 · Brier Score : 0,042 · PSI : 0,18

Limites connues

Changements de population à surveiller et disparités de performance à investiguer.

Décision
Validation avec réserves — exemple

Exemple fictif simplifié. Une Model Card complète comprend aussi les données, les versions, les usages exclus, les évaluations détaillées et les responsables.

## 9. Étape 6 — Mettre en place la gouvernance

La gouvernance concerne la manière dont les modèles sont développés, validés, approuvés et surveillés.

## 9.1. Les rôles

Rôle	Responsabilité principale
Model Developer	Développer et documenter le modèle
Model Validator	Évaluer de manière indépendante ses performances et ses risques
Model Owner	Assumer la responsabilité métier du modèle
MLOps Engineer	Gérer le déploiement et l'exploitation
Risk / Compliance	Vérifier les exigences applicables
Auditor	Examiner la traçabilité et les contrôles

Dans une petite équipe, plusieurs rôles peuvent être exercés par les mêmes personnes, mais il est important de conserver des mécanismes de revue et de séparation des responsabilités adaptés au risque.

## 9.2. Les objets de gouvernance

Un système professionnel devrait idéalement enregistrer :
- Les modèles et leurs versions.
- Les datasets utilisés.
- Les configurations et les métriques.
- Les résultats des tests.
- Les anomalies ou findings.
- Les recommandations.
- Les décisions et approbations.
- Les rapports de validation.
- Les alertes de monitoring.
- Les changements apportés au modèle.

## 9.3. Model Registry

Un Model Registry permet de gérer le cycle de vie des modèles.

Exemple :
Model Lifecycle
Development

Validation

Production

Retirement

Des outils comme MLflow peuvent être utilisés pour le suivi des expériences et la gestion des versions des modèles.

## 10. Étape 7 — Relier validation et monitoring

La validation ne doit pas être considérée comme une activité réalisée une seule fois.

Après le déploiement, le modèle peut être exposé à de nouvelles populations et à de nouveaux comportements.

Il faut donc organiser une boucle continue.
Validation initiale

Tests · Rapport · Approbation

Déploiement

Version approuvée

Monitoring continu

Data Drift · Performance · Alertes

Investigation et décision

Maintien · Correction · Recalibration · Réentraînement

Revalidation avant toute nouvelle mise en production

Quand faut-il revalider un modèle ?
Une revalidation peut être nécessaire lorsque :
- Les performances se dégradent significativement.
- Un changement important de population est observé.
- Le modèle est réentraîné.
- Les variables ou les sources de données changent.
- Le contexte métier évolue.
- Une anomalie importante est détectée.
- Les exigences réglementaires applicables changent.
La fréquence et la profondeur de revalidation dépendent du risque et du contexte.

## 11. Les principales erreurs de gouvernance

Même un modèle techniquement performant peut devenir problématique si sa gouvernance est insuffisante.
Erreur 1 — Ne conserver que les métriques finales
Sans historique, il devient difficile de comprendre pourquoi une version a été approuvée.
Erreur 2 — Modifier les seuils après avoir observé les résultats
Cela peut introduire un biais de sélection et réduire la crédibilité de la validation.
Erreur 3 — Confondre validation et approbation automatique
Un ensemble de tests peut faciliter la décision, mais ne remplace pas nécessairement l'expertise métier et la revue indépendante.
Erreur 4 — Ignorer les limites du modèle
Un modèle peut être valide pour une population et inadapté à une autre.
Erreur 5 — Ne pas relier les alertes aux actions
Une alerte sans responsable, sans procédure d'investigation et sans suivi apporte peu de valeur.

## 12. Projet final du chapitre 5 — ML Trust & Governance Platform

Nous allons maintenant réunir toutes les compétences acquises dans un projet complet.
Projet final : ML Trust & Governance Platform

Une plateforme complète d'évaluation, de validation et de surveillance des modèles ML.

Fonctionnalités attendues

Stack technique proposée
Couche	Technologies
Backend	FastAPI, Python
Frontend	React, Tailwind CSS
Base de données	PostgreSQL
ML Evaluation	scikit-learn, SciPy, statsmodels
Explainability	SHAP
Fairness	Fairlearn
Monitoring	Evidently AI
Model Registry	MLflow
Visualisation	Plotly
Reporting	PDF / HTML

Ce projet peut constituer un excellent exercice d'intégration des concepts du chapitre, à condition de commencer par un MVP limité avant d'ajouter progressivement les modules avancés.

## 13. Synthèse complète du chapitre 5

Nous avons maintenant terminé les dix cours consacrés à l'évaluation avancée, à l'interprétabilité et à la robustesse des modèles ML.
Cours	Sujet	Compétence acquise
1	Advanced Model Evaluation	Évaluer les performances au-delà des métriques classiques
2	Probability Calibration	Vérifier la fiabilité des probabilités
3	Explainable AI — SHAP & LIME	Expliquer les prédictions
4	Model Robustness	Tester la résistance aux perturbations
5	Uncertainty & Conformal Prediction	Quantifier l'incertitude des prédictions
6	Fairness & Responsible ML	Auditer les disparités et les risques
7	Model Diagnostics & Error Analysis	Comprendre les erreurs
8	Data Drift & Model Monitoring	Surveiller les modèles dans le temps
9	Model Validation & Backtesting	Valider statistiquement les résultats
10	End-to-End ML Governance	Intégrer tous les contrôles dans un framework

Ce que tu dois maîtriser avant de passer au chapitre suivant
À ce stade, tu devrais pouvoir :
- Choisir des métriques adaptées à un problème métier.
- Identifier un modèle mal calibré.
- Expliquer les prédictions d'un modèle.
- Réaliser des stress tests.
- Construire des intervalles de prédiction conformes.
- Auditer les disparités entre groupes.
- Diagnostiquer les erreurs.
- Détecter les changements de distribution.
- Comparer statistiquement plusieurs modèles.
- Construire un processus de validation documenté et reproductible.

## 14. Ressources complémentaires

- Scikit-learn — Documentation officielle
- SHAP — Explainability
- Fairlearn — Fairness Assessment
- Evidently AI — Model Monitoring
- MLflow — Model Lifecycle Management

## Suite du parcours

Chapitre 5 : 10 cours disponibles ; 48 cours ML au total sur les cinq premiers chapitres. La progression personnelle dépend de tes quiz et exercices. Chapitre 6 proposé : Fondements du Deep Learning, à venir.


## Coding Lab — implémentation complète

~~~python
"""MVP de gouvernance pédagogique : évaluation, règles configurables et Model Card.
Le verdict est un triage technique, jamais une approbation de production.
"""
import math
import json
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, brier_score_loss, log_loss)

class ClassificationEvaluator:
    def __init__(self, threshold=.5):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Seuil dans [0,1] requis.")
        self.threshold = threshold

    def evaluate(self, y_true, y_proba):
        y, p = np.asarray(y_true), np.asarray(y_proba, dtype=float)
        if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("Cibles binaires et probabilités alignées non vides requises.")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilités finies dans [0,1] requises.")
        pred = (p >= self.threshold).astype(int)
        has_two_classes = len(np.unique(y)) == 2
        return {"n": int(len(y)), "prevalence": float(y.mean()), "threshold": self.threshold,
            "auc": float(roc_auc_score(y, p)) if has_two_classes else None,
            "average_precision": float(average_precision_score(y, p)) if has_two_classes else None,
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}

DEFAULT_RULES = {"auc": (.80, .70), "brier": (.05, .08), "psi": (.10, .25)}

def evaluate_model(metrics, psi, rules=None, critical_findings=None):
    rules = DEFAULT_RULES if rules is None else rules
    statuses, findings = {}, []
    for name, value in {"auc": metrics.get("auc"), "brier": metrics.get("brier"), "psi": psi}.items():
        if name not in rules or len(rules[name]) != 2:
            raise ValueError("Deux seuils requis pour " + name)
        green, amber = rules[name]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (green, amber)):
            raise ValueError("Seuils finis requis.")
        if (name == "auc" and green < amber) or (name != "auc" and green > amber):
            raise ValueError("Ordre des seuils incohérent.")
        valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
        valid = valid and math.isfinite(float(value)) and value >= 0 and (name == "psi" or value <= 1)
        if not valid:
            statuses[name] = "RED"
            findings.append(name + " absent, non défini ou invalide")
            continue
        if name == "auc":
            status = "GREEN" if value >= green else "AMBER" if value >= amber else "RED"
        else:
            status = "GREEN" if value <= green else "AMBER" if value <= amber else "RED"
        statuses[name] = status
        if status != "GREEN":
            findings.append(name + " : " + status + " selon les seuils pédagogiques")
    if critical_findings:
        statuses["critical_controls"] = "RED"
        findings.extend(str(v) for v in critical_findings)
    rank = {"GREEN": 0, "AMBER": 1, "RED": 2}
    verdict = max(statuses.values(), key=rank.get)
    return {"status": verdict, "criteria": statuses, "findings": findings, "requires_human_review": True}

def run():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from ch05_cours08_solution import calculate_psi
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5, random_state=42)
    train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    metrics = ClassificationEvaluator(threshold=.4).evaluate(y[test], model.predict_proba(X[test])[:, 1])
    psi = calculate_psi(X[train, 0], X[test, 0])
    verdict = evaluate_model(metrics, psi)
    card = {"model": "logistique-demo-v1", "dataset": "synthetique-seed42",
        "usage": "Démonstration pédagogique, sans usage de décision de crédit",
        "metrics": metrics, "psi_feature_0": psi, "triage": verdict,
        "not_evaluated": ["SHAP", "robustesse", "fairness", "incertitude conforme", "revue indépendante"],
        "approval": "NON APPROUVÉ POUR LA PRODUCTION",
        "next_steps": ["Compléter les contrôles pertinents", "Revue humaine", "Versionner les artefacts et décisions"]}
    print(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False))
    return card

if __name__ == "__main__":
    run()

~~~

## Hypothèses et limites du lab

Le Coding Lab est un MVP : il calcule les métriques, un PSI, le triage et une Model Card en JSON. Les contrôles non implémentés sont explicitement not_evaluated et le modèle reste NON APPROUVÉ POUR LA PRODUCTION. L'atelier ne construit pas la plateforme complète à ta place : le projet final conserve comme exercices l'intégration SHAP/Fairlearn/Evidently, les droits, le registre, l'historique et les approbations. Une AUC mono-classe vaut None ; les métriques manquantes, non finies ou hors domaine bloquent le triage. https://arxiv.org/abs/1810.03993

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du projet

[[assessment]]