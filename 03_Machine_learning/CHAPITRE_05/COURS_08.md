# Chapitre 5 — Cours 8 : Data Drift, Concept Drift & Model Monitoring



## 1. Introduction — Pourquoi un modèle se dégrade-t-il en production ?

Imaginons que nous avons développé un modèle XGBoost pour prédire le risque de défaut de crédit.

Pendant l'entraînement et la validation, nous obtenons d'excellents résultats.
Exemple fictif : évolution des performances
AUC initiale
0,91

AUC après 6 mois
0,76

Dégradation
−0,15

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Données fictives illustrant une dégradation progressive.

Pourtant, le code du modèle n'a pas changé.

Alors pourquoi les performances diminuent-elles ?

Parce que les données et les relations statistiques du monde réel peuvent évoluer, même lorsqu'un modèle reste identique.

Quelques exemples :
- Les comportements des clients changent.
- Les conditions économiques évoluent.
- De nouveaux produits ou profils apparaissent.
- Certaines variables sont collectées différemment.
- La relation entre les caractéristiques des clients et leur risque réel se transforme.
C'est ici qu'interviennent le Data Drift, le Concept Drift et le Model Monitoring.
Objectifs pédagogiques
À la fin de ce cours, tu sauras :

## 1. Différencier les principaux types de drift.

## 2. Détecter les changements de distribution avec des tests statistiques.

## 3. Comprendre PSI, KS, Jensen-Shannon et les tests multivariés.

## 4. Identifier une dégradation réelle des performances.

## 5. Construire un système de monitoring avec Evidently AI.

## 6. Définir des alertes et des stratégies de maintenance des modèles.

## 2. Les principaux types de Drift

Le terme drift désigne une évolution statistique des données ou du phénomène que le modèle cherche à prédire.

Il existe plusieurs catégories importantes.

## 2.1. Data Drift — Covariate Drift

Le Data Drift désigne un changement dans la distribution des variables d'entrée.
\[
P_{\text{train}}(X)\neq P_{\text{production}}(X)
\]

Exemple : un modèle bancaire a été entraîné sur une population dont le revenu mensuel moyen est de 3 000 DT.

Quelques mois plus tard, le revenu moyen de la population observée passe à 4 200 DT.

La distribution de la variable revenu a changé.
Distribution des revenus — avant et après

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Densités fictives, illustrant un décalage de distribution.

Le modèle peut continuer à fonctionner correctement malgré ce changement, ou au contraire voir ses performances diminuer.

Data Drift ne signifie pas automatiquement Performance Drift.

## 2.2. Label Drift — Prior Probability Shift

Le Label Drift correspond à une modification de la distribution de la variable cible.
\[
P_{\text{train}}(Y)\neq P_{\text{production}}(Y)
\]

Exemple :
Période	Taux de défaut observé
Entraînement	3 %
Production	7 %

Le taux de défaut a augmenté.

On parle plus précisément de Label Shift lorsque \(P(X\mid Y)\) reste stable alors que \(P(Y)\) change.

Il faut donc distinguer un simple changement du taux de défaut de l'hypothèse statistique complète du Label Shift.

## 2.3. Concept Drift

Le Concept Drift apparaît lorsque la relation entre les variables explicatives et la cible évolue.
\[
P_{\text{train}}(Y\mid X)
\neq
P_{\text{production}}(Y\mid X)
\]

Exemple :

En 2024, un certain niveau d'endettement était associé à un risque de défaut relativement faible.

En 2026, avec un contexte économique différent, le même niveau d'endettement peut être associé à un risque plus élevé.

La relation apprise par le modèle n'est plus entièrement représentative de la réalité.

Le Concept Drift est particulièrement important parce qu'il peut apparaître sans changement majeur de la distribution globale des variables d'entrée.

## 2.4. Prediction Drift

Le Prediction Drift désigne un changement de distribution des prédictions du modèle.
\[
P_{\text{train}}(\hat{Y})
\neq
P_{\text{production}}(\hat{Y})
\]

Exemple : la probabilité de défaut moyenne prédite passe de 4 % à 9 %.

Cela peut provenir d'un changement des données d'entrée, mais ne prouve pas que le modèle est devenu incorrect.
Tableau récapitulatif
Type	Ce qui change	Labels nécessaires ?
Data Drift	\(P(X)\)	Non
Label Drift	\(P(Y)\)	Oui, directement
Concept Drift	\(P(Y\mid X)\)	Généralement oui
Prediction Drift	\(P(\hat{Y})\)	Non
Performance Drift	Métriques du modèle	Oui

## 3. Les différentes formes temporelles de Concept Drift

Un changement peut apparaître de plusieurs manières.
Représentation schématique de changements brusques, progressifs et récurrents.
- Sudden Drift : changement brutal du comportement statistique.
- Gradual Drift : transition progressive entre deux distributions.
- Incremental Drift : évolution continue par petites modifications.
- Recurring Drift : retour périodique à des comportements déjà observés.
Exemple de drift récurrent : une variation saisonnière de la demande commerciale.

Une saisonnalité connue ne constitue pas nécessairement une anomalie. Le monitoring doit intégrer les cycles attendus.

## 4. Comment détecter le Data Drift ?

Plusieurs méthodes statistiques permettent de comparer les données de référence et les données courantes.

Nous allons étudier les plus importantes.

## 4.1. Population Stability Index — PSI

Le PSI mesure les différences de proportions entre deux distributions regroupées en classes.

Sa formule est :
\[
PSI=\sum_{i=1}^{K}(P_i-Q_i)\ln\left(\frac{P_i}{Q_i}\right)
\]

Avec :
- \(P_i\) : proportion dans la population de référence.
- \(Q_i\) : proportion dans la population courante.
- \(K\) : nombre de classes.
Le PSI est nul lorsque les proportions sont identiques et augmente lorsque les distributions s'éloignent.
Exemple
Tranche de revenu	Référence	Production
Faible	40 %	25 %
Moyen	35 %	35 %
Élevé	25 %	40 %

Le PSI de cet exemple est approximativement :
\[
PSI\approx0.141
\]

On rencontre souvent les seuils indicatifs suivants :
PSI	Interprétation courante
< 0,10	Changement faible
0,10 à 0,25	Changement modéré
> 0,25	Changement important

Ces seuils sont des conventions pratiques, pas des seuils statistiques universels. Ils doivent être adaptés au domaine, à la taille des échantillons et au découpage choisi.

## 4.2. Test de Kolmogorov-Smirnov

Le test KS à deux échantillons compare les fonctions de répartition empiriques de deux distributions numériques.
\[
D=\sup_x |F_{\text{ref}}(x)-F_{\text{prod}}(x)|
\]

Où \(F_{\text{ref}}\) et \(F_{\text{prod}}\) représentent les fonctions de répartition empiriques.

Plus \(D\) est élevé, plus l'écart maximal entre les distributions est important.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Une p-value supérieure à 0,05 ne prouve pas que les distributions sont identiques.

De plus, sur de très grands échantillons, de petites différences peuvent devenir statistiquement significatives sans être importantes pour le modèle.

## 4.3. Jensen-Shannon Divergence

La divergence Jensen-Shannon compare deux distributions de probabilité.

Elle repose sur la divergence de Kullback-Leibler :
\[
JS(P,Q)=
\frac12 KL(P\|M)+\frac12 KL(Q\|M)
\]

avec :
\[
M=\frac{P+Q}{2}
\]

Elle est symétrique et reste finie pour des distributions discrètes.

Avec un logarithme en base 2, sa valeur est comprise entre 0 et 1.

Elle est utile pour comparer des distributions catégorielles ou des distributions numériques discrétisées.

## 4.4. Méthodes multivariées

Le monitoring variable par variable peut manquer certains changements dans les relations entre variables.

Exemple : les distributions individuelles du revenu et de l'endettement restent stables, mais leur corrélation évolue.

Pour détecter ce type de changement, nous pouvons utiliser :
- Maximum Mean Discrepancy (MMD).
- Classifier Two-Sample Test.
- Tests de changement de corrélation.
- Méthodes de détection multivariée adaptées à la dimension des données.
Ces méthodes complètent les tests univariés.

## 5. Implémentation Python : détecter le Data Drift

Nous allons construire un détecteur simple qui compare une population de référence et une population courante.

## 5.1. Génération de données synthétiques

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Nous avons volontairement introduit un changement sur deux variables : le revenu et l'endettement. L'ancienneté reste statistiquement similaire.

## 5.2. Calcul du PSI

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Cette implémentation est pédagogique. En production, il faut aussi gérer explicitement les valeurs manquantes, les variables catégorielles, les bins dégénérés et la stabilité des résultats.

## 5.3. Détecter automatiquement les variables en drift

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Dans un système professionnel, il est préférable de distinguer la détection statistique, la gravité du changement et son impact métier.

## 6. Comment mesurer le Performance Drift ?

Le Performance Drift correspond à une dégradation des performances du modèle.

Contrairement au Data Drift, il nécessite généralement de disposer des véritables labels.

Pour un modèle de classification, nous pouvons surveiller :
Métrique	Ce qu'elle surveille
AUC-ROC	Pouvoir discriminant
Precision	Fiabilité des prédictions positives
Recall	Capacité à identifier les positifs
F1-score	Compromis precision-recall
Brier Score	Qualité des probabilités
Log Loss	Qualité des probabilités avec pénalisation des erreurs confiantes
Calibration	Correspondance entre probabilités prédites et fréquences observées

Pour un modèle de régression, nous surveillons notamment MAE, RMSE et les erreurs par segment.
Problème : les labels arrivent parfois en retard
Dans un système bancaire, il peut être nécessaire d'attendre plusieurs mois avant de connaître le défaut réel d'un client.

Nous ne pouvons donc pas calculer immédiatement l'AUC réelle des prédictions récentes.

Il faut distinguer deux niveaux de surveillance :
Monitoring sans labels

Disponible rapidement après les prédictions.
- Qualité des données.
- Data Drift.
- Prediction Drift.
- Valeurs manquantes.
- Volumes et anomalies techniques.

Monitoring avec labels

Disponible lorsque les résultats réels sont connus.
- AUC, F1 et Recall.
- Brier Score et calibration.
- Analyse des erreurs.
- Performance par sous-population.
- Détection de changements dans la relation entre variables et cible.

Point essentiel : le monitoring sans labels peut signaler des changements, mais ne remplace pas l'évaluation des performances réelles.

## 7. Construire un système de monitoring avec Evidently AI

Evidently AI est une bibliothèque Python qui permet d'analyser la qualité des données, les changements de distribution et les performances des modèles.

Elle peut être intégrée à un backend Python ou utilisée pour produire des rapports.

## 7.1. Installation

~~~bash
pip install evidently pandas numpy
~~~

## 7.2. Générer un rapport de Data Drift

L'API suivante correspond à la famille d'API Evidently 0.7.x ; selon la version installée, certains imports ou paramètres peuvent évoluer.

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Le rapport permet d'inspecter les distributions et les changements détectés.

Pour un projet de production, il faut également conserver les métadonnées du modèle, les dates des datasets et la configuration des tests.

## 7.3. Architecture d'un système de monitoring

Données de référence et données de production

Features · Prédictions · Labels disponibles

Monitoring Engine — Evidently

Data Quality · Drift · Performance

Historique

Métriques et résultats

Alerting

Seuils et notifications

Dashboard & Model Governance

Visualisation · Investigation · Décisions

Dans une architecture FastAPI, un service de monitoring peut exécuter les contrôles et enregistrer les résultats dans PostgreSQL. Le frontend React peut ensuite afficher les tendances et les alertes.

## 8. Définir une stratégie d'alerting

[[monitoringstatus]]

Détecter un drift ne suffit pas : il faut déterminer quand une intervention est nécessaire.

Voici un exemple de règles configurables.

Seuils illustratifs, non universels et à valider selon le modèle.
Indicateur	Vert	Orange	Rouge
PSI	≤ 0,10	0,10–0,25	> 0,25
Baisse d'AUC	≤ 0,02	0,02–0,05	> 0,05
Valeurs manquantes	≤ 5 %	5–10 %	> 10 %

PSI

0.14

Baisse d'AUC

0.03

Valeurs manquantes

4 %

Data Drift — PSI

0.14

Dégradation AUC

0.03

Valeurs manquantes

4 %

## 9. Que faire lorsqu'un Drift est détecté ?

Un système de monitoring professionnel doit associer les alertes à des actions précises.

## 1. Vérifier les données

Examiner les problèmes de collecte, de schéma, de valeurs manquantes et de transformation.

## 2. Mesurer l'importance du changement

Identifier les variables concernées, les segments touchés et la persistance du drift.

## 3. Évaluer les performances

Lorsque les labels sont disponibles, vérifier les métriques et la calibration.

## 4. Déterminer la cause

Distinguer changement de population, problème technique, évolution du phénomène et variation saisonnière.

## 5. Choisir une action

Maintenir le modèle, corriger les données, recalibrer, réentraîner ou envisager son remplacement.

## 6. Valider avant déploiement

Comparer toute nouvelle version avec la version actuelle sur des données indépendantes et pertinentes.
Le réentraînement automatique sans validation peut introduire de nouveaux risques, notamment si les données récentes sont biaisées ou de mauvaise qualité.

## 10. Monitoring continu : architecture MLOps

Un système MLOps complet doit relier le monitoring au cycle de vie du modèle.
Data Pipeline

Collecte et validation

Model Training

Entraînement et évaluation

Deployment

Mise en production

Monitoring

Drift et performances

Boucle d'amélioration continue

Le monitoring peut déclencher une investigation et, après validation, une nouvelle itération d'entraînement ou de déploiement.
Fréquence de monitoring
La fréquence dépend du contexte :
Type de système	Fréquence indicative
Détection de fraude	Proche du temps réel ou quotidienne
Recommandation	Quotidienne ou hebdomadaire
Prévision commerciale	Hebdomadaire ou mensuelle
Scoring de crédit	Selon la disponibilité des données, des labels et les exigences de validation

Il faut éviter de surveiller trop fréquemment des métriques calculées sur de très petits échantillons, car les variations peuvent être principalement dues au hasard.

## 11. Mini-projet pratique — ML Monitoring Dashboard

Projet : Production ML Monitoring System

Objectif : développer une application de monitoring capable de détecter le drift et de suivre les performances d'un modèle.

Stack proposée : Python, FastAPI, Evidently AI, PostgreSQL et React.

## 12. Les erreurs fréquentes à éviter

- Confondre drift et baisse de performance : une distribution peut changer sans que le modèle devienne moins performant.
- Utiliser uniquement le PSI : il ne détecte pas toutes les formes de changement, notamment les changements multivariés.
- Ignorer les labels retardés : certaines métriques ne peuvent pas être calculées immédiatement.
- Multiplier les tests sans contrôle : tester de nombreuses variables peut générer des alertes statistiques fortuites.
- Réentraîner automatiquement après chaque alerte : le réentraînement doit être justifié et validé.
- Comparer des périodes non comparables : la saisonnalité, les changements de population et les modifications de collecte peuvent fausser l'interprétation.

## 13. À retenir

- Le Data Drift concerne les changements dans la distribution des variables d'entrée.
- Le Label Drift concerne les changements de distribution de la cible.
- Le Concept Drift concerne les changements de la relation entre les variables et la cible.
- Le Prediction Drift concerne les changements dans la distribution des prédictions.
- Le Performance Drift correspond à l'évolution des métriques de performance.
- Le PSI, KS et Jensen-Shannon permettent de comparer des distributions, avec des limites différentes.
- Evidently AI facilite la construction de contrôles automatisés.
- Le monitoring doit être associé à une stratégie d'alerting, d'investigation et de gouvernance.

## 14. Ressources pour approfondir

- Evidently AI — Documentation officielle
- Scikit-learn — ROC AUC
- SciPy — Test KS à deux échantillons

Nous apprendrons à construire un processus de validation indépendant, utiliser le bootstrap et les tests statistiques, comparer plusieurs modèles et vérifier si les performances observées sont réellement fiables.

## Coding Lab — implémentation complète

~~~python
"""Monitoring pédagogique : PSI, KS, JS et absences. --evidently pour le rapport HTML local.
Sans labels, aucune AUC de production ne peut être déduite de ces seuls contrôles.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

def calculate_psi(reference, current, bins=10):
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins doit être un entier >= 2.")
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if ref.ndim != 1 or cur.ndim != 1:
        raise ValueError("Deux vecteurs numériques sont requis.")
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        raise ValueError("Échantillon fini vide.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins+1)))
    if len(edges) < 2:
        raise ValueError("Référence constante : définir des bins métier explicites.")
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(ref, bins=edges)
    b, _ = np.histogram(cur, bins=edges)
    # Lissage positif puis normalisation ; les bornes extrêmes conservent toute la masse.
    p = (a+1e-6)/(a.sum()+1e-6*len(a))
    q = (b+1e-6)/(b.sum()+1e-6*len(b))
    return float(np.sum((p-q)*np.log(p/q)))

def monitoring_status(psi, auc_drop, missing_rate):
    values = np.asarray([psi, auc_drop, missing_rate], dtype=float)
    if not np.isfinite(values).all() or psi < 0 or not -1 <= auc_drop <= 1 or not 0 <= missing_rate <= 1:
        raise ValueError("Indicateurs finis ; PSI >= 0, delta AUC dans [-1,1], absences dans [0,1].")
    # auc_drop = AUC référence − AUC courante ; négatif = amélioration.
    statuses = [
        2 if psi > .25 else 1 if psi > .10 else 0,
        2 if auc_drop > .05 else 1 if auc_drop > .02 else 0,
        2 if missing_rate > .10 else 1 if missing_rate > .05 else 0,
    ]
    return ["GREEN", "AMBER", "RED"][max(statuses)]

def run(include_evidently=False):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"revenu": rng.normal(3000, 700, 800),
        "endettement": rng.normal(35, 10, 800), "anciennete": rng.normal(8, 3, 800)})
    current = pd.DataFrame({"revenu": rng.normal(3800, 800, 800),
        "endettement": rng.normal(42, 12, 800), "anciennete": rng.normal(8, 3, 800)})
    current.loc[:31, "revenu"] = np.nan
    rows = []
    for column in reference:
        a, b = reference[column].dropna().to_numpy(), current[column].dropna().to_numpy()
        statistic, pvalue = ks_2samp(a, b)
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(a, edges)[0]; q = np.histogram(b, edges)[0]
        rows.append({"variable": column, "PSI": calculate_psi(a, b),
            "KS": statistic, "p_value": pvalue,
            "p_bonferroni": min(1., pvalue*len(reference.columns)),
            # scipy retourne une DISTANCE : carré pour obtenir la divergence.
            "JS_divergence_base2": jensenshannon(p/p.sum(), q/q.sum(), base=2)**2,
            "missing_reference": reference[column].isna().mean(),
            "missing_current": current[column].isna().mean()})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("KS significatif ≠ impact métier ; PSI heuristique ≠ test de baisse d'AUC.")
    print("Pas de labels ici : performance et concept drift restent à investiguer.")
    if include_evidently:
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset
        definition = DataDefinition(numerical_columns=list(reference.columns))
        snapshot = Report(metrics=[DataDriftPreset()]).run(
            reference_data=Dataset.from_pandas(reference, data_definition=definition),
            current_data=Dataset.from_pandas(current, data_definition=definition))
        snapshot.save_html("data_drift_report.html")
        print("Rapport local : data_drift_report.html")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidently", action="store_true")
    run(parser.parse_args().evidently)

~~~

## Hypothèses et limites du lab

Le PSI utilise des bins de référence, des bornes infinies et un lissage normalisé ; les absences sont suivies séparément. Les seuils PSI sont heuristiques, non universels. Le lab corrige trois p-values KS par Bonferroni et distingue taille d'effet, détection statistique et impact métier. SciPy retourne la distance Jensen-Shannon : son carré est la divergence. Le drapeau local --evidently utilise la famille d'API 0.7.x ; conserver la version installée avec le rapport. Les groupes temporels doivent être comparables. https://docs.evidentlyai.com/docs/library/report

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du projet

[[assessment]]