# Chapitre 5 — Cours 5 : Incertitude des prédictions et Conformal Prediction



## 1. Introduction : pourquoi une prédiction ne suffit-elle pas ?

Imaginons que nous avons entraîné un modèle de Machine Learning pour prédire le prix d'une maison.

Le modèle annonce :
Prix prédit d'une maison

Modèle A
Sans indication d'incertitude

Modèle B
Intervalle : 320 000–380 000 DT

Les deux modèles fournissent la même estimation ponctuelle, mais le modèle B communique une information supplémentaire : une plage de valeurs plausibles.

Cette information est essentielle dans de nombreux domaines : finance, médecine, industrie, prévisions de demande ou maintenance prédictive.

Cependant, un intervalle n'est utile que si nous comprenons comment il est construit et quelles garanties il possède.

C'est l'objectif de ce cours.

À la fin, tu sauras distinguer les différentes sources d'incertitude, comprendre la Conformal Prediction, construire des intervalles de prédiction et évaluer leur qualité.

## 2. Les deux grandes sources d'incertitude

En Machine Learning, nous distinguons généralement deux types d'incertitude.

Incertitude aléatoire — Aleatoric Uncertainty

Elle provient de la variabilité inhérente aux observations, même si nous connaissions parfaitement le processus de génération des données.

Exemple : deux maisons présentant des caractéristiques similaires peuvent avoir des prix différents en raison de facteurs non observés.

Incertitude épistémique — Epistemic Uncertainty

Elle provient de notre connaissance imparfaite du phénomène ou du modèle, notamment lorsque les données d'entraînement sont limitées.

Exemple : un modèle entraîné principalement sur des appartements peut être moins fiable pour estimer le prix d'une villa très atypique.

Une différence fondamentale :
Aleatoric	Epistemic
Liée à la variabilité des données	Liée aux limites de connaissance du modèle
Généralement irréductible avec davantage de données sur les mêmes variables	Peut souvent diminuer avec des données pertinentes supplémentaires
Peut être constante ou dépendre des entrées	Souvent élevée dans les régions peu représentées

Ces catégories constituent un cadre conceptuel. Leur séparation exacte dépend des hypothèses du modèle.
Visualisation intuitive
Illustration conceptuelle : dispersion autour d'une relation et désaccord possible entre estimations du modèle.

## 3. Quantifier l'incertitude : les approches classiques

Plusieurs méthodes existent pour aller au-delà d'une prédiction ponctuelle.

## 3.1. Intervalles de confiance

Un intervalle de confiance concerne généralement un paramètre inconnu ou une quantité statistique estimée, comme la moyenne conditionnelle.

Exemple : estimer le prix moyen attendu des maisons présentant certaines caractéristiques.

## 3.2. Intervalles de prédiction

Un intervalle de prédiction vise une nouvelle observation individuelle.

Par exemple :
\[
\hat{y}(x)=350\,000
\]

\[
I(x)=[320\,000,\ 380\,000]
\]

Un intervalle de prédiction prend en compte la variabilité des futures observations, et pas uniquement l'incertitude sur la moyenne.

## 3.3. Approches bayésiennes

Les modèles bayésiens représentent l'incertitude sur les paramètres par des distributions de probabilité.

Ils permettent notamment de construire des distributions prédictives.

## 3.4. Ensembles de modèles

On peut entraîner plusieurs modèles, puis analyser la dispersion de leurs prédictions.

Cette dispersion fournit une information sur le désaccord entre modèles, mais elle ne constitue pas automatiquement un intervalle de prédiction correctement calibré.

## 3.5. Conformal Prediction

La Conformal Prediction est une famille de méthodes permettant de construire des ensembles ou intervalles de prédiction avec des garanties de couverture statistique sous certaines hypothèses.

C'est cette approche que nous allons approfondir.

## 4. Qu'est-ce que la Conformal Prediction ?

La Conformal Prediction est une méthode qui utilise les erreurs observées sur un ensemble de calibration pour déterminer l'incertitude associée à de nouvelles prédictions.

Son principal avantage est qu'elle peut être appliquée à de nombreux modèles existants.

On parle souvent de méthode model-agnostic.

Par exemple, elle peut être utilisée avec :
- Régression linéaire.
- Random Forest.
- XGBoost.
- Réseaux de neurones.
- Autres modèles de régression ou classification.
L'idée fondamentale
Supposons que notre modèle commette les erreurs suivantes sur un ensemble de calibration :
Erreurs absolues de calibration

Scores fictifs triés (milliers de DT) : 5, 7, 8, 10, 11, 12, 14, 15, 18, 20.

Données fictives exprimées en milliers de dinars.
Si nous souhaitons construire des intervalles avec une couverture cible de 90 %, nous devons déterminer un quantile approprié des erreurs de calibration.

Ensuite, nous construisons un intervalle autour de la prédiction du modèle.

## 5. Split Conformal Prediction : fonctionnement

La version la plus simple pour la régression est appelée Split Conformal Prediction.

Elle se déroule en cinq étapes.
Étape 1 — Diviser les données

Le modèle est entraîné uniquement sur le premier ensemble. La calibration utilise un ensemble indépendant, et le test final reste réservé à l'évaluation.

Étape 2 — Entraîner le modèle

Nous entraînons un modèle de régression classique :
\[
\hat{f}:X\rightarrow Y
\]

Le modèle produit une prédiction ponctuelle \(\hat{f}(x)\).

Étape 3 — Calculer les scores de non-conformité

Pour chaque observation de calibration :
\[
s_i=|y_i-\hat{f}(x_i)|
\]

Le score mesure ici l'erreur absolue de prédiction.

Étape 4 — Calculer le quantile conforme

Pour \(n\) observations de calibration et un risque \(\alpha\), nous utilisons le rang :
\[
k=\lceil(n+1)(1-\alpha)\rceil
\]

Lorsque \(k\leq n\), nous prenons le \(k\)-ième plus petit score de calibration.

Lorsque \(k=n+1\), le quantile conforme est formellement infini, ce qui signifie que l'échantillon est trop petit pour obtenir une borne finie à ce niveau de couverture sans hypothèses supplémentaires.

Étape 5 — Construire l'intervalle

Pour une nouvelle observation :
\[
C(x)=[\hat{f}(x)-q,\ \hat{f}(x)+q]
\]

Où \(q\) est le quantile conforme obtenu à l'étape précédente.

## 6. Exemple numérique complet

Supposons que les scores de calibration triés soient :
\[
[5,7,8,10,11,12,14,15,18,20]
\]

Nous avons :
\[
n=10,\qquad \alpha=0.1
\]

Le rang du quantile est :
\[
k=\lceil(10+1)\times0.9\rceil=10
\]

Le dixième score est 20.

Donc :
\[
q=20\,000\ \text{DT}
\]

Pour une maison dont le prix prédit est 350 000 DT :
\[
C(x)=[350\,000-20\,000,\ 350\,000+20\,000]
\]

\[
\boxed{C(x)=[330\,000,\ 370\,000]\ \text{DT}}
\]

Ce résultat vise une couverture marginale d'au moins 90 % sous l'hypothèse d'échangeabilité des observations de calibration et de la future observation.

Il ne signifie pas que chaque maison possède individuellement une probabilité garantie de 90 % d'appartenir à son intervalle.

## 7. Simulateur interactif d'intervalles conformes

[[conformalinterval]]

Modifie la prédiction et le niveau de couverture souhaité.

Simulation fondée sur les 10 erreurs fictives de l'exemple. Les quantiles sont discrets en raison de la petite taille de l'échantillon.

## 8. Implémentation Python : Split Conformal Regression

Nous allons construire un exemple complet avec scikit-learn.

Le Coding Lab complet et remis en forme est disponible à la fin du cours, avec les mêmes étapes et les calculs manquants du collage.

Interprétation
La couverture empirique est :
\[
\widehat{\text{Coverage}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\{y_i\in C(x_i)\}
\]

Si la couverture cible est 90 %, nous souhaitons que la couverture observée sur un grand jeu de test représentatif soit proche de cette valeur, sans nécessairement lui être exactement égale.

La garantie de Split Conformal est marginale et porte sur la procédure, sous échangeabilité. Elle ne garantit pas une couverture de 90 % conditionnellement à chaque jeu de calibration fixé.

## 9. Classification : Conformal Prediction Sets

Jusqu'à présent, nous avons étudié la régression.

Mais comment utiliser la Conformal Prediction pour la classification ?

En classification, l'objectif n'est pas de construire un intervalle numérique, mais un ensemble de classes possibles.
Exemple
Un modèle prédit les probabilités suivantes pour une image :
Probabilités prédites
Chat

55 %

Chien

35 %

Lapin

10 %

Une méthode conforme pourrait retourner :
\[
C(x)=\{\text{Chat},\text{Chien}\}
\]

Cela signifie que ces deux classes sont retenues comme candidates par la procédure de prédiction conforme.

Le fait qu'une classe appartienne à l'ensemble ne signifie pas qu'elle possède individuellement une probabilité de 90 %.
Une méthode simple : score \(1-p\)
Pour une observation de calibration dont la vraie classe est \(y_i\), nous définissons :
\[
s_i=1-\hat{p}(y_i\mid x_i)
\]

Nous calculons ensuite le quantile conforme \(q\).

Pour une nouvelle observation :
\[
C(x)=\{k:1-\hat{p}(k\mid x)\leq q\}
\]

Cette méthode peut produire un ensemble contenant une classe, plusieurs classes, voire aucune, selon le seuil.

D'autres méthodes, comme APS (Adaptive Prediction Sets) et RAPS (Regularized Adaptive Prediction Sets), utilisent des scores de non-conformité différents.

## 10. Les limites de la Conformal Prediction

La Conformal Prediction possède des garanties intéressantes, mais elle n'est pas magique.

## 10.1. L'hypothèse d'échangeabilité

La garantie classique suppose que les observations de calibration et la nouvelle observation soient échangeables.

Intuitivement, cela signifie que leur ordre ne doit pas apporter d'information sur leur distribution jointe.

Cette hypothèse peut être compromise dans des données temporelles, des séries financières ou en présence de changements de distribution.

## 10.2. Couverture marginale vs conditionnelle

Une couverture marginale de 90 % ne garantit pas une couverture de 90 % pour chaque sous-population.

Par exemple, un modèle peut atteindre 90 % globalement, mais seulement 75 % sur un segment particulier.

## 10.3. Largeur des intervalles

Un intervalle très large peut atteindre une excellente couverture, tout en étant peu utile.

Il faut donc évaluer simultanément :
- La couverture.
- La largeur moyenne des intervalles.
- La couverture par sous-population.
- La stabilité des résultats.

## 10.4. Changement de distribution

Si les données de production évoluent fortement, les garanties classiques peuvent ne plus s'appliquer.

Des variantes spécialisées existent pour certaines formes de changement de distribution ou de dépendance temporelle, mais elles nécessitent des hypothèses supplémentaires.

## 11. Méthodes avancées de Conformal Prediction

Split Conformal Prediction

Simple, rapide et facile à intégrer à un modèle existant. Elle utilise un ensemble de calibration séparé.

Cross-Conformal et CV+

Exploitent plusieurs partitions ou modèles pour mieux utiliser les données disponibles. Leurs garanties et constructions varient selon la méthode exacte.

Conformalized Quantile Regression (CQR)

Combine la régression quantile avec une calibration conforme afin de produire des intervalles dont la largeur peut varier selon les caractéristiques des observations.

Adaptive Prediction Sets (APS)

Méthode de classification utilisant les probabilités cumulées des classes pour construire des ensembles de prédiction adaptatifs.

Pourquoi CQR est particulièrement intéressant ?
Dans la version simple de Split Conformal avec erreur absolue, tous les intervalles possèdent la même largeur :
\[
C(x)=[\hat{f}(x)-q,\hat{f}(x)+q]
\]

Mais certaines observations sont naturellement plus difficiles à prédire que d'autres.

CQR permet de construire des intervalles adaptatifs :
\[
C(x)=
[\hat{q}_{low}(x)-q,\hat{q}_{high}(x)+q]
\]

Où \(\hat{q}_{low}\) et \(\hat{q}_{high}\) sont des quantiles prédits par un modèle de régression quantile, et \(q\) est une correction conforme.

Cette approche est particulièrement utile lorsque l'incertitude varie fortement selon les observations.

## 12. Mini-projet pratique — Uncertainty-Aware ML

Projet : House Price Prediction with Uncertainty

Objectif : construire un modèle de prédiction immobilière capable de fournir une estimation et un intervalle de prédiction calibré.

## 13. À retenir

- Une prédiction ponctuelle ne fournit pas, à elle seule, une mesure de l'incertitude.
- L'incertitude aléatoire provient de la variabilité des observations ; l'incertitude épistémique provient des limites de connaissance du modèle.
- La Conformal Prediction permet de construire des intervalles ou ensembles de prédiction avec des garanties marginales sous certaines hypothèses.
- Split Conformal utilise les erreurs d'un ensemble de calibration indépendant.
- La couverture et la largeur des intervalles doivent être évaluées conjointement.
- La Conformal Prediction ne garantit pas automatiquement une couverture conditionnelle pour chaque observation ou sous-population.

## 14. Ressources pour approfondir

Pour aller plus loin, voici deux références importantes :
- A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification — Angelopoulos et Bates.
- Conformalized Quantile Regression — Romano, Patterson et Candès.

Nous étudierons comment identifier les biais algorithmiques, mesurer l'équité entre différents groupes et évaluer les compromis entre performance, fairness et exigences métier.

## Coding Lab — implémentation complète

~~~python
"""Split conformal et CQR : données synthétiques hétéroscédastiques.
Aucune recherche de modèle ni de risque alpha sur le test.
--plot pour les figures ; --xgboost ajoute une comparaison locale.
"""
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Un vecteur de scores finis non vide est requis.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement entre 0 et 1.")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        return float("inf")
    return float(np.sort(scores)[k - 1])

def interval_metrics(y, lower, upper):
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    if y.ndim != 1 or not len(y) or y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("Trois vecteurs alignés non vides sont requis.")
    if not np.isfinite(y).all() or np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("Cibles finies et bornes non NaN requises.")
    coverage = np.mean((lower <= y) & (y <= upper))
    # Un intervalle inversé représente un ensemble vide.
    width = np.maximum(upper - lower, 0)
    return {"coverage": float(coverage), "mean_width": float(width.mean())}

def run(plot=False, include_xgboost=False):
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, size=(1000, 4))
    sigma = 4000 + 9000 * np.abs(X[:, 0])
    y = 250000 + 30000 * X[:, 0] + 15000 * X[:, 1] ** 2 + rng.normal(size=len(X)) * sigma
    train_cal, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    train, cal = train_test_split(train_cal, test_size=.25, random_state=7)
    assert not (set(train) & set(cal) or set(cal) & set(test) or set(train) & set(test))
    models = {"Random Forest": RandomForestRegressor(n_estimators=60, min_samples_leaf=5,
                                                    random_state=42, n_jobs=1)}
    if include_xgboost:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=80, max_depth=3, random_state=42, n_jobs=1)
    rows = []
    plot_bounds = None
    for name, model in models.items():
        model.fit(X[train], y[train])
        scores = np.abs(y[cal] - model.predict(X[cal]))
        predicted = model.predict(X[test])
        for alpha in (.2, .1, .05):
            q = conformal_quantile(scores, alpha)
            lower, upper = predicted - q, predicted + q
            metrics = interval_metrics(y[test], lower, upper)
            row = {"model": name, "target": 1-alpha, "q": q, **metrics}
            for group, mask in [("faible_bruit", np.abs(X[test, 0]) < 1.5),
                                ("fort_bruit", np.abs(X[test, 0]) >= 1.5)]:
                row["coverage_"+group] = interval_metrics(y[test][mask], lower[mask], upper[mask])["coverage"]
            rows.append(row)
            if name == "Random Forest" and alpha == .1:
                plot_bounds = (lower, upper, predicted)
    # CQR : modèles quantiles ajustés exclusivement sur train.
    for alpha in (.2, .1, .05):
        low_model = GradientBoostingRegressor(loss="quantile", alpha=alpha/2,
            n_estimators=70, max_depth=2, random_state=42).fit(X[train], y[train])
        high_model = GradientBoostingRegressor(loss="quantile", alpha=1-alpha/2,
            n_estimators=70, max_depth=2, random_state=42).fit(X[train], y[train])
        def quantiles(indices):
            a, b = low_model.predict(X[indices]), high_model.predict(X[indices])
            return np.minimum(a, b), np.maximum(a, b)
        low_cal, high_cal = quantiles(cal)
        # Les scores CQR peuvent être négatifs : ne pas les tronquer.
        scores = np.maximum(low_cal - y[cal], y[cal] - high_cal)
        q = conformal_quantile(scores, alpha)
        low_test, high_test = quantiles(test)
        rows.append({"model": "CQR", "target": 1-alpha, "q": q,
                     **interval_metrics(y[test], low_test-q, high_test+q)})
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("Couverture empirique != garantie conditionnelle. Hypothèse : échangeabilité.")
    print("Comparer largeur, couverture et sous-populations ; aucune AUC ne mesure un intervalle.")
    if plot:
        import matplotlib.pyplot as plt
        low, high, predicted = plot_bounds
        order = np.argsort(predicted)[:60]
        plt.fill_between(np.arange(len(order)), low[order], high[order], alpha=.25, label="Split conformal 90 %")
        plt.plot(predicted[order], label="Prédiction")
        plt.scatter(np.arange(len(order)), y[test][order], s=10, label="Prix observé")
        plt.ylabel("Prix synthétique (DT)")
        plt.legend()
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

## Hypothèses et prolongements du lab

Sources primaires : https://arxiv.org/abs/2107.07511 et https://arxiv.org/abs/1905.03222.

Le lab utilise des prix synthétiques avec bruit hétéroscédastique. Il compare les risques prédéfinis 0,20, 0,10 et 0,05 ; aucun niveau n'est choisi à partir des scores test. CQR utilise des modèles de quantiles entraînés sur train et une correction calculée sur calibration. Les scores CQR peuvent être négatifs. Le modèle Random Forest est disponible dans l'atelier ; --xgboost ajoute la comparaison en local.

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du mini-projet

[[assessment]]