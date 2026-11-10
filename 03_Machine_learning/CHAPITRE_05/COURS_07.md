# Chapitre 5 — Cours 7 : Model Diagnostics & Error Analysis



## 1. Introduction — Pourquoi analyser les erreurs d'un modèle ?

Jusqu'à présent, nous avons appris à entraîner des modèles, à évaluer leurs performances, à expliquer leurs prédictions et à mesurer leur robustesse.

Mais une question essentielle reste :

Lorsqu'un modèle commet des erreurs, comment comprendre précisément pourquoi il se trompe et comment l'améliorer ?

Imaginons un modèle XGBoost qui prédit le défaut de crédit.
Résultats fictifs sur le jeu de test
Accuracy
92 %

AUC-ROC
0,91

F1-score
0,84

Ces métriques indiquent que le modèle fonctionne globalement bien.

Mais elles ne répondent pas à plusieurs questions importantes :
- Quels types d'erreurs le modèle commet-il ?
- Sur quels profils de clients se trompe-t-il le plus ?
- Les erreurs proviennent-elles des données, du modèle ou du seuil de décision ?
- Existe-t-il des observations particulièrement difficiles ?
- Comment améliorer le modèle sans provoquer de nouvelles dégradations ?
C'est le rôle du Model Diagnostics & Error Analysis.
Définition
Le Model Diagnostics désigne l'ensemble des techniques permettant d'examiner le comportement d'un modèle afin d'identifier ses faiblesses, ses erreurs systématiques et les pistes d'amélioration.

L'Error Analysis se concentre plus particulièrement sur les observations mal prédites et les caractéristiques communes à ces erreurs.

L'objectif n'est pas simplement d'obtenir une meilleure métrique, mais de comprendre où, quand et pourquoi le modèle échoue.

## 2. Les principales catégories d'erreurs

Les erreurs ne sont pas toutes de même nature.

Erreurs liées aux données

Valeurs manquantes, labels incorrects, observations atypiques, variables mal encodées ou données non représentatives.

Erreurs liées au modèle

Sous-apprentissage, surapprentissage, complexité insuffisante ou mauvaise représentation des relations entre variables.

Erreurs liées à la décision

Seuil de classification inadapté, coûts d'erreurs mal définis ou mauvaise calibration des probabilités.

Erreurs liées à la population

Performances dégradées sur certains segments, données rares ou changements de distribution.

Il faut distinguer les symptômes observés des causes réelles. Par exemple, un taux d'erreur élevé sur un segment peut provenir d'un problème de données, mais aussi d'un seuil inadapté ou d'un manque d'information.

## 3. Diagnostic des erreurs en classification

## 3.1. La matrice de confusion

La matrice de confusion est l'un des premiers outils à examiner.

Pour un modèle de prédiction de défaut, nous définissons :
- Classe 1 : client en défaut.
- Classe 0 : client non défaillant.
Matrice de confusion — 1 000 clients

Prédit 0

Prédit 1

Réel 0

850
TN

50
FP

Réel 1

30
FN

70
TP

Exemple pédagogique : 900 non-défauts et 100 défauts réels.
À partir de cette matrice :
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}=92\%
\]

\[
Precision=\frac{TP}{TP+FP}\approx58.3\%
\]

\[
Recall=\frac{TP}{TP+FN}=70\%
\]

\[
F1=2\frac{Precision\times Recall}{Precision+Recall}\approx63.6\%
\]

Ce résultat montre pourquoi l'accuracy seule peut être trompeuse, notamment lorsque les classes sont déséquilibrées.

## 3.2. False Positives vs False Negatives

Dans notre exemple :

False Positive (FP) : le modèle prédit un défaut alors que le client ne fait pas défaut.

False Negative (FN) : le modèle prédit une absence de défaut alors que le client fait réellement défaut.

Les conséquences métier sont différentes.

Un faux positif peut conduire à une décision de crédit trop prudente, tandis qu'un faux négatif peut conduire à sous-estimer le risque.

La gravité relative dépend du processus de décision, des coûts et des obligations applicables.

## 4. Analyse du seuil de classification

[[diagnosticthreshold]]

Un modèle de classification produit souvent des probabilités :
\[
\hat{p}=P(Y=1\mid X)
\]

La décision dépend ensuite d'un seuil \(t\) :
\[
\hat{Y}=
\begin{cases}
1 & \text{si }\hat{p}\geq t\\
0 & \text{sinon}
\end{cases}
\]

Le seuil classique de 0,5 n'est pas toujours adapté.
Nous allons utiliser un petit ensemble fictif de 20 clients pour observer l'effet du seuil sur les erreurs.
Seuil de décision
50 %

TP
5

FP
3

FN
5

TN
7

Precision

62.5 %

Recall

50.0 %

F1-score

55.6 %

Les probabilités et labels sont fictifs. Abaisser le seuil augmente généralement le rappel, mais peut aussi augmenter les faux positifs.
Interprétation : il n'existe pas de seuil universellement optimal. Le seuil doit être choisi selon les coûts d'erreur, les contraintes métier et les performances mesurées sur un ensemble de validation indépendant.

Le jeu de test final ne doit pas être utilisé pour optimiser ce seuil.

## 5. Error Analysis par sous-population

Un modèle peut avoir de bonnes performances globales, mais être moins fiable sur certains segments.

Par exemple, nous pouvons comparer les performances selon :
- Les tranches de revenu.
- Les catégories de produits.
- Les régions.
- Les périodes.
- Les groupes de risque.
- Les caractéristiques pertinentes pour un audit d'équité.
Exemple de diagnostic

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Exemple fictif : taux d'erreur par segment de revenu.

Le modèle présente un taux d'erreur plus élevé sur le segment à faible revenu.

Cela peut indiquer un problème, mais nous ne devons pas conclure immédiatement que le revenu est la cause des erreurs.

Il faut examiner la taille des groupes, la fréquence des classes, la qualité des données et les caractéristiques des observations.
Implémentation Python

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Ce code suppose que X_test est un DataFrame contenant une colonne income et que les prédictions sont alignées avec ses lignes.

Pour une analyse rigoureuse, il faut aussi examiner les effectifs, les intervalles d'incertitude et les différences de taux de base.

## 6. Residual Analysis pour la régression

En régression, nous ne disposons pas d'une matrice de confusion.

Nous analysons les résidus.

Un résidu est défini par :
\[
e_i=y_i-\hat{y}_i
\]

Avec :
- \(y_i\) : valeur réelle.
- \(\hat{y}_i\) : valeur prédite.
- \(e_i\) : erreur de prédiction.

## 6.1. Pourquoi analyser les résidus ?

Un modèle correctement spécifié devrait idéalement présenter des résidus sans structure systématique inexpliquée.

Mais plusieurs comportements peuvent révéler des problèmes.
Cas A — Résidus sans structure apparente

Les résidus oscillent autour de zéro sans tendance évidente.

Cas B — Hétéroscédasticité

La dispersion des erreurs augmente avec les valeurs prédites. Le modèle présente une variance résiduelle non constante.

Cas C — Structure non linéaire

Une structure courbe peut indiquer que le modèle n'a pas correctement représenté certaines relations non linéaires.

## 6.2. Implémentation Python

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Il est également utile d'analyser les résidus en fonction de chaque variable importante, du temps et des sous-populations.

## 7. Identifier les observations les plus difficiles

Une approche très utile consiste à classer les observations selon leur erreur.

En régression :
\[
Error_i=|y_i-\hat{y}_i|
\]

En classification probabiliste, on peut examiner la log loss individuelle :
\[
L_i=-\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
\]

Une observation ayant une perte élevée est particulièrement mal prédite.
Exemple Python

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

Après avoir identifié ces observations, nous pouvons examiner leurs caractéristiques, vérifier les labels et chercher des régularités.

Attention : une observation difficile n'est pas nécessairement mal étiquetée. Elle peut simplement être rare, ambiguë ou intrinsèquement difficile à prédire.

## 8. Error Analysis avec SHAP

Dans le cours 3, nous avons étudié SHAP pour expliquer les prédictions.

Nous pouvons maintenant l'utiliser pour comprendre les erreurs.
Méthodologie

## 1. Identifier les observations mal classées.

## 2. Séparer les faux positifs et les faux négatifs.

## 3. Calculer les contributions SHAP sur ces observations.

## 4. Comparer leurs caractéristiques aux observations correctement classées.

[[diagnosticthreshold]]

## 5. Formuler des hypothèses sur les causes possibles des erreurs.

## 6. Vérifier ces hypothèses avec des analyses complémentaires.

Par exemple, si les faux négatifs sont souvent associés à un faible niveau d'endettement, nous pouvons examiner si le modèle accorde trop d'importance à cette variable dans certaines situations.

Cependant, SHAP explique la prédiction du modèle, pas nécessairement la cause de son erreur.

## 9. Learning Curves : détecter Underfitting et Overfitting

Les courbes d'apprentissage permettent de comparer les performances d'entraînement et de validation en fonction du nombre d'observations utilisées.
Exemple de Learning Curve

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Les graphiques du support utilisent des données fictives. Les ateliers et scripts produisent leurs propres résultats calculés.

Données fictives : scores en fonction du nombre d'observations d'entraînement.

Trois situations fréquentes :
Situation	Diagnostic
Train élevé, validation faible	Surapprentissage possible
Train faible, validation faible	Sous-apprentissage possible
Train et validation élevés et proches	Bonne généralisation sur la distribution évaluée

Une courbe d'apprentissage ne prouve pas à elle seule la cause du problème, mais elle oriente les investigations.
Implémentation

L'implémentation complète et exécutable est regroupée dans la section Coding Lab, avec les retours à la ligne et les calculs manquants du collage.

## 10. Data-Centric AI : améliorer les données avant le modèle

Une erreur fréquente en Machine Learning consiste à chercher immédiatement un algorithme plus complexe lorsqu'un modèle ne fonctionne pas correctement.

Pourtant, les problèmes peuvent provenir des données.

L'approche Data-Centric AI consiste à améliorer systématiquement la qualité, la cohérence et la pertinence des données utilisées.

Exemples d'actions :
- Corriger les labels erronés après vérification.
- Améliorer la représentativité des données.
- Identifier les observations dupliquées.
- Ajouter des données pertinentes pour les segments mal représentés.
- Revoir la définition des variables et de la cible.
- Corriger les incohérences entre les pipelines d'entraînement et de production.
Il est important de ne pas supprimer automatiquement les observations difficiles ou atypiques : elles peuvent représenter des cas métier importants.

## 11. Construire un workflow professionnel de diagnostic

Voici une démarche structurée.

## 1. Évaluation globale

Calculer les métriques principales sur un jeu de test indépendant.

## 2. Analyse des erreurs

Étudier les faux positifs, faux négatifs ou résidus.

## 3. Analyse par segment

Identifier les populations présentant des performances insuffisantes.

## 4. Analyse des observations difficiles

[[diagnosticthreshold]]

Examiner les exemples avec les pertes les plus élevées.

## 5. Diagnostic explicatif

Utiliser SHAP, les learning curves et les analyses statistiques.

## 6. Formulation d'hypothèses

Identifier les causes possibles et proposer des corrections.

## 7. Expérimentation contrôlée

Comparer les modifications sur un ensemble de validation, puis confirmer les résultats sur un test final indépendant.

Une bonne pratique consiste à conserver un historique des erreurs identifiées, des hypothèses formulées et des corrections expérimentées.

Cela facilite la reproductibilité et évite de répéter les mêmes investigations.

## 12. Mini-projet pratique — ML Error Analysis Dashboard

Projet : Automated ML Diagnostics

Objectif : développer un tableau de bord permettant d'analyser automatiquement les erreurs d'un modèle de classification.

Technologies possibles : Python, scikit-learn, XGBoost, SHAP, pandas, Plotly et Streamlit.

## 13. À retenir

- Les métriques globales ne suffisent pas à comprendre le comportement d'un modèle.
- La matrice de confusion permet d'identifier les différents types d'erreurs en classification.
- L'analyse des résidus est essentielle pour diagnostiquer les modèles de régression.
- Les performances doivent être étudiées par segment et sur les observations difficiles.
- Les learning curves aident à identifier les problèmes de généralisation.
- SHAP peut aider à expliquer les prédictions erronées, mais ne révèle pas automatiquement leurs causes.
- Une amélioration fiable doit être validée par des expériences contrôlées.

## 14. Ressources pour approfondir

- Scikit-learn — Learning Curves
- Scikit-learn — Model Evaluation
- SHAP — Documentation officielle

Nous étudierons comment détecter les changements de distribution, surveiller les performances des modèles en production et mettre en place un système de monitoring fiable.

## Coding Lab — implémentation complète

~~~python
"""Diagnostics sur validation, puis audit final indépendant. --plot et --curves en option.
Les segments et les données sont synthétiques, sans interprétation causale.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def individual_log_loss(y_true, proba):
    y, p = np.asarray(y_true), np.asarray(proba, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Cibles binaires et probabilités alignées non vides requises.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilités finies dans [0,1] requises.")
    p = np.clip(p, 1e-15, 1-1e-15)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

def segment_report(y_true, proba, segments, threshold=.5):
    losses = individual_log_loss(y_true, proba)
    y, p, segments = np.asarray(y_true), np.asarray(proba), np.asarray(segments)
    if segments.shape != y.shape or pd.isna(segments).any() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Segments alignés non manquants et seuil dans [0,1] requis.")
    pred = (p >= threshold).astype(int)
    frame = pd.DataFrame({"segment": segments, "truth": y, "prediction": pred, "loss": losses})
    rows = []
    for name, group in frame.groupby("segment", observed=True):
        truth, predicted = group["truth"].to_numpy(), group["prediction"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        rows.append({"segment": name, "n": len(group), "prevalence": truth.mean(),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "error_rate": float((truth != predicted).mean()), "log_loss": group["loss"].mean(),
            "precision": tp/(tp+fp) if tp+fp else np.nan,
            "recall": tp/(tp+fn) if tp+fn else np.nan})
    return pd.DataFrame(rows).set_index("segment")

def run(plot=False, curves=False):
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    dev, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    train, valid = train_test_split(dev, test_size=.25, stratify=y[dev], random_state=7)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
    p = model.predict_proba(X[valid])[:, 1]
    # Bornes apprises sur train. Les doublons sont retirés sans labels qcut incompatibles.
    edges = np.unique(np.quantile(X[train, 0], [1/3, 2/3]))
    segments = pd.cut(X[valid, 0], bins=np.r_[-np.inf, edges, np.inf]).astype(str)
    report = segment_report(y[valid], p, segments)
    print("Diagnostic sur VALIDATION :\n", report.to_string())
    hard = pd.DataFrame({"row_id": valid, "y_true": y[valid], "probability": p,
                         "log_loss": individual_log_loss(y[valid], p)}).sort_values("log_loss", ascending=False).head(10)
    print("Cas difficiles à investiguer, sans suppression automatique :\n", hard.to_string(index=False))
    if curves:
        sizes, train_scores, valid_scores = learning_curve(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X[train], y[train],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            train_sizes=[.3, .6, 1.], scoring="roc_auc", n_jobs=1)
        print(pd.DataFrame({"n": sizes, "train_auc": train_scores.mean(axis=1),
                            "cv_auc": valid_scores.mean(axis=1)}).to_string(index=False))
    # L'audit test n'est pas réutilisé pour choisir seuil ou features.
    test_p = model.predict_proba(X[test])[:, 1]
    print("Audit TEST : AUC =", roc_auc_score(y[test], test_p),
          "accuracy =", accuracy_score(y[test], test_p >= .5))
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(individual_log_loss(y[valid], p), bins=20)
        plt.xlabel("Log loss individuelle sur validation")
        plt.show()
        Xr, yr = make_regression(n_samples=300, n_features=3, noise=20, random_state=42)
        a, b = train_test_split(np.arange(len(yr)), test_size=.3, random_state=42)
        predicted = LinearRegression().fit(Xr[a], yr[a]).predict(Xr[b])
        plt.scatter(predicted, yr[b]-predicted, s=10)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Prédiction"); plt.ylabel("Résidu y − prédiction")
        plt.show()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--curves", action="store_true")
    args = parser.parse_args()
    run(args.plot, args.curves)

~~~

## Hypothèses et limites du lab

Les diagnostics qui conduisent à modifier le modèle sont réalisés sur validation. Si tu explores le test pour choisir des corrections, il devient un jeu de développement : réserve alors un nouveau test final. Le lab fixe les segments à partir du train et retire les bornes dupliquées sans imposer trois labels incompatibles. --curves calcule les courbes ; --plot affiche les pertes et les résidus. Les explications SHAP du projet utilisent l'environnement local du cours 3.

## Quiz formatif

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du projet

[[assessment]]