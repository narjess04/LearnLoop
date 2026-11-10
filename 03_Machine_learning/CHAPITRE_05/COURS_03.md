# Chapitre 5 — Cours 3 : Explainable AI (XAI) — SHAP et LIME



## 1. Introduction : pourquoi expliquer un modèle ML ?

Imaginons que nous avons entraîné un modèle XGBoost pour prédire si un client risque de ne pas rembourser son crédit.

Le modèle prédit :
Probabilité de défaut prédite

Risque élevé

Le modèle fournit une prédiction, mais ne nous explique pas directement pourquoi ce client obtient 78 % plutôt que 20 %.
C'est ici qu'intervient l'Explainable Artificial Intelligence (XAI), ou intelligence artificielle explicable.

La XAI regroupe des méthodes permettant de comprendre, d'interpréter et de communiquer le comportement des modèles d'intelligence artificielle.

Elle répond principalement à trois questions :
- Pourquoi ? Pourquoi le modèle a-t-il produit cette prédiction ?
- Comment ? Quelles variables influencent ses décisions ?
- Peut-on lui faire confiance ? Son comportement est-il cohérent, robuste et compatible avec les exigences métier ?
Attention : une explication XAI ne prouve pas, à elle seule, qu'un modèle est fiable, équitable ou causalement correct.

## 2. Interprétabilité vs explicabilité

Ces deux notions sont proches, mais elles ne sont pas exactement identiques.
Interprétabilité

Comprendre directement le fonctionnement interne du modèle.

Exemples : régression linéaire, arbre de décision simple.

Explicabilité

Utiliser une méthode pour expliquer le comportement d'un modèle complexe.

Exemples : SHAP, LIME, permutation importance.

Une régression linéaire est souvent considérée comme intrinsèquement interprétable : ses coefficients permettent de comprendre les relations modélisées, sous certaines conditions.

En revanche, un modèle comme XGBoost ou un réseau de neurones est généralement plus complexe. On utilise alors des méthodes d'explication post-hoc.

## 3. Explications globales et locales

Il existe deux grandes familles d'explications.
Explicabilité globale

Elle cherche à comprendre le comportement général du modèle sur un ensemble de données.

Variable	Importance globale fictive
Endettement	0,20
Revenu	0,15
Historique crédit	0,10
Ancienneté	0,05

Exemple fictif : importance moyenne absolue SHAP.

Ces valeurs mesurent l'ampleur moyenne des contributions, pas leur direction.

Explicabilité locale

Elle explique une prédiction individuelle.

Variable	Importance globale fictive
Endettement	0,20
Revenu	0,15
Historique crédit	0,10
Ancienneté	0,05

Exemple fictif : contributions locales exprimées en points de probabilité.

Valeur de référence : 50 %. Contributions : +28 points. Prédiction finale : 78 %.

L'explicabilité globale répond à « Comment fonctionne le modèle en général ? », tandis que l'explicabilité locale répond à « Pourquoi cette prédiction particulière ? ».

## 4. SHAP — SHapley Additive exPlanations

SHAP est l'une des méthodes les plus utilisées pour expliquer les modèles de Machine Learning.

Elle repose sur les valeurs de Shapley, issues de la théorie des jeux coopératifs.

## 4.1. Intuition

Imaginons une équipe de quatre joueurs qui obtient une récompense collective.

Nous voulons déterminer la contribution équitable de chaque joueur à cette récompense.

En Machine Learning :
- Les joueurs représentent les variables (features).
- La récompense représente la prédiction du modèle.
- Les valeurs de Shapley représentent les contributions attribuées aux variables.
SHAP applique cette logique pour expliquer une prédiction par rapport à une valeur de référence.

## 4.2. Formule mathématique

Pour une prédiction expliquée sur une échelle additive :
\[
f(x)=\phi_0+\sum_{i=1}^{M}\phi_i
\]

Avec :
- \(f(x)\) : sortie du modèle pour l'observation \(x\).
- \(\phi_0\) : valeur de référence (base value).
- \(\phi_i\) : contribution SHAP de la variable \(i\).
- \(M\) : nombre de variables.
La formule théorique d'une valeur de Shapley est :
\[
\phi_i=\sum_{S\subseteq F\setminus\{i\}}
\frac{|S|!(M-|S|-1)!}{M!}
[v(S\cup\{i\})-v(S)]
\]

Où \(S\) est un sous-ensemble de variables et \(v(S)\) mesure la valeur associée à ce sous-ensemble.

L'idée essentielle : SHAP calcule la contribution marginale moyenne d'une variable en considérant les différentes coalitions possibles.

## 4.3. Visualisation interactive de SHAP

[[shapadditive]]

Modifie les contributions pour observer leur influence sur la prédiction.

Variable	Importance globale fictive
Endettement	0,20
Revenu	0,15
Historique crédit	0,10
Ancienneté	0,05

Calcul : 50 + 25 + 18 − 10 − 5 = 78 %
Simulation pédagogique de contributions additives, et non calcul réel de valeurs SHAP.

## 4.4. Les principaux graphiques SHAP

Beeswarm plot

Visualise l'impact des variables sur l'ensemble des observations. Chaque point correspond à une observation, et sa couleur représente généralement la valeur de la variable.

Waterfall plot

Montre comment les contributions des variables font évoluer la prédiction depuis la valeur de référence jusqu'à la sortie du modèle.

Dependence plot

Étudie comment les valeurs d'une variable sont associées à ses contributions SHAP, et peut faire apparaître des interactions.

## 4.5. Implémentation Python avec XGBoost

Nous allons entraîner un modèle de classification sur un dataset synthétique, puis expliquer ses prédictions.

L'implémentation Python complète, remise en forme et exécutable, figure dans la section Coding Lab à la fin du support. Les exemples de cette section sont regroupés dans ce script pour conserver un protocole reproductible.

Interprétation : le beeswarm présente le comportement global du modèle, tandis que le waterfall permet de comprendre la contribution de chaque variable à une prédiction particulière.

Important : pour certains modèles de classification XGBoost, les valeurs retournées par TreeExplainer sont exprimées sur l'échelle des marges brutes, souvent les log-odds, et non directement en probabilités.

## 5. LIME — Local Interpretable Model-agnostic Explanations

LIME est une méthode d'explication locale, indépendante de l'architecture du modèle.

Contrairement à SHAP, LIME ne cherche pas à répartir une prédiction selon les valeurs de Shapley.

Il cherche à approcher localement le comportement d'un modèle complexe à l'aide d'un modèle simple, souvent une régression linéaire.

## 5.1. Principe de fonctionnement

Schéma conceptuel : LIME approxime le comportement du modèle dans le voisinage d'une observation.
Les étapes de LIME sont les suivantes :
1. Sélectionner une observation à expliquer.
2. Générer des observations artificielles autour de celle-ci.
3. Obtenir les prédictions du modèle complexe pour ces observations.
4. Pondérer les observations selon leur proximité avec celle étudiée.
5. Entraîner un modèle simple sur ces prédictions pondérées.
6. Utiliser les coefficients du modèle simple comme explication locale.

## 5.2. Formule mathématique

\[
\xi(x)=\underset{g\in G}{\arg\min}
\left[L(f,g,\pi_x)+\Omega(g)\right]
\]

Où :
- \(f\) : modèle complexe.
- \(g\) : modèle interprétable local.
- \(\pi_x\) : mesure de proximité autour de \(x\).
- \(L\) : erreur entre les prédictions de \(f\) et celles de \(g\).
- \(\Omega(g)\) : pénalité de complexité de l'explication.
L'objectif est de trouver une explication suffisamment simple tout en reproduisant fidèlement le comportement local du modèle.

## 5.3. Implémentation Python avec LIME

Nous réutilisons le modèle XGBoost entraîné précédemment.

L'implémentation Python complète, remise en forme et exécutable, figure dans la section Coding Lab à la fin du support. Les exemples de cette section sont regroupés dans ce script pour conserver un protocole reproductible.

LIME peut produire des explications différentes selon les perturbations générées, le voisinage choisi et la graine aléatoire. Il est donc important de vérifier la stabilité des explications.

## 6. SHAP vs LIME : quelles différences ?

Critère	SHAP	LIME
Fondement	Théorie des jeux	Approximation locale
Explication locale	Oui	Oui
Explication globale	Oui, par agrégation	Pas directement
Modèles compatibles	Large gamme	Modèle agnostique
Coût de calcul	Variable, parfois élevé	Variable selon les perturbations
Stabilité	Dépend de la méthode et des données	Sensible à l'échantillonnage
Principal avantage	Contributions additives cohérentes avec le cadre Shapley	Explications locales intuitives

Quand utiliser SHAP ? Lorsque tu veux analyser à la fois les contributions individuelles et le comportement global du modèle, particulièrement pour les modèles d'arbres avec TreeSHAP.

Quand utiliser LIME ? Lorsque tu veux obtenir une approximation locale simple d'un modèle quelconque et que tu peux vérifier sa fidélité.

Ces méthodes sont complémentaires plutôt que concurrentes.

## 7. Les limites importantes de la XAI

Un ingénieur ML avancé ne doit pas seulement savoir générer des graphiques SHAP ou LIME. Il doit aussi connaître leurs limites.

## 7.1. Corrélation entre variables

Lorsque deux variables sont fortement corrélées, l'attribution des contributions peut devenir difficile à interpréter.

Par exemple, le revenu mensuel et le niveau d'endettement peuvent contenir des informations liées. Selon la méthode utilisée, les contributions peuvent être réparties différemment.

## 7.2. Explication ≠ causalité

Si SHAP indique qu'une variable contribue fortement à une prédiction, cela ne signifie pas qu'une modification de cette variable provoquerait nécessairement une modification équivalente du résultat réel.

## 7.3. Dépendance au jeu de référence

Les valeurs SHAP peuvent varier selon la population de référence utilisée pour calculer les contributions.

## 7.4. Fidélité et stabilité

Une explication visuellement convaincante peut être instable ou mal représenter le modèle. Il faut donc vérifier :
- La stabilité des explications sous de petites perturbations.
- La fidélité locale, particulièrement pour LIME.
- La cohérence des explications entre différents échantillons.
- Les effets des variables corrélées et du choix de la population de référence.

## 8. Cas pratique : explicabilité d'un modèle de crédit

Supposons qu'une banque utilise XGBoost pour prédire la probabilité de défaut d'un client.

Elle souhaite comprendre pourquoi le modèle attribue une probabilité élevée à certains dossiers.

Avec SHAP, elle peut analyser les variables les plus influentes à l'échelle du portefeuille et les contributions pour chaque dossier.

Avec LIME, elle peut obtenir une approximation simple du comportement du modèle autour d'un dossier précis.

Cependant, ces explications doivent être accompagnées de contrôles de performance, de calibration, de stabilité et d'équité. Elles ne remplacent pas une validation complète du modèle.

## 9. Mini-projet pratique

Projet : Explainable Credit Risk Model

Objectif : entraîner un modèle de classification sur des données de crédit, puis comparer ses explications SHAP et LIME.

## 10. À retenir

- XAI permet de mieux comprendre le comportement des modèles de Machine Learning.
- SHAP attribue des contributions aux variables à partir de la théorie des jeux.
- LIME approxime localement un modèle complexe avec un modèle simple.
- Global vs local : l'un explique le comportement général, l'autre une prédiction individuelle.
- Les explications ne prouvent pas la causalité et doivent être évaluées pour leur fidélité et leur stabilité.



## Coding Lab — implémentation complète

~~~python
"""Explainable Credit Risk Model. Local Python: pip install numpy pandas scikit-learn matplotlib shap lime xgboost.
Données synthétiques, sans valeur opérationnelle. Utiliser --plot pour les figures.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer

def run(plot=False):
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5,
                              n_redundant=2, random_state=42)
    names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42)
    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.05,
                              eval_metric="logloss", random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("AUC test:", roc_auc_score(y_test, proba))
    print("F1 au seuil 0.5 fixé avant le test:", f1_score(y_test, proba >= .5))
    # Population de référence issue uniquement du train.
    background = X_train[np.random.default_rng(42).choice(len(X_train), 50, replace=False)]
    explainer = shap.TreeExplainer(model, data=background,
        feature_names=names, feature_perturbation="interventional",
        model_output="probability")
    values = explainer(X_test[:60])
    reconstructed = values.base_values + values.values.sum(axis=1)
    assert np.allclose(reconstructed, model.predict_proba(X_test[:60])[:, 1], atol=1e-5)
    importance = pd.Series(np.abs(values.values).mean(axis=0), index=names).sort_values(ascending=False)
    print("Importance globale (amplitude, pas direction):\n", importance)
    stability = []
    for seed in (7, 42, 99):
        local = LimeTabularExplainer(X_train, feature_names=names,
            class_names=["non-défaut", "défaut"], mode="classification",
            random_state=seed, discretize_continuous=True)
        for i in range(3):
            explanation = local.explain_instance(X_test[i], model.predict_proba,
                labels=(1,), num_features=6, num_samples=2000)
            stability.append({"client": i, "seed": seed,
                "probability": float(proba[i]), "local_prediction": float(explanation.local_pred[0]),
                "local_R2": float(explanation.score), "terms": explanation.as_list(label=1)})
            if plot and seed == 42:
                explanation.as_pyplot_figure(label=1)
    print(pd.DataFrame(stability).to_string(index=False))
    assert np.isfinite([row["local_prediction"] for row in stability]).all()
    if plot:
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(values, show=False)
        for i in range(3):
            shap.plots.waterfall(values[i], show=False)
        plt.show()
    return importance, stability

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    run(parser.parse_args().plot)
~~~

## Atelier navigateur et environnement local

L'atelier ci-dessous vérifie la reconstruction additive et l'importance moyenne absolue avec NumPy. C'est une simulation pédagogique sans calcul de valeurs SHAP. Le mini-projet complet utilise réellement XGBoost, SHAP et LIME en Python local.

Dépendances locales : numpy pandas scikit-learn matplotlib shap lime xgboost. Exécute ch05_cours03_lab.py --plot pour les visualisations réelles.

TreeExplainer utilise explicitement une référence train, feature_perturbation=interventional et model_output=probability. Le script vérifie la reconstruction contre predict_proba pour la classe 1. Les attributions dépendent de cette convention et ne sont pas des effets causaux.

Le score local_R2 de LIME mesure la fidélité du surrogate dans son voisinage pondéré ; ce n'est pas une performance globale.

Sources : https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html et https://lime-ml.readthedocs.io/en/latest/lime.html

## Vérifier ma compréhension

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan du mini-projet

[[assessment]]