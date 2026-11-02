# Gradient Boosting, XGBoost, LightGBM & CatBoost

Dans le cours précédent, nous avons étudié Random Forest, une méthode d'Ensemble Learning qui entraîne plusieurs arbres de manière essentiellement indépendante, puis agrège leurs prédictions.

Aujourd'hui, nous découvrons une approche différente : le Boosting.

Son principe est de construire un modèle progressivement. Chaque nouvel arbre cherche à améliorer les prédictions de l'ensemble déjà construit.

C'est l'une des familles d'algorithmes les plus importantes pour les données tabulaires, notamment en finance, assurance, scoring de crédit et prévision.

## 1. Bagging versus Boosting

Bagging — Random Forest

Arbres entraînés indépendamment, puis combinés. Objectif principal : réduire la variance.

Boosting — Gradient Boosting

Arbres ajoutés successivement. Objectif : optimiser progressivement une fonction de perte.

Critère	Random Forest	Gradient Boosting
Construction	Arbres essentiellement indépendants	Arbres successifs
Principe	Agrégation	Correction progressive
Diversité	Bootstrap et features aléatoires	Contributions successives
Paramètre clé	max_features	learning_rate
Surapprentissage	Souvent assez robuste	Sensible à la complexité et au nombre d'itérations
Early stopping	Moins central	Très important

## Partie I — Comprendre le Gradient Boosting

## 2. L'intuition : apprendre de ses erreurs

Supposons que nous voulions prédire le montant d'une perte financière pour cinq observations.

Observation	Valeur réelle
A	100
B	150
C	200
D	250
E	300

Nous commençons par un modèle très simple qui prédit toujours la moyenne :\[
F_0(x)=\bar y=200
\]

Les erreurs sont :\[
r_i=y_i-F_0(x_i)
\]

Observation	Réel	Prédiction	Résidu
A	100	200	−100
B	150	200	−50
C	200	200	0
D	250	200	+50
E	300	200	+100

Le premier arbre va chercher à prédire ces résidus.

L'ensemble est ensuite mis à jour :\[
\boxed{F_1(x)=F_0(x)+\eta h_1(x)}
\]

Où :

- \(F_0\) est le modèle initial.
- \(h_1\) est le premier arbre correcteur.
- \(\eta\) est le learning rate.

Puis nous recommençons :\[
F_2(x)=F_1(x)+\eta h_2(x)
\]

Jusqu'à :\[
\boxed{
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}h_m(x)
}
\]

C'est la forme additive du Gradient Boosting.

## 3. Simulateur interactif — Correction des erreurs

[[boostsim]]

## 4. Pourquoi parle-t-on de gradient ?

Le Gradient Boosting ne se limite pas à apprendre les résidus \(y-\hat y\).

Il généralise cette idée à différentes fonctions de perte.

Nous cherchons à minimiser :\[
\mathcal L(F)=\sum_{i=1}^{n}L(y_i,F(x_i))
\]

À l'itération \(m\), nous calculons les pseudo-résidus :\[
\boxed{
r_{im}=
-\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
}
\]

Ce sont les gradients négatifs de la perte par rapport aux prédictions courantes.

Ensuite :

1. Nous calculons les pseudo-résidus.
2. Nous entraînons un arbre pour les approximer.
3. Nous calculons une correction.
4. Nous mettons à jour l'ensemble.

Une formulation générale est :\[
F_m(x)=F_{m-1}(x)+\eta\,\gamma_mh_m(x)
\]

Le coefficient \(\gamma_m\) peut être obtenu par une recherche de pas. Dans les implémentations à arbres, les valeurs de correction peuvent aussi être optimisées séparément pour chaque feuille.

## 5. Cas de la régression avec MSE

Considérons :\[
L(y,F)=\frac12(y-F)^2
\]

Sa dérivée par rapport à \(F\) vaut :\[
\frac{\partial L}{\partial F}=F-y
\]

Le gradient négatif est :\[
-\frac{\partial L}{\partial F}=y-F
\]

Donc :\[
\boxed{r_i=y_i-\hat y_i}
\]

Voilà pourquoi, avec la perte quadratique, le Gradient Boosting apprend les résidus.

## 6. Cas de la classification binaire

Pour la classification, nous utilisons souvent la log-loss.

Soit un score \(F(x)\), transformé en probabilité par la sigmoïde :\[
p=\sigma(F)=\frac1{1+e^{-F}}
\]

La perte est :\[
L(y,F)=-y\log(p)-(1-y)\log(1-p)
\]

Sa dérivée par rapport au score \(F\) est :\[
\frac{\partial L}{\partial F}=p-y
\]

Le gradient négatif vaut donc :\[
\boxed{r=y-p}
\]

Attention : en classification, les arbres correcteurs agissent sur le score brut, souvent les log-odds, et pas simplement en additionnant des probabilités.

C'est une distinction importante avec l'exemple de régression.

## Partie III — Learning Rate et nombre d'arbres

## 7. Le rôle du Learning Rate

Le learning rate contrôle l'amplitude de la contribution de chaque nouvel arbre.\[
F_m=F_{m-1}+\eta h_m
\]

Avec :\[
0<\eta\le1
\]

dans les configurations usuelles.
Learning rate élevé

- Corrections plus importantes.
- Moins d'arbres peuvent suffire.
- Risque d'optimisation moins stable ou de surapprentissage.

Learning rate faible

- Corrections progressives.
- Nécessite souvent davantage d'arbres.
- Peut améliorer la généralisation lorsque les autres paramètres sont adaptés.
- Augmente le coût d'entraînement.

## 8. Le compromis learning_rate / n_estimators

Learning rate	Nombre d'arbres	Interprétation
0,3	100	Apprentissage relativement rapide
0,1	300	Configuration courante à tester
0,05	600	Corrections plus progressives
0,01	1500	Beaucoup de petites corrections

Ces combinaisons sont illustratives, pas des configurations universellement optimales.

[[boostcurves]]

La perte d’entraînement peut continuer à diminuer alors que la validation se dégrade.

## 9. Qu'est-ce que XGBoost ?

XGBoost signifie Extreme Gradient Boosting.

C'est une bibliothèque de Gradient Boosting sur arbres, particulièrement connue pour ses performances sur les données tabulaires.

Elle introduit notamment :

- Une fonction objectif régularisée.
- Une optimisation exploitant les gradients et les dérivées secondes.
- Des méthodes efficaces de recherche de splits.
- La gestion des valeurs manquantes.
- La parallélisation de plusieurs opérations.
- Des mécanismes de contrôle de la complexité.

## 10. La fonction objectif de XGBoost

Une formulation classique est :\[
\boxed{
Obj=\sum_{i=1}^{n}L(y_i,\hat y_i)
+\sum_{k=1}^{K}\Omega(f_k)
}
\]

Où :

- \(L\) mesure l'erreur de prédiction.
- \(f_k\) est un arbre de l'ensemble.
- \(\Omega\) pénalise la complexité des arbres.

Une forme de régularisation est :\[
\Omega(f)=\gamma T+\frac12\lambda\sum_{j=1}^{T}w_j^2
\]

Avec :

- \(T\) : nombre de feuilles.
- \(w_j\) : valeur de la feuille \(j\).
- \(\gamma\) : pénalité pour l'ajout de feuilles.
- \(\lambda\) : régularisation L2 sur les valeurs des feuilles.

XGBoost propose aussi une régularisation L1.
Pourquoi est-ce important ?
Parce que nous n'optimisons pas uniquement l'erreur d'entraînement.

Nous contrôlons également la complexité du modèle.

## 11. XGBoost et dérivées secondes

XGBoost utilise une approximation de Taylor d'ordre deux de la perte.

À l'itération \(t\) :\[
L(y_i,\hat y_i^{(t-1)}+f_t(x_i))
\]

est approximée à partir du gradient :\[
g_i=\frac{\partial L}{\partial \hat y_i}
\]

et du Hessien :\[
h_i=\frac{\partial^2 L}{\partial \hat y_i^2}
\]

Ce qui donne, en omettant les termes constants :\[
\boxed{
Obj^{(t)}\approx
\sum_i\left[
g_if_t(x_i)+\frac12h_if_t(x_i)^2
\right]+\Omega(f_t)
}
\]

Cette approximation facilite l'optimisation des valeurs de feuilles et l'évaluation des splits.

## 12. Exemple XGBoost

~~~python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05,
    subsample=.8, colsample_bytree=.8, objective="binary:logistic",
    tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=1)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC", roc_auc_score(y_test, y_proba))
print("AP", average_precision_score(y_test, y_proba))
~~~

Le code suppose que X_train, X_test, y_train et y_test ont déjà été définis.

## Partie V — LightGBM

## 13. Qu'est-ce que LightGBM ?

LightGBM est une bibliothèque de Gradient Boosting développée pour être efficace sur des datasets tabulaires, notamment lorsqu'ils contiennent beaucoup d'observations ou de variables.

Elle utilise notamment des méthodes basées sur les histogrammes pour accélérer la recherche des splits.

## 14. Histogram-Based Learning

Au lieu de considérer chaque valeur numérique distincte comme seuil potentiel, LightGBM regroupe les valeurs en bins.

Par exemple :\[
[0.11,0.13,0.18,0.23,0.29,0.35]
\]

peuvent être regroupées dans des intervalles.

Les splits sont alors recherchés sur ces bins.
Avantages

- Moins de seuils à examiner.
- Calculs souvent plus rapides.
- Consommation mémoire potentiellement réduite.

La précision des seuils est approximée par la discrétisation.

## 15. Leaf-wise versus Level-wise

C'est l'une des différences les plus importantes à connaître.
Level-wise

Croissance par niveaux, illustrée par un arbre équilibré.

Leaf-wise

Croissance ciblée sur la feuille offrant le meilleur gain.

LightGBM utilise une stratégie leaf-wise, c'est-à-dire qu'il privilégie la feuille présentant le gain de split le plus élevé.

Cela peut permettre de réduire rapidement la perte, mais peut aussi produire des arbres profonds sur certaines branches.

D'où l'importance de contrôler num_leaves, max_depth et min_child_samples.

[[boostgrowth]]

## 16. Exemple LightGBM

~~~python
from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31,
    max_depth=-1, min_child_samples=20, subsample=.8, subsample_freq=1,
    colsample_bytree=.8, random_state=42, verbosity=-1, n_jobs=1)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
~~~

Attention : dans LightGBM, définir subsample=0.8 ne suffit pas toujours à activer le bagging. Le paramètre subsample_freq doit aussi être configuré, comme dans cet exemple.

## Partie VI — CatBoost

## 17. Qu'est-ce que CatBoost ?

CatBoost est une bibliothèque de Gradient Boosting particulièrement intéressante pour les datasets contenant des variables catégorielles.

Elle propose notamment :

- Une prise en charge native des variables catégorielles.
- Des méthodes d'encodage basées sur des statistiques ordonnées.
- L'Ordered Boosting, destiné à réduire certains biais d'entraînement.
- Des arbres symétriques, appelés oblivious trees, dans sa configuration classique.

## 18. Pourquoi les variables catégorielles sont-elles difficiles ?

Supposons une variable :
SECTEUR = Industrie, Commerce, Services, Agriculture

Une approche consiste à utiliser One-Hot Encoding.

Mais lorsque nous avons des centaines de catégories, le nombre de variables peut augmenter considérablement.

CatBoost peut traiter les catégories directement.
Ordered Target Statistics
L'idée consiste à calculer des statistiques liées à la cible sans utiliser directement la cible de l'observation encodée.

Des permutations ordonnées sont utilisées pour limiter les fuites d'information liées à l'encodage.

Il ne s'agit pas d'un simple Target Encoding calculé sur tout le dataset.

## 19. Exemple CatBoost

~~~python
import pandas as pd
from catboost import CatBoostClassifier
X_demo = pd.DataFrame({
    "revenu": [3000, 4500, 2200, 7000, 5100, 1800],
    "endettement": [.3, .5, .8, .2, .4, .9],
    "secteur": ["Industrie", "Services", "Commerce", "Industrie", "Services", "Commerce"]})
y_demo = [0, 0, 1, 0, 0, 1]
cat = CatBoostClassifier(iterations=100, depth=3, learning_rate=.05,
    loss_function="Logloss", verbose=False, random_seed=42,
    allow_writing_files=False, thread_count=1)
cat.fit(X_demo, y_demo, cat_features=["secteur"])
print(cat.predict_proba(X_demo)[:, 1])
~~~

Cet exemple est volontairement minuscule et sert uniquement à montrer la syntaxe. Les prédictions sur ces mêmes données ne constituent pas une évaluation fiable du modèle.

## Partie VII — Comparaison des quatre approches

## 20. Gradient Boosting, XGBoost, LightGBM et CatBoost

Critère	Gradient Boosting sklearn	XGBoost	LightGBM	CatBoost
Type	Boosting d'arbres	Boosting régularisé	Boosting efficace sur histogrammes	Boosting avec gestion catégorielle avancée
Données tabulaires	Oui	Oui	Oui	Oui
Variables catégorielles natives	Pas dans GradientBoostingClassifier classique	Support selon configuration	Oui	Oui
Early stopping	Oui	Oui	Oui	Oui
Gestion des NaN	Dépend de l'estimateur sklearn	Oui	Oui	Oui pour les numériques
Particularité	Simplicité	Régularisation et optimisation	Croissance leaf-wise	Ordered Boosting

Quel modèle choisir ?

Pour comprendre le Boosting

Commencer par GradientBoostingClassifier de scikit-learn.

Pour contrôler finement la régularisation

Tester XGBoost, notamment pour les tâches de classification et de régression tabulaires.

Pour de grands datasets tabulaires

LightGBM est un candidat intéressant à benchmarker.

Pour de nombreuses variables catégorielles

CatBoost est souvent un excellent candidat.

Il n'existe pas de gagnant universel. Le meilleur choix dépend du dataset, de la métrique, des ressources disponibles et des contraintes métier.

## Partie VIII — Early Stopping

## 21. Pourquoi l'Early Stopping est-il essentiel ?

Lorsque nous ajoutons des arbres, la perte d'entraînement peut continuer à diminuer.

Mais la performance sur de nouvelles données peut commencer à se dégrader.

L'Early Stopping surveille une métrique sur un ensemble de validation et arrête l'apprentissage lorsque les améliorations deviennent insuffisantes pendant un certain nombre d'itérations.
Exemple XGBoost

~~~python
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
    test_size=.2, stratify=y_train, random_state=42)
model = XGBClassifier(n_estimators=2000, learning_rate=.03, max_depth=4,
    tree_method="hist", eval_metric="logloss", early_stopping_rounds=50,
    random_state=42, n_jobs=1)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print("Best iteration (index zéro)", model.best_iteration)
~~~

Le dataset de test final ne doit pas servir à l'Early Stopping.

## Partie IX — Régularisation

## 22. Comment limiter l'overfitting ?

Les principaux leviers sont :

Hyperparamètre	Effet
learning_rate	Réduit la contribution de chaque arbre
n_estimators	Contrôle le nombre d'itérations
max_depth	Limite la profondeur
min_child_weight (XGBoost)	Contraint les feuilles selon les Hessiennes
reg_lambda (XGBoost)	Régularisation L2
reg_alpha (XGBoost)	Régularisation L1
subsample	Échantillonnage des observations
colsample_bytree	Échantillonnage des features
num_leaves (LightGBM)	Contrôle le nombre de feuilles
min_child_samples (LightGBM)	Limite les petites feuilles

Une bonne stratégie
Commencer par un modèle relativement simple, puis augmenter sa complexité seulement si les résultats de validation le justifient.

Ne pas chercher uniquement la meilleure performance d'entraînement.

## Partie X — Application au risque de crédit

## 23. Prédiction de Probability of Default

Supposons que nous disposions d'un dataset contenant :

- Ratio de solvabilité.
- Capacité de remboursement.
- Endettement.
- Rentabilité financière.
- Secteur d'activité.
- Historique de défaut.
- Variable cible Y : défaut ou non-défaut.

Nous voulons estimer :\[
PD(x)=P(Y=1\mid X=x)
\]

Un modèle de boosting peut apprendre des interactions non linéaires entre les variables.

Par exemple, l'effet de l'endettement peut dépendre de la rentabilité et de la capacité de remboursement.
Attention : discrimination versus calibration
Un modèle peut avoir une excellente ROC-AUC et produire des probabilités mal calibrées.

Pour un modèle de PD, il faut examiner :\[
\text{ROC-AUC},\quad \text{Gini},\quad \text{Brier Score}
\]

ainsi que les courbes de calibration et la stabilité temporelle.

Le coefficient de Gini associé à la ROC-AUC est :\[
\boxed{Gini=2AUC-1}
\]

Ce Gini de discrimination est différent de l'impureté de Gini utilisée dans les arbres de décision.

## 24. Calibration des probabilités

~~~python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
y_proba = model.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
print("Brier Score", brier_score_loss(y_test, y_proba))
~~~

Si une recalibration est nécessaire, elle doit être ajustée sur des données distinctes de celles utilisées pour l'évaluation finale.

En contexte de crédit, la définition de la cible, l'horizon de défaut et la cohérence temporelle sont également essentiels.

## Partie XI — Coding Lab complet

## 25. Comparer Random Forest et XGBoost

Benchmark des cinq familles sur le même jeu synthétique (5 000 observations, 20 features, environ 8 % de positifs). Sélection par ROC-AUC en validation croisée sur train avant l’audit test. Les temps mesurés dépendent de la machine. Installer xgboost, lightgbm et catboost dans l’environnement Python.

~~~python
"""Cours 08. Installer xgboost lightgbm catboost ; --plot pour les figures."""

import argparse
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)

def dataset(n=5000, features=20, informative=10, redundant=5, positive=.08):
    X, y = make_classification(n_samples=n, n_features=features,
        n_informative=informative, n_redundant=redundant, weights=[1-positive, positive],
        flip_y=.01, random_state=42)
    return train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def benchmark(models, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
            scoring={"ROC_AUC": "roc_auc", "AP": "average_precision"})
        rows.append({"Model": name, "ROC_AUC_CV": scores["test_ROC_AUC"].mean(),
            "ROC_AUC_std": scores["test_ROC_AUC"].std(), "AP_CV": scores["test_AP"].mean(),
            "fit_seconds_CV_mean": scores["fit_time"].mean()})
    table = pd.DataFrame(rows).sort_values("ROC_AUC_CV", ascending=False)
    print(table.to_string(index=False))
    chosen = table.iloc[0]["Model"]
    print("Choix figé sur la ROC-AUC CV avant de consulter le test :", chosen)
    return chosen, table

def metrics(model, X, y, threshold=None):
    probabilistic = hasattr(model, "predict_proba")
    score = model.predict_proba(X)[:, 1] if probabilistic else model.decision_function(X)
    pred = model.predict(X) if threshold is None else (score >= threshold).astype(int)
    assert np.isfinite(score).all()
    result = {"ROC_AUC": roc_auc_score(y, score), "AP": average_precision_score(y, score),
        "Gini": 2*roc_auc_score(y, score)-1,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if probabilistic:
        assert ((score >= 0) & (score <= 1)).all()
        result.update(Brier=brier_score_loss(y, score), LogLoss=log_loss(y, score))
    else:
        result.update(Brier=np.nan, LogLoss=np.nan)
    return result

def audit(models, X_train, y_train, X_test, y_test):
    rows=[]
    for name, model in models.items():
        start=perf_counter(); model.fit(X_train, y_train); duration=perf_counter()-start
        rows.append({"Model": name, **metrics(model, X_test, y_test), "fit_seconds": duration})
    table=pd.DataFrame(rows)
    print("Audit final des candidats préspécifiés ; aucune nouvelle sélection à partir du test.")
    print(table.to_string(index=False))
    return table

def figures(models, X, y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    for display, title in [(RocCurveDisplay, "ROC"), (PrecisionRecallDisplay, "Precision-Recall")]:
        fig, ax=plt.subplots()
        for name, model in models.items(): display.from_estimator(model, X, y, ax=ax, name=name)
        ax.set_title(title); fig.tight_layout()
    fig, ax=plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            CalibrationDisplay.from_estimator(model, X, y, n_bins=8, strategy="quantile", ax=ax, name=name)
    fig.tight_layout(); plt.show()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=.05, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=.05, max_depth=4, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbosity=-1, random_state=42, n_jobs=1),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=.05, depth=5, verbose=False, random_seed=42, thread_count=1, allow_writing_files=False),
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--plot", action="store_true"); args=parser.parse_args()
    X_train, X_test, y_train, y_test=dataset()
    candidates=models(); chosen, table=benchmark(candidates, X_train, y_train)
    audit(candidates, X_train, y_train, X_test, y_test)
    if args.plot: figures(candidates, X_test, y_test)
~~~

### Questions à analyser

1. Quel modèle obtient la meilleure ROC-AUC ?
2. Le classement est-il identique avec Average Precision ?
3. Quel modèle produit le meilleur Brier Score ?
4. Le modèle ayant la meilleure discrimination possède-t-il aussi la meilleure calibration ?
5. Que se passe-t-il si nous augmentons fortement n_estimators ?
6. Quel est l'effet d'une diminution du learning rate ?

## Partie XII — Hyperparameter Tuning

## 26. Optimiser XGBoost avec RandomizedSearchCV

~~~python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
model = XGBClassifier(objective="binary:logistic", tree_method="hist",
    eval_metric="logloss", random_state=42, n_jobs=1)
param_distributions = {
    "n_estimators": [100, 200, 400, 600], "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [.01, .03, .05, .1], "subsample": [.7, .8, 1.],
    "colsample_bytree": [.7, .8, 1.], "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
search = RandomizedSearchCV(model, param_distributions, n_iter=30,
    scoring="roc_auc", cv=cv, n_jobs=1, random_state=42, error_score="raise")
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
~~~

Dans cette configuration, nous n'utilisons pas l'Early Stopping à l'intérieur de la recherche.

Si nous voulons combiner recherche d'hyperparamètres et Early Stopping, il faut veiller à ce que chaque fold possède son propre ensemble de validation interne, sans utiliser les observations du fold évalué pour arrêter l'entraînement.

## Partie XIII — Coding Challenge

## 27. Challenge : Boosting Benchmark

Challenge 08 — Boosting Models

+700 XP

Construis un benchmark entre Random Forest, Gradient Boosting, XGBoost, LightGBM et CatBoost. Compare leurs performances, leur calibration et leur temps d'entraînement.

### Questions de réflexion

1. Pourquoi Gradient Boosting apprend-il des pseudo-résidus ?
2. Quelle différence entre résidus et gradients négatifs ?
3. Pourquoi le learning rate et le nombre d'arbres sont-ils liés ?
4. Pourquoi LightGBM peut-il être rapide ?
5. Pourquoi CatBoost est-il intéressant pour les variables catégorielles ?
6. Pourquoi XGBoost utilise-t-il une régularisation explicite ?
7. Pourquoi faut-il surveiller la calibration dans un modèle de PD ?

### Squelette du challenge

~~~python
"""Challenge 08. Conserver les trois fichiers de ce cours dans le même dossier."""
from ch03_cours08_lab import dataset
X_train, X_test, y_train, y_test = dataset()
# TODO : réserver une validation distincte pour le seuil ou l'early stopping.
# TODO : construire les modèles et leurs pipelines sans fuite de données.
# TODO : comparer les candidats en CV sur train, puis figer la sélection.
# TODO : réaliser le tuning sans utiliser le test.
# TODO : auditer ROC-AUC, AP, et Brier uniquement pour des probabilités.
# TODO : produire les figures et une recommandation argumentée.
raise NotImplementedError("Compléter le challenge avant de consulter le corrigé.")
~~~

[[solution]]

## Questions d’entretien

[[interview]]

## 28. Mini-quiz interactif

[[quiz]]

## 29. Validation du cours

[[assessment]]

Passer au cours 9 — Support Vector Machines (SVM)
À retenir : le Gradient Boosting construit un modèle additif en ajoutant successivement des arbres qui contribuent à réduire une fonction de perte. XGBoost, LightGBM et CatBoost sont trois implémentations majeures, avec des choix d'optimisation différents.

Dans le prochain cours, nous passerons à une autre grande famille d'algorithmes supervisés : les Support Vector Machines (SVM), fondés sur la recherche d'une frontière de décision à marge maximale.