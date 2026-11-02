Nous avons étudié trois grandes approches de régression : linéaire, régularisée et polynomiale.

Aujourd'hui, nous passons à un problème différent : prédire une catégorie plutôt qu'une valeur numérique.

La régression logistique est particulièrement importante en Data Science, car elle constitue à la fois un excellent modèle de classification, une baseline robuste et un modèle dont les coefficients peuvent être interprétés.

Pour comprendre son fonctionnement, nous allons utiliser un exemple de risque de crédit : prédire si un emprunteur fera défaut.

## Objectifs du cours

À la fin, tu devras savoir :

- Expliquer pourquoi la régression linéaire n'est pas adaptée directement à la classification binaire.
- Comprendre la Sigmoid, les odds et le logit.
- Interpréter les coefficients et les Odds Ratios.
- Expliquer la Log-Loss et le Maximum de Vraisemblance.
- Comprendre l'apprentissage des coefficients.
- Modifier un seuil de classification.
- Évaluer un modèle avec les métriques appropriées.
- Construire un pipeline de classification avec scikit-learn.
- Comprendre la différence entre discrimination et calibration.

Partie I — Pourquoi la régression logistique ?


## 1. Un problème de classification binaire

Imaginons un dataset bancaire :

Client	Ratio d'endettement	Revenu mensuel	Défaut
A	0,15	4 500	0
B	0,30	3 200	0
C	0,55	2 400	1
D	0,75	1 700	1
E	0,40	3 000	0

La variable cible est :
\[
Y\in\{0,1\}
\]

Avec :

- \(Y=0\) : absence de défaut.
- \(Y=1\) : défaut.

Nous souhaitons estimer :
\[
P(Y=1\mid X)
\]

C'est-à-dire la probabilité conditionnelle de défaut compte tenu des caractéristiques de l'emprunteur.
Pourquoi ne pas utiliser LinearRegression ?
Une régression linéaire pourrait prédire :
\[
\hat y=\beta_0+\beta_1x
\]

Mais elle pourrait produire :
\[
\hat y=-0.25
\]

ou :
\[
\hat y=1.37
\]

Ces valeurs ne sont pas des probabilités valides.

Une probabilité doit respecter :
\[
0\le P(Y=1\mid X)\le1
\]

Nous avons donc besoin d'une fonction qui transforme un score numérique quelconque en une valeur comprise entre 0 et 1.

C'est le rôle de la fonction Sigmoid.

Partie II — La fonction Sigmoid



## 2. Comprendre la Sigmoid

La régression logistique commence par calculer un score linéaire :
\[
z=\beta_0+\beta_1x_1+\cdots+\beta_px_p
\]

Puis elle applique la fonction :
\[
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
\]

La prédiction probabiliste devient :
\[
\boxed{P(Y=1\mid X)=\sigma(\beta_0+\beta^TX)}
\]

Cette fonction transforme tout nombre réel en une valeur strictement comprise entre 0 et 1.
Quelques exemples

Score \(z\)	Sigmoid \(\sigma(z)\)
-4	0,018
-2	0,119
0	0,500
2	0,881
4	0,982

Lorsque \(z\) augmente, la probabilité augmente.

Lorsque \(z\) diminue, la probabilité se rapproche de zéro.



## 3. Explorer la Sigmoid


[[logisticsigmoid]]




## 4. Propriétés mathématiques de la Sigmoid

La Sigmoid possède plusieurs propriétés importantes.
Valeur au centre
\[
\sigma(0)=\frac{1}{2}=0.5
\]

Symétrie
\[
\sigma(-z)=1-\sigma(z)
\]

Dérivée
\[
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
\]

Cette dérivée est particulièrement pratique pour calculer les gradients pendant l'entraînement.
Comportement aux extrêmes
\[
\lim_{z\to+\infty}\sigma(z)=1
\]

\[
\lim_{z\to-\infty}\sigma(z)=0
\]

La Sigmoid approche donc 0 et 1 sans les atteindre mathématiquement pour un score fini.

Partie III — Odds, Log-Odds et Logit



## 5. Qu'est-ce qu'une odds ?

La probabilité et les odds ne sont pas identiques.

Pour une probabilité \(p\), les odds sont :
\[
\boxed{Odds=\frac{p}{1-p}}
\]

Elles représentent le rapport entre la probabilité de l'événement et celle de son complément.
Exemple
Si :
\[
p=0.8
\]

Alors :
\[
Odds=\frac{0.8}{0.2}=4
\]

Cela signifie que les chances de l'événement sont quatre fois celles du non-événement.

À l'inverse, si :
\[
p=0.2
\]

Alors :
\[
Odds=\frac{0.2}{0.8}=0.25
\]




## 6. Le Logit

Le logit est le logarithme des odds :
\[
\boxed{logit(p)=\ln\left(\frac{p}{1-p}\right)}
\]

La régression logistique suppose que le logit est une combinaison linéaire des variables :
\[
\boxed{
\ln\left(\frac{p}{1-p}\right)
=
\beta_0+\beta_1x_1+\cdots+\beta_px_p
}
\]

C'est la relation fondamentale du modèle.

La probabilité n'est donc pas linéaire par rapport aux variables, mais le log-odds l'est.



## 7. Probabilité versus odds


[[logisticodds]]

Lorsque p=50 %, les odds valent 1 et le logit vaut 0.




## 8. Que signifie un coefficient de régression logistique ?

Supposons :
\[
logit(p)=-3+1.2x_1-0.5x_2
\]

Avec :

- \(x_1\) : ratio d'endettement.
- \(x_2\) : indicateur de stabilité financière.

Interprétation des signes
Un coefficient positif augmente les log-odds de l'événement lorsque la variable augmente, toutes choses égales par ailleurs.

Un coefficient négatif les diminue.

Mais il faut faire attention : un coefficient de 1,2 ne signifie pas que la probabilité augmente de 120 %.

Pour interpréter les coefficients, nous utilisons les Odds Ratios.



## 9. Odds Ratio — OR

Pour un coefficient \(\beta_j\) :
\[
\boxed{OR_j=e^{\beta_j}}
\]

L'Odds Ratio indique par quel facteur les odds sont multipliées lorsque la variable augmente d'une unité, les autres variables restant constantes.
Exemple 1 — Coefficient positif
\[
\beta=0.7
\]

\[
OR=e^{0.7}\approx2.01
\]

Les odds sont multipliées par environ 2,01.
Exemple 2 — Coefficient négatif
\[
\beta=-0.5
\]

\[
OR=e^{-0.5}\approx0.61
\]

Les odds sont multipliées par environ 0,61, soit une diminution d'environ 39 %.
Attention à l'unité
Si debt_ratio est compris entre 0 et 1, une augmentation d'une unité représente 100 points de pourcentage.

Pour une augmentation de 0,1, l'Odds Ratio devient :
\[
OR=e^{0.1\beta}
\]

L'interprétation doit toujours tenir compte de l'échelle réelle de la variable.



## 10. Explorateur des Odds Ratios


[[logisticor]]




## 11. Pourquoi ne pas utiliser la MSE ?

Nous avons vu que la régression linéaire minimise souvent les erreurs quadratiques.

Pour la régression logistique, nous utilisons généralement une fonction de perte issue du Maximum de Vraisemblance.

La fonction correspondante est la Binary Cross-Entropy, également appelée Log-Loss.



## 12. La distribution de Bernoulli

En classification binaire :
\[
Y\in\{0,1\}
\]

Nous supposons :
\[
Y_i\mid X_i\sim Bernoulli(p_i)
\]

Avec :
\[
p_i=\sigma(\beta_0+\beta^TX_i)
\]

La probabilité d'observer \(y_i\) peut s'écrire :
\[
P(Y_i=y_i\mid X_i)
=
p_i^{y_i}(1-p_i)^{1-y_i}
\]

Cette formule fonctionne pour les deux classes.

Si \(y_i=1\) :
\[
P(Y_i=1)=p_i
\]

Si \(y_i=0\) :
\[
P(Y_i=0)=1-p_i
\]




## 13. Maximum de Vraisemblance — MLE

Nous cherchons les coefficients qui rendent les observations aussi probables que possible.

Pour \(n\) observations indépendantes conditionnellement aux features, la vraisemblance est :
\[
L(\beta)=\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}
\]

Nous maximisons cette quantité.

Pour faciliter les calculs, nous prenons le logarithme :
\[
\log L(\beta)
=
\sum_{i=1}^{n}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
\]

Maximiser la log-vraisemblance revient à minimiser son opposé.

Nous obtenons la Log-Loss.



## 14. Binary Cross-Entropy — Log-Loss

\[
\boxed{
J(\beta)=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
}
\]

Cette fonction pénalise particulièrement les prédictions très confiantes mais incorrectes.
Exemple
Supposons que la vraie classe soit :
\[
y=1
\]

Si le modèle prédit :
\[
p=0.9
\]

La perte vaut :
\[
-\log(0.9)\approx0.105
\]

Mais s'il prédit :
\[
p=0.01
\]

La perte vaut :
\[
-\log(0.01)\approx4.605
\]

La deuxième prédiction est beaucoup plus pénalisée.



## 15. Explorer la Log-Loss


[[logisticloss]]




## 16. Comment optimiser les coefficients ?

La régression logistique apprend les coefficients en minimisant la Log-Loss, éventuellement avec régularisation.

Le gradient de la perte non régularisée par rapport aux coefficients est :
\[
\boxed{
\nabla_\beta J=
\frac{1}{n}X^T(p-y)
}
\]

Où :

- \(X\) est la matrice des features.
- \(p\) est le vecteur des probabilités prédites.
- \(y\) est le vecteur des vraies classes.

La descente de gradient met à jour les coefficients :
\[
\boxed{
\beta^{(t+1)}
=
\beta^{(t)}-\alpha\nabla_\beta J
}
\]

Intuition
1. Initialiser les coefficients.
2. Calculer les scores linéaires.
3. Appliquer la Sigmoid.
4. Calculer les erreurs probabilistes.
5. Calculer le gradient.
6. Modifier les coefficients.
7. Répéter jusqu'à convergence.
Important
Dans scikit-learn, LogisticRegression utilise des solveurs numériques tels que lbfgs, liblinear, newton-cg, newton-cholesky, sag ou saga, selon la configuration et la version.

Ils ne correspondent pas tous à une simple descente de gradient.

Partie VII — Frontière de décision



## 17. Transformer une probabilité en classe

La régression logistique produit :
\[
p=P(Y=1\mid X)
\]

Mais pour obtenir une classe, nous choisissons un seuil \(t\).
\[
\hat y=
\begin{cases}
1 & \text{si }p\ge t\\
0 & \text{sinon}
\end{cases}
\]

Par défaut, on utilise souvent :
\[
t=0.5
\]

Cependant, ce seuil n'est pas toujours optimal.

Dans un problème de risque de crédit, les conséquences d'un faux négatif et d'un faux positif peuvent être très différentes.



## 18. Seuil de décision interactif


[[logisticthreshold]]

Que se passe-t-il quand le seuil diminue ?
En général :

- Davantage d'observations sont prédites positives.
- Le recall ne diminue pas.
- Le nombre de faux positifs ne diminue pas.

La précision, elle, peut augmenter ou diminuer selon la distribution des scores.
Que se passe-t-il quand le seuil augmente ?

- Moins d'observations sont prédites positives.
- Le nombre de faux positifs ne peut pas augmenter.
- Le recall ne peut pas augmenter.

Le seuil optimal dépend du coût métier des erreurs.

Partie VIII — Évaluer une régression logistique



## 19. La matrice de confusion

	Prédit 0	Prédit 1
Réel 0	TN	FP
Réel 1	FN	TP

À partir de cette matrice, nous calculons plusieurs métriques.
Accuracy
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Precision
\[
Precision=\frac{TP}{TP+FP}
\]

Recall / Sensitivity
\[
Recall=\frac{TP}{TP+FN}
\]

Specificity
\[
Specificity=\frac{TN}{TN+FP}
\]

F1-score
\[
F_1=2\frac{Precision\times Recall}{Precision+Recall}
\]




## 20. ROC-AUC et PR-AUC

ROC-AUC
La courbe ROC représente :
\[
TPR=\frac{TP}{TP+FN}
\]

en fonction de :
\[
FPR=\frac{FP}{FP+TN}
\]

L'AUC mesure la capacité du modèle à classer les observations positives au-dessus des négatives.

Une interprétation courante est :
L'AUC correspond à la probabilité qu'une observation positive choisie au hasard reçoive un score supérieur à celui d'une observation négative choisie au hasard, avec un traitement approprié des égalités.

Precision-Recall
La courbe Precision-Recall est particulièrement informative lorsque la classe positive est rare.

Nous pouvons utiliser average_precision_score pour résumer cette courbe.

Attention : Average Precision (AP) et aire calculée par intégration trapézoïdale de la courbe PR ne sont pas exactement la même métrique.



## 21. Discrimination versus calibration

C'est une distinction fondamentale pour les modèles de probabilité de défaut.

Discrimination : le modèle classe-t-il correctement les emprunteurs du moins risqué au plus risqué ?

Calibration : les probabilités prédites correspondent-elles aux fréquences de défaut observées ?
Exemple
Un modèle prédit une PD de 10 % à un groupe de 1 000 emprunteurs.

Si environ 100 emprunteurs font défaut, la prédiction moyenne est cohérente avec la fréquence observée pour ce groupe.

Mais si 250 emprunteurs font défaut, le modèle sous-estime fortement le risque dans ce groupe.

Un modèle peut obtenir une bonne ROC-AUC tout en ayant une mauvaise calibration.
Brier Score
\[
\boxed{
Brier=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2
}
\]

Plus le Brier Score est faible, meilleure est la qualité globale des prédictions probabilistes selon cette métrique.

Il mesure à la fois des aspects de calibration et de discrimination : il ne constitue pas un test de calibration à lui seul.

Partie IX — Régularisation en régression logistique



## 22. Pourquoi régulariser ?

Comme la régression linéaire, la régression logistique peut surapprendre, particulièrement lorsque :

- Le nombre de features est élevé.
- Les variables sont fortement corrélées.
- Le dataset est petit.
- Le modèle utilise beaucoup d'interactions ou de termes polynomiaux.

Nous pouvons utiliser des pénalités L1, L2 ou Elastic Net.
L2
\[
J_{\mathrm{total}}
=
J_{\mathrm{logloss}}+\lambda\sum_j\beta_j^2
\]

L1
\[
J_{\mathrm{total}}
=
J_{\mathrm{logloss}}+\lambda\sum_j|\beta_j|
\]

Dans scikit-learn, le paramètre C contrôle inversement la force de régularisation.
\[
\boxed{C\propto\frac{1}{\lambda}}
\]

Interprétation

Valeur de C	Régularisation
Petite	Forte
Grande	Faible

C'est l'inverse de alpha utilisé dans Ridge et Lasso.
Exemple
~~~python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1., penalty="l2", max_iter=1000)
~~~




Pour une régularisation L1 :
~~~python
model = LogisticRegression(C=.5, penalty="l1", solver="liblinear", max_iter=1000)
~~~




Tous les solveurs ne prennent pas en charge toutes les pénalités.

Partie X — Class Imbalance



## 23. Pourquoi le déséquilibre est-il important ?

Imaginons :

- 9 800 emprunteurs sans défaut.
- 200 emprunteurs en défaut.

La classe positive représente seulement 2 % du dataset.

Un modèle qui prédit toujours zéro obtient :
\[
Accuracy=98\%
\]

Mais :
\[
Recall=0
\]

Il ne détecte aucun défaut.
Pondération des classes
Une approche consiste à utiliser :
~~~python
LogisticRegression(class_weight="balanced", max_iter=1000)
~~~




Les poids équilibrés sont calculés à partir des effectifs des classes :
\[
w_c=\frac{n}{K n_c}
\]

Avec :

- \(n\) : nombre total d'observations.
- \(K\) : nombre de classes.
- \(n_c\) : nombre d'observations de la classe \(c\).

Cette pondération peut améliorer la prise en compte de la classe minoritaire, mais elle peut également modifier la calibration des probabilités.

Pour une application PD, nous devons donc évaluer séparément la discrimination et la calibration.

Partie XI — Coding Lab complet



## 24. Construire un modèle de classification de défaut

Nous allons créer un dataset synthétique contenant :

- Revenu.
- Ratio d'endettement.
- Ancienneté professionnelle.
- Historique d'incidents.
- Défaut de paiement.

Laboratoire complet : données synthétiques, baseline sans pondération, validation croisée sur train et audit final du modèle fixe. Les graphiques ROC, PR et calibration s’affichent avec l’option --plot.

~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, log_loss, brier_score_loss)

rng = np.random.default_rng(42)
n = 5000
income = rng.lognormal(mean=8.2, sigma=.5, size=n)
debt_ratio = rng.beta(a=2, b=5, size=n)
employment_years = rng.integers(0, 30, size=n)
previous_incidents = rng.binomial(1, .12, size=n)
logit = -3.5 - .00015 * income + 4.0 * debt_ratio - .04 * employment_years + 1.3 * previous_incidents
probability = 1 / (1 + np.exp(-logit))
default = rng.binomial(1, probability)
df = pd.DataFrame({"income": income, "debt_ratio": debt_ratio,
                   "employment_years": employment_years, "previous_incidents": previous_incidents,
                   "default": default})
X, y = df.drop(columns="default"), df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def make_pipeline(C=1., class_weight=None):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, C=C, class_weight=class_weight))])


def probability_metrics(y_true, proba):
    return {"roc_auc": roc_auc_score(y_true, proba),
            "average_precision": average_precision_score(y_true, proba),
            "log_loss": log_loss(y_true, proba, labels=[0, 1]),
            "brier": brier_score_loss(y_true, proba)}


def plot_diagnostics(y_true, proba):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1])
    CalibrationDisplay.from_predictions(y_true, proba, n_bins=10, strategy="quantile", ax=axes[2])
    axes[2].set_title("Calibration : description du test, sans réglage")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher ROC, PR et calibration")
    args = parser.parse_args()
    pipeline = make_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring={"ap": "average_precision", "roc_auc": "roc_auc",
                                     "log_loss": "neg_log_loss", "brier": "neg_brier_score"})
    print("Prévalence train:", y_train.mean())
    print("CV train : AP", scores["test_ap"].mean(), "ROC-AUC", scores["test_roc_auc"].mean(),
          "Log-Loss", -scores["test_log_loss"].mean(), "Brier", -scores["test_brier"].mean())
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = proba >= .5  # Convention explicite pour le seuil de cet exercice.
    assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
    print("Audit final de la baseline fixe:", probability_metrics(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    coefficients = pipeline.named_steps["model"].coef_[0]
    raw_coefficients = coefficients / pipeline.named_steps["scaler"].scale_
    increments = np.array([1000., .1, 5., 1.])
    interpretation = pd.DataFrame({"feature": X.columns, "coefficient_standardise": coefficients,
                                    "OR_par_ecart_type": np.exp(coefficients),
                                    "increment_original": increments,
                                    "OR_par_increment_original": np.exp(raw_coefficients * increments)})
    print(interpretation.to_string(index=False))
    print("Incréments : revenu +1000, ratio +0.1, ancienneté +5 ans, incident de 0 à 1.")
    # Vérifier l'interprétation conditionnelle des odds sur un seul emprunteur fictif.
    client = X_test.iloc[[0]].copy()
    changed = client.copy()
    changed["income"] += 1000.
    p0, p1 = pipeline.predict_proba(client)[0, 1], pipeline.predict_proba(changed)[0, 1]
    assert np.isclose((p1 / (1-p1)) / (p0 / (1-p0)), np.exp(raw_coefficients[0] * 1000.))
    if args.plot:
        plot_diagnostics(y_test, proba)
    print("Données et coefficients générateurs fictifs ; les sorties ne sont pas un score bancaire validé.")
    print("Le Brier mesure une qualité probabiliste globale, pas uniquement la calibration.")
~~~

Interprétation : les OR des coefficients standardisés correspondent à un écart-type des variables transformées. Le tableau fournit aussi des OR pour des incréments dans les unités originales : +1 000 de revenu, +0,1 de ratio, +5 ans d’ancienneté et incident de 0 à 1.

Ces données sont fictives. Une courbe de calibration bruitée sur une classe rare et un Brier faible ne prouvent pas, à eux seuls, la validité d’un modèle bancaire.




## 25. Optimiser C et la pondération des classes

~~~python
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# make_pipeline() est défini dans le laboratoire.
pipeline = make_pipeline()
param_grid = {"model__C": np.logspace(-3, 3, 10), "model__class_weight": [None, "balanced"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="average_precision", cv=cv)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV AP:", search.best_score_)
best_model = search.best_estimator_
# Dans le challenge avec choix du seuil, réserver X_valid AVANT ce tuning et utiliser X_fit ici.
~~~




Ici, nous optimisons l'Average Precision, ce qui peut être pertinent lorsque la classe positive est rare.

Mais dans un problème bancaire réel, il faut également vérifier :

- La ROC-AUC.
- La calibration.
- La stabilité.
- Les coûts métier.
- Les exigences de validation du modèle.

Un modèle optimisé pour l'Average Precision n'est pas automatiquement bien calibré.

Partie XIII — Coding Challenge



## 26. Construire un Credit Default Classifier

Challenge 04 — Logistic Regression

+500 XP

Construis un modèle complet de classification binaire, compare plusieurs configurations et analyse les probabilités prédites.
~~~python
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from ch03_cours04_lab import X_train, X_test, y_train, y_test, make_pipeline

# Réserver la validation du seuil avant de régler C et les poids de classes.
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
pipeline = make_pipeline()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, {"model__C": np.r_[np.logspace(-3, 3, 10), 1.],
                                "model__class_weight": [None, "balanced"]},
                      scoring="average_precision", cv=cv)
# TODO : comparer les configurations sur X_fit, y_fit, sans utiliser X_valid pour le tuning.
# TODO : évaluer Accuracy, Precision, Recall, F1 et les métriques probabilistes.
# TODO : choisir un seuil sur la validation distincte avec des coûts fictifs explicités.
# TODO : figer modèle et seuil ; ne pas réentraîner sans revalider le seuil.
# TODO : réaliser l'audit test puis tracer ROC, PR et calibration.
# BONUS : sigmoid et log-loss NumPy stables, y compris pour des scores extrêmes.
~~~




Questions d'analyse
1. Pourquoi une Accuracy élevée peut-elle être trompeuse ?
2. Pourquoi le seuil 0,5 n'est-il pas toujours optimal ?
3. Quelle différence entre ROC-AUC et Average Precision ?
4. Pourquoi la pondération des classes peut-elle modifier la calibration ?
5. Comment interpréter un Odds Ratio supérieur à 1 ?
6. Pourquoi la Log-Loss pénalise-t-elle fortement les erreurs confiantes ?



[[solution]]

## Questions d’entretien — Régression logistique

[[interview]]




## 27. Mini-quiz interactif


[[quiz]]




## 28. Validation du cours


[[assessment]]

À retenir : la régression logistique transforme une combinaison linéaire de features en une probabilité grâce à la Sigmoid. Elle apprend ses coefficients en minimisant une Log-Loss, et permet d'interpréter les effets des variables via les Odds Ratios.

Pour les applications de risque de crédit, il faut particulièrement distinguer la capacité à classer les emprunteurs (discrimination) et la qualité des probabilités estimées (calibration).

Au prochain cours, nous étudierons K-Nearest Neighbors (KNN), un algorithme qui prend ses décisions en observant les exemples les plus proches, sans apprendre de coefficients comme la régression logistique.

Le support détaillé du cours 5 sera ajouté à sa réception.