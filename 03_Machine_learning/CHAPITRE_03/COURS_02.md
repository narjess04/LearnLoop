Dans le cours précédent, nous avons étudié la régression linéaire :
\[
\hat y=\beta_0+\beta_1x_1+\cdots+\beta_px_p
\]

Nous avons vu comment trouver les coefficients en minimisant les erreurs quadratiques.

Mais imaginons que notre dataset possède 500 variables et seulement 200 observations. La régression linéaire ordinaire peut apprendre des coefficients très instables et s'adapter excessivement aux données d'entraînement.

Comment éviter cela ?

En ajoutant une contrainte sur les coefficients : c'est le principe de la régularisation.


## 1. Pourquoi régulariser un modèle ?

La régression linéaire ordinaire cherche à minimiser :
\[
L_{\mathrm{OLS}}=\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Elle se concentre sur les erreurs d'entraînement.

Mais si certaines variables sont fortement corrélées, bruitées ou trop nombreuses, le modèle peut apprendre des coefficients excessivement élevés.
Exemple
Supposons que deux variables représentent presque la même information :

- revenu_annuel
- revenu_mensuel × 12

Une régression ordinaire pourrait apprendre des coefficients comme :
\[
\beta_1=25.4,\qquad\beta_2=-24.9
\]

Ces coefficients peuvent produire des prédictions correctes sur le train, mais devenir instables lorsque les données changent légèrement.

La régularisation encourage des solutions moins complexes.
Intuition — Ajustement d'un modèle

Modèle simple


Modèle trop flexible


Observations


Illustration conceptuelle du surapprentissage. Ridge et Lasso contrôlent les coefficients ; ils ne rendent pas directement une droite non linéaire.
La régularisation ne garantit pas une meilleure performance. Elle introduit volontairement du biais pour tenter de réduire la variance du modèle.

Partie I — Comprendre la régularisation



## 2. Le principe mathématique

Nous modifions la fonction de coût :
\[
\boxed{L=L_{\mathrm{erreur}}+\lambda L_{\mathrm{pénalité}}}
\]

Deux termes apparaissent.

Erreur : mesure l'écart entre les prédictions et les observations.

Pénalité : décourage certaines valeurs des coefficients.

Le paramètre \(\lambda\) contrôle l'intensité de la régularisation.

Valeur de \(\lambda\)	Effet
Proche de 0	Peu de régularisation
Modérée	Compromis erreur / complexité
Très élevée	Coefficients fortement contraints

Dans scikit-learn, ce paramètre est généralement nommé alpha pour Ridge, Lasso et ElasticNet.
Attention aux conventions
Les fonctions de coût peuvent être écrites avec des facteurs différents selon les bibliothèques.

Pour Ridge dans scikit-learn, l'objectif est :
\[
\|y-Xw\|_2^2+\alpha\|w\|_2^2
\]

Pour Lasso :
\[
\frac{1}{2n}\|y-Xw\|_2^2+\alpha\|w\|_1
\]

Il ne faut donc pas comparer numériquement les valeurs de alpha entre différents algorithmes sans tenir compte de leur formulation.



## 3. Rappel : norme L1 et norme L2

Pour un vecteur de coefficients :
\[
w=(w_1,w_2,\ldots,w_p)
\]

La norme L1 est :
\[
\|w\|_1=\sum_{j=1}^{p}|w_j|
\]

La norme L2 est :
\[
\|w\|_2=\sqrt{\sum_{j=1}^{p}w_j^2}
\]

La régularisation Ridge utilise le carré de la norme L2 :
\[
\|w\|_2^2=\sum_jw_j^2
\]

La régularisation Lasso utilise la norme L1.
Exemple
Avec :
\[
w=(3,-4)
\]

Nous obtenons :
\[
\|w\|_1=|3|+|-4|=7
\]

\[
\|w\|_2^2=3^2+(-4)^2=25
\]

La différence entre ces pénalités explique une grande partie des comportements de Ridge et Lasso.

Partie II — Ridge Regression (L2)



## 4. Qu'est-ce que Ridge ?

Ridge est une régression linéaire avec une pénalité quadratique sur les coefficients.

Sa fonction de coût est :
\[
\boxed{
L_{\mathrm{Ridge}}=
\sum_{i=1}^{n}(y_i-\hat y_i)^2+
\alpha\sum_{j=1}^{p}\beta_j^2
}
\]

L'intercept n'est généralement pas pénalisé.
Intuition
Ridge cherche des coefficients qui :
1. Produisent de bonnes prédictions.
2. Ne deviennent pas inutilement élevés.
Plus alpha augmente, plus les coefficients sont encouragés à se rapprocher de zéro.

Point essentiel : Ridge réduit les coefficients, mais ne les rend généralement pas exactement nuls.



## 5. Visualisation de Ridge


[[ridgeshrink]]




## 6. Ridge et la multicolinéarité

Ridge est particulièrement utile lorsque les variables sont fortement corrélées.

Pour une matrice \(X\) centrée et des variables correctement préparées, une forme de la solution Ridge est :
\[
\boxed{
\hat\beta_{\mathrm{Ridge}}
=
(X^TX+\alpha I)^{-1}X^Ty
}
\]

Avec \(\alpha>0\), le terme ajouté à la diagonale stabilise le problème numérique.

Cela aide à réduire l'instabilité des coefficients en présence de multicolinéarité.

Cependant, Ridge ne supprime pas la corrélation entre les variables : il réduit ses conséquences sur l'estimation.



## 7. Ridge avec scikit-learn

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ridge_pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
ridge_pipeline.fit(X_train, y_train)
y_pred = ridge_pipeline.predict(X_test)
print("Coefficients :", ridge_pipeline.named_steps["model"].coef_)
~~~




Pourquoi StandardScaler ?
Parce que Ridge pénalise les coefficients.

Supposons deux variables :

- Revenu annuel : valeurs entre 10 000 et 100 000.
- Ratio d'endettement : valeurs entre 0 et 1.

Sans standardisation, les coefficients associés aux deux variables ne sont pas sur une échelle comparable.

La régularisation pourrait donc agir de manière déséquilibrée.

Bonne pratique : utiliser un pipeline avec StandardScaler avant Ridge lorsque les variables numériques ont des échelles différentes.

Partie III — Lasso Regression (L1)



## 8. Qu'est-ce que Lasso ?

Lasso signifie Least Absolute Shrinkage and Selection Operator.

Sa fonction de coût est :
\[
\boxed{
L_{\mathrm{Lasso}}=
\frac{1}{2n}\sum_{i=1}^{n}(y_i-\hat y_i)^2+
\alpha\sum_{j=1}^{p}|\beta_j|
}
\]

Contrairement à Ridge, Lasso peut ramener certains coefficients exactement à zéro.
Exemple
Avant régularisation :
\[
\beta=(3.2,1.8,0.7,0.3,0.1)
\]

Après Lasso, nous pourrions obtenir :
\[
\beta=(2.8,1.3,0.4,0,0)
\]

Les deux dernières variables ne contribuent plus directement à la prédiction.

Lasso réalise donc une forme de Feature Selection intégrée.



## 9. Pourquoi Lasso produit-il des coefficients nuls ?

La pénalité L1 n'est pas différentiable en zéro.

Sa géométrie possède des coins sur les axes des coefficients.

Lorsqu'on minimise l'erreur sous une contrainte L1, la solution peut se trouver exactement sur un axe, ce qui annule un ou plusieurs coefficients.
[[penaltygeometry]]

Cette représentation illustre le cas de deux coefficients, sans entraîner un modèle.




## 10. Explorer la sparsité de Lasso


[[lassosparse]]




## 11. Lasso avec scikit-learn

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
lasso_pipeline = Pipeline([("scaler", StandardScaler()),
                           ("model", Lasso(alpha=.1, max_iter=10000))])
lasso_pipeline.fit(X_train, y_train)
coefficients = lasso_pipeline.named_steps["model"].coef_
print("Coefficients :", coefficients)
print("Variables sélectionnées :", (coefficients != 0).sum())
~~~




Limites de Lasso
Lasso n'est pas toujours la meilleure solution.

En présence de variables très fortement corrélées, il peut sélectionner l'une et éliminer les autres de manière instable.

Une petite modification des données peut alors changer les variables retenues.

De plus, une régularisation excessive peut éliminer des variables réellement utiles.

Partie IV — Elastic Net



## 12. Pourquoi combiner Ridge et Lasso ?

Ridge stabilise les coefficients.

Lasso peut effectuer une sélection de variables.

Elastic Net combine les deux pénalités.

Sa fonction de coût, dans la convention scikit-learn, est :
\[
\boxed{
L_{\mathrm{EN}}=
\frac{1}{2n}\|y-Xw\|_2^2+
\alpha\rho\|w\|_1+
\frac{\alpha(1-\rho)}{2}\|w\|_2^2
}
\]

Où :

- \(\alpha\) contrôle l'intensité globale de la régularisation.
- \(\rho\), appelé l1_ratio, contrôle la proportion relative de L1 et L2.

Interprétation de l1_ratio

l1_ratio	Comportement
0	Pénalité L2 seule, mathématiquement de type Ridge
0,25	Principalement L2
0,50	Mélange L1 et L2
0,75	Principalement L1
1	Lasso

Pour ElasticNet de scikit-learn, les valeurs exactement égales ou très proches de zéro peuvent nécessiter des précautions numériques ; pour une pénalité purement L2, il vaut mieux utiliser Ridge.



## 13. Explorateur interactif — Elastic Net


[[elasticmix]]




## 14. Elastic Net avec Python

~~~python
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
elastic_pipeline = Pipeline([("scaler", StandardScaler()),
                             ("model", ElasticNet(alpha=.1, l1_ratio=.5, max_iter=10000))])
elastic_pipeline.fit(X_train, y_train)
~~~




Elastic Net est souvent intéressant lorsque nous avons beaucoup de variables, dont certaines fortement corrélées, et que nous souhaitons combiner stabilité et sélection.

Partie V — Comparer les trois méthodes



## 15. Ridge versus Lasso versus Elastic Net

Critère	Ridge	Lasso	Elastic Net
Pénalité	L2	L1	L1 + L2
Réduit les coefficients	Oui	Oui	Oui
Coefficients exactement nuls	Généralement non	Oui, possible	Oui, possible
Feature Selection intégrée	Non, généralement	Oui	Oui
Variables corrélées	Stabilisation utile	Sélection parfois instable	Compromis souvent intéressant
Hyperparamètres	alpha	alpha	alpha, l1_ratio
Standardisation recommandée	Oui	Oui	Oui

Comment choisir ?
Ridge : lorsque la plupart des variables peuvent contribuer à la prédiction et que nous voulons stabiliser leurs coefficients.

Lasso : lorsque nous souhaitons une représentation plus parcimonieuse et potentiellement éliminer des variables.

Elastic Net : lorsque nous voulons combiner les deux comportements, notamment en présence de variables corrélées.

Le choix final doit être confirmé par validation croisée.



## 16. Comprendre le compromis biais-variance

La régularisation agit sur le compromis biais-variance.

Avec une régularisation très faible :

- Le modèle est plus flexible.
- Il peut obtenir une faible erreur d'entraînement.
- Il peut être sensible aux fluctuations de l'échantillon.

Avec une régularisation forte :

- Le modèle devient plus contraint.
- Ses coefficients sont davantage réduits.
- Il peut sous-apprendre.

[[regularizationtradeoff]]

L'objectif n'est pas de minimiser uniquement l'erreur d'entraînement.

Nous cherchons la configuration qui généralise le mieux à de nouvelles données.

Partie VI — Choisir alpha avec Cross-Validation



## 17. Pourquoi ne pas choisir alpha au hasard ?

Un alpha trop faible peut laisser des coefficients instables.

Un alpha trop élevé peut dégrader la capacité prédictive.

Nous devons donc comparer plusieurs valeurs sur des données de validation.

Il est souvent pertinent de tester une grille logarithmique :
\[
\alpha\in\{10^{-4},10^{-3},10^{-2},\ldots,10^3\}
\]




## 18. RidgeCV et LassoCV

Scikit-learn fournit des estimateurs spécialisés.
RidgeCV
~~~python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ridge_cv = Pipeline([("scaler", StandardScaler()),
                     ("model", RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5))])
ridge_cv.fit(X_train, y_train)
print("Best alpha:", ridge_cv.named_steps["model"].alpha_)
~~~




LassoCV
~~~python
from sklearn.linear_model import LassoCV
lasso_cv = Pipeline([("scaler", StandardScaler()),
                     ("model", LassoCV(cv=5, max_iter=10000, random_state=42))])
lasso_cv.fit(X_train, y_train)
print("Best alpha:", lasso_cv.named_steps["model"].alpha_)
~~~




Ces méthodes sont pratiques, mais le choix du score, des folds et de la stratégie de validation doit rester cohérent avec le problème.


Précaution sur la validation interne : un StandardScaler placé avant RidgeCV ou LassoCV est ajusté sur tout le train avant leur propre validation interne. Ce montage ne réajuste donc pas le scaler dans chacun de leurs folds internes. Pour sélectionner alpha avec un preprocessing réappris à chaque fold, utiliser GridSearchCV autour du pipeline complet, comme dans la section suivante et le challenge. Le test final reste distinct dans tous les cas.




## 19. GridSearchCV pour Elastic Net

~~~python
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
pipeline = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=50000))])
param_grid = {"model__alpha": np.logspace(-3, 2, 10), "model__l1_ratio": [.2, .5, .8, 1.]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="neg_root_mean_squared_error", cv=cv)
search.fit(X_train, y_train)
print(search.best_params_)
print("RMSE de sélection CV:", -search.best_score_)
~~~




Pourquoi un score négatif ?
Scikit-learn utilise une convention où les scores d'évaluation sont maximisés.

Pour les métriques d'erreur, comme RMSE, les scorers correspondants renvoient donc leur opposé.

Ainsi :
~~~python
rmse = -search.best_score_
~~~




Nous retrouvons une RMSE positive.

Partie VII — Coding Lab complet



## 20. Expérience : données corrélées et régularisation

Nous allons construire un dataset contenant :

- Des variables informatives.
- Des variables corrélées.
- Des variables bruitées.

Puis comparer quatre modèles :
1. Linear Regression.
2. Ridge.
3. Lasso.
4. Elastic Net.

Laboratoire complet : comparer les quatre configurations sur les mêmes folds train ; figer le choix par la RMSE CV avant de lire le tableau d’audit final. Le test ne doit pas servir à changer les hyperparamètres.

~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = make_regression(n_samples=500, n_features=30, n_informative=8,
                       noise=25, random_state=42)
rng = np.random.default_rng(42)
X_correlated = X[:, :5] + rng.normal(0, .05, size=(500, 5))
X = np.hstack([X, X_correlated])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


def make_models():
    return {name: Pipeline([("scaler", StandardScaler()), ("model", model)])
            for name, model in [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge(alpha=10)),
                ("Lasso", Lasso(alpha=1, max_iter=50000)),
                ("Elastic Net", ElasticNet(alpha=1, l1_ratio=.5, max_iter=50000))]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les coefficients")
    args = parser.parse_args()
    models = make_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, cv_scores = [], {}
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error", return_estimator=True)
        rmses = -result["test_score"]
        assert np.isfinite(rmses).all()
        cv_scores[name] = rmses.mean()
        # Comparer la stabilité dans les unités originales, car chaque fold a son scaler.
        raw_coefficients = np.array([p.named_steps["model"].coef_ / p.named_steps["scaler"].scale_
                                      for p in result["estimator"]])
        pipeline.fit(X_train, y_train)
        coefficients = pipeline.named_steps["model"].coef_
        rows.append({"model": name, "RMSE_CV": rmses.mean(), "std_RMSE_CV": rmses.std(),
                     "nonzero": np.count_nonzero(coefficients),
                     "mean_coefficient_std_across_folds": raw_coefficients.std(axis=0).mean()})
    print(pd.DataFrame(rows).to_string(index=False))
    chosen = min(cv_scores, key=cv_scores.get)
    print("Modèle choisi sur CV train avant ouverture du test:", chosen)
    # Audit final de configurations fixées ; ne pas changer le choix à partir de ce tableau.
    audit = []
    for name, pipeline in models.items():
        pred = pipeline.predict(X_test)
        audit.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                      "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    print(pd.DataFrame(audit).to_string(index=False))
    if args.plot:
        import matplotlib.pyplot as plt
        for name, pipeline in models.items():
            plt.plot(pipeline.named_steps["model"].coef_, marker="o", label=name)
        plt.xlabel("Indice de la variable")
        plt.ylabel("Coefficient sur variables standardisées")
        plt.legend()
        plt.tight_layout()
        plt.show()
    print("Les alpha des modèles ont des conventions différentes. Le test ne sert pas à les régler.")
~~~

Questions d'analyse
1. Quel modèle obtient la plus faible RMSE ?
2. Quel modèle conserve le moins de coefficients ?
3. Ridge améliore-t-il la stabilité des coefficients ?
4. Lasso élimine-t-il certaines variables ?
5. Elastic Net représente-t-il un bon compromis ?
6. Les différences observées sont-elles suffisamment importantes pour être considérées comme utiles ?
Les résultats dépendent des données et des hyperparamètres. Il ne faut pas supposer à l'avance que Ridge ou Lasso sera toujours meilleur que la régression ordinaire.



## 21. Visualiser les coefficients

~~~python
import matplotlib.pyplot as plt
import numpy as np
# models contient les pipelines déjà ajustés sur le même train.
plt.figure(figsize=(12, 6))
for name, pipeline in models.items():
    coefficients = pipeline.named_steps["model"].coef_
    plt.plot(np.arange(len(coefficients)), coefficients, marker="o", label=name)
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Feature index")
plt.ylabel("Coefficient sur variables standardisées")
plt.title("Coefficient Comparison")
plt.legend()
plt.show()
~~~




Observe les différences entre les coefficients :

- Ridge tend à réduire leur amplitude.
- Lasso peut annuler certaines valeurs.
- Elastic Net peut produire une solution intermédiaire.

Les coefficients sont ici associés à des variables standardisées.

Partie VIII — Coding Challenge



## 22. Régularisation et sélection automatique

Challenge 02 — Regularization Benchmark

+450 XP

Trouve les meilleurs hyperparamètres de Ridge, Lasso et Elastic Net avec cross-validation, puis compare leurs performances sur un test indépendant.
~~~python
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ch03_cours02_lab import X_train, X_test, y_train, y_test

pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
search = GridSearchCV(pipeline, {"model__alpha": np.logspace(-4, 4, 30)},
                      scoring="neg_root_mean_squared_error",
                      cv=KFold(n_splits=5, shuffle=True, random_state=42))
# TODO 1 : ajuster sur X_train, y_train ; afficher alpha et RMSE CV positive.
# TODO 2 : répéter pour Lasso et ElasticNet, avec des grilles adaptées à chaque objectif.
# TODO 3 : figer le choix selon les scores CV, puis réaliser l'audit final sur test.
# TODO 4 : comparer les coefficients et leur nombre de zéros.
# BONUS : refaire la sélection avec 100 observations train seulement.
# BONUS : tracer les coefficients selon alpha, sans utiliser les résultats du test.
~~~




Bonus avancé
Analyse comment les coefficients évoluent lorsque alpha augmente.

Cette représentation s'appelle un regularization path.

Elle permet d'observer quelles variables sont éliminées en premier par Lasso et comment Ridge contracte les coefficients.



[[solution]]

## Questions d’entretien — Régularisation

[[interview]]




## 23. Mini-quiz interactif


[[quiz]]




## 24. Validation du cours


[[assessment]]

À retenir : Ridge, Lasso et Elastic Net ne remplacent pas la régression linéaire : ils ajoutent une pénalité pour contrôler ses coefficients et améliorer potentiellement sa généralisation.

- Ridge privilégie la stabilité.
- Lasso peut sélectionner automatiquement les variables.
- Elastic Net combine les deux approches.

Le prochain cours portera sur la régression polynomiale, qui permet de modéliser des relations non linéaires tout en conservant une structure linéaire dans les coefficients.

Support du cours 3 à venir.