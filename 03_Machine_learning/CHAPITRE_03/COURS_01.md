Nous avons terminé les fondements du Machine Learning et le Data Preprocessing. Nous allons maintenant étudier les algorithmes de Machine Learning supervisé, en comprenant non seulement comment les utiliser, mais surtout comment ils fonctionnent mathématiquement.

Nous commençons par l'algorithme le plus fondamental : la régression linéaire.


## 1. Qu'est-ce que la régression ?

En Machine Learning supervisé, nous disposons d'observations contenant des variables explicatives \(X\) et une cible \(y\).

Nous cherchons à apprendre une fonction :
\[
\hat y=f(X)
\]

La régression sert à prédire une valeur numérique.

Quelques exemples :

Problème	Variables \(X\)	Cible \(y\)
Prix immobilier	Surface, quartier, âge du bien	Prix
Prévision commerciale	Publicité, saison, promotions	Chiffre d'affaires
Consommation électrique	Température, heure, activité	kWh
Risque bancaire	Revenu, dette, ancienneté	Montant de perte
Logistique	Distance, volume, trafic	Temps de livraison

Régression versus classification
La différence principale réside dans la nature de la sortie.
Régression

Prédire une valeur continue : prix, montant, durée.


Classification

Prédire une classe : défaut / non-défaut, fraude / non-fraude.



Point important : malgré son nom, la régression logistique est généralement utilisée pour la classification. Nous l'étudierons au cours 4.

Partie II — Régression linéaire simple



## 2. L'idée fondamentale

Supposons que nous souhaitions prédire le prix d'un appartement à partir de sa surface.

Nous disposons des observations suivantes :

Surface (m²)	Prix (milliers de DT)
50	150
70	190
90	230
110	270
130	310

Nous remarquons que le prix augmente avec la surface.

La régression linéaire cherche à représenter cette relation par une droite.

Son équation est :
\[
\boxed{\hat y=\beta_0+\beta_1x}
\]

Où :

- \(x\) : variable explicative, ici la surface.
- \(\hat y\) : valeur prédite, ici le prix.
- \(\beta_0\) : intercept ou ordonnée à l'origine.
- \(\beta_1\) : coefficient directeur, ou pente.

Exemple numérique
Imaginons que le modèle apprenne :
\[
\hat y=50+2x
\]

Pour un appartement de 100 m² :
\[
\hat y=50+2(100)=250
\]

Le prix prédit est donc de 250 milliers de DT.
Interprétation des coefficients
\(\beta_0=50\) : prix théorique prédit lorsque la surface vaut zéro. Cette interprétation peut ne pas avoir de sens immobilier réel.

\(\beta_1=2\) : dans ce modèle, chaque mètre carré supplémentaire est associé à une augmentation prédite de 2 milliers de DT.

Il s'agit d'une association estimée, pas automatiquement d'une relation causale.



## 3. Visualisation interactive — Construire une droite


[[linearfit]]

Modifier la pente change l’inclinaison ; modifier l’intercept déplace la droite. Chercher les coefficients minimisant les erreurs.




## 4. Les résidus

Pour une observation \(i\), le résidu est :
\[
e_i=y_i-\hat y_i
\]

C'est la différence entre la valeur observée et la valeur prédite.

Exemple :
\[
y_i=230,\quad\hat y_i=220
\]

Alors :
\[
e_i=10
\]

Le modèle sous-estime la valeur de 10 unités.

Un résidu positif signifie que la valeur observée est supérieure à la prédiction.

Un résidu négatif signifie que la prédiction est supérieure à la valeur observée.



## 5. Mean Squared Error — MSE

La régression linéaire ordinaire utilise les moindres carrés pour estimer ses coefficients.

La fonction de coût MSE est :
\[
\boxed{MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

Pourquoi élever les erreurs au carré ?
1. Éviter que les erreurs positives et négatives s'annulent.
2. Pénaliser davantage les grandes erreurs.
3. Obtenir une fonction mathématiquement pratique à optimiser.
Exemple
Supposons trois erreurs :
\[
e=[2,-3,5]
\]

Alors :
\[
MSE=\frac{2^2+(-3)^2+5^2}{3}
\]

\[
MSE=\frac{4+9+25}{3}\approx12.67
\]

L'objectif est de minimiser cette quantité.



## 6. Comment trouver les meilleurs coefficients ?

Il existe deux approches fondamentales.
Approche A — Solution analytique
Dans la régression linéaire simple, la pente estimée par moindres carrés est :
\[
\boxed{\hat\beta_1=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2}}
\]

Et l'intercept :
\[
\boxed{\hat\beta_0=\bar y-\hat\beta_1\bar x}
\]

Cette solution existe lorsque les valeurs de \(x\) ne sont pas toutes identiques.
Approche B — Descente de gradient
Nous pouvons également minimiser progressivement la fonction de coût.

Les mises à jour sont :
\[
\beta_j^{(t+1)}
=
\beta_j^{(t)}
-
\alpha\frac{\partial J}{\partial\beta_j}
\]

Où :

- \(\alpha\) est le learning rate.
- \(J\) est la fonction de coût.
- \(\frac{\partial J}{\partial\beta_j}\) est le gradient.

Pour :
\[
J=\frac{1}{2n}\sum_i(y_i-\hat y_i)^2
\]

Les gradients sont :
\[
\frac{\partial J}{\partial\beta_0}
=
-\frac{1}{n}\sum_i(y_i-\hat y_i)
\]

\[
\frac{\partial J}{\partial\beta_1}
=
-\frac{1}{n}\sum_i x_i(y_i-\hat y_i)
\]

La descente de gradient cherche à diminuer progressivement l'erreur.
Attention
LinearRegression de scikit-learn utilise des méthodes numériques de moindres carrés, pas une boucle de descente de gradient classique.

La descente de gradient est néanmoins essentielle à comprendre pour la suite du Machine Learning et du Deep Learning.



## 7. Visualisation — Fonction de coût


[[linearcost]]

Avec la MSE, le coût est convexe dans les coefficients. Une colinéarité parfaite peut donner plusieurs solutions optimales avec les mêmes prédictions.




## 8. Passer d'une variable à plusieurs

Dans la réalité, une seule variable suffit rarement à expliquer un phénomène.

Pour prédire le prix d'un appartement, nous pouvons utiliser :

- Surface.
- Nombre de chambres.
- Âge du bâtiment.
- Distance du centre-ville.

Le modèle devient :
\[
\boxed{\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p}
\]

Exemple
\[
\hat y=40+1.8\,Surface+12\,Chambres-0.5\,Age
\]

Pour un appartement de 100 m², trois chambres et dix ans :
\[
\hat y=40+1.8(100)+12(3)-0.5(10)
\]

\[
\hat y=251
\]

Le prix prédit est de 251 milliers de DT.
Interpréter les coefficients
Le coefficient de la surface vaut 1,8.

Il signifie que, toutes les autres variables du modèle étant maintenues constantes, une augmentation de 1 m² est associée à une augmentation prédite de 1,8 millier de DT.

Cette précision est essentielle : en régression multiple, chaque coefficient représente une association conditionnelle aux autres variables du modèle.



## 9. Écriture matricielle

Pour \(n\) observations et \(p\) variables, nous pouvons écrire :
\[
\hat{\mathbf y}=X\boldsymbol\beta
\]

En incluant une colonne de 1 pour l'intercept :
\[
X=
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1p}\\
1 & x_{21} & \cdots & x_{2p}\\
\vdots & \vdots & \ddots & \vdots\\
1 & x_{n1} & \cdots & x_{np}
\end{bmatrix}
\]

Et :
\[
\boldsymbol\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{bmatrix}
\]

Nous cherchons à minimiser :
\[
\min_{\boldsymbol\beta}\|X\boldsymbol\beta-\mathbf y\|_2^2
\]

Lorsque \(X^TX\) est inversible, la solution est :
\[
\boxed{\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf y}
\]

En pratique, on privilégie des méthodes numériques plus stables, notamment les décompositions matricielles et les solveurs de moindres carrés.

Partie V — Les hypothèses de la régression linéaire



## 10. Quelles hypothèses faut-il vérifier ?

Il faut distinguer deux objectifs :

- Prédiction : obtenir de bonnes prédictions sur de nouvelles observations.
- Inférence : interpréter les coefficients et réaliser des tests statistiques.

Les hypothèses classiques sont particulièrement importantes pour l'inférence.

### 10.1 Linéarité dans les coefficients

Le modèle est linéaire par rapport aux coefficients :
\[
y=\beta_0+\beta_1x+\varepsilon
\]

Une relation peut être non linéaire en \(x\) tout en restant linéaire dans les coefficients, par exemple :
\[
y=\beta_0+\beta_1x+\beta_2x^2+\varepsilon
\]

Nous étudierons cette situation dans le cours sur la régression polynomiale.

### 10.2 Exogénéité

Pour une interprétation statistique classique des coefficients, on suppose notamment :
\[
E[\varepsilon\mid X]=0
\]

Cela signifie que l'erreur a une espérance conditionnelle nulle compte tenu des variables explicatives.

Une variable importante omise et corrélée avec les variables du modèle peut violer cette hypothèse.

### 10.3 Indépendance des erreurs

Les erreurs ne doivent pas présenter de dépendances problématiques non prises en compte.

Cette question est particulièrement importante dans les séries temporelles ou les données regroupées par client.

### 10.4 Homoscédasticité

L'homoscédasticité signifie que la variance conditionnelle des erreurs est constante :
\[
Var(\varepsilon\mid X)=\sigma^2
\]

À l'inverse, l'hétéroscédasticité apparaît lorsque cette variance dépend des variables explicatives.
Homoscédasticité

Dispersion approximativement constante.


Hétéroscédasticité

Dispersion qui augmente avec X.




### 10.5 Absence de multicolinéarité parfaite

Les variables explicatives ne doivent pas être des combinaisons linéaires exactes les unes des autres si l'on veut identifier de façon unique tous les coefficients.

Une forte multicolinéarité peut rendre les coefficients instables.

### 10.6 Normalité des erreurs

La normalité des erreurs n'est pas nécessaire pour calculer les coefficients par moindres carrés.

Elle intervient surtout dans certaines procédures d'inférence exactes sur petits échantillons.

Il ne faut donc pas affirmer qu'une régression linéaire est automatiquement invalide dès que les résidus ne sont pas normalement distribués.



## 11. Diagnostiquer les résidus

Un bon diagnostic consiste à tracer :
\[
e_i=y_i-\hat y_i
\]

en fonction des prédictions.

Nous cherchons notamment :

- Des motifs non linéaires.
- Une variance changeante.
- Des valeurs aberrantes.
- Des observations influentes.
~~~python
import matplotlib.pyplot as plt
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=.6)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals y - y_pred")
plt.title("Residual Plot")
plt.show()
~~~




Un nuage sans structure apparente est généralement plus rassurant qu'un motif systématique, mais ce graphique ne prouve pas à lui seul que toutes les hypothèses sont satisfaites.

Partie VI — Évaluer un modèle de régression



## 12. Les quatre métriques essentielles


### 12.1 MAE — Mean Absolute Error

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

La MAE mesure l'erreur absolue moyenne.

Elle s'exprime dans la même unité que la cible.

### 12.2 MSE — Mean Squared Error

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Elle pénalise davantage les grandes erreurs.

### 12.3 RMSE — Root Mean Squared Error

\[
RMSE=\sqrt{MSE}
\]

Elle s'exprime dans la même unité que la cible, contrairement à la MSE.

### 12.4 R² — Coefficient de détermination

\[
\boxed{R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}}
\]

Le R² compare les erreurs du modèle à celles d'un prédicteur constant égal à la moyenne de la cible sur l'échantillon évalué.

- \(R^2=1\) : prédictions parfaites.
- \(R^2=0\) : même somme des erreurs quadratiques que cette référence.
- \(R^2<0\) : performance inférieure à cette référence.

Un R² négatif est donc possible, particulièrement sur le test.



## 13. Comparaison des métriques

Métrique	Avantage	Limite
MAE	Facile à interpréter	Moins sensible aux grandes erreurs
MSE	Pénalise fortement les grandes erreurs	Unité au carré
RMSE	Même unité que la cible	Sensible aux outliers
R²	Mesure relative à une baseline	Ne représente pas une erreur métier directe

[[regressionmetrics]]

Partie VII — Implémentation Python complète



## 14. Premier modèle avec scikit-learn

Nous allons utiliser un dataset synthétique de prix immobiliers.
~~~python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
n = 1000
surface = rng.uniform(40, 200, n)
rooms = rng.integers(1, 6, n)
building_age = rng.uniform(0, 40, n)
noise = rng.normal(0, 20, n)
price = 50 + 2.0 * surface + 10 * rooms - .6 * building_age + noise
df = pd.DataFrame({"surface": surface, "rooms": rooms,
                   "building_age": building_age, "price": price})
X, y = df.drop(columns="price"), df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les diagnostics graphiques")
    args = parser.parse_args()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    for name, estimator in [("Moyenne train", DummyRegressor(strategy="mean")),
                             ("Régression linéaire", pipeline)]:
        scores = -cross_val_score(estimator, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        print(name, "RMSE CV train:", scores, "mean:", scores.mean(), "std:", scores.std())
        assert np.isfinite(scores).all()
    # Modèle fixe, évaluation finale sans réglage à partir du test.
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Intercept:", model.intercept_)
    coefficients = pd.Series(model.coef_, index=X.columns)
    print("Coefficients dans les unités originales:", coefficients)
    mse = mean_squared_error(y_test, y_pred)
    print("Test final : MAE", mean_absolute_error(y_test, y_pred),
          "MSE", mse, "RMSE", np.sqrt(mse), "R2", r2_score(y_test, y_pred))
    # Le scaling ne change pas les prédictions OLS ici (matrice de plein rang).
    pipeline.fit(X_train, y_train)
    assert np.allclose(model.predict(X_test), pipeline.predict(X_test))
    # Vérifier la formulation matricielle avec un solveur stable, sans inversion explicite.
    design = np.column_stack([np.ones(len(X_train)), X_train.to_numpy()])
    beta, _, rank, _ = np.linalg.lstsq(design, y_train.to_numpy(), rcond=None)
    assert rank == design.shape[1]
    assert np.allclose(beta, np.r_[model.intercept_, model.coef_])
    if args.plot:
        import matplotlib.pyplot as plt
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(y_test, y_pred, alpha=.6)
        limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[0].plot(limits, limits, linestyle="--", color="black")
        axes[0].set(xlabel="Prix observé", ylabel="Prix prédit")
        axes[1].scatter(y_pred, residuals, alpha=.6)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(xlabel="Prix prédit", ylabel="Résidu y - y_pred")
        coefficients.sort_values().plot(kind="barh", ax=axes[2])
        axes[2].set_title("Coefficients : unités différentes")
        fig.tight_layout()
        plt.show()
    print("Données synthétiques, prix en milliers de DT. Les coefficients sont des associations.")
~~~




Que devons-nous observer ?
Les coefficients appris devraient être proches des coefficients utilisés pour générer les données :
\[
\beta_0\approx50
\]

\[
\beta_{\mathrm{surface}}\approx2
\]

\[
\beta_{\mathrm{rooms}}\approx10
\]

\[
\beta_{\mathrm{age}}\approx-0.6
\]

Ils ne seront pas nécessairement exactement égaux, car les données contiennent du bruit et l'échantillon est fini.



## 15. Visualiser les coefficients

~~~python
import matplotlib.pyplot as plt
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.sort_values().plot(kind="barh")
plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient value")
plt.show()
~~~




Attention : les coefficients ne sont pas directement comparables en importance lorsque les variables ont des unités ou des échelles différentes.

Une variation d'une unité de surface n'est pas comparable à une variation d'une unité du nombre de chambres.

La standardisation peut aider à comparer des coefficients, sans pour autant leur donner une interprétation causale.



## 16. Construire un pipeline propre

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
scores = cross_val_score(pipeline, X_train, y_train,
                         cv=KFold(n_splits=5, shuffle=True, random_state=42),
                         scoring="neg_root_mean_squared_error")
print("CV RMSE :", -scores.mean())
~~~




La standardisation n'est pas indispensable à la régression linéaire ordinaire, mais elle sera importante pour les modèles régularisés que nous étudierons ensuite.

Partie VIII — Limites de la régression linéaire



## 17. Quand la régression linéaire fonctionne-t-elle bien ?

Elle est particulièrement intéressante lorsque :

- La relation peut être raisonnablement représentée par une combinaison linéaire de features.
- Les données sont suffisamment nombreuses et pertinentes.
- Les variables explicatives sont correctement préparées.
- L'interprétabilité est importante.
- Nous recherchons une baseline rapide.



## 18. Quand devient-elle insuffisante ?

Relations non linéaires
Si la cible dépend d'une relation complexe, une simple droite ou un hyperplan peut être insuffisant.
Interactions
L'effet d'une variable peut dépendre d'une autre.

Exemple :
\[
y=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2
\]

Multicolinéarité
Les coefficients peuvent devenir instables lorsque les variables sont fortement corrélées.
Outliers
La minimisation des erreurs quadratiques rend la régression linéaire sensible à certaines observations extrêmes.
Extrapolation
Une droite apprise sur des surfaces comprises entre 40 et 200 m² peut produire des prédictions peu fiables pour un appartement de 1 000 m².
Overfitting
Lorsque le nombre de variables est élevé par rapport au nombre d'observations, une régression linéaire peut surapprendre.

Nous étudierons comment Ridge, Lasso et Elastic Net permettent de contrôler ce problème dans le prochain cours.

Partie IX — Coding Challenge



## 19. Implémenter la régression linéaire from scratch

Challenge 01 — Linear Regression from Scratch

+400 XP

Construis une régression linéaire simple sans utiliser LinearRegression de scikit-learn, puis compare les résultats.

~~~python
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 100)
y = 3 + 2 * X + rng.normal(0, 1, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


def fit_ols(x, y):
    # TODO : vérifier les dimensions et refuser une variable x constante.
    # TODO : calculer les moyennes train.
    # TODO : pente = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)**2).
    # TODO : intercept = y_mean - pente*x_mean.
    raise NotImplementedError

# TODO : prédire sur test et calculer le MSE.
# TODO : comparer les coefficients avec sklearn.LinearRegression sur le même train.
# BONUS : implémenter la descente de gradient sur J = MSE / 2.
~~~




Questions de réflexion
1. Pourquoi la formule de la pente utilise-t-elle la covariance entre \(X\) et \(y\) ?
2. Que se passe-t-il si toutes les valeurs de \(X\) sont identiques ?
3. Pourquoi les coefficients estimés ne correspondent-ils pas exactement à 3 et 2 ?
4. Quel est l'effet d'un outlier extrême sur la pente ?
5. Pourquoi faut-il évaluer le modèle sur des observations non utilisées pendant l'entraînement ?



[[solution]]

## Questions d’entretien — Régression linéaire

[[interview]]




## 20. Mini-quiz interactif


[[quiz]]




## 21. Validation du cours


[[assessment]]

À retenir : la régression linéaire est bien plus qu'un algorithme de prédiction simple. Elle introduit les concepts fondamentaux que nous retrouverons dans presque tout le Machine Learning : fonction de coût, optimisation, coefficients, généralisation, régularisation et évaluation.

Le prochain cours approfondira la manière de contrôler les coefficients et de réduire le surapprentissage avec Ridge, Lasso et Elastic Net.

Le support détaillé du cours 2 sera ajouté à sa réception.