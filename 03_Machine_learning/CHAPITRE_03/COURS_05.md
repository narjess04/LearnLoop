Dans le cours précédent, nous avons étudié la régression logistique. Elle apprend des coefficients qui permettent de calculer une probabilité.

Aujourd'hui, nous découvrons une approche complètement différente.

KNN ne cherche pas à apprendre une équation globale. Il prédit en examinant les observations les plus proches d'un nouvel exemple.

C'est un algorithme simple à comprendre, mais qui introduit plusieurs concepts essentiels du Machine Learning : notion de distance, voisinage, sensibilité à l'échelle, biais-variance et dimensionnalité.

## Objectifs du cours

À la fin, tu devras savoir :

- Expliquer le fonctionnement de KNN en classification et en régression.
- Calculer les distances euclidienne, Manhattan et Minkowski.
- Comprendre le rôle de l'hyperparamètre n_neighbors.
- Distinguer vote uniforme et vote pondéré par la distance.
- Comprendre les frontières de décision.
- Expliquer pourquoi le scaling est essentiel.
- Comprendre la malédiction de la dimension.
- Optimiser KNN avec un pipeline et une validation croisée.
- Comparer KNN à la régression logistique.

Partie I — Comprendre KNN


## 1. Qu'est-ce que K-Nearest Neighbors ?

KNN signifie K-Nearest Neighbors, ou K plus proches voisins.

Son principe est :
Pour prédire la classe d'une nouvelle observation, nous recherchons les K observations d'entraînement les plus proches et utilisons leurs classes pour prendre une décision.

Prenons un exemple.

Nous voulons prédire si un client présente un risque de défaut.

Nous disposons de deux variables :

- Ratio d'endettement.
- Revenu mensuel.

Les clients historiques sont représentés par des points.
Intuition : classer un nouveau client

Non-défaut


Défaut


Client à prédire


Schéma conceptuel. La proximité réelle dépend de la métrique de distance et de l'échelle des variables.
KNN repose sur l'hypothèse que des observations proches dans l'espace des features ont tendance à avoir des cibles similaires.

Cette hypothèse n'est pas toujours vraie : elle dépend de la qualité des variables et de la manière dont nous mesurons la proximité.



## 2. Les étapes de l'algorithme

Pour une nouvelle observation \(x\) :
1. Nouvelle observation \(x\)


2. Calculer les distances aux observations d'entraînement


3. Sélectionner les K voisins les plus proches


4. Agréger leurs classes ou leurs valeurs


5. Produire la prédiction


Exemple
Supposons que les cinq voisins les plus proches soient :

Voisin	Classe
1	Défaut
2	Non-défaut
3	Défaut
4	Défaut
5	Non-défaut

Avec \(K=5\) :
\[
P_{\mathrm{KNN}}(Y=1\mid x)=\frac35=0.6
\]

La classe prédite est donc 1 si nous utilisons un seuil de 0,5.

Cette valeur de 0,6 représente la proportion de voisins positifs, pas nécessairement une probabilité bien calibrée.

Partie II — Les distances



## 3. Pourquoi la distance est-elle essentielle ?

KNN décide quels exemples sont similaires en calculant une distance.

Deux observations proches selon une métrique peuvent être éloignées selon une autre.

Le choix de la métrique influence donc directement les voisins sélectionnés.



## 4. Distance euclidienne — L2

La distance euclidienne correspond à la distance géométrique directe entre deux points.

Pour deux observations :
\[
x=(x_1,\ldots,x_p)
\]

\[
z=(z_1,\ldots,z_p)
\]

La distance est :
\[
\boxed{
d_2(x,z)=\sqrt{\sum_{j=1}^{p}(x_j-z_j)^2}
}
\]

Exemple
Prenons :
\[
A=(2,3)
\]

\[
B=(5,7)
\]

Alors :
\[
d(A,B)=\sqrt{(5-2)^2+(7-3)^2}
\]

\[
d(A,B)=\sqrt{9+16}=5
\]




## 5. Distance Manhattan — L1

La distance Manhattan additionne les différences absolues entre les coordonnées.
\[
\boxed{
d_1(x,z)=\sum_{j=1}^{p}|x_j-z_j|
}
\]

Avec les mêmes points :
\[
d_1(A,B)=|5-2|+|7-3|
\]

\[
d_1(A,B)=3+4=7
\]

Intuition géométrique
Euclidienne

Chemin direct entre A et B.


Manhattan

Somme des déplacements sur les axes.






## 6. Distance Minkowski

La distance Minkowski généralise les distances L1 et L2.
\[
\boxed{
d_p(x,z)=
\left(\sum_{j=1}^{m}|x_j-z_j|^p\right)^{1/p}
}
\]

Attention : ici, \(p\) désigne le paramètre de la distance et \(m\) le nombre de features.

- \(p=1\) : Manhattan.
- \(p=2\) : Euclidienne.
- \(p>2\) : les grandes différences entre coordonnées pèsent davantage.

Pour que cette expression définisse une métrique au sens usuel, on utilise \(p\ge1\).



## 7. Calculateur interactif de distances


[[knndistances]]




## 8. Qu'est-ce que K ?

K représente le nombre de voisins utilisés pour effectuer la prédiction.

Par exemple :

- \(K=1\) : le voisin le plus proche.
- \(K=3\) : les trois voisins les plus proches.
- \(K=5\) : les cinq voisins les plus proches.
- \(K=15\) : les quinze voisins les plus proches.

Le choix de K est l'un des hyperparamètres les plus importants.



## 9. Visualisation interactive — Changer K


[[knnneighbors]]




## 10. K petit versus K grand

K = 1
Le modèle se base sur un seul voisin.

Avantages :

- Très flexible.
- Peut représenter des frontières complexes.
- Erreur d'entraînement souvent très faible.

Inconvénients :

- Sensible au bruit.
- Sensible aux observations atypiques.
- Risque de variance élevée.

K élevé
Le modèle utilise davantage d'observations.

Avantages :

- Prédictions généralement plus lisses.
- Moins sensible à un voisin isolé.

Inconvénients :

- Peut ignorer des structures locales.
- Peut sous-apprendre.
- Peut favoriser la classe majoritaire.

Relation avec le biais-variance

K	Biais	Variance	Risque
Très petit	Faible	Élevée	Overfitting
Modéré	Compromis	Compromis	À sélectionner
Très grand	Élevé	Plus faible	Underfitting

Ces tendances sont générales, pas des garanties pour tous les datasets.



## 11. K doit-il toujours être impair ?

Non.

On recommande parfois un K impair en classification binaire pour réduire le risque d'égalité lors d'un vote uniforme.

Mais ce n'est pas une obligation.

En classification multiclasse, une égalité peut survenir même avec un K impair.

En pratique, K doit être choisi par validation croisée, et non uniquement selon sa parité.

Partie IV — Vote uniforme et vote pondéré



## 12. Vote uniforme

Avec :
~~~python
weights="uniform"
~~~




Chaque voisin possède le même poids.

Si les classes des cinq voisins sont :
\[
[1,1,0,1,0]
\]

La proportion de classe 1 est :
\[
\frac35=0.6
\]




## 13. Vote pondéré par la distance

Avec :
~~~python
weights="distance"
~~~




Les voisins les plus proches ont davantage d'influence.

Dans scikit-learn, les poids sont proportionnels à l'inverse de la distance, avec un traitement particulier lorsque certaines distances sont nulles.

L'idée générale est :
\[
w_i\propto\frac1{d_i}
\]

La proportion pondérée de la classe 1 devient :
\[
\boxed{
\hat p_1=
\frac{\sum_{i\in N_K(x)}w_i\mathbf{1}(y_i=1)}
{\sum_{i\in N_K(x)}w_i}
}
\]

Exemple

Voisin	Distance	Classe
A	0,1	1
B	0,8	0
C	0,9	0

Avec un vote uniforme :

- Classe 0 : deux votes.
- Classe 1 : un vote.

Prédiction : classe 0.

Mais avec une pondération inverse de la distance :

- A reçoit un poids de 10.
- B reçoit un poids de 1,25.
- C reçoit un poids d'environ 1,11.

La classe 1 devient majoritaire en poids.

La prédiction peut donc changer.
Quel mode choisir ?
Le vote pondéré peut être intéressant lorsque les voisins très proches sont plus informatifs que les voisins plus éloignés.

Mais il peut aussi rendre le modèle plus sensible à certaines observations très proches et bruitées.

Il faut comparer les deux approches par validation croisée.

Partie V — Pourquoi le scaling est indispensable


[[knnweighted]]




## 14. Le problème des échelles différentes

Prenons deux variables :

- Âge : entre 18 et 80.
- Revenu annuel : entre 10 000 et 150 000.

Considérons deux clients :

Variable	Client A	Client B
Âge	25	35
Revenu	30 000	31 000

La distance euclidienne brute vaut :
\[
d=\sqrt{(35-25)^2+(31000-30000)^2}
\]

\[
d=\sqrt{100+1\,000\,000}
\]

\[
d\approx1000.05
\]

La différence d'âge contribue très peu à la distance par rapport à celle du revenu.

Cela ne signifie pas que le revenu est nécessairement plus important pour prédire la cible.

C'est simplement une conséquence de l'échelle numérique.



## 15. StandardScaler

Nous pouvons standardiser chaque variable :
\[
\boxed{
z=\frac{x-\mu}{\sigma}
}
\]

Où :

- \(\mu\) est la moyenne de la variable sur le train.
- \(\sigma\) est son écart-type sur le train.

Après transformation, les variables ont généralement une moyenne proche de 0 et une variance proche de 1 sur le train.

Cela rend leurs contributions aux distances plus comparables.
Pipeline recommandé
\[
X
\longrightarrow
StandardScaler
\longrightarrow
KNN
\]

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
~~~




Le scaler doit être ajusté uniquement sur les données d'entraînement.

Un pipeline garantit que les transformations sont réapprises correctement dans chaque fold de cross-validation.



## 16. StandardScaler, MinMaxScaler ou RobustScaler ?

Méthode	Intérêt
StandardScaler	Bon choix général pour KNN
MinMaxScaler	Ramène les valeurs dans un intervalle défini sur le train
RobustScaler	Moins influencé par certaines valeurs extrêmes

Il n'existe pas de scaler universellement optimal.

Le choix dépend de la distribution des variables et de leur pertinence métier.

Partie VI — KNN pour la classification



## 17. KNeighborsClassifier

En Python :
~~~python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
~~~




Les hyperparamètres essentiels

Paramètre	Rôle
n_neighbors	Nombre de voisins K
weights	Vote uniforme ou pondéré
metric	Métrique de distance
p	Paramètre Minkowski
algorithm	Méthode de recherche des voisins
leaf_size	Paramètre des structures BallTree / KDTree

Avec :
~~~python
KNeighborsClassifier(metric="minkowski", p=2)
~~~




Nous utilisons la distance euclidienne.

Avec :
~~~python
KNeighborsClassifier(metric="minkowski", p=1)
~~~




Nous utilisons Manhattan.



## 18. Exemple complet avec un dataset de classification

~~~python
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
                           weights=[.85, .15], class_sep=1.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def make_classifier():
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2))])


def classification_lab():
    pipeline = make_classifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X_train, y_train, cv=cv,
                             scoring={"ap": "average_precision", "roc_auc": "roc_auc", "recall": "recall"})
    print("Classification CV train :", {key: results["test_" + key].mean() for key in ["ap", "roc_auc", "recall"]})
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    assert np.isfinite(proba).all() and np.all((proba >= 0) & (proba <= 1))
    print("Test final, configuration fixe : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    assert np.allclose(pipeline.named_steps["scaler"].mean_, X_train.mean(axis=0))
    return pipeline, proba


pipeline, y_proba = classification_lab()
~~~




Que faut-il observer ?
Le dataset contient une classe minoritaire d'environ 15 %.

Il faut donc examiner :

- Le recall de la classe positive.
- La précision.
- Le F1-score.
- La ROC-AUC.
- Éventuellement l'Average Precision.
- La calibration si les scores sont utilisés comme probabilités.

KNN ne garantit pas une bonne calibration probabiliste.

Partie VII — KNN pour la régression



## 19. KNeighborsRegressor

KNN n'est pas limité à la classification.

Il peut également prédire une valeur continue.

Dans ce cas, il utilise les valeurs des voisins plutôt que leurs classes.
Moyenne uniforme
Pour les K voisins :
\[
\boxed{
\hat y(x)=\frac1K\sum_{i\in N_K(x)}y_i
}
\]

Exemple
Supposons que les trois appartements les plus proches d'un nouveau bien valent :
\[
[200,220,250]
\]

La prédiction KNN est :
\[
\hat y=\frac{200+220+250}{3}
\]

\[
\hat y=223.33
\]

Le prix prédit est d'environ 223,33 milliers de DT.



## 20. Moyenne pondérée

Avec des poids \(w_i\) :
\[
\boxed{
\hat y(x)=
\frac{\sum_{i\in N_K(x)}w_iy_i}
{\sum_{i\in N_K(x)}w_i}
}
\]

Les voisins les plus proches peuvent ainsi contribuer davantage à la prédiction.
Exemple Python
~~~python
def regression_lab():
    # Noms distincts : ne pas remplacer les données du problème de classification.
    X_reg, y_reg = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=.2, random_state=42)
    regressor = Pipeline([("scaler", StandardScaler()),
                          ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance"))])
    scores = cross_validate(regressor, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    print("Régression RMSE CV train :", -scores["test_score"].mean())
    regressor.fit(X_reg_train, y_reg_train)
    prediction = regressor.predict(X_reg_test)
    assert np.all(prediction >= y_reg_train.min()) and np.all(prediction <= y_reg_train.max())
    print("Régression test fixe : MAE", mean_absolute_error(y_reg_test, prediction),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, prediction)), "R2", r2_score(y_reg_test, prediction))
    # Avec des poids positifs, la prédiction reste entre les cibles des voisins retenus.
    transformed = regressor.named_steps["scaler"].transform(X_reg_test)
    indices = regressor.named_steps["knn"].kneighbors(transformed, return_distance=False)
    neighborhood_targets = y_reg_train[indices]
    assert np.all(prediction >= neighborhood_targets.min(axis=1) - 1e-10)
    assert np.all(prediction <= neighborhood_targets.max(axis=1) + 1e-10)
    return y_reg_test, prediction


y_reg_test, y_reg_pred = regression_lab()
~~~




Limite importante
KNN Regression effectue essentiellement une interpolation locale des observations d'entraînement.

Avec des poids positifs, ses prédictions restent dans l'intervalle des valeurs cibles des voisins utilisés.

Il ne réalise donc pas une extrapolation linéaire naturelle comme une régression linéaire.

Partie VIII — La malédiction de la dimension



## 21. Curse of Dimensionality

C'est l'une des limites les plus importantes de KNN.

Lorsque le nombre de dimensions augmente, les observations deviennent souvent plus dispersées dans l'espace.

La notion de proximité peut alors perdre une partie de son pouvoir informatif.
Exemple intuitif
En une dimension, nous cherchons des voisins sur une ligne.

En deux dimensions, nous cherchons des voisins dans un plan.

En trois dimensions, nous cherchons des voisins dans un volume.

En 100 dimensions, nous cherchons des voisins dans un espace beaucoup plus complexe.

Le volume d'un hypercube de côté \(r\) dans un espace de dimension \(d\) vaut :
\[
V=r^d
\]

Si nous voulons couvrir une fraction \(q\) du volume d'un hypercube unité avec un sous-cube de même longueur sur chaque axe :
\[
r=q^{1/d}
\]

Lorsque \(d\) augmente, \(r\) se rapproche de 1.

Cela illustre pourquoi il devient difficile de définir un petit voisinage contenant suffisamment d'observations.



## 22. Explorer l'effet de la dimension


[[knndimension]]




## 23. Comment limiter ce problème ?

Plusieurs stratégies sont possibles :
1. Feature Selection : conserver les variables informatives.
2. PCA : réduire le nombre de dimensions.
3. Feature Engineering : construire des variables plus pertinentes.
4. Choix de la distance : comparer plusieurs métriques.
5. Augmenter les données lorsque c'est possible et pertinent.
6. Comparer avec d'autres modèles moins sensibles à la dimension.
Exemple avec PCA
~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=5)),
                     ("knn", KNeighborsClassifier(n_neighbors=7))])
~~~




Le nombre de composantes PCA doit lui aussi être sélectionné par validation.

Partie IX — Complexité de KNN



## 24. Lazy Learning

KNN est souvent qualifié d'algorithme de lazy learning.

Contrairement à la régression logistique, il ne cherche pas à estimer un vecteur global de coefficients pendant le fit.

Il conserve essentiellement les observations d'entraînement nécessaires à la recherche des voisins.
Conséquences

Aspect	KNN
Entraînement	Souvent relativement rapide
Prédiction	Peut être coûteuse
Mémoire	Peut être importante
Interprétabilité locale	Bonne : voisins consultables
Sensibilité au scaling	Élevée
Haute dimension	Souvent problématique

Complexité
Avec une recherche brute :

- \(n\) : nombre d'observations d'entraînement.
- \(d\) : nombre de features.

Une recherche de voisins pour une nouvelle observation nécessite typiquement de l'ordre de :
\[
O(nd)
\]

pour calculer toutes les distances.

La sélection des K voisins ajoute un coût supplémentaire qui dépend de l'algorithme utilisé.

Scikit-learn peut utiliser différentes stratégies :

- brute
- kd_tree
- ball_tree
- auto

Les structures arborescentes peuvent accélérer certaines recherches, surtout en faible dimension, mais ne sont pas toujours efficaces lorsque la dimension augmente.

Partie X — Hyperparameter Tuning



## 25. Quels hyperparamètres optimiser ?

Les plus importants sont :

Hyperparamètre	Valeurs possibles
n_neighbors	3, 5, 7, 11, 15, 21
weights	uniform, distance
p	1, 2
Scaler	StandardScaler, MinMaxScaler, RobustScaler

Nous pouvons comparer toutes ces configurations avec GridSearchCV.



## 26. Pipeline complet avec GridSearchCV

~~~python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
param_grid = {"scaler": [StandardScaler(), MinMaxScaler(), RobustScaler()],
              "knn__n_neighbors": [3, 5, 7, 11, 15, 21],
              "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="average_precision")
# Utiliser les données de CLASSIFICATION, pas celles de la régression.
search.fit(X_train, y_train)
print(search.best_params_)
print("AP de sélection CV :", search.best_score_)
# Figer les choix avant l’audit test ; AP ne mesure pas directement la calibration.
~~~




Ce code suppose que X_train et y_train correspondent à un problème de classification.
Pourquoi utiliser un pipeline ?
Parce que le scaler doit être ajusté séparément sur chaque fold d'entraînement.

Sans pipeline, une standardisation réalisée sur l'ensemble du dataset avant la validation croisée provoquerait une fuite d'information.

Partie XI — Comparaison avec la régression logistique



## 27. KNN versus Logistic Regression

Critère	KNN	Logistic Regression
Principe	Voisinage local	Modèle paramétrique
Apprend des coefficients globaux	Non	Oui
Frontière de décision	Potentiellement complexe	Linéaire dans les features
Sensibilité au scaling	Forte	Importante surtout avec régularisation
Coût de prédiction	Potentiellement élevé	Généralement faible
Interprétation	Par les voisins	Par les coefficients et Odds Ratios
Probabilités	Fréquences locales pondérées ou non	Sigmoid des scores
Haute dimension	Souvent difficile	Peut mieux se comporter avec régularisation

Quand utiliser KNN ?
KNN peut être intéressant lorsque :

- Le dataset est de taille modérée.
- Le nombre de features est raisonnable.
- Les observations similaires ont des cibles similaires.
- Les frontières de décision sont complexes.
- Nous voulons une baseline non paramétrique.

Quand préférer Logistic Regression ?
La régression logistique est souvent préférable lorsque :

- Nous voulons interpréter les effets des variables.
- Nous avons beaucoup d'observations.
- Nous avons besoin d'une prédiction rapide.
- Nous voulons une baseline probabiliste simple.
- Le problème peut être raisonnablement modélisé par un logit linéaire.

Partie XII — Coding Challenge



## 28. Challenge : KNN Classification Benchmark

Challenge 05 — KNN Benchmark

+500 XP

Construis un classifieur KNN complet, optimise ses hyperparamètres et compare ses performances à celles d'une régression logistique.
~~~python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8, n_redundant=5,
                           weights=[.9, .1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
# TODO : comparer K=5 sans scaling et avec StandardScaler sur les mêmes folds train.
# TODO : optimiser K, weights, p et le scaler avec GridSearchCV.
# TODO : comparer à une régression logistique sous le même protocole.
# TODO : fixer le choix avant le test ; examiner AP, ROC-AUC, recall, F1 et Brier.
# BONUS : inclure PCA dans le pipeline et sélectionner ses composantes sur train.
~~~




Questions d'analyse
1. Quel est l'effet du scaling sur KNN ?
2. Pourquoi K=1 peut-il produire une très faible erreur train ?
3. Pourquoi K=100 peut-il sous-apprendre ?
4. Le vote pondéré améliore-t-il toujours les résultats ?
5. Comment la PCA influence-t-elle les performances ?
6. Quel modèle est le plus rapide en prédiction ?
7. Quel modèle est le plus adapté à un dataset très volumineux ?



[[solution]]

## Questions d’entretien — KNN

[[interview]]




## 29. Mini-quiz interactif


[[quiz]]




## 30. Validation du cours


[[assessment]]

À retenir : KNN est un algorithme fondé sur la proximité. Il peut être performant lorsque des observations proches ont des cibles similaires, mais il dépend fortement du scaling, de la métrique de distance, du nombre de voisins et de la dimension des données.

Au prochain cours, nous étudierons les Decision Trees, qui apprennent des règles de décision successives au lieu de comparer directement les distances entre observations.

Le support détaillé du cours 6 sera ajouté à sa réception.