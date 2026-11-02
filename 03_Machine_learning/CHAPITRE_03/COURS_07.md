Dans le cours précédent, nous avons découvert les Decision Trees. Ils sont interprétables, capables de modéliser des relations non linéaires et ne nécessitent généralement pas de standardisation.

Mais ils présentent une faiblesse importante : une petite modification des données peut produire un arbre très différent.

Random Forest répond à ce problème en combinant les prédictions de nombreux arbres.

L'idée fondamentale est simple :
Au lieu de faire confiance à un seul arbre, nous entraînons plusieurs arbres différents et nous agrégeons leurs prédictions.

Cette approche appartient à une grande famille de techniques : Ensemble Learning.

## Objectifs du cours

À la fin, tu devras maîtriser :

- Ensemble Learning, Bagging et Bootstrap.
- La différence entre Bagging et Random Forest.
- La réduction de variance et le rôle de la corrélation entre arbres.
- Les votes majoritaires et les moyennes de probabilités.
- Le score Out-of-Bag (OOB).
- Les hyperparamètres de Random Forest.
- La classification et la régression.
- Les importances MDI et Permutation Importance.
- La comparaison avec Decision Tree et Gradient Boosting.
- L'optimisation avec GridSearchCV et RandomizedSearchCV.

Partie I — Ensemble Learning


## 1. Pourquoi combiner plusieurs modèles ?

Supposons que nous entraînions un arbre de décision pour prédire le défaut de paiement d'une entreprise.

Cet arbre peut produire une règle comme :
\[
\text{Ratio d'endettement}>0.45
\]

Mais si nous modifions légèrement le dataset, un nouvel arbre peut choisir :
\[
\text{Capacité de remboursement}<1.8
\]

Les deux arbres peuvent avoir appris des relations pertinentes, mais leur structure dépend fortement de l'échantillon d'entraînement.

L'Ensemble Learning consiste à combiner plusieurs modèles pour obtenir une prédiction collective.

Il existe trois grandes approches :

Méthode	Principe	Exemple
Bagging	Modèles entraînés en parallèle sur des échantillons différents	Random Forest
Boosting	Modèles ajoutés séquentiellement pour améliorer la prédiction collective	Gradient Boosting
Stacking	Un méta-modèle combine les prédictions de plusieurs modèles	StackingClassifier

Dans ce cours, nous nous concentrons sur Bagging.



## 2. L'intuition de Bagging

Bagging signifie :

Bootstrap Aggregating.

Deux étapes sont essentielles :
1. Bootstrap : créer plusieurs échantillons à partir du dataset initial.
2. Aggregating : combiner les prédictions des modèles entraînés sur ces échantillons.
Architecture du Bagging
Schéma conceptuel : les arbres sont entraînés indépendamment, puis leurs prédictions sont agrégées.

Partie II — Bootstrap



## 3. Qu'est-ce que le Bootstrap ?

Le Bootstrap est une méthode d'échantillonnage avec remise.

Supposons que notre dataset contienne cinq observations :
\[
D=[A,B,C,D,E]
\]

Nous créons un nouvel échantillon de cinq observations en tirant aléatoirement dans le dataset initial.

Par exemple :
\[
D_1=[A,C,C,E,B]
\]

Un autre échantillon pourrait être :
\[
D_2=[D,A,E,E,D]
\]

Remarque importante : certaines observations apparaissent plusieurs fois, tandis que d'autres ne sont pas sélectionnées.

C'est précisément ce qui différencie le Bootstrap d'un simple échantillonnage sans remise.



## 4. Explorateur interactif — Bootstrap


[[forestbootstrap]]




## 5. Pourquoi environ 63,2 % des observations sont-elles sélectionnées ?

Prenons un dataset de \(n\) observations.

À chaque tirage, la probabilité de ne pas sélectionner une observation précise est :
\[
1-\frac1n
\]

Après \(n\) tirages avec remise :
\[
P(\text{jamais sélectionnée})=
\left(1-\frac1n\right)^n
\]

Lorsque \(n\) devient grand :
\[
\left(1-\frac1n\right)^n\longrightarrow e^{-1}
\]

Donc :
\[
P(\text{jamais sélectionnée})\approx0.368
\]

Par conséquent :
\[
P(\text{sélectionnée au moins une fois})\approx0.632
\]

Observations uniques dans un Bootstrap
≈ 63,2 %


Observations non sélectionnées
≈ 36,8 %



Ces proportions sont des valeurs attendues approximatives pour de grands échantillons Bootstrap de taille \(n\), pas des proportions garanties à chaque tirage.

Les observations non sélectionnées sont appelées Out-of-Bag (OOB).

Nous les réutiliserons pour évaluer Random Forest.

Partie III — Bagging



## 6. Comment Bagging produit-il une prédiction ?

Nous entraînons \(B\) modèles :
\[
f_1(x),f_2(x),\ldots,f_B(x)
\]

Chaque modèle utilise un échantillon Bootstrap différent.
En régression
Nous prenons la moyenne des prédictions :
\[
\boxed{
\hat f_{\mathrm{bag}}(x)=
\frac1B\sum_{b=1}^{B}f_b(x)
}
\]

En classification
Dans le Bagging classique, on peut utiliser un vote majoritaire :
\[
\hat y=
\operatorname{mode}
\left(f_1(x),\ldots,f_B(x)\right)
\]

On peut également agréger les probabilités des modèles.

La distinction est importante : les deux méthodes ne donnent pas toujours exactement la même prédiction.



## 7. Pourquoi Bagging réduit-il la variance ?

Supposons que nous ayons plusieurs modèles produisant des prédictions variables.

La moyenne de leurs prédictions peut être plus stable que chaque prédiction individuelle.

Si les erreurs des modèles ont :

- Une variance commune \(\sigma^2\).
- Une corrélation commune \(\rho\).

Alors la variance de leur moyenne est :
\[
\boxed{
Var(\bar f)=
\sigma^2\left(
\rho+\frac{1-\rho}{B}
\right)
}
\]

Cette formule repose sur l'hypothèse de variances identiques et de corrélations deux à deux identiques.
Cas 1 — Modèles indépendants
Si :
\[
\rho=0
\]

Alors :
\[
Var(\bar f)=\frac{\sigma^2}{B}
\]

Augmenter le nombre de modèles réduit fortement la variance.
Cas 2 — Modèles très corrélés
Si :
\[
\rho=1
\]

Alors :
\[
Var(\bar f)=\sigma^2
\]

Tous les modèles se comportent de la même manière : leur moyenne ne réduit pas la variance.

Conclusion fondamentale : pour construire un ensemble performant, il ne suffit pas d'avoir beaucoup de modèles. Il faut aussi qu'ils soient suffisamment diversifiés.



## 8. Explorateur — Nombre d'arbres et corrélation


[[forestvariance]]




## 9. Pourquoi Random Forest va-t-il plus loin que Bagging ?

Bagging introduit de la diversité en utilisant des échantillons Bootstrap différents.

Mais si certaines variables sont très prédictives, plusieurs arbres peuvent choisir les mêmes variables et produire des structures similaires.

Leurs prédictions restent alors corrélées.

Random Forest introduit une deuxième source d'aléa : la sélection aléatoire des features lors de chaque split.
Bagging classique
À chaque nœud, un arbre peut examiner toutes les variables disponibles.
Random Forest
À chaque nœud, l'arbre examine seulement un sous-ensemble aléatoire des variables candidates.

Cela encourage les arbres à apprendre des structures différentes.



## 10. Architecture de Random Forest


[[forestarchitecture]]




## 11. Bagging versus Random Forest

Critère	Bagging d'arbres	Random Forest
Bootstrap	Généralement oui	Généralement oui
Sélection aléatoire de features par split	Pas nécessairement	Oui, selon max_features
Corrélation entre arbres	Peut être élevée	Souvent réduite
Modèles de base	Divers modèles possibles	Arbres de décision
Agrégation	Vote ou moyenne	Moyenne de probabilités ou de prédictions

Nuance : dans scikit-learn, RandomForestClassifier permet aussi bootstrap=False, et BaggingClassifier peut introduire une sélection aléatoire de features. Le tableau décrit les principes usuels.

Partie V — Random Forest Classification



## 12. Comment Random Forest prédit-il une classe ?

Chaque arbre de la forêt produit une distribution de probabilités sur les classes.

Pour une classification binaire :
\[
\hat p_b(x)=P_b(Y=1\mid x)
\]

Random Forest calcule la moyenne :
\[
\boxed{
\hat p_{\mathrm{RF}}(x)=
\frac1B\sum_{b=1}^{B}\hat p_b(x)
}
\]

La classe prédite correspond à la classe dont la probabilité moyenne est la plus élevée.
Exemple
Trois arbres prédisent :

Arbre	Probabilité de défaut
Arbre 1	0,20
Arbre 2	0,60
Arbre 3	0,40

La probabilité moyenne vaut :
\[
\hat p=\frac{0.20+0.60+0.40}{3}
\]

\[
\boxed{\hat p=0.40}
\]

Avec un seuil de décision de 0,5, la classe prédite est 0.

Attention : les probabilités produites par une forêt ne sont pas nécessairement bien calibrées.



## 13. Premier modèle Random Forest

~~~python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
X, y = make_classification(n_samples=3000, n_features=15, n_informative=8, n_redundant=4,
                           weights=[.9, .1], random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                max_features="sqrt", random_state=42, n_jobs=1)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, model.predict(X_test), zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_proba), "AP:", average_precision_score(y_test, y_proba))
~~~




Pourquoi utiliser plusieurs métriques ?
Dans un problème de défaut bancaire, la classe positive est souvent minoritaire.

L'accuracy seule peut être trompeuse.

Nous devons examiner notamment :

- ROC-AUC : discrimination.
- Average Precision : qualité du classement pour la classe positive.
- Recall : capacité à détecter les positifs.
- Precision : fiabilité des prédictions positives.
- F1-score : compromis entre précision et rappel.
- Calibration : cohérence entre probabilités prédites et fréquences observées.

Partie VI — Random Forest Regression



## 14. Comment fonctionne la régression ?

Chaque arbre produit une prédiction numérique :
\[
f_b(x)
\]

La forêt calcule la moyenne :
\[
\boxed{
\hat y_{\mathrm{RF}}(x)=
\frac1B\sum_{b=1}^{B}f_b(x)
}
\]

Exemple
Nous voulons prédire une perte financière.

Trois arbres prédisent :
\[
[1200,1500,1350]
\]

La forêt prédit :
\[
\hat y=\frac{1200+1500+1350}{3}
\]

\[
\boxed{\hat y=1350}
\]




## 15. Exemple Python

~~~python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def regression_lab():
    X_reg, y_reg = make_regression(n_samples=2500, n_features=10, n_informative=6, noise=25, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=.8,
                                   random_state=42, n_jobs=1)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Forêt régression RMSE CV:", -scores["test_score"].mean(),
          "test : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    average = np.mean([tree.predict(X_reg_test[:5]) for tree in model.estimators_], axis=0)
    assert np.allclose(model.predict(X_reg_test[:5]), average)
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


regression_lab()
~~~




Limite en extrapolation
Comme les arbres de régression classiques prédisent des moyennes locales, une Random Forest ne prolonge pas naturellement une tendance linéaire au-delà de la plage des valeurs cibles d'entraînement.

C'est une limite importante lorsque nous prédisons des valeurs futures susceptibles de dépasser les observations historiques.

Partie VII — Hyperparamètres essentiels



## 16. n_estimators

Définit le nombre d'arbres.
~~~python
n_estimators=200
~~~




En général, augmenter le nombre d'arbres stabilise les prédictions et réduit la variabilité liée à l'échantillonnage aléatoire.

Mais :

- Le temps d'entraînement augmente.
- La mémoire utilisée augmente.
- Le temps de prédiction augmente.
- Les gains finissent souvent par diminuer.

Un nombre d'arbres plus élevé ne garantit pas une amélioration sur chaque dataset.



## 17. max_depth

Contrôle la profondeur maximale de chaque arbre.
~~~python
max_depth=10
~~~




Un arbre très profond est flexible, mais peut apprendre des structures spécifiques au train.

Une forêt réduit une partie de cette variance grâce à l'agrégation, sans supprimer tous les risques de surapprentissage.



## 18. min_samples_leaf

Contrôle le nombre minimal d'observations par feuille.
~~~python
min_samples_leaf=5
~~~




Augmenter cette valeur peut :

- Lisser les prédictions.
- Réduire la variance.
- Limiter les probabilités extrêmes issues de feuilles très petites.

Mais une valeur trop grande peut introduire du biais.



## 19. max_features

C'est un hyperparamètre central de Random Forest.

Il détermine combien de features sont considérées à chaque split.

Par exemple :
~~~python
max_features="sqrt"
~~~




Avec 100 features, cela correspond généralement à environ 10 features candidates par split.

Autres possibilités :
~~~python
# Configurations alternatives, pas simultanées.
choices = [{"max_features": .5}, {"max_features": None}, {"max_features": 5}]
~~~




Interprétation

- Petit max_features : davantage de diversité, mais splits individuels potentiellement moins performants.
- Grand max_features : splits potentiellement meilleurs localement, mais arbres souvent plus corrélés.

Le compromis doit être évalué empiriquement.



## 20. bootstrap et max_samples

~~~python
RandomForestClassifier(bootstrap=True, max_samples=.8, random_state=42)
~~~




Avec cette configuration, chaque arbre utilise un échantillon Bootstrap de taille égale à environ 80 % du train.

max_samples s'applique lorsque bootstrap=True.



## 21. Tableau récapitulatif

Hyperparamètre	Effet principal
n_estimators	Nombre d'arbres
max_depth	Complexité individuelle
min_samples_split	Condition minimale pour un split
min_samples_leaf	Taille minimale des feuilles
max_features	Diversité des arbres
bootstrap	Échantillonnage avec remise
max_samples	Taille du Bootstrap
class_weight	Pondération des classes
oob_score	Évaluation Out-of-Bag
n_jobs	Parallélisation


Partie VIII — Out-of-Bag Evaluation



## 22. Qu'est-ce que le score OOB ?

Lorsqu'un arbre est entraîné sur un échantillon Bootstrap, certaines observations ne sont pas sélectionnées.

Ces observations peuvent servir à évaluer cet arbre.

Pour chaque observation \(i\), nous agrégeons les prédictions des arbres pour lesquels elle était OOB.

Nous obtenons ainsi une prédiction OOB :
\[
\hat f_{\mathrm{OOB}}(x_i)
\]

Ces prédictions permettent d'estimer la performance de généralisation.
Pourquoi est-ce intéressant ?
Parce que nous pouvons obtenir une estimation de performance sans créer systématiquement un jeu de validation séparé pour cette mesure.

Cependant, le score OOB ne remplace pas toujours une validation croisée ou un jeu de test indépendant.



## 23. Activer OOB dans scikit-learn

~~~python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, bootstrap=True, oob_score=True, random_state=42, n_jobs=1)
model.fit(X_train, y_train)
print("OOB accuracy par défaut :", model.oob_score_)
~~~




Par défaut, pour RandomForestClassifier, oob_score_ correspond à une accuracy OOB.

Cela ne signifie pas qu'il s'agit d'une ROC-AUC.
Calculer la ROC-AUC OOB
~~~python
import numpy as np
from sklearn.metrics import roc_auc_score
oob_matrix = model.oob_decision_function_
valid = np.isfinite(oob_matrix).all(axis=1) & (oob_matrix.sum(axis=1) > 0)
print("Couverture OOB:", valid.mean())
print("OOB ROC-AUC:", roc_auc_score(np.asarray(y_train)[valid], oob_matrix[valid, 1]))
~~~




Il faut disposer de suffisamment d'arbres pour que chaque observation ait une bonne chance de recevoir des prédictions OOB. Avec trop peu d'arbres, certaines prédictions OOB peuvent être indisponibles ou peu fiables.
Limites de l'OOB
Le mécanisme OOB ne garantit pas l'absence de fuite d'information si un preprocessing a été ajusté sur toutes les données avant la forêt.

Il faut aussi faire attention aux données temporelles, aux observations regroupées par individu ou entreprise et aux dépendances entre observations.

Partie IX — Feature Importance



## 24. Importance basée sur l'impureté

Random Forest fournit :
~~~python
model.feature_importances_
~~~




Cette mesure agrège les réductions d'impureté attribuées aux variables dans les arbres.
Exemple
~~~python
import pandas as pd
importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
print(importance_df.sort_values("importance", ascending=False))
~~~




feature_names doit correspondre aux colonnes du dataset utilisé pour entraîner le modèle.
Attention
L'importance MDI peut être biaisée en faveur de variables ayant de nombreuses valeurs distinctes.

En présence de variables corrélées, l'importance peut également être répartie entre plusieurs variables.

Une importance élevée ne prouve pas qu'une variable est causalement responsable de la cible.



## 25. Permutation Importance

Une méthode alternative consiste à mélanger les valeurs d'une variable sur un ensemble de validation, puis à mesurer la dégradation du score.
~~~python
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
analysis_model = clone(model).fit(X_fit, y_fit)
result = permutation_importance(analysis_model, X_valid, y_valid, scoring="roc_auc", n_repeats=10,
                                random_state=42, n_jobs=1)
importance_df = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
print(importance_df.sort_values("importance", ascending=False))
~~~




Pour sélectionner les variables ou modifier le modèle à partir de cette analyse, il faut utiliser une validation plutôt que le test final.

Partie X — Class Imbalance



## 26. Random Forest et classes déséquilibrées

Supposons que notre dataset contienne :

- 95 % de non-défauts.
- 5 % de défauts.

Une forêt peut obtenir une accuracy élevée tout en détectant mal les défauts.
Solution 1 — class_weight
~~~python
RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
~~~




Avec balanced, les poids sont calculés approximativement selon :
\[
w_k=\frac{n}{K n_k}
\]

Où :

- \(n\) : nombre total d'observations.
- \(K\) : nombre de classes.
- \(n_k\) : nombre d'observations de la classe \(k\).

Une autre possibilité est :
~~~python
class_weight="balanced_subsample"
~~~




Dans ce cas, les poids sont recalculés à partir de l'échantillon Bootstrap de chaque arbre.
Solution 2 — Ajuster le seuil de décision
Au lieu de toujours utiliser 0,5, nous pouvons sélectionner un seuil en fonction de l'objectif métier.

Par exemple, dans la détection du défaut :

- Un seuil plus faible peut augmenter le recall.
- Mais il peut également augmenter les faux positifs.

Le seuil doit être sélectionné sur des données de validation.
Solution 3 — Métriques adaptées
Examiner notamment :

- ROC-AUC.
- Average Precision.
- Precision.
- Recall.
- F1-score.
- Matrice de confusion.
- Calibration.

Solution 4 — Resampling
Nous pouvons également comparer des approches de sur-échantillonnage ou de sous-échantillonnage.

Mais toute méthode de resampling doit être appliquée uniquement aux partitions d'entraînement, idéalement à l'intérieur d'un pipeline adapté.

Partie XI — Random Forest versus Decision Tree



## 27. Comparaison

Critère	Decision Tree	Random Forest
Nombre d'arbres	1	Plusieurs
Variance	Souvent élevée	Généralement réduite
Interprétabilité	Très bonne si arbre simple	Plus complexe
Overfitting	Peut être important	Souvent mieux contrôlé
Entraînement	Rapide	Plus coûteux
Prédiction	Rapide	Plus coûteuse
Feature Importance	Oui	Oui
Scaling	Généralement inutile	Généralement inutile
Non-linéarité	Oui	Oui

Pourquoi Random Forest est-il souvent plus performant ?
Parce qu'il réduit l'instabilité d'un arbre unique en agrégeant des arbres diversifiés.

Mais il ne garantit pas de meilleures performances dans tous les contextes.

Un arbre simple peut être préférable lorsque l'explicabilité des règles est prioritaire et que sa performance est suffisante.

Partie XII — Random Forest versus Boosting



## 28. Quelle différence ?

C'est une question fréquente en entretien.
Bagging / Random Forest
Les arbres sont entraînés de manière essentiellement indépendante.

L'objectif principal est de réduire la variance grâce à l'agrégation.
Boosting
Les arbres sont ajoutés séquentiellement.

Chaque nouvel arbre contribue à améliorer l'ensemble selon une fonction de perte.

Le Boosting cherche à corriger progressivement les erreurs du modèle collectif, selon la méthode utilisée.

Critère	Random Forest	Gradient Boosting
Entraînement des arbres	Parallélisable	Séquentiel dans le boosting classique
Diversité	Bootstrap + random features	Contributions successives
Objectif principal	Réduction de variance	Optimisation progressive de la perte
Hyperparamètre clé	n_estimators, max_features	learning_rate, n_estimators
Sensibilité au tuning	Souvent modérée	Peut être plus importante
Exemples	RandomForest	XGBoost, LightGBM, CatBoost

Nous étudierons le Gradient Boosting dans le prochain cours.

Partie XIII — Hyperparameter Tuning



## 29. Optimiser Random Forest

Les principaux hyperparamètres à explorer sont :
~~~python
param_grid = {"n_estimators": [100, 200, 400], "max_depth": [5, 10, 20, None],
              "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", .5, None]}
~~~




Une recherche exhaustive peut devenir coûteuse.

Dans ce cas, RandomizedSearchCV est souvent plus pratique.



## 30. RandomizedSearchCV complet

~~~python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
model = RandomForestClassifier(random_state=42, n_jobs=1)
param_distributions = {"n_estimators": [100, 200, 300, 500], "max_depth": [5, 10, 15, 20, None],
                       "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 5, 10],
                       "max_features": ["sqrt", .5, .8], "class_weight": [None, "balanced"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(model, param_distributions, n_iter=30, scoring="roc_auc", cv=cv,
                            random_state=42, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
~~~




Pourquoi n_jobs=1 dans la forêt ?
Parce que nous utilisons déjà n_jobs=-1 dans RandomizedSearchCV.

Cela permet d'éviter une parallélisation imbriquée trop importante, qui peut consommer inutilement les ressources de la machine.

Partie XIV — Coding Lab



## 31. Comparer Decision Tree et Random Forest


Laboratoire complet : la comparaison de complexité se fait sur les mêmes folds train. Les choix sont figés avant l’audit du test. Les figures s’affichent avec --plot.

~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=4000, n_features=20, n_informative=10, n_redundant=5,
                           weights=[.9, .1], class_sep=1., random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]


def regression_lab():
    X_reg, y_reg = make_regression(n_samples=2500, n_features=10, n_informative=6, noise=25, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=.8,
                                   random_state=42, n_jobs=1)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Forêt régression RMSE CV:", -scores["test_score"].mean(),
          "test : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    average = np.mean([tree.predict(X_reg_test[:5]) for tree in model.estimators_], axis=0)
    assert np.allclose(model.predict(X_reg_test[:5]), average)
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {"Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
                  "Bagging 200": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8),
                                                   n_estimators=200, random_state=42, n_jobs=1)}
    for trees in [1, 5, 10, 25, 50, 100, 200]:
        candidates[f"Random Forest {trees}"] = RandomForestClassifier(n_estimators=trees, max_depth=8,
                                                                       random_state=42, n_jobs=1)
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring={"auc": "roc_auc", "ap": "average_precision"})
        rows.append({"model": name, "ROC_AUC_CV": scores["test_auc"].mean(),
                     "AP_CV": scores["test_ap"].mean()})
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    chosen = results.loc[results["ROC_AUC_CV"].idxmax(), "model"]
    model = candidates[chosen].fit(X_train, y_train)
    pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    print("Choix par CV avant test:", chosen, "audit test : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "F1", f1_score(y_test, pred))
    oob = RandomForestClassifier(n_estimators=300, bootstrap=True, oob_score=True, random_state=42, n_jobs=1)
    oob.fit(X_train, y_train)
    oob_matrix = oob.oob_decision_function_
    valid = np.isfinite(oob_matrix).all(axis=1) & (oob_matrix.sum(axis=1) > 0)
    print("OOB accuracy (défaut sklearn):", oob.oob_score_, "couverture OOB:", valid.mean())
    print("OOB ROC-AUC:", roc_auc_score(y_train[valid], oob_matrix[valid, 1]),
          "OOB AP:", average_precision_score(y_train[valid], oob_matrix[valid, 1]))
    # La forêt moyenne les probabilités des arbres, pas nécessairement leurs décisions de classe.
    average = np.mean([tree.predict_proba(X_test[:5]) for tree in oob.estimators_], axis=0)
    assert np.allclose(oob.predict_proba(X_test[:5]), average)
    regression_lab()
    if args.plot:
        import matplotlib.pyplot as plt
        counts = [1, 5, 10, 25, 50, 100, 200]
        aucs = [float(results.loc[results["model"] == f"Random Forest {n}", "ROC_AUC_CV"].iloc[0]) for n in counts]
        plt.plot(counts, aucs, marker="o")
        plt.xlabel("Nombre d’arbres")
        plt.ylabel("ROC-AUC CV train")
        plt.title("Stabilisation empirique, sans choisir sur test")
        plt.show()
    print("OOB et CV ne sont pas des garanties pour le temps ou les données groupées ; test final distinct.")
~~~

Questions à analyser
1. La performance augmente-t-elle toujours avec le nombre d'arbres ?
2. À partir de combien d'arbres observe-t-on une stabilisation ?
3. Random Forest surpasse-t-il toujours Decision Tree ?
4. Quel est l'effet de max_depth ?
5. Quel est l'effet de max_features ?
6. Pourquoi un grand nombre d'arbres peut-il devenir coûteux ?

Partie XV — Coding Challenge



## 32. Challenge : Random Forest Credit Risk

Challenge 07 — Random Forest Benchmark

+600 XP

Construis un modèle de classification du risque de défaut avec Random Forest, optimise ses hyperparamètres et compare ses résultats à ceux d'un arbre unique.
~~~python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=25, n_informative=12, n_redundant=6,
                           weights=[.95, .05], flip_y=.01, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
# TODO : comparer arbre et forêt sur les mêmes folds train avec ROC-AUC et AP.
# TODO : tester n_estimators, max_features, profondeur et taille de feuilles.
# TODO : RandomizedSearchCV avec budget explicite et sans parallélisation imbriquée.
# TODO : analyser OOB et permutation importance sur validation distincte.
# TODO : figer les choix avant audit test, puis examiner seuil et calibration.
~~~







[[solution]]

## Questions d’entretien — Random Forest et Bagging

[[interview]]




## 33. Mini-quiz interactif


[[quiz]]




## 34. Validation du cours


[[assessment]]

À retenir : Random Forest améliore la stabilité des arbres de décision grâce à deux mécanismes complémentaires : le Bootstrap, qui diversifie les observations utilisées, et la sélection aléatoire de features, qui diversifie les règles apprises.

Dans le prochain cours, nous verrons une autre approche de l'Ensemble Learning : Gradient Boosting, ainsi que ses implémentations majeures, XGBoost, LightGBM et CatBoost.