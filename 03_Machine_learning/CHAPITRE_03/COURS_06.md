Dans le cours précédent, nous avons étudié K-Nearest Neighbors (KNN), qui prend ses décisions à partir des observations les plus proches.

Aujourd'hui, nous allons découvrir un algorithme fondamental, très utilisé en Machine Learning et à la base de modèles puissants comme Random Forest, XGBoost et LightGBM.

Contrairement à KNN, un arbre de décision apprend une succession de règles conditionnelles.

Par exemple :
Si le ratio d'endettement est élevé, examiner la capacité de remboursement. Sinon, examiner l'historique des incidents.

L'algorithme apprend automatiquement ces règles à partir des données.


## 1. Qu'est-ce qu'un Decision Tree ?

Un arbre de décision est un algorithme supervisé utilisable pour deux types de problèmes :

- Classification : prédire une classe, par exemple défaut ou non-défaut.
- Régression : prédire une valeur continue, par exemple un montant de perte.

L'arbre divise progressivement les observations en groupes de plus en plus homogènes.
Exemple : prédiction du défaut
Un arbre de décision simplifié
Exemple pédagogique fictif. Les seuils et les règles ne proviennent pas d'un entraînement réel.
Un arbre contient trois éléments fondamentaux.

Élément	Rôle
Root node	Premier nœud, contenant les observations d'entraînement
Internal node	Nœud qui applique une condition
Leaf node	Feuille qui produit une prédiction

Chaque séparation est appelée un split.



## 2. Comment un arbre apprend-il ses règles ?

L'algorithme cherche, à chaque nœud, une variable et un seuil permettant de séparer les observations.

Pour une variable numérique :
\[
x_j\le t
\]

Les observations sont réparties en deux groupes :
\[
D_L=\{(x_i,y_i):x_{ij}\le t\}
\]

\[
D_R=\{(x_i,y_i):x_{ij}>t\}
\]

Le problème est donc :

Comment choisir la meilleure variable et le meilleur seuil ?

Pour cela, l'arbre utilise un critère d'impureté.

Partie I — Les critères de séparation



## 3. Gini Impurity

L'indice de Gini mesure l'hétérogénéité des classes dans un nœud.

Pour \(K\) classes :
\[
\boxed{Gini(D)=1-\sum_{k=1}^{K}p_k^2}
\]

Où \(p_k\) représente la proportion de la classe \(k\) dans le nœud.
Classification binaire
Avec deux classes :
\[
Gini=1-p^2-(1-p)^2
\]

En simplifiant :
\[
\boxed{Gini=2p(1-p)}
\]

Exemple 1 — Nœud parfaitement pur
Supposons :

- 10 observations de classe 0.
- 0 observation de classe 1.

Alors :
\[
p_0=1,\qquad p_1=0
\]

\[
Gini=1-1^2-0^2=0
\]

Un Gini de 0 signifie que toutes les observations appartiennent à la même classe.
Exemple 2 — Nœud équilibré
Supposons :

- 5 observations de classe 0.
- 5 observations de classe 1.

Alors :
\[
Gini=1-0.5^2-0.5^2
\]

\[
\boxed{Gini=0.5}
\]

En classification binaire, 0,5 est la valeur maximale du Gini.



## 4. Explorateur interactif — Gini


[[treeimpurity]]

Un arbre cherche des enfants moins impurs que leur parent ; Gini et entropie sont deux critères possibles.




## 5. Entropy

L'entropie mesure l'incertitude associée à la distribution des classes.
\[
\boxed{
H(D)=-\sum_{k=1}^{K}p_k\log_2(p_k)
}
\]

Par convention :
\[
0\log_2(0)=0
\]

Exemple : deux classes équilibrées
\[
p_0=p_1=0.5
\]

\[
H=-0.5\log_2(0.5)-0.5\log_2(0.5)
\]

\[
\boxed{H=1}
\]

Exemple : nœud pur
\[
p_0=1,\qquad p_1=0
\]

\[
\boxed{H=0}
\]

Comme Gini, l'entropie est minimale pour un nœud pur.



## 6. Gini versus Entropy

Critère	Gini	Entropy
Formule	\(1-\sum p_k^2\)	\(-\sum p_k\log_2p_k\)
Nœud pur	0	0
Maximum binaire	0,5	1
Interprétation	Impureté	Incertitude
scikit-learn	criterion="gini"	criterion="entropy"

En pratique, les deux critères donnent souvent des arbres similaires, mais ils peuvent choisir des séparations différentes.

Il n'existe pas de règle universelle selon laquelle Entropy serait toujours meilleur que Gini.

Partie II — Information Gain et choix du split



## 7. Comment mesurer la qualité d'une séparation ?

Supposons qu'un nœud parent \(D\) soit divisé en deux groupes \(D_L\) et \(D_R\).

L'impureté pondérée après séparation est :
\[
I_{\mathrm{après}}=
\frac{|D_L|}{|D|}I(D_L)
+
\frac{|D_R|}{|D|}I(D_R)
\]

Le gain d'impureté est :
\[
\boxed{
Gain=
I(D)-I_{\mathrm{après}}
}
\]

Lorsque \(I\) est l'entropie, on parle généralement d'Information Gain.

Avec Gini, on parle plutôt de réduction de l'impureté de Gini.
Pourquoi pondérer les nœuds enfants ?
Parce qu'un split créant un nœud pur de deux observations et un autre nœud très impur de 998 observations n'est pas nécessairement une bonne séparation.

La pondération tient compte de la taille de chaque groupe.



## 8. Exemple de calcul

Nous avons un nœud parent contenant 10 observations :

- 5 de classe 0.
- 5 de classe 1.

Son Gini est :
\[
Gini(D)=0.5
\]

Un split produit :

Nœud gauche :

- 4 observations de classe 0.
- 1 observation de classe 1.

\[
Gini(D_L)=1-(4/5)^2-(1/5)^2
\]

\[
Gini(D_L)=0.32
\]

Nœud droit :

- 1 observation de classe 0.
- 4 observations de classe 1.

\[
Gini(D_R)=0.32
\]

L'impureté pondérée vaut :
\[
I_{\mathrm{après}}=
\frac5{10}(0.32)+\frac5{10}(0.32)
\]

\[
I_{\mathrm{après}}=0.32
\]

Le gain vaut :
\[
\boxed{Gain=0.5-0.32=0.18}
\]

Ce split réduit donc l'impureté.



## 9. Simulateur interactif de split


[[treesplit]]

Le meilleur gain local parmi les candidats admissibles ne garantit pas une meilleure généralisation de l’arbre complet.




## 10. Classification and Regression Trees

L'algorithme utilisé par les arbres de décision de scikit-learn appartient à la famille CART.

CART signifie :

Classification and Regression Trees.

Ses caractéristiques principales :

- Arbre binaire : chaque split produit deux enfants.
- Recherche de séparations selon un critère.
- Construction récursive.
- Prédiction à partir des feuilles.

Comment CART construit-il l'arbre ?
À chaque nœud :
1. Examiner les variables candidates.
2. Examiner les seuils candidats.
3. Calculer la qualité des séparations.
4. Choisir un split selon le critère d'optimisation.
5. Répéter sur les enfants.
6. Arrêter selon les conditions définies.
C'est une approche gloutonne (greedy).

L'algorithme optimise les décisions localement, sans garantir de trouver l'arbre globalement optimal.



## 11. Pourquoi un arbre est-il non linéaire ?

Considérons une règle :
\[
x_1\le0.4
\]

Puis une autre :
\[
x_2>0.7
\]

Chaque split découpe l'espace des features.

En répétant ces découpages, l'arbre peut construire une frontière de décision complexe.
Frontière de décision d'un arbre
Illustration de partitions rectangulaires obtenues avec des splits sur des variables numériques.
Différence avec la régression logistique : une régression logistique standard possède une frontière linéaire dans l'espace des features, alors qu'un arbre peut créer une frontière en escalier.

Partie IV — Decision Tree Classification



## 12. Comment une feuille prédit-elle une classe ?

Une feuille contient un ensemble d'observations d'entraînement.

Supposons qu'une feuille contienne :

- 18 observations de classe 0.
- 2 observations de classe 1.

L'estimation de probabilité de la classe 1 est :
\[
\hat P(Y=1\mid X\in feuille)=\frac2{20}=0.1
\]

La classe prédite est généralement la classe majoritaire.
Limite importante
Ces probabilités correspondent aux proportions de classes dans les feuilles.

Elles ne sont pas nécessairement bien calibrées.

Un arbre très profond peut produire des feuilles contenant très peu d'observations, et donc des probabilités extrêmes comme 0 ou 1.



## 13. Premier arbre avec scikit-learn

~~~python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
                           weights=[.85, .15], random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
model = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_proba), "AP:", average_precision_score(y_test, y_proba))
~~~




Pourquoi max_depth=4 ?
Parce que nous voulons éviter que l'arbre crée trop de niveaux et apprenne des règles extrêmement spécifiques aux observations d'entraînement.
Pourquoi min_samples_leaf=10 ?
Parce que chaque feuille doit contenir au moins 10 observations d'entraînement.

Cela limite les feuilles trop petites.

Partie V — Decision Tree Regression



## 14. Comment fonctionne un arbre de régression ?

En classification, l'arbre cherche à séparer les classes.

En régression, il cherche à former des groupes dans lesquels les valeurs cibles sont similaires.

Une feuille prédit généralement la moyenne des valeurs cibles des observations qui lui appartiennent lorsque le critère est squared_error.
Exemple
Une feuille contient trois valeurs :
\[
[200,220,250]
\]

La prédiction vaut :
\[
\hat y=\frac{200+220+250}{3}
\]

\[
\hat y=223.33
\]




## 15. Critère MSE

Pour un nœud \(D\), nous pouvons mesurer :
\[
\boxed{
MSE(D)=\frac1{|D|}
\sum_{i\in D}(y_i-\bar y_D)^2
}
\]

Où :
\[
\bar y_D=\frac1{|D|}\sum_{i\in D}y_i
\]

L'arbre recherche les séparations qui réduisent la dispersion des valeurs cibles.
Exemple Python
~~~python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def regression_lab():
    X_reg, y_reg = make_regression(n_samples=1000, n_features=8, noise=20, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, criterion="squared_error", random_state=42)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    print("Arbre régression RMSE CV train:", -scores["test_score"].mean())
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Arbre régression test fixe : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    leaf_ids = model.apply(X_reg_test)
    for leaf in np.unique(leaf_ids):
        assert np.allclose(pred[leaf_ids == leaf], pred[leaf_ids == leaf][0])
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


regression_lab()
~~~




Une limite de l'arbre de régression
Un arbre de régression classique prédit une valeur constante à l'intérieur de chaque feuille.

Sa fonction de prédiction est donc généralement en escalier.

Il n'extrapole pas naturellement une tendance linéaire au-delà des observations d'entraînement.

Partie VI — Overfitting des arbres



## 16. Pourquoi les arbres surapprennent-ils facilement ?

Un arbre sans contraintes peut continuer à séparer les observations jusqu'à obtenir des feuilles très petites.

Il peut alors apprendre des règles comme :
Si ratio d'endettement ≤ 0,437 et revenu > 2 813 et ancienneté ≤ 4,7...

Ces règles peuvent correspondre à des particularités de l'échantillon plutôt qu'à des relations stables.
Arbre peu profond

- Moins complexe.
- Moins de variance.
- Peut sous-apprendre.

Arbre très profond

- Très flexible.
- Peut obtenir une excellente performance sur le train.
- Peut généraliser moins bien.



## 17. Explorateur — Profondeur et complexité


[[treedepth]]




## 18. Pre-pruning

Le pre-pruning consiste à limiter la croissance de l'arbre pendant sa construction.

Scikit-learn propose plusieurs hyperparamètres.

Paramètre	Rôle
max_depth	Profondeur maximale
min_samples_split	Nombre minimal d'observations requis pour envisager un split
min_samples_leaf	Nombre minimal d'observations dans chaque feuille
max_leaf_nodes	Nombre maximal de feuilles
min_impurity_decrease	Réduction minimale d'impureté nécessaire pour un split
max_features	Nombre de features considérées lors de la recherche du meilleur split

Exemple
~~~python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10,
                                max_leaf_nodes=30, random_state=42)
~~~




Différence entre min_samples_split et min_samples_leaf
Supposons :
~~~python
min_samples_split = 20
min_samples_leaf = 10
~~~




min_samples_split=20 signifie qu'un nœud doit contenir au moins 20 observations pour pouvoir être séparé.

min_samples_leaf=10 impose qu'après séparation, chaque enfant terminal admissible contienne au moins 10 observations.

Ces deux contraintes agissent différemment.



## 19. Post-pruning

Le post-pruning consiste à réduire un arbre après sa construction.

L'une des méthodes est le Minimal Cost-Complexity Pruning.

Elle cherche un compromis entre l'erreur et la complexité :
\[
\boxed{
R_\alpha(T)=R(T)+\alpha|T|
}
\]

Où :

- \(R(T)\) représente le coût de l'arbre.
- \(|T|\) représente le nombre de feuilles.
- \(\alpha\) contrôle la pénalisation de la complexité.

Dans scikit-learn, l'hyperparamètre est :
~~~python
ccp_alpha = 0.001
~~~




Interprétation

- ccp_alpha=0 : pas de pénalisation de complexité par ce mécanisme.
- ccp_alpha plus élevé : davantage de pruning possible.



## 20. Cost-Complexity Pruning avec Python

~~~python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
base_tree = DecisionTreeClassifier(random_state=42)
path = base_tree.cost_complexity_pruning_path(X_train, y_train)
print("Chemin descriptif sur train:", path.ccp_alphas)
# Grille fixée à l'avance : les cibles des folds validation ne définissent pas les candidats.
search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      {"ccp_alpha": [0., .0001, .001, .01, .02]},
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="roc_auc")
search.fit(X_train, y_train)
print("Best alpha:", search.best_params_)
~~~




Pour une évaluation particulièrement rigoureuse, la génération des candidats de pruning doit elle aussi respecter les partitions de validation. Un chemin calculé une fois sur tout le train peut influencer le choix des candidats utilisés dans les folds.

Une alternative pratique consiste à tester une grille prédéfinie de ccp_alpha, sans la calculer à partir des données de validation.

Partie VIII — Feature Importance



## 21. Comment mesurer l'importance des variables ?

Un arbre peut fournir une mesure appelée :

Mean Decrease in Impurity (MDI).

Elle repose sur la réduction totale d'impureté attribuable aux splits utilisant chaque variable.

En scikit-learn :
~~~python
model.feature_importances_
~~~




Exemple
~~~python
import pandas as pd
importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
print(importance.sort_values("importance", ascending=False))
~~~




Dans cet exemple, feature_names doit contenir les noms des variables utilisées lors de l'entraînement.
Limites de MDI
Les importances basées sur l'impureté peuvent favoriser certaines variables ayant de nombreuses valeurs distinctes.

Elles peuvent également être difficiles à interpréter lorsque les variables sont fortement corrélées.

Une variable importante pour les splits n'est pas nécessairement une cause du phénomène étudié.



## 22. Permutation Importance

Une alternative consiste à mesurer la dégradation de performance lorsque nous mélangeons les valeurs d'une variable.
Principe
1. Évaluer le modèle sur un dataset de validation.
2. Mélanger les valeurs d'une variable.
3. Réévaluer le modèle.
4. Mesurer la perte de performance.
Si le score se dégrade fortement, la variable est probablement utile au modèle.
~~~python
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
analysis_model = clone(model).fit(X_fit, y_fit)
result = permutation_importance(analysis_model, X_valid, y_valid, scoring="roc_auc",
                                n_repeats=10, random_state=42)
print(result.importances_mean)
~~~




Dans un projet réel, on privilégiera un ensemble de validation pour analyser et ajuster le modèle, en conservant le test final pour l'évaluation indépendante.

La permutation importance peut elle aussi être trompeuse en présence de variables fortement corrélées.

Partie IX — Visualiser un arbre



## 23. plot_tree

Scikit-learn permet de visualiser les règles apprises.
~~~python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
plt.figure(figsize=(18, 9))
plot_tree(model, filled=True, rounded=True, feature_names=feature_names,
          class_names=["Non-default", "Default"], fontsize=10)
plt.show()
~~~




Les noms de variables doivent correspondre aux colonnes réellement utilisées.
Comment lire un nœud ?
Un nœud peut afficher :
~~~text
debt_ratio <= 0.42
gini = 0.32
samples = 450
value = [360, 90]
class = Non-default
~~~

Cela signifie :

- La condition est debt_ratio <= 0.42.
- L'impureté Gini est 0,31.
- Le nœud contient 450 observations.
- Les effectifs sont 360 pour la classe 0 et 90 pour la classe 1, dans le cas non pondéré.
- La classe majoritaire est 0.

Avec sample_weight ou class_weight, certaines valeurs affichées peuvent correspondre à des sommes pondérées selon la configuration.

Partie X — Decision Trees et scaling



## 24. Faut-il standardiser les variables ?

En général, les arbres de décision classiques n'ont pas besoin de StandardScaler.

Pourquoi ?

Parce qu'ils utilisent des conditions de seuil :
\[
x_j\le t
\]

Une transformation monotone croissante conserve l'ordre des observations.

Par exemple, si nous transformons un revenu de DT en milliers de DT, les mêmes observations restent ordonnées de la même manière.

Les splits peuvent donc être équivalents après adaptation des seuils.
Comparaison

Algorithme	Scaling généralement nécessaire ?
KNN	Oui
Logistic Regression	Recommandé, surtout avec régularisation
SVM	Généralement oui
Decision Tree	Non
Random Forest	Non
Gradient Boosting sur arbres	Non

Attention : les transformations non monotones, les approximations numériques ou certaines méthodes de construction des arbres peuvent modifier les résultats.

Partie XI — Hyperparameter Tuning



## 25. Optimiser un arbre avec GridSearchCV

Nous allons sélectionner automatiquement :

- La profondeur.
- Le nombre minimal d'observations par feuille.
- Le nombre minimal d'observations pour un split.
- Le critère Gini ou Entropy.
~~~python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
param_grid = {"criterion": ["gini", "entropy"], "max_depth": [3, 5, 7, 10, None],
              "min_samples_split": [2, 10, 20], "min_samples_leaf": [1, 5, 10, 20],
              "ccp_alpha": [0., .001, .01]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, scoring="roc_auc", cv=cv)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("AUC de sélection CV:", search.best_score_)
~~~




Pourquoi utiliser ROC-AUC ?
Pour un problème de classification binaire, la ROC-AUC permet d'évaluer la capacité du modèle à classer les observations positives au-dessus des négatives.

Mais en cas de forte rareté de la classe positive, il est également pertinent d'examiner l'Average Precision et les métriques au seuil opérationnel.

Pour un modèle de risque, il faut aussi vérifier la calibration des probabilités.

Partie XII — Comparaison avec KNN et Logistic Regression



## 26. Quel modèle choisir ?

Critère	Logistic Regression	KNN	Decision Tree
Principe	Modèle linéaire dans le logit	Proximité	Règles conditionnelles
Non-linéarité	Via feature engineering	Oui	Oui
Scaling	Recommandé	Important	Généralement inutile
Interprétabilité	Coefficients	Voisins	Règles
Overfitting	Contrôlé par régularisation	Dépend de K	Dépend de la profondeur et des feuilles
Prédiction	Rapide	Potentiellement coûteuse	Généralement rapide
Variables corrélées	Peut être sensible	Affectent la distance	Peuvent rendre les splits instables

Les limites d'un arbre unique
Un arbre de décision présente plusieurs faiblesses :

- Instabilité : une petite modification du dataset peut changer la structure.
- Surapprentissage si l'arbre est trop profond.
- Frontières de décision en escalier.
- Performances parfois inférieures aux méthodes d'ensemble.
- Probabilités souvent mal calibrées lorsque les feuilles sont petites.

Ces limites expliquent pourquoi nous utilisons fréquemment Random Forest ou Gradient Boosting.

Partie XIII — Coding Lab complet



## 27. Construire un arbre de classification


Laboratoire complet : la comparaison de complexité se fait sur les mêmes folds train. Les choix sont figés avant l’audit du test. Les figures s’affichent avec --plot.

~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error, r2_score)

X, y = make_classification(n_samples=3000, n_features=12, n_informative=7, n_redundant=3,
                           weights=[.9, .1], flip_y=.02, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)


def regression_lab():
    X_reg, y_reg = make_regression(n_samples=1000, n_features=8, noise=20, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, criterion="squared_error", random_state=42)
    scores = cross_validate(model, X_reg_train, y_reg_train,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="neg_root_mean_squared_error")
    print("Arbre régression RMSE CV train:", -scores["test_score"].mean())
    model.fit(X_reg_train, y_reg_train)
    pred = model.predict(X_reg_test)
    print("Arbre régression test fixe : MAE", mean_absolute_error(y_reg_test, pred),
          "RMSE", np.sqrt(mean_squared_error(y_reg_test, pred)), "R2", r2_score(y_reg_test, pred))
    leaf_ids = model.apply(X_reg_test)
    for leaf in np.unique(leaf_ids):
        assert np.allclose(pred[leaf_ids == leaf], pred[leaf_ids == leaf][0])
    assert np.all(pred >= y_reg_train.min()) and np.all(pred <= y_reg_train.max())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Afficher les AUC CV et un arbre de profondeur 3")
    args = parser.parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, models = [], {}
    for depth in [2, 3, 5, 7, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_validate(model, X_train, y_train, cv=cv, return_train_score=True,
                                scoring={"auc": "roc_auc", "ap": "average_precision"})
        model.fit(X_train, y_train)
        rows.append({"depth": str(depth), "train_auc_CV": scores["train_auc"].mean(),
                     "validation_auc_CV": scores["test_auc"].mean(), "AP_CV": scores["test_ap"].mean(),
                     "leaves_on_train": model.get_n_leaves()})
        models[str(depth)] = model
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    chosen = results.loc[results["validation_auc_CV"].idxmax(), "depth"]
    model = models[chosen]
    print("Profondeur choisie par CV avant le test:", chosen)
    proba, pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)
    print("Audit final : ROC-AUC", roc_auc_score(y_test, proba),
          "AP", average_precision_score(y_test, proba), "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    print(pd.DataFrame({"feature": feature_names, "MDI": model.feature_importances_}).sort_values("MDI", ascending=False))
    regression_lab()
    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        axes[0].plot(results["depth"], results["train_auc_CV"], marker="o", label="Train CV")
        axes[0].plot(results["depth"], results["validation_auc_CV"], marker="o", label="Validation CV")
        axes[0].set(xlabel="Profondeur", ylabel="ROC-AUC")
        axes[0].legend()
        shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42).fit(X_train, y_train)
        plot_tree(shallow, feature_names=feature_names, class_names=["Classe 0", "Classe 1"],
                  filled=True, rounded=True, fontsize=7, ax=axes[1])
        fig.tight_layout()
        plt.show()
    print("La sélection de profondeur utilise les folds train ; le test ne sert pas à la modifier.")
~~~

Questions à analyser
1. Comment évolue l'AUC train lorsque la profondeur augmente ?
2. Comment évolue l'AUC de validation ou de test ?
3. Combien de feuilles possède un arbre sans profondeur maximale ?
4. Les variables les plus importantes sont-elles forcément les seules utiles ?
5. Pourquoi un arbre peu profond peut-il mieux généraliser ?

Partie XIV — Coding Challenge



## 28. Challenge : construire un arbre optimisé

Challenge 06 — Decision Tree Optimization

+550 XP

Construis un classifieur CART, optimise sa complexité et explique les règles qui influencent ses prédictions.
~~~python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from ch03_cours06_lab import X_train, X_test, y_train, y_test

X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
model = DecisionTreeClassifier(random_state=42)
grid = {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"], "ccp_alpha": [0., .001, .01]}
# TODO : GridSearchCV sur X_fit et StratifiedKFold, refit selon ROC-AUC.
# TODO : examiner AP, règles, MDI et permutation importance sur X_valid.
# TODO : figer les décisions puis auditer le test indépendant.
# BONUS : comparer pre-pruning et post-pruning sans générer de candidats sur validation.
~~~




Questions de réflexion
1. Pourquoi un arbre profond peut-il obtenir une AUC train proche de 1 ?
2. Quelle différence entre Gini et Entropy ?
3. Pourquoi min_samples_leaf peut-il améliorer la généralisation ?
4. Pourquoi les arbres n'ont-ils généralement pas besoin de scaling ?
5. Pourquoi la feature importance n'implique-t-elle pas une relation causale ?
6. Quelle différence entre pre-pruning et post-pruning ?



[[solution]]

## Questions d’entretien — Decision Trees

[[interview]]




## 29. Mini-quiz interactif


[[quiz]]




## 30. Validation du cours


[[assessment]]

À retenir : les arbres de décision apprennent des règles successives en cherchant à réduire l'impureté ou la dispersion des valeurs cibles. Ils sont interprétables et capables de modéliser des relations non linéaires, mais leur principal défaut est leur instabilité et leur sensibilité au surapprentissage.

Le prochain cours sera consacré à Random Forest et Bagging : nous verrons comment combiner plusieurs arbres pour obtenir des prédictions généralement plus stables et plus performantes.