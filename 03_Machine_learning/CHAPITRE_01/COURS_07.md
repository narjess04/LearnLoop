Dans le cours précédent, nous avons appris à détecter et éviter le Data Leakage.

Aujourd'hui, nous allons répondre à une question fondamentale :

Comment trouver la meilleure configuration d'un algorithme de Machine Learning ?

Par exemple, pourquoi choisir un arbre de décision de profondeur 5 plutôt que 10 ? Comment choisir la régularisation d'une régression logistique ? Et comment automatiser ces décisions ?

C'est le rôle de l'optimisation des hyperparamètres, appelée Hyperparameter Tuning.

## 1. Paramètres versus hyperparamètres

Il faut absolument distinguer ces deux concepts.
	Paramètres	Hyperparamètres
Définition	Valeurs apprises pendant l'entraînement	Valeurs qui configurent l'apprentissage ou le modèle
Exemples	Poids \(w\), biais \(b\)	Learning rate, profondeur, régularisation
Comment les obtenir ?	Optimisation lors de fit()	Choix ou recherche sur validation
Exemple scikit-learn	model.coef_	max_depth=5

Prenons une régression logistique :

~~~python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
print("Paramètres appris :", model.coef_, model.intercept_)
~~~

Ici :
- C contrôle la force de régularisation : une petite valeur de C correspond à une régularisation plus forte.
- max_iter fixe le nombre maximal d'itérations du solveur.
- Les coefficients appris sont disponibles dans model.coef_.
La valeur C=1.0 n'est pas apprise automatiquement pendant fit(). Nous devons la choisir.
2. Pourquoi optimiser les hyperparamètres ?
Supposons que nous entraînions un arbre de décision.

Avec max_depth=1, l'arbre risque d'être trop simple : underfitting.

Avec max_depth=30, il peut devenir très complexe : overfitting.

Une profondeur intermédiaire peut fournir une meilleure généralisation.
Explorateur interactif
Decision Tree — Hyperparameter Explorer
Simulation

Profondeur maximale (max_depth)5

Train Accuracy
90.4%

Validation Accuracy
86.6%

Train

Validation

Une profondeur intermédiaire peut offrir un compromis intéressant entre capacité et généralisation.
Courbes simulées à des fins pédagogiques, et non issues d'un entraînement réel.

## 3. Formalisation mathématique du Hyperparameter Tuning

Nous cherchons un ensemble d'hyperparamètres :
\[
\lambda^*=\arg\min_{\lambda\in\Lambda}
\hat R_{\mathrm{val}}(f_{\theta^*(\lambda)})
\]

Avec :
\[
\theta^*(\lambda)=
\arg\min_{\theta}L_{\mathrm{train}}(\theta;\lambda)
\]

Cela signifie que nous avons deux niveaux d'optimisation.

Niveau interne : entraîner les paramètres du modèle pour une configuration donnée.

Niveau externe : comparer différentes configurations sur des données de validation.

C'est pourquoi l'optimisation des hyperparamètres est parfois appelée optimisation bi-niveau (bilevel optimization).

En pratique, nous ne calculons généralement pas ces solutions exactes : les algorithmes d'entraînement et les recherches d'hyperparamètres produisent des solutions approximatives.

## 4. Grid Search — Recherche exhaustive sur une grille

Grid Search consiste à définir plusieurs valeurs possibles pour chaque hyperparamètre, puis à tester toutes les combinaisons.

Prenons une régression logistique :
\[
C\in\{0.01,\ 0.1,\ 1,\ 10\}
\]

Et deux choix de pondération des classes :
\[
class\_weight\in\{None,\ balanced\}
\]

Nous avons :
\[
4\times2=8\text{ combinaisons}
\]

Avec une cross-validation à 5 folds :
\[
8\times5=40\text{ entraînements}
\]

[[gridsearch]]

### 4.1 Implémentation avec GridSearchCV

~~~python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="roc_auc",
    n_jobs=1, refit=True
)
search.fit(X_train, y_train)
print("Meilleurs réglages :", search.best_params_)
print("Score de sélection CV (potentiellement optimiste) :", search.best_score_)
# Garder le test final intact si l'on compare encore plusieurs méthodes de recherche.
~~~

Comprendre les arguments importants
Argument	Rôle
estimator	Modèle ou pipeline à optimiser
param_grid	Combinaisons à tester
scoring	Métrique de sélection
cv	Stratégie de validation croisée
n_jobs=-1	Utilise les processeurs disponibles en parallèle
refit=True	Réentraîne la meilleure configuration sur tout le train transmis

Le préfixe model__C signifie que nous modifions le paramètre C de l'étape nommée model dans le pipeline.
Point important
grid_search.best_score_ correspond au meilleur score moyen obtenu en cross-validation.

Ce n'est pas le score du test final.

De plus, puisque nous avons sélectionné le maximum parmi plusieurs configurations, ce meilleur score CV peut être légèrement optimiste.

## 5. Randomized Search — Recherche aléatoire

Grid Search devient coûteux lorsque nous avons beaucoup d'hyperparamètres.

Imaginons :
- 10 valeurs pour le premier hyperparamètre.
- 8 valeurs pour le deuxième.
- 5 valeurs pour le troisième.
- 4 valeurs pour le quatrième.
Cela représente :
\[
10\times8\times5\times4=1600
\]

Avec 5 folds :
\[
1600\times5=8000\text{ entraînements}
\]

Random Search sélectionne un nombre limité de configurations selon des distributions définies.
Grid Search

Teste toutes les combinaisons prédéfinies.

Random Search

Teste un nombre limité de configurations.

### 5.1 Implémentation avec RandomizedSearchCV

~~~python
# Reprendre pipeline, cv, X_train et y_train du Coding Lab.
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    pipeline,
    {"model__C": loguniform(1e-4, 1e3),
     "model__class_weight": [None, "balanced"]},
    n_iter=20, scoring="roc_auc", cv=cv, random_state=42,
    n_jobs=1, refit=True
)
random_search.fit(X_train, y_train)
print(random_search.best_params_, random_search.best_score_)
~~~

Pourquoi loguniform ?
Le paramètre C peut varier sur plusieurs ordres de grandeur.

Une distribution log-uniforme permet d'explorer des valeurs comme \(10^{-4}\), \(10^{-2}\), \(1\), \(10^2\) sans concentrer presque tous les essais sur les plus grandes valeurs.

Grid Search versus Random Search : Grid Search est utile pour de petites grilles précises. Random Search est souvent plus efficace lorsque l'espace de recherche est grand et que le budget d'essais est limité.

## 6. Optuna — Optimisation adaptative des hyperparamètres

Optuna est une bibliothèque qui permet de construire des recherches d'hyperparamètres plus flexibles.

Contrairement à une recherche purement exhaustive ou aléatoire, certains samplers d'Optuna, notamment TPE, utilisent les résultats des essais précédents pour orienter les suivants.

### 6.1 Fonctionnement général

Schéma conceptuel. Le sampler utilise l'historique des essais pour proposer de nouvelles configurations selon sa stratégie.

### 6.2 Installer Optuna

~~~bash
python -m pip install optuna
~~~

Optuna est optionnel dans l’environnement de ces supports. Les blocs ont été vérifiés syntaxiquement ; leur étude n’a pas été exécutée dans la vérification locale faute de cette dépendance.
pip install optuna scikit-learn

### 6.3 Premier exemple complet

~~~python
# Dépendance optionnelle à installer dans l'environnement du cours : optuna.
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def objective(trial):
    C = trial.suggest_float("C", 1e-4, 1e3, log=True)
    weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=weight, max_iter=2000)
    )
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=30, timeout=120)
print(study.best_params, study.best_value)
~~~

Comprendre trial
Un trial représente une tentative d'entraînement avec une configuration particulière.

La méthode :
trial.suggest_float("C", 1e-4, 1e3, log=True)

demande au sampler de proposer une valeur de C dans l'intervalle défini.

Optuna évalue ensuite la fonction objective() et enregistre le score.

### 6.4 Réentraîner et tester le meilleur modèle

La fonction objective() ne remplace pas l'étape finale de réentraînement.

~~~python
# À exécuter une seule fois une fois la procédure figée.
from sklearn.metrics import roc_auc_score
best_params = study.best_params
final_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=best_params["C"], class_weight=best_params["class_weight"],
        max_iter=2000
    )
)
final_model.fit(X_train, y_train)
proba = final_model.predict_proba(X_test)[:, list(final_model.classes_).index(1)]
print("ROC-AUC finale, classe positive 1 :", roc_auc_score(y_test, proba))
~~~

Nous avons donc bien séparé la sélection des hyperparamètres et l'évaluation finale.

## 7. Bayesian Optimization — Comprendre l'idée mathématique

Optuna avec TPE appartient à la famille des méthodes d'optimisation adaptative fondées sur un modèle probabiliste.

L'idée générale consiste à exploiter les résultats déjà observés pour choisir plus intelligemment les prochaines configurations.

Il existe plusieurs approches, dont les processus gaussiens et le TPE (Tree-structured Parzen Estimator).
Exploration versus exploitation
Deux objectifs sont en tension.

Exploration : tester des zones encore peu connues de l'espace de recherche.

Exploitation : tester des configurations proches des meilleures déjà observées.
Exploration ou exploitation ?

ExplorationExploitation

L'algorithme choisit une zone moins explorée pour recueillir de nouvelles informations.
Illustration conceptuelle, pas une représentation exacte du fonctionnement de TPE.
Le principe du TPE
Pour un hyperparamètre \(x\), TPE construit des modèles de densité associés aux configurations ayant obtenu de bons scores et aux autres configurations.

Il cherche ensuite à privilégier les valeurs susceptibles d'améliorer l'objectif.

Cette stratégie peut être plus efficace qu'une recherche aléatoire, mais elle n'est pas systématiquement supérieure : son efficacité dépend de l'espace de recherche, du budget et de la nature du problème.

## 8. Les erreurs fréquentes dans le Hyperparameter Tuning

### 8.1 Optimiser sur le test set

C'est une erreur méthodologique majeure.

Le test final ne doit pas servir à sélectionner les hyperparamètres.

### 8.2 Tester trop de configurations sur une petite validation

Même sans fuite technique, comparer énormément de configurations peut conduire à sélectionner celle qui a bénéficié du hasard sur les données de validation.

C'est une forme d'overfitting au processus de sélection.

La Nested Cross-Validation, étudiée au cours 5, peut aider à obtenir une estimation plus rigoureuse.

### 8.3 Choisir une métrique inadaptée

Sur un dataset très déséquilibré, l'accuracy peut masquer une mauvaise détection de la classe minoritaire.

Selon le problème, on peut utiliser :
- ROC-AUC pour mesurer la discrimination.
- PR-AUC / Average Precision pour certains problèmes à événements rares.
- F1 pour équilibrer précision et rappel selon leur importance.
- Log Loss pour évaluer les probabilités.
- MAE ou RMSE pour la régression.
La métrique doit être choisie selon l'objectif métier, pas uniquement parce qu'elle produit le meilleur score.

### 8.4 Ignorer le coût de calcul

Une optimisation exhaustive peut être inutilement coûteuse.

Il est souvent préférable de commencer avec une recherche grossière, puis d'affiner l'espace autour des régions prometteuses.

### 8.5 Oublier la reproductibilité

Pour rendre une expérience reproductible, il faut enregistrer les seeds, versions des bibliothèques, paramètres, scores et stratégie de validation.

## 9. Grid Search, Random Search ou Optuna ?

Critère	Grid Search	Random Search	Optuna (TPE)
Stratégie	Grille exhaustive	Échantillonnage	Recherche adaptative
Petit espace discret	Très adapté	Adapté	Adapté
Grand espace	Coûteux	Généralement pratique	Souvent intéressant
Hyperparamètres continus	Grille discrétisée	Naturel	Naturel
Essais conditionnels	Plus contraignant	Possible selon implémentation	Flexible
Utilise les essais précédents	Non	Non	Oui, avec TPE
Complexité d'utilisation	Faible	Faible	Moyenne

Règle pratique : commence par une baseline et un espace de recherche raisonnable. Le choix de l'outil vient ensuite.

## 10. Coding Challenge — Optimiser un Random Forest

Challenge 07 — Hyperparameter Optimization
Projet guidé · validation par livrables
Objectif : entraîner un RandomForestClassifier, optimiser ses hyperparamètres avec Optuna et comparer la performance obtenue à une baseline non optimisée.

~~~python
import optuna
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    # TODO : autres hyperparamètres, estimateur seedé et F1 macro en CV.
    raise NotImplementedError("Compléter la fonction objective")
# TODO : étude avec budget explicite, refit puis une évaluation finale.
~~~

Critères de réussite
Tu dois pouvoir expliquer pourquoi tu as choisi f1_macro, interpréter les meilleurs hyperparamètres et déterminer si l'optimisation a réellement apporté un gain sur le test final.

Un modèle optimisé ne surpasse pas nécessairement la baseline, particulièrement sur un petit dataset.

## 11. Mini-quiz interactif

[[quiz]]

## 12. Questions d'entretien — ML Engineer / Data Scientist

[[interview]]

## 13. Validation du cours

[[assessment]]

À retenir : le Hyperparameter Tuning consiste à rechercher une configuration qui généralise bien, et non simplement celle qui maximise un score observé. Un protocole de validation rigoureux reste plus important qu'un algorithme de recherche sophistiqué.

Le prochain cours sera le cours 8 — Méthodologie complète d'un projet Machine Learning, qui clôturera le premier chapitre avant le projet final de synthèse.