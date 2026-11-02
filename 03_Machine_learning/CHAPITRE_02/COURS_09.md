Dans le cours précédent, nous avons étudié la réduction de dimension avec PCA, SVD, t-SNE et UMAP.

Aujourd'hui, nous allons aborder l'un des problèmes les plus importants en Machine Learning appliqué : comment entraîner et évaluer correctement un modèle lorsque la classe que nous voulons détecter est très rare ?

C'est une situation fréquente dans la détection de fraude, la maintenance prédictive et la modélisation du risque de crédit.


## 1. Qu'est-ce qu'un dataset déséquilibré ?

Un dataset est déséquilibré lorsqu'une classe possède beaucoup plus d'observations qu'une autre.

Prenons un dataset de 10 000 clients :
Exemple : prédiction du défaut de paiement
Clients sans défaut
9 800
Classe majoritaire


Clients en défaut
200
Classe minoritaire






98 % sans défaut

2 % en défaut


Le problème apparaît lorsqu'un modèle apprend principalement à prédire la classe majoritaire.
Exemple trompeur
Imaginons un modèle qui prédit toujours :
\[
\hat y=0
\]

C'est-à-dire « aucun défaut ».

Son accuracy est :
\[
Accuracy=\frac{9800}{10000}=98\%
\]

98 % d'accuracy, mais aucun défaut détecté.

Un score élevé peut masquer un modèle inutilisable pour la détection de la classe minoritaire.

C'est pourquoi nous devons comprendre les métriques avant même de choisir une technique de rééquilibrage.



## 2. La matrice de confusion

La matrice de confusion compare les prédictions aux vraies classes.

Dans notre exemple, nous considérons :

- Classe 0 : client sans défaut.
- Classe 1 : client en défaut.


	Prédit : sans défaut	Prédit : défaut
Réel : sans défaut	TN	FP
Réel : défaut	FN	TP

Les quatre termes :

- TP — True Positive : défaut correctement détecté.
- TN — True Negative : non-défaut correctement détecté.
- FP — False Positive : client sans défaut signalé comme en défaut.
- FN — False Negative : défaut non détecté.

Exemple
Supposons que le modèle produise :

	Prédit 0	Prédit 1
Réel 0	9 500	300
Réel 1	80	120

L'accuracy est :
\[
Accuracy=\frac{9500+120}{10000}=96.2\%
\]

Elle est inférieure aux 98 % du modèle qui prédit toujours zéro.

Mais ce nouveau modèle détecte 120 défauts sur 200.

Il peut donc être bien plus utile selon le coût métier des erreurs.




## 3. Precision, Recall et F1-score


### 3.1 Precision

La précision mesure la proportion de prédictions positives qui sont réellement positives.
\[
Precision=\frac{TP}{TP+FP}
\]

Dans notre exemple :
\[
Precision=\frac{120}{120+300}\approx28.6\%
\]

Parmi les clients signalés comme en défaut, environ 28,6 % le sont réellement.

### 3.2 Recall — Sensibilité

Le recall mesure la proportion de vrais positifs détectés.
\[
Recall=\frac{TP}{TP+FN}
\]

\[
Recall=\frac{120}{120+80}=60\%
\]

Le modèle détecte 60 % des défauts.

### 3.3 F1-score

Le F1-score combine Precision et Recall à l'aide de leur moyenne harmonique :
\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]

Dans notre exemple :
\[
F1\approx38.7\%
\]

Le F1-score pénalise les situations où l'une des deux métriques est très faible.
Tableau récapitulatif

Métrique	Question à laquelle elle répond
Accuracy	Quelle proportion totale de prédictions est correcte ?
Precision	Parmi les positifs prédits, combien sont réellement positifs ?
Recall	Parmi les vrais positifs, combien sont détectés ?
F1-score	Quel est l'équilibre entre Precision et Recall ?
Specificity	Parmi les vrais négatifs, combien sont correctement identifiés ?

La spécificité est :
\[
Specificity=\frac{TN}{TN+FP}
\]




## 4. Explorateur interactif — Precision versus Recall


[[metrics]]

Les curseurs modifient indépendamment une matrice fictive, sans simuler le seuil d’un modèle.




## 5. ROC-AUC

La courbe ROC représente :
\[
TPR=\frac{TP}{TP+FN}
\]

en fonction de :
\[
FPR=\frac{FP}{FP+TN}
\]

pour différents seuils de classification.

L'aire sous cette courbe est appelée ROC-AUC.

Une ROC-AUC de 0,5 correspond généralement à une discrimination équivalente au hasard, tandis qu'une valeur de 1 représente une séparation parfaite.
Limite avec les classes rares
La ROC-AUC peut rester élevée alors que le nombre de faux positifs est important par rapport au nombre de vrais positifs.

En présence d'une classe positive très rare, il est donc utile de compléter la ROC-AUC avec la courbe Precision-Recall.



## 6. PR-AUC et Average Precision

La courbe Precision-Recall représente la precision en fonction du recall pour différents seuils.

Elle est particulièrement informative lorsque la classe positive est rare.
Pourquoi ?
Avec 2 % de positifs, un modèle qui prédit au hasard selon une sélection indépendante de la cible possède une précision attendue d'environ 2 %.

Un modèle atteignant une précision de 30 % peut donc représenter une amélioration importante par rapport à cette référence.
Python
~~~python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)
print("ROC-AUC :", roc_auc)
print("Average Precision :", average_precision)
~~~




Nuance importante : average_precision_score calcule l'Average Precision (AP), une somme pondérée des précisions selon les variations du recall. Ce n'est pas exactement la même chose qu'une aire calculée par interpolation trapézoïdale sous la courbe Precision-Recall.

En pratique, AP est une métrique couramment utilisée pour évaluer les classifieurs sur des données déséquilibrées.



## 7. Balanced Accuracy

La Balanced Accuracy est la moyenne du recall de chaque classe.

En classification binaire :
\[
BalancedAccuracy=
\frac{Sensitivity+Specificity}{2}
\]

Cette métrique évite qu'une classe majoritaire domine autant l'évaluation que dans l'accuracy classique.
~~~python
from sklearn.metrics import balanced_accuracy_score
score = balanced_accuracy_score(y_test, y_pred)
~~~







## 8. Choisir la métrique selon le métier

Objectif	Métriques utiles
Détecter un maximum de défauts	Recall, PR-AUC
Limiter les fausses alertes	Precision
Équilibrer Precision et Recall	F1 ou F-beta
Évaluer la discrimination globale	ROC-AUC
Équilibrer les performances des classes	Balanced Accuracy
Prédire des probabilités fiables	Brier Score, Log Loss, calibration

Dans un projet de risque de crédit, il faut distinguer la capacité à classer les clients de la qualité des probabilités prédites.

Un modèle peut obtenir une excellente ROC-AUC tout en fournissant des probabilités mal calibrées.

C'est particulièrement important si la sortie du modèle est utilisée comme une probabilité de défaut.

Partie II — Les méthodes de rééquilibrage



## 9. Random Oversampling

Le Random Oversampling consiste à augmenter le nombre d'observations de la classe minoritaire en rééchantillonnant des observations existantes avec remise.
Exemple
Dataset initial :
Majoritaire
900

Minoritaire
100


Après oversampling jusqu'à l'équilibre :
Majoritaire
900

Minoritaire
900


Implémentation
~~~python
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
~~~




Avantages

- Simple à comprendre.
- Ne supprime pas les observations majoritaires.
- Peut améliorer l'apprentissage de la classe minoritaire.

Limites

- Duplique des observations.
- Peut augmenter le coût d'entraînement.
- Peut favoriser le surapprentissage de certaines observations.



## 10. Random Undersampling

Le Random Undersampling consiste à réduire le nombre d'observations de la classe majoritaire.
Exemple
Avant :
\[
900\text{ majoritaires},\quad100\text{ minoritaires}
\]

Après :
\[
100\text{ majoritaires},\quad100\text{ minoritaires}
\]

Implémentation
~~~python
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
~~~




Avantages

- Réduit le temps d'entraînement.
- Peut être utile sur de très grands datasets.
- Simplifie l'apprentissage.

Limites
Nous supprimons potentiellement beaucoup d'informations provenant de la classe majoritaire.

Cela peut détériorer la capacité de généralisation.

Partie III — SMOTE



## 11. Qu'est-ce que SMOTE ?

SMOTE signifie Synthetic Minority Over-sampling Technique.

Contrairement au Random Oversampling, qui duplique des observations, SMOTE crée de nouvelles observations synthétiques par interpolation entre des exemples de la classe minoritaire.
Principe
Pour une observation minoritaire \(x_i\), SMOTE choisit un voisin minoritaire \(x_j\).

Il génère :
\[
x_{\mathrm{new}}=x_i+\lambda(x_j-x_i)
\]

Avec :
\[
\lambda\sim U(0,1)
\]

Le nouveau point se trouve donc sur le segment reliant les deux observations.
Exemple
Supposons deux observations :
\[
x_i=(2,3)
\]

\[
x_j=(6,7)
\]

Avec :
\[
\lambda=0.25
\]

Nous obtenons :
\[
x_{\mathrm{new}}=(2,3)+0.25(4,4)
\]

\[
x_{\mathrm{new}}=(3,4)
\]

[[smote]]




## 12. SMOTE avec Python

La bibliothèque imbalanced-learn fournit une implémentation de SMOTE.
~~~shell
pip install imbalanced-learn
~~~



~~~python
from imblearn.over_sampling import SMOTE
from collections import Counter
print("Avant :", Counter(y_train))
smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("Après :", Counter(y_resampled))
# Numérique uniquement ; mise à l’échelle ajustée sur train avant les distances.
~~~




Le paramètre k_neighbors
Il définit le nombre de voisins minoritaires considérés pour la génération des observations synthétiques.

La valeur par défaut est généralement 5.

Lorsque la classe minoritaire contient très peu d'observations, il peut être nécessaire de réduire ce paramètre.



## 13. Les limites de SMOTE

SMOTE est utile, mais il ne faut pas le considérer comme une solution automatique.
Limite 1 — Génération de points peu réalistes
Une interpolation entre deux clients peut produire une combinaison de variables qui ne correspond pas à une situation métier cohérente.
Limite 2 — Zones de chevauchement
Si les classes sont proches, SMOTE peut générer des observations synthétiques dans des zones ambiguës.
Limite 3 — Variables catégorielles
Le SMOTE classique n'est pas conçu pour interpoler directement des catégories nominales.

Pour des datasets contenant à la fois des variables numériques et catégorielles, il existe notamment SMOTENC.
Limite 4 — Distances et scaling
SMOTE utilise des voisins proches.

Les échelles des variables numériques peuvent donc influencer la sélection des voisins.

Il est souvent pertinent de standardiser les variables numériques avant d'appliquer SMOTE.
Limite 5 — Calibration
Le rééquilibrage modifie la distribution des classes utilisée pendant l'entraînement.

Les probabilités produites par le modèle peuvent alors ne plus correspondre à la fréquence réelle des événements.

Cela nécessite une attention particulière lorsque l'objectif est d'estimer des probabilités.




## 14. Variantes de SMOTE

Méthode	Principe
SMOTE	Interpolation entre voisins minoritaires
BorderlineSMOTE	Se concentre davantage sur les observations proches des frontières
ADASYN	Génère davantage d'observations dans certaines zones difficiles
SMOTENC	Adapté aux données mixtes numériques et catégorielles
SMOTEENN	Combine SMOTE et nettoyage par Edited Nearest Neighbours
SMOTETomek	Combine SMOTE et suppression de certains liens de Tomek

Ces méthodes peuvent être intéressantes, mais leurs performances doivent être vérifiées expérimentalement.

Partie IV — Class Weights



## 15. Rééquilibrer sans générer de nouvelles observations

Une alternative consiste à attribuer un poids plus élevé aux observations de la classe minoritaire.

L'idée est de pénaliser davantage les erreurs sur cette classe.
Exemple avec la régression logistique
La fonction de perte logistique pondérée peut s'écrire :
\[
L=-\frac{1}{n}\sum_{i=1}^{n}w_{y_i}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
\]

Où \(w_{y_i}\) représente le poids associé à la classe de l'observation.
Avec scikit-learn
~~~python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)
~~~




Pour le mode balanced, scikit-learn utilise une pondération inversement proportionnelle à la fréquence des classes :
\[
w_c=\frac{n}{K\,n_c}
\]

Avec :

- \(n\) : nombre total d'observations.
- \(K\) : nombre de classes.
- \(n_c\) : nombre d'observations de la classe \(c\).

[[classweights]]

Class Weights versus SMOTE

Critère	Class Weights	SMOTE
Ajoute des observations	Non	Oui
Modifie la fonction d'apprentissage	Oui	Indirectement
Dépend des distances	Non	Oui
Risque de points synthétiques irréalistes	Non	Oui
Compatible avec tous les modèles	Non, selon l'estimateur	Selon les données et le pipeline
Calibration à vérifier	Oui	Oui

Conseil pratique : tester d'abord une baseline sans rééquilibrage, puis class_weight="balanced" et enfin SMOTE ou d'autres méthodes si nécessaire.

Partie V — Le seuil de décision



## 16. Pourquoi le seuil 0,5 n'est-il pas toujours optimal ?

Un classifieur probabiliste prédit une probabilité :
\[
P(Y=1\mid X)
\]

Par défaut, une classification binaire est souvent obtenue avec :
\[
\hat y=
\begin{cases}
1 & \text{si }P(Y=1\mid X)\geq0.5\\
0 & \text{sinon}
\end{cases}
\]

Mais le seuil 0,5 n'est pas nécessairement adapté à l'objectif métier.
Exemple
Supposons trois clients :

Client	Probabilité prédite
A	0,12
B	0,35
C	0,72

Avec un seuil de 0,5, seul C est classé positif.

Avec un seuil de 0,3, B et C sont classés positifs.

Le seuil permet donc de modifier le compromis entre les faux positifs et les faux négatifs.



## 17. Explorateur interactif — Modifier le seuil


[[threshold]]

Sur ce jeu fixe, baisser le seuil ne peut pas diminuer le recall ; la precision peut varier dans les deux sens.




## 18. Comment choisir le seuil optimal ?

Cela dépend du coût des erreurs.

Dans certains contextes, manquer un événement positif peut coûter plus cher qu'une fausse alerte.

Nous pouvons définir une fonction de coût :
\[
Cost= C_{FN}\times FN+C_{FP}\times FP
\]

Où :

- \(C_{FN}\) : coût d'un faux négatif.
- \(C_{FP}\) : coût d'un faux positif.

Nous pouvons choisir un seuil qui minimise ce coût sur un jeu de validation.
Exemple Python
~~~python
import numpy as np
from sklearn.metrics import confusion_matrix
thresholds = np.linspace(.05, .95, 19)
cost_fn, cost_fp = 10, 1
results = []
for threshold in thresholds:
    y_pred = (y_valid_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()
    results.append((threshold, cost_fn * fn + cost_fp * fp))
best_threshold, best_cost = min(results, key=lambda x: x[1])
print("Meilleur seuil :", best_threshold)
print("Coût estimé :", best_cost)
~~~




Les coûts doivent être justifiés par le contexte métier.

Le seuil doit être choisi sur des données de validation, pas directement sur le test final.

Partie VI — Data Leakage et pipelines



## 19. L'erreur la plus grave avec SMOTE

Voici une erreur très fréquente :
~~~python
# MAUVAISE APPROCHE : ne pas exécuter ce protocole pour évaluer un modèle.
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
~~~




Pourquoi est-ce incorrect ?

Parce que des observations synthétiques peuvent être générées en utilisant des observations qui se retrouveront ensuite dans le test.

Nous introduisons ainsi une fuite d'information.

De plus, le test ne représente plus nécessairement la distribution réelle des classes.
Bonne approche
Nous séparons d'abord les données.

Puis nous appliquons SMOTE uniquement sur le train.

Mais en cross-validation, il faut aller plus loin : SMOTE doit être appliqué uniquement au fold d'entraînement de chaque itération.



## 20. Pipeline correct avec imbalanced-learn

~~~python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)),
                     ("model", LogisticRegression(max_iter=1000))])
scores = cross_val_score(pipeline, X_train, y_train,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                         scoring="average_precision")
print(scores.mean())
~~~




Le pipeline imblearn applique SMOTE pendant l'entraînement, sans rééchantillonner les observations de validation ou de test lors de la prédiction.

Il faut utiliser imblearn.pipeline.Pipeline et non le Pipeline classique de scikit-learn lorsque nous intégrons un sampler comme SMOTE.

Partie VII — Coding Lab complet



## 21. Comparer quatre stratégies sur un dataset déséquilibré

Nous allons comparer :
1. Régression logistique classique.
2. Régression logistique avec poids équilibrés.
3. Random Oversampling.
4. SMOTE.
Nous utiliserons un dataset synthétique contenant 5 % de positifs.
~~~python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, average_precision_score,
                             roc_auc_score, balanced_accuracy_score, brier_score_loss)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

X, y = make_classification(n_samples=5000, n_features=20, n_informative=8,
                           n_redundant=5, weights=[.95, .05], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"ap": "average_precision", "roc_auc": "roc_auc",
           "balanced_accuracy": "balanced_accuracy", "recall": "recall"}


def make_models():
    return {name: Pipeline([("scaler", StandardScaler()), ("sampler", sampler),
                             ("model", LogisticRegression(class_weight=weight, max_iter=1000))])
            for name, sampler, weight in [
                ("Baseline", "passthrough", None),
                ("Balanced weights", "passthrough", "balanced"),
                ("Random oversampling", RandomOverSampler(random_state=42), None),
                ("SMOTE", SMOTE(random_state=42), None)]}


if __name__ == "__main__":
    candidates = {}
    for name, pipeline in make_models().items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        assert np.isfinite(scores["test_ap"]).all()
        print(name, {metric: (scores["test_" + metric].mean(), scores["test_" + metric].std())
                     for metric in scoring})
        candidates[name] = (scores["test_ap"].mean(), pipeline)
    name, (_, model) = max(candidates.items(), key=lambda pair: pair[1][0])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    assert len(pred) == len(y_test)  # Le test n'est pas rééchantillonné.
    print("Choix par Average Precision CV :", name)
    print("Test final : AP", average_precision_score(y_test, proba),
          "ROC-AUC", roc_auc_score(y_test, proba), "Balanced Accuracy", balanced_accuracy_score(y_test, pred),
          "Brier", brier_score_loss(y_test, proba))
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(classification_report(y_test, pred, zero_division=0))
    # Optionnel : python ch02_cours09_lab.py --plot
    import sys
    if "--plot" in sys.argv:
        from sklearn.metrics import PrecisionRecallDisplay
        import matplotlib.pyplot as plt
        PrecisionRecallDisplay.from_predictions(y_test, proba)
        plt.show()
~~~




Questions d'analyse
1. Quelle méthode obtient la meilleure Average Precision ?
2. Les poids équilibrés et SMOTE produisent-ils des résultats similaires ?
3. Quelle méthode obtient la meilleure Balanced Accuracy ?
4. Pourquoi la méthode ayant la meilleure ROC-AUC n'est-elle pas nécessairement celle ayant la meilleure Average Precision ?
5. Pourquoi devons-nous vérifier la calibration après rééquilibrage ?
Étape supplémentaire : évaluation finale
Après avoir sélectionné une stratégie avec la cross-validation, entraîne-la sur l'ensemble du train, puis évalue-la une seule fois sur le test indépendant.

Compare également les courbes Precision-Recall et les matrices de confusion.




## 22. Coding Challenge — Construire un système de classification robuste

Challenge 09 — Imbalanced Classification

+400 XP

Construis un pipeline complet de classification déséquilibrée, avec comparaison des méthodes et choix du seuil.
~~~python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from ch02_cours09_lab import X_train, X_test, y_train, y_test

# Réserver la validation du seuil AVANT la recherche des hyperparamètres.
X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=.25,
                                               stratify=y_train, random_state=43)
pipeline = Pipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)),
                     ("model", LogisticRegression(max_iter=1000))])
search = GridSearchCV(pipeline, {"smote__k_neighbors": [3, 5, 7], "model__C": [.01, .1, 1, 10]},
                      scoring="average_precision",
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
# TODO : fit sur X_fit, y_fit ; afficher les paramètres et scores.
# TODO : choisir le seuil minimisant 10*FN + FP sur X_valid, y_valid uniquement.
# TODO : figer modèle et seuil, puis évaluer une seule fois sur le test.
# BONUS : comparer RandomForest(class_weight="balanced") sur le train.
~~~




Bonus
Compare les résultats avec un modèle RandomForestClassifier(class_weight="balanced").

Analyse également la stabilité des performances lorsque la proportion de positifs diminue.



[[solution]]




## 23. Questions d'entretien — Data Scientist / ML Engineer


[[interview]]




## 24. Mini-quiz interactif


[[quiz]]




## 25. Validation du cours


[[assessment]]

À retenir : dans un problème de classification déséquilibrée, le rééquilibrage n'est pas toujours nécessaire. Il faut d'abord choisir les bonnes métriques, construire une baseline solide et comparer les stratégies sur une validation représentative.

Le prochain cours sera le dernier cours du chapitre 2 : Pipelines avancés et automatisation du preprocessing, suivi d'un projet pratique réunissant toutes les techniques étudiées.