# Machine Learning — roadmap en douze chapitres

Ordre demandé par l’apprenant le 1 novembre 2026. Les introductions existantes sont conservées ; les chapitres complets restent à développer et à valider.

## Chapitre 1 — Introduction et fondements du Machine Learning

Cadrer une tâche, distinguer apprentissage et généralisation, définir une baseline et séparer train, validation et test.

Preuve attendue : Concevoir et défendre un protocole sans fuite de données.

État : ressource introductive disponible ; chapitre complet à produire.

## Chapitre 2 — Préparation des données et Feature Engineering

Types, absences, jointures, transformations, encodages et pipelines ajustés exclusivement sur train.

Preuve attendue : Livrer un pipeline de préparation avec contrats et tests.

État : ressource introductive disponible ; chapitre complet à produire.

## Chapitre 3 — Supervised Learning

Programme fourni le 2 novembre 2026 : dix cours, de la régression linéaire à la comparaison des modèles supervisés.

Cours 01 disponible : Régression linéaire simple et multiple, avec laboratoire synthétique, challenge OLS NumPy, descente de gradient, quiz et autoévaluation.

Cours 02 disponible : Ridge, Lasso et Elastic Net, avec laboratoire, recherche d’hyperparamètres et régularisation illustrée.

Cours 03 disponible : régression polynomiale, interactions et sélection du degré avec régularisation.

Cours 04 disponible : régression logistique, probabilités, odds ratios, log-loss, seuil, régularisation et calibration.

Cours 05 disponible : KNN, distances, voisinage, classification/régression, scaling, dimension et comparaison à la logistique.

Cours 06 et 07 disponibles : Decision Trees et Random Forest/Bagging, avec pruning, bootstrap, OOB, validation et laboratoires.

Cours 08 à 10 : boosting ; SVM ; comparaison, évaluation et sélection. Supports détaillés à venir.

Preuve attendue : implémenter, entraîner et comparer des modèles sous un protocole indépendant avec les métriques adaptées.

## Chapitre 4 — Régression linéaire et régularisation

Moindres carrés, gradient, hypothèses, Ridge, Lasso, ElasticNet et analyse des résidus.

Preuve attendue : Implémenter une régression et comparer les effets de régularisation.

État : ressource introductive disponible ; chapitre complet à produire.

## Chapitre 5 — Régression logistique

Sigmoïde, vraisemblance, entropie croisée, gradient, softmax et régularisation.

Preuve attendue : Dériver et implémenter la logistique puis vérifier ses gradients.

État : à rédiger.

## Chapitre 6 — KNN, Naive Bayes et SVM

Distances et échelles, hypothèses d’indépendance, marges et noyaux.

Preuve attendue : Comparer les méthodes avec une validation équitable et expliquer leurs erreurs.

État : à rédiger.

## Chapitre 7 — Decision Trees et Random Forest

Impureté, splits, élagage, bagging, sélection de variables et variance.

Preuve attendue : Coder un arbre minimal, puis comparer arbre et forêt.

État : à rédiger.

## Chapitre 8 — Gradient Boosting, XGBoost, LightGBM et CatBoost

Construction additive, pertes, régularisation, early stopping et particularités des outils.

Preuve attendue : Comparer des modèles à budget de recherche défini et documenter les versions.

État : à rédiger.

## Chapitre 9 — Clustering et apprentissage non supervisé

k-means, GMM/EM, DBSCAN, hiérarchique et limites des métriques internes.

Preuve attendue : Interpréter des groupes sans les confondre avec une vérité terrain.

État : à rédiger.

## Chapitre 10 — PCA, réduction de dimension et détection d’anomalies

Projection, variance, SVD, visualisations non linéaires, Isolation Forest et seuils.

Preuve attendue : Expliquer une projection et évaluer un détecteur selon le cas d’usage.

État : à rédiger.

## Chapitre 11 — Séries temporelles

Tendance, saisonnalité, lags, validation temporelle, modèles naïfs et backtesting.

Preuve attendue : Comparer des prévisions sans utiliser des informations du futur.

État : à rédiger.

## Chapitre 12 — ML avancé, interprétabilité, robustesse et études de cas

Explications, permutation, SHAP/LIME, décalage de distribution, robustesse, incertitude et décision métier.

Preuve attendue : Défendre un système complet, ses limites et son comportement après changement des données.

État : à rédiger.


Chapitre 3 — Supervised Learning : les 10 cours fournis sont disponibles, avec scripts, quiz, questions d’entretien et projet final Credit Default Prediction Benchmark.
