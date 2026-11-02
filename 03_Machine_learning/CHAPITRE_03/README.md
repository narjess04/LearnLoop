# Chapitre 03 — Supervised Learning

Comprendre, entraîner et comparer les modèles supervisés, avec mathématiques, Python et validation reproductible.

01. Régression linéaire simple et multiple — disponible
02. Ridge, Lasso et Elastic Net — disponible
03. Régression polynomiale et relations non linéaires — disponible
04. Régression logistique — disponible
05. K-Nearest Neighbors (KNN) — disponible
06. Decision Trees — disponible
07. Random Forest et Bagging — disponible
08. Gradient Boosting, XGBoost, LightGBM et CatBoost — disponible
09. Support Vector Machines (SVM) — disponible
10. Comparaison des modèles, évaluation et sélection — disponible

Cours 1 : 3–4 heures indicatives, laboratoire synthétique de prix immobiliers, OLS NumPy, descente de gradient, quiz et autoévaluation. Les scripts sont à exécuter dans Python ; --plot active les figures du laboratoire.

Cours 2 : laboratoire de comparaison OLS/Ridge/Lasso/Elastic Net, recherche d’hyperparamètres, 5 questions de quiz, 8 réponses d’entretien et 12 objectifs. Options des scripts : --plot ; --small pour le bonus à 100 observations.

Cours 3 : degrés polynomiaux, interactions, comptage combinatoire, choix du degré et Ridge dans un pipeline. Laboratoire quadratique, challenge cubique et option --nested pour la validation imbriquée ; --plot active les graphiques Python.

Cours 4 : sigmoïde, odds/logit, odds ratios, log-loss, seuil et calibration. Laboratoire synthétique, tuning sur train et validation de seuil distincte, bonus NumPy stable ; --plot active ROC, PR et calibration. 5 questions de quiz, 8 questions d’entretien et 14 objectifs.

Cours 5 : distances, choix de K, votes, scaling et dimension. Laboratoires de classification/régression, recherche et comparaison à la logistique, option --pca pour le challenge ; --plot pour les figures du laboratoire. 6 questions de quiz, 8 questions d’entretien et 14 objectifs.

Cours 6 : CART, Gini/Entropy, pruning, classification/régression, validation et importances. 6 questions et 16 objectifs.

Cours 7 : Bagging, bootstrap, forêt, OOB et importances. 6 questions et 18 objectifs. Les scripts proposent --plot ; le challenge forêt utilise --iterations (défaut 30) pour fixer le budget de recherche.

Cours 8 : cinq modèles, gradient négatif, régularisation, early stopping et tuning. Dépendances : xgboost, lightgbm, catboost. --iterations fixe le budget (défaut 30). 6 questions, 19 objectifs.

Cours 9 : marge, noyaux, hinge et epsilon loss ; grille complète de 25 configurations. 6 questions et 17 objectifs.

Cours 10 : benchmark de cinq familles et projet final à données mixtes. Sélection CV, tuning, calibration et choix du seuil sur validation, puis audit indépendant. Le projet nécessite xgboost. 7 questions et 17 objectifs.

Pour chaque cours, conserver lab/starter/solution dans le même dossier. --plot active les figures. Les données sont synthétiques ; les tests aléatoires ne démontrent pas une validation hors période.
