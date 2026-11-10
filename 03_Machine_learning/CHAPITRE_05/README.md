# Chapitre 5 — Machine Learning avancé

Mis à jour le 10 novembre 2026. Les dix cours sont intégrés à LearnLoop. Les cours 1 et 2 s'appuient sur les scripts existants ; les cours 3 à 10 reprennent les supports fournis, avec code remis en forme, quiz et interactions.

## Programme

1. [Évaluation avancée des modèles](COURS_01.md)
2. [Calibration des probabilités](COURS_02.md)
3. [Explainable AI (XAI) — SHAP et LIME](COURS_03.md)
4. [Robustesse des modèles Machine Learning](COURS_04.md)
5. [Incertitude des prédictions et Conformal Prediction](COURS_05.md)
6. [Fairness, Bias & Responsible Machine Learning](COURS_06.md)
7. [Model Diagnostics & Error Analysis](COURS_07.md)
8. [Data Drift, Concept Drift & Model Monitoring](COURS_08.md)
9. [Model Validation, Backtesting & Statistical Testing](COURS_09.md)
10. [End-to-End ML Evaluation & Governance Framework](COURS_10.md)

## Pratiquer les TODO

Dans le cours ou l'onglet Exercices, complète le fichier de départ puis clique sur « Exécuter et vérifier ». La console affiche les sorties et erreurs avec leur ligne ; les graphiques apparaissent sous le code. Après le premier démarrage, la syntaxe est vérifiée pendant la saisie. Tab indente et Ctrl/⌘ + Entrée exécute. Le code est sauvegardé dans le compte ; le bouton « Mon fichier .py » permet de l'exporter.

Les tests disponibles vérifient des comportements précis. Une exécution réussie sans tests ne prouve pas que toute la consigne est satisfaite ; le message de l'atelier le précise. Les tests du chapitre 5 ciblent notamment additivité, perturbations, quantile conforme, fairness, pertes, PSI, bootstrap et triage.

## Scripts locaux

Chaque cours possède ch05_coursNN_lab.py (démonstration), _starter.py (TODO) et _solution.py (corrigé). Télécharge les fichiers auxiliaires requis dans le même dossier. Certains starters utilisent leur _solution.py comme banc d'essai ; cela permet de tester la fonction complétée dans un protocole reproductible.

Dépendances de base : Python, numpy, pandas, scipy, scikit-learn >= 1.6 et matplotlib. Les exemples sont synthétiques. Les durées du site sont indicatives, hors mini-projets.

- Cours 3 : l'atelier NumPy explique l'additivité ; SHAP, LIME et XGBoost réels se travaillent en Python local. Installe shap, lime et xgboost, puis lance python ch05_cours03_lab.py --plot.
- Cours 4 : python ch05_cours04_lab.py --plot ; --xgboost ajoute le modèle local.
- Cours 5 : python ch05_cours05_lab.py --plot ; --xgboost ajoute la comparaison locale.
- Cours 6 : --fairlearn illustre l'audit et ThresholdOptimizer sur validation ; nécessite fairlearn. --xgboost ajoute ce modèle.
- Cours 7 : --curves calcule les courbes d'apprentissage, --plot affiche pertes et résidus.
- Cours 8 : --evidently produit un rapport HTML local avec la famille d'API Evidently 0.7.x.
- Cours 9 : --plot affiche les différences AUC bootstrap appariées.
- Cours 10 : le lab produit un MVP d'évaluation, un triage explicite et une Model Card JSON. Le projet final ML Trust & Governance Platform reste un exercice d'intégration, avec contrôles non évalués et revue humaine à documenter.

## Suite

Le parcours comporte désormais 48 cours ML sur cinq chapitres : 8 + 10 + 10 + 10 + 10. Cela décrit les supports disponibles, pas les cours personnellement validés. Le chapitre 6 proposé porte sur les fondements du Deep Learning et reste à produire.
