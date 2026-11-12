# Chapitre 6 — Séries temporelles et Machine Learning spécialisé

Chapitre complet, mis à jour le 12 novembre 2026 : **12 cours disponibles**, 60 questions de quiz, supports détaillés, interactions et exercices Python.

| Partie | Contenu | Cours | Nombre |
| --- | --- | --- | --- |
| A | Séries temporelles | 01 à 07 | 7 |
| B | Machine Learning spécialisé | 08 à 12 | 5 |
| Total | | 01 à 12 | 12 |

## Programme

| Cours | Support | Partie | Durée indicative |
| --- | --- | --- | --- |
| 01 | [Fondamentaux des séries temporelles](COURS_01.md) | A | 2–3 h |
| 02 | [Baselines et validation temporelle](COURS_02.md) | A | 2–3 h |
| 03 | [Modèles statistiques classiques : ETS, ARIMA, SARIMA et ARIMAX](COURS_03.md) | A | 3–4 h |
| 04 | [Forecasting avec le Machine Learning](COURS_04.md) | A | 3–4 h |
| 05 | [Multi-séries, prévision hiérarchique et demande intermittente](COURS_05.md) | A | 3–4 h |
| 06 | [Prévision probabiliste : quantiles, incertitude et Conformal Prediction](COURS_06.md) | A | 3–4 h |
| 07 | [Détection d'anomalies, ruptures temporelles et drift](COURS_07.md) | A | 3–4 h |
| 08 | [Ensemble Learning avancé : Voting, Stacking et Blending](COURS_08.md) | B | 3–4 h |
| 09 | [Cost-Sensitive Learning et prise de décision optimale](COURS_09.md) | B | 3–4 h |
| 10 | [Survival Analysis : Kaplan-Meier, Cox et Machine Learning de survie](COURS_10.md) | B | 3–4 h |
| 11 | [Recommender Systems & Learning to Rank](COURS_11.md) | B | 3–4 h |
| 12 | [Data-Centric Machine Learning](COURS_12.md) | B | 3–4 h |

Prévoir environ 34 à 46 h pour les supports, selon le rythme personnel ; les exercices et approfondissements peuvent demander davantage de temps. La durée affichée par le site correspond à une estimation centrale de 40 h.

## Travailler avec les scripts

Chaque cours possède trois fichiers :

- `ch06_coursNN_lab.py` : démonstration complète exécutable.
- `ch06_coursNN_starter.py` : exercice avec TODO.
- `ch06_coursNN_solution.py` : corrigé du challenge.

Les 36 scripts sont présents dans ce dossier et dans `site/public/downloads/`. Dans LearnLoop, les exercices NumPy/pandas sont exécutables dans l’atelier du navigateur avec vérifications comportementales. Les labs complets et les extensions utilisant des bibliothèques natives se lancent localement. La réussite d’un quiz débloque le cours suivant selon la progression existante.

## Dépendances

Socle : Python, NumPy, pandas, Matplotlib, SciPy et scikit-learn. Pour les labs des cours 03 à 12 :

~~~shell
python -m pip install numpy pandas matplotlib scipy scikit-learn statsmodels lightgbm ruptures lifelines cleanlab
~~~

Extensions optionnelles des supports : `shap` (cours 04), `scikit-survival` (cours 10, option `--rsf`), `scikit-surprise` et `implicit` (cours 11). Leur installation dépend de la version de Python et de la disponibilité de roues adaptées. Le lab 11 inclut une factorisation sur notes observées et un ALS implicite en NumPy sans imposer Surprise ou implicit.

## Exécuter les labs

Depuis ce dossier, remplacer NN par le numéro du cours :

~~~shell
python ch06_cours01_lab.py
python ch06_cours03_lab.py --plot
python ch06_cours04_lab.py --plot
python ch06_cours05_lab.py --plot
python ch06_cours06_lab.py --plot
python ch06_cours07_lab.py --plot
python ch06_cours08_lab.py
python ch06_cours09_lab.py --plot
python ch06_cours10_lab.py --plot
python ch06_cours11_lab.py
python ch06_cours12_lab.py
~~~

Le cours 02 affiche directement ses figures. Les cours 04 et 05 acceptent `--engine browser`, une variante Random Forest à distinguer du lab LightGBM. Le cours 10 accepte `--rsf` pour l’extension Random Survival Forest et l’IBS IPCW. Cette extension a été exécutée avec scikit-survival 0.25.0 et scikit-learn 1.7.1. Installer une combinaison compatible : les versions plus récentes de scikit-survival peuvent exiger une version plus récente de scikit-learn.

## Protocoles à respecter

- Cours 01 : STL sur la série complète est exploratoire ; ADF/KPSS sont des diagnostics, pas une preuve automatique de stationnarité.
- Cours 02 à 05 : respecter origine, horizon, chronologie et séparation des séries. Les prévisions à un pas avec historique actualisé diffèrent des prévisions multi-horizon à origine fixe. Sélection SARIMA et réglages ML utilisent le développement, jamais le test final.
- Cours 06 : calibration séparée, réarrangement des quantiles et correction CQR conservatrice. La dépendance temporelle empêche d’affirmer une couverture universelle sous la garantie split conformal classique.
- Cours 07 : rolling/CUSUM en ligne ; STL/PELT rétrospectifs. Le drift ne prouve pas une baisse de performance et les seuils de démonstration ne sont pas universels.
- Cours 08 : méta-modèle entraîné sur OOF ou holdout ; CV stratifiée destinée ici aux données tabulaires indépendantes.
- Cours 09 : calibration dans le train, choix du seuil sur validation, audit final indépendant. Matrice de coûts illustrative ; DCA sans unité monétaire.
- Cours 10 : censure à droite et orientation du risque explicites. Rossi concerne l’arrestation, pas le churn. Vérifier les risques proportionnels ; le diagnostic du lab peut signaler des violations à investiguer. Discrimination et calibration sont distinctes.
- Cours 11 : groupes de requêtes distincts ; NDCG avec gain exponentiel, AP@K normalisée par le total pertinent et ex æquo stables. Exclure les items déjà consommés. Ne pas traiter une interaction inconnue comme une note observée égale à zéro.
- Cours 12 : probabilités OOF sur train, revue simulée avec oracle uniquement dans les données synthétiques, comparaison avec une revue aléatoire de même budget ; aucune correction automatique des labels réels.

## Sources et vérifications

Les fichiers `supports/coursNN_source.txt` conservent les douze pièces jointes sans modification. Les blocs Python aplatis ou tronqués sont remis en forme ou regroupés dans les labs autonomes ; les corrections de protocole sont explicitées dans les supports. Les sections internes des cours ne modifient pas les deux parties A/B du chapitre.

Depuis la racine du dépôt : `python -B site/tests/verify-chapter6-python.py`. Ce script vérifie le catalogue, les quiz, la syntaxe des scripts et les comportements des corrigés ; il vérifie aussi que les TODO non complétés échouent. Les labs utilisant LightGBM, ruptures, lifelines et Cleanlab se vérifient dans un environnement local équipé de leurs dépendances.

Le chapitre 6 termine ce module de ML ; le cours 12 annonce un prochain module consacré au Deep Learning, sans prétendre que ses supports sont déjà disponibles.
