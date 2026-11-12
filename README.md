# Recap ML — comprendre, retenir et construire

Ce dossier sert à consolider les acquis de l'année scolaire et à développer une autonomie pratique en machine learning, deep learning et IA générative.

Le but ne se limite pas à mémoriser des cours : il faut savoir expliquer une méthode, la programmer, choisir une évaluation correcte et analyser ses limites. L'expertise se construit progressivement, avec un socle large et des approfondissements ciblés.

## Organisation

| Dossier | Contenu prévu |
| --- | --- |
| 00_Bilan | Inventaire des cours suivis, acquis, lacunes et diagnostic initial |
| 01_Python_et_outils | Python, NumPy, pandas, visualisation, Git, débogage et organisation du code |
| 02_Maths_et_statistiques | Algèbre linéaire, dérivées, probabilités, statistiques et optimisation |
| 03_Machine_learning | Préparation des données, modèles supervisés et non supervisés, validation et métriques |
| 04_Deep_learning | Réseaux de neurones, rétropropagation, entraînement et architectures |
| 05_GenAI | Transformers, modèles génératifs, embeddings, RAG, adaptation et évaluation |
| 06_Projets | Projets personnels avec code, résultats, erreurs et documentation |
| 07_Veille | Notes datées sur les nouveautés, sources originales et essais reproductibles |
| 08_Revisions | Questions de rappel, fiches synthétiques et journal des erreurs |

## Méthode pour chaque notion

1. Répondre à quelques questions sans regarder le cours pour identifier les acquis réels.
2. Étudier le concept : problème traité, intuition, hypothèses et mathématiques utiles.
3. Fermer le support et expliquer la notion avec ses propres mots.
4. Coder un exemple minimal. Pour les algorithmes fondamentaux, réaliser aussi une version simplifiée avec NumPy lorsque cela aide à comprendre.
5. Utiliser une bibliothèque adaptée sur un cas plus réaliste et vérifier les résultats.
6. Modifier les données ou les contraintes et expliquer ce qui change.
7. Consigner les erreurs, puis refaire un exercice sans solution sous les yeux.

La documentation reste autorisée : l'autonomie consiste à résoudre le problème et vérifier son code, pas à retenir toutes les API. Si une aide ou une IA fournit une solution, la relire, la tester, puis refaire l'exercice sans cette solution.

## Révisions

Point de départ ajustable : revoir une notion le lendemain, puis après une semaine et après un mois. Commencer chaque révision par du rappel actif ou un exercice, avant de relire. Rapprocher les révisions lorsque la notion reste fragile.

## Ordre de progression

Commencer par le bilan de l'année dans `00_Bilan/BILAN_INITIAL.md`, puis suivre `PARCOURS.md`. Un thème déjà maîtrisé peut être validé par un exercice et une explication, sans reprendre tout le cours.

Le rythme et le contenu détaillé seront adaptés aux cours réellement suivis, au temps disponible et aux objectifs personnels. Aucun niveau ni délai d'expertise n'est présumé.


## État du parcours — 12 novembre 2026

Le site LearnLoop se trouve dans [site/](site/README.md). Le catalogue ML comprend **60 cours disponibles sur six chapitres**.

| Chapitre | Contenu | Cours |
| --- | --- | --- |
| 1 | Introduction et fondements | 8 |
| 2 | Préparation des données | 10 |
| 3 | Apprentissage supervisé | 10 |
| 4 | Apprentissage non supervisé | 10 |
| 5 | Évaluation avancée, XAI, robustesse et gouvernance | 10 |
| 6 | Séries temporelles et Machine Learning spécialisé | 12 — chapitre complet |

[Catalogue ML](03_Machine_learning/README.md) · [Programme du chapitre 5](03_Machine_learning/CHAPITRE_05/README.md) · [Chapitre 6 complet](03_Machine_learning/CHAPITRE_06/README.md).

### Atelier Python dans les cours

Complète les TODO dans l'éditeur intégré, puis clique sur « Exécuter et vérifier ». Tu retrouves la console, les erreurs avec leur ligne, les graphiques et les tests disponibles. Après le premier lancement, la syntaxe est contrôlée pendant la saisie. Ton code est enregistré dans ton compte et téléchargeable en .py. Le premier chargement du moteur nécessite Internet.

Une exécution sans erreur ne valide pas à elle seule toute la solution : les exercices sans tests le signalent. Les projets utilisant SHAP/LIME/XGBoost ou les intégrations avancées Fairlearn/Evidently se poursuivent dans un IDE Python local ; les scripts et dépendances sont décrits dans le README du chapitre.


Le chapitre 6 comprend deux parties : séries temporelles (cours 01–07) et Machine Learning spécialisé (cours 08–12). Ses douze supports, 60 questions de quiz et 36 scripts Python sont disponibles. Les dépendances, commandes et limites méthodologiques sont documentées dans son README.
