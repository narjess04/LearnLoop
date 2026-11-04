# Chapitre 04 — Apprentissage non supervisé

10 cours + 1 projet ; 11 h 30 hors exercices.

01. Introduction à l’apprentissage non supervisé — 45 min — disponible
02. K-Means Clustering — 75 min — disponible
03. Clustering hiérarchique — 60 min — disponible
04. DBSCAN et HDBSCAN — 75 min — disponible
05. Gaussian Mixture Models (GMM) — 75 min — disponible
06. Évaluation des algorithmes de clustering — 60 min — disponible
07. Analyse en composantes principales (PCA) — 90 min — disponible
08. t-SNE et UMAP — 75 min — disponible
09. Détection d’anomalies — 75 min — disponible
10. Pipelines avancés et bonnes pratiques — 60 min — disponible

Cours 1 à 10 : supports, quiz, autoévaluation, laboratoires Python et challenges corrigés. Les scripts lab et solution s’exécutent sans affichage ; --plot ouvre les figures. Les squelettes starter sont volontairement à compléter. Dépendances : numpy, pandas, scipy, scikit-learn et matplotlib.

Les données sont synthétiques. Le from-scratch K-Means conserve un centroïde vide et ne garantit pas un optimum global. La coupe du dendrogramme d’exemple est corrigée : 1,5 donne trois groupes ; 2,5 donne deux groupes. Les métriques sont calculées en exécutant les scripts, sans résultat préfabriqué.

Projet final : Customer Segmentation & Anomaly Detection System ; cahier des charges dans LearnLoop. Les dix cours et les corrigés pédagogiques sont disponibles ; le projet final reste à réaliser.


Cours 4 : DBSCAN NumPy complet, HDBSCAN sklearn, bruit et métriques valides. Cours 5 : EM univarié stable, probabilités et comparaison de covariances. Cours 6 : benchmark, métriques au périmètre explicite et stabilité sur intersections d’indices. Les supports approfondis fournis pour les cours 4 à 6 annoncent 90–120 min ; les repères de roadmap d’origine sont conservés séparément. Dépendance : scikit-learn >= 1.3 pour HDBSCAN.

## Cours 7 à 10

- Cours 7 : PCA NumPy complet, lien SVD, coefficients/loadings, reconstruction, seuils 90/95/99 % et comparaison du clustering dans un espace commun. Support : 120 min.
- Cours 8 : PCA/t-SNE/UMAP sur le même sous-échantillon Digits, paramètres, trustworthiness, stabilité des voisinages et temps de calcul. Support : 90–120 min. Options : --samples (600 par défaut), --iterations (750 par défaut), --plot.
- Cours 9 : Isolation Forest, LOF outlier/novelty et One-Class SVM. Lab sur référence normale synthétique, validation séparée pour le choix du modèle et du seuil dans le corrigé, audit final avec precision/recall/F1/AP. Support : 120–150 min.
- Cours 10 : pipeline mixte, PCA/K-Means, anomalies, sélection sur validation, stabilité, inférence, PSI et sauvegarde joblib vérifiée. Support : 120–150 min. --output chemin.joblib conserve le bundle du lab ; sans cette option la sauvegarde est testée dans un répertoire temporaire.

Les 11 h 30 de roadmap sont les repères d’origine ; les durées des supports approfondis sont affichées dans les cours. Dépendances supplémentaires : umap-learn et joblib ; scikit-learn >= 1.6 couvre les nouveaux exemples.

Les sources fournies sont archivées dans supports/. Les blocs Python tronqués ont été reconstruits. Chaque corrigé utilisant un module lab demande de conserver les fichiers du cours dans le même dossier. Les squelettes lèvent volontairement NotImplementedError jusqu’à leur réalisation.

La checklist de 15 livrables du projet final sera accessible dans le cours 10. L’API FastAPI, le dashboard et le monitoring opérationnel restent à développer par l’apprenant ; les téléchargements sont des laboratoires et corrigés pédagogiques.
