# Recap ML — contrat pédagogique et programme de formation

Ce document intègre le programme proposé le 1 novembre 2026. Il définit le contenu à concevoir ; il ne prétend pas que tous les cours sont déjà écrits. Les premières leçons de l'application sont des introductions, à approfondir selon ce standard.

## 1. Résultat recherché

Former une personne capable de cadrer un problème, comprendre les mathématiques utiles, programmer une solution, l'évaluer honnêtement, la déployer et expliquer ses limites.

Le statut de senior dépend également de l'expérience sur plusieurs systèmes, de la responsabilité et de la collaboration. Ni une durée de formation, ni des XP, ni un certificat ne permettent de le garantir. L'hypothèse de planification proposée de 2 à 3 ans à 10–15 heures par semaine représente environ 1 040 à 2 340 heures : un budget de travail, pas une promesse de niveau.

La cible est une compétence démontrée. Pour chaque objectif, conserver une preuve : exercice inédit, code testé, rapport, décision argumentée ou revue externe.

## 2. Architecture d'un cours

Un parcours contient des modules ; un module traite une compétence cohérente ; une leçon couvre un objectif assez précis pour être exercé et vérifié. Une leçon ne doit pas devenir un catalogue d'outils.

Chaque leçon comprend :

1. **Fiche d'entrée** : objectif observable, prérequis, diagnostic court, durée indicative et place dans le projet.
2. **Problème concret** : quelle difficulté essaie-t-on de résoudre ? Quelle solution simple existe déjà ?
3. **Intuition** : explication accessible, exemple et visualisation, avec les limites de l'analogie.
4. **Notations** : symboles, dimensions, unités et hypothèses définis avant leur utilisation.
5. **Mathématiques** : dérivation étape par étape, conditions de validité, calcul numérique et cas limite.
6. **Code expliqué** : implémentation minimale, formes des données, invariants, tests utiles et résultats effectivement observés.
7. **Application réaliste** : passage à une bibliothèque, protocole d'évaluation, erreurs et arbitrages.
8. **Pratique graduée** : calcul à la main, implémentation, diagnostic d'un bug, transfert vers un cas inédit.
9. **Corrections** : raisonnement complet, alternatives acceptables, erreurs fréquentes et critères d'évaluation.
10. **Quiz** : questions de rappel, interprétation et transfert, avec explication de chaque distracteur.
11. **Questions-réponses** : objections et confusions réelles, conditions où la méthode échoue.
12. **Scénario vidéo** : séquences, visuels, narration, pauses actives et sous-titres. Le texte reste disponible.
13. **Preuve de maîtrise** : tâche sans guide, puis explication orale ou écrite des choix.
14. **Révisions** : rappel actif et variantes à J+1, J+7 et J+30, à ajuster aux difficultés.
15. **Sources et maintenance** : source primaire, version de l'environnement, date de vérification et historique des corrections.

Les durées sont indicatives. Une leçon peut demander plusieurs séances. Les prérequis manquants doivent renvoyer vers une ressource précise, et non laisser l'élève deviner.

## 3. Progression et évaluation

Utiliser les états : à découvrir → en cours → exercé avec aide → démontré sans guide → revérifié dans un autre contexte.

Une réussite au quiz donne un retour formatif et des XP. Elle ne valide pas le module à elle seule. Une case cochée dans un projet représente une déclaration personnelle ; la revue des livrables apporte la validation technique.

Grille de projet : cadrage et données 20 points ; protocole et baseline 25 ; implémentation et tests 20 ; analyse des résultats et limites 20 ; reproduction et communication 15. Seuil pédagogique proposé : 80/100 après revue, avec correction obligatoire de toute fuite de données, métrique inventée, résultat non reproductible ou défaut critique identifié. Ce seuil devra être calibré sur des travaux réels.

Une seconde tentative utilise une variante de l'exercice et demande une explication de l'erreur précédente. Les indices progressent du concept à une piste de code ; la solution complète reste consultable après une tentative.

La gamification valorise la pratique, les révisions et les livrables. Éviter d'assimiler un niveau de jeu à un titre professionnel ou de pénaliser les pauses.

## 4. Parcours complet

Les blocs suivent des dépendances. Les mathématiques se travaillent en parallèle de la programmation et se réactivent au moment des modèles.

### A. Programmation, outils et données

Modules : fonctions et structures ; erreurs et débogage ; modules et environnements ; POO et composition ; itérateurs, générateurs et décorateurs ; typage ; tests pytest ; complexité ; tri et graphes ; programmation dynamique ; terminal Linux ; Git et collaboration ; SQL et modélisation ; NumPy ; pandas ; visualisation ; notebooks reproductibles.

Exercices : expliquer la complexité d'un traitement, déboguer une mutation involontaire, écrire une jointure SQL sans multiplier les lignes, tester une transformation avec valeurs manquantes.

Projet : pipeline d'analyse de données depuis fichiers bruts vers tables validées et rapport. Livrables : CLI, schéma des données, requêtes SQL, tests, environnement et README. Passage : reproduction sur un nouveau fichier et justification des règles de nettoyage.

### B. Mathématiques, probabilités, statistiques et optimisation

Modules : vecteurs, matrices et formes ; projections ; valeurs propres et SVD ; dérivées ; gradient, jacobien et hessien ; règle de chaîne ; convexité ; probabilités conditionnelles ; variables et lois ; espérance, variance et covariance ; vraisemblance ; entropie et divergence KL ; estimation et intervalles ; tests et puissance ; A/B testing ; biais et variance ; introduction bayésienne ; SGD, momentum et Adam.

Chaque concept relie un calcul à la main, une expérience numérique et une utilisation dans un modèle. Enseigner les hypothèses des tests avant leurs commandes.

Projet : laboratoire d'optimisation et d'incertitude. Comparer convergence, échelle des variables, estimation analytique et simulation. Passage : dériver une perte simple, vérifier un gradient et interpréter un intervalle sans confusion sur sa signification.

### C. Machine learning classique

Modules : cadrage et baseline ; partitions et fuite ; prétraitement par pipeline ; régressions linéaire et logistique ; régularisation ; k-NN et Naive Bayes ; arbres et ensembles ; gradient boosting ; SVM ; déséquilibre ; calibration et choix de seuil ; validation croisée et sélection ; feature engineering ; k-means et DBSCAN ; PCA et visualisations non linéaires ; anomalies ; séries temporelles et backtesting ; interprétabilité et limites causales.

Les outils particuliers sont choisis après le problème : scikit-learn en profondeur, puis comparaison raisonnée d'outils de boosting. L'apprentissage ne doit pas dépendre de la mémorisation de noms de bibliothèques.

Projets : classification tabulaire avec coût d'erreur ; prévision temporelle avec backtesting ; segmentation et interprétation. Quelques jeux publics peuvent servir de terrain, mais le score d'une compétition ne remplace pas le protocole.

Passage : défendre la métrique, le split, la baseline, l'incertitude et l'analyse des erreurs sur des données inédites.

### D. Deep learning et vision

Modules : tenseurs et autograd ; perceptron et MLP ; rétropropagation manuelle ; initialisation ; activations et pertes ; boucle PyTorch ; surajustement d'un petit lot ; régularisation ; normalisations ; optimiseurs et schedulers ; précision mixte ; CNN et résidus ; transfert ; détection ; segmentation ; séquences récurrentes ; attention et Transformers ; autoencodeurs ; VAE ; GAN ; diffusion.

Projets : classifieur d'images avec analyse par classe ; segmentation ; reproduction limitée d'un papier avec baseline et ablations.

Passage : expliquer les gradients, diagnostiquer une boucle défaillante, reproduire une expérience et distinguer amélioration mesurée et variation aléatoire.

### E. Modèles de langage et génération

Modules : tokenisation et BPE ; embeddings ; attention causale et multi-head ; positions ; blocs Transformer ; mini-GPT ; préparation des données ; préentraînement ; scaling ; modèles experts ; SFT ; préférences et DPO ; principes du RLHF et des méthodes de politique ; évaluation du raisonnement et calcul à l'inférence ; cache KV ; quantification ; attention efficace ; serving ; multimodal.

Chaque mécanisme avancé demande ses propres prérequis. Présenter les objectifs d'optimisation, données requises, coûts et limites, avant les recettes.

Projets : mini-modèle de langage reproductible sur petit corpus autorisé ; adaptation efficace d'un petit modèle avec comparaison au modèle initial. Prévoir une alternative peu coûteuse avant toute expérience GPU.

Passage : expliquer masque causal et shapes, vérifier l'absence de contamination de l'évaluation, mesurer les compromis de qualité, mémoire et latence.

### F. AI engineering

Modules : prompting et contexte ; recherche documentaire ; découpage ; indexation ; recherche lexicale, dense et hybride ; reranking ; RAG ; métriques de récupération ; réponses et citations ; jeux d'évaluation ; juges automatiques et limites ; adaptation versus récupération ; LoRA ; outils et schémas ; agents ; planification ; mémoire ; orchestration ; protocoles d'intégration ; injections ; confidentialité ; observabilité ; cache et routage.

Projets : assistant documentaire évalué avec questions sans réponse ; agent pour une tâche bornée avec permissions et traces ; expérience LoRA mesurée.

Passage : comparer une solution simple à une solution plus complexe, tester les échecs et rendre explicites les actions nécessitant une autorisation.

### G. Data engineering et MLOps

Modules : contrats de données ; pipelines batch ; SQL et transformations ; orchestration ; traitement distribué selon besoin ; suivi d'expériences ; versionnement ; tests ML ; FastAPI ; conteneurs ; CI/CD ; serving batch et temps réel ; cloud choisi ; GPU ; déploiement progressif ; monitoring ; dérive et labels retardés ; rollback ; coûts ; profiling ; distribution de l'entraînement.

Projet : service de prédiction conteneurisé avec validation d'entrée, tests, version du modèle, métriques et procédure de retour arrière. Projet avancé : déclencher et suivre une expérience depuis un pipeline reproductible.

Passage : simuler une panne, un changement de schéma et une régression, puis restaurer un service connu comme stable.

### H. Expertise, recherche et spécialisation

Modules : cadrage métier ; conception de systèmes ; expérimentation et ablations ; lecture critique de papiers ; reproduction ; communication ; revue de code ; mentorat ; confidentialité et équité ; étude des contraintes applicables avec sources à jour.

Choisir ensuite un axe profond : vision, NLP/LLM, recommandation, RL, systèmes d'inférence, agents, MLOps ou recherche.

Projet final : besoin ouvert, budget de ressources, données documentées, architecture défendue, baseline, évaluation, déploiement, surveillance et bilan après usage. La revue doit comprendre des questions inédites et une évolution des contraintes.

## 5. Les défis de code deviennent des projets

Chaque projet possède un brief avec utilisateur cible, problème, données et droits d'usage, contraintes, critères d'acceptation, baseline et livrables.

Les petits exercices de code restent utiles pour isoler un mécanisme. Ils alimentent un projet suggéré ; ils ne constituent pas toute l'expérience. Exemple : vectoriser un calcul → entraîner une régression → comparer une baseline → exposer le modèle → surveiller ses entrées.

Prévoir trois degrés d'aide : jalons détaillés ; brief avec indices ; brief ouvert. Une même tâche peut être reprise à chaque degré pour montrer une autonomie croissante.

## 6. Production des vidéos

Avant toute génération, préparer scénario, faits vérifiés, storyboard, formules et exemple exécuté. Produire d'abord une leçon pilote, puis faire un essai avec un apprenant.

Séquence proposée : situation concrète ; intuition visuelle ; notation ; démonstration ; code ; pause exercice ; correction ; question de transfert. La durée dépend du contenu ; découper au lieu de compresser une démonstration.

Une vidéo n'est considérée disponible qu'après production, vérification des calculs et du code, sous-titrage, transcription et contrôle de cohérence avec le texte. L'application doit afficher honnêtement les scénarios encore à produire.

## 7. Veille et maintenance

Séparer les fondamentaux stables des ateliers portant sur des outils évolutifs. Un atelier mentionne ses versions et la date de vérification. Une nouveauté entre au programme si elle résout un besoin identifiable et si ses preuves et limites sont examinées.

Une fiche de veille contient : source primaire, date, contribution revendiquée, protocole, résultat, limites, reproduction possible et conséquence pour un projet. Ne pas qualifier une information de « récente » sans vérification.

Prévoir une revue périodique et après toute erreur signalée ou rupture de dépendance. Les sujets juridiques nécessitent des sources officielles vérifiées au moment de rédaction.

## 8. Définition de terminé

Un cours est publiable lorsque :

- les objectifs correspondent aux exercices et à l'évaluation ;
- les prérequis et toutes les notations sont explicités ;
- les démonstrations sont vérifiées indépendamment ;
- le code a été exécuté depuis un environnement propre et ses versions sont consignées ;
- les corrections expliquent pourquoi, pas seulement quoi ;
- le quiz évite les ambiguïtés et couvre autre chose que la reconnaissance ;
- les limites, coûts et conditions d'utilisation sont indiqués ;
- une personne autre que l'auteur peut suivre les consignes ;
- le contenu est lisible au clavier et sur mobile ;
- les sources, la date de vérification et les éléments encore manquants sont visibles.

« Parfait » devient ainsi un processus de vérification, de retours et de correction. Le premier lot approfondi devrait couvrir Python/NumPy, gradient, régression, validation et un projet tabulaire avant d'étendre la bibliothèque.

## 9. État de la première version

L'interface réutilise les leçons introductives du dossier : intuition, théorie, exemple de code, exercice et correction, un quiz par leçon, carnet et scénario vidéo. Trois projets proposent des jalons et livrables. La progression est locale au navigateur et exportable.

Restent à produire : les modules complets de ce programme, plusieurs exercices et quiz par objectif, les vidéos, la revue de code, l'exécution Python isolée, l'adaptation pédagogique, la synchronisation des comptes et une veille alimentée par des sources vérifiées. Le prototype ne délivre aucune validation professionnelle.
