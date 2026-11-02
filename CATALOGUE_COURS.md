# Catalogue détaillé des modules — version de conception

Mis à jour le 1 novembre 2026 à partir des deux listes fournies par l'apprenant.
Statut : programme à produire. Les noms d'outils sont des exemples de sujets, sans recommandation de version ni affirmation sur leur actualité.
Référence qualité : CONCEPTION_PEDAGOGIQUE.md.
Les cours réellement disponibles restent identifiés séparément dans l'application.

## Organisation

Le tronc commun comprend programmation, outils, maths, ML, réseaux, langage, évaluation et systèmes. Les approfondissements en vision, génération, RL, graphes et GPU sont des branches de spécialisation après les prérequis correspondants. On peut explorer leurs introductions sans devoir maîtriser toutes les branches.

Pour chaque module ci-dessous : diagnostic → leçons détaillées → calcul ou simulation → code minimal → exercices corrigés → quiz justifiés → projet → revue → révision différée.

Le nombre de questions et la durée sont adaptés aux objectifs. Une formule complexe peut demander une séance complète. Ne pas transformer les listes ci-dessous en diapositives survolant dix notions à la fois.

## Bloc 1 — Python et génie logiciel

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| P01 — Bases Python | valeurs, variables, types, conditions, boucles, fonctions, portée, f-strings | prédire une exécution, gérer un cas vide, écrire un calculateur de statistiques simples |
| P02 — Structures | listes, tuples, dictionnaires, ensembles, mutabilité, copie, compréhensions | éliminer un alias involontaire, compter et agréger sans perdre le sens des données |
| P03 — Fichiers et modules | JSON/CSV, chemins, encodage, with, exceptions, packages, lambda | importer un fichier incorrect et produire un diagnostic exploitable |
| P04 — Conception Python | classes, composition, héritage, dataclasses, itérateurs, générateurs, décorateurs, gestionnaires de contexte | concevoir un lecteur de données en flux et justifier les abstractions |
| P05 — Qualité | annotations, typing, contrôle statique, pytest, logging, docstrings, style, lint | package installé dans un environnement propre avec tests et documentation |
| P06 — Concurrence | async/await, I/O et CPU, processus, annulation, erreurs et synchronisation | comparer des stratégies sur une charge définie et expliquer les limites |

Projet transversal : package de préparation et validation de données. Les mécanismes avancés ne sont introduits que lorsqu'un besoin du projet les motive.

## Bloc 2 — Environnement, algorithmique et SQL

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| O01 — Système | navigation, filtres, pipes, scripts shell, permissions, SSH | traiter un journal et expliquer les effets de chaque commande |
| O02 — Collaboration | commits, branches, fusion, rebase, PR, conflits | résoudre un conflit, relire un diff et documenter la décision |
| O03 — Environnements | dépendances, venv, conda, uv, pyproject, verrouillage, image et conteneur | reproduire un projet avec une commande documentée |
| A01 — Structures et coût | Big-O, tableaux, piles, files, hash maps, arbres, graphes, heaps | comparer coût temporel et mémoire, et les vérifier sur des tailles croissantes |
| A02 — Algorithmes | tri, recherche binaire, récursivité, BFS/DFS, programmation dynamique | résoudre des variantes et expliquer l'invariant et la complexité |
| S01 — SQL analytique | filtres, agrégats, GROUP BY/HAVING, jointures, sous-requêtes, CTE, fenêtres et LAG | produire une analyse temporelle et détecter une multiplication de lignes |
| S02 — Stockage | normalisation, index, transactions, notions document et clé-valeur | modéliser un petit domaine et justifier les index et transactions |

Un objectif d'environ 100 problèmes peut organiser la pratique algorithmique ; il ne constitue pas un seuil de compétence. La validation porte sur l'explication, les variantes inédites et le transfert dans un projet.

## Bloc 3 — Outils de données

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| D01 — NumPy | tableaux, dimensions, indexation, broadcasting, vectorisation, algèbre linéaire | corriger une erreur silencieuse de shape et tester les invariants |
| D02 — pandas | index, types, groupby, merge, pivot, séries temporelles, manquants | analyse exploratoire avec règles de nettoyage justifiées |
| D03 — Visualisation | Matplotlib, Seaborn, graphiques interactifs, choix d'échelles | montrer une distribution sans induire une conclusion trompeuse |
| D04 — Traitements efficaces | colonnes, mémoire, traitement en lots, comparaison de moteurs dont Polars | mesurer un coût sur un jeu donné au lieu d'affirmer une supériorité générale |

Projet : analyse exploratoire d'un jeu réel, vérification des droits d'usage, rapport des anomalies, transformations testées et notebook réexécutable.

## Bloc 4 — Fondations mathématiques

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| M01 — Géométrie vectorielle | produit scalaire, normes, projection, espaces, bases et indépendance | projeter à la main et relier projection et approximation |
| M02 — Matrices | produits, inverse, déterminant, rang, systèmes linéaires | vérifier les dimensions et résoudre sans inverse explicite quand approprié |
| M03 — Décompositions | valeurs propres, diagonalisation, SVD, QR, LU et Cholesky | reconstruire une matrice et expliquer une approximation de rang réduit |
| M04 — Calcul | dérivées, chaîne, dérivées partielles, gradient, jacobien, hessien, Taylor, intégrales | calculer le gradient d'un petit réseau et contrôler par différences finies |
| M05 — Autodiff | graphes, modes forward et reverse, accumulation | coder un moteur scalaire minimal et vérifier ses dérivées |
| M06 — Probabilités | axiomes, conditionnement, indépendance, Bayes, variables et lois, moments | simuler un taux de base et expliquer une probabilité conditionnelle |
| M07 — Lois et limites | Bernoulli, binomiale, Poisson, normale, uniforme, exponentielle, normale multivariée, LGN/TCL | distinguer hypothèse de loi, fréquence empirique et approximation |
| M08 — Information et estimation | entropie, cross-entropy, KL, information mutuelle, MLE, MAP, chaînes de Markov | dériver une vraisemblance simple et traiter les supports incompatibles |
| M09 — Statistiques | échantillonnage, estimateurs, biais/variance, intervalles, tests, puissance, multiplicité, bootstrap | concevoir une simulation de couverture et interpréter un test |
| M10 — Expérimentation | randomisation, confusion causale, A/B, bases bayésiennes | définir une expérience, sa métrique et ses limites avant les données |
| M11 — Optimisation | convexité, extrema et selles, GD/SGD, momentum, RMSprop, Adam/AdamW, schedules, contraintes et Lagrange | comparer convergence, mise à l'échelle et influence des hyperparamètres |

Les liens entre régularisation et priors sont traités avec leurs hypothèses. Les méthodes avancées s'appuient sur une dérivation simple avant leurs versions générales.

## Bloc 5 — Machine learning

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| ML01 — Protocole | supervisé/non supervisé/RL, baseline, coût, splits, CV stratifiée/groupée/temporelle, fuite, recherche d'hyperparamètres | choisir une séparation cohérente avec l'usage futur |
| ML02 — Modèles linéaires | OLS, régression polynomiale, Ridge/Lasso/ElasticNet, logistique, softmax | coder la logistique, vérifier son gradient et comparer à une bibliothèque |
| ML03 — Méthodes classiques | k-NN, Naive Bayes, SVM et noyaux | comparer hypothèses, coût, frontière et calibration |
| ML04 — Arbres et ensembles | arbre, bagging, forêt, AdaBoost, gradient boosting, stacking/blending | coder un arbre minimal et empêcher une fuite dans un ensemble |
| ML05 — Représentations et groupes | k-means, GMM/EM, DBSCAN, hiérarchique, PCA, t-SNE, UMAP | distinguer regroupement, visualisation et preuve d'une structure |
| ML06 — Features | encodage, standardisation, manquants, outliers, sélection, features cycliques, pipelines | mesurer l'effet d'une transformation à protocole constant |
| ML07 — Évaluation | précision/rappel/F1, ROC/PR, calibration, seuils, MAE/RMSE/R²/MAPE, ranking NDCG/MAP | traiter les cas indéfinis, les classes rares et les coûts métier |
| ML08 — Interprétation | importance par permutation, dépendance partielle, SHAP/LIME, analyse d'erreurs | expliquer une prédiction sans la transformer en conclusion causale |
| ML09 — Anomalies et recommandation | Isolation Forest, one-class SVM, associations, filtrage collaboratif, factorisation | construire une baseline et un split sans fuite utilisateur/temps |
| ML10 — Temps | tendance, saisonnalité, stationnarité, autocorrélation, lissage, ARIMA/SARIMA, features retardées, backtesting | comparer à une prévision naïve avec plusieurs origines temporelles |

Le rééquilibrage et les transformations sont effectués dans le train de chaque pli. Les modèles de fondation temporels constituent un atelier ultérieur, évalué avec la même baseline.

## Bloc 6 — Deep learning et spécialisations

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| DL01 — Fondations | neurone, MLP, expressivité et limites, forward, backprop, pertes, activations | micrograd scalaire, puis MLP NumPy avec tests de gradients |
| DL02 — Entraînement | initialisation, BatchNorm/LayerNorm, dropout, weight decay, optimiseurs | expliquer une expérience de surapprentissage d'un petit lot |
| DL03 — PyTorch | tenseurs, autograd, Module, Dataset/DataLoader, checkpoints, GPU/AMP | boucle fiable, modes train/eval, métriques et reprise |
| DL04 — Diagnostic et performance | NaN, gradients, profiling, compilation, suivi d'expériences ; notions JAX/Flax | localiser un goulot mesuré et tester une optimisation |
| CV01 — Vision | convolution, pooling, champ réceptif, résidus, familles de CNN, augmentation et transfert | classification avec analyse par classe et contamination contrôlée |
| CV02 — Vision avancée | ViT, embeddings contrastifs, CLIP/DINO/SAM comme études, détection et segmentation | étudier une architecture et reproduire une comparaison bornée |
| SQ01 — Séquences | RNN, gradients, LSTM/GRU, seq2seq, attention historique, notions SSM | comparer une tâche séquentielle à une baseline simple |
| TR01 — Transformer | Q/K/V, masques, multi-head, positions sinusoïdales/RoPE/ALiBi, résidus, normalisation, MQA/GQA/éparse | implémentation complète avec tests de causalité et de formes |
| GE01 — Génération | autoencodeur, VAE, GAN, autorégression, flows, diffusion, score, guidance, latents, flow matching | reproduire un petit mécanisme et définir son évaluation |
| RL01 — Renforcement | MDP, Bellman, Q-learning/DQN, REINFORCE, actor-critic, PPO, exploration et récompense | environnement simple, seeds multiples et contrôle du reward hacking |
| GR01 — Graphes | message passing, GCN/GAT/GraphSAGE, auto-supervision | tâche sur graphes avec split adapté au problème |

RL, GNN, génération avancée et performance GPU sont des spécialisations. Elles demandent un choix de profondeur plutôt qu'un survol obligatoire de toutes les architectures.

## Bloc 7 — NLP, LLM et multimodalité

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| NLP01 — Langage | prétraitement, BPE/WordPiece/Unigram, SentencePiece, TF-IDF/BM25, word2vec/GloVe, contexte | construire une baseline de recherche et un tokenizer simple |
| NLP02 — Tâches et encodeurs | classification, NER, QA, résumé, traduction, encodeurs, perplexité/BLEU/ROUGE et limites | choisir une évaluation qui correspond à la tâche |
| LLM01 — Préentraînement | decoder-only, next-token, données et déduplication, MoE, scaling, multilingue et contexte | préentraîner un petit GPT avec courbes et partition non contaminée |
| LLM02 — Post-training | SFT, templates, récompense, RLHF, DPO/KTO, GRPO, données synthétiques, distillation, méthodes constitutionnelles | comparaison bornée et recherche de sycophancy/reward hacking |
| LLM03 — Inférence | cache KV, batching, attention efficace, pagination, décodage spéculatif, quantification, pruning, serving | mesurer TTFT, débit, mémoire et qualité à charge définie |
| LLM04 — Évaluation | benchmarks comme études de cas, contamination, juge automatique/humain, factualité | jeu d'évaluation spécifique, accord et erreurs du juge |
| MM01 — Multimodal | vision-langage, audio, TTS, vidéo, OCR, embeddings multimodaux | extraction documentaire mesurée et analyse d'échecs |

Les noms de modèles et moteurs sont étudiés dans des ateliers datés, avec versions et conditions d'exécution. Leur présence dans la liste de départ ne prouve ni leur disponibilité ni leur supériorité actuelle.

## Bloc 8 — AI engineering

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| AI01 — Contexte et sorties | zero/few-shot, rôles, schémas JSON, contexte, compaction, cache et versionnement | tests de conformité et comparaison sur jeu fixe |
| AI02 — Ingestion et recherche | PDF/HTML, chunking, embeddings, index HNSW, BM25, hybride, reranking, reformulation | mesurer le rappel séparément de la génération |
| AI03 — RAG | citations, sans réponse, multi-hop, GraphRAG/agentic comme approfondissements | RAG évalué avec sources autorisées et tests d'injection |
| AI04 — Adaptation | prompting/RAG/fine-tuning, LoRA/QLoRA/PEFT, préparation, oubli, évaluation | comparaison avant/après avec données isolées et coût déclaré |
| AI05 — Outils et agents | schémas, boucle d'outils, planification, mémoire, workflows, protocoles MCP/A2A comme sujets | tâche bornée, permissions, reprise et validation humaine |
| AI06 — Fiabilité | traces, datasets, régressions, grounding, injections directes/indirectes, exfiltration, PII | échecs contrôlés, limites, sandbox et critères d'arrêt |
| AI07 — Produit et coût | streaming, latence, cache, routage, feedback et UX | expérience utilisateur qui rend visibles les incertitudes utiles |

Les SDK et frameworks mentionnés dans la proposition sont des supports interchangeables. Le cours explique d'abord les mécanismes et les tests.

## Bloc 9 — Data engineering, MLOps et performance

| Module | Leçons à concevoir | Exercices et preuve attendue |
| --- | --- | --- |
| SYS01 — Infrastructure | un cloud choisi, compute, stockage, IAM, réseau, GPU, Docker, Kubernetes, Terraform | environnement limité en permissions et coût, restauration documentée |
| DATA01 — Pipelines | ETL/ELT, orchestration, dbt, Spark, streaming/Kafka, entrepôt/lakehouse, Parquet/Delta | pipeline relançable, idempotence, qualité et gestion d'erreur |
| OPS01 — Expériences | MLflow/W&B comme supports, DVC, registry, configuration, seeds, artefacts | reproduire et comparer deux expériences |
| OPS02 — Déploiement | API, batch/temps réel, CI/CD, canary/shadow, A/B, contrats d'entrée | service testé avec version et rollback |
| OPS03 — Exploitation | drift, performance réelle, labels retardés, alertes, réentraînement, LLMOps | incident simulé et décision d'intervention justifiée |
| GPU01 — Calcul distribué | DDP/FSDP/ZeRO, parallélisme données/tenseurs/pipeline, multi-nœuds | comparaison mesurée sur ressources accessibles ou analyse instrumentée |
| GPU02 — Performance | mémoire, bande passante, profiling, notions CUDA/Triton | rapport de goulot et gain mesuré avec validation numérique |

Les travaux coûteux ont une variante locale ou une simulation explicite. Une simulation n'est jamais présentée comme une expérience multi-GPU réellement exécutée.

## Bloc 10 — Compétences de conception et expertise

| Module | Leçons à concevoir | Preuve attendue |
| --- | --- | --- |
| EXP01 — Cadrage | besoin, métrique, contraintes, architecture, coût/qualité/latence, solution de repli | design doc défendu sur fraude, recherche, ranking ou assistant |
| EXP02 — Recherche | lecture, baselines, ablations, incertitude et reproduction | rapport distinguant résultat du papier et résultat reproduit |
| EXP03 — Responsabilité | biais, confidentialité, fédéré, confidentialité différentielle, model cards, gouvernance | analyse adaptée au cas ; cadre légal vérifié séparément avant rédaction |
| EXP04 — Travail collectif | revue de code, mentorat, estimation, documentation, communication non technique | revue argumentée et présentation des décisions et résultats |
| EXP05 — Spécialisation | axe choisi, objectifs, papiers et système réel | projet long avec retours, contraintes nouvelles et suivi après déploiement |

## File de production proposée

1. P01–P05, O02–O03 et D01–D02 : obtenir un package de données testé.
2. M01–M04 puis optimisation élémentaire : relier calcul, code et erreurs.
3. ML01–ML02 et ML06–ML08 : mener un projet tabulaire rigoureux.
4. DL01–DL03 et TR01 : comprendre et programmer un petit réseau puis l'attention.
5. NLP01, LLM01 et AI01–AI03 : construire puis évaluer un assistant documentaire.
6. OPS01–OPS03 : rendre les projets reproductibles, déployables et observables.
7. Choisir les branches avancées à partir d'un projet et des prérequis réels.

Les autres modules restent au catalogue : leur ordre sera ajusté au diagnostic initial, au temps disponible et aux preuves de maîtrise.
