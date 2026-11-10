export const mlChapters = [
  {
    "number": 1,
    "title": "Introduction et fondements du Machine Learning",
    "description": "Cadrer une tâche, distinguer apprentissage et généralisation, définir une baseline et séparer train, validation et test.",
    "lessonIds": [
      "ml-introduction",
      "ml-first-model",
      "ml-learning-optimization",
      "ml-generalization",
      "ml-validation-strategies",
      "ml-data-leakage",
      "ml-hyperparameter-tuning",
      "ml-project-methodology"
    ],
    "validation": "Concevoir et défendre un protocole sans fuite de données."
  },
  {
    "number": 2,
    "title": "Maîtriser la préparation des données",
    "description": "Transformer des données brutes en variables fiables, pertinentes et exploitables par les modèles de Machine Learning.",
    "lessonIds": ["ml-data-understanding", "ml-missing-values", "ml-outliers", "ml-feature-scaling", "ml-encoding", "ml-feature-engineering", "ml-feature-selection", "ml-dimensionality-reduction", "ml-class-imbalance", "ml-advanced-pipelines"],
    "validation": "Livrer un pipeline de préparation avec contrats et tests."
  },
  {
    "number": 3,
    "title": "Supervised Learning",
    "description": "Comprendre les modèles supervisés : régressions, voisins, arbres, ensembles et SVM, avec mathématiques, Python et comparaison sous validation.",
    "lessonIds": ["ml-linear-regression", "ml-regularization", "ml-polynomial-regression", "ml-logistic-regression", "ml-knn", "ml-decision-trees", "ml-random-forest", "ml-gradient-boosting", "ml-svm", "ml-model-evaluation"],
    "validation": "Implémenter, entraîner et comparer des modèles supervisés avec des métriques adaptées et un protocole indépendant."
  },
  {
    "number": 4,
    "title": "Apprentissage non supervisé (Unsupervised Learning)",
    "description": "10 cours + 1 mini-projet · 11 h 30 hors exercices. Clustering, réduction de dimension, anomalies et bonnes pratiques.",
    "lessonIds": ["ml-unsupervised-introduction","ml-kmeans-clustering","ml-hierarchical-clustering","ml-dbscan-hdbscan","ml-gaussian-mixture-models","ml-clustering-evaluation","ml-pca","ml-tsne-umap","ml-anomaly-detection","ml-unsupervised-pipelines"],
    "validation": "Comparer des segmentations, interpréter PCA et investiguer des anomalies en justifiant prétraitement, stabilité et limites."
  },
  {
    "number": 5,
    "title": "Machine Learning avancé",
    "description": "10 cours disponibles : évaluation, calibration, XAI, robustesse, incertitude, fairness, diagnostic, monitoring, validation et gouvernance.",
    "lessonIds": ["ml-advanced-evaluation", "ml-probability-calibration", "ml-explainable-ai", "ml-model-robustness", "ml-conformal-prediction", "ml-fairness-responsible-ml", "ml-model-diagnostics", "ml-drift-monitoring", "ml-validation-backtesting", "ml-evaluation-governance"],
    "validation": "Auditer performance, calibration, explications et robustesse avec des protocoles indépendants."
  },
  {
    "number": 6,
    "title": "Fondements du Deep Learning",
    "description": "À venir : perceptron, réseaux multicouches, activations, rétropropagation, optimiseurs, régularisation et PyTorch.",
    "lessonIds": [],
    "validation": "Entraîner un réseau, vérifier ses gradients et documenter sa généralisation."
  },
  {
    "number": 7,
    "title": "Decision Trees et Random Forest",
    "description": "Impureté, splits, élagage, bagging, sélection de variables et variance.",
    "lessonIds": [],
    "validation": "Coder un arbre minimal, puis comparer arbre et forêt."
  },
  {
    "number": 8,
    "title": "Gradient Boosting, XGBoost, LightGBM et CatBoost",
    "description": "Construction additive, pertes, régularisation, early stopping et particularités des outils.",
    "lessonIds": [],
    "validation": "Comparer des modèles à budget de recherche défini et documenter les versions."
  },
  {
    "number": 9,
    "title": "Clustering et apprentissage non supervisé",
    "description": "k-means, GMM/EM, DBSCAN, hiérarchique et limites des métriques internes.",
    "lessonIds": [],
    "validation": "Interpréter des groupes sans les confondre avec une vérité terrain."
  },
  {
    "number": 10,
    "title": "PCA, réduction de dimension et détection d’anomalies",
    "description": "Projection, variance, SVD, visualisations non linéaires, Isolation Forest et seuils.",
    "lessonIds": [],
    "validation": "Expliquer une projection et évaluer un détecteur selon le cas d’usage."
  },
  {
    "number": 11,
    "title": "Séries temporelles",
    "description": "Tendance, saisonnalité, lags, validation temporelle, modèles naïfs et backtesting.",
    "lessonIds": [],
    "validation": "Comparer des prévisions sans utiliser des informations du futur."
  },
  {
    "number": 12,
    "title": "ML avancé, interprétabilité, robustesse et études de cas",
    "description": "Explications, permutation, SHAP/LIME, décalage de distribution, robustesse, incertitude et décision métier.",
    "lessonIds": [],
    "validation": "Défendre un système complet, ses limites et son comportement après changement des données."
  }
] as const;
