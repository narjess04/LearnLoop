import chapterFour from "@/lib/ml-chapter-four.json";
import chapterThree from "@/lib/ml-chapter-three.json";
import chapterTwo from "@/lib/ml-chapter-two.json";
export type Lesson={id:string;module:string;title:string;minutes:number;goal:string;intuition:string;theory:string;example:string;code:string;pitfalls:string[];exercise:string;solution:string;question:string;options:string[];answer:number;explanation:string;source:string};
export const modules = [
  {
    "id": "python",
    "name": "Python & données",
    "label": "LES FONDATIONS",
    "color": "#4a927e",
    "icon": "code",
    "description": "Penser en tableaux et comprendre ses données."
  },
  {
    "id": "maths",
    "name": "Maths pour comprendre",
    "label": "LE RAISONNEMENT",
    "color": "#bb8a3a",
    "icon": "sigma",
    "description": "Relier les équations au comportement des modèles."
  },
  {
    "id": "ml",
    "name": "Machine learning",
    "label": "LES MODÈLES",
    "color": "#638ec4",
    "icon": "network",
    "description": "Prédire, évaluer et éviter les conclusions trompeuses."
  },
  {
    "id": "dl",
    "name": "Deep learning",
    "label": "LES RÉSEAUX",
    "color": "#9274bc",
    "icon": "brain",
    "description": "Comprendre les gradients et entraîner des réseaux."
  },
  {
    "id": "genai",
    "name": "IA générative",
    "label": "LA GÉNÉRATION",
    "color": "#c97965",
    "icon": "sparkles",
    "description": "De l’attention à un assistant documentaire évalué."
  },
  {
    "id": "engineering",
    "name": "Du modèle au projet",
    "label": "L’AUTONOMIE",
    "color": "#71886a",
    "icon": "terminal",
    "description": "Reproduire les expériences et observer les erreurs."
  }
];
export const lessons:Lesson[] = [
  {
    "minutes": 120,
    "intuition": "Construire un programme explicite, vérifiable et partageable.",
    "theory": "Le support complet développe les notions avec exemples, exercices, corrections et projet fil rouge.",
    "example": "Construire Taskflow puis le rendre reproductible.",
    "pitfalls": [
      "Oublier les cas limites.",
      "Confondre une démonstration et une validation réelle."
    ],
    "code": "def moyenne(valeurs):\n    if not valeurs:\n        raise ValueError('Au moins une valeur')\n    return sum(valeurs) / len(valeurs)",
    "exercise": "Construis Taskflow à partir du cahier des charges du support. Documente une règle métier, ses tests et une variante réalisée sans guide.",
    "solution": "Séparer contrat, logique et persistance. Tester ajout, liste, identifiant inconnu, données absentes et fichier corrompu. Fournir les commandes de reproduction et justifier les décisions.",
    "module": "python",
    "id": "python-foundations",
    "title": "Python : des bases au package Taskflow",
    "goal": "Écrire un package Python propre, testé et documenté.",
    "question": "Pourquoi éviter une liste comme valeur par défaut d’un paramètre ?",
    "options": [
      "Elle est partagée entre les appels",
      "Elle est copiée automatiquement",
      "Python interdit les valeurs par défaut"
    ],
    "answer": 0,
    "explanation": "La valeur par défaut est créée lors de la définition. Utiliser None et créer une liste dans le corps évite le partage involontaire.",
    "source": "https://docs.python.org/3/tutorial/controlflow.html"
  },
  {
    "minutes": 120,
    "intuition": "Construire un programme explicite, vérifiable et partageable.",
    "theory": "Le support complet développe les notions avec exemples, exercices, corrections et projet fil rouge.",
    "example": "Construire Taskflow puis le rendre reproductible.",
    "pitfalls": [
      "Oublier les cas limites.",
      "Confondre une démonstration et une validation réelle."
    ],
    "code": "def moyenne(valeurs):\n    if not valeurs:\n        raise ValueError('Au moins une valeur')\n    return sum(valeurs) / len(valeurs)",
    "exercise": "Construis Taskflow à partir du cahier des charges du support. Documente une règle métier, ses tests et une variante réalisée sans guide.",
    "solution": "Séparer contrat, logique et persistance. Tester ajout, liste, identifiant inconnu, données absentes et fichier corrompu. Fournir les commandes de reproduction et justifier les décisions.",
    "module": "python",
    "id": "tools-foundations",
    "title": "Terminal, Git et environnements reproductibles",
    "goal": "Partager et reconstruire un projet avec Git, un environnement isolé, une CI et Docker.",
    "question": "Un .gitignore protège-t-il un secret déjà commité ?",
    "options": [
      "Oui, rétroactivement",
      "Non, il faut révoquer le secret",
      "Oui après n’importe quel nouveau commit"
    ],
    "answer": 1,
    "explanation": "Le secret reste dans l’historique et ses copies. Il faut révoquer la clé ; ignorer le fichier ne suffit pas.",
    "source": "https://git-scm.com/book/en/v2"
  },
  {
    "id": "numpy",
    "module": "python",
    "title": "Penser en tableaux avec NumPy",
    "minutes": 30,
    "goal": "Prévoir les dimensions avant de lancer un calcul.",
    "intuition": "Dans X de forme (n,d), une ligne est une observation et une colonne une variable. Une opération acceptée par NumPy peut pourtant répondre à la mauvaise question si les axes sont mal interprétés.",
    "theory": "Pour X ∈ ℝⁿˣᵈ et w ∈ ℝᵈ, X @ w produit n prédictions. * multiplie élément par élément. Le broadcasting compare les dimensions depuis la droite : elles doivent être égales ou l’une doit valoir 1. axis=0 réduit l’axe des observations : X.mean(axis=0) donne une moyenne par variable.",
    "example": "100 logements avec 3 variables donnent X de forme (100,3). Trois coefficients donnent 100 prix. Attention : soustraire (100,1) à (100,) produit (100,100), sans erreur de syntaxe.",
    "code": "import numpy as np\nX = np.array([[20., 1., 5.], [40., 2., 10.]])\nw = np.array([2., 3., -0.5])\npred = X @ w + 1.\ncentered = X - X.mean(axis=0)\nassert pred.shape == (2,)\nassert np.allclose(centered.mean(axis=0), 0)\nprint(pred)  # [41.5, 82.]",
    "pitfalls": [
      "Confondre @ et *.",
      "Confondre (n,) et (n,1).",
      "Oublier le sens des axes et les unités."
    ],
    "exercise": "Crée X (4,3), calcule les moyennes par colonne et quatre prédictions. Montre le problème d’une soustraction entre (4,) et (4,1).",
    "solution": "X.mean(axis=0) est (3,), X @ w est (4,). La soustraction avec (4,1) produit (4,4). Aligner explicitement les formes avant la perte.",
    "question": "X est (8,3), w est (3,). X @ w est…",
    "options": [
      "(8,)",
      "(3,)",
      "(8,3)"
    ],
    "answer": 0,
    "explanation": "Une combinaison des trois variables par ligne : huit prédictions.",
    "source": "https://numpy.org/doc/stable/user/basics.broadcasting.html"
  },
  {
    "id": "data",
    "module": "python",
    "title": "Nettoyer sans perdre le sens des données",
    "minutes": 35,
    "goal": "Diagnostiquer types, clés et informations disponibles.",
    "intuition": "Une valeur manquante peut être un oubli ou une information en soi. Remplacer par zéro change son sens. Définir d’abord ce que représente une ligne et quand ses variables sont connues.",
    "theory": "Examiner types, unités, manquants et clés. Deux achats identiques peuvent être réels. Une jointure plusieurs-à-plusieurs peut multiplier les lignes. Les statistiques d’imputation d’un modèle seront apprises sur train uniquement.",
    "example": "Deux lignes pour un même client peuvent doubler ses transactions à la jointure et gonfler les ventes. validate vérifie la relation attendue.",
    "code": "import pandas as pd\norders = pd.DataFrame({'client': [1,1,2], 'amount': [10,20,15]})\nclients = pd.DataFrame({'client': [1,2], 'city': ['Tunis','Sfax']})\nassert clients.client.is_unique\njoined = orders.merge(clients, on='client',\n                      how='left', validate='many_to_one')\nassert len(joined) == len(orders)\nprint(orders.isna().mean())",
    "pitfalls": [
      "Supprimer les valeurs extrêmes sans enquête.",
      "Supprimer arbitrairement un doublon.",
      "Utiliser une variable connue après la cible."
    ],
    "exercise": "Ajoute un client en double. Observe l’échec de validate et propose une règle métier pour résoudre le conflit.",
    "solution": "Utiliser une correction vérifiée, une clé composée ou la version valide à la date de transaction. drop_duplicates sans règle peut conserver la mauvaise information.",
    "question": "Pourquoi validate=many_to_one ?",
    "options": [
      "Imputer les manquants",
      "Détecter plusieurs correspondances client par transaction",
      "Accélérer tous les calculs"
    ],
    "answer": 1,
    "explanation": "La validation contrôle la cardinalité, donc une possible multiplication des lignes.",
    "source": "https://pandas.pydata.org/docs/user_guide/merging.html"
  },
  {
    "id": "gradient",
    "module": "maths",
    "title": "Du gradient à la descente de gradient",
    "minutes": 40,
    "goal": "Expliquer le signe et la taille d’une mise à jour.",
    "intuition": "La dérivée indique comment une petite variation du paramètre change localement la perte. Si elle est positive, augmenter le paramètre augmente la perte : on part dans l’autre sens.",
    "theory": "L(w)=(w−3)² ; ∂L/∂w=2(w−3). Mise à jour : w ← w−η∇L(w). Pour w=0 et η=0,1, le gradient vaut −6, donc w devient 0,6. Ici l’erreur est multipliée par 1−2η ; la convergence nécessite 0<η<1.",
    "example": "η=0,8 oscille mais converge. η=1,1 diverge. La borne dépend de la courbure : elle ne se transfère pas telle quelle aux réseaux.",
    "code": "w, eta = 0., 0.1\nfor step in range(20):\n    loss = (w - 3) ** 2\n    gradient = 2 * (w - 3)\n    w -= eta * gradient\n    print(step, w, loss)\n# Perte affichée AVANT la mise à jour.",
    "pitfalls": [
      "Ajouter le gradient.",
      "Confondre progrès local et optimum global.",
      "Utiliser la même borne pour toutes les pertes."
    ],
    "exercise": "Compare η=0,01, 0,8 et 1,1. Dérive ensuite L=(2w−6)² et retrouve sa borne de convergence.",
    "solution": "Le nouveau gradient vaut 8(w−3). Il faut |1−8η|<1, soit 0<η<0,25.",
    "question": "Gradient=−4, η=0,1 : déplacement ?",
    "options": [
      "−0,4",
      "+0,4",
      "+4"
    ],
    "answer": 1,
    "explanation": "−η×gradient = −0,1×(−4)=+0,4.",
    "source": "https://www.deeplearningbook.org/contents/optimization.html"
  },
  {
    "id": "probability",
    "module": "maths",
    "title": "Probabilités et taux de base",
    "minutes": 35,
    "goal": "Interpréter une alerte en tenant compte de la fréquence.",
    "intuition": "Repérer presque tous les événements ne garantit pas que les alertes soient fiables lorsque ces événements sont rares. Les deux probabilités conditionnelles ne s’inversent pas.",
    "theory": "Bayes : P(A|B)=P(B|A)P(A)/P(B), pour P(B)>0. P(B)=P(B|A)P(A)+P(B|non-A)P(non-A). Le dénominateur rassemble vraies et fausses alertes.",
    "example": "Sur 10 000 opérations, 100 fraudes. On détecte 90 fraudes et on alerte sur 5 % des 9 900 autres : 495 fausses alertes. Précision=90/585≈15,4 %, rappel=90 %.",
    "code": "prevalence, recall, fpr = 0.01, 0.90, 0.05\ntrue_alerts = prevalence * recall\nfalse_alerts = (1 - prevalence) * fpr\nprecision = true_alerts / (true_alerts + false_alerts)\nprint(precision)  # environ 0.154",
    "pitfalls": [
      "Inverser P(A|B) et P(B|A).",
      "Ignorer un changement de prévalence.",
      "Confondre score et probabilité calibrée."
    ],
    "exercise": "Refais le calcul à 10 % de prévalence, avec rappel et taux de faux positifs identiques.",
    "solution": "Vraies alertes=0,09 ; fausses=0,045 ; précision=0,09/0,135≈66,7 %. La population a changé.",
    "question": "P(alerte | fraude) correspond à…",
    "options": [
      "La précision",
      "Le rappel",
      "La prévalence"
    ],
    "answer": 1,
    "explanation": "Le rappel est la fraction de fraudes détectées parmi les fraudes.",
    "source": "https://scikit-learn.org/stable/modules/model_evaluation.html"
  },
  {
    "id": "ml-introduction",
    "module": "ml",
    "title": "Cours 1 — Qu’est-ce que le Machine Learning ?",
    "minutes": 90,
    "goal": "Comprendre les fondements du ML, ses principales familles, son fonctionnement et les situations dans lesquelles il est réellement utile.",
    "intuition": "Le ML apprend des modèles à partir de données. L’IA est le domaine général ; le ML en est une approche et le deep learning une famille de méthodes. La GenAI décrit une capacité de génération, pas une quatrième couche strictement emboîtée.",
    "theory": "Une observation est une instance ; les features X sont les variables d’entrée et y la cible. Un modèle fθ apprend ses paramètres θ pendant le training. Les hyperparamètres règlent la procédure. L’inférence utilise le modèle appris. Minimiser la perte empirique ne garantit ni généralisation ni valeur en production. Les familles supervisée, non supervisée, par renforcement et auto-supervisée sont des cadres utiles, sans être des catégories toujours disjointes.",
    "example": "Comparer une règle dette/revenu > 0,4 à un modèle appris sur un historique de crédit. Quatre exemples illustrent le raisonnement mais ne suffisent pas pour valider un système réel.",
    "code": "clients = [\n    {'income': 2500, 'debt': 1200, 'default': 1},\n    {'income': 5000, 'debt': 500, 'default': 0},\n    {'income': 3200, 'debt': 1600, 'default': 1},\n    {'income': 7000, 'debt': 1000, 'default': 0},\n]\n\ndef debt_ratio(client):\n    if client['income'] <= 0:\n        raise ValueError('Revenu strictement positif requis')\n    return client['debt'] / client['income']\n\ndef predict_default(client, threshold=0.4):\n    return int(debt_ratio(client) > threshold)\n\nfor client in clients:\n    prediction = predict_default(client)\n    print(f\"Ratio: {debt_ratio(client):.2f} | \"\n          f\"Prediction: {prediction} | \"\n          f\"Actual: {client['default']}\")",
    "pitfalls": [
      "Confondre une règle choisie manuellement et une relation apprise.",
      "Déduire une fiabilité réelle de quatre observations.",
      "Confondre performance sur train et généralisation.",
      "Choisir un modèle avant de cadrer le besoin."
    ],
    "exercise": "Exécute la règle sur les quatre clients. Prédis les sorties avant l’exécution. Compare des seuils de 0,4 et 0,5. Explique pourquoi aucun apprentissage n’a lieu et pourquoi le résultat sur ces quatre lignes ne mesure pas la généralisation.",
    "solution": "Les ratios sont 0,48 ; 0,10 ; 0,50 ; environ 0,143. À 0,4, les prédictions sont 1,0,1,0 ; à 0,5, avec une comparaison strictement supérieure, elles sont toutes nulles. Le seuil est fixé par le programmeur ; rien n’est ajusté sur les données. Aucun test indépendant n’est effectué.",
    "question": "Le programme fixe le seuil dette/revenu à 0,4 sans l’ajuster sur les données. S’agit-il déjà de Machine Learning ?",
    "options": [
      "Oui, car il utilise des données",
      "Non, c’est une règle déterministe définie manuellement",
      "Oui, car il produit une prédiction"
    ],
    "answer": 1,
    "explanation": "Manipuler des données et prédire ne suffit pas : ici, aucun paramètre n’est appris. Une règle reste une baseline pertinente à comparer à un modèle entraîné.",
    "source": "https://developers.google.com/machine-learning/intro-to-ml"
  },
  {
    "module": "ml",
    "intuition": "Un modèle doit être construit et évalué avec un protocole explicite. Comprendre chaque étape permet de déboguer et justifier les résultats.",
    "pitfalls": [
      "Apprendre des transformations sur le test.",
      "Confondre ajustement sur train et généralisation.",
      "Annoncer une métrique sans l’avoir mesurée."
    ],
    "id": "ml-first-model",
    "title": "Cours 2 — Construire son premier modèle de ML de A à Z",
    "minutes": 180,
    "goal": "Construire, expliquer et évaluer une classification supervisée avec un pipeline scikit-learn.",
    "theory": "Charger Breast Cancer Wisconsin (569 observations, 30 features ; 0=malignant, 1=benign), séparer 80/20 avec stratification, apprendre StandardScaler et LogisticRegression sur train, puis évaluer sans réglage sur le test. Préciser la classe positive. Le dataset sert uniquement à la pédagogie.",
    "example": "Standardisation z=(x−μ_train)/σ_train ; pipeline scaler puis logistique ; accuracy, rapport par classe, AUC et matrice de confusion.",
    "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.datasets import load_breast_cancer\n\ndataset = load_breast_cancer(as_frame=True)\nX, y = dataset.data, dataset.target\nprint(\"Dimensions X :\", X.shape)\nprint(\"Dimensions y :\", y.shape)\nprint(X.head())\nprint(y.value_counts())\nprint(\"Valeurs manquantes :\", X.isna().sum().sum())\nprint(X.describe())\nprint(\"Classes :\", dict(enumerate(dataset.target_names)))\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.20, stratify=y, random_state=42\n)\nprint(\"X_train :\", X_train.shape)\nprint(\"X_test :\", X_test.shape)\nprint(\"Répartition train :\", y_train.value_counts(normalize=True))\nprint(\"Répartition test :\", y_test.value_counts(normalize=True))\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = make_pipeline(\n    StandardScaler(),\n    LogisticRegression(max_iter=1000)\n)\nmodel.fit(X_train, y_train)\nfrom sklearn.metrics import (\n    accuracy_score, classification_report,\n    confusion_matrix, roc_auc_score\n)\n\ny_pred = model.predict(X_test)\n# Vérifier l'ordre des colonnes plutôt que le supposer.\npositive_column = list(model.classes_).index(1)\ny_proba = model.predict_proba(X_test)[:, positive_column]\n\nprint(\"Accuracy :\", accuracy_score(y_test, y_pred))\nprint(\"ROC-AUC (classe positive 1) :\", roc_auc_score(y_test, y_proba))\nprint(classification_report(\n    y_test, y_pred, labels=[0, 1],\n    target_names=[\"malignant\", \"benign\"], zero_division=0\n))\nprint(\"Lignes = réel, colonnes = prédit, ordre [0, 1]\")\nprint(confusion_matrix(y_test, y_pred, labels=[0, 1]))\n",
    "exercise": "Réalise le projet Wine : split stratifié, pipeline, baseline, accuracy, F1 macro, probabilités et analyse des erreurs. Explique chaque choix.",
    "solution": "Le support fournit un squelette et des critères, pas la solution complète. Contrôler le périmètre de fit, l’ordre de classes_ et les dimensions des probabilités. Comparer à une baseline et documenter les versions.",
    "question": "Sur quelles données le StandardScaler doit-il apprendre sa moyenne et son écart-type ?",
    "options": [
      "Tout le dataset avant le split",
      "Le train uniquement, dans la pipeline",
      "Le test pour représenter les nouvelles données"
    ],
    "answer": 1,
    "explanation": "Les statistiques sont apprises sur train, puis appliquées aux autres observations. En validation croisée, la pipeline doit être ajustée dans chaque pli train.",
    "source": "https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html"
  },
  {
    "module": "ml",
    "intuition": "Un modèle doit être construit et évalué avec un protocole explicite. Comprendre chaque étape permet de déboguer et justifier les résultats.",
    "pitfalls": [
      "Apprendre des transformations sur le test.",
      "Confondre ajustement sur train et généralisation.",
      "Annoncer une métrique sans l’avoir mesurée."
    ],
    "id": "ml-learning-optimization",
    "title": "Cours 3 — Comment un modèle apprend-il réellement ?",
    "minutes": 240,
    "goal": "Comprendre les pertes, dériver les gradients et implémenter une descente de gradient avec NumPy.",
    "theory": "Pour ŷ=wx+b, J=mean((wx+b−y)^2). La règle de chaîne donne dw=2 mean(error*x), db=2 mean(error). Mettre à jour simultanément les paramètres par θ←θ−η∇J. Choisir un pas adapté ; les garanties dépendent de la fonction. Tous les modèles ne sont pas appris par descente de gradient.",
    "example": "Pour L(θ)=θ² et η=0,2, θ=4 devient 2,4 en une mise à jour ; la perte passe de 16 à 5,76.",
    "code": "import numpy as np\n\nrng = np.random.default_rng(42)\nx = np.linspace(-2.0, 2.0, 120)\ny = 3.0 * x + 2.0 + rng.normal(scale=0.2, size=x.shape)\nw, b = 0.0, 0.0\nlearning_rate = 0.05\nlosses = []\n\nfor step in range(500):\n    prediction = w * x + b\n    error = prediction - y\n    # Les deux gradients utilisent les mêmes paramètres.\n    dw = 2.0 * np.mean(error * x)\n    db = 2.0 * np.mean(error)\n    w -= learning_rate * dw\n    b -= learning_rate * db\n    # Conserver une perte correspondant aux paramètres après mise à jour.\n    losses.append(float(np.mean((w * x + b - y) ** 2)))\n\nprint(\"w :\", w, \"b :\", b)\nprint(\"MSE finale :\", losses[-1])\nassert np.isfinite(losses).all()\nassert abs(w - 3.0) < 0.1\nassert abs(b - 2.0) < 0.1\n",
    "exercise": "Réalise le projet multivarié : 200 lignes, trois features, gradient matriciel, dimensions explicites et vérification par différences finies. Compare plusieurs learning rates.",
    "solution": "La forme attendue de X.T @ error est (3,), celle de prediction est (200,). Le gradient du biais agrège les erreurs. Les deux gradients doivent être calculés avant la mise à jour. Le bruit implique des poids proches, pas nécessairement identiques aux paramètres générateurs.",
    "question": "Pour L(θ)=θ², θ=4 et η=0,2, quelle est la nouvelle valeur de θ ?",
    "options": [
      "5,6",
      "2,4",
      "3,2"
    ],
    "answer": 1,
    "explanation": "Le gradient vaut 2θ=8. La mise à jour est 4−0,2×8=2,4, avec une nouvelle perte de 5,76.",
    "source": "https://www.deeplearningbook.org/contents/optimization.html"
  },
  {
    "id": "ml-generalization",
    "module": "ml",
    "title": "Cours 4 — Généralisation, Overfitting, Underfitting et Biais-Variance",
    "minutes": 240,
    "goal": "Comprendre pourquoi un modèle performant à l'entraînement peut échouer sur de nouvelles données, comment le diagnostiquer et comment améliorer sa généralisation.",
    "intuition": "Comprendre le problème, poser un protocole et tester des hypothèses avant de choisir une solution.",
    "theory": "Dans le cours précédent, nous avons appris à minimiser une fonction de perte grâce à la descente de gradient. Mais minimiser l'erreur d'entraînement ne garantit pas qu'un modèle sera performant sur de nouvelles données. C'est précisément le problème de la généralisation, l'un des concepts les plus importants du Machine Learning. ## 1. Qu'est-ce que la généralisation ? Un modèle généralise lorsqu'il parvient à produire de bonnes prédictions sur des observations qu'il n'a jamais utilisées pour apprendre. Prenons un exemple : nous entraînons un modèle pour prédire les prix immobiliers à partir de la surface, de la localisation et d'autres caractéristiques. Un modèle qui prédit parfaitement les prix des maisons de son dataset d'entraînement, mais se trompe fortement sur de nouvelles maisons, n'a pas bien appris à généraliser. ### 1.1 Training Error versus Generalization Error On distingue deux notions. Erreur empirique (Training Error) : Elle mesure la perte moyenne sur les observations d'entraînement. Risque de généralisation (Expected Risk) : Il mesure la perte attendue sur la distribution réelle des données. Cette seconde quantité n'est généralement pas directement calculable, puisque nous ne connaissons pas toute la distribution des données. Nous l'estimons donc à partir d'observations indépendantes, notamment avec un ensemble de validation ou de test. L'objectif du ML n'est pas seulement de minimiser \\(\\hat R_{\\text{train}}\\), mais d'obtenir un modèle dont le risque de généralisation est faible. ## 2. Underfitting et Overfitting Il existe deux problèmes classiques lorsqu'on entraîne un modèle. Underfitting Modèle trop simple pour capturer la structure. Overfitting Modèle trop ajusté aux observations. Illustrations conceptuelles : les points représentent les observations et les courbes les fonctions apprises. ### 2.1 Underfitting — Sous-apprentissage Un modèle est en underfitting lorsqu'il n'arrive pas à capturer suffisamment les relations présentes dans les données. Les causes possibles sont une complexité insuffisante, des variables peu informatives, une régularisation excessive ou un entraînement insuffisant. Symptôme fréquent : erreurs élevées sur train et v",
    "example": "Le Coding Lab et le projet guidé du cours relient les concepts aux données, aux métriques et aux limites.",
    "code": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import PolynomialFeatures, StandardScaler\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import mean_squared_error\n\nrng = np.random.default_rng(42)\nX = rng.uniform(-3, 3, size=(100, 1))\ny = np.sin(X[:, 0]) + rng.normal(0, 0.25, 100)\nX_train, X_val, y_train, y_val = train_test_split(\n    X, y, test_size=0.3, random_state=42\n)\ndegrees = [1, 3, 15]\nx_plot = np.linspace(-3, 3, 300).reshape(-1, 1)\nfig, axes = plt.subplots(1, 3, figsize=(12, 4))\nfor ax, degree in zip(axes, degrees):\n    model = make_pipeline(\n        PolynomialFeatures(degree=degree, include_bias=False),\n        StandardScaler(), LinearRegression()\n    )\n    model.fit(X_train, y_train)\n    train_mse = mean_squared_error(y_train, model.predict(X_train))\n    val_mse = mean_squared_error(y_val, model.predict(X_val))\n    print(degree, \"MSE train:\", train_mse, \"MSE validation:\", val_mse)\n    ax.scatter(X_train[:, 0], y_train, label=\"train\", s=14)\n    ax.scatter(X_val[:, 0], y_val, label=\"validation\", s=14)\n    ax.plot(x_plot[:, 0], model.predict(x_plot), color=\"olive\")\n    ax.set_title(f\"Degré {degree}\")\n    ax.set_ylim(-3, 3)\n    ax.legend()\nfig.tight_layout()\nplt.show()\n",
    "pitfalls": [
      "Orienter les choix à partir du test final.",
      "Confondre un score de sélection et une estimation indépendante.",
      "Oublier les groupes, la chronologie ou la disponibilité des variables."
    ],
    "exercise": "Réalise le projet guidé à la fin du support ; fournis code reproductible, protocole, résultats observés et analyse des limites.",
    "solution": "Utiliser le squelette comme point de départ, vérifier les invariants du protocole et documenter chaque décision. La solution complète demande une revue des livrables.",
    "question": "Un modèle de churn atteint 0,99 de ROC-AUC sur train et 0,71 sur validation. Premier diagnostic ?",
    "options": [
      "Le modèle est nécessairement performant",
      "Un problème de généralisation est possible",
      "Il faut immédiatement ajouter des paramètres"
    ],
    "answer": 1,
    "explanation": "L’écart motive une enquête sur le protocole, les distributions et la complexité ; il ne suffit pas pour conclure à une cause unique.",
    "source": "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  {
    "id": "ml-validation-strategies",
    "module": "ml",
    "title": "Cours 5 — Train, Validation, Test et Cross-Validation",
    "minutes": 240,
    "goal": "Apprendre à évaluer correctement un modèle, choisir une stratégie de découpage adaptée et éviter les erreurs qui faussent les performances.",
    "intuition": "Comprendre le problème, poser un protocole et tester des hypothèses avant de choisir une solution.",
    "theory": "Dans le cours précédent, nous avons compris pourquoi un modèle peut être excellent sur les données d'entraînement et mauvais sur de nouvelles observations. Aujourd'hui, nous allons apprendre à mesurer correctement la capacité de généralisation d'un modèle. C'est un sujet fondamental pour les Data Scientists et ML Engineers, mais également une source fréquente de questions techniques en entretien. ## 1. Pourquoi découper les données ? Supposons que nous construisions un modèle de prédiction du défaut de crédit sur 10 000 observations. Nous entraînons le modèle sur les 10 000 observations, puis nous calculons une accuracy de 98 % sur ces mêmes observations. Pouvons-nous affirmer que notre modèle est performant ? Non. Ce résultat mesure uniquement sa performance sur les données utilisées pendant l'apprentissage. Il ne permet pas d'estimer honnêtement comment il se comportera sur de nouveaux clients. Pour cela, nous devons séparer les données en plusieurs ensembles. ## 2. Training, Validation et Test Set ### 2.1 Training Set Le training set est utilisé pour apprendre les paramètres du modèle. Par exemple, dans une régression logistique, il sert à estimer les coefficients \\(w\\) et le biais \\(b\\). ### 2.2 Validation Set Le validation set sert à comparer plusieurs configurations, sélectionner un modèle et régler ses hyperparamètres. Exemples : choisir max_depth d'un arbre ou la force de régularisation d'une régression logistique. ### 2.3 Test Set Le test set est réservé à l'évaluation finale, une fois les choix de modèle et d'hyperparamètres terminés. Si nous modifions plusieurs fois notre modèle en fonction des résultats du test, nous commençons à surajuster nos décisions à ce test. Principe essentiel : le test set ne doit pas devenir un second validation set. ## 3. Holdout Validation — Le découpage simple La stratégie la plus simple consiste à réserver une partie des données pour la validation ou le test. En Python : Comprendre les paramètres Paramètre Signification test_size=0.2 Réserve 20 % des données random_state=42 Rend le découpage reproductible stratify=y Préserve approximativement les proportions des classes shuffle=True Mélange les observations avant le déco",
    "example": "Le Coding Lab et le projet guidé du cours relient les concepts aux données, aux métriques et aux limites.",
    "code": "import numpy as np\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import KFold, StratifiedKFold, cross_val_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\n\nX, y = make_classification(\n    n_samples=1000, n_features=20, n_informative=8,\n    n_redundant=4, weights=[0.9, 0.1], random_state=42\n)\nmodel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\nstrategies = {\n    \"KFold\": KFold(n_splits=5, shuffle=True, random_state=42),\n    \"StratifiedKFold\": StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n}\nfor name, cv in strategies.items():\n    scores = cross_val_score(model, X, y, cv=cv, scoring=\"roc_auc\")\n    print(name, scores, scores.mean(), scores.std())\n# Aucun des deux protocoles n'est adapté automatiquement au temps ou aux groupes.\n",
    "pitfalls": [
      "Orienter les choix à partir du test final.",
      "Confondre un score de sélection et une estimation indépendante.",
      "Oublier les groupes, la chronologie ou la disponibilité des variables."
    ],
    "exercise": "Réalise le projet guidé à la fin du support ; fournis code reproductible, protocole, résultats observés et analyse des limites.",
    "solution": "Utiliser le squelette comme point de départ, vérifier les invariants du protocole et documenter chaque décision. La solution complète demande une revue des livrables.",
    "question": "Le test final peut-il servir à régler les hyperparamètres ?",
    "options": [
      "Oui, à chaque essai",
      "Non, il doit rester indépendant des choix",
      "Seulement si le score augmente"
    ],
    "answer": 1,
    "explanation": "La sélection se fait sur train/validation. Réutiliser le test rend son estimation optimiste.",
    "source": "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  {
    "id": "ml-data-leakage",
    "module": "ml",
    "title": "Cours 6 — Data Leakage : détecter et prévenir les fuites de données",
    "minutes": 240,
    "goal": "Pourquoi un modèle peut afficher 99 % de performance en validation, puis échouer en production ? Apprendre à identifier les fuites d'informations et construire des pipelines fiables.",
    "intuition": "Comprendre le problème, poser un protocole et tester des hypothèses avant de choisir une solution.",
    "theory": "Chapitre 1 : Fondements du Machine Learning MACHINE LEARNING · CHAPITRE 01 · COURS 06 Data Leakage — Comprendre, détecter et éviter les fuites de données Pourquoi un modèle peut afficher 99 % de performance en validation, puis échouer en production ? Apprendre à identifier les fuites d'informations et construire des pipelines fiables. 3–4 heures Théorie + Python Entretiens ML Engineer Progression du chapitre Cours 6 / 8 Dans le cours 5, nous avons appris à séparer les données avec train_test_split, StratifiedKFold, GroupKFold et TimeSeriesSplit. Mais même avec un bon découpage, une erreur de préparation peut fausser toute l'évaluation. C'est ce que nous allons étudier aujourd'hui : le Data Leakage. ## 1. Qu'est-ce que le Data Leakage ? Le Data Leakage, ou fuite de données, se produit lorsque des informations qui ne devraient pas être disponibles au moment de l'entraînement ou de la prédiction influencent le modèle ou son évaluation. Cela peut créer une performance artificiellement élevée. Exemple concret Imaginons que nous souhaitions prédire si un client remboursera son crédit. Notre dataset contient : Feature Description Disponible au moment de la demande ? income Revenu du client Oui debt_ratio Ratio d'endettement Oui credit_history Historique de crédit Oui recovery_amount Montant récupéré après un défaut Non default_flag Variable cible Non, pour une nouvelle demande Si nous utilisons recovery_amount pour prédire le défaut au moment de l'octroi, le modèle dispose d'une information future. Il pourrait apprendre une relation très forte, mais cette variable ne sera pas disponible lors de la prédiction réelle. Résultat : excellente performance pendant l'évaluation, puis échec au déploiement. ## 2. Les principales catégories de Data Leakage Target Leakage Une feature révèle directement ou indirectement la réponse que nous souhaitons prédire. Exemple : utiliser le statut final d'un dossier pour prédire son statut à l'ouverture. Train–Test Contamination Les données de validation ou de test influencent une étape d'apprentissage. Exemple : effectuer une sélection de features sur tout le dataset avant la cross-validation. Temporal Leakage Le modèle exploite des informa",
    "example": "Le Coding Lab et le projet guidé du cours relient les concepts aux données, aux métriques et aux limites.",
    "code": "import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n\nX_array, y = make_classification(\n    n_samples=3000, n_features=10, n_informative=5,\n    n_redundant=2, flip_y=0.05, random_state=42\n)\nX = pd.DataFrame(X_array, columns=[f\"feature_{i}\" for i in range(10)])\ntrain_idx, test_idx = train_test_split(\n    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42\n)\ndef evaluate(features):\n    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\n    model.fit(features.iloc[train_idx], y[train_idx])\n    proba = model.predict_proba(features.iloc[test_idx])[:, 1]\n    return roc_auc_score(y[test_idx], proba)\n\nprint(\"Features originales :\", evaluate(X))\nleaked = X.copy()\nleaked[\"future_target\"] = y  # INFORMATION INTERDITE à la prédiction.\nprint(\"Avec fuite volontaire :\", evaluate(leaked))\n# Le même split ne rend pas une feature issue de la cible légitime.\n",
    "pitfalls": [
      "Orienter les choix à partir du test final.",
      "Confondre un score de sélection et une estimation indépendante.",
      "Oublier les groupes, la chronologie ou la disponibilité des variables."
    ],
    "exercise": "Réalise le projet guidé à la fin du support ; fournis code reproductible, protocole, résultats observés et analyse des limites.",
    "solution": "Utiliser le squelette comme point de départ, vérifier les invariants du protocole et documenter chaque décision. La solution complète demande une revue des livrables.",
    "question": "On calcule une moyenne d’imputation sur toutes les lignes avant de séparer train et test.",
    "options": [
      "Fuite ou contamination",
      "Protocole correct",
      "Informations insuffisantes"
    ],
    "answer": 0,
    "explanation": "La transformation apprenante incorpore les observations de test.",
    "source": "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  {
    "id": "ml-hyperparameter-tuning",
    "module": "ml",
    "title": "Cours 7 — Hyperparameter Tuning : Grid Search, Random Search et Optuna",
    "minutes": 240,
    "goal": "Apprendre à optimiser les hyperparamètres d'un modèle de manière rigoureuse, efficace et reproductible, sans surajuster les données de validation.",
    "intuition": "Comprendre le problème, poser un protocole et tester des hypothèses avant de choisir une solution.",
    "theory": "Dans le cours précédent, nous avons appris à détecter et éviter le Data Leakage. Aujourd'hui, nous allons répondre à une question fondamentale : Comment trouver la meilleure configuration d'un algorithme de Machine Learning ? Par exemple, pourquoi choisir un arbre de décision de profondeur 5 plutôt que 10 ? Comment choisir la régularisation d'une régression logistique ? Et comment automatiser ces décisions ? C'est le rôle de l'optimisation des hyperparamètres, appelée Hyperparameter Tuning. ## 1. Paramètres versus hyperparamètres Il faut absolument distinguer ces deux concepts. Paramètres Hyperparamètres Définition Valeurs apprises pendant l'entraînement Valeurs qui configurent l'apprentissage ou le modèle Exemples Poids \\(w\\), biais \\(b\\) Learning rate, profondeur, régularisation Comment les obtenir ? Optimisation lors de fit() Choix ou recherche sur validation Exemple scikit-learn model.coef_ max_depth=5 Prenons une régression logistique : Ici : - C contrôle la force de régularisation : une petite valeur de C correspond à une régularisation plus forte. - max_iter fixe le nombre maximal d'itérations du solveur. - Les coefficients appris sont disponibles dans model.coef_. La valeur C=1.0 n'est pas apprise automatiquement pendant fit(). Nous devons la choisir. 2. Pourquoi optimiser les hyperparamètres ? Supposons que nous entraînions un arbre de décision. Avec max_depth=1, l'arbre risque d'être trop simple : underfitting. Avec max_depth=30, il peut devenir très complexe : overfitting. Une profondeur intermédiaire peut fournir une meilleure généralisation. Explorateur interactif Decision Tree — Hyperparameter Explorer Simulation Profondeur maximale (max_depth)5 Train Accuracy 90.4% Validation Accuracy 86.6% Train Validation Une profondeur intermédiaire peut offrir un compromis intéressant entre capacité et généralisation. Courbes simulées à des fins pédagogiques, et non issues d'un entraînement réel. ## 3. Formalisation mathématique du Hyperparameter Tuning Nous cherchons un ensemble d'hyperparamètres : Avec : Cela signifie que nous avons deux niveaux d'optimisation. Niveau interne : entraîner les paramètres du modèle pour une configuration donnée. Niveau externe ",
    "example": "Le Coding Lab et le projet guidé du cours relient les concepts aux données, aux métriques et aux limites.",
    "code": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import roc_auc_score\n\nX, y = load_breast_cancer(return_X_y=True)\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, stratify=y, random_state=42\n)\npipeline = Pipeline([\n    (\"scaler\", StandardScaler()),\n    (\"model\", LogisticRegression(max_iter=2000))\n])\nparam_grid = {\n    \"model__C\": [0.01, 0.1, 1, 10],\n    \"model__class_weight\": [None, \"balanced\"]\n}\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nsearch = GridSearchCV(\n    pipeline, param_grid, cv=cv, scoring=\"roc_auc\",\n    n_jobs=1, refit=True\n)\nsearch.fit(X_train, y_train)\nprint(\"Meilleurs réglages :\", search.best_params_)\nprint(\"Score de sélection CV (potentiellement optimiste) :\", search.best_score_)\n# Garder le test final intact si l'on compare encore plusieurs méthodes de recherche.\n",
    "pitfalls": [
      "Orienter les choix à partir du test final.",
      "Confondre un score de sélection et une estimation indépendante.",
      "Oublier les groupes, la chronologie ou la disponibilité des variables."
    ],
    "exercise": "Réalise le projet guidé à la fin du support ; fournis code reproductible, protocole, résultats observés et analyse des limites.",
    "solution": "Utiliser le squelette comme point de départ, vérifier les invariants du protocole et documenter chaque décision. La solution complète demande une revue des livrables.",
    "question": "Que distingue paramètres et hyperparamètres ?",
    "options": [
      "Les premiers sont appris ; les seconds règlent la procédure",
      "Les hyperparamètres sont tous appris par fit",
      "Il n’y a aucune différence"
    ],
    "answer": 0,
    "explanation": "Un coefficient est appris ; C ou max_depth sont des réglages sélectionnés dans un protocole.",
    "source": "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  {
    "id": "ml-project-methodology",
    "module": "ml",
    "title": "Cours 8 — Méthodologie complète d’un projet Machine Learning",
    "minutes": 300,
    "goal": "De la définition du problème métier à l'évaluation finale : apprendre à construire un projet ML rigoureux, reproductible et présentable en entretien.",
    "intuition": "Comprendre le problème, poser un protocole et tester des hypothèses avant de choisir une solution.",
    "theory": "Nous arrivons au dernier cours théorique du chapitre 1. Jusqu'à présent, nous avons étudié les concepts séparément : apprentissage, fonctions de perte, généralisation, validation croisée, data leakage et optimisation des hyperparamètres. Aujourd'hui, nous allons assembler toutes ces connaissances dans un véritable workflow de Machine Learning. L'objectif est de comprendre non seulement comment entraîner un modèle, mais aussi comment conduire un projet de manière professionnelle. 1. Le cycle de vie d'un projet Machine Learning Un projet ML ne commence pas avec model.fit(). Il commence par une question métier, puis progresse vers une solution évaluée et exploitable. Workflow inspiré de CRISP-DM, complété par le déploiement et le monitoring. En pratique, les étapes sont itératives : on peut revenir à la préparation des données après l'évaluation. ## 2. Étape 1 — Business Understanding Avant de construire un modèle, nous devons comprendre ce que nous souhaitons résoudre. Prenons un cas de prédiction du churn : une entreprise veut identifier les clients susceptibles de résilier leur abonnement. Il faut répondre à plusieurs questions : - Quelle décision sera prise grâce à la prédiction ? - À quel moment la prédiction sera-t-elle effectuée ? - Quelles données seront disponibles à cet instant ? - Quel est le coût des faux positifs et des faux négatifs ? - Comment mesurera-t-on la réussite du projet ? ### 2.1 Transformer le besoin métier en problème ML Question métier Traduction ML Quels clients risquent de partir ? Classification binaire Quel sera le montant des ventes ? Régression Quels clients ont des comportements similaires ? Clustering Quels produits recommander ? Recommandation / ranking ### 2.2 Définir les métriques La métrique doit refléter le besoin métier. Par exemple, dans une détection de fraude, manquer une fraude peut coûter cher, mais déclencher trop de fausses alertes peut également poser problème. Nous pouvons alors examiner le rappel, la précision, la courbe précision-rappel et les coûts associés aux erreurs. Une accuracy élevée ne suffit pas toujours. ## 3. Étape 2 — Data Understanding Nous devons ensuite comprendre les données. L'Exploratory Data Ana",
    "example": "Le Coding Lab et le projet guidé du cours relient les concepts aux données, aux métriques et aux limites.",
    "code": "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix\n\ndata = load_breast_cancer(as_frame=True)\nX, y = data.data, data.target\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, stratify=y, random_state=42\n)\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nmodels = {\n    \"Dummy\": DummyClassifier(strategy=\"most_frequent\"),\n    \"Logistic\": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),\n    \"RandomForest\": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)\n}\nfor name, model in models.items():\n    results = cross_validate(\n        model, X_train, y_train, cv=cv,\n        scoring={\"auc\": \"roc_auc\", \"ap\": \"average_precision\"},\n        n_jobs=1\n    )\n    print(name, \"AUC CV :\", results[\"test_auc\"].mean(),\n          \"AP CV :\", results[\"test_ap\"].mean())\n# Choix didactique annoncé avant l'expérience : approfondir la logistique.\nsearch = GridSearchCV(\n    models[\"Logistic\"], {\"logisticregression__C\": [0.01, 0.1, 1, 10]},\n    cv=cv, scoring=\"roc_auc\", refit=True, n_jobs=1\n)\nsearch.fit(X_train, y_train)\nbest_model = search.best_estimator_\npredictions = best_model.predict(X_test)\nproba = best_model.predict_proba(X_test)[:, list(best_model.classes_).index(1)]\nprint(\"Test final AUC :\", roc_auc_score(y_test, proba))\nprint(\"Test final AP :\", average_precision_score(y_test, proba))\nprint(classification_report(y_test, predictions, labels=[0, 1],\n      target_names=[\"malignant\", \"benign\"], zero_division=0))\nprint(confusion_matrix(y_test, predictions, labels=[0, 1]))\n# La classe positive est 1, donc benign dans ce dataset pédagogique.\n",
    "pitfalls": [
      "Orienter les choix à partir du test final.",
      "Confondre un score de sélection et une estimation indépendante.",
      "Oublier les groupes, la chronologie ou la disponibilité des variables."
    ],
    "exercise": "Réalise le projet guidé à la fin du support ; fournis code reproductible, protocole, résultats observés et analyse des limites.",
    "solution": "Utiliser le squelette comme point de départ, vérifier les invariants du protocole et documenter chaque décision. La solution complète demande une revue des livrables.",
    "question": "Première étape d’un projet ML ?",
    "options": [
      "Choisir XGBoost",
      "Comprendre le besoin métier",
      "Optimiser les hyperparamètres"
    ],
    "answer": 1,
    "explanation": "Définir la décision et les contraintes précède le choix d’un modèle.",
    "source": "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  ...chapterTwo.lessons,
  ...chapterThree.lessons,
  ...chapterFour.lessons,
  {
    "id": "backprop",
    "module": "dl",
    "title": "Comprendre la rétropropagation",
    "minutes": 45,
    "goal": "Retrouver un gradient à la main et avec autograd.",
    "intuition": "Le réseau compose des calculs. La rétropropagation applique la règle de chaîne pour propager la sensibilité de la perte vers les paramètres. Elle ne met pas les poids à jour.",
    "theory": "ŷ=wx+b, L=(ŷ−y)². ∂L/∂ŷ=2(ŷ−y), ∂ŷ/∂w=x, donc ∂L/∂w=2(ŷ−y)x. Les contributions de plusieurs chemins s’additionnent. Les activations non linéaires permettent de dépasser une simple transformation linéaire.",
    "example": "x=2,w=1,b=0,y=6 : prédiction=2, perte=16, gradient de w=−16 et de b=−8.",
    "code": "import torch\nw = torch.tensor(1., requires_grad=True)\nb = torch.tensor(0., requires_grad=True)\nx, y = torch.tensor(2.), torch.tensor(6.)\nloss = (w*x+b-y)**2\nloss.backward()\nprint(w.grad.item(), b.grad.item())  # -16.0, -8.0",
    "pitfalls": [
      "Croire que backward modifie les poids.",
      "Détacher le graphe pendant le calcul.",
      "Accumuler les gradients involontairement."
    ],
    "exercise": "Pour x=3,w=2,b=1,y=5, calcule les gradients et vérifie avec autograd et une différence finie centrale.",
    "solution": "Prédiction=7, erreur=2 : ∂L/∂w=12, ∂L/∂b=4. (L(w+h)−L(w−h))/(2h) approche 12 avec un h adapté.",
    "question": "loss.backward()…",
    "options": [
      "Calcule les gradients",
      "Choisit le learning rate",
      "Évalue le test"
    ],
    "answer": 0,
    "explanation": "L’optimiseur effectue la mise à jour dans step.",
    "source": "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"
  },
  {
    "id": "training",
    "module": "dl",
    "title": "Écrire une boucle d’entraînement",
    "minutes": 45,
    "goal": "Maîtriser les étapes et distinguer eval de no_grad.",
    "intuition": "Un lot produit une perte, des gradients et une mise à jour. La validation observe sans apprendre sur ses exemples. Le mode du modèle et l’enregistrement des gradients sont deux réglages distincts.",
    "theory": "zero_grad → forward → loss → backward → step. train active les comportements d’entraînement comme dropout. eval change ces comportements mais ne désactive pas autograd. no_grad évite la construction du graphe. Choisir les poids sur validation et garder le test isolé.",
    "example": "Un cas synthétique vérifie qu’un réseau peut ajuster quelques points. La perte sur ces mêmes points ne mesure pas la généralisation.",
    "code": "import torch\nfrom torch import nn\ntorch.manual_seed(42)\nX = torch.randn(64,2)\ny = 2*X[:,:1] - X[:,1:2]\nmodel = nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,1))\noptimizer = torch.optim.Adam(model.parameters(),lr=0.01)\nfor epoch in range(200):\n    model.train()\n    optimizer.zero_grad(set_to_none=True)\n    loss = nn.functional.mse_loss(model(X),y)\n    loss.backward()\n    optimizer.step()\nmodel.eval()\nwith torch.no_grad():\n    print(nn.functional.mse_loss(model(X),y).item())",
    "pitfalls": [
      "step avant backward.",
      "Confondre ajustement et généralisation.",
      "Croire que eval désactive autograd."
    ],
    "exercise": "Sépare train et validation, enregistre les deux pertes et conserve les meilleurs poids sur validation.",
    "solution": "Validation avec eval et no_grad. Conserver deepcopy(model.state_dict()) lorsque la validation s’améliore pour éviter de garder une référence aux poids en cours de modification.",
    "question": "Qu’est-ce qui désactive la construction du graphe ?",
    "options": [
      "model.eval()",
      "torch.no_grad()",
      "optimizer.step()"
    ],
    "answer": 1,
    "explanation": "eval règle les modules, no_grad règle autograd.",
    "source": "https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"
  },
  {
    "id": "overfit",
    "module": "dl",
    "title": "Diagnostiquer le surapprentissage",
    "minutes": 35,
    "goal": "Choisir une expérience à partir des courbes.",
    "intuition": "Le modèle peut mémoriser des détails du train et se dégrader ailleurs. Les courbes sont des indices à confronter aux données, pas une preuve automatique de la cause.",
    "theory": "Écart train/validation croissant : possible surapprentissage. Deux pertes hautes : sous-ajustement, optimisation ratée ou bug. Arrêt anticipé : choisir l’époque sur validation. La régularisation contraint l’apprentissage, sans réparer les fuites.",
    "example": "Train passe de 0,8 à 0,1 ; validation atteint 0,4 puis remonte à 0,7. Conserver le minimum de validation, puis tester un réglage à la fois.",
    "code": "from copy import deepcopy\nbest_loss = float('inf')\nbest_weights = None\n# Après chaque validation :\n# if validation_loss < best_loss:\n#     best_loss = validation_loss\n#     best_weights = deepcopy(model.state_dict())\n# Puis : model.load_state_dict(best_weights)",
    "pitfalls": [
      "Changer plusieurs facteurs simultanément.",
      "Surinterpréter une validation minuscule.",
      "Régulariser avant de vérifier la boucle."
    ],
    "exercise": "Dessine trois paires de courbes : pertes hautes, écart croissant, divergence. Propose un diagnostic et une expérience pour chacune.",
    "solution": "Pertes hautes : surajuster un minuscule lot pour tester le code. Écart : vérifier split puis régularisation. Divergence : chercher NaN, problèmes d’échelle, gradients et taux trop élevé.",
    "question": "Avec arrêt anticipé, retenir…",
    "options": [
      "Les derniers poids",
      "Les poids choisis sur validation",
      "Le meilleur test consulté chaque époque"
    ],
    "answer": 1,
    "explanation": "Le test doit rester indépendant du choix de l’époque.",
    "source": "https://www.deeplearningbook.org/contents/regularization.html"
  },
  {
    "id": "attention",
    "module": "genai",
    "title": "L’attention, étape par étape",
    "minutes": 50,
    "goal": "Expliquer Q,K,V, les formes et le masque causal.",
    "intuition": "Une requête Q est comparée aux clés K pour déterminer comment mélanger les valeurs V d’autres positions. Les poids d’attention ne sont pas automatiquement une explication causale.",
    "theory": "Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V. Q,K de forme (T,dₖ) donnent des scores (T,T). Softmax agit sur les positions consultées, par ligne. Le masque causal interdit le futur. Plusieurs têtes apprennent plusieurs mélanges.",
    "example": "Avec trois tokens, le premier consulte lui-même ; le deuxième les positions 1 et 2 ; le troisième les trois. Les scores interdits valent −∞ avant softmax, donc leurs poids sont nuls.",
    "code": "import torch, math\ntorch.manual_seed(42)\nQ,K,V = [torch.randn(3,4) for _ in range(3)]\nscores = Q @ K.T / math.sqrt(K.shape[-1])\nmask = torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)\nweights = scores.masked_fill(mask,float('-inf')).softmax(-1)\noutput = weights @ V\nassert torch.allclose(weights.sum(-1),torch.ones(3))\nassert output.shape == (3,4)\nprint(weights)",
    "pitfalls": [
      "Softmax sur le mauvais axe.",
      "Oublier le masque autorégressif.",
      "Oublier l’information positionnelle."
    ],
    "exercise": "Retire le masque et explique la fuite créée pendant un apprentissage à prédire le token suivant.",
    "solution": "Une position peut consulter la réponse future, absente en génération réelle. Le protocole devient incohérent avec l’usage.",
    "question": "Q,K=(10,64). Q @ K.T est…",
    "options": [
      "(10,10)",
      "(64,64)",
      "(10,64)"
    ],
    "answer": 0,
    "explanation": "Dix requêtes sont comparées à dix clés.",
    "source": "https://arxiv.org/abs/1706.03762"
  },
  {
    "id": "generation",
    "module": "genai",
    "title": "Du token à la génération",
    "minutes": 40,
    "goal": "Distinguer contexte, apprentissage et échantillonnage.",
    "intuition": "Le modèle prédit une distribution sur le prochain token. Un texte plausible peut être faux. Un token n’est pas nécessairement un mot entier.",
    "theory": "P(séquence)=∏P(tᵢ|t<ᵢ). L’entraînement minimise souvent une log-vraisemblance négative. Une température positive divise les logits avant softmax ; plus basse, elle concentre la distribution. Le choix glouton prend l’argmax. Le contexte ne modifie pas les poids.",
    "example": "Les logits [2,0] donnent [0,881;0,119] à température 1 et [0,982;0,018] à 0,5. Moins de variété ne garantit pas la vérité.",
    "code": "import numpy as np\nlogits = np.array([2.,0.])\nfor temperature in [0.5,1.,2.]:\n    scaled = logits/temperature\n    exp = np.exp(scaled-scaled.max())\n    print(temperature,exp/exp.sum())",
    "pitfalls": [
      "Confondre probabilité de token et fiabilité factuelle.",
      "Supposer que plus de contexte aide toujours.",
      "Comparer sans contrôler prompt et paramètres."
    ],
    "exercise": "Explique la différence entre ajouter un document au prompt et modifier le modèle par fine-tuning.",
    "solution": "Le contexte conditionne la génération sans modifier les poids. Le fine-tuning modifie les paramètres. Aucun des deux ne garantit automatiquement une réponse fidèle et vraie.",
    "question": "Température basse = réponse vraie ?",
    "options": [
      "Oui, sous 1",
      "Non, elle modifie l’échantillonnage",
      "Oui, avec beaucoup de contexte"
    ],
    "answer": 1,
    "explanation": "L’échantillonnage ne vérifie pas les affirmations.",
    "source": "https://huggingface.co/learn/llm-course/chapter1/1"
  },
  {
    "id": "rag",
    "module": "genai",
    "title": "Construire et évaluer un RAG",
    "minutes": 50,
    "goal": "Séparer récupération pertinente et réponse fidèle.",
    "intuition": "Le RAG fournit des passages au générateur. Il faut trouver les preuves puis répondre à partir d’elles. Une réponse fluide peut échouer à l’une ou l’autre étape.",
    "theory": "Documents → découpage → index → recherche → contexte → génération → évaluation. Les embeddings capturent des similarités, pas une garantie de vérité. Recall@k mesure les passages pertinents retrouvés. Évaluer justesse, fidélité et citations. Les documents sont des données non fiables, jamais des instructions.",
    "example": "Une recherche lexicale sert de baseline. Conserver identifiants et sources. Prévoir des questions non couvertes pour vérifier la capacité à signaler une absence de réponse.",
    "code": "import re\npassages = {'p1':'Le gradient est une variation locale.',\n            'p2':'La validation choisit les hyperparamètres.',\n            'p3':'Le test mesure la performance finale.'}\ndef tokens(text):\n    return set(re.findall(r'\\w+',text.lower()))\nquery = tokens('À quoi sert la validation ?')\nranked = sorted(passages,key=lambda k:\n                len(query & tokens(passages[k])),reverse=True)\nprint([(k,passages[k]) for k in ranked[:2]])\n# Baseline lexicale, sans générateur.",
    "pitfalls": [
      "Évaluer uniquement la fluidité.",
      "Perdre les sources au découpage.",
      "Suivre une instruction malveillante dans un document."
    ],
    "exercise": "Crée dix questions avec passages attendus, dont deux sans réponse. Mesure la récupération et les fausses réponses sur les cas non couverts.",
    "solution": "Rappel=pertinents retrouvés / pertinents attendus pour les cas couverts. Évaluer séparément les questions sans réponse, dont le dénominateur serait nul. Un score de similarité seul ne garantit pas un contexte suffisant.",
    "question": "La preuve n’est pas récupérée : examiner d’abord…",
    "options": [
      "Index et récupération",
      "Couleur de l’interface",
      "Uniquement la température"
    ],
    "answer": 0,
    "explanation": "La première défaillance intervient avant la génération.",
    "source": "https://arxiv.org/abs/2005.11401"
  },
  {
    "id": "reproducibility",
    "module": "engineering",
    "title": "Rendre une expérience reproductible",
    "minutes": 35,
    "goal": "Relier résultats, code, données et configuration.",
    "intuition": "Un score dans un notebook ne suffit pas. Une autre personne doit comprendre ce qui a été exécuté et pouvoir relancer l’expérience.",
    "theory": "Conserver versions, configuration, graine, partitions, métriques et artefacts. Une graine ne garantit pas l’identité numérique entre matériels. Tests de code et évaluation statistique répondent à des questions différentes. Le test final reste isolé de la sélection.",
    "example": "Identifier une expérience par commit, données et configuration. Chaque métrique indique partition et calcul. Ne jamais annoncer un score non observé.",
    "code": "import json\nfrom pathlib import Path\nconfig = {'seed':42,'model':'baseline','split':'fixed-v1'}\nresult = {'config':config,'status':'example_not_executed'}\nPath('artifacts').mkdir(exist_ok=True)\nPath('artifacts/run.json').write_text(\n    json.dumps(result,indent=2),encoding='utf-8')",
    "pitfalls": [
      "Inventer une métrique.",
      "Versionner des secrets.",
      "Ne pas réexécuter le notebook depuis zéro."
    ],
    "exercise": "Écris les instructions de reproduction : environnement, données, commande et artefacts. Fais-les tester dans un environnement propre.",
    "solution": "Préciser versions, accès autorisé aux données et commande. Mentionner étapes manuelles et ressources manquantes. Vérifier que les artefacts annoncés sont retrouvés.",
    "question": "Une graine suffit-elle à reproduire exactement partout ?",
    "options": [
      "Toujours",
      "Non",
      "Seulement avec un grand modèle"
    ],
    "answer": 1,
    "explanation": "Matériel, versions et opérations non déterministes peuvent modifier les résultats.",
    "source": "https://docs.pytorch.org/docs/stable/notes/randomness.html"
  },
  {
    "id": "monitoring",
    "module": "engineering",
    "title": "Observer un modèle après déploiement",
    "minutes": 40,
    "goal": "Distinguer dérive, baisse de qualité et panne.",
    "intuition": "Le modèle appartient à un système : les données changent et les labels arrivent parfois tard. Un service rapide peut retourner des prédictions inutiles.",
    "theory": "Suivre schéma, manquants, distributions, latence, erreurs et performance quand les labels arrivent. Une dérive ne prouve pas une dégradation et son absence ne garantit pas la qualité. Comparer des cohortes pertinentes et prévoir un retour à une version stable.",
    "example": "Une surface passant de m² à cm² produit des prédictions absurdes sans panne serveur. Vérifier les unités et les plages plausibles avant d’attendre les vrais prix.",
    "code": "def validate_features(row):\n    if not {'surface_m2','rooms'}.issubset(row):\n        raise ValueError('Variables manquantes')\n    if not 5 <= row['surface_m2'] <= 2000:\n        raise ValueError('Surface à vérifier')\n    if not 1 <= row['rooms'] <= 50:\n        raise ValueError('Pièces à vérifier')\n    return row\n# Plages illustratives à adapter au métier.",
    "pitfalls": [
      "Confondre disponibilité et qualité.",
      "Réentraîner sans diagnostic.",
      "Journaliser des contenus sensibles inutilement."
    ],
    "exercise": "Définis cinq indicateurs, leur fréquence et les actions associées. Prévois les labels retardés.",
    "solution": "Schéma invalide, manquants, latence p95, erreurs et MAE sur cohortes labellisées. Donner une action à chaque alerte ; distinguer signaux indirects et performance réellement mesurée.",
    "question": "Une dérive prouve-t-elle une baisse de qualité ?",
    "options": [
      "Toujours",
      "Non, examiner son effet",
      "Seulement en numérique"
    ],
    "answer": 1,
    "explanation": "La dérive est un signal d’enquête, pas un verdict de performance.",
    "source": "https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring"
  }
];
export const projects = [
  {"id":"customer-segmentation-anomaly","title":"Customer Segmentation & Anomaly Detection System","tag":"CHAPITRE 04 · MINI-PROJET","description":"Cahier des charges du projet de fin de chapitre : combiner clustering, PCA, anomalies et visualisations. À réaliser après les dix cours.","steps":["Documenter les données clients, leur provenance et les variables disponibles sans cible","Diagnostiquer les absences et les échelles puis construire un prétraitement reproductible","Comparer K-Means, hiérarchique et méthodes de densité sur les mêmes données","Évaluer séparation, stabilité, effectifs et interprétabilité des groupes","Étudier PCA et distinguer projection exploratoire et espace d’apprentissage","Comparer des détecteurs d’anomalies et définir une procédure d’investigation","Construire des visualisations interactives des segments et des points atypiques","Livrer notebook, scripts, versions, résultats observés et limites ; défendre la recommandation"]},
  {id:"data-preparation-pipeline",title:"Projet final — Data Preparation Pipeline",tag:"CHAPITRE 02 · SYNTHÈSE",description:"Projet guidé synthétique fourni au cours 10 : nettoyage, validation et sauvegarde. Prolongement sur un dataset réel documenté.",steps:["Réaliser le projet guidé du cours 10 et vérifier la sauvegarde du pipeline","Choisir un dataset réel et documenter les variables et la cible","Réserver le test et produire le diagnostic de qualité sur train","Justifier imputation, outliers, scaling et encodage","Construire et sélectionner des variables sans fuite","Assembler un ColumnTransformer et un pipeline évalué en validation","Livrer le code reproductible, les résultats et les limites"]},
  {
    "id": "tabular",
    "title": "Un modèle tabulaire de bout en bout",
    "tag": "MACHINE LEARNING",
    "description": "Comparer une baseline et un modèle, puis expliquer les erreurs.",
    "steps": [
      "Définir cible, données et métrique",
      "Créer une séparation et une baseline",
      "Construire une pipeline sans fuite",
      "Comparer sur validation et analyser dix erreurs",
      "Évaluer sur test et documenter les limites"
    ]
  },
  {
    "id": "network",
    "title": "Mon premier réseau, vraiment compris",
    "tag": "DEEP LEARNING",
    "description": "Écrire la boucle et expliquer les courbes.",
    "steps": [
      "Fixer données et partitions",
      "Vérifier les formes et surajuster un petit lot",
      "Séparer entraînement et validation",
      "Comparer un réglage à la baseline",
      "Sauvegarder les poids et fournir une prédiction reproductible"
    ]
  },
  {
    "id": "assistant",
    "title": "Un assistant pour mes propres cours",
    "tag": "GENAI",
    "description": "Construire un RAG sur tes supports et mesurer ses limites.",
    "steps": [
      "Préparer des documents autorisés, identifiés et sourcés",
      "Créer des questions, y compris sans réponse",
      "Évaluer une baseline de récupération",
      "Générer avec citations, refus et gestion des documents non fiables",
      "Mesurer qualité, coûts, latence et échecs"
    ]
  },
  {
    "id": "wine",
    "title": "Wine : mon premier pipeline de classification",
    "tag": "MACHINE LEARNING · CHAPITRE 01",
    "description": "Un projet reproductible : données, split stratifié, baseline, pipeline, métriques multiclasses et analyse d’erreurs.",
    "steps": [
      "Charger Wine et documenter les variables et classes",
      "Réserver un test stratifié 80/20 sans fuite",
      "Établir une baseline et une pipeline StandardScaler + LogisticRegression",
      "Entraîner sur train et mesurer accuracy et F1 macro sur le test final",
      "Examiner les probabilités de cinq observations et l’ordre de classes_",
      "Analyser des erreurs et documenter les versions et limites",
      "Fournir un script reproductible, un README et une explication des décisions"
    ]
  },
  {
    "id": "numpy-multivariate",
    "title": "Descente de gradient multivariée avec NumPy",
    "tag": "MACHINE LEARNING · CHAPITRE 01",
    "description": "Dériver le gradient matriciel, vérifier les dimensions et diagnostiquer la convergence sur trois features.",
    "steps": [
      "Générer les données synthétiques et conserver leur configuration",
      "Dériver les gradients et préciser toutes les dimensions",
      "Implémenter des mises à jour simultanées de w et b",
      "Vérifier les gradients par différences finies",
      "Comparer plusieurs learning rates avec courbes et résultats observés",
      "Analyser l’effet du bruit et de l’échelle des variables",
      "Fournir le code reproductible et un rapport expliquant les limites"
    ]
  },
  {
    "id": "ridge-generalization",
    "title": "Régulariser une régression polynomiale",
    "tag": "CHAPITRE 01 · COURS 04",
    "description": "Comparer les erreurs train/validation et justifier le choix de alpha sans utiliser de test final.",
    "steps": [
      "Reproduire les données et le split du Coding Lab",
      "Comparer les degrés 1, 3 et 15",
      "Comparer une baseline non régularisée à Ridge",
      "Tracer MSE train et validation selon alpha",
      "Documenter le réglage retenu, ses limites et les versions"
    ]
  },
  {
    "id": "time-validation",
    "title": "Construire une validation temporelle",
    "tag": "CHAPITRE 01 · COURS 05",
    "description": "Définir l’horizon et comparer une régression à une baseline retardée, sans informations futures.",
    "steps": [
      "Définir une prévision à un pas et la disponibilité des variables",
      "Construire uniquement des lags causaux",
      "Utiliser des plis respectant la chronologie",
      "Comparer modèle et baseline lag_1 en MAE",
      "Expliquer ce qui changerait en multi-pas"
    ]
  },
  {
    "id": "leakage-free",
    "title": "Auditer et construire un pipeline sans fuite",
    "tag": "CHAPITRE 01 · COURS 06",
    "description": "Imputation, mise à l’échelle et sélection ajustées dans chaque pli train.",
    "steps": [
      "Cartographier la disponibilité de chaque variable",
      "Réserver le protocole de validation",
      "Construire imputation, scaler, sélection et logistique",
      "Évaluer dans StratifiedKFold sans ajustement préalable",
      "Auditer groupes, doublons, temps et sélection de modèles"
    ]
  },
  {
    "id": "random-forest-tuning",
    "title": "Optimiser une forêt avec un budget explicite",
    "tag": "CHAPITRE 01 · COURS 07",
    "description": "Recherche Optuna sur Wine, F1 macro et test final réservé.",
    "steps": [
      "Réserver un test stratifié",
      "Définir métrique, espace de recherche et budget",
      "Écrire la fonction objective en CV sur train",
      "Tracer et conserver les essais avec leurs seeds",
      "Réentraîner la configuration retenue",
      "Évaluer une fois sur test et expliquer le biais de sélection"
    ]
  },
  {
    "id": "capstone-chapter-one",
    "title": "Projet final — Classification tabulaire de A à Z",
    "tag": "CHAPITRE 01 · SYNTHÈSE",
    "description": "Sur Wine, livrer EDA, baseline, comparaison en CV, tuning, analyse des erreurs et dépôt reproductible.",
    "steps": [
      "Cadrer le problème et documenter les données Wine",
      "Réserver le test final stratifié et explorer le train",
      "Construire une baseline Dummy et des pipelines sans fuite",
      "Comparer plusieurs modèles sur les mêmes plis",
      "Optimiser avec un budget fixé et sans utiliser le test",
      "Réentraîner et réaliser une seule évaluation finale",
      "Analyser les erreurs et les probabilités multiclasses",
      "Fournir notebook, scripts, tableau comparatif, tests et README",
      "Défendre la méthode et ses limites lors d’une revue"
    ]
  }
];
