export const mlIntroLesson = {
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
};
export const mlScenarios = [
  {
    "question": "Une entreprise veut calculer une taxe à partir d’un taux connu et d’une base définie.",
    "options": [
      "Machine Learning",
      "Règles déterministes"
    ],
    "answer": 1,
    "explanation": "Appliquer une formule connue fournit un calcul exact. Il n’y a pas de relation inconnue à apprendre."
  },
  {
    "question": "Une plateforme veut prédire la résiliation d’abonnements à partir d’historiques et de résiliations connues.",
    "options": [
      "Machine Learning supervisé",
      "Algorithme de tri"
    ],
    "answer": 0,
    "explanation": "Une cible historique est disponible. Une approche supervisée peut être comparée à une règle simple, avec un protocole adapté."
  },
  {
    "question": "Une entreprise veut regrouper ses clients selon leurs habitudes sans catégories prédéfinies.",
    "options": [
      "Classification supervisée",
      "Clustering non supervisé"
    ],
    "answer": 1,
    "explanation": "On recherche une structure sans labels imposés. Les groupes doivent ensuite être interprétés et vérifiés."
  },
  {
    "question": "Un service veut calculer le total exact des ventes enregistrées pendant une journée.",
    "options": [
      "Agrégation SQL",
      "Deep Learning"
    ],
    "answer": 0,
    "explanation": "Une somme sur les transactions répond au besoin. Prédire les ventes futures serait une autre tâche."
  }
];
export const mlInterview = [
  {
    "level": "Fondamental",
    "question": "Quelle différence entre AI, ML et DL ?",
    "answer": "L’IA est le domaine général. Le ML désigne des méthodes apprenant à partir de données. Le DL est une famille du ML utilisant des réseaux à plusieurs couches. La GenAI désigne la génération de contenu et ne forme pas une quatrième couche nécessairement emboîtée."
  },
  {
    "level": "Fondamental",
    "question": "Quelle différence entre un paramètre et un hyperparamètre ?",
    "answer": "Un paramètre, par exemple un coefficient de régression, est ajusté pendant l’apprentissage. Un hyperparamètre, comme la force de régularisation, règle la procédure et peut être choisi à l’aide de la validation. Le test final reste à l’écart de cette sélection."
  },
  {
    "level": "Intermédiaire",
    "question": "Pourquoi être performant sur train et mauvais sur de nouvelles données ?",
    "answer": "Le modèle peut surapprendre des particularités du train. Une fuite, un échantillon peu représentatif ou un changement de distribution peuvent aussi fausser l’évaluation. Il faut examiner le split, les données, la complexité et le protocole ; le surapprentissage n’est pas l’unique explication."
  },
  {
    "level": "Intermédiaire",
    "question": "Dans quel cas déconseillerais-tu le ML à une entreprise ?",
    "answer": "Si une règle déterministe suffit, si les données pertinentes manquent ou si l’amélioration espérée ne justifie pas les coûts et risques. Un coût d’erreur élevé demande des contrôles adaptés, une solution de repli et parfois une validation humaine."
  },
  {
    "level": "Avancé",
    "question": "Quelle différence entre minimiser la loss d’entraînement et réussir en production ?",
    "answer": "La loss d’entraînement mesure l’ajustement sur les exemples observés. En production comptent aussi la généralisation, les coûts d’erreur, la calibration, les sous-populations, les données disponibles à temps, la latence, le coût et la robustesse aux changements."
  }
];
export const mlObjectives = [
  "Je peux expliquer AI, ML, DL et GenAI.",
  "Je connais le vocabulaire X, y, modèle, paramètres et hyperparamètres.",
  "Je sais distinguer les quatre cadres d’apprentissage.",
  "Je peux expliquer quand utiliser ou éviter le ML.",
  "J’ai compris et exécuté le premier exemple Python.",
  "Je peux répondre oralement aux questions d’entretien."
];
export const mlIntroHeadings = ["Comprendre le Machine Learning","Le vocabulaire fondamental","Les familles d’apprentissage","Quand utiliser le ML ?","Coding Lab : une première baseline","Questions d’entretien","Validation du cours"];
