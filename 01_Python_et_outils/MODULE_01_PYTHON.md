# Module P01 — Python pour construire un premier outil fiable

Statut : conception détaillée du module. Les futurs supports et corrections devront être exécutés et relus avant publication.
Prérequis : aucun en programmation ; savoir utiliser les quatre opérations.
Résultat : un outil qui lit une liste de mesures, détecte les entrées invalides et produit un rapport.
Ce module précède la conception du package complet : modules, fichiers et pytest sont approfondis dans P03/P05.

## Projet fil rouge

Construire « DataCheck », un outil de contrôle de mesures. Version initiale : valeurs directement définies dans le programme. Versions suivantes : CSV, exceptions explicites, package, tests et ligne de commande.

Contrat initial : accepter une liste de nombres finis, compter les éléments, calculer moyenne/min/max, signaler une entrée vide, et afficher un rapport lisible. Le traitement des valeurs non numériques doit être choisi et expliqué ; ne pas les transformer silencieusement en zéro.

## Leçon 1 — Valeurs, noms et types

Objectifs : distinguer une valeur et le nom qui y fait référence ; différencier int, float, str, bool et None ; prédire une conversion ; expliquer qu'un nombre flottant a une précision limitée.

Déroulé : problème d'un prix reçu comme texte ; affectation ; type ; opérations ; conversion explicite ; cas de 0.1+0.2 ; affichage avec f-string.

Exercices :
1. Prédire le résultat de "12"+"3" puis int("12")+3.
2. Expliquer pourquoi remplacer toutes les chaînes invalides par 0 fausse une moyenne.
3. Choisir les types pour une mesure, son unité, un identifiant et l'absence de mesure.
4. Déboguer une division exécutée sur une chaîne.

Correction attendue : concaténation "123" versus addition 15 ; zéro représente une vraie mesure, pas une absence ; ne pas convertir un identifiant à zéros initiaux en entier sans raison.

Quiz : type et valeur, conversions, différence entre absence et zéro. Les mauvaises réponses doivent correspondre à des confusions expliquées.

Projet : définir trois mesures et produire un rapport avec unités.

## Leçon 2 — Conditions et cas limites

Objectifs : évaluer une condition booléenne ; utiliser if/elif/else ; distinguer bornes incluses et exclues ; traiter une entrée vide.

Intuition : une règle métier devient une décision explicite. Expliquer and/or/not, comparaisons et court-circuit à partir de règles compréhensibles.

Exercices : seuil d'alerte ; valeurs aux bornes ; condition mal parenthésée ; confusion entre chaîne vide et nombre zéro.

Maths : intervalle fermé [a,b] versus intervalle ouvert ; traduire 0≤x≤100 en condition.

Projet : classer une mesure comme acceptable ou à vérifier. Écrire cinq cas attendus avant le programme.

## Leçon 3 — Boucles et invariants

Objectifs : parcourir une collection ; expliquer compteur et accumulateur ; prévoir ce qui reste vrai à chaque étape ; éviter une erreur de borne.

Démonstration : après k éléments, total est la somme des k premières valeurs et count vaut k. Initialisation puis conservation de l'invariant.

Exercices : somme manuelle ; compteur de valeurs dépassant un seuil ; range ; boucle infinie à diagnostiquer ; calcul d'une moyenne avec et sans valeurs.

Correction : ne pas diviser avant de connaître le dénominateur ; tester le cas vide séparément.

Projet : calculer les statistiques sans bibliothèque externe et comparer au calcul manuel.

## Leçon 4 — Fonctions et contrats

Objectifs : distinguer paramètres, arguments et valeur de retour ; séparer calcul et affichage ; éviter les dépendances cachées.

Exercices : transformer un script en fonctions ; corriger un return manquant ; expliquer print versus return ; proposer une politique explicite pour le cas vide.

Projet : une fonction de statistiques et une fonction de présentation. L'annotation de type sert de lecture ; elle ne valide pas une entrée à l'exécution.

Critère : appeler la fonction sur une autre liste sans modifier son corps.

## Leçon 5 — Structures et mutabilité

Objectifs : choisir liste, tuple, dictionnaire ou ensemble ; comprendre référence partagée et mutation ; distinguer déduplication légitime et perte de données.

Exercices : compter les catégories ; montrer deux noms pour une même liste ; trier une copie ; conserver des mesures répétées qui représentent de vraies observations.

Projet : retourner un dictionnaire de statistiques et conserver les données originales intactes, selon le contrat annoncé.

Approfondissement P02 : copies imbriquées, compréhensions et coût des opérations.

## Leçon 6 — Erreurs et débogage

Objectifs : lire un traceback ; isoler un exemple minimal ; distinguer erreur de type, syntaxe et logique ; vérifier une hypothèse avant de modifier le code.

Exercices : trois bugs de natures différentes ; moyenne calculée avec le mauvais compteur ; changement d'unité silencieux ; chaîne non convertible.

Projet : fournir un message compréhensible pour une entrée invalide. Les exceptions, fichiers et journaux structurés seront détaillés dans P03/P05.

## Leçon 7 — Vérifier son programme

Objectifs : écrire des cas avec résultats attendus ; vérifier des invariants ; expliquer pourquoi un exemple heureux ne suffit pas.

Cas minimaux : liste vide, un élément, valeurs identiques, négatifs, valeurs répétées et entrée invalide. La politique d'acceptation des booléens et des valeurs infinies doit être explicitée dans le contrat.

Exercice de transfert : déplacer toutes les mesures de +c ; expliquer pourquoi la moyenne se déplace de +c. Multiplier par une constante positive et prévoir min/max.

Projet : documenter les vérifications ; migrer ensuite vers pytest en P05. Ne pas annoncer de test passé sans exécution.

## Leçon 8 — Soutenance et variante

Livrables : script lisible, fonctions, contrat, cas attendus, sorties observées, explication des erreurs rencontrées et limites.

Variante sans guide : adapter DataCheck à des mesures dont certaines sont absentes. Afficher séparément nombre total, nombre valide et nombre manquant. Ne pas mélanger ces dénominateurs.

Questions :
- Pourquoi choisir ce type et cette structure ?
- Que signifie une ligne ou une mesure ?
- Qu'arrive-t-il si la liste est vide ?
- Une valeur zéro est-elle manquante ?
- Comment démontrer que la liste d'origine est conservée ?
- Quelle règle métier justifie le traitement d'une entrée invalide ?

## Évaluation

Le quiz vérifie le vocabulaire et le raisonnement. La validation exige une variante du projet réalisée avec documentation mais sans solution guidée, puis une explication de ses choix.

Répartition proposée : contrat et cas limites 25 ; logique et fonctions 30 ; vérification 25 ; lisibilité et explication 20. Les erreurs silencieuses de dénominateur et de traitement des absences doivent être corrigées avant passage.

## Scénarios vidéo à produire

Une vidéo courte par objectif : situation ; prédiction sans code ; exécution ; explication ; pause exercice ; correction. Pour la mutabilité, représenter deux noms reliés au même objet. Pour la boucle, afficher simultanément indice, valeur, total et compteur.

Chaque vidéo aura une transcription et un exercice accessible sans lecture du média. Le scénario ne compte pas comme une vidéo produite.

## Révisions

J+1 : expliquer print/return, calculer une boucle à la main.
J+7 : réparer une variante incorrecte et ajouter un cas limite.
J+30 : reconstruire la fonction de statistiques et comparer ses choix au premier essai.

## Suite du projet

P03 : lecture CSV/JSON, gestion d'erreurs et modules.
P04 : lecteur itérable lorsque le volume le justifie.
P05 : package, pyproject, annotations, pytest, journalisation et documentation.
P06 : concurrence uniquement si le profil de charge démontre un besoin.
