# LearnLoop — cours et atelier Python

Application pédagogique en français : parcours, supports détaillés, quiz, progression par compte et exercices exécutables. Le catalogue ML comprend 60 cours répartis sur les six premiers chapitres, dont les 12 cours du chapitre 6.

## Développement local

Prérequis : Node.js >= 22.13, npm et la base D1 configurée pour les comptes. Les schémas et migrations figurent dans db/ et drizzle/.

~~~powershell
npm ci
npm run dev
~~~

Le serveur portable écoute normalement sur http://127.0.0.1:5173. Les réglages d'hébergement existants sont dans .openai/hosting.json. Aucun secret ne doit être ajouté au dépôt.

## Exercices Python

L'atelier apparaît dans les supports et l'onglet Exercices. Il charge les squelettes de public/downloads/, exécute Python dans un worker Pyodide et affiche console, ligne d'erreur, graphiques et tests disponibles. Le premier lancement télécharge Python et ses bibliothèques. L'arrêt détruit le worker ; une limite de 180 secondes couvre chargement et calcul. Le code est enregistré dans le compte, exportable, réinitialisable et comparé à un corrigé révélé à la demande.

La syntaxe est contrôlée après le premier lancement. Les tests sont comportementaux et indépendants du texte de la solution. Un exercice sans tests indique explicitement qu'une exécution réussie ne prouve pas sa correction pédagogique. Les projets XAI complets et les intégrations optionnelles nécessitent un environnement Python local ; voir le README du chapitre 5 à la racine du dépôt.

## Vérification

~~~powershell
npx tsc --noEmit --incremental false
python -B tests/verify-chapter5-python.py
python -B tests/verify-chapter6-python.py
npm run build
~~~

Les tests navigateur de l'atelier sont dans tests/python-workshop.cjs. Le script utilise un navigateur Chrome/Edge local en débogage et un compte de test local pour ne pas modifier les comptes réels.

## Sources du parcours

- lib/curriculum.ts : catalogue et métadonnées.
- lib/ml-chapter-five.json : dix cours, quiz, objectifs et corrigés.
- lib/ml-chapter-six.json : douze cours, 60 questions de quiz, objectifs, scripts et corrigés.
- lib/ml-roadmap.ts et lib/learning-paths.ts : chapitres et prérequis.
- components/python-workshop.tsx : éditeur, brouillons et interface d'exécution.
- public/python/runner.js et checks.py : moteur et tests.
- public/downloads/ : labs, starters et solutions.


## Chapitre 6 complet

Partie A : 7 cours de séries temporelles. Partie B : 5 cours de ML spécialisé. Les supports, quiz et métadonnées sont dans `lib/ml-chapter-six.json`. Les 36 scripts sont dans `public/downloads/`. Le [README du chapitre](../03_Machine_learning/CHAPITRE_06/README.md) détaille les installations, commandes, extensions optionnelles et protocoles.
