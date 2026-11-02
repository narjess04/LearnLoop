# Outils professionnels — rendre Taskflow reproductible

Support adapté de ton exemple. Durées indicatives ; chapitres avancés à approfondir. Les commandes ne sont pas exécutées dans cette page.

Durée : 3 à 4 semaines · Prérequis : bases de Python (Cours 1) · Objectif : travailler comme un développeur professionnel : terminal à l’aise, Git au quotidien, environnements reproductibles, premier conteneur Docker.
Ce cours est souvent négligé par les débutants, mais c’est lui qui sépare “ça marche sur mon PC” d’un travail réellement partageable et déployable.
## 1. Terminal et système

Les serveurs, le cloud, les GPU et Docker tournent presque tous sous Linux. Sous Windows, installe WSL2 (Ubuntu). Sous macOS, le terminal est déjà proche de Linux.
~~~bash
pwd                      # où suis-je ?
ls -la                   # lister (fichiers cachés + détails)
cd projets/ml            # changer de dossier
cd ..                    # remonter ; cd ~ = dossier personnel ; cd - = dossier précédent

mkdir -p data/raw        # créer (avec parents)
touch notes.txt          # créer un fichier vide
cp a.txt b.txt           # copier ; cp -r pour un dossier
mv a.txt data/           # déplacer / renommer
rm fichier.txt           # supprimer (IRRÉVERSIBLE, pas de corbeille)
rm -r dossier/           # supprimer un dossier
~~~
~~~bash
cat fichier.txt          # afficher
head -n 20 data.csv      # 20 premières lignes
tail -f logs.txt         # suivre un fichier en direct (logs)
less gros_fichier.txt    # lecture paginée (q pour quitter)
wc -l data.csv           # nombre de lignes

grep "erreur" logs.txt              # chercher un motif
grep -rn "def train" src/           # récursif, avec numéros de ligne
find . -name "*.py" -size +1M       # chercher des fichiers
~~~
## 2. Pipes et diagnostic

Le principe clé : enchaîner de petits outils.
~~~bash
cat logs.txt | grep ERROR | sort | uniq -c | sort -rn | head
# → les 10 erreurs les plus fréquentes

echo "hello" > f.txt     # écrase
echo "suite" >> f.txt    # ajoute
python train.py > out.log 2> err.log    # sépare sortie et erreurs
python train.py > all.log 2>&1          # tout dans un fichier
~~~
~~~bash
ls -l                    # -rwxr-xr-- : propriétaire / groupe / autres
chmod +x script.sh       # rendre exécutable
chmod 600 ~/.ssh/id_rsa  # clé privée : lisible par toi seul

ps aux | grep python     # processus
top / htop               # utilisation CPU/RAM
kill 1234                # arrêter un processus (kill -9 en dernier recours)
nvidia-smi               # état des GPU
df -h                    # espace des systèmes de fichiers
du -sh data/             # taille d’un dossier

curl -s https://api.exemple.com/data | head
ssh user@serveur         # connexion à distance
scp modele.pt user@serveur:/data/    # copier via SSH
tmux                     # sessions persistantes (indispensable pour les longs entraînements)
~~~
~~~bash
export OPENAI_API_KEY="..."     # JAMAIS écrit dans le code ni dans Git
echo $PATH

#!/usr/bin/env bash
set -euo pipefail               # options utiles, avec exceptions selon le contexte
for f in data/*.csv; do
    echo "Traitement de $f"
    python process.py "$f"
done
~~~
1. Dans un dossier de logs, trouve les 5 adresses IP les plus fréquentes avec un pipe (awk, sort, uniq).
2. Écris un script bash qui sauvegarde un dossier dans une archive .tar.gz datée.
3. Lance un script Python long dans tmux, détache la session, déconnecte-toi, reviens.
Validation : reproduire un scénario au terminal et expliquer les commandes. La souris et les outils d’accessibilité restent autorisés.
## 3. Git et ses trois zones

Git enregistre des instantanés (commits) de ton projet. Trois zones :
~~~
Working directory  →  git add  →  Staging area  →  git commit  →  Dépôt local  →  git push  →  GitHub
~~~
~~~bash
git config --global user.name "Ton Nom"
git config --global user.email "toi@exemple.com"

git init                     # nouveau dépôt
git clone <url>              # copier un dépôt existant
~~~
~~~bash
git status                   # où en suis-je ?
git add src/train.py         # ajouter un fichier précis (évite `git add .` aveugle)
git commit -m "Ajoute la boucle d'entraînement"
git log --oneline --graph    # historique
git diff                     # voir les modifications non ajoutées
git diff --staged            # voir ce qui sera commité
~~~
## 4. Ignorer et corriger

À créer dès le début :
~~~
__pycache__/
.venv/
.env
*.pt
*.ckpt
data/raw/
.ipynb_checkpoints/
.DS_Store
~~~
Règle d’or ML : ne commite jamais de gros fichiers (datasets, poids de modèles) ni de secrets. Utilise DVC ou Git LFS pour les gros fichiers (vu en Phase 6).
~~~bash
git restore fichier.py            # abandonne les modifs non ajoutées
git restore --staged fichier.py   # retire du staging
git commit --amend                # corrige le dernier commit (non poussé)
git revert <hash>                 # crée un commit qui annule un commit (sûr, même poussé)
git stash / git stash pop         # met de côté du travail en cours
~~~
⚠️ git reset --hard et git push --force détruisent du travail : ne les utilise qu’en sachant exactement pourquoi.
- Impératif, court, précis : Corrige la fuite de données dans le split
- Un commit = une idée logique
- Évite : fix, update, wip2
1. Crée un dépôt, fais 5 commits cohérents, lis l’historique.
2. Fais volontairement une erreur, puis annule-la avec restore, puis avec revert.
3. Ajoute un .gitignore après avoir commité un fichier à ignorer, puis retire-le du suivi (git rm --cached).
## 5. Branches et collaboration

~~~bash
git switch -c feature/ajout-metriques     # créer + basculer
git switch main
git branch -d feature/ajout-metriques     # supprimer après fusion
~~~
~~~bash
git switch main
git merge feature/ajout-metriques         # conserve l'historique tel quel

git switch feature/x
git rebase main                           # rejoue tes commits au-dessus de main (historique linéaire)
~~~
Règle : ne jamais rebaser des commits déjà partagés avec d’autres.
Quand deux personnes modifient les mêmes lignes :
~~~
<<<<<<< HEAD
version de la branche actuelle
=======
version de l'autre branche
>>>>>>> feature/x
~~~
Tu édites à la main, tu gardes ce qui est juste, tu supprimes les marqueurs, puis git add et git commit (ou git rebase --continue).
~~~bash
git remote add origin git@github.com:toi/projet.git
git push -u origin main
git pull --rebase           # récupérer les changements des autres
git fetch                   # récupérer sans fusionner
~~~
- SSH : génère une clé (ssh-keygen -t ed25519) et ajoute la clé publique à GitHub
- Fork : ta copie d’un dépôt tiers, pour contribuer à l’open source
1. Créer une branche depuis main
2. Commits petits et clairs
3. git push puis ouvrir une Pull Request
4. Revue de code, corrections, CI verte
5. Merge (squash ou merge commit) puis suppression de la branche
À connaître : Conventional Commits (feat:, fix:, docs:), protection de la branche main, git bisect (trouver quel commit a cassé quelque chose), git cherry-pick, git worktree.
1. Crée un conflit volontaire avec deux branches, puis résous-le.
2. Fais une vraie contribution à un dépôt open source (même une correction de typo dans la doc).
3. Ouvre une PR sur ton propre dépôt et relis-la comme si c’était celle d’un collègue.
Validation : préparer et défendre une PR avec revue. L’acceptation par un projet externe ne conditionne pas la réussite.
## 6. Intégration continue

Un pipeline qui lance lint et tests à chaque push :
~~~yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest
~~~
Vérifie les versions actuelles des actions utilisées dans leur documentation.
Notions : secrets GitHub, matrice de versions Python, badge de statut dans le README.
## 7. Environnements Python

Le projet A a besoin de numpy 1.26, le projet B de numpy 2.x. Sans isolation, tout casse. Un environnement virtuel donne à chaque projet ses propres paquets.
~~~bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install numpy pandas
pip freeze > requirements.txt
pip install -r requirements.txt
deactivate
~~~
~~~bash
uv init mon-projet
uv add numpy pandas scikit-learn
uv add --dev pytest ruff
uv sync                          # reconstruit l'env depuis le lockfile
uv run python train.py
uv python install 3.12           # gère aussi les versions de Python
~~~
Le fichier pyproject.toml liste les dépendances, et le lockfile (uv.lock) fige les versions exactes pour la reproductibilité.
Utile quand tu as besoin de dépendances non-Python (CUDA, bibliothèques C) ou pour certains environnements scientifiques.
~~~bash
conda create -n ml python=3.12
conda activate ml
conda install pytorch -c pytorch
conda env export > environment.yml
~~~
- Un environnement par projet, jamais pip install dans le Python système
- Toujours versionner pyproject.toml + lockfile, pas le dossier .venv
- Épingler les versions pour les projets d’application, rester souple pour les bibliothèques
- Jupyter : uv add --dev ipykernel pour associer l’environnement à un notebook
- Secrets dans un fichier .env (ignoré par Git), chargé avec python-dotenv
1. Crée un projet avec uv, ajoute 3 dépendances, supprime .venv, reconstruis-le avec uv sync.
2. Reproduis le même environnement avec venv + requirements.txt et compare.
3. Clone un dépôt ML ancien et fais-le tourner dans un environnement propre.
Une image contient les fichiers et dépendances de l’application ; un conteneur est un processus isolé lancé depuis cette image. Les conteneurs partagent le noyau du système qui les exécute, directement ou dans une VM. Architecture et pilotes restent des contraintes de portabilité.
~~~bash
docker run hello-world
docker run -it --rm python:3.12-slim bash     # shell interactif, supprimé à la sortie
docker ps                                      # conteneurs actifs ; -a pour tous
docker images
docker build -t mon-app:1.0 .
docker run --rm -p 8000:8000 mon-app:1.0       # expose le port
docker run -v $(pwd)/data:/app/data mon-app    # monte un dossier (volume)
docker run -e API_KEY=xxx mon-app              # variable d'environnement
docker logs -f <id>
docker exec -it <id> bash                      # entrer dans un conteneur actif
# docker system prune : étudier ses effets avant de l’exécuter
~~~
~~~dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copier d'abord les dépendances : profite du cache des couches
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "monapp"]
~~~
Fichier .dockerignore (comme .gitignore) :
~~~
.venv/
.git/
__pycache__/
data/
*.pt
~~~
- Couches et cache : l’ordre des instructions influence la vitesse de build (ce qui change rarement en haut)
- Volumes : les données d’un conteneur disparaissent avec lui, sauf si montées
- Ports : -p hôte:conteneur
- Taille des images : préfère -slim, évite d’embarquer données et poids de modèles inutiles
- Sécurité : ne pas mettre de secrets dans l’image, éviter de tourner en root (USER)
- Docker Compose : plusieurs services ensemble (API + base de données)
~~~yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [db]
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: example
    volumes: [pgdata:/var/lib/postgresql/data]
volumes:
  pgdata:
~~~
- GPU : docker run --gpus all nécessite le NVIDIA Container Toolkit (utile plus tard pour le DL)
1. Conteneurise un script Python qui lit un CSV monté en volume et écrit un résultat.
2. Conteneurise une mini-API FastAPI et appelle-la depuis ton navigateur.
3. Écris un docker-compose.yml avec ton API + une base PostgreSQL.
4. Réduis la taille de ton image de moitié (images slim, .dockerignore, ordre des couches).
## 9. Projet — industrialiser Taskflow

Prends le projet taskflow du Cours 1 et industrialise-le :
1. Dépôt GitHub propre : .gitignore, README, licence, historique de commits lisible
2. Workflow Git : au moins 3 branches de fonctionnalités, 3 PR avec revue (demande à quelqu’un de relire, ou relis-toi), un conflit résolu
3. Environnement : pyproject.toml + lockfile uv, setup en une commande (uv sync)
4. CI GitHub Actions : ruff + mypy + pytest à chaque push, badge dans le README
5. Docker : Dockerfile fonctionnel, docker run taskflow list marche
6. Script Makefile ou bash : make test, make lint, make docker
Critères de réussite : quelqu’un clone ton dépôt sur une machine vierge et obtient un projet qui tourne en moins de 5 minutes, avec CI verte.
## 10. Ressources

- Linux : The Linux Command Line (William Shotts, gratuit en ligne), le cours MIT The Missing Semester of Your CS Education (excellent, à faire)
- Git : Pro Git (gratuit en ligne), le site Learn Git Branching (interactif), Oh Shit, Git!?!
- Environnements : documentation d’uv et du Python Packaging User Guide
- Docker : documentation officielle (Get Started), Play with Docker (bac à sable en ligne)
## 11. Corrections guidées

IP fréquentes : identifier d’abord le format du journal. Si l’IP est le premier champ, extraire ce champ avec awk, trier, compter avec uniq -c, trier numériquement à rebours et sélectionner cinq lignes. Vérifier sur un petit journal annoté ; plusieurs IP par ligne demandent une règle différente.

Archive : citer les chemins, vérifier le retour de tar, lister le contenu et restaurer dans un autre dossier de laboratoire. La présence d’une archive ne démontre pas sa restauration.

tmux : une session peut survivre à une déconnexion du client, mais pas à toutes les pannes de la machine. Vérifier le processus et les sorties.

Git : diff compare le travail à l’index ; diff --staged compare l’index au commit. restore peut supprimer des changements non commitées. Pratiquer dans un dépôt jetable.

Conflit : comprendre les deux intentions, produire la règle correcte, retirer les marqueurs et tester le comportement. L’absence de marqueurs ne garantit pas une résolution logique.

Secret commité : révoquer la clé. .gitignore ne supprime ni l’historique ni les copies.

Environnement : vérifier la reconstruction dans un dossier propre. Une installation réussie sur la machine d’origine ne suffit pas.

Docker : publier un port avec -p ; EXPOSE seul ne le publie pas. Mesurer une optimisation de taille plutôt que promettre un gain arbitraire.

## 12. Compléments avant production

Le Dockerfile est pédagogique. La première synchronisation exclut le projet car les sources ne sont pas encore copiées ; la seconde l’installe. Copier les métadonnées nécessaires au packaging, comme README si requis, figer la version d’uv et l’image de base, puis utiliser un utilisateur sans privilèges. Référence : documentation uv, « Using uv in Docker ».

Ajouter .env et .env.* à .dockerignore. Aucun secret ne doit être inscrit dans l’image. Dans Compose, depends_on ne garantit pas à lui seul que PostgreSQL accepte les connexions : prévoir santé et reprise. Le mot de passe example est réservé à l’illustration.

Vérifier les versions officielles des actions CI et les instructions d’installation des dépendances lors de l’exercice. Sous PowerShell, activer un venv avec .venv\Scripts\Activate.ps1 ; sous cmd avec activate.bat ; sous Bash avec source .venv/bin/activate.

## 13. Soutenance et révisions

Montrer une installation propre, un test échouant avant correction, un conflit résolu et la persistance des données après relancement du conteneur. Fournir commandes, versions et temps réellement mesuré. Les cibles de deux ou cinq minutes restent des objectifs mesurés.

J+1 : expliquer l’index Git. J+7 : résoudre un conflit inédit. J+30 : reconstruire Taskflow depuis le dépôt.

Sources : documentation Docker « What is a container? » et documentation uv « Using uv in Docker ». Aucune commande de ce cours n’est exécutée par le site.
