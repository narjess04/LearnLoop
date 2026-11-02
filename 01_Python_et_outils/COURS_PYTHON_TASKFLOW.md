# Python — des bases au package Taskflow

Support adapté de ton exemple. Durées indicatives ; chapitres avancés à approfondir. Les commandes ne sont pas exécutées dans cette page.

Durée : 4 à 6 semaines · Prérequis : aucun · Objectif : écrire un package Python propre, testé et documenté.
Pour chaque module : explication, exemples de code, exercices. Je commence par les modules 1 à 5 en détail, puis je donne les modules 6 à 9 sous une forme plus condensée.
## 1. Variables, types et opérations

Une variable est un nom qui référence une valeur. Python est typé dynamiquement : le type est attaché à la valeur, pas au nom.
~~~python
age = 25              # int
prix = 19.99          # float
nom = "Amine"         # str
est_actif = True      # bool
rien = None           # NoneType

print(type(age))      # <class 'int'>
~~~
Types de base : int, float, str, bool, None.
 Conversions : int("42"), float("3.14"), str(10), bool(0) donne False.
~~~python
7 + 2    # 9
7 / 2    # 3.5  (division réelle)
7 // 2   # 3    (quotient arrondi vers moins l’infini)
7 % 2    # 1    (modulo)
2 ** 10  # 1024 (puissance)

5 > 3 and 2 > 1   # True
not True          # False
"a" in "banane"   # True
~~~
~~~python
nom, score = "Sara", 0.9234
print(f"Bonjour {nom}, score = {score:.2f}")   # Bonjour Sara, score = 0.92

s = "  Machine Learning  "
s.strip().lower().replace(" ", "_")   # 'machine_learning'
s[2:9]                                # 'Machine'  (slicing)
"a,b,c".split(",")                    # ['a', 'b', 'c']
~~~
1. Écris un programme qui demande le prénom et l’année de naissance, puis affiche l’âge.
2. Convertis une température de Celsius en Fahrenheit avec 1 décimale.
3. Inverse une chaîne sans boucle (indice : slicing [::-1]).
## 2. Conditions et boucles

~~~python
note = 14
if note >= 16:
    mention = "Très bien"
elif note >= 12:
    mention = "Bien"
else:
    mention = "À améliorer"
~~~
Exemples de valeurs falsy usuelles : 0, "", [], {}, set(), None et False. Les objets personnalisés peuvent définir leur valeur de vérité.
~~~python
if liste:               # plus pythonique que `if len(liste) > 0`
    print("non vide")
~~~
~~~python
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

for i, lettre in enumerate("abc"):
    print(i, lettre)            # 0 a / 1 b / 2 c

noms = ["a", "b"]; scores = [10, 20]
for n, s in zip(noms, scores):
    print(n, s)

n = 0
while n < 3:
    n += 1
~~~
break sort de la boucle, continue passe à l’itération suivante.
1. FizzBuzz de 1 à 100.
2. Jeu “devine le nombre” : l’ordinateur choisit un nombre (module random), l’utilisateur devine, le programme répond “plus grand” ou “plus petit”.
3. Affiche tous les nombres premiers inférieurs à 100.
## 3. Fonctions et contrats

~~~python
def moyenne(valeurs: list[float], arrondi: int = 2) -> float:
    """Calcule la moyenne d'une liste de nombres.

    Args:
        valeurs: liste non vide de nombres.
        arrondi: nombre de décimales.
    """
    if not valeurs:
        raise ValueError("Au moins une valeur est requise")
    return round(sum(valeurs) / len(valeurs), arrondi)

moyenne([10, 12, 15])             # 12.33
moyenne([10, 12, 15], arrondi=0)  # 12.0
~~~
Notions clés
- Arguments positionnels, nommés, valeurs par défaut
- *args (tuple d’arguments) et **kwargs (dictionnaire d’arguments nommés)
- Portée des variables (locale, globale), à éviter : global
- Fonctions de première classe : on peut les passer en argument
- lambda : trier = sorted(users, key=lambda u: u["age"])
Piège classique : ne jamais utiliser un objet mutable comme valeur par défaut.
~~~python
def ajoute(x, liste=[]):      # MAUVAIS : la liste est partagée entre les appels
    liste.append(x); return liste

def ajoute(x, liste=None):    # BON
    liste = [] if liste is None else liste
    liste.append(x); return liste
~~~
1. Écris est_palindrome(texte) en ignorant casse et espaces.
2. Écris stats(*nombres) qui renvoie min, max et moyenne.
3. Écris une fonction qui applique une fonction passée en paramètre à chaque élément d’une liste.
## 4. Structures et complexité

~~~python
# Listes
nums = [3, 1, 2]
nums.append(4); nums.sort(); nums[-1]    # dernier élément

# Dictionnaires
user = {"nom": "Sara", "age": 30}
user.get("email", "inconnu")             # accès sûr
for cle, val in user.items(): ...

# Sets
{1, 2, 3} & {2, 3, 4}                    # {2, 3} intersection
{1, 2} | {2, 3}                          # {1, 2, 3} union

# Compréhensions (très utilisées)
carres = [x**2 for x in range(10) if x % 2 == 0]
longueurs = {mot: len(mot) for mot in ["chat", "chien"]}
uniques = {x % 3 for x in range(10)}
~~~
Complexité à retenir : accès par index dans une liste en O(1), recherche x in liste en O(n), x in set ou x in dict en O(1) en moyenne.
Modules utiles : collections (Counter, defaultdict, deque), itertools, heapq.
1. Compte la fréquence des mots d’un texte avec un dict, puis avec Counter.
2. Regroupe des anagrammes (["tea", "eat", "tan", "nat"]).
3. Trouve les doublons d’une liste en O(n).
## 5. Fichiers, exceptions et modules

~~~python
from pathlib import Path
import json, csv

p = Path("data") / "notes.txt"
p.parent.mkdir(exist_ok=True)

with open(p, "w", encoding="utf-8") as f:   # `with` ferme le fichier automatiquement
    f.write("bonjour\n")

texte = p.read_text(encoding="utf-8")

with open("users.json", encoding="utf-8") as f:
    data = json.load(f)

with open("data.csv", newline="", encoding="utf-8") as f:
    for ligne in csv.DictReader(f):
        print(ligne["nom"])
~~~
~~~python
try:
    valeur = int(input("Nombre : "))
    resultat = 100 / valeur
except ValueError:
    print("Ce n'est pas un nombre")
except ZeroDivisionError:
    print("Division par zéro")
else:
    print(resultat)          # exécuté si aucune exception
finally:
    print("Toujours exécuté")

class ErreurValidation(Exception):
    """Exception métier personnalisée."""

raise ErreurValidation("âge négatif")
~~~
Règle : n’attrape jamais except: ou except Exception: à nu sans une bonne raison. Capture l’erreur précise que tu sais gérer.
~~~python
import math
from collections import Counter
import numpy as np

if __name__ == "__main__":     # code exécuté seulement en script
    main()
~~~
Un package est un dossier avec des modules (et souvent un __init__.py).
1. Lis un CSV, calcule la moyenne par colonne et sauvegarde le résultat en JSON.
2. Écris un script qui parcourt un dossier et liste les 10 plus gros fichiers.
3. Écris une fonction lire_config(chemin) qui lève des exceptions explicites si le fichier est absent ou invalide.
## 6. Objets et dataclasses

~~~python
from dataclasses import dataclass, field

class Compte:
    def __init__(self, titulaire: str, solde: float = 0.0):
        self.titulaire = titulaire
        self._solde = solde            # convention : _ = "privé"

    @property
    def solde(self) -> float:
        return self._solde

    def deposer(self, montant: float) -> None:
        if montant <= 0:
            raise ValueError("montant positif requis")
        self._solde += montant

    def __repr__(self) -> str:
        return f"Compte({self.titulaire!r}, {self._solde})"

class CompteEpargne(Compte):           # héritage
    def __init__(self, titulaire, taux):
        super().__init__(titulaire)
        self.taux = taux

@dataclass
class Point:                           # génère __init__, __repr__, __eq__
    x: float
    y: float
    tags: list[str] = field(default_factory=list)
~~~
À maîtriser : attributs de classe vs d’instance, @classmethod, @staticmethod, méthodes magiques (__len__, __getitem__, __eq__, __lt__, __iter__), composition vs héritage (préfère la composition), abc.ABC pour les interfaces.
Exercice : modélise un système de bibliothèque (Livre, Membre, Emprunt) avec des règles métier.
## 7. Itérateurs, décorateurs et concurrence

Itérateurs et générateurs : traitement paresseux : évite de charger tout le fichier, mais la mémoire dépend de la taille des éléments et des objets conservés.
~~~python
def lire_gros_fichier(chemin):
    with open(chemin) as f:
        for ligne in f:
            yield ligne.strip()

somme = sum(x**2 for x in range(10_000_000))    # expression génératrice
~~~
Décorateurs
~~~python
import time, functools

def chronometre(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        res = fn(*args, **kwargs)
        print(f"{fn.__name__}: {time.perf_counter() - t:.3f}s")
        return res
    return wrapper

@chronometre
def calcul(): ...
~~~
Context managers
~~~python
from contextlib import contextmanager

@contextmanager
def minuteur():
    t = time.perf_counter()
    try:
        yield
    finally:
        print(f"{time.perf_counter() - t:.3f}s")
~~~
Typage : list[int], dict[str, float], Optional[str] / str | None, Protocol, TypedDict, vérification avec mypy ou pyright.
Concurrence
- threading : tâches liées aux I/O (limité par le GIL pour le calcul)
- multiprocessing / concurrent.futures : calcul CPU parallèle
- asyncio (async/await) : beaucoup d’I/O réseau (appels d’API de LLM, par exemple)
~~~python
import asyncio

async def appel(i):
    await asyncio.sleep(1)
    return i

async def main():
    return await asyncio.gather(*(appel(i) for i in range(10)))   # appels concurrents ; durée dépendante de l’environnement
~~~
## 8. Tests et packaging

- pytest : tests unitaires, fixtures, parametrize, couverture avec pytest-cov
- logging au lieu de print : niveaux DEBUG/INFO/WARNING/ERROR
- ruff (lint + format), mypy (types), pre-commit
- Documentation : docstrings (style Google ou NumPy), README
- Packaging : pyproject.toml, structure src/, gestion des dépendances avec uv
~~~python
# tests/test_stats.py
import pytest
from monpaquet.stats import moyenne

@pytest.mark.parametrize("entree, attendu", [([1, 2, 3], 2), ([10], 10)])
def test_moyenne(entree, attendu):
    assert moyenne(entree) == attendu

def test_moyenne_vide():
    with pytest.raises(ValueError):
        moyenne([])
~~~
## 9. Profiling et bonnes pratiques

- Lire du code pythonique : The Zen of Python (import this), PEP 8
- Profiler avant d’optimiser : cProfile, timeit
- Vectoriser (NumPy) plutôt que boucler
- Éviter la copie inutile, comprendre mutabilité et références (copy vs deepcopy)
- Gérer les secrets avec des variables d’environnement (jamais dans le code)
## 10. Projet complet — Taskflow

Construis taskflow, un gestionnaire de tâches en ligne de commande :
- Package installable (pyproject.toml, structure src/)
- Classes Tache et GestionnaireTaches (POO + dataclasses)
- Sauvegarde JSON, gestion d’erreurs personnalisées
- CLI avec argparse ou typer (taskflow add "...", taskflow list)
- Au moins 15 tests pytest, couverture supérieure à 80 %
- Logging, typage vérifié par mypy, lint propre avec ruff
- README avec exemples, dépôt GitHub
Critères de réussite : tu peux expliquer chaque ligne de ton code, et un inconnu peut l’installer et l’utiliser en 2 minutes.
## 11. Ressources

- Livre : Automate the Boring Stuff (débutant), Fluent Python (avancé)
- Cours : CS50P (Harvard), documentation officielle Python
- Pratique : Exercism, LeetCode (easy/medium), Advent of Code
## 12. Corrections guidées et cas limites

Âge : soustraire l’année de naissance à l’année courante donne un âge approximatif. L’âge exact exige la date complète et la prise en compte de l’anniversaire.

Température : F=C×9/5+32. Vérifier 0 °C → 32 °F et 100 °C → 212 °F. Arrondir l’affichage plutôt que les calculs intermédiaires.

Inversion : texte[::-1]. Attention : les graphèmes Unicode complexes peuvent contenir plusieurs points de code.

FizzBuzz : tester le multiple de 15 avant les multiples de 3 et de 5, sinon la première condition absorbe le cas commun.

Nombres premiers : n<2 n’est pas premier. Tester les diviseurs de 2 à la racine carrée incluse suffit ; dans tout couple de diviseurs, l’un est inférieur ou égal à cette racine.

Palindrome : préciser le contrat. Ignorer casse et espaces ne signifie pas supprimer aussi les accents et la ponctuation. Comparer "".join(texte.casefold().split()) à son inverse.

stats : pour zéro nombre, lever ValueError. Sinon retourner min, max et sum(nombres)/len(nombres). Tester un seul nombre, des négatifs et des répétitions.

Application d’une fonction : [fonction(x) for x in valeurs]. Une fonction se transmet comme une valeur ; la compréhension construit ensuite une nouvelle liste.

Fréquences : compteur[mot] = compteur.get(mot, 0)+1. Définir la normalisation avant de compter. Counter automatise le comptage mais ne choisit pas la règle métier.

Anagrammes : regrouper selon "".join(sorted(mot)). Le tri coûte O(k log k) pour un mot de longueur k ; préciser la longueur des mots dans l’analyse.

Doublons : conserver un ensemble vus et un ensemble doublons. Le coût est linéaire en moyenne avec les hypothèses usuelles de hachage, et la mémoire dépend des valeurs distinctes.

CSV : convertir explicitement les colonnes numériques, traiter les cellules vides selon un contrat et compter les valeurs valides par colonne. Un dénominateur commun est faux si les colonnes n’ont pas les mêmes absences.

Configuration : distinguer fichier absent, JSON mal formé et schéma invalide. Un JSON valide syntaxiquement peut rester une mauvaise configuration.

## 13. Exemple autonome à vérifier

Le contrat ci-dessous accepte une liste non vide de nombres réels finis ordinaires. Les booléens sont exclus explicitement. Les annotations ne remplacent pas cette validation.

~~~python
from math import isfinite

def moyenne_mesures(valeurs):
    if not valeurs:
        raise ValueError("Au moins une mesure")
    if any(type(v) not in (int, float) or not isfinite(v)
           for v in valeurs):
        raise ValueError("Mesures réelles et finies requises")
    return sum(valeurs) / len(valeurs)

assert moyenne_mesures([10, 12, 14]) == 12
assert moyenne_mesures([-2, 2]) == 0
for invalide in ([], [True], [float("nan")], ["12"]):
    try:
        moyenne_mesures(invalide)
    except ValueError:
        pass
    else:
        raise AssertionError("Entrée invalide acceptée")
~~~

## 14. Soutenance et révisions

Quinze tests et 80 % de couverture sont des repères de pratique ; vérifier aussi la pertinence des assertions. Présenter un cas absent, un fichier corrompu et la persistance de Taskflow. Ajouter une fonctionnalité depuis un brief inédit.

J+1 : expliquer print et return. J+7 : corriger une valeur mutable par défaut. J+30 : ajouter une règle métier sans solution guidée.

Sources à consulter : tutoriel Python « Control Flow » et documentation contextlib. Le comportement des threads dépend notamment de la version et de la configuration de Python. Les commandes et exemples de ce support demandent une vérification dans l’environnement retenu.
