# Leçon pilote — dériver, programmer et vérifier une descente de gradient

Statut : support pédagogique rédigé ; deux blocs Python exécutés avec succès le 1 novembre 2026, assertions de convergence et différences finies validées. Relecture pédagogique indépendante encore à effectuer.
Prérequis : fonction Python, boucle, carré d'un nombre, lecture d'un repère, dérivée d'une fonction simple.
Objectif : calculer une mise à jour, justifier son signe, déterminer un pas convergent dans un cas quadratique et vérifier le gradient numériquement.
Durée indicative : deux séances de 45 à 60 minutes, puis une révision.

## Diagnostic

Sans exécuter de code : combien vaut f(0) pour f(w)=(w−3)² ? Où cette fonction est-elle minimale ? Si sa dérivée est négative, faut-il augmenter ou diminuer w pour diminuer localement f ?

Attendus : 9 ; w=3 ; augmenter w. Une erreur sur la dernière question appelle une étude du signe avant la boucle Python.

## Le problème

On cherche une valeur w qui minimise une perte. Pour cet exemple seulement, on connaît la solution exacte ; elle servira à vérifier notre méthode. En apprentissage, les pertes sont souvent multivariées et leur minimisation n'a pas une formule aussi simple.

Notations : w est un scalaire réel ; L(w)=(w−3)² est la perte ; g(w)=dL/dw est sa dérivée ; η est un pas strictement positif ; t est l'indice de l'itération.

## Intuition et dérivation

La dérivée mesure la variation locale de la perte lorsque w varie légèrement. Un signe positif signifie qu'une augmentation de w augmente localement L ; pour diminuer L, on diminue w.

Développons : L(w)=w²−6w+9. Dériver terme par terme donne g(w)=2w−6=2(w−3).

La règle est w_(t+1)=w_t−η g(w_t). Le signe moins choisit la direction opposée à la dérivée. Il ne garantit pas qu'un déplacement de taille quelconque améliore la perte : l'approximation est locale.

Pour w₀=0 et η=0,1 : g(0)=−6 ; w₁=0−0,1×(−6)=0,6 ; L(0,6)=5,76. La perte passe bien de 9 à 5,76.

Pour la deuxième étape : g(0,6)=−4,8 ; w₂=1,08 ; L(1,08)=3,6864.

## Pourquoi le pas change tout

Posons e_t=w_t−3. Alors :

w_(t+1)−3 = w_t−3−2η(w_t−3)
e_(t+1) = (1−2η)e_t
e_t = (1−2η)^t e_0

Pour converger vers zéro depuis toute valeur initiale, il faut |1−2η|<1.
Résolution : −1<1−2η<1, donc 0<η<1.

- η=0 : aucun mouvement.
- 0<η<0,5 : convergence sans changement de signe de l'erreur.
- η=0,5 : arrivée en une étape, dans cette quadratique précise.
- 0,5<η<1 : alternance autour de la solution avec amplitude décroissante.
- η=1 : alternance à amplitude constante, sauf départ exactement à l'optimum.
- η>1 : divergence pour toute erreur initiale non nulle.

La dérivée seconde vaut 2. Si la perte est multipliée par 4, sa courbure est multipliée par 4 et le pas admissible change. La borne obtenue ici ne doit pas être appliquée directement à un réseau.

## Implémentation avec invariants

Exécuter ce fichier dans Python. Sortie observée lors de la vérification : w=2.9999999993888893 et L=3.7345630914781746e-19 après 100 étapes. Les assertions des deux blocs passent.

~~~python
from math import isclose

def loss(w: float) -> float:
    return (w - 3.0) ** 2

def grad(w: float) -> float:
    return 2.0 * (w - 3.0)

def optimize(w: float, eta: float, steps: int):
    history = []
    for step in range(steps):
        before = w
        gradient = grad(before)
        w = before - eta * gradient
        history.append({
            "step": step,
            "before": before,
            "gradient": gradient,
            "after": w,
            "loss_after": loss(w),
        })
    return w, history

w, history = optimize(0.0, 0.1, 100)
assert isclose(history[0]["after"], 0.6, abs_tol=1e-12)
assert isclose(history[0]["loss_after"], 5.76, abs_tol=1e-12)
assert abs(w - 3.0) < 1e-8
assert all(
    history[i+1]["loss_after"] <= history[i]["loss_after"] + 1e-12
    for i in range(len(history)-1)
)
print(w, loss(w))
~~~

Chaque ligne d'historique associe la perte au paramètre après mise à jour. Mélanger le paramètre après l'étape avec une perte avant l'étape rend les journaux trompeurs.

## Vérifier le gradient sans lui faire confiance

La différence finie centrale est [L(w+h)−L(w−h)]/(2h). Elle approche la dérivée pour une fonction assez régulière et un h adapté.

~~~python
def finite_difference(w: float, h: float = 1e-5) -> float:
    return (loss(w + h) - loss(w - h)) / (2.0 * h)

for point in [-2.0, 0.0, 2.5, 5.0]:
    assert isclose(grad(point), finite_difference(point),
                   rel_tol=1e-7, abs_tol=1e-7)
~~~

Un h trop petit amplifie les erreurs d'arrondi par soustraction de nombres proches. Tester plusieurs ordres de grandeur plutôt que d'affirmer que plus petit est toujours meilleur.

## Exercices et corrections

### 1 — Calcul manuel

Depuis w=5 avec η=0,2, calculer le gradient, le nouveau w et la nouvelle perte.

Correction : gradient=4 ; w'=4,2 ; perte=1,44 contre 4 au départ. Le paramètre diminue parce que le gradient est positif.

### 2 — Prédiction avant exécution

Comparer η=0,01 ; 0,8 ; 1 ; 1,1 depuis w=0. Prévoir le sens et l'amplitude de l'erreur avant de tracer la courbe.

Correction : facteurs d'erreur 0,98 ; −0,6 ; −1 ; −1,2. Respectivement convergence lente sans alternance ; convergence avec alternance ; oscillation sans convergence ; divergence. Tracer aussi la perte : le signe de l'erreur ne s'y voit plus.

### 3 — Transfert mathématique

Pour J(w)=(2w−6)², dériver et retrouver la plage de pas convergents.

Correction : J=4(w−3)² ; J'=8(w−3) ; e_(t+1)=(1−8η)e_t. La condition devient 0<η<0,25. L'optimum est identique, le pas admissible est différent.

### 4 — Débogage

Une boucle écrit w += eta * grad(w). Depuis w=0, expliquer pourquoi elle échoue avant toute exécution.

Correction : elle suit la montée. e_(t+1)=(1+2η)e_t ; pour η>0 et e₀≠0 l'amplitude augmente. Corriger le signe puis vérifier une première étape à la main.

### 5 — Généralisation

Pour L(w)=(aw−b)² avec a≠0, trouver l'optimum et la borne.

Correction : g=2a(aw−b), optimum b/a. En posant e=w−b/a, e'=(1−2ηa²)e. Il faut 0<η<1/a². Pour a=0, la perte est constante : tout w minimise et le gradient est nul.

## Quiz avec explication des distracteurs

1. Gradient=−4 et η=0,1 : déplacement −0,4 ; +0,4 ; +4 ?
   Réponse +0,4. −0,4 oublie le signe moins ; +4 oublie l'échelle du pas.
2. Une perte qui baisse à chaque itération prouve-t-elle la généralisation ?
   Non. Ici on étudie l'optimisation d'une fonction. La généralisation demande un protocole sur des données indépendantes.
3. η=1 est-il convergent pour L=(w−3)² depuis w=0 ?
   Non : w alterne entre 0 et 6. La perte reste 9.
4. Multiplier la perte par 4 change-t-il son minimum et son gradient ?
   Le minimum reste au même endroit ; le gradient est multiplié par 4. Il faut adapter le pas.

## Questions-réponses

**Pourquoi ne pas utiliser directement w=3 ?** Ici, cette solution connue fournit un contrôle. La méthode prépare les problèmes dont la solution ne s'obtient pas aussi directement.

**Gradient nul signifie-t-il minimum ?** Pas en général. Il peut aussi s'agir d'un maximum ou d'un autre point stationnaire. Dans cette quadratique strictement convexe, l'unique point stationnaire est le minimum.

**Une oscillation indique-t-elle une divergence ?** Non. Examiner l'amplitude : un facteur négatif de module inférieur à 1 produit une alternance convergente.

**Et avec plusieurs paramètres ?** Le gradient rassemble les dérivées partielles. La courbure dépend alors des directions. Il faudra étudier la géométrie et les valeurs propres pour aller plus loin.

## Projet associé

Créer un petit laboratoire comparant trois pertes quadratiques et plusieurs pas. Livrer les formules, courbes, tests de différences finies, journaux cohérents et une explication de chaque comportement. Puis remplacer la perte par la MSE d'une régression à une variable.

Critères : calculs corrects ; prédictions formulées avant les essais ; code reproductible ; au moins un bug diagnostiqué ; explication de la différence entre optimisation et généralisation.

## Vidéo à produire ensemble

- Ouverture : choisir un paramètre pour réduire une erreur.
- Visualisation : parabole, tangente et déplacement à partir de w=0.
- Construction : dérivée et première mise à jour à la main.
- Expérience : trois tailles de pas, prédiction avant animation.
- Pause active : calcul depuis w=5.
- Correction : gradient, signe et nouvelle perte.
- Transfert : multiplier la perte et expliquer le changement de pas.

## Révisions

J+1 : calculer une étape sans support.
J+7 : retrouver la borne sur une quadratique modifiée.
J+30 : vérifier un gradient erroné et l'expliquer à une autre personne.

Référence à vérifier lors de la revue éditoriale : Deep Learning, chapitre Numerical Computation / Optimization for Training Deep Models. Les deux blocs de code ont été exécutés avec succès ; une relecture indépendante reste à organiser.
