Dans le cours précédent, nous avons entraîné une régression logistique avec :

~~~python
model.fit(X_train, y_train)
~~~

Mais une question essentielle reste sans réponse : que se passe-t-il mathématiquement pendant fit() ?

Ce cours va nous permettre de comprendre le mécanisme d'apprentissage plutôt que d'utiliser les algorithmes comme des boîtes noires.

## 1. L'apprentissage est un problème d'optimisation

De nombreux modèles sont formulés comme une optimisation de paramètres ; ce cours étudie ce cadre pour les modèles différentiables. Tous les algorithmes ML ne suivent pas cette même procédure.

Un modèle possède des paramètres. L'entraînement consiste à trouver des valeurs qui permettent de produire de bonnes prédictions selon un objectif mathématique.

Prenons la régression linéaire simple :
\[
\hat y=wx+b
\]

- \(x\) : caractéristique d'entrée.
- \(w\) : poids (weight).
- \(b\) : biais ou intercept.
- \(\hat y\) : prédiction du modèle.
Supposons que nous cherchions à prédire une valeur à partir d'une seule caractéristique.
[[linefit]]

Lorsque tu modifies \(w\) ou \(b\), tu modifies les prédictions. Il nous faut donc une manière de mesurer quantitativement la qualité de ces prédictions.

C'est le rôle de la fonction de perte.

## 2. Les fonctions de perte (Loss Functions)

Une fonction de perte mesure à quel point les prédictions diffèrent des valeurs réelles.

### 2.1 Mean Squared Error — MSE

En régression, une fonction très utilisée est l'erreur quadratique moyenne :
\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Prenons un exemple :
Observation	Réel \(y\)	Prédit \(\hat y\)	Erreur au carré
1	3	2	1
2	5	4	1
3	7	9	4

Alors :
\[
MSE=\frac{1+1+4}{3}=2
\]

Pourquoi élever les erreurs au carré ? Cela évite que les erreurs positives et négatives s'annulent, tout en pénalisant davantage les grandes erreurs.

Limite : le MSE est sensible aux valeurs aberrantes.

### 2.2 MAE et RMSE

Deux autres fonctions importantes :
\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

Fonction	Avantage	Limite
MSE	Différentiable, pénalise fortement les grandes erreurs	Sensible aux outliers
MAE	Plus robuste aux grandes erreurs	Non différentiable exactement en zéro
RMSE	Même unité que la cible	Sensible aux outliers

### 2.3 Log Loss — Classification binaire

Pour une classification probabiliste binaire, on utilise souvent la binary cross-entropy :
\[
L=-\frac{1}{n}\sum_{i=1}^{n}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
\]

Ici, \(p_i\) représente la probabilité prédite pour la classe 1.

Si \(y=1\), la perte devient :
\[
L=-\log(p)
\]

Si \(y=0\), elle devient :
\[
L=-\log(1-p)
\]

[[crossentropy]]

Remarque importante : une prédiction incorrecte faite avec une très forte confiance entraîne une perte beaucoup plus élevée qu'une prédiction incertaine.

## 3. Comment trouver les meilleurs paramètres ?

Nous savons maintenant mesurer l'erreur.

Mais comment trouver les paramètres \(w\) et \(b\) qui la minimisent ?

Il existe plusieurs méthodes : solutions analytiques, méthodes itératives, optimisation numérique, etc.

Nous allons commencer par la descente de gradient (Gradient Descent).

### 3.1 Comprendre les dérivées

Une dérivée mesure comment une fonction varie lorsqu'on modifie légèrement son entrée.

Prenons :
\[
f(x)=x^2
\]

Sa dérivée est :
\[
f'(x)=2x
\]

À \(x=3\), la pente vaut 6 : augmenter légèrement \(x\) augmente la fonction.

À \(x=-3\), la pente vaut -6 : augmenter légèrement \(x\) diminue la fonction.

À \(x=0\), la dérivée vaut 0 : nous sommes au minimum global de cette fonction.

### 3.2 Descente de gradient

L'idée est de mettre à jour les paramètres dans la direction opposée au gradient :
\[
\boxed{\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)}
\]

Où :
- \(\theta_t\) représente les paramètres à l'itération \(t\).
- \(\eta\) est le learning rate.
- \(\nabla L\) est le gradient de la fonction de perte.
[[gradientstep]]

### 3.3 Le learning rate

Le learning rate contrôle la taille des mises à jour.

Un learning rate trop petit ralentit la convergence. Un learning rate trop grand peut provoquer des oscillations ou une divergence.
[[learningrates]]

Le learning rate optimal dépend de la fonction objectif, de l'échelle des variables et de la méthode d'optimisation.

## 4. Dériver le gradient de la régression linéaire

Passons maintenant à un vrai modèle.

Notre fonction prédictive :
\[
\hat y_i=wx_i+b
\]

Notre fonction de coût :
\[
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2
\]

Nous devons calculer les dérivées partielles par rapport à \(w\) et \(b\).

### 4.1 Dérivée par rapport à w

En utilisant la règle de la chaîne :
\[
\frac{\partial J}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}(wx_i+b-y_i)x_i
\]

### 4.2 Dérivée par rapport à b

\[
\frac{\partial J}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(wx_i+b-y_i)
\]

### 4.3 Mise à jour des paramètres

\[
w \leftarrow w-\eta\frac{\partial J}{\partial w}
\]

\[
b \leftarrow b-\eta\frac{\partial J}{\partial b}
\]

Nous avons maintenant toutes les équations nécessaires pour implémenter un algorithme d'apprentissage.

## 5. Coding Lab — Linear Regression from Scratch

Nous allons coder notre propre régression linéaire, sans utiliser LinearRegression de scikit-learn.

~~~python
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(-2.0, 2.0, 120)
y = 3.0 * x + 2.0 + rng.normal(scale=0.2, size=x.shape)
w, b = 0.0, 0.0
learning_rate = 0.05
losses = []

for step in range(500):
    prediction = w * x + b
    error = prediction - y
    # Les deux gradients utilisent les mêmes paramètres.
    dw = 2.0 * np.mean(error * x)
    db = 2.0 * np.mean(error)
    w -= learning_rate * dw
    b -= learning_rate * db
    # Conserver une perte correspondant aux paramètres après mise à jour.
    losses.append(float(np.mean((w * x + b - y) ** 2)))

print("w :", w, "b :", b)
print("MSE finale :", losses[-1])
assert np.isfinite(losses).all()
assert abs(w - 3.0) < 0.1
assert abs(b - 2.0) < 0.1
~~~

Comprendre le code
Les données contiennent un petit bruit : on attend des coefficients proches, pas une égalité exacte. Une perte nulle n’est pas un objectif réaliste ici.

Le forward pass calcule les prédictions à partir des paramètres actuels.

La fonction de coût mesure l'erreur.

Le calcul des gradients indique comment chaque paramètre influence la perte.

Enfin, les mises à jour modifient \(w\) et \(b\) pour tenter de réduire l'erreur à l'itération suivante.

Dans cet exemple, les valeurs apprises devraient être proches de \(w=3\) et \(b=2\), puisque ce sont les paramètres utilisés pour générer les données.

[[verification]]

## 6. Batch Gradient Descent, SGD et Mini-Batch

Il existe plusieurs façons de calculer les gradients.
Méthode	Données par mise à jour	Caractéristiques
Batch GD	Tout le dataset	Gradient déterministe à paramètres et données fixés, calcul potentiellement coûteux
SGD	Une observation	Mises à jour bruitées, fréquentes
Mini-Batch GD	Un petit groupe d'observations	Compromis entre stabilité et efficacité

En Deep Learning, on utilise très fréquemment le Mini-Batch Gradient Descent avec des optimiseurs comme Adam ou SGD avec momentum.

Il faut aussi distinguer les termes iteration, batch et epoch :
- Dans la boucle simple étudiée ici, une iteration correspond à une mise à jour des paramètres.
- Un batch est le groupe d'observations utilisé pour calculer cette mise à jour.
- Une epoch correspond à un passage sur l'ensemble des données d'entraînement.

## 7. Approfondissement : pourquoi le gradient ne suffit-il pas toujours ?

La descente de gradient utilise des informations locales. Le point obtenu et la convergence dépendent des hypothèses sur la fonction, du pas et du protocole ; convexité seule ne justifie pas n’importe quelle taille de pas.

Pour une fonction convexe différentiable, tout minimum local est global. En revanche, pour une fonction non convexe, l'optimisation peut être plus complexe.

C'est particulièrement important en Deep Learning, où les fonctions de perte sont généralement non convexes.

Autre distinction essentielle : tous les modèles ML ne sont pas entraînés par descente de gradient. Les arbres de décision, par exemple, utilisent des critères de partitionnement et des procédures de construction différentes.

## 8. Projet guidé — descente de gradient multivariée

Challenge 03 — Gradient Descent NumPy
Projet guidé · validation par livrables

Cette fois, notre dataset contient trois features.

Tu dois remplacer le poids scalaire par un vecteur de poids, puis utiliser des opérations matricielles pour calculer le gradient.

La prédiction devient :
\[
\hat y=Xw+b
\]

Avec :
\[
X\in\mathbb{R}^{n\times p},\qquad w\in\mathbb{R}^{p}
\]

Le gradient par rapport au vecteur de poids est :
\[
\nabla_w J=\frac{2}{n}X^T(Xw+b-y)
\]

~~~python
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
true_w = np.array([2.0, -1.5, 4.0])
true_b = 3.0
y = X @ true_w + true_b
y += rng.normal(scale=0.2, size=200)

w = np.zeros(3)
b = 0.0
lr, epochs = 0.1, 200
for epoch in range(epochs):
    # TODO : predictions de forme (200,)
    # TODO : error de forme (200,)
    # TODO : dw = ... avec X.T, de forme (3,)
    # TODO : db = ...
    # TODO : mise a jour simultanee de w et b
    pass
# TODO : comparer les poids aux paramètres générateurs
# TODO : tester les gradients par différences finies
~~~

Critère de réussite : retrouver des poids proches de [2.0, -1.5, 4.0] et un biais proche de 3.0.

## 9. Questions d'entretien technique

[[interview]]

## 10. Validation du cours

[[assessment]]

L'idée essentielle de ce cours : un modèle apprend en ajustant ses paramètres pour optimiser une fonction objectif. Comprendre les pertes, les gradients et les mises à jour des paramètres te donne les fondations nécessaires pour comprendre la régression, les réseaux de neurones et la rétropropagation.

Le cours 4 sera consacré à une autre question fondamentale : pourquoi un modèle peut-il apprendre parfaitement ses données d'entraînement et pourtant échouer sur de nouvelles données ?

## Livrables du projet NumPy

- Une implémentation vectorisée avec dimensions vérifiées.
- Des courbes de perte et un tableau comparant plusieurs learning rates.
- Un contrôle de gradient par différences finies centrales.
- Une comparaison des paramètres appris aux paramètres générateurs.
- Un README et une analyse de l’effet du bruit et de la mise à l’échelle.

Ne pas confondre retrouver des paramètres sur un exemple synthétique et démontrer une performance sur des observations futures.