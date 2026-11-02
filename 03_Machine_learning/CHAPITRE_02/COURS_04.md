Dans le cours précédent, nous avons appris à détecter et traiter les valeurs atypiques.

Aujourd'hui, nous allons comprendre pourquoi deux variables ayant des échelles différentes peuvent perturber certains algorithmes, et comment les transformer correctement.

À la fin de ce cours, tu devras être capable de répondre à trois questions fondamentales :
- Pourquoi faut-il parfois standardiser les données ?
- Quelle différence entre StandardScaler, MinMaxScaler, RobustScaler et Normalizer ?
- Comment choisir la bonne transformation selon le modèle et la distribution des variables ?


## 1. Pourquoi le Feature Scaling est-il important ?

Imaginons un dataset bancaire avec deux variables :

Client	Âge	Revenu annuel
A	25	20 000
B	35	50 000
C	45	80 000
D	55	120 000

L'âge varie entre 25 et 55, tandis que le revenu varie entre 20 000 et 120 000.

Si nous utilisons un algorithme basé sur la distance euclidienne :
\[
d(x_i,x_j)=\sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}
\]

La différence entre les revenus risque de dominer la distance.
Exemple
Entre les clients A et B :
\[
d(A,B)=\sqrt{(35-25)^2+(50000-20000)^2}
\]

\[
d(A,B)\approx30000
\]

La différence d'âge contribue très peu à cette distance.

Ce n'est pas forcément souhaitable : l'algorithme traite les unités numériques comme si elles étaient directement comparables.
[[distance]]




## 2. Quels algorithmes ont besoin de Feature Scaling ?

Tous les algorithmes ne sont pas sensibles de la même manière aux échelles.

Algorithme	Scaling généralement utile ?	Pourquoi ?
KNN	Oui	Basé sur les distances
K-Means	Oui	Basé sur les distances aux centroïdes
SVM	Oui	Distances, marges et noyaux
Régression logistique régularisée	Oui	Régularisation et optimisation
Régression linéaire régularisée	Oui	Régularisation des coefficients
Réseaux de neurones	Oui	Stabilité et vitesse d'entraînement
PCA	Souvent	Sensible aux variances
Decision Tree	Généralement non	Utilise des seuils de séparation
Random Forest	Généralement non	Ensemble d'arbres
Gradient Boosting à base d'arbres	Généralement non	Séparations par seuils

Règle pratique : les méthodes basées sur les distances, les produits scalaires, la régularisation ou l'optimisation numérique bénéficient souvent du scaling.

Les modèles à base d'arbres sont généralement beaucoup moins sensibles aux transformations monotones des variables.



## 3. StandardScaler — Standardisation Z-score

StandardScaler est l'une des transformations les plus utilisées en Machine Learning.

Sa formule est :
\[
z_i=\frac{x_i-\mu}{\sigma}
\]

Avec :
- \(x_i\) : valeur originale.
- \(\mu\) : moyenne calculée sur le train.
- \(\sigma\) : écart-type calculé sur le train.
Après standardisation, une variable d'entraînement non constante possède approximativement :
\[
\mu_z=0,\qquad \sigma_z=1
\]

Exemple
Supposons :
\[
X=[10,20,30,40,50]
\]

La moyenne vaut :
\[
\mu=30
\]

L'écart-type de population vaut :
\[
\sigma=\sqrt{200}\approx14.14
\]

La valeur 50 devient :
\[
z=\frac{50-30}{14.14}\approx1.41
\]

Cela signifie que 50 est située environ 1,41 écart-type au-dessus de la moyenne.
Implémentation

~~~python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~




Quand utiliser StandardScaler ?
Il est particulièrement adapté aux régressions régularisées, SVM, KNN, réseaux de neurones et à la PCA lorsque les variables doivent être comparées sur des échelles similaires.
Limite
La moyenne et l'écart-type sont sensibles aux outliers.

Une valeur extrêmement élevée peut modifier les statistiques utilisées pour transformer toute la colonne.



## 4. MinMaxScaler — Normalisation entre 0 et 1

MinMaxScaler transforme les valeurs selon :
\[
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

Par défaut, les valeurs observées dans le train sont projetées dans l'intervalle \([0,1]\), sauf cas particuliers comme les variables constantes.
Exemple
Supposons :
\[
X=[10,20,30,40,50]
\]

Pour \(x=30\) :
\[
x'=\frac{30-10}{50-10}=0.5
\]

Pour \(x=50\) :
\[
x'=\frac{50-10}{50-10}=1
\]

Implémentation

~~~python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~




Attention aux nouvelles observations
Si une valeur du test dépasse le maximum observé dans le train, sa valeur transformée peut dépasser 1.

Par exemple, avec un train compris entre 10 et 50 :
\[
x_{\mathrm{test}}=70
\]

\[
x'=\frac{70-10}{50-10}=1.5
\]

C'est un comportement normal de MinMaxScaler.
Limite principale
Les valeurs extrêmes influencent directement le minimum et le maximum.



## 5. RobustScaler — Scaling robuste aux outliers

RobustScaler utilise la médiane et l'écart interquartile.

Sa formule habituelle est :
\[
x'=\frac{x-\operatorname{Median}(X)}{Q_3-Q_1}
\]

Contrairement à StandardScaler, il ne repose pas sur la moyenne et l'écart-type.
Exemple
Supposons :
\[
X=[10,12,13,15,16,18,100]
\]

La valeur 100 est très éloignée des autres.

La médiane vaut 15.

Avec la définition usuelle des quartiles utilisée ici :
\[
Q_1=12.5,\qquad Q_3=17
\]

\[
IQR=4.5
\]

Pour \(x=18\) :
\[
x'=\frac{18-15}{4.5}\approx0.67
\]

La transformation reste centrée sur les observations habituelles.
Implémentation

~~~python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~




Attention : RobustScaler ne supprime pas les outliers et ne les ramène pas nécessairement dans un intervalle limité. Il réduit simplement leur influence sur les statistiques de centrage et de mise à l'échelle.



## 6. Comparaison interactive des trois scalers


[[scalers]]




## 7. Normalizer — Normalisation par observation

C'est une différence importante à connaître en entretien.

StandardScaler, MinMaxScaler et RobustScaler travaillent généralement colonne par colonne.

Normalizer, lui, transforme chaque ligne afin de lui donner une norme unitaire.

Pour la norme L2 :
\[
x_i'=\frac{x_i}{\|x_i\|_2}
\]

Avec :
\[
\|x_i\|_2=\sqrt{\sum_{j=1}^{p}x_{ij}^2}
\]

Exemple
Considérons :
\[
x=[3,4]
\]

Sa norme vaut :
\[
\|x\|_2=\sqrt{3^2+4^2}=5
\]

Après normalisation :
\[
x'=[0.6,0.8]
\]

La nouvelle norme vaut 1.
Implémentation

~~~python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm="l2")
X_normalized = normalizer.fit_transform(X)
~~~




Contrairement à StandardScaler, Normalizer n'apprend pas une moyenne ou un écart-type à partir du train.
Quand l'utiliser ?
Il est utile lorsque la direction d'un vecteur est plus importante que sa magnitude.

Par exemple :
- Représentations vectorielles de documents.
- Données textuelles TF-IDF.
- Similarité cosinus.
- Certains systèmes de recommandation.
Il n'est pas interchangeable avec StandardScaler.



## 8. Scaling versus transformation de distribution

Jusqu'à présent, nous avons principalement modifié l'échelle des variables.

Mais certaines distributions présentent une forte asymétrie.

Dans ce cas, nous pouvons envisager une transformation non linéaire.

### 8.1 Transformation logarithmique

Pour une variable positive ou nulle :
\[
x'=\log(1+x)
\]

Cette transformation réduit l'écart numérique entre les grandes valeurs.
[[log]]

Exemple Python

~~~python
import numpy as np
df["income_log"] = np.log1p(df["income"])
~~~




np.log1p() calcule \(\log(1+x)\) avec une bonne précision numérique, notamment pour les petites valeurs.

Cette transformation exige \(x>-1\), et elle est couramment utilisée sur des données non négatives.



## 9. PowerTransformer — Yeo-Johnson et Box-Cox

PowerTransformer applique une transformation de puissance afin de réduire certaines asymétries et de stabiliser la variance.

Il propose notamment deux méthodes.
Box-Cox
La transformation Box-Cox est définie pour les valeurs strictement positives :
\[
T(x;\lambda)=
\begin{cases}
\frac{x^\lambda-1}{\lambda} & \lambda\neq0\\
\log(x) & \lambda=0
\end{cases}
\]

Yeo-Johnson
Yeo-Johnson est une généralisation qui accepte également les valeurs nulles et négatives.
Implémentation

~~~python
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method="yeo-johnson", standardize=True)
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)
~~~




Par défaut, standardize=True applique également une standardisation après la transformation de puissance.
Quand l'utiliser ?
Lorsque certaines variables sont fortement asymétriques et que le modèle peut bénéficier d'une distribution plus régulière.

Mais cette transformation ne garantit pas une distribution normale ni une amélioration de la performance.



## 10. QuantileTransformer — Transformer selon les rangs

QuantileTransformer utilise la distribution empirique des observations d'entraînement.

Il peut transformer les valeurs pour obtenir une distribution cible approximativement uniforme ou normale.
Exemple

~~~python
from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100, random_state=42)
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)
~~~




Principe
Au lieu d'utiliser directement la valeur numérique, la transformation repose sur sa position dans la distribution.

Les valeurs les plus faibles sont associées à des quantiles faibles, et les plus grandes à des quantiles élevés.
Avantages
- Peut réduire l'effet de valeurs extrêmes.
- Peut transformer certaines distributions très asymétriques.
- Peut aider certains modèles sensibles à la distribution des variables.
Limites
- Modifie les distances entre observations.
- Peut écraser certaines différences importantes.
- Les valeurs situées hors de la plage d'entraînement sont ramenées aux bornes de la transformation apprise.



## 11. Tableau récapitulatif — Quelle méthode choisir ?


Méthode	Transformation	Utilisation fréquente
StandardScaler	Moyenne 0, écart-type 1	SVM, régressions régularisées, PCA
MinMaxScaler	Intervalle défini, souvent [0,1]	Données nécessitant une échelle bornée sur le train
RobustScaler	Médiane et IQR	Variables avec outliers
Normalizer	Norme unitaire par ligne	Texte, vecteurs, similarité cosinus
log1p	Transformation logarithmique	Variables positives asymétriques
PowerTransformer	Transformation de puissance	Réduction de certaines asymétries
QuantileTransformer	Transformation par rangs	Distributions fortement asymétriques

Il faut éviter une règle du type « StandardScaler est toujours le meilleur ».

Le bon choix dépend du modèle, de la distribution et du protocole de validation.



## 12. Feature Scaling et Data Leakage

Comme pour l'imputation et les outliers, les paramètres du scaling doivent être appris uniquement sur le train.
Mauvaise approche

~~~python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Le scaler a déjà vu les statistiques du futur test.
~~~

Le scaler a déjà utilisé les informations statistiques du futur test.

Bonne approche

~~~python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~

Encore mieux : Pipeline

~~~python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipeline.fit(X_train, y_train)
~~~




Avec une cross-validation, le scaler sera réajusté séparément sur chaque fold d'entraînement.



## 13. Coding Lab — Comparer les scalers sur un modèle

Nous allons utiliser le dataset Wine de scikit-learn.

Il contient des variables numériques ayant des échelles différentes.

Nous comparerons la performance d'une régression logistique avec plusieurs méthodes de scaling.

~~~python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scalers = {"No Scaling": "passthrough", "StandardScaler": StandardScaler(),
           "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, scaler in scalers.items():
        model = Pipeline([("scaler", scaler),
                          ("model", LogisticRegression(max_iter=10000))])
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        assert scores.shape == (5,) and np.isfinite(scores).all()
        print(name, scores, "mean:", scores.mean(), "std:", scores.std())
    print("Le test final reste réservé pendant cette comparaison.")
~~~




Questions d'analyse
Après exécution, réponds aux questions suivantes :
1. Quel scaler obtient la meilleure accuracy moyenne ?
2. Quelle différence observes-tu avec l'absence de scaling ?
3. Les écarts sont-ils suffisamment importants pour conclure ?
4. Pourquoi les résultats peuvent-ils changer si nous utilisons KNN au lieu de Logistic Regression ?
5. Pourquoi faut-il conserver le test final hors de cette comparaison ?
Les scores dépendent des données et des hyperparamètres. Un scaler n'est pas universellement supérieur aux autres.



## 14. Coding Challenge — StandardScaler from Scratch

Challenge 04 — Build Your Own Scaler
+250 XP

Implémente toi-même un scaler compatible avec l'interface de scikit-learn, sans utiliser StandardScaler à l'intérieur.


~~~python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MyStandardScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # TODO : moyenne et écart-type par colonne (ddof=0).
        # TODO : remplacer l'échelle des colonnes constantes par 1.
        return self

    def transform(self, X):
        # TODO : vérifier le fit, les dimensions et standardiser.
        raise NotImplementedError

    def inverse_transform(self, X):
        # BONUS : retrouver les valeurs originales.
        raise NotImplementedError

# TODO : comparer à StandardScaler avec np.allclose, puis utiliser un Pipeline.
~~~




Bonus
Ajoute une méthode inverse_transform() permettant de retrouver les valeurs originales.

La formule inverse est :
\[
x=z\sigma+\mu
\]



[[solution]]




## 15. Questions d'entretien — ML Engineer / Data Scientist


Réponds oralement avant de révéler la correction.

[[interview]]




## 16. Mini-quiz interactif


[[quiz]]




## 17. Validation du cours


[[assessment]]

À retenir : le Feature Scaling ne sert pas à améliorer automatiquement toutes les performances. Il permet d'adapter la représentation numérique des données aux hypothèses et au fonctionnement des algorithmes.

Le prochain cours sera consacré à l'Encoding des variables catégorielles, avec One-Hot, Ordinal, Label, Target et Frequency Encoding.