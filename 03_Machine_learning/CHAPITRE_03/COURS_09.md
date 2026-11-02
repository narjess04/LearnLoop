# Support Vector Machines (SVM)

Dans les cours précédents, nous avons étudié les arbres de décision, Random Forest et Gradient Boosting.

Aujourd'hui, nous changeons complètement d'approche.

Les Support Vector Machines (SVM) cherchent une frontière de décision qui sépare les classes en maximisant la marge entre elles.

L'idée fondamentale est :
Une bonne frontière de classification ne doit pas seulement séparer les classes. Elle doit aussi être aussi éloignée que possible des observations critiques qui définissent cette séparation.

C'est ce que nous appelons le principe de la marge maximale.

### Objectifs du cours

À la fin, tu sauras expliquer les notions suivantes :

- Hyperplan et frontière de décision.
- Marge maximale et Support Vectors.
- Hard Margin et Soft Margin.
- Hinge Loss et régularisation.
- Le rôle du paramètre C.
- Kernel Trick et noyaux linéaire, polynomial et RBF.
- Le rôle de gamma.
- SVC, LinearSVC et SVR.
- Feature Scaling et pipelines.
- Hyperparameter Tuning avec GridSearchCV.
- Avantages, limites et cas d'utilisation des SVM.

## Partie I — Comprendre les SVM

## 1. Qu'est-ce qu'un SVM ?

Un Support Vector Machine est un algorithme supervisé utilisé principalement pour :

- La classification binaire ou multiclasse.
- La régression, avec SVR.
- Certaines méthodes de détection d'anomalies, comme One-Class SVM.

Prenons un exemple de classification binaire.

Nous avons deux classes :

Classe 0

Classe 1

Classification linéaire en deux dimensions
Illustration conceptuelle : la ligne verte centrale représente une frontière de décision, et les lignes pointillées les marges.
Dans cet exemple, plusieurs droites pourraient séparer les deux classes.

Mais laquelle choisir ?

Le SVM cherche celle qui maximise la distance minimale aux observations des deux classes.

## 2. Qu'est-ce qu'un hyperplan ?

En deux dimensions, un hyperplan est une droite.

En trois dimensions, c'est un plan.

Dans un espace de dimension \(d\), c'est un hyperplan de dimension \(d-1\).

Son équation est :\[
\boxed{w^\top x+b=0}
\]

Où :

- \(x\) : vecteur des features.
- \(w\) : vecteur des coefficients.
- \(b\) : biais.

Le SVM calcule une fonction de décision :\[
f(x)=w^\top x+b
\]

En classification binaire, nous prédisons généralement :\[
\hat y=
\begin{cases}
+1 & \text{si } f(x)>0\\
-1 & \text{si } f(x)<0
\end{cases}
\]

La convention exacte pour les points sur la frontière dépend de l'implémentation.

## Partie II — La marge maximale

## 3. Qu'est-ce que la marge ?

La marge est la distance entre la frontière de décision et les observations les plus proches, dans le cas linéaire séparable.

Les deux hyperplans de marge sont :\[
w^\top x+b=1
\]

et :\[
w^\top x+b=-1
\]

La distance entre ces deux hyperplans est :\[
\boxed{\text{Largeur de marge}=\frac{2}{\|w\|}}
\]

Maximiser la marge revient donc à minimiser la norme de \(w\).

C'est pourquoi le SVM classique résout :\[
\boxed{
\min_{w,b}\frac12\|w\|^2
}
\]

sous les contraintes :\[
y_i(w^\top x_i+b)\ge1
\]

pour toutes les observations.

Ici, les labels sont codés en \(-1\) et \(+1\).

## 4. Pourquoi une grande marge est-elle intéressante ?

Une frontière très proche des observations peut être sensible à de petites variations des données.

Une marge plus grande peut améliorer la robustesse et la généralisation, sous des hypothèses appropriées.

Mais cela ne signifie pas qu'une marge maximale garantit toujours la meilleure performance sur des données futures.

## 5. Explorateur interactif — Marge et frontière

[[svmmargin]]

## 6. Que sont les Support Vectors ?

Les Support Vectors sont les observations qui déterminent la solution SVM.

Dans le cas Hard Margin, ce sont les observations situées sur les hyperplans de marge.

Elles satisfont :\[
y_i(w^\top x_i+b)=1
\]

Les observations plus éloignées de la frontière n'influencent pas directement l'hyperplan optimal dans la formulation duale.

C'est une caractéristique fondamentale des SVM : la solution dépend principalement d'un sous-ensemble d'observations.
Pourquoi est-ce important ?
Imaginons un dataset de 10 000 observations.

Le SVM peut déterminer sa frontière à partir d'un ensemble beaucoup plus petit de Support Vectors.

Cependant, avec un Soft Margin ou un noyau non linéaire, le nombre de Support Vectors peut devenir important.

## 7. Visualiser les Support Vectors avec Python

~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
X_demo, y_demo = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=42)
model = SVC(kernel="linear", C=1).fit(X_demo, y_demo)
fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(model, X_demo, response_method="decision_function",
    plot_method="contour", levels=[-1, 0, 1], ax=ax)
ax.scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap="coolwarm", s=40)
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
    s=160, facecolors="none", edgecolors="black")
plt.show()
~~~

Les cercles noirs entourent les Support Vectors.

## Partie IV — Hard Margin versus Soft Margin

## 8. Hard Margin SVM

Le Hard Margin suppose que les classes sont parfaitement séparables par un hyperplan.

Il impose :\[
y_i(w^\top x_i+b)\ge1
\]

pour toutes les observations.

Mais les données réelles sont rarement parfaitement séparables.

Nous pouvons avoir :

- Du bruit.
- Des observations atypiques.
- Des classes qui se chevauchent.
- Des erreurs de mesure.

Dans ces situations, le Hard Margin peut être impossible à satisfaire.

## 9. Soft Margin SVM

Le Soft Margin introduit des variables d'écart :\[
\xi_i\ge0
\]

Nous résolvons :\[
\boxed{
\min_{w,b,\xi}
\frac12\|w\|^2+C\sum_{i=1}^{n}\xi_i
}
\]

sous les contraintes :\[
y_i(w^\top x_i+b)\ge1-\xi_i
\]

et :\[
\xi_i\ge0
\]

Les variables \(\xi_i\) autorisent des violations de marge, voire des erreurs de classification.

Le paramètre \(C\) contrôle la pénalisation de ces violations.

## 10. Le rôle du paramètre C

C'est l'un des hyperparamètres les plus importants des SVM.
Petit C

- Régularisation plus forte.
- Violations de marge davantage tolérées.
- Frontière souvent plus simple.
- Risque de sous-apprentissage si C est trop petit.

Grand C

- Violations davantage pénalisées.
- Modèle cherchant à mieux satisfaire les observations d'entraînement.
- Frontière potentiellement plus complexe.
- Risque de surapprentissage.

Influence conceptuelle de C
C faible

Tolérance accrue aux violations de marge.

C élevé

Pénalisation plus forte des violations.

Schémas illustratifs : la forme exacte des marges dépend des données et de l'optimisation.

## Partie V — Hinge Loss

## 11. Comprendre la fonction de perte

Le Soft Margin SVM peut s'exprimer à l'aide de la Hinge Loss.

Pour une observation :\[
\boxed{
L(y,f(x))=\max(0,1-yf(x))
}
\]

avec :\[
y\in\{-1,+1\}
\]

Cas 1 — Bonne classification avec marge suffisante
Si :\[
yf(x)\ge1
\]

Alors :\[
L=0
\]

Cas 2 — Bonne classification mais marge insuffisante
Si :\[
0<yf(x)<1
\]

Alors :\[
L>0
\]

Cas 3 — Mauvaise classification
Si :\[
yf(x)<0
\]

Alors :\[
L>1
\]

## 12. Explorateur interactif — Hinge Loss

[[svmhinge]]

## 13. Et si les données ne sont pas linéairement séparables ?

Jusqu'ici, nous avons supposé qu'une droite pouvait séparer les deux classes.

Mais imaginons que les données forment deux cercles concentriques.
Un problème non linéaire

Dans ce cas, un SVM linéaire n'est pas adapté.

Nous avons besoin d'une frontière non linéaire.

## 14. Transformer l'espace des features

L'idée consiste à transformer les données :\[
x\longrightarrow\phi(x)
\]

Dans cet espace transformé, les classes peuvent devenir linéairement séparables.

Par exemple :\[
\phi(x_1,x_2)=(x_1,x_2,x_1^2+x_2^2)
\]

Des points formant des cercles dans l'espace original peuvent être séparés plus facilement dans un espace de dimension supérieure.

Mais calculer explicitement toutes les transformations peut être coûteux.

C'est là qu'intervient le Kernel Trick.

## 15. Le Kernel Trick

Un noyau calcule directement un produit scalaire dans un espace de features transformé :\[
\boxed{
K(x,z)=\langle\phi(x),\phi(z)\rangle
}
\]

Nous n'avons pas nécessairement besoin de calculer explicitement \(\phi(x)\).

C'est une idée extrêmement puissante.

Le modèle peut alors produire des frontières non linéaires dans l'espace original.

## Partie VII — Les noyaux SVM

## 16. Linear Kernel

Le noyau linéaire est :\[
\boxed{K(x,z)=x^\top z}
\]

Il convient lorsque les classes sont raisonnablement séparables par une frontière linéaire.

~~~python
from sklearn.svm import SVC
model = SVC(kernel="linear")
~~~

Il est souvent intéressant lorsque le nombre de features est élevé.

## 17. Polynomial Kernel

Le noyau polynomial est :\[
\boxed{
K(x,z)=(\gamma x^\top z+r)^d
}
\]

Où :

- \(d\) : degré du polynôme.
- \(\gamma\) : facteur d'échelle.
- \(r\) : coefficient indépendant.

~~~python
from sklearn.svm import SVC
model = SVC(kernel="poly", degree=3, gamma="scale", coef0=1)
~~~

Il permet de représenter des interactions polynomiales.

## 18. RBF Kernel

Le noyau RBF (Radial Basis Function) est :\[
\boxed{
K(x,z)=\exp(-\gamma\|x-z\|^2)
}
\]

C'est l'un des noyaux les plus utilisés.

Il permet de modéliser des frontières de décision non linéaires complexes.

~~~python
from sklearn.svm import SVC
model = SVC(kernel="rbf", gamma="scale")
~~~

## 19. Le rôle de gamma

Le paramètre \(\gamma\) contrôle la portée de l'influence des observations.
Gamma faible

- Influence plus étendue.
- Frontière généralement plus lisse.
- Risque de sous-apprentissage si trop faible.

Gamma élevé

- Influence plus localisée.
- Frontière potentiellement plus complexe.
- Risque de surapprentissage si trop élevé.

## 20. Explorateur interactif — RBF Kernel

[[svmrbf]]

## 21. Pourquoi le Scaling est-il indispensable ?

Les SVM dépendent fortement des distances et des produits scalaires.

Supposons deux variables :

- Revenu annuel : entre 20 000 et 200 000.
- Ratio d'endettement : entre 0 et 1.

Sans Scaling, le revenu peut dominer la géométrie du modèle.

Nous devons donc généralement standardiser les features.

La standardisation est :\[
\boxed{z=\frac{x-\mu}{\sigma}}
\]

## 22. Pipeline SVM

~~~python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1, gamma="scale"))])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
~~~

Le Pipeline évite le Data Leakage en ajustant le scaler uniquement sur les données d'entraînement de chaque fold.

C'est particulièrement important avec les SVM.

## Partie IX — SVC, LinearSVC et SVR

## 23. SVC

SVC est l'implémentation classique des SVM de classification utilisant libsvm.

Il permet différents noyaux :

~~~python
from sklearn.svm import SVC
models = [SVC(kernel=k) for k in ["linear", "poly", "rbf", "sigmoid"]]
~~~

Il est très utile pour les datasets de taille petite à moyenne.

Mais son coût d'entraînement peut devenir élevé sur de grands datasets.

## 24. LinearSVC

LinearSVC est conçu pour la classification linéaire.

~~~python
from sklearn.svm import LinearSVC
model = LinearSVC(C=1, max_iter=10000)
~~~

Il peut être plus adapté à des datasets comportant beaucoup d'observations ou de features.

Attention : LinearSVC et SVC(kernel="linear") ne résolvent pas nécessairement exactement le même problème, car leurs fonctions de perte et paramètres d'optimisation peuvent différer.

## 25. SVR — Support Vector Regression

Les SVM peuvent aussi être utilisés en régression.

C'est le principe de Support Vector Regression (SVR).

L'objectif est de construire une fonction de prédiction avec une tolérance \(\varepsilon\) autour des valeurs réelles.

La perte epsilon-insensitive est :\[
\boxed{
L_\varepsilon(y,\hat y)=
\max(0,|y-\hat y|-\varepsilon)
}
\]

Si l'erreur absolue est inférieure à \(\varepsilon\), la perte est nulle.

## 26. Explorateur interactif — Epsilon Loss

[[svmepsilon]]

## 27. Exemple SVR

~~~python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
X_reg, y_reg = make_regression(n_samples=1000, n_features=8, noise=20, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
model_reg = Pipeline([("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10, epsilon=.1, gamma="scale"))])
model_reg.fit(Xr_train, yr_train)
yr_pred = model_reg.predict(Xr_test)
print("MAE", mean_absolute_error(yr_test, yr_pred), "R²", r2_score(yr_test, yr_pred))
~~~

Dans un projet réel, il peut également être utile de standardiser la cible de régression avec TransformedTargetRegressor, notamment lorsque son échelle est très grande.

## Partie X — Classification complète avec Python

## 28. Construire un modèle SVM de A à Z

Le laboratoire suivant crée 2 000 observations et 15 variables, puis compare logistique, SVM linéaire, RBF et forêt par CV. decision_function produit des scores utilisables pour ROC-AUC et AP ; aucun Brier n’est calculé sur ces scores. Activer probability=True ajoute une estimation de probabilités qu’il faut aussi évaluer. Le laboratoire présente également SVR avec une cible standardisée.

~~~python
"""Cours 09. Classification SVM et régression SVR ; --plot optionnel."""

import argparse
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)

def dataset(n=5000, features=20, informative=10, redundant=5, positive=.08):
    X, y = make_classification(n_samples=n, n_features=features,
        n_informative=informative, n_redundant=redundant, weights=[1-positive, positive],
        flip_y=.01, random_state=42)
    return train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def benchmark(models, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
            scoring={"ROC_AUC": "roc_auc", "AP": "average_precision"})
        rows.append({"Model": name, "ROC_AUC_CV": scores["test_ROC_AUC"].mean(),
            "ROC_AUC_std": scores["test_ROC_AUC"].std(), "AP_CV": scores["test_AP"].mean(),
            "fit_seconds_CV_mean": scores["fit_time"].mean()})
    table = pd.DataFrame(rows).sort_values("ROC_AUC_CV", ascending=False)
    print(table.to_string(index=False))
    chosen = table.iloc[0]["Model"]
    print("Choix figé sur la ROC-AUC CV avant de consulter le test :", chosen)
    return chosen, table

def metrics(model, X, y, threshold=None):
    probabilistic = hasattr(model, "predict_proba")
    score = model.predict_proba(X)[:, 1] if probabilistic else model.decision_function(X)
    pred = model.predict(X) if threshold is None else (score >= threshold).astype(int)
    assert np.isfinite(score).all()
    result = {"ROC_AUC": roc_auc_score(y, score), "AP": average_precision_score(y, score),
        "Gini": 2*roc_auc_score(y, score)-1,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if probabilistic:
        assert ((score >= 0) & (score <= 1)).all()
        result.update(Brier=brier_score_loss(y, score), LogLoss=log_loss(y, score))
    else:
        result.update(Brier=np.nan, LogLoss=np.nan)
    return result

def audit(models, X_train, y_train, X_test, y_test):
    rows=[]
    for name, model in models.items():
        start=perf_counter(); model.fit(X_train, y_train); duration=perf_counter()-start
        rows.append({"Model": name, **metrics(model, X_test, y_test), "fit_seconds": duration})
    table=pd.DataFrame(rows)
    print("Audit final des candidats préspécifiés ; aucune nouvelle sélection à partir du test.")
    print(table.to_string(index=False))
    return table

def figures(models, X, y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    for display, title in [(RocCurveDisplay, "ROC"), (PrecisionRecallDisplay, "Precision-Recall")]:
        fig, ax=plt.subplots()
        for name, model in models.items(): display.from_estimator(model, X, y, ax=ax, name=name)
        ax.set_title(title); fig.tight_layout()
    fig, ax=plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            CalibrationDisplay.from_estimator(model, X, y, n_bins=8, strategy="quantile", ax=ax, name=name)
    fig.tight_layout(); plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def scaled(model): return Pipeline([("scaler",StandardScaler()),("svm",model)])
def models():
    return {"Logistic Regression":scaled(LogisticRegression(max_iter=2000,random_state=42)),
        "SVM linéaire":scaled(SVC(kernel="linear", C=1)),
        "SVM RBF":scaled(SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")),
        "Random Forest":RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42,n_jobs=1)}

def regression():
    from sklearn.datasets import make_regression
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    X,y=make_regression(n_samples=1000,n_features=8,noise=20,random_state=42)
    X_fit,X_eval,y_fit,y_eval=train_test_split(X,y,test_size=.2,random_state=42)
    reg=TransformedTargetRegressor(regressor=Pipeline([("scaler",StandardScaler()),
        ("svr",SVR(C=10,epsilon=.1,gamma="scale"))]),transformer=StandardScaler())
    scores=cross_validate(reg,X_fit,y_fit,cv=KFold(5,shuffle=True,random_state=42),scoring="neg_root_mean_squared_error")
    reg.fit(X_fit,y_fit); pred=reg.predict(X_eval)
    print("SVR cible standardisée : RMSE CV",-scores["test_score"].mean(),
        "test MAE",mean_absolute_error(y_eval,pred),"RMSE",root_mean_squared_error(y_eval,pred),"R²",r2_score(y_eval,pred))
    assert np.isfinite(pred).all()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--plot",action="store_true");args=parser.parse_args()
    X_train,X_test,y_train,y_test=dataset(2000,15,8,4,.1)
    candidates=models(); chosen,table=benchmark(candidates,X_train,y_train)
    audit(candidates,X_train,y_train,X_test,y_test)
    regression()
    if args.plot:
        figures(candidates,X_test,y_test)
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from sklearn.inspection import DecisionBoundaryDisplay
        X_demo,y_demo=make_blobs(n_samples=100,centers=2,cluster_std=1.2,random_state=42)
        model=SVC(kernel="linear",C=1).fit(X_demo,y_demo)
        fig,ax=plt.subplots();DecisionBoundaryDisplay.from_estimator(model,X_demo,response_method="decision_function",plot_method="contour",levels=[-1,0,1],ax=ax)
        ax.scatter(*X_demo.T,c=y_demo,cmap="coolwarm")
        ax.scatter(*model.support_vectors_.T,s=160,facecolors="none",edgecolors="black")
        plt.show()
~~~

## 29. GridSearchCV pour SVM

Les paramètres les plus importants sont :

- C : régularisation.
- kernel : type de noyau.
- gamma : portée du noyau RBF.
- degree : degré du noyau polynomial.

~~~python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
param_grid = [
    {"svm__kernel": ["linear"], "svm__C": [.1, 1, 10]},
    {"svm__kernel": ["rbf"], "svm__C": [.1, 1, 10, 100], "svm__gamma": [.001, .01, .1, 1]},
    {"svm__kernel": ["poly"], "svm__C": [.1, 1, 10], "svm__degree": [2, 3], "svm__gamma": ["scale"]}]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=1, error_score="raise")
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
~~~

Pourquoi le Pipeline est-il important ici ?
Parce que chaque combinaison d'hyperparamètres est évaluée sur plusieurs folds.

Le StandardScaler doit être ajusté uniquement sur les observations d'entraînement de chaque fold.

Sinon, nous introduisons du Data Leakage.

## Partie XII — SVM et risque de crédit

## 30. Application à la Probability of Default

Supposons que nous voulons prédire le défaut d'une entreprise à partir de :

- Rentabilité financière.
- Ratio de solvabilité.
- Capacité de remboursement.
- Endettement.
- Charge financière.
- Secteur d'activité.

Un SVM peut apprendre une frontière non linéaire entre les entreprises en défaut et les entreprises sans défaut.
Avantages potentiels

- Bonne capacité de classification sur des datasets de taille modérée.
- Modélisation de relations non linéaires avec RBF.
- Régularisation contrôlée par C.
- Possibilité d'utiliser des features nombreuses.

Limites

- Sensibilité au Scaling.
- Coût d'entraînement sur de grands datasets.
- Interprétation plus difficile avec les noyaux non linéaires.
- Probabilités non directement disponibles dans la formulation SVM standard.
- Calibration à vérifier pour un usage en estimation de PD.

Un SVM peut être un bon modèle de comparaison, mais le choix d'un modèle de PD nécessite aussi de considérer l'explicabilité, la stabilité temporelle et les exigences de validation.

## Partie XIII — Comparaison avec les autres algorithmes

## 31. SVM versus Logistic Regression, KNN et Random Forest

Critère	Logistic Regression	KNN	SVM	Random Forest
Frontière linéaire	Oui par défaut	Non nécessairement	Selon noyau	Non nécessairement
Scaling	Recommandé	Essentiel	Très important	Généralement inutile
Non-linéarité	Via features transformées	Oui	Oui avec kernels	Oui
Interprétabilité	Bonne	Limitée	Bonne en linéaire, faible en RBF	Modérée à faible
Probabilités natives	Oui	Proportions de voisins	Pas directement	Oui
Grands datasets	Souvent efficace	Prédiction coûteuse	RBF parfois coûteux	Souvent adapté
Hyperparamètres clés	C	K, distance	C, gamma, kernel	Nombre et complexité des arbres

Quand choisir un SVM ?
Un SVM est particulièrement intéressant lorsque :

- Le dataset est de taille petite à moyenne.
- Les features sont principalement numériques.
- Une frontière complexe peut être nécessaire.
- Le Scaling est correctement réalisé.
- Nous voulons comparer un modèle à marge maximale aux autres familles d'algorithmes.

## Partie XIV — Coding Challenge

## 32. Challenge : SVM Credit Risk

Challenge 09 — Support Vector Machines

+700 XP

Construis un benchmark entre Logistic Regression, SVM linéaire, SVM RBF et Random Forest sur un dataset synthétique déséquilibré.

### Questions d'analyse

1. Le noyau RBF surpasse-t-il toujours le noyau linéaire ?
2. Que se passe-t-il lorsque C devient très grand ?
3. Que se passe-t-il lorsque gamma devient très élevé ?
4. Pourquoi le Scaling modifie-t-il fortement les résultats ?
5. Pourquoi un SVM RBF peut-il devenir coûteux ?
6. Pourquoi ROC-AUC et Average Precision peuvent-elles classer les modèles différemment ?

### Squelette du challenge

~~~python
"""Challenge 09. Conserver les trois fichiers de ce cours dans le même dossier."""
from ch03_cours09_lab import dataset
X_train, X_test, y_train, y_test = dataset(3000, 20, 10, 5, .07)
# TODO : réserver une validation distincte pour le seuil ou l'early stopping.
# TODO : construire les modèles et leurs pipelines sans fuite de données.
# TODO : comparer les candidats en CV sur train, puis figer la sélection.
# TODO : réaliser le tuning sans utiliser le test.
# TODO : auditer ROC-AUC, AP, et Brier uniquement pour des probabilités.
# TODO : produire les figures et une recommandation argumentée.
raise NotImplementedError("Compléter le challenge avant de consulter le corrigé.")
~~~

[[solution]]

## Questions d’entretien

[[interview]]

## 33. Mini-quiz interactif

[[quiz]]

## 34. Validation du cours

[[assessment]]

Passer au cours 10 — Comparaison, évaluation et sélection des modèles
À retenir : les SVM construisent des frontières de décision en maximisant la marge, tandis que le Kernel Trick leur permet de modéliser des séparations non linéaires. Les paramètres C, gamma et kernel jouent un rôle essentiel dans leur comportement.

Le prochain cours sera le dernier du chapitre 3 : nous apprendrons à évaluer, comparer et sélectionner rigoureusement les modèles supervisés, comme dans un véritable projet de Data Science.