# Chapitre 5 — Cours 1 : Évaluation avancée des modèles

## 1. Séparer estimation et décision

Cette fiche s'appuie sur les scripts des cours 1 et 2 déjà préparés. Une probabilité, un classement et une décision à seuil répondent à des besoins différents. Un modèle peut classer correctement sans produire de probabilités calibrées.

## 2. Train, validation et test

Le train apprend les paramètres et le prétraitement. La validation permet de choisir modèle, hyperparamètres et seuil. Le test sert à une mesure finale indépendante. Pour des clients répétés, sépare par groupe ; pour une prédiction future, respecte le temps.

## 3. Données déséquilibrées

L'accuracy peut être élevée en prédisant seulement la classe majoritaire. Rapporte prévalence, matrice de confusion, précision, rappel, F1 ou F-beta et MCC. L'AUC ROC et l'average precision évaluent des scores ; elles ne choisissent pas le seuil opérationnel.

## 4. Seuil et coûts

Pour des coûts FP et FN fixés, minimise le coût sur validation. F1 et Youden correspondent à d'autres objectifs. Fixe l'objectif avant la comparaison.

\[
\mathrm{coût}=C_{FP}FP+C_{FN}FN
\]

## 5. Precision@K et Recall@K

Quand une équipe ne peut examiner que K dossiers, mesure précision et rappel parmi les K scores les plus élevés. Documente K, les ex æquo et la prévalence.

## 6. Coding Lab

Le lab compare les modèles sur données synthétiques, choisit sur validation, puis fige le modèle et le seuil pour l'audit test. Conserve les résultats réellement obtenus.

~~~python
"""Évaluation avancée : train/validation/test et métriques binaires. --plot pour les figures."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix)

def dataset():
    X,y=make_classification(n_samples=5000,n_features=20,n_informative=10,n_redundant=5,weights=[0.95,0.05],flip_y=0.01,random_state=42)
    train_valid,test=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
    train,valid=train_test_split(train_valid,test_size=0.25,stratify=y[train_valid],random_state=42)
    assert not (set(train)&set(valid) or set(train)&set(test) or set(valid)&set(test))
    return X[train],X[valid],X[test],y[train],y[valid],y[test]

def models():
    return {"Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),
            "Random Forest":RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=1)}

def evaluate(y,probabilities,threshold=0.5,fp_cost=100,fn_cost=1000):
    p=np.asarray(probabilities,dtype=float);y=np.asarray(y)
    if p.ndim!=1 or y.shape!=p.shape or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
        raise ValueError("Cible binaire et probabilités finies alignées requises.")
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"Accuracy":accuracy_score(y,pred),"Balanced Accuracy":balanced_accuracy_score(y,pred),
            "Precision":precision_score(y,pred,zero_division=0),"Recall":recall_score(y,pred,zero_division=0),
            "Specificity":tn/(tn+fp) if tn+fp else np.nan,
            "F1":f1_score(y,pred,zero_division=0),"F2":fbeta_score(y,pred,beta=2,zero_division=0),
            "MCC":matthews_corrcoef(y,pred),"ROC-AUC":roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
            "AP":average_precision_score(y,p) if np.any(y==1) else np.nan,
            "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"Cost":int(fp_cost*fp+fn_cost*fn)}

def plot_evaluation(y,p,threshold=0.5):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay,PrecisionRecallDisplay,ConfusionMatrixDisplay
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    RocCurveDisplay.from_predictions(y,p,ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y,p,ax=axes[1])
    ConfusionMatrixDisplay.from_predictions(y,(p>=threshold).astype(int),labels=[0,1],ax=axes[2])
    fig.tight_layout();plt.show()

def run(plot=False):
    X_train,X_valid,X_test,y_train,y_valid,y_test=dataset()
    fitted=models();rows=[]
    for name,model in fitted.items():
        model.fit(X_train,y_train)
        rows.append({"Model":name,**evaluate(y_valid,model.predict_proba(X_valid)[:,1])})
    print("Comparaison sur validation, pas sur test :")
    print(pd.DataFrame(rows).to_string(index=False))
    # Exemple final fixé à l'avance : RF, seuil 0.5. Le corrigé choisit sur validation.
    p=fitted["Random Forest"].predict_proba(X_test)[:,1]
    result=evaluate(y_test,p)
    print("Audit final RF au seuil fixé 0.5 :",result)
    assert result["TN"]+result["FP"]+result["FN"]+result["TP"]==len(y_test)
    if plot:plot_evaluation(y_test,p)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true")
    run(parser.parse_args().plot)
~~~

## Quiz

[[quiz]]

## Questions d'entretien

[[interview]]

## Bilan

[[assessment]]