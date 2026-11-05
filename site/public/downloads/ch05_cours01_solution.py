"""Sélection conjointe modèle/seuil sur validation ; audit final unique. --plot."""
import argparse
import numpy as np
import pandas as pd
from ch05_cours01_lab import dataset,models,evaluate,plot_evaluation

def threshold_table(y,p):
    # Les prédictions changent aux scores observés ; inclure aussi zéro et aucune alerte.
    thresholds=np.r_[0.0,np.unique(p),np.nextafter(float(np.max(p)),np.inf)]
    return pd.DataFrame([{"threshold":t,**evaluate(y,p,t)} for t in thresholds])

def precision_at_k(y,p,k=100):
    if not 1<=k<=len(y):raise ValueError("K doit être entre 1 et n.")
    order=np.argsort(-p,kind="stable")[:k]
    return float(np.mean(y[order])),float(y[order].sum()/y.sum()) if y.sum() else np.nan

def run(plot=False):
    X_train,X_valid,X_test,y_train,y_valid,y_test=dataset()
    fitted=models();candidates=[]
    for name,model in fitted.items():
        model.fit(X_train,y_train)
        table=threshold_table(y_valid,model.predict_proba(X_valid)[:,1])
        for rule,column in (("F1","F1"),("Youden","Youden"),("Cost","Cost")):
            table["Youden"]=table["Recall"]+table["Specificity"]-1
            index=table[column].idxmin() if rule=="Cost" else table[column].idxmax()
            row=table.loc[index]
            candidates.append({"model":name,"rule":rule,"threshold":row.threshold,"F1":row.F1,"Cost":row.Cost})
    selection=pd.DataFrame(candidates);print(selection.to_string(index=False))
    # Critère métier annoncé avant accès au test : FP=100, FN=1000.
    winner=min((r for r in candidates if r["rule"]=="Cost"),key=lambda r:r["Cost"])
    p=fitted[winner["model"]].predict_proba(X_test)[:,1]
    audit=evaluate(y_test,p,winner["threshold"])
    print("Choix figé sur validation :",winner)
    print("Audit final indépendant :",audit)
    print("Precision@100 et Recall@100 :",precision_at_k(y_test,p,100))
    print("Les coûts sont fictifs ; aucune interprétation de risque réel ni décision de crédit n'est déduite de ce dataset synthétique.")
    if plot:plot_evaluation(y_test,p,winner["threshold"])
    return winner,audit

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true")
    run(parser.parse_args().plot)
