"""Choix du calibrateur sur validation indépendante : train50/cal20/valid15/test15."""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ch05_cours02_lab import data,fit_versions,probability_metrics,plot_calibration

def run(plot=False):
    X,y=data();remaining,test=train_test_split(np.arange(len(y)),test_size=0.15,stratify=y,random_state=42)
    remaining,valid=train_test_split(remaining,test_size=0.15/0.85,stratify=y[remaining],random_state=13)
    train,cal=train_test_split(remaining,test_size=0.20/0.70,stratify=y[remaining],random_state=7)
    groups=[set(ids) for ids in (train,cal,valid,test)]
    assert sum(map(len,groups))==len(y) and len(set.union(*groups))==len(y)
    versions=fit_versions(X[train],y[train],X[cal],y[cal]);rows=[]
    for name,model in versions.items():rows.append({"Model":name,**probability_metrics(y[valid],model.predict_proba(X[valid])[:,1],y[train].mean())})
    selection=pd.DataFrame(rows);print("Validation :",selection.to_string(index=False),sep="\n")
    winner=selection.loc[selection.Brier.idxmin(),"Model"]
    p=versions[winner].predict_proba(X[test])[:,1]
    audit=probability_metrics(y[test],p,y[train].mean())
    print("Méthode figée, critère Brier validation :",winner)
    print("Audit final indépendant :",audit)
    print("Aucune garantie que la calibration améliore tous les scores. Réserver encore un jeu adapté si un seuil doit être réglé ensuite.")
    if plot:plot_calibration(y[test],{winner:p})
    return winner,audit

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true")
    run(parser.parse_args().plot)
