"""Calibration holdout : RF figé, sigmoid/isotonic et audit descriptif. sklearn >=1.6. --plot."""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,average_precision_score

def calibration_bins(y,p,n_bins=10,strategy="uniform"):
    y,p=np.asarray(y),np.asarray(p,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Cible binaire et probabilités finies alignées dans [0,1] requises.")
    if not isinstance(n_bins,(int,np.integer)) or n_bins<1 or strategy not in ("uniform","quantile"):
        raise ValueError("Nombre de bins entier positif et stratégie uniform/quantile requis.")
    edges=np.linspace(0,1,n_bins+1) if strategy=="uniform" else np.quantile(p,np.linspace(0,1,n_bins+1))
    ids=np.digitize(p,edges[1:-1],right=False)
    rows=[]
    for k in range(n_bins):
        mask=ids==k
        if mask.any():rows.append({"bin":k,"count":int(mask.sum()),"mean_p":float(p[mask].mean()),"frequency":float(y[mask].mean())})
    return pd.DataFrame(rows)

def calibration_errors(y,p,n_bins=10,strategy="uniform"):
    table=calibration_bins(y,p,n_bins,strategy)
    gaps=np.abs(table.mean_p-table.frequency)
    return float(np.sum(table["count"]/len(y)*gaps)),float(gaps.max())

def expected_calibration_error(y_true,y_proba,n_bins=10):
    return calibration_errors(y_true,y_proba,n_bins)[0]

def murphy(y,p,n_bins=10):
    table=calibration_bins(y,p,n_bins);weights=table["count"]/len(y);mean=np.mean(y)
    reliability=float(np.sum(weights*(table.mean_p-table.frequency)**2))
    resolution=float(np.sum(weights*(table.frequency-mean)**2));uncertainty=float(mean*(1-mean))
    grouped=reliability-resolution+uncertainty
    return {"REL":reliability,"RES":resolution,"UNC":uncertainty,"grouped_BS":grouped,
            "actual_BS":brier_score_loss(y,p),"binning_residual":brier_score_loss(y,p)-grouped}

def probability_metrics(y,p,reference_probability):
    reference=np.full(len(y),reference_probability)
    ref_bs=brier_score_loss(y,reference)
    ece,mce=calibration_errors(y,p)
    return {"Brier":brier_score_loss(y,p),"Log Loss":log_loss(y,p,labels=[0,1]),
            "ROC-AUC":roc_auc_score(y,p),"AP":average_precision_score(y,p),
            "ECE_10_uniform":ece,"MCE_10_uniform":mce,
            "BSS_train_prevalence":1-brier_score_loss(y,p)/ref_bs if ref_bs>0 else np.nan}

def fit_versions(X_train,y_train,X_cal,y_cal):
    base=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=5,random_state=42,n_jobs=1).fit(X_train,y_train)
    tree_ids=[id(tree) for tree in base.estimators_]
    versions={"Original":base}
    for method in ("sigmoid","isotonic"):
        versions[method]=CalibratedClassifierCV(estimator=FrozenEstimator(base),method=method).fit(X_cal,y_cal)
    assert tree_ids==[id(tree) for tree in base.estimators_]
    return versions

def data():
    return make_classification(n_samples=10000,n_features=20,n_informative=12,n_redundant=4,weights=[0.90,0.10],flip_y=0.01,random_state=42)

def plot_calibration(y,predictions):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,p in predictions.items():
        table=calibration_bins(y,p,10,"quantile")
        axes[0].plot(table.mean_p,table.frequency,marker="o",label=name)
        axes[1].hist(p,bins=np.linspace(0,1,11),histtype="step",label=name)
        print(name,"effectifs quantiles :",table["count"].tolist())
    axes[0].plot([0,1],[0,1],"k--")
    axes[0].set(xlabel="Probabilité moyenne",ylabel="Fréquence observée")
    axes[1].set(xlabel="Probabilité",ylabel="Effectif")
    for ax in axes:ax.legend()
    fig.tight_layout();plt.show()

def run(plot=False):
    X,y=data();remaining,test=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
    train,cal=train_test_split(remaining,test_size=0.25,stratify=y[remaining],random_state=42)
    assert not(set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test))
    versions=fit_versions(X[train],y[train],X[cal],y[cal]);predictions={name:m.predict_proba(X[test])[:,1] for name,m in versions.items()}
    table=pd.DataFrame([{"Model":name,**probability_metrics(y[test],p,y[train].mean())} for name,p in predictions.items()])
    print("Audit descriptif de trois variantes fixées ; ne pas choisir sur test :")
    print(table.to_string(index=False))
    for name,p in predictions.items():print(name,"Murphy groupé (approximation) :",murphy(y[test],p))
    print("Référence BSS : prévalence train, évaluée sur le même test. Brier/Log Loss évaluent la qualité probabiliste globale.")
    if plot:plot_calibration(y[test],predictions)
    return table

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true")
    run(parser.parse_args().plot)
