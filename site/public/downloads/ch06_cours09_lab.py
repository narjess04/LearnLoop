import numpy as np

def binary_inputs(y,probabilities):
    y=np.asarray(y);probabilities=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or not len(y) or probabilities.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Cible binaire et probabilités alignées requises.')
    return y,probabilities

def compute_cost(y_true,y_proba,threshold,cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    y,probabilities=binary_inputs(y_true,y_proba)
    if not np.isfinite([threshold,cost_fp,cost_fn,cost_tp,cost_tn]).all():raise ValueError('Seuil et coûts finis requis.')
    action=(probabilities>=threshold).astype(int)
    costs=np.array([[cost_tn,cost_fn],[cost_fp,cost_tp]],dtype=float)
    return float(costs[action,y.astype(int)].sum())

def bayes_threshold(cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    denominator=(cost_fp-cost_tn)+(cost_fn-cost_tp)
    if denominator<=0:raise ValueError('Dénominateur positif requis ; comparer directement les risques sinon.')
    return (cost_fp-cost_tn)/denominator

def optimal_actions(probabilities,costs):
    probabilities=np.asarray(probabilities,dtype=float);costs=np.asarray(costs,dtype=float)
    if probabilities.ndim!=1 or costs.shape!=(2,2) or not np.isfinite(costs).all() or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Probabilités et matrice action x état 2x2 requises.')
    risks=(1-probabilities[:,None])*costs[:,0]+probabilities[:,None]*costs[:,1]
    return risks.argmin(axis=1)

def select_threshold(y_val,proba_val,**costs):
    y_val,proba_val=binary_inputs(y_val,proba_val)
    # Inclure toutes les politiques distinctes, notamment signaler tout / rien.
    thresholds=np.unique(np.r_[0.,proba_val,np.nextafter(1.,np.inf)])
    values=np.array([compute_cost(y_val,proba_val,t,**costs) for t in thresholds])
    index=int(np.argmin(values))
    return float(thresholds[index]),float(values[index])

def net_benefit(y_true,y_proba,threshold):
    y,proba=binary_inputs(y_true,y_proba)
    if not 0<threshold<1:raise ValueError('Seuil strictement entre 0 et 1 requis.')
    action=proba>=threshold
    tp=np.sum(action&(y==1));fp=np.sum(action&(y==0))
    return float(tp/len(y)-fp/len(y)*threshold/(1-threshold))

def run(plot=False):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV,calibration_curve
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix
    X,y=make_classification(n_samples=4500,n_features=20,n_informative=12,n_redundant=4,weights=[.95,.05],random_state=42)
    dev,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
    # Calibration croisée dans le train ; validation indépendante pour le seuil.
    model=CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,random_state=42),method='sigmoid',cv=3)
    model.fit(X[train],y[train]);pv=model.predict_proba(X[val])[:,1]
    print('Validation AUC/AP/Brier/Log Loss :',roc_auc_score(y[val],pv),average_precision_score(y[val],pv),brier_score_loss(y[val],pv),log_loss(y[val],pv))
    threshold,cost=select_threshold(y[val],pv)
    print('Seuil choisi sur validation :',threshold,'Coût validation :',cost,'Seuil théorique :',bayes_threshold())
    pt=model.predict_proba(X[test])[:,1]
    # Audit final des règles fixées : aucune sélection sur le test.
    result={}
    for name,t in [('Défaut .5',.5),('Seuil validation',threshold),('Théorique',bayes_threshold())]:
        result[name]=compute_cost(y[test],pt,t);print(name,'coût test',result[name],confusion_matrix(y[test],pt>=t,labels=[0,1]).tolist())
    print('Ne rien signaler / tout signaler :',compute_cost(y[test],pt,np.nextafter(1.,np.inf)),compute_cost(y[test],pt,0))
    print('Bénéfice net à .1 :',net_benefit(y[test],pt,.1),'(sans unité monétaire).')
    if plot:
        import matplotlib.pyplot as plt
        thresholds=np.linspace(0,1,101);plt.plot(thresholds,[compute_cost(y[val],pv,t) for t in thresholds])
        plt.axvline(threshold,linestyle='--');plt.xlabel('Seuil');plt.ylabel('Coût sur validation');plt.tight_layout();plt.show()
        observed,predicted=calibration_curve(y[val],pv,n_bins=8,strategy='quantile')
        plt.plot(predicted,observed,'o-');plt.plot([0,1],[0,1],'--');plt.title('Calibration sur validation');plt.show()
    return {'threshold':threshold,'test_costs':result}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
