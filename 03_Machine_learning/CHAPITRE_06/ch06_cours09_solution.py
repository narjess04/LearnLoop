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

if __name__=='__main__':
    print(bayes_threshold())
