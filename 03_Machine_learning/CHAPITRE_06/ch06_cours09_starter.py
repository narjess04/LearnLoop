import numpy as np

def binary_inputs(y,probabilities):
    y=np.asarray(y);probabilities=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or not len(y) or probabilities.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Cible binaire et probabilités alignées requises.')
    return y,probabilities

def compute_cost(y_true,y_proba,threshold,cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    # TODO : compléter compute_cost.
    raise NotImplementedError('Complète compute_cost')

def bayes_threshold(cost_fp=10,cost_fn=500,cost_tp=20,cost_tn=0):
    denominator=(cost_fp-cost_tn)+(cost_fn-cost_tp)
    if denominator<=0:raise ValueError('Dénominateur positif requis ; comparer directement les risques sinon.')
    return (cost_fp-cost_tn)/denominator

def optimal_actions(probabilities,costs):
    # TODO : compléter optimal_actions.
    raise NotImplementedError('Complète optimal_actions')

def select_threshold(y_val,proba_val,**costs):
    # TODO : compléter select_threshold.
    raise NotImplementedError('Complète select_threshold')

def net_benefit(y_true,y_proba,threshold):
    y,proba=binary_inputs(y_true,y_proba)
    if not 0<threshold<1:raise ValueError('Seuil strictement entre 0 et 1 requis.')
    action=proba>=threshold
    tp=np.sum(action&(y==1));fp=np.sum(action&(y==0))
    return float(tp/len(y)-fp/len(y)*threshold/(1-threshold))

if __name__=='__main__':
    print(bayes_threshold())
