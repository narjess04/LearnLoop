import numpy as np
import pandas as pd

def rolling_zscore(values,window=28):
    # TODO : implémenter rolling_zscore selon la consigne.
    raise NotImplementedError('Complète rolling_zscore')

def robust_zscore(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Vecteur fini non vide requis.')
    median=np.median(values);mad=np.median(np.abs(values-median))
    if mad==0:return np.where(values==median,0.,np.copysign(np.inf,values-median))
    return .6745*(values-median)/mad

def cusum_detector(values,target_mean,k,h):
    # TODO : implémenter cusum_detector selon la consigne.
    raise NotImplementedError('Complète cusum_detector')

def population_stability_index(reference,current,bins=10,epsilon=1e-6):
    # TODO : implémenter population_stability_index selon la consigne.
    raise NotImplementedError('Complète population_stability_index')

if __name__=='__main__':
    print(cusum_detector([0,0,3,3,3],0,1,3))
