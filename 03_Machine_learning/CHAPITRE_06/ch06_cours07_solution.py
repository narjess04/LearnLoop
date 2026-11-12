import numpy as np
import pandas as pd

def rolling_zscore(values,window=28):
    series=pd.Series(values,dtype=float).copy()
    if not isinstance(window,int) or window<2:raise ValueError('Fenêtre entière >= 2 requise.')
    past=series.shift(1);mean=past.rolling(window).mean();std=past.rolling(window).std()
    return (series-mean)/std.replace(0,np.nan)

def robust_zscore(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Vecteur fini non vide requis.')
    median=np.median(values);mad=np.median(np.abs(values-median))
    if mad==0:return np.where(values==median,0.,np.copysign(np.inf,values-median))
    return .6745*(values-median)/mad

def cusum_detector(values,target_mean,k,h):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite([target_mean,k,h]).all() or k<0 or h<=0:
        raise ValueError('Données finies, k>=0 et h>0 requis.')
    positive=negative=0.;alerts=[]
    for t,value in enumerate(values):
        positive=max(0.,positive+value-target_mean-k)
        negative=max(0.,negative+target_mean-value-k)
        if positive>h or negative>h:
            alerts.append(t);positive=negative=0.
    return alerts

def population_stability_index(reference,current,bins=10,epsilon=1e-6):
    reference,current=[np.asarray(v,dtype=float) for v in (reference,current)]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in (reference,current)) or bins<2 or epsilon<=0:
        raise ValueError('Distributions finies non vides et paramètres valides requis.')
    # Frontières apprises sur la référence ; bornes infinies pour conserver les nouvelles valeurs extrêmes.
    inner=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)[1:-1]))
    edges=np.r_[-np.inf,inner,np.inf]
    p=np.histogram(reference,edges)[0].astype(float);q=np.histogram(current,edges)[0].astype(float)
    p=np.maximum(p/p.sum(),epsilon);q=np.maximum(q/q.sum(),epsilon);p/=p.sum();q/=q.sum()
    return float(np.sum((p-q)*np.log(p/q)))

if __name__=='__main__':
    print(cusum_detector([0,0,3,3,3],0,1,3))
