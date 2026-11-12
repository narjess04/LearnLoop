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

def run(plot=False):
    from statsmodels.tsa.seasonal import STL
    from sklearn.ensemble import IsolationForest
    from scipy.stats import ks_2samp,wasserstein_distance
    import ruptures as rpt
    rng=np.random.default_rng(42);n=400;t=np.arange(n)
    values=100+.02*t+8*np.sin(2*np.pi*t/7)+rng.normal(0,3,n)
    values[250:]+=25;values[80]+=45;values[320]-=40
    df=pd.DataFrame({'value':values},index=pd.date_range('2025-01-01',periods=n,freq='D'))
    df['z']=rolling_zscore(df['value'],28);df['anomaly']=df['z'].abs()>3
    reference=df['value'].iloc[:60]
    alerts=cusum_detector(df['value'],reference.mean(),k=4,h=35)
    print('Alertes CUSUM, seuils fixés à l’avance :',alerts)
    # Rétrospectif : STL et PELT exploitent la série complète.
    stl=STL(df['value'],period=7,robust=True).fit();signal=df['value'].to_numpy()-stl.seasonal.to_numpy()
    breakpoints=rpt.Pelt(model='l2',min_size=30).fit(signal).predict(pen=300)
    binary=rpt.Binseg(model='l2',min_size=30).fit(signal).predict(n_bkps=1)
    print('PELT rétrospectif (fin du signal incluse) :',breakpoints,'Binary Segmentation :',binary)
    df['rolling_mean']=df['value'].shift(1).rolling(28).mean();df['rolling_std']=df['value'].shift(1).rolling(28).std()
    features=['value','rolling_mean','rolling_std'];clean=df.dropna(subset=features)
    train=clean.loc[clean.index<df.index[70]];test=clean.loc[clean.index>=df.index[70]]
    model=IsolationForest(n_estimators=100,contamination=.03,random_state=42).fit(train[features])
    print('Isolation Forest :',int((model.predict(test[features])==-1).sum()),'alertes après observation courante.')
    current=df['value'].iloc[300:]
    print('Drift descriptif : PSI',population_stability_index(reference,current),'KS',ks_2samp(reference,current).statistic,
          'Wasserstein',wasserstein_distance(reference,current))
    print('Tests KS usuels : dépendance sérielle et tests multiples à considérer avant interprétation des p-values.')
    actual=np.zeros(n,dtype=bool);actual[[80,320]]=True;detected=df['anomaly'].to_numpy()
    tp=np.sum(actual&detected);fp=np.sum(~actual&detected);fn=np.sum(actual&~detected)
    print('Pics injectés : TP/FP/FN',tp,fp,fn,'; changement durable évalué séparément autour de 250.')
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5));plt.plot(df.index,values,label='Signal')
        plt.scatter(df.index[detected],values[detected],color='red',label='Rolling Z')
        for bp in breakpoints[:-1]:plt.axvline(df.index[bp],color='orange',linestyle='--')
        plt.legend();plt.tight_layout();plt.show()
    return {'breakpoints':breakpoints,'alerts':alerts}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
