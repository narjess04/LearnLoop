import numpy as np

def finite_vector(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Vecteur fini non vide requis.')
    return values

def pinball_loss(y_true,y_pred,tau):
    y_true,y_pred=finite_vector(y_true),finite_vector(y_pred)
    if y_true.shape!=y_pred.shape or not 0<tau<1:raise ValueError('Formes alignées et tau dans ]0,1[ requis.')
    error=y_true-y_pred
    return float(np.maximum(tau*error,(tau-1)*error).mean())

def conformal_quantile(scores,alpha=.1):
    scores=finite_vector(scores)
    if not 0<alpha<1:raise ValueError('alpha dans ]0,1[ requis.')
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=len(scores) else np.inf

def conformal_interval(y_cal,pred_cal,pred_test,alpha=.1):
    y_cal,pred_cal=finite_vector(y_cal),finite_vector(pred_cal)
    if y_cal.shape!=pred_cal.shape:raise ValueError('Calibration alignée requise.')
    q=conformal_quantile(np.abs(y_cal-pred_cal),alpha)
    pred_test=finite_vector(pred_test)
    return pred_test-q,pred_test+q

def interval_metrics(y,lower,upper,alpha=.1):
    y=finite_vector(y);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if y.shape!=lower.shape or y.shape!=upper.shape or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>upper) or not 0<alpha<1:
        raise ValueError('Intervalles ordonnés alignés et alpha dans ]0,1[ requis.')
    width=upper-lower
    # Masquer les pénalités pour éviter inf*0 avec un intervalle non borné.
    penalty=np.where(y<lower,lower-y,0)+np.where(y>upper,y-upper,0)
    return {'coverage':float(((lower<=y)&(y<=upper)).mean()),'width':float(width.mean()),
            'interval_score':float(np.mean(width+2/alpha*penalty))}

def crps_ensemble(y,samples):
    samples=np.sort(finite_vector(samples));m=len(samples)
    pair_half=np.dot(2*np.arange(1,m+1)-m-1,samples)/(m*m)
    return float(np.abs(samples-y).mean()-pair_half)

def cqr_correction(y,lower,upper,alpha=.1):
    y,lower,upper=map(finite_vector,(y,lower,upper))
    if y.shape!=lower.shape or y.shape!=upper.shape or np.any(lower>upper):raise ValueError('Quantiles ordonnés alignés requis.')
    # Variante conservatrice : ne rétrécit pas l'intervalle si le quantile est négatif.
    return max(0.,conformal_quantile(np.maximum(lower-y,y-upper),alpha))

def run(plot=False):
    import pandas as pd
    from lightgbm import LGBMRegressor
    rng=np.random.default_rng(42);t=np.arange(1200)
    sales=100+.02*t+15*np.sin(2*np.pi*t/7)+rng.normal(0,4+.008*t)
    df=pd.DataFrame({'sales':sales},index=pd.date_range('2022-01-01',periods=len(t),freq='D'))
    for lag in [1,7,14,28]:df[f'lag_{lag}']=df['sales'].shift(lag)
    df['rolling_mean_7']=df['sales'].shift(1).rolling(7).mean()
    df['rolling_std_7']=df['sales'].shift(1).rolling(7).std()
    df['day_of_week']=df.index.dayofweek;df['month']=df.index.month;data=df.dropna()
    a,b=int(.65*len(data)),int(.8*len(data));train,cal,test=data.iloc[:a],data.iloc[a:b],data.iloc[b:]
    features=[c for c in data if c!='sales'];models={}
    for tau in [.05,.5,.95]:
        model=LGBMRegressor(objective='quantile',alpha=tau,n_estimators=180,learning_rate=.05,num_leaves=15,random_state=42,verbosity=-1,n_jobs=1)
        model.fit(train[features],train['sales']);models[tau]=model
    # Même règle de réarrangement sur calibration et test avant toute évaluation.
    cal_q=np.sort(np.column_stack([models[q].predict(cal[features]) for q in [.05,.5,.95]]),axis=1)
    test_q=np.sort(np.column_stack([models[q].predict(test[features]) for q in [.05,.5,.95]]),axis=1)
    qhat=cqr_correction(cal['sales'],cal_q[:,0],cal_q[:,2],.1)
    lower,median,upper=test_q[:,0]-qhat,test_q[:,1],test_q[:,2]+qhat
    print('Correction CQR conservatrice :',qhat)
    print('Quantiles bruts :',interval_metrics(test['sales'],test_q[:,0],test_q[:,2]))
    print('CQR :',interval_metrics(test['sales'],lower,upper))
    print('Pinball :',{q:pinball_loss(test['sales'],test_q[:,j],q) for j,q in enumerate([.05,.5,.95])})
    # Mise à jour après observation, jamais avant : protocole à un pas uniquement.
    scores=list(np.abs(cal['sales'].to_numpy()-cal_q[:,1]));rolling_lower=[];rolling_upper=[]
    for actual,pred in zip(test['sales'],median):
        radius=conformal_quantile(scores[-100:],.1)
        rolling_lower.append(pred-radius);rolling_upper.append(pred+radius)
        scores.append(abs(actual-pred))
    print('Rolling conformal empirique :',interval_metrics(test['sales'],rolling_lower,rolling_upper))
    print('Dépendance temporelle : ces résultats empiriques ne garantissent pas une couverture universelle.')
    print('CRPS illustratif :',crps_ensemble(108,[95,100,105,110,115]))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5));plt.plot(test.index,test['sales'],label='Réel')
        plt.plot(test.index,median,label='Médiane');plt.fill_between(test.index,lower,upper,alpha=.2,label='CQR nominal 90 %')
        plt.legend();plt.tight_layout();plt.show()
    return {'qhat':qhat,'metrics':interval_metrics(test['sales'],lower,upper)}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');run(parser.parse_args().plot)
