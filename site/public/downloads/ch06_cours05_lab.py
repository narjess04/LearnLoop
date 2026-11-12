import numpy as np
import pandas as pd

def demand_vector(y):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError('Demande non négative, finie, non vide et unidimensionnelle requise.')
    return y

def croston_forecast(y,alpha=.1):
    y=demand_vector(y)
    if not 0<alpha<=1:
        raise ValueError('alpha dans ]0,1] requis.')
    positive=np.flatnonzero(y>0)
    if not len(positive):return 0.
    first=positive[0];z_hat=y[first];p_hat=float(first+1);last=first
    for idx in positive[1:]:
        z_hat=alpha*y[idx]+(1-alpha)*z_hat
        p_hat=alpha*(idx-last)+(1-alpha)*p_hat
        last=idx
    return float(z_hat/p_hat)

def sba_forecast(y,alpha=.1):
    return (1-alpha/2)*croston_forecast(y,alpha)

def tsb_forecast(y,alpha=.1,beta=.1):
    y=demand_vector(y)
    if not 0<alpha<=1 or not 0<beta<=1:
        raise ValueError('alpha et beta dans ]0,1] requis.')
    positive=np.flatnonzero(y>0)
    if not len(positive):return 0.
    first=positive[0];size=y[first];probability=1./(first+1)
    for value in y[first+1:]:
        probability=beta*float(value>0)+(1-beta)*probability
        if value>0:size=alpha*value+(1-alpha)*size
    return float(probability*size)

def demand_characteristics(y):
    y=demand_vector(y);positive=y[y>0]
    if not len(positive):return {'ADI':np.inf,'CV2':np.nan,'zero_rate':1.}
    return {'ADI':len(y)/len(positive),'CV2':float(np.var(positive)/np.mean(positive)**2),
            'zero_rate':float(np.mean(y==0))}

def classify_demand(adi,cv2):
    if np.isinf(adi):return 'All-zero'
    if not np.isfinite(adi) or not np.isfinite(cv2) or adi<1 or cv2<0:
        raise ValueError('ADI >= 1 et CV2 >= 0 finis requis.')
    if adi<1.32:return 'Smooth' if cv2<.49 else 'Erratic'
    return 'Intermittent' if cv2<.49 else 'Lumpy'

def create_panel_features(df):
    data=df.sort_values(['series_id','date']).copy()
    data['date']=pd.to_datetime(data['date'])
    if data.duplicated(['series_id','date']).any():raise ValueError('Dates dupliquées dans une série.')
    gaps=data.groupby('series_id',observed=True)['date'].diff().dropna()
    if not (gaps==pd.Timedelta(days=1)).all():raise ValueError('Compléter les calendriers sans confondre zéro et manquant.')
    grouped=data.groupby('series_id',observed=True)['sales']
    for lag in [1,7,14,28]:data[f'lag_{lag}']=grouped.shift(lag)
    data['rolling_mean_7']=grouped.transform(lambda s:s.shift(1).rolling(7).mean())
    data['rolling_std_7']=grouped.transform(lambda s:s.shift(1).rolling(7).std())
    data['day_of_week']=data['date'].dt.dayofweek;data['month']=data['date'].dt.month
    data['series_id']=data['series_id'].astype('category')
    return data.dropna(subset=['sales','lag_28','rolling_mean_7','rolling_std_7'])

def bottom_up(bottom,S):
    bottom=np.asarray(bottom,dtype=float);S=np.asarray(S,dtype=float)
    return S@bottom

def top_down(total,shares):
    shares=np.asarray(shares,dtype=float)
    if shares.ndim!=1 or not np.isfinite(shares).all() or np.any(shares<0) or not np.isclose(shares.sum(),1):
        raise ValueError('Proportions positives de somme 1 requises, estimées sur train.')
    return float(total)*shares

def mint_reconcile(base,S,W):
    base=np.asarray(base,dtype=float);S=np.asarray(S,dtype=float);W=np.asarray(W,dtype=float)
    if W.shape!=(S.shape[0],S.shape[0]) or base.shape!=(S.shape[0],):raise ValueError('Dimensions incompatibles.')
    if not np.isfinite(W).all() or not np.allclose(W,W.T) or np.linalg.eigvalsh(W).min()<=0:
        raise ValueError('Covariance symétrique définie positive requise ; régulariser si nécessaire.')
    weighted=np.linalg.solve(W,S)
    bottom=np.linalg.solve(S.T@weighted,weighted.T@base)
    return S@bottom

def dataset():
    rng=np.random.default_rng(42);dates=pd.date_range('2024-01-01',periods=500,freq='D');records=[]
    for product in range(20):
        base=rng.uniform(20,100);trend=rng.uniform(-.01,.04);amplitude=rng.uniform(5,20)
        values=np.maximum(0,base+trend*np.arange(500)+amplitude*np.sin(2*np.pi*np.arange(500)/7)+rng.normal(0,5,500))
        records.extend({'date':date,'series_id':f'P{product:02d}','sales':value} for date,value in zip(dates,values))
    return pd.DataFrame(records)

def intermittent_dataset():
    rng=np.random.default_rng(42);t=np.arange(180)
    return pd.DataFrame({'regular':np.maximum(0,30+rng.normal(0,4,180)),
        'seasonal':np.maximum(0,25+12*np.sin(2*np.pi*t/7)+rng.normal(0,3,180)),
        'intermittent':np.where(rng.random(180)<.12,rng.integers(1,15,180),0)},
        index=pd.date_range('2025-01-01',periods=180,freq='D'))

def run(plot=False,engine='lightgbm'):
    from sklearn.metrics import mean_absolute_error
    data=create_panel_features(dataset())
    train=data[data['date']<'2025-01-01'];test=data[data['date']>='2025-01-01']
    assert train['date'].max()<test['date'].min()
    features=[c for c in data if c not in ['sales','date']]
    if engine=='lightgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(n_estimators=180,learning_rate=.05,num_leaves=15,random_state=42,verbosity=-1,n_jobs=1)
        model.fit(train[features],train['sales'],categorical_feature=['series_id'])
        pred=model.predict(test[features])
    else:
        from sklearn.ensemble import RandomForestRegressor
        # Variante pédagogique navigateur : encoder explicitement l'identifiant.
        xtrain=pd.get_dummies(train[features],columns=['series_id'],dtype=float)
        xtest=pd.get_dummies(test[features],columns=['series_id'],dtype=float).reindex(columns=xtrain.columns,fill_value=0)
        model=RandomForestRegressor(n_estimators=40,max_depth=10,random_state=42,n_jobs=1)
        model.fit(xtrain,train['sales']);pred=model.predict(xtest)
    results=test[['date','series_id','sales','lag_7']].copy();results['prediction']=pred
    results['error']=np.abs(results['sales']-results['prediction'])
    results['baseline_error']=np.abs(results['sales']-results['lag_7'])
    per_series=results.groupby('series_id',observed=True)[['error','baseline_error']].mean()
    print('Évaluation à un pas par série :');print(per_series.round(3))
    print('MAE micro :',results['error'].mean(),'MAE macro :',per_series['error'].mean())
    print('Les hyperparamètres sont fixés à l’avance ; aucun choix sur le test.')
    demands=intermittent_dataset();rows=[]
    for name in demands:
        history=demands[name].iloc[:140].to_numpy();actual=demands[name].iloc[140:].to_numpy()
        stats=demand_characteristics(history)
        print(name,stats,classify_demand(stats['ADI'],stats['CV2']))
        estimates={'Zero':0.,'Mean':history.mean(),'Croston':croston_forecast(history),
                   'SBA':sba_forecast(history),'TSB':tsb_forecast(history)}
        for method,value in estimates.items():
            predicted=np.repeat(value,len(actual))
            rows.append({'series':name,'method':method,'MAE':mean_absolute_error(actual,predicted),
                         'cumulative_error':abs(predicted.sum()-actual.sum()),'bias':float((predicted-actual).mean())})
    print('Audit à origine fixe, 40 jours, méthodes fixées :');print(pd.DataFrame(rows).round(3).to_string(index=False))
    S=np.array([[1.,1.],[1.,0.],[0.,1.]])
    # W illustrative, pas estimée sur le test ; W=I donne une réconciliation OLS.
    base=np.array([1000.,600.,450.]);W=np.diag([100.,64.,36.])
    coherent=mint_reconcile(base,S,W)
    print('Bottom-Up :',bottom_up([600,450],S),'Top-Down :',top_down(1000,[.6,.4]))
    print('Réconciliation pondérée, W illustrative :',coherent)
    assert np.isclose(coherent[0],coherent[1:].sum())
    if plot:
        import matplotlib.pyplot as plt
        demands.plot(subplots=True,figsize=(12,7));plt.tight_layout();plt.show()
    return {'per_series':per_series,'intermittent':pd.DataFrame(rows),'reconciled':coherent}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true')
    parser.add_argument('--engine',choices=['lightgbm','browser'],default='lightgbm')
    args=parser.parse_args();run(args.plot,args.engine)
