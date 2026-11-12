import numpy as np
import pandas as pd

def demand_vector(y):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError('Demande non négative, finie, non vide et unidimensionnelle requise.')
    return y

def croston_forecast(y,alpha=.1):
    # TODO : implémenter croston_forecast selon la consigne.
    raise NotImplementedError('Complète croston_forecast')

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
    # TODO : implémenter mint_reconcile selon la consigne.
    raise NotImplementedError('Complète mint_reconcile')

if __name__=='__main__':
    print(croston_forecast([0,0,5,0,0,8]))
