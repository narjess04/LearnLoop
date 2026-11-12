import numpy as np
import pandas as pd

FEATURES=['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_std_7','day_of_week','month','is_weekend']

def create_features_for_future(df):
    if not isinstance(df.index,pd.DatetimeIndex) or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError('Dates uniques et triées requises.')
    if len(df)>1 and not (df.index.to_series().diff().iloc[1:]==pd.Timedelta(days=1)).all():
        raise ValueError('Calendrier journalier régulier requis ; traiter les dates absentes.')
    data=df.copy()
    for lag in [1,7,14,28]:
        data[f'lag_{lag}']=data['sales'].shift(lag)
    past=data['sales'].shift(1)
    data['rolling_mean_7']=past.rolling(7).mean()
    data['rolling_std_7']=past.rolling(7).std()
    data['day_of_week']=data.index.dayofweek
    data['month']=data.index.month
    data['is_weekend']=(data.index.dayofweek>=5).astype(int)
    return data

def create_features(df):
    return create_features_for_future(df).dropna(subset=FEATURES+['sales'])

def recursive_forecast(model,history,horizon,feature_cols=FEATURES):
    if not isinstance(horizon,(int,np.integer)) or horizon<1:
        raise ValueError('Horizon entier positif requis.')
    history=history[['sales']].copy()
    if len(history)<28 or not np.isfinite(history['sales']).all():
        raise ValueError('Au moins 28 observations finies requises.')
    predictions=[]
    for step in range(horizon):
        next_date=history.index[-1]+pd.Timedelta(days=1)
        future=pd.DataFrame({'sales':[np.nan]},index=[next_date])
        data=create_features_for_future(pd.concat([history,future]))
        X_next=data.loc[[next_date],feature_cols]
        if not np.isfinite(X_next.to_numpy(dtype=float)).all():
            raise ValueError('Features futures incomplètes.')
        pred=float(model.predict(X_next)[0])
        if not np.isfinite(pred):
            raise ValueError('Prévision non finie.')
        history.loc[next_date,'sales']=pred
        predictions.append(pred)
    return pd.Series(predictions,index=pd.date_range(history.index[-horizon],periods=horizon,freq='D'),name='prediction')

def direct_targets(df,horizon=3):
    # Ligne datée t : origine t-1 ; cible de l'horizon h = Y[t+h-1].
    data=create_features_for_future(df)
    columns=[]
    for h in range(1,horizon+1):
        name=f'target_{h}';data[name]=data['sales'].shift(-(h-1));columns.append(name)
    return data.dropna(subset=FEATURES+columns)

def dataset():
    rng=np.random.default_rng(42);t=np.arange(1000)
    sales=100+.03*t+20*np.sin(2*np.pi*t/7)+10*np.sin(2*np.pi*t/365.25)+rng.normal(0,5,len(t))
    return pd.DataFrame({'sales':sales},index=pd.date_range('2022-01-01',periods=len(t),freq='D'))

def make_model(engine='lightgbm',iterations=200):
    if engine=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=iterations,learning_rate=.03,num_leaves=15,random_state=42,verbosity=-1,n_jobs=1)
    if engine=='browser':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=60,max_depth=8,random_state=42,n_jobs=1)
    raise ValueError('Moteur inconnu.')

def run(plot=False,engine='lightgbm'):
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    df=dataset();data=create_features(df)
    a,b=int(len(data)*.70),int(len(data)*.85)
    train,validation,test=data.iloc[:a],data.iloc[a:b],data.iloc[b:]
    model=make_model(engine,500)
    if engine=='lightgbm':
        import lightgbm as lgb
        model.fit(train[FEATURES],train['sales'],eval_set=[(validation[FEATURES],validation['sales'])],
                  eval_metric='l1',callbacks=[lgb.early_stopping(30,verbose=False)])
        iterations=max(1,model.best_iteration_)
    else:
        model.fit(train[FEATURES],train['sales']);iterations=60
    # Modèle final et réglage fixés avant l'audit test.
    development=data.iloc[:b]
    model=make_model(engine,iterations);model.fit(development[FEATURES],development['sales'])
    pred=model.predict(test[FEATURES])
    baseline=df['sales'].shift(7).loc[test.index]
    scores={'MAE':mean_absolute_error(test['sales'],pred),'RMSE':np.sqrt(mean_squared_error(test['sales'],pred)),
            'baseline_MAE':mean_absolute_error(test['sales'],baseline)}
    print('Test à un pas, historique actualisé :',scores)
    history=df.loc[df.index<test.index[0]]
    recursive=recursive_forecast(model,history,14)
    fixed_baseline=np.resize(history['sales'].iloc[-7:].to_numpy(),14)
    print('Origine fixe 14 jours, MAE modèle / baseline :',mean_absolute_error(df.loc[recursive.index,'sales'],recursive),
          mean_absolute_error(df.loc[recursive.index,'sales'],fixed_baseline))
    # Folds uniquement dans le développement : aucun tuning sur le test final.
    folds=[]
    for tr,va in TimeSeriesSplit(n_splits=3,test_size=70).split(development):
        fold_model=make_model(engine,iterations)
        fold_model.fit(development[FEATURES].iloc[tr],development['sales'].iloc[tr])
        folds.append(mean_absolute_error(development['sales'].iloc[va],fold_model.predict(development[FEATURES].iloc[va])))
    print('MAE par fold à un pas :',folds)
    print('Importance native, association et non causalité :',dict(zip(FEATURES,model.feature_importances_)))
    # Extension directe / multi-output : ne garder que les cibles disponibles avant la frontière.
    multi=direct_targets(df.loc[df.index<test.index[0]],3)
    print('Exemples multi-output sur développement :',multi.shape)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4));plt.plot(test.index,test['sales'],label='Réel')
        plt.plot(test.index,pred,label=engine+' à un pas',alpha=.8);plt.legend();plt.tight_layout();plt.show()
    return {'scores':scores,'recursive':recursive,'fold_MAE':folds}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true')
    parser.add_argument('--engine',choices=['lightgbm','browser'],default='lightgbm')
    args=parser.parse_args();run(args.plot,args.engine)
