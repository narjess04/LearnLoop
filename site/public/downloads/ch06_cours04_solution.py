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

if __name__=='__main__':
    print(create_features(pd.DataFrame({'sales':np.arange(40.)},index=pd.date_range('2025-01-01',periods=40))))
