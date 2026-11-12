import numpy as np
import pandas as pd

FEATURES=['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_std_7','day_of_week','month','is_weekend']

def create_features_for_future(df):
    # TODO : implémenter create_features_for_future selon la consigne.
    raise NotImplementedError('Complète create_features_for_future')

def create_features(df):
    return create_features_for_future(df).dropna(subset=FEATURES+['sales'])

def recursive_forecast(model,history,horizon,feature_cols=FEATURES):
    # TODO : implémenter recursive_forecast selon la consigne.
    raise NotImplementedError('Complète recursive_forecast')

def direct_targets(df,horizon=3):
    # Ligne datée t : origine t-1 ; cible de l'horizon h = Y[t+h-1].
    data=create_features_for_future(df)
    columns=[]
    for h in range(1,horizon+1):
        name=f'target_{h}';data[name]=data['sales'].shift(-(h-1));columns.append(name)
    return data.dropna(subset=FEATURES+columns)

if __name__=='__main__':
    print(create_features(pd.DataFrame({'sales':np.arange(40.)},index=pd.date_range('2025-01-01',periods=40))))
