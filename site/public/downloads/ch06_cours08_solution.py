import numpy as np

def soft_vote(probabilities,weights=None):
    probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.ndim!=2 or not probabilities.size or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Matrice observations x modèles de probabilités requise.')
    weights=np.ones(probabilities.shape[1]) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(probabilities.shape[1],) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Poids positifs ou nuls, somme strictement positive requise.')
    return probabilities@weights/weights.sum()

def hard_vote(predictions):
    predictions=np.asarray(predictions)
    if predictions.ndim!=2 or not predictions.size or not np.isin(predictions,[0,1]).all():raise ValueError('Prédictions binaires requises.')
    # Convention explicite : égalité en faveur de la classe 0.
    return (predictions.mean(axis=1)>.5).astype(int)

def generate_oof(estimator,X,y,splits):
    from sklearn.base import clone
    X,y=np.asarray(X),np.asarray(y)
    if len(X)!=len(y) or not np.isin(y,[0,1]).all():raise ValueError('Cible binaire alignée requise.')
    predictions=np.full(len(y),np.nan);visits=np.zeros(len(y),dtype=int)
    for train_idx,val_idx in splits:
        train_idx,val_idx=np.asarray(train_idx),np.asarray(val_idx)
        if np.intersect1d(train_idx,val_idx).size or not len(train_idx) or not len(val_idx):raise ValueError('Folds disjoints non vides requis.')
        if np.any(train_idx<0) or np.any(val_idx<0) or np.any(train_idx>=len(y)) or np.any(val_idx>=len(y)) or np.any(visits[val_idx]):
            raise ValueError('Indices valides et une seule visite par observation requis.')
        fitted=clone(estimator).fit(X[train_idx],y[train_idx])
        column=np.flatnonzero(fitted.classes_==1)
        if len(column)!=1:raise ValueError('Chaque train doit contenir la classe positive.')
        predictions[val_idx]=fitted.predict_proba(X[val_idx])[:,column[0]];visits[val_idx]+=1
    if not np.all(visits==1):raise ValueError('Toutes les observations doivent recevoir une prédiction OOF.')
    return predictions

if __name__=='__main__':
    print(soft_vote([[.8,.6,.7]],[1,2,3]))
