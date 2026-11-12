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

def run(plot=False):
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split,StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,VotingClassifier,StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
    X,y=make_classification(n_samples=1800,n_features=20,n_informative=12,n_redundant=4,random_state=42)
    dev,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
    Xtr,ytr=X[train],y[train]
    estimators=[('lr',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))),
                ('rf',RandomForestClassifier(n_estimators=70,random_state=42,n_jobs=1)),
                ('hgb',HistGradientBoostingClassifier(max_iter=80,random_state=42))]
    cv=StratifiedKFold(3,shuffle=True,random_state=42);splits=list(cv.split(Xtr,ytr))
    Z=np.column_stack([generate_oof(model,Xtr,ytr,splits) for name,model in estimators])
    meta=LogisticRegression(max_iter=1000).fit(Z,ytr)
    bases=[clone(model).fit(Xtr,ytr) for name,model in estimators]
    candidates={name:model for (name,_),model in zip(estimators,bases)}
    candidates['Soft Voting']=VotingClassifier(estimators=estimators,voting='soft').fit(Xtr,ytr)
    candidates['Hard Voting']=VotingClassifier(estimators=estimators,voting='hard').fit(Xtr,ytr)
    candidates['Stacking sklearn']=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(max_iter=1000),cv=splits,stack_method='predict_proba').fit(Xtr,ytr)
    base_idx,blend_idx=train_test_split(np.arange(len(ytr)),test_size=.25,stratify=ytr,random_state=42)
    blend_bases=[clone(model).fit(Xtr[base_idx],ytr[base_idx]) for name,model in estimators]
    blend_meta=LogisticRegression(max_iter=1000).fit(np.column_stack([m.predict_proba(Xtr[blend_idx])[:,1] for m in blend_bases]),ytr[blend_idx])
    def proba(name,indices):
        if name=='Stacking manuel':return meta.predict_proba(np.column_stack([m.predict_proba(X[indices])[:,1] for m in bases]))[:,1]
        if name=='Blending':return blend_meta.predict_proba(np.column_stack([m.predict_proba(X[indices])[:,1] for m in blend_bases]))[:,1]
        return candidates[name].predict_proba(X[indices])[:,1]
    names=[name for name in candidates if name!='Hard Voting']+['Stacking manuel','Blending'];rows=[]
    for name in names:rows.append({'model':name,'AUC_validation':roc_auc_score(y[val],proba(name,val))})
    comparison=pd.DataFrame(rows).sort_values('AUC_validation',ascending=False);winner=comparison.iloc[0]['model']
    print(comparison.round(4).to_string(index=False));print('Hard Voting, accuracy validation :',accuracy_score(y[val],candidates['Hard Voting'].predict(X[val])))
    # Décision fixée avant de consulter le test. Les bases de blending restent sur leur sous-train.
    probabilities=proba(winner,test)
    print('Audit final du modèle choisi :',winner,'AUC',roc_auc_score(y[test],probabilities),'Log Loss',log_loss(y[test],probabilities))
    print('Corrélation des erreurs probabilistes sur validation :')
    print(np.corrcoef(np.column_stack([y[val]-m.predict_proba(X[val])[:,1] for m in bases]),rowvar=False))
    return {'winner':winner,'comparison':comparison}

if __name__=='__main__':run()
