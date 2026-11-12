import numpy as np
import pandas as pd

def audit_dataset(df,target_col):
    if target_col not in df:raise ValueError('Colonne cible absente.')
    return {'n_rows':len(df),'n_columns':df.shape[1],'duplicates':int(df.duplicated().sum()),
            'missing_rate':(100*df.isna().mean()).round(2).to_dict(),
            'target_distribution':df[target_col].value_counts(normalize=True).to_dict()}

def label_quality(labels,oof_probabilities):
    labels=np.asarray(labels);p=np.asarray(oof_probabilities,dtype=float)
    if labels.ndim!=1 or p.ndim!=2 or len(p)!=len(labels) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=p.shape[1]) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Labels entiers et probabilités OOF normalisées alignées requis.')
    return p[np.arange(len(labels)),labels]

def select_for_annotation(probabilities,budget=20,strategy='entropy'):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or not len(p) or p.shape[1]<2 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1) or not isinstance(budget,int) or budget<0:
        raise ValueError('Probabilités multiclasses normalisées et budget entier positif ou nul requis.')
    if strategy=='entropy':
        terms=np.zeros_like(p);np.log(p,out=terms,where=p>0);score=-(p*terms).sum(axis=1)
    elif strategy=='uncertainty':score=1-p.max(axis=1)
    elif strategy=='margin':
        ordered=np.sort(p,axis=1);score=-(ordered[:,-1]-ordered[:,-2])
    else:raise ValueError('Stratégie inconnue.')
    return np.argsort(-score,kind='stable')[:budget]

def weak_majority(votes):
    votes=np.asarray(votes)
    if votes.ndim!=2 or not np.isin(votes,[-1,0,1]).all():raise ValueError('Votes -1 (abstention), 0 ou 1 requis.')
    positive=(votes==1).sum(axis=1);negative=(votes==0).sum(axis=1)
    return np.where(positive>negative,1,np.where(negative>positive,0,-1))

def lf_promotion(text):
    return 1 if any(term in text.lower() for term in ['free prize','limited offer','win money']) else -1

def lf_trusted_sender(sender):
    return 0 if sender.split('@')[-1].lower() in ['company.com','university.edu'] else -1

def run(plot=False):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from cleanlab.filter import find_label_issues
    X,y=make_classification(n_samples=2400,n_features=20,n_informative=12,n_redundant=4,n_classes=3,class_sep=1.2,random_state=42)
    train,test=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=42)
    # Le test propre reste réservé. La politique de revue et son budget sont fixés à l'avance.
    truth=y[train].copy();noisy=truth.copy();rng=np.random.default_rng(42)
    corrupted=rng.choice(len(noisy),int(.15*len(noisy)),replace=False)
    noisy[corrupted]=(noisy[corrupted]+rng.integers(1,3,len(corrupted)))%3
    def estimator():return RandomForestClassifier(n_estimators=70,max_depth=10,random_state=42,n_jobs=1)
    cv=StratifiedKFold(3,shuffle=True,random_state=42)
    oof=cross_val_predict(estimator(),X[train],noisy,cv=cv,method='predict_proba',n_jobs=1)
    mask=find_label_issues(labels=noisy,pred_probs=oof)
    budget=100;quality=label_quality(noisy,oof)
    suspects=np.flatnonzero(mask);selected=suspects[np.argsort(quality[suspects],kind='stable')[:budget]]
    random_review=rng.choice(len(noisy),len(selected),replace=False)
    corrected=noisy.copy();corrected[selected]=truth[selected]
    random_corrected=noisy.copy();random_corrected[random_review]=truth[random_review]
    # Oracle simulé seulement pour les labels du train ; jamais pour une correction automatique réelle.
    results={}
    for name,labels in [('Bruit initial',noisy),('Revue OOF',corrected),('Revue aléatoire même budget',random_corrected)]:
        model=estimator().fit(X[train],labels);results[name]=f1_score(y[test],model.predict(X[test]),average='macro')
    print('Audit final, politiques fixées sans consulter le test :',results)
    print('Labels revus :',len(selected),'erreurs confirmées :',int(np.sum(noisy[selected]!=truth[selected])))
    print('Active Learning, entropie sur pool fictif :',select_for_annotation(oof[:20],5))
    print('Weak supervision :',weak_majority([[1,1,-1],[1,0,-1],[-1,-1,-1],[0,-1,0]]))
    frame=pd.DataFrame(X[train,:3],columns=['x1','x2','x3']);frame['label']=noisy
    print('Audit :',audit_dataset(frame,'label'))
    print('Une simulation unique ne garantit pas un gain ; répéter avec plusieurs seeds et validation indépendante pour choisir une politique.')
    return {'test_macro_F1':results,'reviewed':len(selected)}

if __name__=='__main__':run()
