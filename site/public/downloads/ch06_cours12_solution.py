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

if __name__=='__main__':
    print(weak_majority([[1,1,-1],[1,0,-1]]))
