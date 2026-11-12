import numpy as np
import pandas as pd

def audit_dataset(df,target_col):
    # TODO : compléter audit_dataset.
    raise NotImplementedError('Complète audit_dataset')

def label_quality(labels,oof_probabilities):
    labels=np.asarray(labels);p=np.asarray(oof_probabilities,dtype=float)
    if labels.ndim!=1 or p.ndim!=2 or len(p)!=len(labels) or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=p.shape[1]) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Labels entiers et probabilités OOF normalisées alignées requis.')
    return p[np.arange(len(labels)),labels]

def select_for_annotation(probabilities,budget=20,strategy='entropy'):
    # TODO : compléter select_for_annotation.
    raise NotImplementedError('Complète select_for_annotation')

def weak_majority(votes):
    # TODO : compléter weak_majority.
    raise NotImplementedError('Complète weak_majority')

def lf_promotion(text):
    return 1 if any(term in text.lower() for term in ['free prize','limited offer','win money']) else -1

def lf_trusted_sender(sender):
    return 0 if sender.split('@')[-1].lower() in ['company.com','university.edu'] else -1

if __name__=='__main__':
    print(weak_majority([[1,1,-1],[1,0,-1]]))
