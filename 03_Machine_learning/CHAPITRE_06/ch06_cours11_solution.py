import numpy as np

def ranking_metrics(relevance,scores,k=5,total_relevant=None):
    relevance=np.asarray(relevance,dtype=float);scores=np.asarray(scores,dtype=float)
    if relevance.ndim!=1 or not len(relevance) or scores.shape!=relevance.shape or not np.isfinite(relevance).all() or not np.isfinite(scores).all() or np.any(relevance<0) or not isinstance(k,int) or k<1:
        raise ValueError('Pertinences et scores finis alignés, pertinences >=0 et K entier positif requis.')
    order=np.argsort(-scores,kind='stable');limit=min(k,len(relevance));ranked=relevance[order[:limit]]
    binary=ranked>0;available=int(np.sum(relevance>0))
    total_relevant=available if total_relevant is None else total_relevant
    if total_relevant<available:raise ValueError('Total pertinent incompatible avec les candidats.')
    discounts=np.log2(np.arange(2,limit+2))
    dcg=np.sum(np.expm1(np.log(2)*ranked)/discounts)
    ideal=np.sort(relevance)[::-1][:limit];idcg=np.sum(np.expm1(np.log(2)*ideal)/discounts)
    precisions=np.cumsum(binary)/np.arange(1,limit+1)
    # AP@K normalisée par le total pertinent ; 0 quand aucune pertinence.
    return {'precision':float(binary.sum()/k),'recall':float(binary.sum()/total_relevant) if total_relevant else 0.,
            'AP':float(np.sum(precisions*binary)/total_relevant) if total_relevant else 0.,
            'NDCG':float(dcg/idcg) if idcg>0 else 0.}

def recommend_unseen(scores,seen,k=3):
    scores=np.asarray(scores,dtype=float);seen=np.asarray(seen,dtype=bool)
    if scores.ndim!=1 or scores.shape!=seen.shape or not np.isfinite(scores).all() or k<1:raise ValueError('Scores et masque alignés requis.')
    candidates=np.flatnonzero(~seen)
    return candidates[np.argsort(-scores[candidates],kind='stable')[:k]]

def explicit_factorization(ratings,factors=3,epochs=120,learning_rate=.015,regularization=.05):
    ratings=np.asarray(ratings,dtype=float)
    if ratings.ndim!=2 or not np.isfinite(ratings[~np.isnan(ratings)]).all():raise ValueError('Matrice de notes avec NaN pour inconnus requise.')
    rng=np.random.default_rng(42);U=rng.normal(0,.2,(ratings.shape[0],factors));V=rng.normal(0,.2,(ratings.shape[1],factors))
    observed=np.argwhere(~np.isnan(ratings))
    for epoch in range(epochs):
        for i,j in observed[rng.permutation(len(observed))]:
            old=U[i].copy();error=ratings[i,j]-old@V[j]
            U[i]+=learning_rate*(error*V[j]-regularization*old)
            V[j]+=learning_rate*(error*old-regularization*V[j])
    return U,V

def implicit_als(interactions,factors=3,alpha=10.,regularization=.1,iterations=15):
    interactions=np.asarray(interactions,dtype=float)
    if interactions.ndim!=2 or not np.isfinite(interactions).all() or np.any(interactions<0):raise ValueError('Comptages non négatifs finis requis.')
    rng=np.random.default_rng(42);U=rng.normal(0,.1,(interactions.shape[0],factors));V=rng.normal(0,.1,(interactions.shape[1],factors))
    eye=np.eye(factors);preference=(interactions>0).astype(float);confidence=1+alpha*interactions
    for iteration in range(iterations):
        for i in range(len(U)):
            c=confidence[i];U[i]=np.linalg.solve((V.T*c)@V+regularization*eye,V.T@(c*preference[i]))
        for j in range(len(V)):
            c=confidence[:,j];V[j]=np.linalg.solve((U.T*c)@U+regularization*eye,U.T@(c*preference[:,j]))
    return U,V

if __name__=='__main__':
    print(ranking_metrics([3,0,2,1],[.9,.8,.7,.6],4))
