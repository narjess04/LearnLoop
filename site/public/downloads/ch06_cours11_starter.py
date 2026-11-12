import numpy as np

def ranking_metrics(relevance,scores,k=5,total_relevant=None):
    # TODO : compléter ranking_metrics.
    raise NotImplementedError('Complète ranking_metrics')

def recommend_unseen(scores,seen,k=3):
    # TODO : compléter recommend_unseen.
    raise NotImplementedError('Complète recommend_unseen')

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
