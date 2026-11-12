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

def run(plot=False):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from lightgbm import LGBMRanker
    courses=['Régression','Forêt','Boosting','CNN','Transformers']
    descriptions=['supervised classification linear model','supervised classification ensemble trees',
                  'supervised boosting decision trees','deep learning computer vision images','deep learning attention language models']
    text=TfidfVectorizer().fit_transform(descriptions);similarity=cosine_similarity(text)
    print('Contenu similaire à Forêt :',[courses[i] for i in recommend_unseen(similarity[1],np.arange(5)==1,2)])
    ratings=np.array([[5,4,np.nan,5],[5,5,np.nan,4],[1,2,5,np.nan],[np.nan,4,5,2.]])
    U,V=explicit_factorization(ratings);print('Notes prédites sur interactions inconnues :',(U@V.T)[np.isnan(ratings)])
    interactions=np.array([[3,1,0,0],[2,2,0,1],[0,0,4,3],[0,1,3,2.]])
    A,B=implicit_als(interactions);print('Items non consommés, utilisateur 0 :',recommend_unseen((A@B.T)[0],interactions[0]>0,3))
    rng=np.random.default_rng(42);rows=[]
    for query in range(90):
        for item in range(8):
            similarity,popularity,match=rng.random(3)
            relevance=int(np.clip(np.floor(2*similarity+match+.3*popularity+rng.normal(0,.2)),0,3))
            rows.append({'query':query,'similarity':similarity,'popularity':popularity,'match':match,'relevance':relevance})
    data=pd.DataFrame(rows).sort_values('query',kind='stable');features=['similarity','popularity','match']
    train=data[data['query']<54];val=data[(data['query']>=54)&(data['query']<72)];test=data[data['query']>=72]
    def fit(frame,trees):
        model=LGBMRanker(objective='lambdarank',n_estimators=trees,num_leaves=7,min_child_samples=5,random_state=42,verbosity=-1,n_jobs=1)
        model.fit(frame[features],frame['relevance'],group=frame.groupby('query',sort=False).size().to_numpy())
        return model
    def score(model,frame):
        scored=frame.copy();scored['score']=model.predict(frame[features])
        return np.mean([ranking_metrics(g['relevance'],g['score'],5)['NDCG'] for _,g in scored.groupby('query')])
    candidates=[(trees,score(fit(train,trees),val)) for trees in [30,80]]
    best=max(candidates,key=lambda pair:pair[1])[0];print('NDCG@5 exponentiel sur validation :',candidates)
    final=fit(pd.concat([train,val]),best);test_score=score(final,test)
    print('Test avec requêtes distinctes : NDCG@5',test_score)
    print('Convention : gain 2**rel-1 ; sklearn.ndcg_score utilise un gain linéaire si rel est fourni directement.')
    return {'test_NDCG':test_score,'trees':best}

if __name__=='__main__':run()
