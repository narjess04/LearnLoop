import numpy as np

def soft_vote(probabilities,weights=None):
    # TODO : compléter soft_vote.
    raise NotImplementedError('Complète soft_vote')

def hard_vote(predictions):
    predictions=np.asarray(predictions)
    if predictions.ndim!=2 or not predictions.size or not np.isin(predictions,[0,1]).all():raise ValueError('Prédictions binaires requises.')
    # Convention explicite : égalité en faveur de la classe 0.
    return (predictions.mean(axis=1)>.5).astype(int)

def generate_oof(estimator,X,y,splits):
    # TODO : compléter generate_oof.
    raise NotImplementedError('Complète generate_oof')

if __name__=='__main__':
    print(soft_vote([[.8,.6,.7]],[1,2,3]))
