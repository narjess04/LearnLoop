import numpy as np
import pandas as pd

def survival_inputs(durations,events):
    durations=np.asarray(durations,dtype=float);events=np.asarray(events)
    if durations.ndim!=1 or not len(durations) or events.shape!=durations.shape or not np.isfinite(durations).all() or np.any(durations<0) or not np.isin(events,[0,1]).all():
        raise ValueError('Durées finies non négatives et événements binaires alignés requis.')
    return durations,events.astype(bool)

def kaplan_meier(durations,events):
    # TODO : compléter kaplan_meier.
    raise NotImplementedError('Complète kaplan_meier')

def survival_at(curve,time):
    known=curve[curve['time']<=time]
    return 1. if known.empty else float(known.iloc[-1]['survival'])

def risk_concordance(durations,events,risk):
    # TODO : compléter risk_concordance.
    raise NotImplementedError('Complète risk_concordance')

if __name__=='__main__':
    print(kaplan_meier([2,4,5,7,8],[1,1,0,1,0]))
