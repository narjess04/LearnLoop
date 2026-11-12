import numpy as np
import pandas as pd

def survival_inputs(durations,events):
    durations=np.asarray(durations,dtype=float);events=np.asarray(events)
    if durations.ndim!=1 or not len(durations) or events.shape!=durations.shape or not np.isfinite(durations).all() or np.any(durations<0) or not np.isin(events,[0,1]).all():
        raise ValueError('Durées finies non négatives et événements binaires alignés requis.')
    return durations,events.astype(bool)

def kaplan_meier(durations,events):
    durations,events=survival_inputs(durations,events);survival=1.;rows=[]
    for time in np.unique(durations):
        at_risk=int(np.sum(durations>=time));deaths=int(np.sum((durations==time)&events))
        censored=int(np.sum((durations==time)&~events))
        # Les censures au même temps restent dans le risque lors des événements.
        survival*=1-deaths/at_risk
        rows.append({'time':time,'at_risk':at_risk,'events':deaths,'censored':censored,'survival':survival})
    return pd.DataFrame(rows)

def survival_at(curve,time):
    known=curve[curve['time']<=time]
    return 1. if known.empty else float(known.iloc[-1]['survival'])

def risk_concordance(durations,events,risk):
    durations,events=survival_inputs(durations,events);risk=np.asarray(risk,dtype=float)
    if risk.shape!=durations.shape or not np.isfinite(risk).all():raise ValueError('Risques finis alignés requis.')
    concordant=0.;comparable=0
    # Convention pédagogique : exclure les durées égales ; risque élevé = événement précoce.
    for i in range(len(durations)):
        for j in range(i+1,len(durations)):
            if durations[i]==durations[j]:continue
            early,late=(i,j) if durations[i]<durations[j] else (j,i)
            if events[early]:
                comparable+=1
                concordant+=1. if risk[early]>risk[late] else .5 if risk[early]==risk[late] else 0.
    return concordant/comparable if comparable else np.nan

if __name__=='__main__':
    print(kaplan_meier([2,4,5,7,8],[1,1,0,1,0]))
