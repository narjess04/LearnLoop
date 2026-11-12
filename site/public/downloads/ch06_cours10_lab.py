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

def run_rsf(train,test):
    # Extension locale optionnelle : python -m pip install scikit-survival
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored,integrated_brier_score
    features=[c for c in train if c not in ['week','arrest']]
    y_train=Surv.from_dataframe('arrest','week',train);y_test=Surv.from_dataframe('arrest','week',test)
    rsf=RandomSurvivalForest(n_estimators=100,min_samples_leaf=5,random_state=42,n_jobs=1)
    rsf.fit(train[features],y_train);risk=rsf.predict(test[features])
    score=concordance_index_censored(test['arrest'].astype(bool),test['week'],risk)[0]
    print('RSF test C-index :',score)
    # Horizon strictement dans le support de suivi observé train/test.
    end=min(train['week'].max(),test['week'].max())
    times=np.linspace(max(test['week'].min(),train['week'].min())+1,end-1,20)
    predicted=np.array([fn(times) for fn in rsf.predict_survival_function(test[features])])
    print('IBS IPCW :',integrated_brier_score(y_train,y_test,predicted,times))
    return score

def run(plot=False,rsf=False):
    from lifelines.datasets import load_rossi
    from lifelines import KaplanMeierFitter,CoxPHFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import logrank_test
    from sklearn.model_selection import train_test_split
    df=load_rossi()
    print('Données Rossi intégrées à lifelines : temps avant arrestation, pas churn client.')
    train,test=train_test_split(df,test_size=.25,stratify=df['arrest'],random_state=42)
    km=KaplanMeierFitter().fit(train['week'],train['arrest'],label='Survie sans événement, train')
    own=kaplan_meier(train['week'],train['arrest'])
    np.testing.assert_allclose([survival_at(own,t) for t in own['time']],km.predict(own['time']).to_numpy())
    model=CoxPHFitter(penalizer=.01).fit(train,duration_col='week',event_col='arrest')
    features=[c for c in df if c not in ['week','arrest']]
    risk=model.predict_partial_hazard(test[features]).to_numpy().ravel()
    cindex=concordance_index(test['week'],-risk,test['arrest'])
    print('Test C-index lifelines (signe de risque inversé) :',cindex)
    print('Hazard ratios associés :');print(model.hazard_ratios_)
    print('Diagnostic des risques proportionnels sur train :')
    model.check_assumptions(train,p_value_threshold=.05,show_plots=False)
    a,b=train[train['fin']==0],train[train['fin']==1]
    print('Log-rank descriptif sur train :',logrank_test(a['week'],b['week'],event_observed_A=a['arrest'],event_observed_B=b['arrest']).p_value)
    individual=model.predict_survival_function(test[features].iloc[[0]])
    if rsf:run_rsf(train,test)
    if plot:
        import matplotlib.pyplot as plt
        km.plot_survival_function();plt.title('Kaplan-Meier sur train');plt.show()
        individual.plot();plt.title('Survie individuelle sur test');plt.show()
    return {'C_index':cindex,'survival_curve':individual}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');parser.add_argument('--rsf',action='store_true')
    args=parser.parse_args();run(args.plot,args.rsf)
