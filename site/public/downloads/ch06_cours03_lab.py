import numpy as np

def ses_forecast(y, alpha=0.3):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not 0 <= alpha <= 1:
        raise ValueError('Série finie non vide et alpha dans [0,1] requis.')
    level = y[0]
    for value in y[1:]:
        level = alpha*value + (1-alpha)*level
    return float(level)

def select_best(results):
    eligible = [row for row in results if row.get('converged', True) and np.isfinite(row['MAE'])]
    if not eligible:
        raise ValueError('Aucun modèle convergé avec un score fini.')
    return min(eligible, key=lambda row: row['MAE']).copy()

def residual_model_df(order, seasonal_order):
    p, d, q = order
    P, D, Q, m = seasonal_order
    return p+q+P+Q

import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

CANDIDATES = [((0,1,1),(0,1,1,12)), ((1,1,1),(0,1,1,12)),
              ((1,1,0),(1,1,0,12)), ((1,0,1),(1,1,0,12))]

def dataset():
    rng = np.random.default_rng(42)
    t = np.arange(96)
    dates = pd.date_range('2018-01-01', periods=96, freq='MS')
    sales = 100+0.8*t+20*np.sin(2*np.pi*t/12)+rng.normal(0,5,96)
    series = pd.Series(sales,index=dates,name='sales')
    # Promotion planifiée à l'avance : variable connue à chaque origine.
    exog = pd.DataFrame({'promotion': (dates.month==12).astype(float)},index=dates)
    return series, exog

def fit_candidate(train, order, seasonal_order, exog=None):
    fit = SARIMAX(train,exog=exog,order=order,seasonal_order=seasonal_order,
                  enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
    if not fit.mle_retvals.get('converged',False):
        raise ValueError('Optimisation non convergée : configuration écartée.')
    return fit

def run(plot=False):
    series, exog = dataset()
    train, validation, test = series.iloc[:72],series.iloc[72:84],series.iloc[84:]
    # Comparaison descriptive sur la même validation ; pas de classement AIC entre familles.
    fitted = {
        'SES': SimpleExpSmoothing(train,initialization_method='estimated').fit(),
        'Holt': Holt(train,initialization_method='estimated').fit(),
        'Holt-Winters': ExponentialSmoothing(train,trend='add',seasonal='add',seasonal_periods=12).fit(),
        'ETS(A,A,A)': ETSModel(train,error='add',trend='add',seasonal='add',seasonal_periods=12).fit(disp=False),
        'ARIMA(1,1,1)': ARIMA(train,order=(1,1,1)).fit()}
    rows = [{'Model':name,'MAE':float(np.abs(validation-fit.forecast(12)).mean())}
            for name,fit in fitted.items()]
    rows.append({'Model':'Naïve saisonnière','MAE':float(np.abs(validation.to_numpy()-train.iloc[-12:].to_numpy()).mean())})
    print('Comparaison descriptive, validation uniquement :')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    results=[]
    for order, seasonal in CANDIDATES:
        try:
            fit=fit_candidate(train,order,seasonal)
            pred=fit.forecast(12)
            results.append({'order':order,'seasonal_order':seasonal,'converged':True,
                            'MAE':float(np.abs(validation-pred).mean()),'AIC':fit.aic,'BIC':fit.bic})
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
            print('Configuration écartée :',order,seasonal,error)
    best=select_best(results)  # Minimum MAE ; l'ordre de la liste ne décide pas du choix.
    print('Candidat SARIMA choisi sur validation :',best)
    development=series.iloc[:84]
    final_fit=fit_candidate(development,best['order'],best['seasonal_order'])
    result=final_fit.get_forecast(len(test))
    pred=result.predicted_mean
    intervals=result.conf_int(alpha=0.05)
    baseline=np.resize(development.iloc[-12:].to_numpy(),len(test))
    mae=float(np.abs(test-pred).mean())
    print('Audit test SARIMA MAE / RMSE :',mae,float(np.sqrt(np.mean((test-pred)**2))))
    print('Audit test baseline MAE :',float(np.abs(test.to_numpy()-baseline).mean()))
    print('Couverture descriptive sur 12 mois :',float(((test>=intervals.iloc[:,0]) & (test<=intervals.iloc[:,1])).mean()))
    # Les paramètres AR et MA saisonniers sont inclus dans l'ajustement des degrés de liberté.
    burn=max(final_fit.loglikelihood_burn,12)
    residuals=final_fit.resid.iloc[burn:].dropna()
    model_df=residual_model_df(best['order'],best['seasonal_order'])
    lb=acorr_ljungbox(residuals,lags=[12,18],model_df=model_df,return_df=True)
    print('Ljung-Box, model_df =',model_df,'; résidus écartés =',burn)
    print(lb)
    # Illustration SARIMAX : configuration fixée, exog connue ; comparaison sur validation.
    with_exog=fit_candidate(train,(0,1,1),(0,1,1,12),exog.iloc[:72])
    exog_pred=with_exog.get_forecast(12,exog=exog.iloc[72:84]).predicted_mean
    print('SARIMAX validation MAE :',float(np.abs(validation-exog_pred).mean()))
    if plot:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(development.index,development,label='Historique')
        ax.plot(test.index,test,label='Réel')
        ax.plot(pred.index,pred,'--',label='SARIMA')
        ax.fill_between(pred.index,intervals.iloc[:,0],intervals.iloc[:,1],alpha=.2,label='Intervalle prédictif nominal 95 %')
        ax.legend();plt.tight_layout();plt.show()
        final_fit.plot_diagnostics(figsize=(12,8));plt.tight_layout();plt.show()
    return {'best':best,'MAE':mae,'predictions':pred,'intervals':intervals,'ljung_box':lb}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true')
    run(parser.parse_args().plot)
