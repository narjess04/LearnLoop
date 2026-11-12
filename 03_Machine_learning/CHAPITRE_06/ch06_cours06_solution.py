import numpy as np

def finite_vector(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Vecteur fini non vide requis.')
    return values

def pinball_loss(y_true,y_pred,tau):
    y_true,y_pred=finite_vector(y_true),finite_vector(y_pred)
    if y_true.shape!=y_pred.shape or not 0<tau<1:raise ValueError('Formes alignées et tau dans ]0,1[ requis.')
    error=y_true-y_pred
    return float(np.maximum(tau*error,(tau-1)*error).mean())

def conformal_quantile(scores,alpha=.1):
    scores=finite_vector(scores)
    if not 0<alpha<1:raise ValueError('alpha dans ]0,1[ requis.')
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=len(scores) else np.inf

def conformal_interval(y_cal,pred_cal,pred_test,alpha=.1):
    y_cal,pred_cal=finite_vector(y_cal),finite_vector(pred_cal)
    if y_cal.shape!=pred_cal.shape:raise ValueError('Calibration alignée requise.')
    q=conformal_quantile(np.abs(y_cal-pred_cal),alpha)
    pred_test=finite_vector(pred_test)
    return pred_test-q,pred_test+q

def interval_metrics(y,lower,upper,alpha=.1):
    y=finite_vector(y);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if y.shape!=lower.shape or y.shape!=upper.shape or np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>upper) or not 0<alpha<1:
        raise ValueError('Intervalles ordonnés alignés et alpha dans ]0,1[ requis.')
    width=upper-lower
    # Masquer les pénalités pour éviter inf*0 avec un intervalle non borné.
    penalty=np.where(y<lower,lower-y,0)+np.where(y>upper,y-upper,0)
    return {'coverage':float(((lower<=y)&(y<=upper)).mean()),'width':float(width.mean()),
            'interval_score':float(np.mean(width+2/alpha*penalty))}

def crps_ensemble(y,samples):
    samples=np.sort(finite_vector(samples));m=len(samples)
    pair_half=np.dot(2*np.arange(1,m+1)-m-1,samples)/(m*m)
    return float(np.abs(samples-y).mean()-pair_half)

def cqr_correction(y,lower,upper,alpha=.1):
    y,lower,upper=map(finite_vector,(y,lower,upper))
    if y.shape!=lower.shape or y.shape!=upper.shape or np.any(lower>upper):raise ValueError('Quantiles ordonnés alignés requis.')
    # Variante conservatrice : ne rétrécit pas l'intervalle si le quantile est négatif.
    return max(0.,conformal_quantile(np.maximum(lower-y,y-upper),alpha))

if __name__=='__main__':
    print(pinball_loss([120],[100],.9))
