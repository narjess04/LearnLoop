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

if __name__=='__main__':
    print(ses_forecast([10,20,15],.3))
