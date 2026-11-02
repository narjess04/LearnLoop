# Reprendre X_train, X_val, y_train, y_val du Coding Lab.
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(
        PolynomialFeatures(degree=15, include_bias=False),
        StandardScaler(), Ridge(alpha=alpha)
    )
    # TODO : fit sur train, puis MSE train et validation.
    # TODO : tracer les erreurs en fonction de alpha.
    pass
# Comparer les réglages sur validation, pas sur un test final.
