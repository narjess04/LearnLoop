"""Cours 09. Classification SVM et régression SVR ; --plot optionnel."""

import argparse
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)

def dataset(n=5000, features=20, informative=10, redundant=5, positive=.08):
    X, y = make_classification(n_samples=n, n_features=features,
        n_informative=informative, n_redundant=redundant, weights=[1-positive, positive],
        flip_y=.01, random_state=42)
    return train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def benchmark(models, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
            scoring={"ROC_AUC": "roc_auc", "AP": "average_precision"})
        rows.append({"Model": name, "ROC_AUC_CV": scores["test_ROC_AUC"].mean(),
            "ROC_AUC_std": scores["test_ROC_AUC"].std(), "AP_CV": scores["test_AP"].mean(),
            "fit_seconds_CV_mean": scores["fit_time"].mean()})
    table = pd.DataFrame(rows).sort_values("ROC_AUC_CV", ascending=False)
    print(table.to_string(index=False))
    chosen = table.iloc[0]["Model"]
    print("Choix figé sur la ROC-AUC CV avant de consulter le test :", chosen)
    return chosen, table

def metrics(model, X, y, threshold=None):
    probabilistic = hasattr(model, "predict_proba")
    score = model.predict_proba(X)[:, 1] if probabilistic else model.decision_function(X)
    pred = model.predict(X) if threshold is None else (score >= threshold).astype(int)
    assert np.isfinite(score).all()
    result = {"ROC_AUC": roc_auc_score(y, score), "AP": average_precision_score(y, score),
        "Gini": 2*roc_auc_score(y, score)-1,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if probabilistic:
        assert ((score >= 0) & (score <= 1)).all()
        result.update(Brier=brier_score_loss(y, score), LogLoss=log_loss(y, score))
    else:
        result.update(Brier=np.nan, LogLoss=np.nan)
    return result

def audit(models, X_train, y_train, X_test, y_test):
    rows=[]
    for name, model in models.items():
        start=perf_counter(); model.fit(X_train, y_train); duration=perf_counter()-start
        rows.append({"Model": name, **metrics(model, X_test, y_test), "fit_seconds": duration})
    table=pd.DataFrame(rows)
    print("Audit final des candidats préspécifiés ; aucune nouvelle sélection à partir du test.")
    print(table.to_string(index=False))
    return table

def figures(models, X, y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    for display, title in [(RocCurveDisplay, "ROC"), (PrecisionRecallDisplay, "Precision-Recall")]:
        fig, ax=plt.subplots()
        for name, model in models.items(): display.from_estimator(model, X, y, ax=ax, name=name)
        ax.set_title(title); fig.tight_layout()
    fig, ax=plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            CalibrationDisplay.from_estimator(model, X, y, n_bins=8, strategy="quantile", ax=ax, name=name)
    fig.tight_layout(); plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def scaled(model): return Pipeline([("scaler",StandardScaler()),("svm",model)])
def models():
    return {"Logistic Regression":scaled(LogisticRegression(max_iter=2000,random_state=42)),
        "SVM linéaire":scaled(SVC(kernel="linear", C=1)),
        "SVM RBF":scaled(SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")),
        "Random Forest":RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42,n_jobs=1)}

def regression():
    from sklearn.datasets import make_regression
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    X,y=make_regression(n_samples=1000,n_features=8,noise=20,random_state=42)
    X_fit,X_eval,y_fit,y_eval=train_test_split(X,y,test_size=.2,random_state=42)
    reg=TransformedTargetRegressor(regressor=Pipeline([("scaler",StandardScaler()),
        ("svr",SVR(C=10,epsilon=.1,gamma="scale"))]),transformer=StandardScaler())
    scores=cross_validate(reg,X_fit,y_fit,cv=KFold(5,shuffle=True,random_state=42),scoring="neg_root_mean_squared_error")
    reg.fit(X_fit,y_fit); pred=reg.predict(X_eval)
    print("SVR cible standardisée : RMSE CV",-scores["test_score"].mean(),
        "test MAE",mean_absolute_error(y_eval,pred),"RMSE",root_mean_squared_error(y_eval,pred),"R²",r2_score(y_eval,pred))
    assert np.isfinite(pred).all()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--plot",action="store_true");args=parser.parse_args()
    X_train,X_test,y_train,y_test=dataset(2000,15,8,4,.1)
    candidates=models(); chosen,table=benchmark(candidates,X_train,y_train)
    audit(candidates,X_train,y_train,X_test,y_test)
    regression()
    if args.plot:
        figures(candidates,X_test,y_test)
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from sklearn.inspection import DecisionBoundaryDisplay
        X_demo,y_demo=make_blobs(n_samples=100,centers=2,cluster_std=1.2,random_state=42)
        model=SVC(kernel="linear",C=1).fit(X_demo,y_demo)
        fig,ax=plt.subplots();DecisionBoundaryDisplay.from_estimator(model,X_demo,response_method="decision_function",plot_method="contour",levels=[-1,0,1],ax=ax)
        ax.scatter(*X_demo.T,c=y_demo,cmap="coolwarm")
        ax.scatter(*model.support_vectors_.T,s=160,facecolors="none",edgecolors="black")
        plt.show()
