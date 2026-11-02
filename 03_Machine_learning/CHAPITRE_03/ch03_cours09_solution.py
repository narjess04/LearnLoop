"""Challenge 09. Grille complète de 25 configurations, 5 folds. --plot optionnel."""
from ch03_cours09_lab import *
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true");args=parser.parse_args()
    X_train,X_test,y_train,y_test=dataset(3000,20,10,5,.07)
    candidates=models();chosen,table=benchmark(candidates,X_train,y_train)
    grid=[{"svm__kernel":["linear"],"svm__C":[.1,1,10]},
        {"svm__kernel":["rbf"],"svm__C":[.1,1,10,100],"svm__gamma":[.001,.01,.1,1]},
        {"svm__kernel":["poly"],"svm__C":[.1,1,10],"svm__degree":[2,3],"svm__gamma":["scale"]}]
    search=GridSearchCV(scaled(SVC()),grid,scoring={"auc":"roc_auc","ap":"average_precision"},
        refit="auc",cv=StratifiedKFold(5,shuffle=True,random_state=42),n_jobs=1,error_score="raise")
    search.fit(X_train,y_train)
    print("Paramètres CV",search.best_params_,"AUC sélection",search.best_score_)
    print("Score de sélection interne ; une validation externe est nécessaire pour évaluer la procédure de tuning.")
    # Comparaison préspécifiée, sans sélection répétée sur le test.
    audit(candidates,X_train,y_train,X_test,y_test)
    best=search.best_estimator_
    print("SVM optimisé, test indépendant",metrics(best,X_test,y_test))
    np.testing.assert_allclose(best.named_steps["scaler"].mean_,X_train.mean(axis=0))
    assert not hasattr(best,"predict_proba")
    print("Scaling ajusté sur train ; scores de décision conservés, aucun Brier sur ces scores.")
    if args.plot: figures({**candidates,"SVM optimisé":best},X_test,y_test)
