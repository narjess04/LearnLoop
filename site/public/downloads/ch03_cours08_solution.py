"""Challenge 08. --iterations 30 par défaut ; --plot optionnel.
Recherche CV sur X_fit, puis early stopping sur X_stop distinct ; test indépendant.
Les scores internes servent à sélectionner et ne mesurent pas la procédure complète.
"""
from ch03_cours08_lab import *
from sklearn.model_selection import RandomizedSearchCV

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--plot", action="store_true"); args=parser.parse_args()
    if args.iterations < 1: parser.error("--iterations doit être positif")
    X_train, X_test, y_train, y_test=dataset()
    X_fit, X_stop, y_fit, y_stop=train_test_split(X_train, y_train, test_size=.2, stratify=y_train, random_state=42)
    candidates=models(); chosen, table=benchmark(candidates, X_fit, y_fit)
    distribution={"n_estimators":[100,200,400,600], "max_depth":[2,3,4,5,6],
        "learning_rate":[.01,.03,.05,.1], "subsample":[.7,.8,1.],
        "colsample_bytree":[.7,.8,1.], "reg_lambda":[1,5,10], "min_child_weight":[1,3,5]}
    search=RandomizedSearchCV(XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=1),
        distribution, n_iter=args.iterations, scoring="roc_auc", cv=StratifiedKFold(5,shuffle=True,random_state=42),
        n_jobs=1, random_state=42, error_score="raise")
    search.fit(X_fit, y_fit)
    print("Budget", args.iterations, "paramètres", search.best_params_, "AUC sélection CV", search.best_score_)
    # Choix du nombre d'arbres uniquement sur la validation réservée.
    params={**search.best_params_, "n_estimators":2000}
    early=XGBClassifier(**params, tree_method="hist", eval_metric="logloss", early_stopping_rounds=50, random_state=42, n_jobs=1)
    early.fit(X_fit, y_fit, eval_set=[(X_stop,y_stop)], verbose=False)
    assert 0 <= early.best_iteration < 2000
    print("Itération optimale (index à partir de zéro)", early.best_iteration)
    # Deux analyses prévues à l'avance : benchmark fixe et procédure XGBoost optimisée.
    audit(candidates, X_train, y_train, X_test, y_test)
    print("XGBoost optimisé, audit test unique", metrics(early, X_test, y_test))
    if args.plot: figures({**candidates, "XGBoost early stopping":early}, X_test, y_test)
    print("Aucune révision des paramètres après cet audit.")
