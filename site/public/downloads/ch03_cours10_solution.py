"""Projet final synthétique : données mixtes, valeurs manquantes, sélection, calibration,
seuil selon coûts fictifs et audit indépendant. Installer xgboost. --plot optionnel.
Aucune date n'est simulée : ce découpage ne démontre pas une validation hors période.
"""
from ch03_cours10_lab import *
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--plot",action="store_true");args=parser.parse_args()
    X,y=make_classification(n_samples=5000,n_features=20,n_informative=10,n_redundant=5,
        weights=[.92,.08],flip_y=.01,random_state=42)
    X=pd.DataFrame(X,columns=[f"feature_{i}" for i in range(20)])
    rng=np.random.default_rng(42)
    X["secteur"]=rng.choice(["Industrie","Commerce","Services"],len(X))
    X.loc[rng.random(len(X))<.05,"feature_0"]=np.nan
    X.loc[rng.random(len(X))<.03,"secteur"]=np.nan
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    X_fit,X_valid,y_fit,y_valid=train_test_split(X_train,y_train,test_size=.25,stratify=y_train,random_state=42)
    numeric=[f"feature_{i}" for i in range(20)]
    prep=ColumnTransformer([
        ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
        ("category",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),["secteur"])])
    # Préparation ajustée dans chaque fold. Une standardisation commune est sans effet
    # sur les splits des arbres et simplifie ici la comparaison des sept familles.
    base={"Logistic Regression":LogisticRegression(max_iter=2000,random_state=42),
        "Decision Tree":DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42,n_jobs=1),
        "XGBoost":XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.05,tree_method="hist",eval_metric="logloss",n_jobs=1,random_state=42),
        "SVM":SVC(kernel="rbf",C=1,gamma="scale")}
    candidates={name:Pipeline([("prepare",clone(prep)),("model",model)]) for name,model in base.items()}
    chosen,table=benchmark(candidates,X_fit,y_fit)
    print("Le critère prédéfini est ROC-AUC CV ; le temps et l'explicabilité restent à discuter.")
    # Tuning limité mais réel de la famille retenue, exclusivement sur X_fit.
    grids={"Logistic Regression":{"model__C":[.1,1,10]},
        "Decision Tree":{"model__max_depth":[3,5,8]},
        "Random Forest":{"model__min_samples_leaf":[1,5,10]},
        "XGBoost":{"model__max_depth":[2,4,6]},
        "SVM":{"model__C":[.1,1,10]}}
    search=GridSearchCV(candidates[chosen],grids[chosen],scoring="roc_auc",
        cv=StratifiedKFold(5,shuffle=True,random_state=42),n_jobs=1,error_score="raise")
    search.fit(X_fit,y_fit);print("Tuning",search.best_params_)
    # La calibration CV réajuste aussi le pipeline dans chaque fold.
    selected=CalibratedClassifierCV(clone(search.best_estimator_),method="sigmoid",cv=5)
    selected.fit(X_fit,y_fit)
    p=selected.predict_proba(X_valid)[:,1]
    thresholds=np.linspace(0,1,201)
    def cost(t):
        pred=p>=t;return 100*np.sum((pred==1)&(y_valid==0))+2000*np.sum((pred==0)&(y_valid==1))
    threshold=float(min(thresholds,key=cost))
    print("Coûts fictifs FP=100/FN=2000 ; seuil validation figé",threshold,"coût",cost(threshold))
    # Ne pas réentraîner après choix du seuil : conserver le modèle évalué en validation.
    report=metrics(selected,X_test,y_test,threshold)
    pred=(selected.predict_proba(X_test)[:,1]>=threshold).astype(int)
    print("AUDIT FINAL UNIQUE",chosen,report,"matrice",confusion_matrix(y_test,pred))
    assert np.isfinite(list(report.values())).all()
    if args.plot:figures({chosen+" calibré":selected},X_test,y_test)
    print("Rapport : comparer au taux de base ; justifier seuil, stabilité, latence et limites.")
    print("Données synthétiques, aucune preuve de causalité ou de validité opérationnelle en crédit.")
