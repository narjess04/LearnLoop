import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scalers = {"No Scaling": "passthrough", "StandardScaler": StandardScaler(),
           "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":
    for name, scaler in scalers.items():
        model = Pipeline([("scaler", scaler),
                          ("model", LogisticRegression(max_iter=10000))])
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        assert scores.shape == (5,) and np.isfinite(scores).all()
        print(name, scores, "mean:", scores.mean(), "std:", scores.std())
    print("Le test final reste réservé pendant cette comparaison.")
