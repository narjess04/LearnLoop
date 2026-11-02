import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(100, 1))
y = np.sin(X[:, 0]) + rng.normal(0, 0.25, 100)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)
degrees = [1, 3, 15]
x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, degree in zip(axes, degrees):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(), LinearRegression()
    )
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    val_mse = mean_squared_error(y_val, model.predict(X_val))
    print(degree, "MSE train:", train_mse, "MSE validation:", val_mse)
    ax.scatter(X_train[:, 0], y_train, label="train", s=14)
    ax.scatter(X_val[:, 0], y_val, label="validation", s=14)
    ax.plot(x_plot[:, 0], model.predict(x_plot), color="olive")
    ax.set_title(f"Degré {degree}")
    ax.set_ylim(-3, 3)
    ax.legend()
fig.tight_layout()
plt.show()
