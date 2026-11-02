import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(-2.0, 2.0, 120)
y = 3.0 * x + 2.0 + rng.normal(scale=0.2, size=x.shape)
w, b = 0.0, 0.0
learning_rate = 0.05
losses = []

for step in range(500):
    prediction = w * x + b
    error = prediction - y
    # Les deux gradients utilisent les mêmes paramètres.
    dw = 2.0 * np.mean(error * x)
    db = 2.0 * np.mean(error)
    w -= learning_rate * dw
    b -= learning_rate * db
    # Conserver une perte correspondant aux paramètres après mise à jour.
    losses.append(float(np.mean((w * x + b - y) ** 2)))

print("w :", w, "b :", b)
print("MSE finale :", losses[-1])
assert np.isfinite(losses).all()
assert abs(w - 3.0) < 0.1
assert abs(b - 2.0) < 0.1
