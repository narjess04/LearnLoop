import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
true_w = np.array([2.0, -1.5, 4.0])
true_b = 3.0
y = X @ true_w + true_b
y += rng.normal(scale=0.2, size=200)

w = np.zeros(3)
b = 0.0
lr, epochs = 0.1, 200
for epoch in range(epochs):
    # TODO : predictions de forme (200,)
    # TODO : error de forme (200,)
    # TODO : dw = ... avec X.T, de forme (3,)
    # TODO : db = ...
    # TODO : mise a jour simultanee de w et b
    pass
# TODO : comparer les poids aux paramètres générateurs
# TODO : tester les gradients par différences finies
