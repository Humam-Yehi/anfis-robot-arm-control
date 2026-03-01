import numpy as np


def train_supervised(model, X, Y, epochs=100, lr=0.115):
    for epoch in range(epochs):
        total_loss = 0

        for x, y_true in zip(X, Y):
            y_pred = model.forward(x)
            error = y_pred - y_true
            total_loss += np.sum(error ** 2)

            firing = model.compute_firing_strengths(x)
            x_ext = np.append(x, 1.0)

            for i in range(model.n_rules):
                grad = np.outer(x_ext, error) * firing[i]
                model.weights[i] -= lr * grad

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.6f}")
