"""Gradient-based optimisation for the MSE loss."""
import numpy as np


def mse_loss(Xb, y, w):
    r = Xb @ w - y
    return float(r @ r / len(y))


def mse_gradient(Xb, y, w):
    return (2.0 / len(y)) * (Xb.T @ (Xb @ w - y))


def numerical_gradient(f, w, eps=1e-5):
    """Central finite-difference gradient of scalar function f at w."""
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return g


def batch_gradient_descent(Xb, y, lr=0.1, epochs=1000, tol=1e-10):
    Xb = np.asarray(Xb, float)
    y = np.asarray(y, float)
    w = np.zeros(Xb.shape[1])
    history = []
    for _ in range(epochs):
        loss = mse_loss(Xb, y, w)
        history.append(loss)
        if not np.isfinite(loss):
            break
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break
        w = w - lr * mse_gradient(Xb, y, w)
    return w, history


def unscale_weights(w_scaled, mean, std):
    """Map weights learned on standardized features back to original units."""
    w = np.empty_like(np.asarray(w_scaled, float))
    w[1:] = w_scaled[1:] / std
    w[0] = w_scaled[0] - np.sum(w_scaled[1:] * mean / std)
    return w
