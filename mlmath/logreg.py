"""Logistic regression from the negative log-likelihood."""
import numpy as np
from mlmath.linreg import add_bias


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))                 # always in (0, 1]: no overflow
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_loss(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def loss_and_grad(Xb, y, w, l2=0.0):
    y = np.asarray(y, float)
    p = sigmoid(Xb @ w)
    loss = log_loss(y, p) + 0.5 * l2 * float(np.sum(w[1:] ** 2))
    grad = Xb.T @ (p - y) / len(y)
    grad[1:] += l2 * w[1:]
    return loss, grad


class LogisticRegressionGD:
    def __init__(self, lr=1.0, epochs=5000, l2=0.0, tol=1e-12):
        self.lr, self.epochs, self.l2, self.tol = lr, epochs, l2, tol

    def fit(self, X, y):
        Xb = add_bias(X)
        y = np.asarray(y, float)
        w = np.zeros(Xb.shape[1])
        self.history_ = []
        for _ in range(self.epochs):
            loss, grad = loss_and_grad(Xb, y, w, self.l2)
            self.history_.append(loss)
            if len(self.history_) > 1 and abs(self.history_[-2] - loss) < self.tol:
                break
            w = w - self.lr * grad
        self.w_ = w
        self.n_iter_ = len(self.history_)
        return self

    def decision_function(self, X):
        return add_bias(X) @ self.w_

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
