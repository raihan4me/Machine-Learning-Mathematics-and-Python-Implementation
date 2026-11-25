"""Linear regression primitives."""
import numpy as np


def add_bias(X):
    """Prepend a column of ones."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def train_test_split_idx(n, test_frac=0.2, seed=0):
    """Return (train_idx, test_idx) from a seeded permutation."""
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def fit_normal_equation(Xb, y):
    Xb = np.asarray(Xb, float)
    y = np.asarray(y, float)
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def fit_lstsq(Xb, y):
    """SVD-based least squares; returns the minimum-norm solution."""
    w, *_ = np.linalg.lstsq(np.asarray(Xb, float), np.asarray(y, float), rcond=None)
    return w


def predict(Xb, w):
    return np.asarray(Xb, float) @ w


def mse(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean((y - yhat) ** 2))


def r2_score(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)
