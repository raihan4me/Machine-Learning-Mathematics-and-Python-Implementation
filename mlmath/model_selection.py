"""Polynomial features, ridge regression and k-fold cross-validation."""
import numpy as np
from mlmath.linreg import add_bias, mse

def polynomial_features(x, degree):
    """Columns x, x^2, ..., x^degree (no bias column)."""
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x ** k for k in range(1, degree + 1)])

def ridge_fit(Xb, y, lam):
    Xb = np.asarray(Xb, float)
    y = np.asarray(y, float)
    p = Xb.shape[1]
    D = np.eye(p)
    D[0, 0] = 0.0
    A = np.vstack([Xb, np.sqrt(lam) * D])
    b = np.concatenate([y, np.zeros(p)])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w

def kfold_indices(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, folds[i]

def cross_val_mse(X, y, lam, k=5, seed=0):
    scores = []
    for tr, va in kfold_indices(len(y), k, seed):
        w = ridge_fit(add_bias(X[tr]), y[tr], lam)
        scores.append(mse(y[va], add_bias(X[va]) @ w))
    return float(np.mean(scores))
