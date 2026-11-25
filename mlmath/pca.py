"""PCA via SVD and via covariance eigendecomposition."""
import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        ev = S ** 2 / (n - 1)
        k = self.n_components or len(S)
        self.explained_variance_all_ = ev
        self.explained_variance_ = ev[:k]
        self.explained_variance_ratio_ = ev[:k] / ev.sum()
        self.components_ = Vt[:k]
        self.n_components_ = k
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


def pca_eig(X):
    """Eigenvalues/eigenvectors of the sample covariance, sorted descending."""
    C = np.cov(np.asarray(X, float), rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def n_components_for_variance(ratio, target):
    return int(np.searchsorted(np.cumsum(ratio), target) + 1)
