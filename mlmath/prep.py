"""Data preparation helpers."""
import numpy as np
import pandas as pd


def impute_median(df, columns):
    out = df.copy()
    medians = {}
    for col in columns:
        med = float(out[col].median())
        out[col] = out[col].fillna(med)
        medians[col] = med
    return out, medians


class Standardizer:
    """z = (x - mean) / std, column-wise, population std (ddof=0)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
