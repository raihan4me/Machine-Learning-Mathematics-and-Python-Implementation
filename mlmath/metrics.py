"""Classification metrics from first principles."""
import numpy as np
from scipy.stats import rankdata


def confusion_matrix(y_true, y_pred):
    t = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(np.sum((t == 1) & (p == 1)))
    tn = int(np.sum((t == 0) & (p == 0)))
    fp = int(np.sum((t == 0) & (p == 1)))
    fn = int(np.sum((t == 1) & (p == 0)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred):
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def roc_curve(y_true, scores):
    """Return fpr, tpr, thresholds; positive means score >= threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_block = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[last_of_block]
    fps = (last_of_block + 1) - tps
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    thresholds = np.r_[np.inf, s[last_of_block]]
    return fpr, tpr, thresholds


def auc_trapezoid(x, y):
    return float(np.trapz(y, x))


def roc_auc_rank(y_true, scores):
    y = np.asarray(y_true).astype(int)
    r = rankdata(np.asarray(scores, float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
