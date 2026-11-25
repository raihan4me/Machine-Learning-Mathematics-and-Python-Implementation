"""Task 5: metrics, ROC/AUC, threshold selection and the accuracy paradox."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from mlmath.metrics import (confusion_matrix, accuracy, precision, recall, f1,
                            roc_curve, auc_trapezoid, roc_auc_rank)

d = pd.read_csv("outputs/churn_test_scores.csv")
y = d["y_true"].to_numpy(int)
p = d["p_churn"].to_numpy(float)
yhat = (p >= 0.5).astype(int)

print("confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y, yhat))
ours = {"accuracy": accuracy(y, yhat), "precision": precision(y, yhat),
        "recall": recall(y, yhat), "f1": f1(y, yhat)}
sk = {"accuracy": skm.accuracy_score(y, yhat),
      "precision": skm.precision_score(y, yhat, zero_division=0),
      "recall": skm.recall_score(y, yhat, zero_division=0),
      "f1": skm.f1_score(y, yhat, zero_division=0)}
print(f"\n{'metric':10s} {'ours':>8s} {'sklearn':>8s}")
for k in ours:
    print(f"{k:10s} {ours[k]:8.4f} {sk[k]:8.4f}")
cm_equal = bool(np.array_equal(confusion_matrix(y, yhat), skm.confusion_matrix(y, yhat)))

fpr, tpr, thr = roc_curve(y, p)
a_trap, a_rank, a_sk = auc_trapezoid(fpr, tpr), roc_auc_rank(y, p), skm.roc_auc_score(y, p)
print(f"\nAUC trapezoid = {a_trap:.6f}  rank (Mann-Whitney) = {a_rank:.6f}  sklearn = {a_sk:.6f}")

idx = int(np.argmax(tpr >= 0.80))
t80 = float(thr[idx])
y80 = (p >= t80).astype(int)
r80, p80 = recall(y, y80), precision(y, y80)
print(f"\nhighest threshold with recall >= 0.80: {t80:.4f}")
print(f"  recall = {r80:.4f}  precision = {p80:.4f}  customers flagged = {y80.sum()} of {len(y)}")

rng = np.random.default_rng(5)
neg, pos = np.where(y == 0)[0], np.where(y == 1)[0]
keep = np.r_[neg, rng.choice(pos, size=max(1, len(neg) // 19), replace=False)]
yi, pi = y[keep], p[keep]
always0 = np.zeros_like(yi)
print(f"\n--- accuracy paradox: subset with {yi.mean():.1%} churners ---")
print(f"'never churn' model: accuracy = {accuracy(yi, always0):.4f}  recall = {recall(yi, always0):.4f}")
print(f"our model @0.5     : accuracy = {accuracy(yi, (pi>=0.5).astype(int)):.4f}  "
      f"recall = {recall(yi, (pi>=0.5).astype(int)):.4f}  precision = {precision(yi, (pi>=0.5).astype(int)):.4f}")
print(f"AUC on subset      : {roc_auc_rank(yi, pi):.4f}  (ranking quality is prevalence-independent)")

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"model (AUC={a_trap:.3f})")
plt.plot([0, 1], [0, 1], "--", label="random")
plt.scatter([fpr[idx]], [tpr[idx]], color="red", zorder=3, label=f"t={t80:.2f}")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/roc_churn.png", dpi=100)

json.dump({"ours": ours, "sklearn": sk, "cm_equal": cm_equal, "auc_trapezoid": a_trap,
           "auc_rank": a_rank, "auc_sklearn": a_sk, "recall80_threshold": t80,
           "recall80_recall": r80, "recall80_precision": p80,
           "paradox_accuracy": accuracy(yi, always0), "paradox_recall": recall(yi, always0)},
          open("outputs/task5_metrics.json", "w"), indent=2)
print("saved outputs/roc_churn.png")
