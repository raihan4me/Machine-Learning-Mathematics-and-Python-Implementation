"""Task 4: logistic regression from scratch vs scikit-learn."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlmath.prep import Standardizer
from mlmath.linreg import train_test_split_idx, add_bias
from mlmath.gd import numerical_gradient, unscale_weights
from mlmath.logreg import sigmoid, log_loss, loss_and_grad, LogisticRegressionGD

FEATURES = ["tenure_months", "monthly_charge", "support_calls"]
df = pd.read_csv("data/churn.csv")
X = df[FEATURES].to_numpy(float)
y = df["churned"].to_numpy(float)
tr, te = train_test_split_idx(len(y), 0.2, seed=11)
sc = Standardizer().fit(X[tr])
Ztr, Zte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]

print("--- numerical stability ---")
z = np.array([-1000.0, 0.0, 1000.0])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    naive = 1.0 / (1.0 + np.exp(-z))
print(f"naive sigmoid : {naive}  warnings raised: {len(caught)}")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    stable = sigmoid(z)
print(f"stable sigmoid: {stable}  warnings raised: {len(caught)}")
print(f"log loss, positive example predicted p=0 (clipped): {log_loss(np.array([1.0]), np.array([0.0])):.4f}")

w0 = np.random.default_rng(1).normal(size=4)
_, g = loss_and_grad(add_bias(Ztr), ytr, w0)
gn = numerical_gradient(lambda v: loss_and_grad(add_bias(Ztr), ytr, v)[0], w0)
print(f"\ngradient check relative error: {np.linalg.norm(g - gn) / np.linalg.norm(g + gn):.2e}")

print("\n--- unregularised fit ---")
ours = LogisticRegressionGD(lr=1.0, epochs=20000).fit(Ztr, ytr)
sk = LogisticRegression(penalty=None, max_iter=10000, tol=1e-8).fit(Ztr, ytr)
w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
d0 = float(np.max(np.abs(ours.w_ - w_sk)))
print(f"ours    (scaled): {np.round(ours.w_, 4)}  iterations: {ours.n_iter_}")
print(f"sklearn (scaled): {np.round(w_sk, 4)}")
print(f"max abs difference: {d0:.2e}")

print("\n--- L2-regularised fit: l2 = 0.05  <=>  C = 1/(n*l2) ---")
l2 = 0.05
C = 1.0 / (len(ytr) * l2)
ours_l2 = LogisticRegressionGD(lr=1.0, epochs=20000, l2=l2).fit(Ztr, ytr)
sk_l2 = LogisticRegression(C=C, max_iter=10000, tol=1e-8).fit(Ztr, ytr)
d1 = float(np.max(np.abs(ours_l2.w_ - np.r_[sk_l2.intercept_, sk_l2.coef_.ravel()])))
print(f"C = {C:.5f}; max abs difference: {d1:.2e}")
print(f"||w|| without / with L2: {np.linalg.norm(ours.w_[1:]):.4f} / {np.linalg.norm(ours_l2.w_[1:]):.4f}")

w_orig = unscale_weights(ours.w_, sc.mean_, sc.std_)
print("\ncoefficients in original units (true: -1.0, -0.06, 0.03, 0.5):")
for name, v in zip(["intercept"] + FEATURES, w_orig):
    print(f"  {name:15s} {v:8.4f}")

p_te = ours.predict_proba(Zte)
acc = float(np.mean((p_te >= 0.5) == yte))
ll = log_loss(yte, p_te)
print(f"\ntest accuracy = {acc:.4f}   test log loss = {ll:.4f}")
pd.DataFrame({"y_true": yte.astype(int), "p_churn": p_te}).to_csv("outputs/churn_test_scores.csv", index=False)

json.dump({"max_diff_sklearn_noreg": d0, "max_diff_sklearn_l2": d1, "test_accuracy": acc,
           "test_log_loss": ll, "coef_original_units": w_orig.tolist()},
          open("outputs/task4_logreg.json", "w"), indent=2)
