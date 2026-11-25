"""Task 2: OLS via the normal equation, residual orthogonality, collinearity."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mlmath.linreg import (add_bias, train_test_split_idx, fit_normal_equation,
                           fit_lstsq, predict, mse, r2_score)

FEATURES = ["size_m2", "rooms", "age_years", "distance_km"]
TRUE = [50.0, 2.0, 10.0, -0.8, -3.0]
df = pd.read_csv("data/housing_clean.csv")
X = df[FEATURES].to_numpy(float)
y = df["price_k"].to_numpy(float)
tr, te = train_test_split_idx(len(y), 0.2, seed=7)
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
Xb_tr, Xb_te = add_bias(Xtr), add_bias(Xte)

w_ne = fit_normal_equation(Xb_tr, ytr)
w_ls = fit_lstsq(Xb_tr, ytr)
sk = LinearRegression().fit(Xtr, ytr)
w_sk = np.r_[sk.intercept_, sk.coef_]

print(f"{'term':12s} {'true':>8s} {'normal_eq':>10s} {'lstsq':>10s} {'sklearn':>10s}")
for name, t, a, b, c in zip(["intercept"] + FEATURES, TRUE, w_ne, w_ls, w_sk):
    print(f"{name:12s} {t:8.2f} {a:10.4f} {b:10.4f} {c:10.4f}")
max_diff = float(np.max(np.abs(w_ne - w_sk)))
print(f"max |w_normal_eq - w_sklearn| = {max_diff:.2e}")

r = ytr - predict(Xb_tr, w_ne)
orth = float(np.linalg.norm(Xb_tr.T @ r) / np.linalg.norm(Xb_tr.T @ ytr))
print(f"mean training residual        = {r.mean():.2e}")
print(f"||Xb^T r|| / ||Xb^T y||       = {orth:.2e}")

yhat = predict(Xb_te, w_ne)
test_mse = mse(yte, yhat)
print(f"test MSE = {test_mse:.2f}  RMSE = {np.sqrt(test_mse):.2f}  R^2 = {r2_score(yte, yhat):.4f}")

print("\n--- collinearity experiment: add size_m2 * 2 as an extra column ---")
Xb_col = add_bias(np.column_stack([Xtr, 2.0 * Xtr[:, 0]]))
rank = int(np.linalg.matrix_rank(Xb_col))
print(f"rank = {rank} of {Xb_col.shape[1]} columns, cond(Xb^T Xb) = {np.linalg.cond(Xb_col.T @ Xb_col):.2e}")
try:
    w_bad = fit_normal_equation(Xb_col, ytr)
    print("np.linalg.solve returned (unreliable):", np.round(w_bad, 3))
except np.linalg.LinAlgError as exc:
    print("np.linalg.solve failed:", exc)
w_min = fit_lstsq(Xb_col, ytr)
size_sum = float(w_min[1] + 2.0 * w_min[5])
print(f"lstsq min-norm: size coef = {w_min[1]:.4f}, duplicate coef = {w_min[5]:.4f}")
print(f"size + 2*duplicate = {size_sum:.4f}  (OLS size coef = {w_ne[1]:.4f})")
print(f"max prediction difference vs original model = "
      f"{np.max(np.abs(Xb_col @ w_min - Xb_tr @ w_ne)):.2e}")

json.dump({"coef_normal_eq": w_ne.tolist(), "max_diff_sklearn": max_diff,
           "residual_orthogonality": orth, "test_rmse": float(np.sqrt(test_mse)),
           "test_r2": r2_score(yte, yhat), "collinear_rank": rank,
           "collinear_size_sum": size_sum},
          open("outputs/task2_ols.json", "w"), indent=2)
