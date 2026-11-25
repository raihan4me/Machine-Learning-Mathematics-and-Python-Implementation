"""Task 3: gradient check, divergence, Hessian-based learning-rate bound."""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlmath.prep import Standardizer
from mlmath.linreg import add_bias, train_test_split_idx, fit_normal_equation
from mlmath.gd import (mse_loss, mse_gradient, numerical_gradient,
                       batch_gradient_descent, unscale_weights)

FEATURES = ["size_m2", "rooms", "age_years", "distance_km"]
df = pd.read_csv("data/housing_clean.csv")
X = df[FEATURES].to_numpy(float)
y = df["price_k"].to_numpy(float)
tr, _ = train_test_split_idx(len(y), 0.2, seed=7)      # same split as Task 2
Xtr, ytr = X[tr], y[tr]
sc = Standardizer().fit(Xtr)
Ztr = add_bias(sc.transform(Xtr))

# 1. Gradient check
w0 = np.random.default_rng(0).normal(size=Ztr.shape[1])
g_an = mse_gradient(Ztr, ytr, w0)
g_num = numerical_gradient(lambda w: mse_loss(Ztr, ytr, w), w0)
grad_rel = float(np.linalg.norm(g_an - g_num) / np.linalg.norm(g_an + g_num))
print(f"gradient check relative error: {grad_rel:.2e}")

# 2. Raw (unscaled) features: a classic failure. The RuntimeWarning is NOT suppressed here.
print("\n--- raw (unscaled) features, lr=0.01 ---")
_, hist_raw = batch_gradient_descent(add_bias(Xtr), ytr, lr=0.01, epochs=500)
raw_diverged = not np.isfinite(hist_raw[-1])
print(f"stopped after {len(hist_raw)} epochs, last loss = {hist_raw[-1]}, diverged = {raw_diverged}")

# 3. Hessian eigenvalues give the stability bound
H = (2.0 / len(ytr)) * Ztr.T @ Ztr
eig = np.linalg.eigvalsh(H)
lr_max = float(2.0 / eig.max())
print(f"\nHessian eigenvalues (scaled data): {np.round(eig, 3)}")
print(f"stability bound: lr < 2/lambda_max = {lr_max:.4f}")
best_loss = mse_loss(Ztr, ytr, fit_normal_equation(Ztr, ytr))
print(f"optimal training MSE: {best_loss:.4f}")

lrs = [0.0005, 0.01, 0.1, 0.5, 0.9 * lr_max, 1.1 * lr_max]
sweep, histories = {}, {}
print(f"\n{'lr':>8s} {'epochs':>7s} {'final_loss':>14s}  status")
for lr in lrs:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the divergent run overflows on purpose
        _, hist = batch_gradient_descent(Ztr, ytr, lr=lr, epochs=5000, tol=1e-12)
    final = hist[-1]
    diverged = (not np.isfinite(final)) or final > hist[0]
    converged = (not diverged) and abs(final - best_loss) / best_loss < 1e-6
    status = "DIVERGED" if diverged else ("converged" if converged else "not yet converged")
    print(f"{lr:8.4f} {len(hist):7d} {final:14.4f}  {status}")
    sweep[f"{lr:.4g}"] = {"epochs": len(hist), "final_loss": float(final) if np.isfinite(final) else None,
                          "diverged": bool(diverged), "converged": bool(converged)}
    histories[lr] = np.array(hist)

# 4. Map back to original units and compare with Task 2
w_gd, _ = batch_gradient_descent(Ztr, ytr, lr=0.5, epochs=5000, tol=1e-12)
w_orig = unscale_weights(w_gd, sc.mean_, sc.std_)
w_ols = np.array(json.load(open("outputs/task2_ols.json"))["coef_normal_eq"])
max_diff = float(np.max(np.abs(w_orig - w_ols)))
print(f"\nGD (lr=0.5) weights in original units: {np.round(w_orig, 4)}")
print(f"max |w_gd - w_ols| = {max_diff:.2e}")

plt.figure(figsize=(7, 4))
for lr, h in histories.items():
    h = h[np.isfinite(h)]
    plt.semilogy(h[:300], label=f"lr={lr:.3g}")
plt.xlabel("epoch"); plt.ylabel("training MSE (log)"); plt.legend(); plt.title("Learning-rate sweep")
plt.tight_layout(); plt.savefig("outputs/gd_lr_sweep.png", dpi=100)

json.dump({"grad_rel_error": grad_rel, "raw_diverged": bool(raw_diverged), "lr_max": lr_max,
           "sweep": sweep, "max_diff_vs_ols": max_diff,
           "below_bound_converged": sweep[f"{0.9*lr_max:.4g}"]["converged"],
           "above_bound_diverged": sweep[f"{1.1*lr_max:.4g}"]["diverged"]},
          open("outputs/task3_gd.json", "w"), indent=2)
print("saved outputs/gd_lr_sweep.png")
