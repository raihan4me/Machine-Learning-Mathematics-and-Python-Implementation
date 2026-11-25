"""Task 6: bias-variance trade-off, ridge path, k-fold model selection."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from mlmath.linreg import add_bias, mse
from mlmath.model_selection import polynomial_features, ridge_fit, cross_val_mse

rng = np.random.default_rng(42)
truth = lambda x: np.sin(np.pi * x)
x = rng.uniform(-1, 1, 40); y = truth(x) + rng.normal(0, 0.3, 40)
xt = rng.uniform(-1, 1, 2000); yt = truth(xt) + rng.normal(0, 0.3, 2000)

print("irreducible error (noise variance) = 0.09")
print(f"\n{'degree':>6s} {'train_mse':>10s} {'cv_mse':>10s} {'test_mse':>10s}")
deg_res = {}
for d in [1, 3, 5, 9, 15]:
    P, Pt = polynomial_features(x, d), polynomial_features(xt, d)
    w = ridge_fit(add_bias(P), y, 1e-8)
    r = {
        "train_mse": mse(y, add_bias(P) @ w),
        "cv_mse": cross_val_mse(P, y, 1e-8),
        "test_mse": mse(yt, add_bias(Pt) @ w),
    }
    deg_res[str(d)] = r
    print(f"{d:6d} {r['train_mse']:10.4f} {r['cv_mse']:10.4f} {r['test_mse']:10.4f}")

best_deg = int(min(deg_res, key=lambda k: deg_res[k]["cv_mse"]))
print(f"best degree by 5-fold CV (lambda~0): {best_deg}")

print("\n--- ridge path, degree 15 ---")
P15, P15t = polynomial_features(x, 15), polynomial_features(xt, 15)
lams = np.logspace(-4, 1, 15)
cv_scores, norms = [], []
print(f"{'lambda':>10s} {'cv_mse':>10s} {'||w||':>10s}")
for lam in lams:
    cv = cross_val_mse(P15, y, lam)
    nrm = float(np.linalg.norm(ridge_fit(add_bias(P15), y, lam)[1:]))
    cv_scores.append(cv); norms.append(nrm)
    print(f"{lam:10.1e} {cv:10.4f} {nrm:10.4f}")

best_lam = float(lams[int(np.argmin(cv_scores))])
w_best = ridge_fit(add_bias(P15), y, best_lam)
test_best = mse(yt, add_bias(P15t) @ w_best)
monotone = all(norms[i + 1] <= norms[i] * (1 + 1e-6) for i in range(len(norms) - 1))
print(f"best lambda = {best_lam:.1e}  test MSE (degree 15 + ridge) = {test_best:.4f}")
print(f"||w|| non-increasing in lambda: {monotone}")

P5 = polynomial_features(x, 5)
w_ours = ridge_fit(add_bias(P5), y, 0.5)
skr = Ridge(alpha=0.5).fit(P5, y)
diff = float(np.max(np.abs(w_ours - np.r_[skr.intercept_, skr.coef_])))
print(f"\nmax |ours - sklearn Ridge(alpha=0.5)| (degree 5) = {diff:.2e}")

grid = np.linspace(-1, 1, 300)
plt.figure(figsize=(7, 4))
plt.scatter(x, y, s=12, color="k", label="train")
plt.plot(grid, truth(grid), "g--", label="truth")
for d in [1, 3, 15]:
    w = ridge_fit(add_bias(polynomial_features(x, d)), y, 1e-8)
    plt.plot(grid, add_bias(polynomial_features(grid, d)) @ w, label=f"deg {d}")
plt.plot(grid, add_bias(polynomial_features(grid, 15)) @ w_best, lw=2, label="deg 15 + ridge")
plt.ylim(-2, 2); plt.legend(); plt.tight_layout(); plt.savefig("outputs/bias_variance.png", dpi=100)

json.dump({
    "degree_results": deg_res,
    "best_degree_cv": best_deg,
    "best_lambda": best_lam,
    "test_mse_best": test_best,
    "norm_monotone": monotone,
    "max_diff_sklearn": diff,
}, open("outputs/task6_ridge.json", "w"), indent=2)
print("saved outputs/bias_variance.png")
