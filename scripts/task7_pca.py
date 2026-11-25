"""Task 7: PCA on the handwritten digits dataset."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SkPCA
from mlmath.pca import PCA, pca_eig, n_components_for_variance

digits = load_digits()
X, labels = digits.data, digits.target
n, p = X.shape
print(f"digits: n={n}, p={p}")

full = PCA().fit(X)
ev_eig, _ = pca_eig(X)
eig_diff = float(np.max(np.abs(full.explained_variance_ - ev_eig)) / ev_eig.max())
print(f"SVD vs covariance-eigh eigenvalues, max relative diff: {eig_diff:.2e}")

sk = SkPCA().fit(X)
ratio_diff = float(np.max(np.abs(full.explained_variance_ratio_ - sk.explained_variance_ratio_)))
cos = np.abs(np.sum(full.components_[:10] * sk.components_[:10], axis=1))
print(f"explained-variance ratio vs sklearn, max diff: {ratio_diff:.2e}")
print(f"|cos| between our and sklearn top-10 components: min = {cos.min():.8f}")

C = full.components_
orth = float(np.max(np.abs(C @ C.T - np.eye(C.shape[0]))))
print(f"orthonormality error max|V V^T - I|: {orth:.2e}")

n90 = n_components_for_variance(full.explained_variance_ratio_, 0.90)
n90_sk = int(SkPCA(n_components=0.90).fit(X).n_components_)
print(f"components for 90% variance: ours = {n90}, sklearn = {n90_sk}")
print("first 5 ratios:", np.round(full.explained_variance_ratio_[:5], 4))

print(f"\n{'k':>3s} {'recon_mse':>10s} {'predicted':>10s}")
worst = 0.0
for k in [2, 5, 10, n90, 40, 64]:
    pk = PCA(k).fit(X)
    rec = float(np.mean((X - pk.inverse_transform(pk.transform(X))) ** 2))
    pred = float((n - 1) / (n * p) * full.explained_variance_all_[k:].sum())
    if pred > 1e-9:
        worst = max(worst, abs(rec - pred) / pred)
    print(f"{k:3d} {rec:10.4f} {pred:10.4f}")
print(f"max relative mismatch (theory vs measurement): {worst:.2e}")

Z = PCA(2).fit(X).transform(X)
plt.figure(figsize=(6, 5))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=6)
plt.colorbar(sc, label="digit"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("outputs/pca_digits.png", dpi=100)

json.dump({"eig_svd_max_diff": eig_diff, "max_ratio_diff": ratio_diff, "min_abs_cos_top10": float(cos.min()),
           "orthonormal_err": orth, "n90_ours": n90, "n90_sklearn": n90_sk,
           "recon_identity_err": worst}, open("outputs/task7_pca.json", "w"), indent=2)
print("saved outputs/pca_digits.png")
