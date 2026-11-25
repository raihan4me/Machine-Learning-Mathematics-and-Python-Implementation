"""Task 1: inspect, clean and analyse the housing design matrix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mlmath.prep import impute_median, Standardizer

FEATURES = ["size_m2", "rooms", "age_years", "distance_km"]
TARGET = "price_k"
Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv("data/housing.csv")
print("shape:", df.shape)
print("missing values per column:")
print(df.isna().sum().to_string())

# A single NaN poisons every entry of X^T X it touches
Xb_dirty = np.column_stack([np.ones(len(df)), df[FEATURES].to_numpy(float)])
print("X^T X contains NaN (dirty data):", bool(np.isnan(Xb_dirty.T @ Xb_dirty).any()))

clean, medians = impute_median(df, FEATURES)
print("medians used:", {k: round(v, 2) for k, v in medians.items()})
print("missing after imputation:", int(clean.isna().sum().sum()))
clean.to_csv("data/housing_clean.csv", index=False)

print("\nsummary statistics:")
print(clean.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
print("\ncorrelation with target:")
print(clean.corr()[TARGET].drop(TARGET).round(3).to_string())

X = clean[FEATURES].to_numpy(float)
Xb_raw = np.column_stack([np.ones(len(X)), X])
Xb_std = np.column_stack([np.ones(len(X)), Standardizer().fit_transform(X)])

rank = int(np.linalg.matrix_rank(Xb_raw))
cond_raw = float(np.linalg.cond(Xb_raw))
cond_std = float(np.linalg.cond(Xb_std))
XtX = Xb_std.T @ Xb_std
eig = np.linalg.eigvalsh(XtX)
print(f"\nrank(Xb) = {rank} of {Xb_raw.shape[1]} columns")
print(f"cond(Xb) raw          = {cond_raw:,.1f}")
print(f"cond(Xb) standardized = {cond_std:.3f}")
print(f"cond(X^T X) raw       = {cond_raw**2:,.0f}")
print("X^T X symmetric:", bool(np.allclose(XtX, XtX.T)))
print(f"min eigenvalue of X^T X (standardized): {eig.min():.1f}")
print("X^T X positive definite:", bool(eig.min() > 0))

json.dump({"rank": rank, "cond_raw": cond_raw, "cond_std": cond_std,
           "min_eig_std": float(eig.min()), "medians": medians},
          open("outputs/task1.json", "w"), indent=2)
