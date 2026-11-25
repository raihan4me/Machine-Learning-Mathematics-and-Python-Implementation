"""Generate the lab datasets with known ground-truth parameters."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = Path("data")
out.mkdir(exist_ok=True)

# Housing: price = 50 + 2*size + 10*rooms - 0.8*age - 3*dist + N(0, 15^2)
n = 1000
size = rng.uniform(40, 220, n)
rooms = rng.integers(1, 7, n)
age = rng.uniform(0, 60, n)
dist = rng.uniform(0.5, 30, n)
noise = rng.normal(0, 15, n)
price = 50 + 2.0 * size + 10 * rooms - 0.8 * age - 3.0 * dist + noise
housing = pd.DataFrame({
    "size_m2": size.round(1), "rooms": rooms, "age_years": age.round(1),
    "distance_km": dist.round(2), "price_k": price.round(2),
})
missing_idx = rng.choice(n, size=12, replace=False)
housing.loc[missing_idx, "age_years"] = np.nan
housing.to_csv(out / "housing.csv", index=False)

# Churn. True logit: -1.0 - 0.06*tenure + 0.03*charge + 0.5*calls
m = 2000
tenure = rng.uniform(0, 72, m)
charge = rng.uniform(20, 120, m)
calls = rng.poisson(2.0, m)
logit = -1.0 - 0.06 * tenure + 0.03 * charge + 0.5 * calls
p = 1.0 / (1.0 + np.exp(-logit))
churned = (rng.uniform(0, 1, m) < p).astype(int)
churn = pd.DataFrame({
    "tenure_months": tenure.round(1), "monthly_charge": charge.round(2),
    "support_calls": calls, "churned": churned,
})
churn.to_csv(out / "churn.csv", index=False)

print(f"housing: {housing.shape}, missing age_years: {housing['age_years'].isna().sum()}")
print(f"churn:   {churn.shape}, churn rate: {churn['churned'].mean():.3f}")
