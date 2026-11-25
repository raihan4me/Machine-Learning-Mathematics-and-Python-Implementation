"""Task 8: tuned scikit-learn pipeline for churn, persisted with joblib."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlmath.prep import Standardizer
from mlmath.logreg import LogisticRegressionGD
from mlmath.metrics import roc_auc_rank

FEATURES = ["tenure_months", "monthly_charge", "support_calls"]
df = pd.read_csv("data/churn.csv")
X, y = df[FEATURES], df["churned"].to_numpy(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)

pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000, tol=1e-8))])
search = GridSearchCV(pipe, {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                      scoring="roc_auc", n_jobs=1)
search.fit(Xtr, ytr)
print(f"{'C':>8s} {'mean_cv_auc':>12s} {'std':>8s}")
for C, m, s in zip(search.cv_results_["param_clf__C"], search.cv_results_["mean_test_score"],
                   search.cv_results_["std_test_score"]):
    print(f"{C:8g} {m:12.4f} {s:8.4f}")
best = search.best_estimator_
best_C = float(search.best_params_["clf__C"])
p_te = best.predict_proba(Xte)[:, 1]
test_auc = float(roc_auc_score(yte, p_te))
print(f"best C = {best_C:g}   CV AUC = {search.best_score_:.4f}   held-out test AUC = {test_auc:.4f}")

# Cross-check against the from-scratch model using l2 = 1/(n*C)
sc = Standardizer().fit(Xtr.to_numpy(float))
ours = LogisticRegressionGD(lr=1.0, epochs=20000, l2=1.0 / (len(ytr) * best_C))
ours.fit(sc.transform(Xtr.to_numpy(float)), ytr)
p_ours = ours.predict_proba(sc.transform(Xte.to_numpy(float)))
max_prob_diff = float(np.max(np.abs(p_ours - p_te)))
print(f"from-scratch AUC = {roc_auc_rank(yte, p_ours):.4f}   max |p_ours - p_sklearn| = {max_prob_diff:.2e}")

Path("models").mkdir(exist_ok=True)
joblib.dump({"model": best, "features": FEATURES, "threshold": 0.5,
             "sklearn_version": sklearn.__version__}, "models/churn_model.joblib")
json.dump({"best_C": best_C, "cv_auc": float(search.best_score_), "test_auc": test_auc,
           "auc_ours": roc_auc_rank(yte, p_ours), "max_prob_diff": max_prob_diff},
          open("outputs/churn_metrics.json", "w"), indent=2)
print("saved models/churn_model.joblib")
