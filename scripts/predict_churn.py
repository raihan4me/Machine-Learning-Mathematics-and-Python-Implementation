"""Score new customers with the persisted churn model."""
import argparse
import sys
import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Score customers for churn risk.")
    ap.add_argument("--model", default="models/churn_model.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    df = pd.read_csv(a.input)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        print(f"ERROR: input is missing required columns: {missing}", file=sys.stderr)
        return 2
    X = df[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.index[X.isna().any(axis=1)].tolist()
        print(f"ERROR: non-numeric or empty feature values in rows {bad}", file=sys.stderr)
        return 3
    thr = a.threshold if a.threshold is not None else bundle["threshold"]
    proba = bundle["model"].predict_proba(X)[:, 1]
    df["p_churn"] = proba.round(4)
    df["flag_for_retention"] = (proba >= thr).astype(int)
    df.to_csv(a.output, index=False)
    print(f"scored {len(df)} customers, flagged {int(df['flag_for_retention'].sum())} at threshold {thr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
