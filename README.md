# Machine Learning Mathematics and Python Implementation

A complete, self-contained lab that implements the core mathematics of
supervised learning **from scratch in NumPy**, validates every component
against scikit-learn, and finishes with a production-style pipeline,
CLI, and unit-test suite.

Scenario. You have joined the analytics team of Northwind Utilities, a regional energy and broadband provider. The team uses scikit-learn in production. Your lead wants every new engineer to understand what the library does underneath before shipping models. You will build a small, tested Python package called mlmath. It implements the core algorithms of classical machine learning from their mathematical definitions:

linear algebra diagnostics
least squares
gradient descent
logistic regression
evaluation metrics
ridge regularisation with cross-validation
PCA
You will verify each implementation numerically against scikit-learn. Then you will build, tune, persist and serve a production-style churn model from the command line.

What you will build

Two synthetic datasets: house prices for regression and customer churn for classification. 

The scikit-learn digits dataset (bundled with the library) is used for PCA.
A Python package mlmath/ with from-scratch implementations, each checked against scikit-learn.

Analysis scripts that save JSON results and PNG plots to outputs/. Plots use the Agg backend and savefig (never plt.show).

A tuned scikit-learn Pipeline, saved with joblib, and a CLI scoring tool.

A pytest suite and a final validation script.


## Environment

- Python **3.12.x** (tested on 3.12.3 / Ubuntu 24.04)
- Pinned: `numpy==1.26.4`, `scipy==1.13.1`, `pandas==2.2.2`,
  `scikit-learn==1.5.1`, `matplotlib==3.8.4`, `joblib==1.4.2`, `pytest==8.2.2`
