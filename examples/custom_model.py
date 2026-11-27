"""Example "bring your own model": any class with fit / predict / predict_proba works.

Referenced from scenarios/15_bring_your_own_model_csv.yaml as
    model:
      python: examples/custom_model.py:GradientBoostedStumps

To test a scikit-learn model instead, point at a factory, e.g. in my_models.py:

    from sklearn.ensemble import RandomForestClassifier
    def make_rf(**params):
        return RandomForestClassifier(n_estimators=200, **params)

and use `python: my_models.py:make_rf`. aitesteval calls .fit(X_train, y_train) for you
(set `fit: false` if your model is already trained, or use `pickle: model.pkl`).
"""
from __future__ import annotations

import numpy as np


class GradientBoostedStumps:
    """Minimal gradient boosting (logistic loss) with depth-1 trees. Handles NaN by median imputation."""

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.3, n_thresholds: int = 16) -> None:
        self.n_estimators, self.learning_rate, self.n_thresholds = n_estimators, learning_rate, n_thresholds

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nan_to_num(np.nanmedian(X, axis=0))
        X = self._impute(X)
        self.classes_ = np.unique(y)
        t = (np.asarray(y) == self.classes_[-1]).astype(float)
        w = np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight, float)
        p0 = np.clip(np.average(t, weights=w), 1e-6, 1 - 1e-6)
        self.init_ = np.log(p0 / (1 - p0))
        F = np.full(len(t), self.init_)
        self.stumps_ = []
        cand = [np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, self.n_thresholds))) for j in range(X.shape[1])]
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            g = (t - p) * w  # negative gradient
            h = np.maximum(p * (1 - p) * w, 1e-9)
            best = None
            for j, ths in enumerate(cand):
                for th in ths:
                    left = X[:, j] <= th
                    gl, hl = g[left].sum(), h[left].sum()
                    gr, hr = g[~left].sum(), h[~left].sum()
                    gain = gl * gl / (hl + 1) + gr * gr / (hr + 1)
                    if best is None or gain > best[0]:
                        best = (gain, j, th, gl / (hl + 1), gr / (hr + 1))
            _, j, th, vl, vr = best
            self.stumps_.append((j, th, vl, vr))
            F += self.learning_rate * np.where(X[:, j] <= th, vl, vr)
        return self

    def _impute(self, X):
        X = np.array(X, dtype=float, copy=True)
        m = np.isnan(X)
        if m.any():
            X[m] = np.take(self.medians_, np.where(m)[1])
        return X

    def decision_function(self, X):
        X = self._impute(X)
        F = np.full(len(X), self.init_)
        for j, th, vl, vr in self.stumps_:
            F += self.learning_rate * np.where(X[:, j] <= th, vl, vr)
        return F

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, self.classes_[-1], self.classes_[0])

    def size_bytes(self) -> int:
        return len(self.stumps_) * 4 * 8
