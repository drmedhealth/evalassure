"""Built-in numpy ML models, model wrappers used by auto-fixes, and compression utilities.

Any object with `predict(X)` (and optionally `predict_proba(X)`) works with evalassure,
including scikit-learn, XGBoost, LightGBM or PyTorch models wrapped in a small adapter.
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np


class _Preprocess:
    """Median imputation + standardisation, fitted on training data."""

    def fit(self, X: np.ndarray) -> "_Preprocess":
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        Xi = self._impute(X)
        self.mean_ = Xi.mean(axis=0)
        self.std_ = Xi.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        mask = np.isnan(X)
        if mask.any():
            X[mask] = np.take(self.medians_, np.where(mask)[1])
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._impute(X) - self.mean_) / self.std_


class LogisticRegressionNP:
    """Multinomial (softmax) logistic regression trained with full-batch gradient descent."""

    def __init__(self, lr: float = 0.5, epochs: int = 400, l2: float = 1e-3,
                 class_weight: str | dict | None = None, seed: int = 0, **_: Any) -> None:
        self.lr, self.epochs, self.l2, self.class_weight, self.seed = lr, epochs, l2, class_weight, seed

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticRegressionNP":
        self.pre_ = _Preprocess().fit(X)
        Z = self.pre_.transform(X)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        n, d = Z.shape
        k = len(self.classes_)
        Y = np.eye(k)[yi]
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        if self.class_weight == "balanced":
            counts = np.bincount(yi, minlength=k)
            w = w * (n / (k * np.maximum(counts, 1)))[yi]
        elif isinstance(self.class_weight, dict):
            w = w * np.array([self.class_weight.get(c, 1.0) for c in self.classes_.tolist()])[yi]
        w = w / w.mean()
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0, 0.01, (d, k))
        self.b_ = np.zeros(k)
        for _ in range(self.epochs):
            P = self._softmax(Z @ self.W_ + self.b_)
            G = (P - Y) * w[:, None] / n
            self.W_ -= self.lr * (Z.T @ G + self.l2 * self.W_)
            self.b_ -= self.lr * G.sum(axis=0)
        return self

    @staticmethod
    def _softmax(S: np.ndarray) -> np.ndarray:
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pre_.transform(X) @ self.W_ + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_bytes(self) -> int:
        return int(self.W_.astype(np.float32).nbytes + self.b_.astype(np.float32).nbytes)


class KNNClassifierNP:
    """k-nearest-neighbours (useful as a high-variance challenger / overfitting example)."""

    def __init__(self, k: int = 1, **_: Any) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierNP":
        self.pre_ = _Preprocess().fit(X)
        self.Xtr_ = self.pre_.transform(X)
        self.classes_ = np.unique(y)
        self.ytr_ = np.searchsorted(self.classes_, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.pre_.transform(X)
        out = np.zeros((len(Z), len(self.classes_)))
        for s in range(0, len(Z), 512):
            chunk = Z[s:s + 512]
            d = (chunk ** 2).sum(1)[:, None] - 2 * chunk @ self.Xtr_.T + (self.Xtr_ ** 2).sum(1)[None, :]
            nn = np.argpartition(d, kth=min(self.k, d.shape[1] - 1) - 1, axis=1)[:, :self.k]
            for c in range(len(self.classes_)):
                out[s:s + 512, c] = (self.ytr_[nn] == c).mean(axis=1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_bytes(self) -> int:
        return int(self.Xtr_.astype(np.float32).nbytes + self.ytr_.nbytes)


class RidgeRegressionNP:
    def __init__(self, alpha: float = 1.0, **_: Any) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionNP":
        self.pre_ = _Preprocess().fit(X)
        Z = self.pre_.transform(X)
        self.y_mean_ = float(np.mean(y))
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.W_ = np.linalg.solve(A, Z.T @ (y - self.y_mean_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pre_.transform(X) @ self.W_ + self.y_mean_

    def size_bytes(self) -> int:
        return int(self.W_.astype(np.float32).nbytes)


BUILTIN_MODELS = {
    "logreg": LogisticRegressionNP,
    "logistic_regression": LogisticRegressionNP,
    "knn": KNNClassifierNP,
    "ridge": RidgeRegressionNP,
    "linear_regression": RidgeRegressionNP,
}


def make_trainer(kind: str):
    cls = BUILTIN_MODELS[kind]

    def trainer(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, **params: Any):
        model = cls(**params)
        if sample_weight is not None:
            return model.fit(X, y, sample_weight=sample_weight)
        return model.fit(X, y)

    trainer.kind = kind  # type: ignore[attr-defined]
    return trainer


# ------------------------------------------------------------------ wrappers used by fixes

class _Wrapper:
    def __init__(self, base: Any) -> None:
        self.base = base

    @property
    def classes_(self) -> np.ndarray:
        return self.base.classes_

    def size_bytes(self) -> int:
        return model_size_bytes(self.base)

    def __getattr__(self, item: str) -> Any:  # delegate unknown attrs (e.g. pre_)
        if item == "base":
            raise AttributeError(item)
        return getattr(self.base, item)


class ThresholdModel(_Wrapper):
    """Binary classifier with a tuned decision threshold."""

    def __init__(self, base: Any, threshold: float) -> None:
        super().__init__(base)
        self.threshold = float(threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict_proba(X)

    def predict(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return np.where(p >= self.threshold, self.classes_[1], self.classes_[0])


class GroupThresholdModel(_Wrapper):
    """Post-processing fairness mitigation: per-group thresholds (needs group membership)."""

    needs_groups = True

    def __init__(self, base: Any, attribute: str, thresholds: dict[str, float], default: float = 0.5) -> None:
        super().__init__(base)
        self.group_attribute = attribute
        self.thresholds = thresholds
        self.default = default

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict_proba(X)

    def predict(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        if groups is None:
            t = np.full(len(p), self.default)
        else:
            t = np.array([self.thresholds.get(str(g), self.default) for g in groups])
        return np.where(p >= t, self.classes_[1], self.classes_[0])


class TemperatureScaledModel(_Wrapper):
    """Probability calibration by temperature scaling of log-probabilities."""

    def __init__(self, base: Any, temperature: float) -> None:
        super().__init__(base)
        self.temperature = float(temperature)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logp = np.log(np.clip(self.base.predict_proba(X), 1e-12, 1)) / self.temperature
        logp -= logp.max(axis=1, keepdims=True)
        e = np.exp(logp)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        if hasattr(self.base, "threshold") or getattr(self.base, "needs_groups", False):
            return self.base.predict(X, groups=groups) if getattr(self.base, "needs_groups", False) else self.base.predict(X)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_temperature(proba: np.ndarray, y_idx: np.ndarray) -> float:
    logp = np.log(np.clip(proba, 1e-12, 1))
    best_t, best_loss = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 120)):
        s = logp / t
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1)))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


# --------------------------------------------------------------------- compression

class QuantizedLogReg:
    """Symmetric per-tensor integer quantisation of a LogisticRegressionNP."""

    def __init__(self, base: LogisticRegressionNP, bits: int = 8) -> None:
        self.bits = bits
        self.pre_, self.classes_ = base.pre_, base.classes_
        qmax = 2 ** (bits - 1) - 1
        self.scale_ = float(np.abs(base.W_).max() / qmax) or 1.0
        self.Wq_ = np.clip(np.round(base.W_ / self.scale_), -qmax, qmax).astype(np.int8 if bits <= 8 else np.int16)
        self.b_ = base.b_.astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        S = self.pre_.transform(X) @ (self.Wq_.astype(float) * self.scale_) + self.b_
        return LogisticRegressionNP._softmax(S)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_bytes(self) -> int:
        return int(np.ceil(self.Wq_.size * self.bits / 8) + self.b_.nbytes + 4)


class PrunedLogReg:
    """Magnitude pruning: zero the smallest |w|, store as sparse (value + int16 index)."""

    def __init__(self, base: LogisticRegressionNP, sparsity: float = 0.5) -> None:
        self.sparsity = sparsity
        self.pre_, self.classes_, self.b_ = base.pre_, base.classes_, base.b_
        W = base.W_.copy()
        k = int(round(sparsity * W.size))
        if k > 0:
            cutoff = np.sort(np.abs(W).ravel())[k - 1]
            W[np.abs(W) <= cutoff] = 0.0
        self.W_ = W

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return LogisticRegressionNP._softmax(self.pre_.transform(X) @ self.W_ + self.b_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_bytes(self) -> int:
        nnz = int(np.count_nonzero(self.W_))
        return int(nnz * (4 + 2) + self.b_.astype(np.float32).nbytes)


def _unwrap(model: Any) -> Any:
    while isinstance(model, _Wrapper):
        model = model.base
    return model


def compress(model: Any, method: str, **params: Any) -> Any:
    base = _unwrap(model)
    if not isinstance(base, LogisticRegressionNP):
        raise TypeError(
            f"Built-in compression supports LogisticRegressionNP; got {type(base).__name__}. "
            "Provide your own compressed model via `compressed_models:` in the scenario.")
    if method == "quantize":
        return QuantizedLogReg(base, bits=int(params.get("bits", 8)))
    if method == "prune":
        return PrunedLogReg(base, sparsity=float(params.get("sparsity", 0.5)))
    raise ValueError(f"Unknown compression method '{method}' (use quantize | prune)")


def model_size_bytes(model: Any) -> int:
    if hasattr(model, "size_bytes"):
        return int(model.size_bytes())
    try:
        return len(pickle.dumps(model))
    except Exception:
        return -1
