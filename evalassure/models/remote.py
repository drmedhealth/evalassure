"""Remote ML model: test a model that is already deployed behind an HTTP endpoint.

Request (POST, JSON):
    {"feature_names": ["age", "income", "region=north", ...],
     "instances": [[34, 52000, 1, ...], ...]}          # missing values are sent as null
Response (JSON), either or both of:
    {"predictions": [0, 1, ...]}
    {"probabilities": [[0.8, 0.2], ...]}               # columns in sorted class order

Features are the numeric matrix evalassure builds from your CSV (categorical columns
one-hot encoded as `column=value`). The model is treated as pre-trained: retraining
fixes are skipped, and post-processing fixes (threshold, calibration, fairness) wrap it.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any, Callable

import numpy as np


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


def _clean(v: float) -> float | None:
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


class RemoteModel:
    """predict() only. Use RemoteProbaModel when the endpoint returns probabilities."""

    def __init__(self, url: str, feature_names: list[str], classes: np.ndarray | None, api_key: str | None = None,
                 timeout_s: float = 30, batch_size: int = 500, follow_redirects: bool = True,
                 max_requests: int = 2000, name: str | None = None,
                 url_check: Callable[[str], None] | None = None) -> None:
        if url_check:
            url_check(url)
        self.url, self.feature_names, self.api_key = url, list(feature_names), api_key
        self.timeout, self.batch_size, self.max_requests = timeout_s, batch_size, max_requests
        self.classes_ = classes
        self.name = name or "remote-endpoint"
        self.requests_made = 0
        self._opener = urllib.request.build_opener() if follow_redirects else urllib.request.build_opener(_NoRedirect)

    def _call(self, X: np.ndarray) -> dict[str, Any]:
        preds, probas = [], []
        for start in range(0, len(X), self.batch_size):
            if self.requests_made >= self.max_requests:
                raise RuntimeError(f"remote endpoint request budget ({self.max_requests}) exhausted")
            chunk = X[start:start + self.batch_size]
            body = json.dumps({"feature_names": self.feature_names,
                               "instances": [[_clean(v) for v in row] for row in chunk.tolist()]}).encode()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            self.requests_made += 1
            with self._opener.open(req, timeout=self.timeout) as r:
                data = json.loads(r.read(20_000_000))
            if "predictions" in data:
                preds.extend(data["predictions"])
            if "probabilities" in data:
                probas.extend(data["probabilities"])
        out: dict[str, Any] = {}
        if preds:
            out["predictions"] = np.asarray(preds)
        if probas:
            out["probabilities"] = np.asarray(probas, dtype=float)
        if not out:
            raise ValueError("endpoint response needs 'predictions' and/or 'probabilities'")
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        data = self._call(np.asarray(X, dtype=float))
        if "predictions" in data:
            return data["predictions"]
        p = data["probabilities"]
        return self.classes_[p.argmax(axis=1)] if self.classes_ is not None else p.argmax(axis=1)


class RemoteProbaModel(RemoteModel):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        data = self._call(np.asarray(X, dtype=float))
        p = data["probabilities"]
        return p if p.ndim == 2 else np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return self.classes_[p.argmax(axis=1)] if self.classes_ is not None else p.argmax(axis=1)


def build_remote_model(spec: dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                       feature_names: list[str], task: str) -> Any:
    """Probe the endpoint with a few rows to see whether it returns probabilities."""
    classes = np.unique(y_train) if task == "classification" else None
    kwargs = {k: spec[k] for k in ("api_key", "timeout_s", "batch_size", "follow_redirects", "max_requests",
                                   "name", "url_check") if k in spec}
    probe = RemoteModel(spec["endpoint"], feature_names, classes, **kwargs)
    data = probe._call(np.asarray(X_train[:5], dtype=float))
    use_proba = task == "classification" and "probabilities" in data
    cls = RemoteProbaModel if use_proba else RemoteModel
    model = cls(spec["endpoint"], feature_names, classes, **{k: v for k, v in kwargs.items() if k != "url_check"})
    model.requests_made = probe.requests_made
    return model
