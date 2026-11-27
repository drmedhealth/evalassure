"""EvalContext: everything an evaluator needs (data splits, model, LLM cases, workflow)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray | None
    sensitive: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, idx: np.ndarray) -> "Split":
        return Split(
            X=self.X[idx],
            y=None if self.y is None else self.y[idx],
            sensitive={k: v[idx] for k, v in self.sensitive.items()},
        )


@dataclass
class EvalContext:
    """Shared state for one evaluation run.

    task: "classification" | "regression" | "llm" | "automation"
    """

    task: str
    model: Any = None
    splits: dict[str, Split] = field(default_factory=dict)  # train / val / test / production
    feature_names: list[str] = field(default_factory=list)
    raw_data: Split | None = None  # full dataset before splitting (for data-quality checks)
    trainer: Callable[..., Any] | None = None  # trainer(X, y, **overrides) -> fitted model
    trainer_params: dict[str, Any] = field(default_factory=dict)
    challenger: Any = None
    # LLM
    llm_cases: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | None = None
    # automation
    automation_cases: list[dict[str, Any]] = field(default_factory=list)
    # governance metadata
    model_card: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    applied_fixes: list[str] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def clone(self, **changes: Any) -> "EvalContext":
        """Shallow copy with a fresh prediction cache (use after swapping model/data)."""
        new = copy.copy(self)
        new.cache = {}
        new.splits = dict(self.splits)
        new.applied_fixes = list(self.applied_fixes)
        for k, v in changes.items():
            setattr(new, k, v)
        return new

    def split(self, name: str) -> Split | None:
        return self.splits.get(name)

    @property
    def classes(self) -> np.ndarray:
        ys = [s.y for s in self.splits.values() if s.y is not None]
        return np.unique(np.concatenate(ys)) if ys else np.array([])

    @property
    def is_binary(self) -> bool:
        return len(self.classes) == 2

    @property
    def retrainable(self) -> bool:
        return self.trainer is not None

    def retrain(self, X: np.ndarray | None = None, y: np.ndarray | None = None, **overrides: Any) -> Any:
        if self.trainer is None:
            raise RuntimeError("Model is not retrainable (bring-your-own model). Apply the recommendation manually.")
        tr = self.splits["train"]
        params = {**self.trainer_params, **overrides}
        return self.trainer(tr.X if X is None else X, tr.y if y is None else y, **params)

    # ----------------------------------------------------------- predictions
    def _groups_for(self, split: Split) -> np.ndarray | None:
        attr = getattr(self.model, "group_attribute", None)
        return split.sensitive.get(attr) if attr else None

    def predict(self, split: str = "test", X: np.ndarray | None = None, model: Any = None) -> np.ndarray:
        model = model or self.model
        s = self.splits[split]
        key = ("pred", id(model), split) if X is None else None
        if key and key in self.cache:
            return self.cache[key]
        Xi = s.X if X is None else X
        if getattr(model, "needs_groups", False):
            out = np.asarray(model.predict(Xi, groups=self._groups_for(s)))
        else:
            out = np.asarray(model.predict(Xi))
        if key:
            self.cache[key] = out
        return out

    def predict_proba(self, split: str = "test", X: np.ndarray | None = None, model: Any = None) -> np.ndarray | None:
        model = model or self.model
        if not hasattr(model, "predict_proba"):
            return None
        s = self.splits[split]
        key = ("proba", id(model), split) if X is None else None
        if key and key in self.cache:
            return self.cache[key]
        out = np.asarray(model.predict_proba(s.X if X is None else X), dtype=float)
        if key:
            self.cache[key] = out
        return out

    def positive_scores(self, split: str = "test", model: Any = None) -> np.ndarray | None:
        p = self.predict_proba(split, model=model)
        if p is None:
            return None
        return p[:, -1] if p.ndim == 2 else p

    # ------------------------------------------------------------ LLM helpers
    def llm_responses(self, model: Any = None) -> dict[str, Any]:
        """Generate (once) and cache the model's response to every LLM case."""
        from ..models.llm import build_prompt  # local import avoids a cycle

        model = model or self.model
        key = ("llm", id(model))
        if key in self.cache:
            return self.cache[key]
        out: dict[str, Any] = {}
        for case in self.llm_cases:
            prompt = build_prompt(case)
            try:
                out[case["id"]] = model.generate(prompt, system=self.system_prompt, context=case.get("context"))
            except Exception as exc:  # recorded so evaluators can report error rates
                from ..models.llm import LLMResponse

                out[case["id"]] = LLMResponse(text="", latency_s=0.0, error=f"{type(exc).__name__}: {exc}")
        self.cache[key] = out
        return out
