"""Plugin registries: add your own evaluator / fix with a decorator."""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from .context import EvalContext
from .types import EvalResult, Finding, Severity

EVALUATORS: dict[str, type["Evaluator"]] = {}
FIXES: dict[str, type] = {}


def register_evaluator(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        EVALUATORS[name] = cls
        return cls

    return deco


def register_fix(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.id = name
        FIXES[name] = cls
        return cls

    return deco


class Evaluator:
    """Base class. Subclasses implement `evaluate(ctx, result)` and fill `result`."""

    name = "base"
    tasks: tuple[str, ...] = ("classification", "regression", "llm", "automation")
    description = ""

    def __init__(self, **config: Any) -> None:
        self.config = config

    def cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def threshold(self, key: str, default: float | None = None) -> float | None:
        return self.config.get("thresholds", {}).get(key, default)

    def finding(self, result: EvalResult, code: str, title: str, severity: Severity | str, detail: str = "",
                metric: str | None = None, value: float | None = None, threshold: float | None = None,
                tags: list[str] | None = None) -> None:
        result.findings.append(Finding(
            code=code, title=title, severity=Severity(severity), evaluator=self.name, detail=detail,
            metric=metric, value=None if value is None else float(value),
            threshold=None if threshold is None else float(threshold), tags=tags or [],
        ))

    def run(self, ctx: EvalContext) -> EvalResult:
        result = EvalResult(evaluator=self.name)
        t0 = time.perf_counter()
        if ctx.task not in self.tasks:
            result.skipped = f"not applicable to task '{ctx.task}'"
            return result
        try:
            reason = self.evaluate(ctx, result)
            if isinstance(reason, str):
                result.skipped = reason
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.artifacts["traceback"] = traceback.format_exc(limit=6)
        result.duration_s = time.perf_counter() - t0
        return result

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> str | None:  # pragma: no cover
        raise NotImplementedError
