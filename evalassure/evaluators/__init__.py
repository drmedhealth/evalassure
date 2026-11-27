"""Importing this package registers every built-in evaluator."""
from . import automation, comparison, data_checks, fairness, llm, ml_performance, robustness  # noqa: F401
from ..core.registry import EVALUATORS

__all__ = ["EVALUATORS"]
