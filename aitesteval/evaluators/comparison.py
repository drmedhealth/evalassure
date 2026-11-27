"""Champion/challenger comparison (A/B with significance tests) and model-compression analysis."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.context import EvalContext
from ..core.registry import Evaluator, register_evaluator
from ..core.types import EvalResult
from ..models.ml import compress, model_size_bytes
from . import metrics as M
from .fairness import fairness_summary, group_rates
from .ml_performance import classification_metrics


def _latency_ms_per_1k(ctx: EvalContext, model: Any, repeats: int = 5) -> float:
    X = ctx.splits["test"].X
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        ctx.predict("test", X=X, model=model)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best * 1000 / max(1, len(X))


def _min_di(ctx: EvalContext, model: Any) -> dict[str, float]:
    test = ctx.splits["test"]
    if ctx.task != "classification" or not test.sensitive:
        return {}
    pred = ctx.predict("test", model=model)
    out = {}
    for attr, g in test.sensitive.items():
        s = fairness_summary(group_rates(test.y, pred, g, ctx.classes[-1]))
        if s:
            out[attr] = s["disparate_impact_ratio"]
    return out


def _group_acc(ctx: EvalContext, model: Any) -> dict[str, float]:
    test = ctx.splits["test"]
    pred = ctx.predict("test", model=model)
    out = {}
    for attr, g in test.sensitive.items():
        for v in np.unique(g):
            m = g == v
            out[f"{attr}={v}"] = float((pred[m] == test.y[m]).mean())
    return out


@register_evaluator("comparison")
class ComparisonEvaluator(Evaluator):
    tasks = ("classification", "regression", "llm")
    description = "Champion vs challenger: metric deltas with paired bootstrap CI, McNemar test, fairness & latency deltas, promotion decision."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> str | None:
        ch = ctx.challenger
        if ch is None:
            return "no `challenger:` model in scenario"
        if ctx.task == "llm":
            return self._llm(ctx, result, ch)
        y = ctx.splits["test"].y
        pa, pb = ctx.predict("test"), ctx.predict("test", model=ch)
        if ctx.task == "regression":
            sa, sb = -np.abs(pa - y), -np.abs(pb - y)  # higher is better
            primary = "neg_abs_error"
            m = {"champion_mae": float(-sa.mean()), "challenger_mae": float(-sb.mean())}
        else:
            sa, sb = (pa == y).astype(float), (pb == y).astype(float)
            primary = "accuracy"
            ma, mb = classification_metrics(ctx, "test"), classification_metrics(ctx, "test", model=ch)
            m = {}
            for k in ("accuracy", "balanced_accuracy", "f1", "recall", "precision", "roc_auc", "ece"):
                if k in ma and k in mb:
                    m[f"champion_{k}"], m[f"challenger_{k}"], m[f"delta_{k}"] = ma[k], mb[k], mb[k] - ma[k]
            m["mcnemar_p"] = M.mcnemar_p(sa.astype(bool), sb.astype(bool))
            m["agreement_rate"] = float((pa == pb).mean())
        delta, lo, hi = M.paired_bootstrap_delta(sa, sb, n_boot=int(self.cfg("bootstrap", 1000)))
        m.update({f"delta_{primary}_mean": delta, "delta_ci95": [lo, hi]})
        m["champion_latency_ms_per_1k"] = _latency_ms_per_1k(ctx, ctx.model)
        m["challenger_latency_ms_per_1k"] = _latency_ms_per_1k(ctx, ch)
        m["champion_size_bytes"], m["challenger_size_bytes"] = model_size_bytes(ctx.model), model_size_bytes(ch)
        di_a, di_b = _min_di(ctx, ctx.model), _min_di(ctx, ch)
        if di_a:
            m["champion_disparate_impact"], m["challenger_disparate_impact"] = di_a, di_b
        fairness_regression = [a for a in di_a if di_b.get(a, 1) < min(0.8, di_a[a] - 0.05)]
        if fairness_regression:
            self.finding(result, "CHALLENGER_FAIRNESS_REGRESSION", "Challenger is less fair than the champion", "high",
                         f"disparate impact worsens on {fairness_regression}", tags=["comparison", "fairness"])
        m["decision"] = self._decide(lo, hi, bool(fairness_regression))
        result.metrics = m
        if m["decision"] == "KEEP_CHAMPION":
            self.finding(result, "CHALLENGER_WORSE", "Challenger is significantly worse than the champion", "medium",
                         f"delta {primary} = {delta:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})", tags=["comparison"])
        elif m["decision"] == "INCONCLUSIVE":
            self.finding(result, "COMPARISON_INCONCLUSIVE", "No statistically significant difference", "info",
                         f"delta {primary} = {delta:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}); collect more data or prefer the cheaper model",
                         tags=["comparison"])
        return None

    @staticmethod
    def _decide(lo: float, hi: float, fairness_regression: bool) -> str:
        if fairness_regression:
            return "KEEP_CHAMPION"
        if lo > 0:
            return "PROMOTE_CHALLENGER"
        if hi < 0:
            return "KEEP_CHAMPION"
        return "INCONCLUSIVE"

    def _llm(self, ctx: EvalContext, result: EvalResult, ch: Any) -> None:
        ra, rb = ctx.llm_responses(), ctx.llm_responses(model=ch)
        cases = [c for c in ctx.llm_cases if c.get("reference")]
        sa = np.array([M.token_f1(ra[c["id"]].text, c["reference"]) for c in cases])
        sb = np.array([M.token_f1(rb[c["id"]].text, c["reference"]) for c in cases])
        delta, lo, hi = M.paired_bootstrap_delta(sa, sb)
        la = [r.latency_s for r in ra.values()]
        lb = [r.latency_s for r in rb.values()]
        result.metrics = {"champion_token_f1": float(sa.mean()), "challenger_token_f1": float(sb.mean()),
                          "delta_token_f1": delta, "delta_ci95": [lo, hi],
                          "champion_latency_p95_s": M.percentile(la, 95), "challenger_latency_p95_s": M.percentile(lb, 95),
                          "champion_output_tokens": sum(r.output_tokens for r in ra.values()),
                          "challenger_output_tokens": sum(r.output_tokens for r in rb.values()),
                          "decision": self._decide(lo, hi, False)}
        if result.metrics["decision"] == "KEEP_CHAMPION":
            self.finding(result, "CHALLENGER_WORSE", "Challenger LLM answers are significantly worse", "medium",
                         f"delta token_f1={delta:+.3f} (95% CI {lo:+.3f}..{hi:+.3f})", tags=["comparison"])
        return None


@register_evaluator("compression")
class CompressionEvaluator(Evaluator):
    tasks = ("classification",)
    description = "Quantization / pruning / distillation trade-off: size, latency, accuracy, fidelity, per-group degradation."

    DEFAULT_METHODS = [{"method": "quantize", "bits": 8}, {"method": "quantize", "bits": 4},
                       {"method": "quantize", "bits": 2}, {"method": "prune", "sparsity": 0.5},
                       {"method": "prune", "sparsity": 0.9}]

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        variants = []
        methods = ctx.config.get("compression_methods_override") or self.cfg("methods", self.DEFAULT_METHODS)
        for spec in methods:
            label = spec.get("label") or f"{spec['method']}-" + "-".join(f"{k}{v}" for k, v in spec.items() if k != "method")
            try:
                variants.append((label, compress(ctx.model, **spec)))
            except TypeError as exc:
                result.artifacts.setdefault("skipped_methods", []).append(f"{label}: {exc}")
        for label, model in ctx.config.get("extra_compressed", []):  # user-supplied compressed models
            variants.append((label, model))
        if not variants:
            return None
        budget = {"max_accuracy_drop": 0.01, "max_group_accuracy_drop": 0.02, "min_agreement": 0.97,
                  **self.config.get("budget", {})}
        result.metrics.update(self.analyse(ctx, variants, budget))
        for v in result.metrics["variants"]:
            if not v["within_budget"]:
                sev = "high" if v["accuracy_drop"] > 3 * budget["max_accuracy_drop"] else "medium"
                self.finding(result, "COMPRESSION_DEGRADATION", f"'{v['label']}' exceeds the quality budget", sev,
                             f"{v['label']}: accuracy drop {v['accuracy_drop']:+.4f}, worst group drop {v['worst_group_drop']:+.4f}, "
                             f"agreement {v['agreement']:.3f} at {v['compression_ratio']:.1f}x smaller",
                             f"{v['label']}.accuracy_drop", v["accuracy_drop"], budget["max_accuracy_drop"],
                             ["compression"])
            elif v["worst_group_drop"] > 2 * max(v["accuracy_drop"], 0.002):
                self.finding(result, "COMPRESSION_DISPARATE_DEGRADATION",
                             f"'{v['label']}' hurts some groups more than others", "low",
                             f"overall drop {v['accuracy_drop']:+.4f} vs worst group {v['worst_group']} {v['worst_group_drop']:+.4f}",
                             tags=["compression", "fairness"])
        if result.metrics.get("recommended") is None:
            self.finding(result, "NO_COMPRESSION_WITHIN_BUDGET", "No compressed variant meets the quality budget", "medium",
                         tags=["compression"])

    @staticmethod
    def analyse(ctx: EvalContext, variants: list, budget: dict) -> dict:
        y = ctx.splits["test"].y
        base_pred = ctx.predict("test")
        base_score = float((base_pred == y).mean()) if ctx.task != "regression" else 0.0
        base_size = model_size_bytes(ctx.model)
        base_lat = _latency_ms_per_1k(ctx, ctx.model)
        base_groups = _group_acc(ctx, ctx.model) if ctx.task != "regression" else {}
        base_proba = ctx.predict_proba("test")
        rows = []
        for label, model in variants:
            pred = ctx.predict("test", model=model)
            acc = float((pred == y).mean())
            groups = _group_acc(ctx, model)
            drops = {g: base_groups[g] - groups[g] for g in base_groups}
            worst = max(drops, key=drops.get) if drops else None
            proba = ctx.predict_proba("test", model=model)
            size = model_size_bytes(model)
            row = {"label": label, "size_bytes": size, "compression_ratio": base_size / max(size, 1),
                   "latency_ms_per_1k": _latency_ms_per_1k(ctx, model), "accuracy": acc,
                   "accuracy_drop": base_score - acc, "agreement": float((pred == base_pred).mean()),
                   "mean_abs_proba_shift": float(np.abs(proba - base_proba).mean()) if proba is not None and base_proba is not None else None,
                   "worst_group": worst, "worst_group_drop": drops.get(worst, 0.0) if worst else 0.0}
            row["speedup"] = base_lat / max(row["latency_ms_per_1k"], 1e-9)
            row["within_budget"] = (row["accuracy_drop"] <= budget["max_accuracy_drop"]
                                    and row["worst_group_drop"] <= budget["max_group_accuracy_drop"]
                                    and row["agreement"] >= budget["min_agreement"])
            rows.append(row)
        ok = [r for r in rows if r["within_budget"]]
        best = max(ok, key=lambda r: r["compression_ratio"]) if ok else None
        # Pareto frontier on (size, accuracy)
        pareto = [r["label"] for r in rows
                  if not any(o["size_bytes"] <= r["size_bytes"] and o["accuracy"] >= r["accuracy"]
                             and (o["size_bytes"] < r["size_bytes"] or o["accuracy"] > r["accuracy"]) for o in rows)]
        return {"baseline_size_bytes": base_size, "baseline_accuracy": base_score, "baseline_latency_ms_per_1k": base_lat,
                "variants": rows, "pareto_frontier": pareto, "recommended": best["label"] if best else None,
                "budget": budget}
