"""Auto-fixes. Each fix transforms the EvalContext (retrain, wrap the model, add guardrails or
middleware); the runner re-evaluates and records before/after evidence."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.context import EvalContext, Split
from ..core.registry import FIXES, register_fix
from ..core.types import EvalResult
from ..models.automation import WorkflowWithMiddleware
from ..models.guardrails import GUARDS
from ..models.ml import GroupThresholdModel, TemperatureScaledModel, ThresholdModel, fit_temperature


class Fix:
    id = "fix"
    title = ""
    order = 100
    tasks: tuple[str, ...] = ()
    watch: list[tuple[str, str]] = []  # (evaluator, metric path) shown before/after

    def __init__(self, **config: Any) -> None:
        self.config = config

    def not_applicable(self, ctx: EvalContext, results: list[EvalResult]) -> str | None:
        if self.tasks and ctx.task not in self.tasks:
            return f"not for task '{ctx.task}'"
        return None

    def apply(self, ctx: EvalContext, results: list[EvalResult]) -> tuple[EvalContext, EvalContext, str]:
        """Return (baseline_ctx, fixed_ctx, description)."""
        raise NotImplementedError


def _metric(results: list[EvalResult], ev: str) -> dict:
    return next((r.metrics for r in results if r.evaluator == ev), {})


def _findings(results: list[EvalResult], code: str):
    return [f for r in results for f in r.findings if f.code == code]


def _drop_features(ctx: EvalContext, names: list[str]) -> EvalContext:
    keep = [i for i, n in enumerate(ctx.feature_names) if n not in names]
    splits = {k: Split(s.X[:, keep], s.y, s.sensitive) for k, s in ctx.splits.items()}
    raw = Split(ctx.raw_data.X[:, keep], ctx.raw_data.y, ctx.raw_data.sensitive) if ctx.raw_data else None
    return ctx.clone(splits=splits, raw_data=raw, feature_names=[ctx.feature_names[i] for i in keep])


def _require_retrain(ctx: EvalContext) -> str | None:
    return None if ctx.retrainable else "model is not retrainable from aitesteval (bring-your-own); apply the recommendation in your training pipeline"


# ============================================================== ML fixes
@register_fix("data_cleaning")
class DataCleaningFix(Fix):
    title = "Remove leaking features, de-duplicate and remove train/test contamination, then retrain"
    order, tasks = 10, ("classification", "regression")
    watch = [("classification", "accuracy"), ("classification", "roc_auc"), ("data_quality", "train_test_overlap")]

    def not_applicable(self, ctx, results):
        return super().not_applicable(ctx, results) or _require_retrain(ctx)

    def apply(self, ctx, results):
        from ..evaluators.data_checks import _row_keys

        leaks = [f.metric.split(".", 1)[1] for f in _findings(results, "TARGET_LEAKAGE")]
        new = _drop_features(ctx, leaks) if leaks else ctx.clone()
        tr = new.splits["train"]
        keys = _row_keys(tr.X)
        _, first = np.unique(np.array(keys, dtype=object), return_index=True)
        train = tr.subset(np.sort(first))
        train_keys = set(_row_keys(train.X))
        cleaned = {"train": train}
        removed = {}
        for name in ("val", "test"):
            s = new.splits[name]
            mask = np.array([k not in train_keys for k in _row_keys(s.X)])
            removed[name] = int((~mask).sum())
            cleaned[name] = s.subset(np.where(mask)[0])
        new.splits.update(cleaned)
        if new.raw_data is not None:
            rk = _row_keys(new.raw_data.X)
            _, f2 = np.unique(np.array(rk, dtype=object), return_index=True)
            new.raw_data = new.raw_data.subset(np.sort(f2))
        new.model = new.retrain()
        return ctx, new, (f"dropped leaking features {leaks or 'none'}; removed {len(tr) - len(train)} duplicate train rows "
                          f"and {removed} contaminated val/test rows; retrained")


@register_fix("drop_proxy_features")
class DropProxyFix(Fix):
    title = "Remove features that act as proxies for protected attributes, then retrain"
    order, tasks = 15, ("classification", "regression")
    watch = [("fairness", "*.disparate_impact_ratio"), ("classification", "accuracy")]

    def not_applicable(self, ctx, results):
        return super().not_applicable(ctx, results) or _require_retrain(ctx)

    def apply(self, ctx, results):
        proxies = sorted({f.metric.split(".", 1)[1].split("~")[0] for f in _findings(results, "PROXY_FEATURE")})
        new = _drop_features(ctx, proxies)
        new.model = new.retrain()
        return ctx, new, f"dropped proxy features {proxies}; retrained"


@register_fix("class_reweighting")
class ClassReweightFix(Fix):
    title = "Retrain with balanced class weights"
    order, tasks = 20, ("classification",)
    watch = [("classification", "recall"), ("classification", "f1"), ("classification", "balanced_accuracy"),
             ("classification", "accuracy")]

    def not_applicable(self, ctx, results):
        return super().not_applicable(ctx, results) or _require_retrain(ctx)

    def apply(self, ctx, results):
        y = ctx.splits["train"].y
        classes, counts = np.unique(y, return_counts=True)
        w = (len(y) / (len(classes) * counts))[np.searchsorted(classes, y)]
        if getattr(ctx.trainer, "kind", "") in ("logreg", "logistic_regression"):
            model = ctx.retrain(class_weight="balanced")
            new = ctx.clone(model=model, trainer_params={**ctx.trainer_params, "class_weight": "balanced"})
        else:
            new = ctx.clone(model=ctx.retrain(sample_weight=w))
        return ctx, new, "retrained with inverse-frequency class weights"


@register_fix("regularization")
class RegularizationFix(Fix):
    title = "Increase regularisation / reduce model variance"
    order, tasks = 25, ("classification", "regression")
    watch = [("classification", "train_test_gap"), ("classification", "f1"), ("regression", "train_test_gap")]

    def not_applicable(self, ctx, results):
        if r := super().not_applicable(ctx, results) or _require_retrain(ctx):
            return r
        if getattr(ctx.trainer, "kind", "") not in ("logreg", "logistic_regression", "knn", "ridge", "linear_regression"):
            return "custom model: regularisation hyper-parameters unknown - tune them in your own pipeline"
        return None

    def apply(self, ctx, results):
        kind = getattr(ctx.trainer, "kind", "")
        p = dict(ctx.trainer_params)
        if kind == "knn":
            p["k"] = max(15, int(p.get("k", 1)) * 10)
        elif kind in ("ridge", "linear_regression"):
            p["alpha"] = float(p.get("alpha", 1.0)) * 10
        else:
            p["l2"] = float(p.get("l2", 1e-3)) * 30
        new = ctx.clone(model=ctx.trainer(ctx.splits["train"].X, ctx.splits["train"].y, **p), trainer_params=p)
        return ctx, new, f"retrained with {p}"


@register_fix("robust_training")
class RobustTrainingFix(Fix):
    title = "Noise-augmented training with stronger regularisation"
    order, tasks = 27, ("classification", "regression")
    watch = [("robustness", "noise.0.1"), ("robustness", "noise.0.05")]

    def not_applicable(self, ctx, results):
        return super().not_applicable(ctx, results) or _require_retrain(ctx)

    def apply(self, ctx, results):
        rng = np.random.default_rng(0)
        tr = ctx.splits["train"]
        X = tr.X.astype(float)
        std = np.nanstd(X, axis=0)
        Xa = np.vstack([X, X + rng.normal(0, 1, X.shape) * std * 0.1])
        ya = np.concatenate([tr.y, tr.y])
        p = dict(ctx.trainer_params)
        if getattr(ctx.trainer, "kind", "") in ("logreg", "logistic_regression"):
            p["l2"] = float(p.get("l2", 1e-3)) * 10
        new = ctx.clone(model=ctx.trainer(Xa, ya, **p))
        return ctx, new, "trained on original + 10%-noise-augmented copies"


@register_fix("retrain_recent")
class RetrainRecentFix(Fix):
    title = "Retrain on training data + recent labelled production data"
    order, tasks = 30, ("classification", "regression")
    watch = [("drift", "production_accuracy"), ("drift", "production_r2"), ("drift", "performance_drop")]

    def not_applicable(self, ctx, results):
        if r := super().not_applicable(ctx, results) or _require_retrain(ctx):
            return r
        p = ctx.splits.get("production")
        return None if p is not None and p.y is not None else "needs labelled production data (`production` split with labels)"

    def apply(self, ctx, results):
        prod = ctx.splits["production"]
        idx = np.random.default_rng(0).permutation(len(prod))
        fit_idx, eval_idx = idx[: len(idx) // 2], idx[len(idx) // 2:]
        base = ctx.clone()
        base.splits["production"] = prod.subset(eval_idx)  # fair comparison on the same held-out window
        pf = prod.subset(fit_idx)
        tr = ctx.splits["train"]
        X = np.vstack([tr.X, pf.X])
        y = np.concatenate([tr.y, pf.y])
        weight = float(self.config.get("recent_weight", 3.0))
        w = np.concatenate([np.ones(len(tr)), np.full(len(pf), weight)])
        new = base.clone(model=ctx.trainer(X, y, sample_weight=w, **ctx.trainer_params))
        sens = {k: np.concatenate([v, pf.sensitive[k]]) for k, v in tr.sensitive.items() if k in pf.sensitive}
        new.splits["reference"] = Split(X, y, sens)  # drift reference = data the new model was trained on
        return base, new, (f"retrained on {len(tr)} train + {len(pf)} recent production rows (recent weight {weight}); "
                           f"evaluated on {len(eval_idx)} held-out production rows")


@register_fix("calibration")
class CalibrationFix(Fix):
    title = "Temperature-scaling calibration fitted on the validation split"
    order, tasks = 40, ("classification",)
    watch = [("classification", "ece"), ("classification", "log_loss"), ("classification", "accuracy")]

    def not_applicable(self, ctx, results):
        if r := super().not_applicable(ctx, results):
            return r
        return None if hasattr(ctx.model, "predict_proba") else "model has no predict_proba"

    def apply(self, ctx, results):
        val = ctx.splits["val"]
        t = fit_temperature(ctx.model.predict_proba(val.X), np.searchsorted(ctx.classes, val.y))
        return ctx, ctx.clone(model=TemperatureScaledModel(ctx.model, t)), f"temperature T={t:.3f} fitted on validation data"


@register_fix("threshold_tuning")
class ThresholdTuningFix(Fix):
    title = "Tune decision threshold for F1 on the validation split"
    order, tasks = 50, ("classification",)
    watch = [("classification", "f1"), ("classification", "recall"), ("classification", "precision")]

    def not_applicable(self, ctx, results):
        if r := super().not_applicable(ctx, results):
            return r
        if not ctx.is_binary:
            return "binary classification only"
        return None if hasattr(ctx.model, "predict_proba") else "model has no predict_proba"

    def apply(self, ctx, results):
        val = ctx.splits["val"]
        p = ctx.model.predict_proba(val.X)[:, 1]
        pos = val.y == ctx.classes[1]
        min_recall = self.config.get("min_recall")
        best_t, best_f1, best_rec = 0.5, -1.0, 0.0
        for t in np.linspace(0.02, 0.98, 97):
            pred = p >= t
            tp = (pred & pos).sum()
            f1 = 2 * tp / max(1, pred.sum() + pos.sum())
            rec = tp / max(1, pos.sum())
            if min_recall is not None and rec < float(min_recall):
                continue
            if f1 > best_f1:
                best_t, best_f1, best_rec = float(t), float(f1), float(rec)
        cur = getattr(ctx.model, "threshold", 0.5)
        floor = f", recall floor {min_recall}" if min_recall is not None else ""
        return ctx, ctx.clone(model=ThresholdModel(ctx.model, best_t)), (
            f"threshold {cur:.2f} -> {best_t:.2f} (val F1 {best_f1:.3f}, val recall {best_rec:.3f}{floor})")


@register_fix("fairness_mitigation")
class FairnessMitigationFix(Fix):
    title = "Post-processing: group-specific thresholds to equalise selection rates"
    order, tasks = 60, ("classification",)
    watch = [("fairness", "*.disparate_impact_ratio"), ("fairness", "*.equal_opportunity_diff"),
             ("classification", "accuracy")]

    def not_applicable(self, ctx, results):
        if r := super().not_applicable(ctx, results):
            return r
        if not ctx.is_binary or not hasattr(ctx.model, "predict_proba"):
            return "binary classifier with predict_proba required"
        return None

    def apply(self, ctx, results):
        bad = _findings(results, "DISPARATE_IMPACT") or _findings(results, "EQUAL_OPPORTUNITY_GAP") or \
            _findings(results, "DEMOGRAPHIC_PARITY_GAP")
        attr = bad[0].metric.split(".")[0] if bad else next(iter(ctx.splits["val"].sensitive))
        val = ctx.splits["val"]
        p = ctx.model.predict_proba(val.X)[:, 1]
        base_t = getattr(ctx.model, "threshold", 0.5)
        target = float((p >= base_t).mean())
        groups = val.sensitive[attr]
        thresholds = {}
        mode = self.config.get("mode", "demographic_parity")
        for g in np.unique(groups):
            pg = p[groups == g]
            if mode == "equal_opportunity":
                pos = val.y[groups == g] == ctx.classes[1]
                overall_tpr = float((p[val.y == ctx.classes[1]] >= base_t).mean())
                thresholds[str(g)] = float(np.quantile(pg[pos], 1 - overall_tpr)) if pos.any() else base_t
            else:
                thresholds[str(g)] = float(np.quantile(pg, 1 - target))
        model = GroupThresholdModel(ctx.model, attr, thresholds, default=base_t)
        return ctx, ctx.clone(model=model), (f"{mode} thresholds on '{attr}': "
                                            + ", ".join(f"{k}={v:.3f}" for k, v in thresholds.items())
                                            + ". NOTE: group-aware decisions may be restricted in some jurisdictions/sectors "
                                              "(e.g. US credit) - review with legal before deploying.")


@register_fix("compression_search")
class CompressionSearchFix(Fix):
    title = "Search for the smallest compressed model that stays within the quality budget"
    order, tasks = 70, ("classification",)
    watch = [("compression", "recommended")]

    def apply(self, ctx, results):
        from ..evaluators.comparison import CompressionEvaluator
        from ..models.ml import compress

        cfg = next((r for r in results if r.evaluator == "compression"), None)
        budget = (cfg.metrics.get("budget") if cfg else None) or {"max_accuracy_drop": 0.01, "max_group_accuracy_drop": 0.02,
                                                                 "min_agreement": 0.97}
        grid = [{"method": "quantize", "bits": b} for b in (16, 12, 8, 6, 5, 4, 3, 2)] + \
               [{"method": "prune", "sparsity": s} for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
        variants = []
        for spec in grid:
            try:
                variants.append(("-".join(f"{v}" for v in spec.values()), compress(ctx.model, **spec), spec))
            except TypeError:
                return ctx, ctx.clone(), "model type not supported by built-in compression"
        res = CompressionEvaluator.analyse(ctx, [(l, m) for l, m, _ in variants], budget)
        best = res["recommended"]
        new = ctx.clone()
        if best:
            spec = next(s for l, _, s in variants if l == best)
            new.config = {**ctx.config, "compression_methods_override": [spec]}
            row = next(v for v in res["variants"] if v["label"] == best)
            return ctx, new, (f"best within budget: {spec} -> {row['compression_ratio']:.1f}x smaller, "
                              f"accuracy drop {row['accuracy_drop']:+.4f}, agreement {row['agreement']:.3f}")
        return ctx, new, "no setting in the search grid meets the budget - consider distillation or QAT"


# ============================================================== LLM guardrails
def _guard_fix(gid: str, order: int, title: str, watch: list[tuple[str, str]]):
    @register_fix(gid)
    class _G(Fix):
        tasks = ("llm",)

        def apply(self, ctx, results):
            return ctx, ctx.clone(model=GUARDS[gid](ctx.model, **self.config)), f"wrapped model with {gid} guardrail"

    _G.order, _G.title, _G.watch, _G.__name__ = order, title, watch, f"{gid}_fix"
    return _G


_guard_fix("injection_guard", 10, "Prompt-injection input screening + secret leak filter",
           [("llm_security", "attack_success_rate"), ("llm_security", "canary_leak_rate")])
_guard_fix("safety_guard", 20, "Harmful-request pre-filter", [("llm_safety", "refusal_rate"), ("llm_safety", "over_refusal_rate")])
_guard_fix("pii_redaction", 30, "Output PII redaction", [("llm_safety", "pii_leak_rate")])
_guard_fix("grounding_guard", 40, "Drop unsupported sentences (faithfulness filter)",
           [("llm_grounding", "hallucination_rate"), ("llm_quality", "token_f1")])
_guard_fix("toxicity_filter", 50, "Toxic sentence filter", [("llm_safety", "toxicity_rate")])
_guard_fix("counterfactual_fairness", 55, "Neutralise demographic wording", [("llm_bias", "mean_similarity")])
_guard_fix("json_repair", 60, "Extract/repair JSON + strict retry", [("llm_format", "schema_valid_rate")])
_guard_fix("self_consistency", 70, "Self-consistency (medoid of 3 samples)", [("llm_performance", "consistency")])
_guard_fix("response_cache", 75, "Response cache for repeated questions (deterministic answers)",
           [("llm_performance", "consistency"), ("llm_performance", "latency_p95_s")])


# ============================================================== automation
def _mw_fix(fid: str, mw: str, order: int, title: str, watch):
    @register_fix(fid)
    class _M(Fix):
        tasks = ("automation",)

        def apply(self, ctx, results):
            wf = ctx.model
            if isinstance(wf, WorkflowWithMiddleware):
                wf = WorkflowWithMiddleware(wf.inner, wf.middleware + [mw])
            else:
                wf = WorkflowWithMiddleware(wf, [mw])
            return ctx, ctx.clone(model=wf), f"added '{mw}' middleware around tool calls"

    _M.order, _M.title, _M.watch, _M.__name__ = order, title, watch, f"{fid}_fix"
    return _M


_mw_fix("workflow_retry", "retry", 10, "Retry transient tool failures with back-off",
        [("automation", "fault_recovery_rate"), ("automation", "crash_rate")])
_mw_fix("workflow_idempotency", "idempotency", 20, "Idempotency keys for side-effecting tools",
        [("automation", "idempotency_rate")])

__all__ = ["FIXES", "Fix"]
