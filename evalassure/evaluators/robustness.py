"""Robustness (noise, feature outage, missing values) and drift (PSI, KS, prediction & performance drift)."""
from __future__ import annotations

import numpy as np

from ..core.context import EvalContext
from ..core.registry import Evaluator, register_evaluator
from ..core.types import EvalResult
from . import metrics as M


def _score(ctx: EvalContext, X: np.ndarray, y: np.ndarray, model=None) -> float:
    p = ctx.predict("test", X=X, model=model)
    if ctx.task == "regression":
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - np.sum((p - y) ** 2) / ss_tot)
    return float((p == y).mean())


@register_evaluator("robustness")
class RobustnessEvaluator(Evaluator):
    tasks = ("classification", "regression")
    description = "Gaussian-noise stress test, prediction stability, single-feature outage, missing-value handling."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        rng = np.random.default_rng(int(self.cfg("seed", 0)))
        test, train = ctx.splits["test"], ctx.splits["train"]
        X, y = test.X.astype(float), test.y
        std = np.nanstd(train.X.astype(float), axis=0)
        continuous = np.array([len(np.unique(train.X[:, j][~np.isnan(train.X[:, j].astype(float))])) > 2
                               for j in range(X.shape[1])])
        base = _score(ctx, X, y)
        base_pred = ctx.predict("test")
        metric = "r2" if ctx.task == "regression" else "accuracy"
        result.metrics["baseline_" + metric] = base
        noise = {}
        for level in self.cfg("noise_levels", [0.05, 0.1, 0.2]):
            Xn = X + rng.normal(0, 1, X.shape) * std * level * continuous
            sc = _score(ctx, Xn, y)
            entry = {metric: sc, "drop": base - sc}
            if ctx.task != "regression":
                entry["flip_rate"] = float((ctx.predict("test", X=Xn) != base_pred).mean())
            noise[str(level)] = entry
        result.metrics["noise"] = noise
        max_drop = float(self.threshold("max_drop_at_10pct_noise", 0.05))
        ref = noise.get("0.1") or list(noise.values())[len(noise) // 2]
        if ref["drop"] > max_drop:
            self.finding(result, "NOISE_SENSITIVITY", "Performance collapses under small input noise", "high",
                         f"{metric} drops {ref['drop']:.3f} with 10%-of-std Gaussian noise", f"noise_drop", ref["drop"],
                         max_drop, ["robustness"])
        flip = (noise.get("0.05") or {}).get("flip_rate")
        if flip is not None and flip > float(self.threshold("max_flip_rate_at_5pct_noise", 0.08)):
            self.finding(result, "UNSTABLE_PREDICTIONS", "Predictions flip under tiny perturbations", "medium",
                         f"{flip:.1%} of decisions change with 5%-of-std noise", "flip_rate", flip,
                         self.threshold("max_flip_rate_at_5pct_noise", 0.08), ["robustness"])
        # single-feature outage (sensor/API down -> value replaced by median)
        med = np.nanmedian(train.X.astype(float), axis=0)
        outage = {}
        for j, name in enumerate(ctx.feature_names):
            Xo = X.copy()
            Xo[:, j] = med[j]
            outage[name] = base - _score(ctx, Xo, y)
        result.metrics["feature_outage_drop"] = dict(sorted(outage.items(), key=lambda kv: -kv[1]))
        worst = max(outage, key=outage.get)
        if outage[worst] > float(self.threshold("max_single_feature_drop", 0.15)):
            self.finding(result, "SINGLE_FEATURE_DEPENDENCY", f"Model over-relies on '{worst}'", "medium",
                         f"{metric} drops {outage[worst]:.3f} if '{worst}' is unavailable", "feature_outage_drop",
                         outage[worst], self.threshold("max_single_feature_drop", 0.15), ["robustness"])
        # missing values
        Xm = X.copy()
        Xm[rng.random(X.shape) < 0.1] = np.nan
        try:
            pm = ctx.predict("test", X=Xm)
            pm = np.asarray(pm)
            if pm.dtype.kind == "f" and np.isnan(pm).any():
                raise ValueError("model returned NaN predictions")
            result.metrics["missing_values_" + metric] = _score(ctx, Xm, y)
        except Exception as exc:
            self.finding(result, "MISSING_VALUE_CRASH", "Model fails when inputs contain missing values", "high",
                         f"{type(exc).__name__}: {exc}", tags=["robustness", "data"])


@register_evaluator("drift")
class DriftEvaluator(Evaluator):
    tasks = ("classification", "regression")
    description = "Feature drift (PSI, KS), prediction drift and production performance decay."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        ref_name = "reference" if "reference" in ctx.splits else "train"
        ref = ctx.splits[ref_name]
        cur_name = "production" if "production" in ctx.splits else "test"
        cur = ctx.splits[cur_name]
        result.metrics["current_window"] = cur_name
        feats = {}
        for j, name in enumerate(ctx.feature_names):
            a, b = ref.X[:, j].astype(float), cur.X[:, j].astype(float)
            ks, p = M.ks_2samp(a, b)
            feats[name] = {"psi": M.psi(a, b), "ks": ks, "ks_p": p,
                           "ref_mean": float(np.nanmean(a)), "cur_mean": float(np.nanmean(b))}
        result.metrics["features"] = feats
        t_med, t_high = float(self.threshold("psi_warn", 0.1)), float(self.threshold("psi_alert", 0.25))
        for name, f in feats.items():
            if f["psi"] > t_high:
                self.finding(result, "FEATURE_DRIFT", f"Significant drift in '{name}'", "high",
                             f"PSI={f['psi']:.3f}; mean {f['ref_mean']:.2f} -> {f['cur_mean']:.2f}", f"psi.{name}",
                             f["psi"], t_high, ["drift"])
            elif f["psi"] > t_med:
                self.finding(result, "FEATURE_DRIFT_MODERATE", f"Moderate drift in '{name}'", "low",
                             f"PSI={f['psi']:.3f}", f"psi.{name}", f["psi"], t_med, ["drift"])
        # prediction drift
        s_ref, s_cur = ctx.positive_scores(ref_name), ctx.positive_scores(cur_name)
        if s_ref is None:
            s_ref, s_cur = ctx.predict(ref_name).astype(float), ctx.predict(cur_name).astype(float)
        pred_psi = M.psi(s_ref, s_cur)
        result.metrics["prediction_psi"] = pred_psi
        if pred_psi > t_high:
            self.finding(result, "PREDICTION_DRIFT", "Distribution of model outputs has shifted", "medium",
                         f"prediction PSI={pred_psi:.3f}", "prediction_psi", pred_psi, t_high, ["drift"])
        # performance decay when production labels are available
        if cur_name == "production" and cur.y is not None:
            if ctx.task == "regression":
                def r2(split):
                    y, p = ctx.splits[split].y, ctx.predict(split)
                    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
                ref_perf, cur_perf, name = r2("test"), r2("production"), "r2"
            else:
                ref_perf = float((ctx.predict("test") == ctx.splits["test"].y).mean())
                cur_perf = float((ctx.predict("production") == cur.y).mean())
                name = "accuracy"
            result.metrics.update({f"test_{name}": ref_perf, f"production_{name}": cur_perf,
                                   "performance_drop": ref_perf - cur_perf})
            if ctx.task != "regression":
                lab_psi = abs(float(np.mean(cur.y == ctx.classes[-1])) - float(np.mean(ref.y == ctx.classes[-1])))
                result.metrics["label_rate_shift"] = lab_psi
            if ref_perf - cur_perf > float(self.threshold("max_performance_drop", 0.03)):
                self.finding(result, "PERFORMANCE_DECAY", "Model performance has degraded in production", "critical"
                             if ref_perf - cur_perf > 0.1 else "high",
                             f"{name}: test {ref_perf:.3f} -> production {cur_perf:.3f}", "performance_drop",
                             ref_perf - cur_perf, self.threshold("max_performance_drop", 0.03), ["drift", "performance"])
