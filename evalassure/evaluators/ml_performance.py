"""Classification and regression performance evaluators."""
from __future__ import annotations

import numpy as np

from ..core.context import EvalContext
from ..core.registry import Evaluator, register_evaluator
from ..core.types import EvalResult
from . import metrics as M


def classification_metrics(ctx: EvalContext, split: str = "test", model=None) -> dict:
    s = ctx.splits[split]
    y = s.y
    pred = ctx.predict(split, model=model)
    labels = np.unique(np.concatenate([ctx.classes, np.unique(pred)]))
    cm = M.confusion(y, pred, labels)
    prec, rec, f1 = M.prf_per_class(cm)
    out: dict = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    support = cm.sum(axis=1)
    present = support > 0
    out["balanced_accuracy"] = float(rec[present].mean())
    if len(labels) == 2:
        out.update(precision=float(prec[1]), recall=float(rec[1]), f1=float(f1[1]), positive_label=labels[1].item())
    else:
        out.update(precision=float(prec[present].mean()), recall=float(rec[present].mean()),
                   f1=float(f1[present].mean()), averaging="macro")
    out["per_class"] = {str(l): {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                 "f1": round(float(f), 4), "support": int(n)}
                        for l, p, r, f, n in zip(labels.tolist(), prec, rec, f1, support)}
    out["confusion_matrix"] = {"labels": [str(l) for l in labels.tolist()], "matrix": cm.tolist()}
    proba = ctx.predict_proba(split, model=model)
    if proba is not None and proba.ndim == 2 and proba.shape[1] == len(ctx.classes):
        y_idx = np.searchsorted(ctx.classes, y)
        out["log_loss"] = M.log_loss(y_idx, proba)
        out["ece"] = M.expected_calibration_error(y_idx, proba)
        if proba.shape[1] == 2:
            out["roc_auc"] = M.roc_auc((y == ctx.classes[1]).astype(int), proba[:, 1])
            out["brier"] = float(np.mean((proba[:, 1] - (y == ctx.classes[1])) ** 2))
    return out


@register_evaluator("classification")
class ClassificationEvaluator(Evaluator):
    tasks = ("classification",)
    description = "Accuracy, precision/recall/F1, ROC-AUC, calibration (ECE), baseline & overfitting checks."

    DEFAULTS = {"accuracy": 0.75, "f1": 0.6, "recall": None, "precision": None, "roc_auc": 0.75,
                "ece_max": 0.08, "train_test_gap_max": 0.08}

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        m = classification_metrics(ctx, "test")
        tr = classification_metrics(ctx, "train")
        y = ctx.splits["test"].y
        majority = float(np.max(np.unique(y, return_counts=True)[1]) / len(y))
        m["majority_baseline_accuracy"] = majority
        m["train_accuracy"] = tr["accuracy"]
        m["train_f1"] = tr["f1"]
        m["train_test_gap"] = tr["f1"] - m["f1"]
        result.metrics = m
        t = {**self.DEFAULTS, **self.config.get("thresholds", {})}

        for key in ("accuracy", "f1", "recall", "precision", "roc_auc"):
            thr = t.get(key)
            if thr is not None and key in m and m[key] < thr:
                sev = "high" if m[key] < thr - 0.1 else "medium"
                self.finding(result, f"LOW_{key.upper()}", f"{key} below target", sev,
                             f"{key}={m[key]:.3f} < target {thr}", key, m[key], thr, ["performance"])
        chance = 1.0 / max(len(ctx.classes), 2)
        m["chance_balanced_accuracy"] = chance
        if m["balanced_accuracy"] <= chance + 0.03:
            self.finding(result, "NO_BETTER_THAN_BASELINE", "Model is no better than chance / always predicting the majority class",
                         "critical", f"balanced accuracy {m['balanced_accuracy']:.3f} vs chance {chance:.3f} "
                         f"(accuracy {m['accuracy']:.3f}, majority baseline {majority:.3f})",
                         "balanced_accuracy", m["balanced_accuracy"], chance, ["performance", "imbalance"])
        if majority > 0.85 and m["accuracy"] > 0.9 and m["balanced_accuracy"] < 0.75:
            self.finding(result, "ACCURACY_PARADOX", "High accuracy hides poor minority-class performance", "high",
                         f"accuracy {m['accuracy']:.3f} but balanced accuracy {m['balanced_accuracy']:.3f} "
                         f"(majority class = {majority:.1%}, majority-baseline accuracy {majority:.3f})", "balanced_accuracy", m["balanced_accuracy"], 0.75,
                         ["imbalance"])
        if m.get("ece") is not None and t["ece_max"] is not None and m["ece"] > t["ece_max"]:
            self.finding(result, "POORLY_CALIBRATED", "Predicted probabilities are not calibrated",
                         "medium" if m["ece"] < 2 * t["ece_max"] else "high",
                         f"ECE={m['ece']:.3f} > {t['ece_max']}; confidence scores cannot be trusted as probabilities",
                         "ece", m["ece"], t["ece_max"], ["calibration"])
        if m["train_test_gap"] > t["train_test_gap_max"]:
            self.finding(result, "OVERFITTING", "Large train/test performance gap", "high",
                         f"train F1 {tr['f1']:.3f} vs test F1 {m['f1']:.3f}", "train_test_gap",
                         m["train_test_gap"], t["train_test_gap_max"], ["generalization"])
        if m.get("roc_auc", 0) > 0.995 or m["accuracy"] > 0.995:
            self.finding(result, "SUSPICIOUSLY_HIGH_PERFORMANCE", "Near-perfect scores - check for target leakage", "high",
                         "Real-world problems rarely reach AUC/accuracy > 0.995", "roc_auc", m.get("roc_auc"), 0.995,
                         ["leakage"])


@register_evaluator("regression")
class RegressionEvaluator(Evaluator):
    tasks = ("regression",)
    description = "MAE, RMSE, R², MAPE, residual bias and overfitting checks."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        def stats(split: str) -> dict:
            y = ctx.splits[split].y.astype(float)
            p = ctx.predict(split).astype(float)
            err = p - y
            ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y - y.mean()) ** 2))
            return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
                    "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9))),
                    "median_ae": float(np.median(np.abs(err))), "mean_residual": float(np.mean(err)),
                    "residual_bias_pct": float(np.mean(err) / max(abs(y.mean()), 1e-9))}

        m, tr = stats("test"), stats("train")
        m["train_r2"], m["train_test_gap"] = tr["r2"], tr["r2"] - m["r2"]
        result.metrics = m
        t = {"r2": 0.7, "mape_max": 0.2, "bias_max": 0.05, "train_test_gap_max": 0.1, **self.config.get("thresholds", {})}
        if m["r2"] < t["r2"]:
            self.finding(result, "LOW_R2", "R² below target", "high" if m["r2"] < t["r2"] - 0.2 else "medium",
                         f"R²={m['r2']:.3f}", "r2", m["r2"], t["r2"], ["performance"])
        if m["mape"] > t["mape_max"]:
            self.finding(result, "HIGH_MAPE", "Mean absolute percentage error above target", "medium",
                         f"MAPE={m['mape']:.1%}", "mape", m["mape"], t["mape_max"], ["performance"])
        if abs(m["residual_bias_pct"]) > t["bias_max"]:
            self.finding(result, "SYSTEMATIC_BIAS", "Predictions are systematically too high/low", "medium",
                         f"mean residual = {m['residual_bias_pct']:.1%} of mean target", "residual_bias_pct",
                         m["residual_bias_pct"], t["bias_max"], ["bias"])
        if m["train_test_gap"] > t["train_test_gap_max"]:
            self.finding(result, "OVERFITTING", "Large train/test R² gap", "high", f"train R² {tr['r2']:.3f} vs test {m['r2']:.3f}",
                         "train_test_gap", m["train_test_gap"], t["train_test_gap_max"], ["generalization"])
