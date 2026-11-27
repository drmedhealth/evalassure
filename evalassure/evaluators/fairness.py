"""Group fairness: demographic parity, disparate impact (4/5ths rule), equal opportunity,
equalized odds, per-group accuracy; error disparity for regression."""
from __future__ import annotations

import numpy as np

from ..core.context import EvalContext
from ..core.registry import Evaluator, register_evaluator
from ..core.types import EvalResult


def group_rates(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, favorable) -> dict[str, dict]:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        yt, yp = (y[m] == favorable), (pred[m] == favorable)
        out[str(g)] = {
            "n": int(m.sum()),
            "selection_rate": float(yp.mean()),
            "base_rate": float(yt.mean()),
            "tpr": float(yp[yt].mean()) if yt.any() else float("nan"),
            "fpr": float(yp[~yt].mean()) if (~yt).any() else float("nan"),
            "accuracy": float((y[m] == pred[m]).mean()),
        }
    return out


def fairness_summary(rates: dict[str, dict], min_group_size: int = 30) -> dict:
    valid = {g: r for g, r in rates.items() if r["n"] >= min_group_size}
    if len(valid) < 2:
        return {}
    sel = {g: r["selection_rate"] for g, r in valid.items()}
    tpr = {g: r["tpr"] for g, r in valid.items() if not np.isnan(r["tpr"])}
    fpr = {g: r["fpr"] for g, r in valid.items() if not np.isnan(r["fpr"])}
    acc = {g: r["accuracy"] for g, r in valid.items()}
    priv = max(sel, key=sel.get)
    unpriv = min(sel, key=sel.get)
    return {
        "privileged_group": priv, "unprivileged_group": unpriv,
        "demographic_parity_diff": max(sel.values()) - min(sel.values()),
        "disparate_impact_ratio": (sel[unpriv] / sel[priv]) if sel[priv] > 0 else float("nan"),
        "equal_opportunity_diff": (max(tpr.values()) - min(tpr.values())) if len(tpr) > 1 else 0.0,
        "equalized_odds_diff": max((max(tpr.values()) - min(tpr.values())) if len(tpr) > 1 else 0.0,
                                   (max(fpr.values()) - min(fpr.values())) if len(fpr) > 1 else 0.0),
        "accuracy_diff": max(acc.values()) - min(acc.values()),
        "worst_group_accuracy": min(acc.values()),
        "worst_group": min(acc, key=acc.get),
    }


@register_evaluator("fairness")
class FairnessEvaluator(Evaluator):
    tasks = ("classification", "regression")
    description = "Demographic parity, disparate impact (80% rule), equal opportunity, equalized odds, error disparity."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> str | None:
        test = ctx.splits["test"]
        attrs = self.cfg("sensitive") or list(test.sensitive)
        if not attrs:
            return "no sensitive attributes provided (add `sensitive:` to the data spec)"
        t = {"disparate_impact_min": 0.8, "demographic_parity_diff_max": 0.1, "equal_opportunity_diff_max": 0.1,
             "equalized_odds_diff_max": 0.15, "accuracy_diff_max": 0.05, **self.config.get("thresholds", {})}
        pred = ctx.predict("test")
        if ctx.task == "regression":
            return self._regression(ctx, result, attrs, pred, t)
        favorable = self.cfg("favorable_label", ctx.classes[-1].item() if ctx.is_binary else None)
        if favorable is None:
            favorable = ctx.classes[-1].item()
        result.metrics["favorable_label"] = favorable
        for attr in attrs:
            if attr not in test.sensitive:
                self.finding(result, "MISSING_SENSITIVE_ATTRIBUTE", f"Sensitive attribute '{attr}' not found", "low")
                continue
            rates = group_rates(test.y, pred, test.sensitive[attr], favorable)
            summ = fairness_summary(rates, int(self.cfg("min_group_size", 30)))
            result.metrics[attr] = {"groups": rates, **summ}
            if not summ:
                continue
            di = summ["disparate_impact_ratio"]
            if di < t["disparate_impact_min"]:
                self.finding(result, "DISPARATE_IMPACT", f"Disparate impact on '{attr}' fails the 4/5ths rule",
                             "critical" if di < 0.6 else "high",
                             f"selection rate {summ['unprivileged_group']}={rates[summ['unprivileged_group']]['selection_rate']:.2%} "
                             f"vs {summ['privileged_group']}={rates[summ['privileged_group']]['selection_rate']:.2%} (ratio {di:.2f})",
                             f"{attr}.disparate_impact_ratio", di, t["disparate_impact_min"], ["fairness", attr])
            if summ["demographic_parity_diff"] > t["demographic_parity_diff_max"]:
                self.finding(result, "DEMOGRAPHIC_PARITY_GAP", f"Selection-rate gap across '{attr}'", "medium",
                             f"gap={summ['demographic_parity_diff']:.3f}", f"{attr}.demographic_parity_diff",
                             summ["demographic_parity_diff"], t["demographic_parity_diff_max"], ["fairness", attr])
            if summ["equal_opportunity_diff"] > t["equal_opportunity_diff_max"]:
                self.finding(result, "EQUAL_OPPORTUNITY_GAP", f"True-positive-rate gap across '{attr}'", "high",
                             f"qualified members of some groups are approved less often (TPR gap {summ['equal_opportunity_diff']:.3f})",
                             f"{attr}.equal_opportunity_diff", summ["equal_opportunity_diff"],
                             t["equal_opportunity_diff_max"], ["fairness", attr])
            if summ["accuracy_diff"] > t["accuracy_diff_max"]:
                self.finding(result, "GROUP_PERFORMANCE_GAP", f"Model is less accurate for some '{attr}' groups", "medium",
                             f"worst group '{summ['worst_group']}' accuracy {summ['worst_group_accuracy']:.3f}; gap {summ['accuracy_diff']:.3f}",
                             f"{attr}.accuracy_diff", summ["accuracy_diff"], t["accuracy_diff_max"], ["fairness", attr])
            small = [g for g, r in rates.items() if r["n"] < int(self.cfg("min_group_size", 30))]
            if small:
                self.finding(result, "SMALL_GROUP_SAMPLE", f"Groups too small to assess reliably: {small}", "low",
                             f"'{attr}' groups {small} have < {self.cfg('min_group_size', 30)} test rows; collect more data",
                             tags=["fairness", "data"])
        return None

    def _regression(self, ctx, result, attrs, pred, t):
        test = ctx.splits["test"]
        for attr in attrs:
            groups = test.sensitive.get(attr)
            if groups is None:
                continue
            per = {}
            for g in np.unique(groups):
                m = groups == g
                err = pred[m] - test.y[m]
                per[str(g)] = {"n": int(m.sum()), "mae": float(np.mean(np.abs(err))),
                               "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(test.y[m]), 1e-9))),
                               "mean_residual": float(np.mean(err))}
            mapes = [v["mape"] for v in per.values()]
            ratio = max(mapes) / max(min(mapes), 1e-9)
            result.metrics[attr] = {"groups": per, "error_ratio": ratio}
            if ratio > float(self.threshold("error_ratio_max", 1.25)):
                self.finding(result, "ERROR_DISPARITY", f"Prediction error differs across '{attr}'", "medium",
                             f"worst/best MAPE ratio {ratio:.2f}", f"{attr}.error_ratio", ratio,
                             self.threshold("error_ratio_max", 1.25), ["fairness", attr])
        return None
