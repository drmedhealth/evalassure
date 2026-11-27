"""Data quality / leakage checks and model explainability (permutation importance + proxy detection)."""
from __future__ import annotations

import numpy as np

from ..core.context import EvalContext
from ..core.registry import Evaluator, register_evaluator
from ..core.types import EvalResult
from . import metrics as M


def _row_keys(X: np.ndarray) -> list[bytes]:
    Z = np.where(np.isnan(X.astype(float)), -9.87654321e17, np.round(X.astype(float), 8))
    return [r.tobytes() for r in Z]


@register_evaluator("data_quality")
class DataQualityEvaluator(Evaluator):
    tasks = ("classification", "regression")
    description = "Missing values, duplicates, train/test contamination, target leakage, imbalance, outliers, constant & collinear features."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        raw = ctx.raw_data or ctx.splits["train"]
        X, y = raw.X.astype(float), raw.y
        n, d = X.shape
        names = ctx.feature_names
        miss = {names[j]: float(np.isnan(X[:, j]).mean()) for j in range(d)}
        result.metrics.update(rows=n, features=d, missing_rate=miss)
        for k, v in miss.items():
            if v > 0:
                self.finding(result, "MISSING_VALUES", f"Missing values in '{k}'", "medium" if v > 0.05 else "low",
                             f"{v:.1%} missing", f"missing_rate.{k}", v, 0.0, ["data"])
        keys = _row_keys(X)
        dup = 1 - len(set(keys)) / n
        result.metrics["duplicate_rate"] = dup
        if dup > 0.001:
            self.finding(result, "DUPLICATE_ROWS", "Duplicate records in dataset", "medium" if dup > 0.01 else "low",
                         f"{dup:.2%} of rows are exact duplicates", "duplicate_rate", dup, 0.001, ["data"])
        train_keys = set(_row_keys(ctx.splits["train"].X))
        test_keys = _row_keys(ctx.splits["test"].X)
        overlap = float(np.mean([k in train_keys for k in test_keys]))
        result.metrics["train_test_overlap"] = overlap
        if overlap > 0.001:
            self.finding(result, "TRAIN_TEST_CONTAMINATION", "Test rows also appear in training data", "high",
                         f"{overlap:.2%} of test rows are in train -> optimistic metrics", "train_test_overlap", overlap,
                         0.0, ["data", "leakage"])
        const = [names[j] for j in range(d) if np.nanstd(X[:, j]) == 0]
        if const:
            self.finding(result, "CONSTANT_FEATURES", f"Constant features carry no information: {const}", "low", tags=["data"])
        # leakage: single feature almost perfectly predicts the target
        leak = {}
        yv = y
        for j in range(d):
            col = X[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 10:
                continue
            if ctx.task == "classification" and len(np.unique(yv)) == 2:
                auc = M.roc_auc((yv[ok] == np.unique(yv)[1]).astype(int), col[ok])
                power = max(auc, 1 - auc)
            else:
                c = np.corrcoef(col[ok], yv[ok].astype(float))[0, 1]
                power = abs(c) if not np.isnan(c) else 0.0
            leak[names[j]] = float(power)
        result.metrics["single_feature_predictive_power"] = dict(sorted(leak.items(), key=lambda kv: -kv[1]))
        for k, v in leak.items():
            if v > float(self.threshold("leakage_power", 0.97)):
                self.finding(result, "TARGET_LEAKAGE", f"Feature '{k}' almost perfectly predicts the target", "critical",
                             f"single-feature AUC/|corr| = {v:.3f}; likely only known after the outcome", f"leakage.{k}",
                             v, self.threshold("leakage_power", 0.97), ["data", "leakage"])
        # collinearity
        Xi = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(Xi, rowvar=False)
        pairs = [(names[i], names[j], float(C[i, j])) for i in range(d) for j in range(i + 1, d)
                 if not np.isnan(C[i, j]) and abs(C[i, j]) > 0.95]
        if pairs:
            result.metrics["collinear_pairs"] = pairs
            self.finding(result, "COLLINEAR_FEATURES", f"{len(pairs)} highly correlated feature pair(s)", "low",
                         ", ".join(f"{a}~{b} ({c:.2f})" for a, b, c in pairs[:5]), tags=["data"])
        if ctx.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            minority = float(counts.min() / counts.sum())
            result.metrics["class_distribution"] = {str(v): int(c) for v, c in zip(vals.tolist(), counts)}
            result.metrics["minority_share"] = minority
            if minority < float(self.threshold("min_minority_share", 0.1)):
                self.finding(result, "CLASS_IMBALANCE", "Severe class imbalance", "medium",
                             f"minority class is {minority:.2%} of data", "minority_share", minority,
                             self.threshold("min_minority_share", 0.1), ["data", "imbalance"])
        z = np.abs((Xi - Xi.mean(0)) / np.where(Xi.std(0) == 0, 1, Xi.std(0)))
        out_rate = float((z > 5).any(axis=1).mean())
        result.metrics["extreme_outlier_rate"] = out_rate
        if out_rate > 0.01:
            self.finding(result, "OUTLIERS", "Many extreme outliers (|z| > 5)", "low", f"{out_rate:.2%} of rows",
                         "extreme_outlier_rate", out_rate, 0.01, ["data"])
        if n < 10 * d:
            self.finding(result, "SMALL_SAMPLE", "Too few rows for the number of features", "medium",
                         f"{n} rows / {d} features", tags=["data"])


@register_evaluator("explainability")
class ExplainabilityEvaluator(Evaluator):
    tasks = ("classification", "regression")
    description = "Global permutation importance, proxy-discrimination detection, sensitive-feature usage."

    def evaluate(self, ctx: EvalContext, result: EvalResult) -> None:
        rng = np.random.default_rng(0)
        test = ctx.splits["test"]
        X, y = test.X.astype(float), test.y

        def score(Xp):
            p = ctx.predict("test", X=Xp)
            if ctx.task == "regression":
                return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
            return float((p == y).mean())

        base = score(X)
        imp = {}
        for j, name in enumerate(ctx.feature_names):
            drops = []
            for _ in range(int(self.cfg("repeats", 3))):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - score(Xp))
            imp[name] = float(np.mean(drops))
        ranking = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        result.metrics["permutation_importance"] = ranking
        top = list(ranking)[: int(self.cfg("top_k", 3))]
        result.metrics["top_features"] = top
        total = sum(max(v, 0) for v in imp.values()) or 1.0
        if ranking[top[0]] / total > 0.8:
            self.finding(result, "DOMINANT_FEATURE", f"One feature ('{top[0]}') drives almost all decisions", "low",
                         f"{ranking[top[0]] / total:.0%} of total importance", tags=["explainability"])
        # proxy discrimination
        proxies = {}
        for attr, groups in test.sensitive.items():
            for g in np.unique(groups)[:10]:
                ind = (groups == g).astype(float)
                for j, name in enumerate(ctx.feature_names):
                    col = np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j]))
                    if np.std(col) == 0 or np.std(ind) == 0:
                        continue
                    c = abs(float(np.corrcoef(col, ind)[0, 1]))
                    key = f"{name}~{attr}"
                    proxies[key] = max(proxies.get(key, 0.0), c)
        result.metrics["sensitive_correlation"] = dict(sorted(proxies.items(), key=lambda kv: -kv[1])[:10])
        thr = float(self.threshold("proxy_corr", 0.3))
        for key, c in proxies.items():
            feat, attr = key.split("~")
            if c > thr and feat in top:
                self.finding(result, "PROXY_FEATURE", f"Influential feature '{feat}' is a proxy for '{attr}'", "high",
                             f"|corr|={c:.2f} with {attr} and ranked in top-{len(top)} importance -> indirect discrimination risk",
                             f"proxy.{key}", c, thr, ["fairness", "explainability"])
        lowered = {n.lower() for n in ctx.feature_names}
        used = [a for a in test.sensitive if a.lower() in lowered]
        if used:
            self.finding(result, "SENSITIVE_FEATURE_USED", f"Protected attributes used directly as model inputs: {used}",
                         "high", tags=["fairness", "privacy"])
