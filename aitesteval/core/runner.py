"""Scenario runner: build context -> evaluate -> compliance -> recommend -> diagnose -> auto-fix -> re-evaluate."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

from .. import evaluators as _evaluators  # noqa: F401  (registers evaluators)
from ..compliance import DISCLAIMER, run_compliance
from ..data.loaders import load_dataset, load_records, stratified_split
from ..models.loader import build_llm, build_ml_model, build_workflow
from ..remediation import fixes as _fixes  # noqa: F401  (registers fixes)
from ..remediation.recommendations import PLAYBOOK, recommend
from ..troubleshoot import diagnose
from .context import EvalContext, Split
from .registry import EVALUATORS, FIXES
from .types import EvalResult, FixOutcome, Report

VALID_TASKS = {"classification", "regression", "llm", "automation"}
DEFAULT_EVALUATORS = {
    "classification": ["data_quality", "classification", "fairness", "robustness", "explainability", "drift"],
    "regression": ["data_quality", "regression", "fairness", "robustness", "explainability", "drift"],
    "llm": ["llm_quality", "llm_grounding", "llm_safety", "llm_security", "llm_bias", "llm_format", "llm_performance"],
    "automation": ["automation"],
}


def load_scenario(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        sc = yaml.safe_load(fh)
    validate_scenario(sc)
    return sc, path.resolve().parent


def validate_scenario(sc: Any) -> None:
    errors = []
    if not isinstance(sc, dict):
        raise ValueError("Scenario must be a YAML mapping")
    for key in ("name", "task", "model"):
        if key not in sc:
            errors.append(f"missing required key '{key}'")
    if sc.get("task") not in VALID_TASKS:
        errors.append(f"task must be one of {sorted(VALID_TASKS)}")
    if sc.get("task") in ("classification", "regression") and "data" not in sc:
        errors.append("ML scenarios need a `data:` section")
    if sc.get("task") == "llm" and not sc.get("llm", {}).get("cases"):
        errors.append("LLM scenarios need `llm.cases`")
    if sc.get("task") == "automation" and not sc.get("automation", {}).get("cases"):
        errors.append("automation scenarios need `automation.cases`")
    for ev in sc.get("evaluators", []):
        name = ev if isinstance(ev, str) else ev.get("name")
        if name not in EVALUATORS:
            errors.append(f"unknown evaluator '{name}' (available: {sorted(EVALUATORS)})")
    if errors:
        raise ValueError("Invalid scenario:\n  - " + "\n  - ".join(errors))


def _get_path(d: Any, path: str) -> Any:
    if not isinstance(d, dict):
        return None
    if path in d:
        return d[path]
    if path.startswith("*."):
        rest = path[2:]
        out = {k: _get_path(v, rest) for k, v in d.items() if isinstance(v, dict)}
        out = {k: v for k, v in out.items() if v is not None}
        return out or None
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in d:
            return _get_path(d[head], ".".join(parts[i:]))
    return None


class Runner:
    def __init__(self, scenario: dict[str, Any], base_dir: Path | str = ".") -> None:
        validate_scenario(scenario)
        self.sc = scenario
        self.base_dir = Path(base_dir).resolve()

    @classmethod
    def from_file(cls, path: str | Path) -> "Runner":
        sc, base = load_scenario(path)
        return cls(sc, base)

    # ------------------------------------------------------------ paths
    def resolve(self, p: str) -> Path:
        cand = Path(p)
        if cand.is_absolute():
            return cand
        for root in (self.base_dir, self.base_dir.parent, Path.cwd()):
            if (root / cand).exists():
                return root / cand
        raise FileNotFoundError(f"Cannot find '{p}' (looked in {self.base_dir}, {self.base_dir.parent}, {Path.cwd()})")

    @property
    def code_root(self) -> Path:
        return self.base_dir.parent

    # ------------------------------------------------------------ context
    def build_context(self) -> EvalContext:
        sc = self.sc
        task = sc["task"]
        ctx = EvalContext(task=task, model_card=sc.get("model_card", {}), config=dict(sc.get("settings", {})))
        if task in ("classification", "regression"):
            data_spec = dict(sc["data"])
            if "path" in data_spec:
                data_spec["path"] = str(self.resolve(data_spec["path"]))
                if "production_path" in data_spec:
                    data_spec["production_path"] = str(self.resolve(data_spec["production_path"]))
            data = load_dataset(data_spec, self.base_dir)
            split_cfg = sc["data"].get("split", {})
            ctx.splits = stratified_split(data, stratify=task == "classification", **split_cfg)
            ctx.raw_data = Split(data["X"], data["y"], data.get("sensitive", {}))
            ctx.feature_names = list(data["feature_names"])
            ctx.config["data_description"] = data.get("description", "")
            tr = ctx.splits["train"]
            fn = ctx.feature_names
            ctx.model, ctx.trainer, ctx.trainer_params = build_ml_model(sc["model"], tr.X, tr.y, self.code_root, fn, task)
            if sc.get("challenger"):
                ctx.challenger, _, _ = build_ml_model(sc["challenger"], tr.X, tr.y, self.code_root, fn, task)
            extra = []
            for spec in sc.get("compressed_models", []):
                m, _, _ = build_ml_model({**spec, "fit": spec.get("fit", False)}, tr.X, tr.y, self.code_root)
                extra.append((spec.get("label", "custom"), m))
            ctx.config["extra_compressed"] = extra
        elif task == "llm":
            llm = sc["llm"]
            ctx.llm_cases = load_records(self.resolve(llm["cases"]))
            canary = llm.get("canary")
            sp = llm.get("system_prompt")
            ctx.system_prompt = sp.replace("{canary}", canary) if sp and canary else sp
            ctx.config["canary"] = canary
            ctx.model = build_llm(sc["model"], self.code_root)
            if sc.get("challenger"):
                ctx.challenger = build_llm(sc["challenger"], self.code_root)
        else:
            ctx.automation_cases = load_records(self.resolve(sc["automation"]["cases"]))
            ctx.model = build_workflow(sc["model"], self.code_root)
        return ctx

    # ------------------------------------------------------------ evaluate
    def evaluator_specs(self) -> list[tuple[str, dict[str, Any]]]:
        specs = self.sc.get("evaluators") or DEFAULT_EVALUATORS[self.sc["task"]]
        out = []
        for ev in specs:
            if isinstance(ev, str):
                out.append((ev, {}))
            else:
                cfg = {k: v for k, v in ev.items() if k != "name"}
                out.append((ev["name"], cfg))
        return out

    def evaluate(self, ctx: EvalContext) -> list[EvalResult]:
        return [EVALUATORS[name](**cfg).run(ctx) for name, cfg in self.evaluator_specs()]

    # ------------------------------------------------------------ main
    def run(self, autofix: bool | None = None, progress=None) -> Report:
        say = progress or (lambda *_: None)
        sc = self.sc
        say(f"Building context for '{sc['name']}' ({sc['task']})")
        ctx = self.build_context()
        say("Running evaluators: " + ", ".join(n for n, _ in self.evaluator_specs()))
        results = self.evaluate(ctx)
        frameworks = sc.get("compliance", [])
        compliance = run_compliance(results, ctx.model_card, ctx.task, frameworks) if frameworks else []
        findings = [f for r in results for f in r.findings]
        report = Report(
            scenario=sc["name"], use_case=sc.get("use_case", ""), task=sc["task"],
            timestamp=dt.datetime.now().isoformat(timespec="seconds"), results=results, compliance=compliance,
            recommendations=recommend(findings, compliance), diagnoses=diagnose(results),
            metadata={"description": sc.get("description", ""), "model": _describe(ctx.model),
                      "data": ctx.config.get("data_description", ""), "frameworks": frameworks,
                      "compliance_disclaimer": DISCLAIMER if frameworks else "", "model_card": ctx.model_card},
        )
        do_fix = sc.get("autofix", False) if autofix is None else autofix
        if do_fix:
            say("Applying auto-fixes")
            final_ctx, final_results = self._autofix(ctx, results, report, say)
            if any(f.applied for f in report.fixes):
                report.post_fix_results = final_results
                report.post_fix_compliance = (run_compliance(final_results, final_ctx.model_card, final_ctx.task, frameworks)
                                              if frameworks else [])
                report.metadata["fixed_model"] = _describe(final_ctx.model)
        return report

    def _autofix(self, ctx: EvalContext, results: list[EvalResult], report: Report, say):
        cfg = self.sc.get("fixes", {}) or {}
        exclude = set(self.sc.get("fix_exclude", []))
        wanted = []
        for rec in report.recommendations:
            if rec.auto_fix and rec.auto_fix not in wanted and rec.auto_fix not in exclude and rec.auto_fix in FIXES:
                wanted.append(rec.auto_fix)
        wanted.sort(key=lambda f: FIXES[f].order)
        cur_ctx, cur_res = ctx, results
        for fid in wanted:
            fix = FIXES[fid](**cfg.get(fid, {}))
            reason = fix.not_applicable(cur_ctx, cur_res)
            if reason:
                report.fixes.append(FixOutcome(fid, False, fix.title, notes=f"skipped: {reason}"))
                continue
            targeted = {c for c, v in PLAYBOOK.items() if v[3] == fid}
            try:
                base_ctx, new_ctx, desc = fix.apply(cur_ctx, cur_res)
            except Exception as exc:
                report.fixes.append(FixOutcome(fid, False, fix.title, notes=f"failed: {type(exc).__name__}: {exc}"))
                continue
            base_res = cur_res if base_ctx is cur_ctx else self.evaluate(base_ctx)
            new_res = self.evaluate(new_ctx)
            before, after = self._snapshot(fix, base_res, targeted), self._snapshot(fix, new_res, targeted)
            improved = (after["targeted_findings"] < before["targeted_findings"]
                        or (after["targeted_findings"] == before["targeted_findings"]
                            and after["health_score"] > before["health_score"]))
            accepted = improved and after["health_score"] >= before["health_score"]
            note = ("accepted" if accepted else
                    "not needed: issue already resolved by an earlier fix" if before["targeted_findings"] == 0 else
                    "rolled back: no net improvement")
            report.fixes.append(FixOutcome(fid, accepted, f"{fix.title}: {desc}", before, after, improved, note))
            say(f"  fix {fid}: {note} (health {before['health_score']} -> {after['health_score']})")
            if accepted:
                new_ctx.applied_fixes.append(fid)
                cur_ctx, cur_res = new_ctx, new_res
        return cur_ctx, cur_res

    @staticmethod
    def _snapshot(fix, res: list[EvalResult], targeted: set[str]) -> dict[str, Any]:
        snap: dict[str, Any] = {"health_score": Report.health_score_of(res),
                                "targeted_findings": sum(1 for r in res for f in r.findings if f.code in targeted)}
        by = {r.evaluator: r.metrics for r in res}
        for ev, path in fix.watch:
            if ev in by:
                v = _get_path(by[ev], path)
                if v is not None:
                    snap[f"{ev}.{path}"] = v
        return snap


def _describe(model: Any) -> str:
    if model is None:
        return ""
    chain = getattr(model, "guard_chain", None)
    if chain:
        inner = model
        while hasattr(inner, "inner"):
            inner = inner.inner
        return f"{getattr(inner, 'name', type(inner).__name__)} + guards[{', '.join(chain)}]"
    parts = []
    m = model
    while m is not None:
        parts.append(getattr(m, "name", None) or type(m).__name__)
        m = getattr(m, "base", None) if "base" in getattr(m, "__dict__", {}) else None
    return " <- ".join(parts)


def run_scenario(path: str | Path, autofix: bool | None = None, progress=None) -> Report:
    return Runner.from_file(path).run(autofix=autofix, progress=progress)
