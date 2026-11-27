"""Run compliance controls against evaluation evidence."""
from __future__ import annotations

from typing import Any

from ..core.types import ComplianceResult, ControlStatus, EvalResult
from .frameworks import FRAMEWORKS, Control, Evidence

DISCLAIMER = ("Compliance results are indicative technical evidence mapped to published requirements. "
              "They are not legal advice and do not certify conformity.")


def run_compliance(results: list[EvalResult], model_card: dict[str, Any], task: str,
                   frameworks: list[str]) -> list[ComplianceResult]:
    ev = Evidence(results, model_card, task)
    out: list[ComplianceResult] = []
    for fw in frameworks:
        if fw not in FRAMEWORKS:
            raise KeyError(f"Unknown framework '{fw}'. Available: {sorted(FRAMEWORKS)}")
        for c in FRAMEWORKS[fw]():
            out.append(_evaluate(c, ev))
    return out


def _evaluate(c: Control, ev: Evidence) -> ComplianceResult:
    try:
        if not c.applies(ev):
            return ComplianceResult(c.id, c.framework, c.title, c.reference, ControlStatus.NOT_APPLICABLE,
                                    "not applicable to this system's risk level / task")
        status, evidence, remediation = c.check(ev)
    except Exception as exc:  # a broken control must not break the run
        status, evidence, remediation = ControlStatus.MANUAL, f"check error: {exc}", "Review manually"
    return ComplianceResult(c.id, c.framework, c.title, c.reference, status, evidence, remediation)


__all__ = ["run_compliance", "FRAMEWORKS", "DISCLAIMER"]
