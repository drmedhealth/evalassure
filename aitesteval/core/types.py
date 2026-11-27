"""Core result types shared by every evaluator, compliance check and fix."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def weight(self) -> int:
        return {"critical": 25, "high": 12, "medium": 5, "low": 2, "info": 0}[self.value]

    @property
    def rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[self.value]


@dataclass
class Finding:
    """A single issue detected by an evaluator."""

    code: str
    title: str
    severity: Severity
    evaluator: str
    detail: str = ""
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class EvalResult:
    evaluator: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_s: float = 0.0
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return not any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "passed": self.passed,
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.findings],
            "artifacts": self.artifacts,
            "error": self.error,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 4),
        }


class ControlStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    MANUAL = "manual_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceResult:
    control_id: str
    framework: str
    title: str
    reference: str
    status: ControlStatus
    evidence: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Recommendation:
    code: str
    title: str
    priority: str  # P0 (now) .. P3 (nice to have)
    rationale: str
    actions: list[str] = field(default_factory=list)
    auto_fix: str | None = None
    related: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixOutcome:
    fix: str
    applied: bool
    description: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    improved: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    title: str
    root_cause: str
    evidence: list[str]
    next_steps: list[str]
    confidence: str = "medium"
    kb_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    scenario: str
    use_case: str
    task: str
    timestamp: str
    results: list[EvalResult]
    compliance: list[ComplianceResult] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    fixes: list[FixOutcome] = field(default_factory=list)
    post_fix_results: list[EvalResult] = field(default_factory=list)
    post_fix_compliance: list[ComplianceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- scoring -------------------------------------------------------
    @staticmethod
    def health_score_of(results: list[EvalResult]) -> int:
        penalty = 0
        for r in results:
            if r.error:
                penalty += 10
            for f in r.findings:
                penalty += f.severity.weight
        return max(0, 100 - penalty)

    @staticmethod
    def compliance_score_of(items: list[ComplianceResult]) -> float | None:
        scored = [c for c in items if c.status in (ControlStatus.PASS, ControlStatus.FAIL, ControlStatus.WARN)]
        if not scored:
            return None
        pts = sum(1.0 if c.status == ControlStatus.PASS else 0.5 if c.status == ControlStatus.WARN else 0.0 for c in scored)
        return round(100 * pts / len(scored), 1)

    @property
    def all_findings(self) -> list[Finding]:
        out = [f for r in self.results for f in r.findings]
        return sorted(out, key=lambda f: f.severity.rank)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {s.value: 0 for s in Severity}
        for f in self.all_findings:
            counts[f.severity.value] += 1
        s = {
            "health_score": self.health_score_of(self.results),
            "compliance_score": self.compliance_score_of(self.compliance),
            "findings_by_severity": counts,
            "evaluators_passed": sum(r.passed for r in self.results if not r.skipped),
            "evaluators_run": sum(1 for r in self.results if not r.skipped),
            "verdict": self.verdict,
        }
        if self.post_fix_results:
            s["post_fix_health_score"] = self.health_score_of(self.post_fix_results)
            s["post_fix_compliance_score"] = self.compliance_score_of(self.post_fix_compliance)
        return s

    @property
    def verdict(self) -> str:
        sev = {f.severity for f in self.all_findings}
        if any(r.error for r in self.results) or Severity.CRITICAL in sev:
            return "BLOCK"
        if Severity.HIGH in sev or any(c.status == ControlStatus.FAIL for c in self.compliance):
            return "NEEDS_WORK"
        if Severity.MEDIUM in sev:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "use_case": self.use_case,
            "task": self.task,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
            "compliance": [c.to_dict() for c in self.compliance],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "fixes": [f.to_dict() for f in self.fixes],
            "post_fix_results": [r.to_dict() for r in self.post_fix_results],
            "post_fix_compliance": [c.to_dict() for c in self.post_fix_compliance],
            "metadata": self.metadata,
        }
