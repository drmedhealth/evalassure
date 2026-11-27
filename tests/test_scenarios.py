"""End-to-end: every offline scenario must detect its planted problem, and auto-fix must help."""
from pathlib import Path

import pytest

from aitesteval import run_scenario
from aitesteval.reporting import to_html, to_json, to_markdown

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scenarios"


def codes(results):
    return {f.code for r in results for f in r.findings}


@pytest.mark.parametrize("name, expected, must_improve", [
    ("01_credit_scoring_fairness.yaml", {"DISPARATE_IMPACT", "PROXY_FEATURE"}, True),
    ("02_fraud_detection_imbalance.yaml", {"ACCURACY_PARADOX", "LOW_RECALL"}, True),
    ("03_churn_drift_monitoring.yaml", {"FEATURE_DRIFT", "PERFORMANCE_DECAY"}, True),
    ("04_medical_triage_multiclass.yaml", {"GROUP_PERFORMANCE_GAP"}, False),
    ("06_data_leakage_detection.yaml", {"TARGET_LEAKAGE", "TRAIN_TEST_CONTAMINATION"}, True),
    ("07_model_comparison_ab.yaml", {"CHALLENGER_WORSE"}, False),
    ("08_model_compression.yaml", {"COMPRESSION_DEGRADATION"}, True),
    ("09_llm_rag_support_bot.yaml", {"HALLUCINATION", "PII_LEAKAGE", "PROMPT_INJECTION", "COUNTERFACTUAL_BIAS",
                                     "INVALID_STRUCTURED_OUTPUT"}, True),
    ("10_llm_security_redteam.yaml", {"PROMPT_INJECTION", "SYSTEM_PROMPT_LEAKAGE", "HARMFUL_COMPLIANCE"}, True),
    ("13_automation_invoice_agent.yaml", {"EXCESSIVE_AGENCY", "HUMAN_OVERSIGHT_BYPASS", "NON_IDEMPOTENT",
                                          "FAULT_INTOLERANCE"}, True),
])
def test_scenario_detects_and_fixes(name, expected, must_improve):
    rep = run_scenario(S / name)
    found = codes(rep.results)
    assert expected <= found, f"missing {expected - found}"
    if must_improve:
        s = rep.summary()
        assert s["post_fix_health_score"] > s["health_score"]
        assert codes(rep.post_fix_results).isdisjoint({"PROMPT_INJECTION", "PII_LEAKAGE", "TARGET_LEAKAGE"})
    # all report formats render
    assert to_json(rep) and to_markdown(rep) and "<html" in to_html(rep)


def test_fixed_workflow_passes_regression_gate():
    rep = run_scenario(S / "14_automation_invoice_v2_regression.yaml")
    assert rep.verdict == "PASS"


def test_compliance_flags_missing_governance():
    rep = run_scenario(S / "04_medical_triage_multiclass.yaml", autofix=False)
    failed = {c.control_id for c in rep.compliance if c.status.value == "fail"}
    assert {"EUAI-ART11", "EUAI-ART12", "GDPR-ART35"} <= failed
