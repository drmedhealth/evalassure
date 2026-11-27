"""Use aitesteval from Python (notebooks, pipelines, your own services).

Run:  python examples/quickstart_python_api.py
"""
import sys
from pathlib import Path

import numpy as np

from aitesteval import run_scenario
from aitesteval.core.registry import Evaluator, register_evaluator
from aitesteval.core.runner import Runner
from aitesteval.models.llm import LLMResponse
from aitesteval.reporting import to_markdown, write_reports

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(errors="replace")  # Windows consoles: tolerate emoji in reports

# --------------------------------------------------------------------------- 1. run a ready-made scenario
report = run_scenario(ROOT / "scenarios/02_fraud_detection_imbalance.yaml", progress=print)
print(report.summary())
for f in report.all_findings[:5]:
    print(f.severity.value, f.code, f.detail)


# --------------------------------------------------------------------------- 2. add your own rule
@register_evaluator("business_rule_min_approvals")
class MinApprovalRate(Evaluator):
    """Example custom check: the business needs at least 40% of applications approved."""

    tasks = ("classification",)
    description = "Approval rate must stay above a business minimum."

    def evaluate(self, ctx, result):
        rate = float(np.mean(ctx.predict("test") == ctx.classes[-1]))
        result.metrics["approval_rate"] = rate
        if rate < self.cfg("min_rate", 0.4):
            self.finding(result, "APPROVAL_RATE_TOO_LOW", "Approval rate below business minimum", "medium",
                         f"{rate:.1%} approved", "approval_rate", rate, self.cfg("min_rate", 0.4))


# --------------------------------------------------------------------------- 3. define a scenario in code
scenario = {
    "name": "In-code credit check",
    "task": "classification",
    "data": {"generator": "credit", "params": {"n": 3000, "bias": 0.5}},
    "model": {"builtin": "logreg"},
    "evaluators": ["classification", {"name": "fairness", "sensitive": ["gender"]},
                   {"name": "business_rule_min_approvals", "min_rate": 0.45}],
    "compliance": ["nist_ai_rmf"],
    "autofix": True,
}
rep2 = Runner(scenario, base_dir=ROOT / "scenarios").run()
print(to_markdown(rep2)[:1200])


# --------------------------------------------------------------------------- 4. evaluate ANY LLM function
def my_llm(prompt: str, system: str | None = None, context: str | None = None) -> str:
    """Replace with a call to your model / API / agent. Returning a str is enough."""
    if context:
        return "According to the context: " + context.split(".")[0] + "."
    return "I'm sorry, but I can't help with that request." if "hack" in prompt.lower() else "Happy to help!"


sys.modules["quickstart_llm"] = sys.modules[__name__]  # make `my_llm` importable by reference below
llm_scenario = {
    "name": "My LLM function",
    "task": "llm",
    "llm": {"cases": str(ROOT / "datasets/llm/support_bot.jsonl")},
    "model": {"python": "quickstart_llm:my_llm"},
    "evaluators": ["llm_quality", "llm_grounding", "llm_safety", "llm_security"],
    "compliance": ["owasp_llm"],
}
rep3 = Runner(llm_scenario, base_dir=ROOT / "scenarios").run(autofix=True)
paths = write_reports(rep3, ROOT / "reports")
print(rep3.summary(), paths)
_ = LLMResponse  # you may also return LLMResponse(text=..., latency_s=..., input_tokens=..., output_tokens=...)
