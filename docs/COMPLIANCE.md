# Compliance checks

> **Disclaimer**: aitesteval maps technical test evidence and model-card facts to published requirements. Its results support audits and readiness reviews. They are **not legal advice** and do not certify conformity. Applicability depends on your role (provider or deployer), jurisdiction, sector and risk classification.

Enable frameworks per scenario:

```yaml
compliance: [eu_ai_act, nist_ai_rmf, gdpr, iso_42001, owasp_llm, sector]
model_card:            # governance facts the checks use
  owner: team@example.com
  version: 1.2.0
  domain: credit               # credit | lending | employment | healthcare | medical | education | insurance ...
  use_case_type: creditworthiness_assessment
  risk_level: high             # high | limited | minimal | unacceptable
  intended_use: ...
  limitations: ...
  training_data: ...
  risk_assessment: link-or-summary
  human_oversight: who can override and how
  logging_enabled: true
  monitoring_plan: ...
  dpia_completed: true
  ai_disclosure: true          # users are told they talk to an AI (GenAI)
  generative_ai: true
  automated_decision: true
  affects_individuals: true
```

Each control returns **pass**, **fail**, **warn**, **manual_review** (no evidence; add the suggested evaluator) or **not_applicable** (for example, EU AI Act Art. 9–15 apply only when `risk_level: high` or the domain is a high-risk domain). The **compliance score** is (pass + 0.5 × warn) ÷ scored controls. Reports show controls whose status changed after auto-fix (❌ → ✅).

## Control catalogue

### EU AI Act (Regulation (EU) 2024/1689)
| Control | Requirement | Evidence used |
|---|---|---|
| EUAI-ART5 | Prohibited practices | `use_case_type` checked against the prohibited list |
| EUAI-ART9 | Risk management system | `risk_assessment` + evaluations executed |
| EUAI-ART10 | Data & data governance | `data_quality` (leakage, contamination) + `fairness` / proxy findings |
| EUAI-ART11 | Technical documentation (Annex IV) | model card: intended_use, limitations, training_data, owner, version |
| EUAI-ART12 | Record-keeping | `logging_enabled` |
| EUAI-ART13 | Transparency to deployers | explainability evidence + documented limitations |
| EUAI-ART14 | Human oversight | `human_oversight` and **automation tests** (`HUMAN_OVERSIGHT_BYPASS`, `EXCESSIVE_AGENCY` fail it) |
| EUAI-ART15 | Accuracy, robustness, cybersecurity | performance, robustness, drift decay, prompt injection, hallucination findings |
| EUAI-ART50 | Transparency for GenAI | `ai_disclosure` |

### NIST AI RMF 1.0
GOVERN-1 (accountable owner), MAP-1 (context and intended use), MEASURE-2.3 (performance), MEASURE-2.5 (validity and reliability: robustness, drift, consistency, fault tolerance), MEASURE-2.7 (security), MEASURE-2.9 (explainability), MEASURE-2.10 (privacy), MEASURE-2.11 (fairness), MANAGE-4 (post-deployment monitoring).

### GDPR
Art. 5(1)(c) data minimisation (protected attributes used as inputs), Art. 22 automated decisions (human intervention), Art. 32 security of processing (PII / system-prompt leakage in LLM outputs), Art. 35 DPIA (high-risk processing).

### ISO/IEC 42001 (AI management system), Annex A
A.5 impact assessment, A.6.2.4 verification and validation, A.6.2.6 operation and monitoring, A.7 data for AI systems.

### OWASP Top 10 for LLM Applications (2025)
LLM01 prompt injection, LLM02 sensitive information disclosure, LLM05 improper output handling, LLM06 excessive agency (automation), LLM07 system-prompt leakage, LLM09 misinformation, LLM10 unbounded consumption. (LLM03 supply chain, LLM04 poisoning and LLM08 vector/embedding weaknesses need pipeline-level review and are out of scope for black-box tests.)

### Sector rules (`sector`)
| Control | Rule | Applies when |
|---|---|---|
| US-EEOC-4/5 | Four-fifths adverse-impact rule (29 CFR 1607.4(D)); disparate-impact practice under ECOA / Reg B | classification tasks |
| US-ECOA-ADVERSE | Adverse-action reasons can be produced (Reg B 1002.9) | `domain: credit` or `lending` |
| HIPAA-PHI | No PHI in LLM outputs (45 CFR 164.502) | `domain: healthcare` or `medical`, LLM task |

## Adding your own controls

Append a `Control(...)` to a framework function in `aitesteval/compliance/frameworks.py`, or register a new framework:

```python
from aitesteval.compliance.frameworks import FRAMEWORKS, Control
from aitesteval.core.types import ControlStatus as S

def my_policy():
    return [Control("ACME-01", "ACME AI Policy", "Model must beat baseline by 5 pts", "Policy §3.2",
                    lambda e: (S.PASS, "ok", "") if (e.m("classification", "balanced_accuracy", 0) > 0.55)
                    else (S.FAIL, "below policy", "Improve model"))]

FRAMEWORKS["acme"] = my_policy   # then: compliance: [acme]
```

`Evidence` gives you `e.ran(...)`, `e.has(*finding_codes)`, `e.m(evaluator, metric)`, `e.card`, `e.missing(*fields)`, `e.high_risk`, `e.is_genai`.
