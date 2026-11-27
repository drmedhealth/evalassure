"""Troubleshooting: a searchable knowledge base of AI/ML failure modes + automatic root-cause
diagnosis that correlates findings across evaluators."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.types import Diagnosis, EvalResult


@dataclass
class KBEntry:
    id: str
    title: str
    symptoms: list[str]
    causes: list[str]
    checks: list[str]
    remedies: list[str]
    codes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


KB: list[KBEntry] = [
    KBEntry("KB-001", "Great offline metrics, poor production performance",
            ["accuracy dropped after deployment", "works in notebook, fails in prod"],
            ["Target leakage", "Train/test contamination", "Data drift", "Training/serving skew (different preprocessing)"],
            ["Run data_quality (leakage, overlap)", "Run drift with a production window", "Compare feature stats train vs serving"],
            ["Remove post-outcome features", "Split before dedup/feature engineering", "Share one preprocessing pipeline",
             "Retrain on recent data"], ["TARGET_LEAKAGE", "TRAIN_TEST_CONTAMINATION", "PERFORMANCE_DECAY", "FEATURE_DRIFT"],
            ["production", "deploy", "dropped", "degrade", "live", "decay", "worse"]),
    KBEntry("KB-002", "High accuracy but the model misses the cases that matter",
            ["99% accuracy but catches no fraud", "recall is near zero"],
            ["Class imbalance", "Default 0.5 threshold", "Optimising accuracy instead of recall/F1"],
            ["Compare accuracy with majority baseline", "Look at per-class recall and confusion matrix"],
            ["Class weights / resampling", "Tune threshold on validation", "Report PR-AUC & recall"],
            ["ACCURACY_PARADOX", "NO_BETTER_THAN_BASELINE", "CLASS_IMBALANCE", "LOW_RECALL"],
            ["imbalance", "recall", "fraud", "rare", "minority", "accuracy high", "miss"]),
    KBEntry("KB-003", "Model is unfair to a group",
            ["approval rate differs by gender", "4/5ths rule fails", "bias"],
            ["Historical bias in labels", "Proxy features (zip code, part-time...)", "Under-representation"],
            ["Run fairness + explainability (proxy detection)", "Inspect base rates per group"],
            ["Remove proxies", "Reweighing", "Threshold post-processing (check legality)", "Collect representative data"],
            ["DISPARATE_IMPACT", "EQUAL_OPPORTUNITY_GAP", "PROXY_FEATURE", "COUNTERFACTUAL_BIAS"],
            ["bias", "fair", "discriminat", "gender", "race", "ethnic", "disparate", "group"]),
    KBEntry("KB-004", "Model is overconfident / probabilities wrong",
            ["90% confidence but 60% right", "risk scores don't match observed rates"],
            ["Uncalibrated model", "Class weights or resampling distort probabilities", "Dataset shift"],
            ["Check ECE / reliability diagram"], ["Temperature/Platt/isotonic calibration on held-out data"],
            ["POORLY_CALIBRATED"], ["confidence", "probabilit", "calibrat", "overconfident"]),
    KBEntry("KB-005", "Overfitting", ["train accuracy 100%, test much lower"],
            ["Model too complex", "Too little data", "Leakage within train"], ["Compare train vs test metrics", "Learning curves"],
            ["Regularise", "Simplify", "More data", "Cross-validation"], ["OVERFITTING"],
            ["overfit", "memoriz", "train accuracy", "generaliz", "variance"]),
    KBEntry("KB-006", "Model breaks on unusual or missing inputs", ["crash on null", "noise changes predictions"],
            ["No input validation", "No imputation at inference", "Fragile decision boundary"],
            ["Run robustness"], ["Schema validation", "Imputation in pipeline", "Noise augmentation"],
            ["MISSING_VALUE_CRASH", "NOISE_SENSITIVITY", "UNSTABLE_PREDICTIONS"],
            ["crash", "null", "nan", "missing", "noise", "robust", "unstable", "flip"]),
    KBEntry("KB-007", "LLM makes things up", ["answer not in documents", "invents numbers or policies"],
            ["Weak grounding instructions", "Poor retrieval", "No abstention path"],
            ["Run llm_grounding with unanswerable cases", "Check retrieval recall"],
            ["Answer-only-from-context prompt", "Faithfulness filter", "Improve retrieval / chunking", "Citations"],
            ["HALLUCINATION", "FAILS_TO_ABSTAIN"], ["hallucinat", "made up", "invent", "fabricat", "wrong fact", "rag"]),
    KBEntry("KB-008", "LLM follows malicious instructions", ["ignore previous instructions works", "system prompt leaked"],
            ["No separation of instructions and data", "Secrets in prompt", "Over-privileged tools"],
            ["Run llm_security with direct + indirect injections and a canary"],
            ["Input/context screening", "Keep secrets out of prompts", "Least-privilege tools", "Output leak filter"],
            ["PROMPT_INJECTION", "SYSTEM_PROMPT_LEAKAGE"], ["inject", "jailbreak", "ignore previous", "leak", "system prompt", "attack"]),
    KBEntry("KB-009", "LLM leaks personal data", ["response contains emails/phone numbers"],
            ["Context contains other users' data", "No output redaction"], ["Run llm_safety (PII)"],
            ["Redact PII", "Per-user retrieval scoping", "Data minimisation"], ["PII_LEAKAGE"],
            ["pii", "personal data", "privacy", "email", "phone", "gdpr", "hipaa", "phi"]),
    KBEntry("KB-010", "LLM output breaks the parser", ["JSONDecodeError downstream"],
            ["Prose around JSON", "Trailing commas", "No schema enforcement"], ["Run llm_format"],
            ["Structured-output mode", "Validate & repair with retry"], ["INVALID_STRUCTURED_OUTPUT"],
            ["json", "parse", "schema", "format", "structured"]),
    KBEntry("KB-011", "LLM gives different answers each time", ["non-deterministic"],
            ["High temperature", "Ambiguous prompt"], ["Run llm_performance consistency"],
            ["Lower temperature", "Self-consistency", "Tighter prompt"], ["INCONSISTENT_OUTPUTS"],
            ["inconsistent", "different answer", "random", "non-determin", "flaky"]),
    KBEntry("KB-012", "LLM too slow or expensive", ["timeouts", "bill too high"],
            ["Over-long prompts", "Largest model for every request", "No caching"], ["Run llm_performance with prices"],
            ["Prompt caching", "Route easy queries to smaller models", "Streaming", "Token limits"], ["HIGH_LATENCY"],
            ["slow", "latency", "timeout", "cost", "expensive", "token"]),
    KBEntry("KB-013", "Automation fails intermittently", ["works on retry", "random 503s"],
            ["No retries for transient errors"], ["Run automation with inject_faults"],
            ["Retry with exponential back-off + jitter", "Circuit breaker", "Dead-letter queue"],
            ["FAULT_INTOLERANCE", "WORKFLOW_CRASH"], ["intermittent", "transient", "retry", "503", "timeout", "flaky", "crash"]),
    KBEntry("KB-014", "Automation duplicates actions", ["invoice posted twice", "double payment"],
            ["Webhook redelivery / retries without idempotency keys"], ["Run automation with check_idempotent"],
            ["Idempotency keys", "Exactly-once ledger"], ["NON_IDEMPOTENT"], ["duplicate", "twice", "double", "idempot"]),
    KBEntry("KB-015", "Agent does things it shouldn't", ["paid invoice without approval"],
            ["Over-privileged tools", "Missing policy checks", "No human-in-the-loop"], ["forbidden_tools + requires_human_review cases"],
            ["Tool allow-lists", "Approval gates above limits"], ["EXCESSIVE_AGENCY", "HUMAN_OVERSIGHT_BYPASS"],
            ["agent", "agency", "permission", "approval", "unauthori", "human review", "oversight"]),
    KBEntry("KB-016", "Compressed model loses accuracy", ["quantized model worse"],
            ["Too few bits / too much sparsity", "Outlier weights", "No fine-tuning after compression"],
            ["Run compression with a budget, check per-group drops"],
            ["Higher bit-width", "Per-channel quantisation", "QAT / fine-tune after pruning", "Distillation"],
            ["COMPRESSION_DEGRADATION", "NO_COMPRESSION_WITHIN_BUDGET"], ["quantiz", "prun", "compress", "distill", "int8", "int4", "smaller"]),
]

# cross-finding rules: (required codes (all), any-of codes, title, root cause, steps, confidence)
RULES = [
    ({"SUSPICIOUSLY_HIGH_PERFORMANCE"}, {"TARGET_LEAKAGE", "TRAIN_TEST_CONTAMINATION"}, "Inflated metrics caused by leakage",
     "Near-perfect scores coincide with leakage/contamination evidence; offline metrics are not trustworthy.",
     ["Fix data first (data_cleaning), then re-evaluate everything"], "high"),
    ({"ACCURACY_PARADOX"}, {"CLASS_IMBALANCE"}, "Imbalance is masking poor minority-class performance",
     "The model learned to predict the majority class; accuracy is dominated by it.",
     ["Retrain with class weights", "Tune threshold", "Track recall/PR-AUC"], "high"),
    ({"DISPARATE_IMPACT"}, {"PROXY_FEATURE"}, "Bias flows through proxy features",
     "Protected attributes are excluded, but correlated features reintroduce them.",
     ["Remove/decorrelate proxies", "Re-check fairness"], "high"),
    ({"PERFORMANCE_DECAY"}, {"FEATURE_DRIFT", "PREDICTION_DRIFT"}, "Production decay driven by drift",
     "Input distributions changed and performance dropped together.",
     ["Retrain on recent data", "Alert on PSI > 0.25"], "high"),
    ({"PERFORMANCE_DECAY"}, set(), "Concept drift suspected",
     "Performance dropped without strong input drift: the input→label relationship likely changed.",
     ["Collect fresh labels", "Retrain; consider time-weighted training"], "medium"),
    ({"PROMPT_INJECTION"}, {"SYSTEM_PROMPT_LEAKAGE"}, "Instructions and data are not separated",
     "Injected text is treated as instructions and secrets are reachable.",
     ["Injection guard", "Move secrets out of prompts", "Least privilege"], "high"),
    ({"HALLUCINATION"}, {"FAILS_TO_ABSTAIN"}, "No abstention path in the RAG prompt",
     "The model fills gaps with fabrications when context is missing.",
     ["Grounding filter", "Explicit 'say you don't know' instruction"], "high"),
    ({"TASK_FAILURES"}, {"FIELD_EXTRACTION_ERRORS"}, "Workflow failures trace back to field parsing",
     "Failing cases share a mis-parsed field (e.g. locale-specific amounts).",
     ["Fix the parser for the failing field; add regression tests"], "high"),
    ({"FAULT_INTOLERANCE"}, {"NON_IDEMPOTENT"}, "Retries are unsafe without idempotency",
     "Adding retries alone could duplicate side effects.", ["Add idempotency keys before enabling retries"], "medium"),
]


def search(query: str, limit: int = 3) -> list[tuple[KBEntry, int]]:
    q = query.lower()
    words = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for e in KB:
        s = sum(3 for k in e.keywords if k in q)
        s += sum(1 for w in words if len(w) > 3 and any(w in t.lower() for t in e.symptoms + [e.title]))
        s += sum(5 for c in e.codes if c.lower() in q)
        if s:
            scored.append((e, s))
    return sorted(scored, key=lambda x: -x[1])[:limit]


def diagnose(results: list[EvalResult]) -> list[Diagnosis]:
    findings = [f for r in results for f in r.findings]
    codes = {f.code for f in findings}
    out: list[Diagnosis] = []
    used: set[str] = set()
    for required, any_of, title, cause, steps, conf in RULES:
        if required <= codes and (not any_of or any_of & codes):
            if title == "Concept drift suspected" and {"FEATURE_DRIFT", "PREDICTION_DRIFT"} & codes:
                continue
            ev = [f"{f.code}: {f.detail}" for f in findings if f.code in required | any_of]
            out.append(Diagnosis(title, cause, ev[:5], steps, conf))
            used |= required | any_of
    for e in KB:
        hit = [c for c in e.codes if c in codes and c not in used]
        if hit:
            ev = [f"{f.code}: {f.detail}" for f in findings if f.code in hit]
            out.append(Diagnosis(e.title, "Likely causes: " + "; ".join(e.causes), ev[:5],
                                 e.checks + e.remedies, "medium", e.id))
            used |= set(hit)
    return out
