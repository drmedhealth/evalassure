"""Turn findings and failed compliance controls into prioritised, actionable recommendations."""
from __future__ import annotations

from ..core.types import ComplianceResult, ControlStatus, Finding, Recommendation, Severity

# code -> (title, rationale, actions, auto_fix_id)
PLAYBOOK: dict[str, tuple[str, str, list[str], str | None]] = {
    "LOW_ACCURACY": ("Improve predictive performance", "Model misses its accuracy target.",
                     ["Add informative features / more data", "Try a higher-capacity model and tune hyper-parameters",
                      "Check label quality"], None),
    "LOW_F1": ("Improve precision/recall balance", "F1 below target.",
               ["Tune the decision threshold on a validation set", "Use class weights for imbalanced data"], "threshold_tuning"),
    "LOW_RECALL": ("Increase recall of the positive class", "Too many positives are missed.",
                   ["Lower the decision threshold", "Class-weighted training / oversampling"], "threshold_tuning"),
    "NO_BETTER_THAN_BASELINE": ("Model adds no value over a trivial baseline", "Predicting the majority class is as good.",
                                ["Use class-balanced training", "Optimise for recall/F1, not accuracy"], "class_reweighting"),
    "ACCURACY_PARADOX": ("Stop using accuracy as the headline metric", "Accuracy hides minority-class failure.",
                         ["Report balanced accuracy, recall, PR-AUC", "Retrain with class weights", "Tune threshold for F1"],
                         "class_reweighting"),
    "POORLY_CALIBRATED": ("Calibrate probabilities", "Scores are used as probabilities/risk but are miscalibrated.",
                          ["Fit temperature/Platt scaling or isotonic regression on a held-out set"], "calibration"),
    "OVERFITTING": ("Reduce overfitting", "Model memorises the training data.",
                    ["Stronger regularisation", "Simpler model / fewer features", "More training data, cross-validation"],
                    "regularization"),
    "SUSPICIOUSLY_HIGH_PERFORMANCE": ("Audit for target leakage", "Near-perfect metrics usually mean leakage.",
                                      ["Check features computed after the outcome", "Check train/test row overlap"], "data_cleaning"),
    "TARGET_LEAKAGE": ("Remove leaking features", "A feature encodes the label; production performance will collapse.",
                       ["Drop features unavailable at prediction time", "Rebuild the feature pipeline with point-in-time joins"],
                       "data_cleaning"),
    "TRAIN_TEST_CONTAMINATION": ("Deduplicate before splitting", "Test rows seen in training inflate metrics.",
                                 ["Drop duplicates before splitting", "Use group/time-based splits"], "data_cleaning"),
    "DUPLICATE_ROWS": ("Deduplicate the dataset", "Duplicates bias training and evaluation.", ["Drop exact duplicates"],
                       "data_cleaning"),
    "MISSING_VALUES": ("Handle missing values explicitly", "Missing data can crash or bias models.",
                       ["Impute (median/most-frequent) inside the pipeline", "Add missing-indicator features"], None),
    "CLASS_IMBALANCE": ("Address class imbalance", "Minority class is under-represented.",
                        ["Class weights / focal loss", "Stratified sampling", "Evaluate with PR-AUC"], "class_reweighting"),
    "DISPARATE_IMPACT": ("Mitigate disparate impact", "Selection rates fail the 4/5ths rule - legal & ethical risk.",
                         ["Remove proxy features", "Reweighing (pre-processing)", "Group-aware thresholds (post-processing) "
                          "- confirm legality in your jurisdiction", "Document business necessity"], "fairness_mitigation"),
    "EQUAL_OPPORTUNITY_GAP": ("Equalise true-positive rates", "Qualified people in some groups are rejected more often.",
                              ["Equal-opportunity post-processing", "Collect more data for under-served groups"], "fairness_mitigation"),
    "DEMOGRAPHIC_PARITY_GAP": ("Reduce selection-rate gap", "Outcomes differ by group.", ["See disparate-impact actions"],
                               "fairness_mitigation"),
    "GROUP_PERFORMANCE_GAP": ("Improve worst-group performance", "Model is less accurate for some groups.",
                              ["Collect more/cleaner data for the worst group", "Group-balanced training"], None),
    "PROXY_FEATURE": ("Remove or constrain proxy features", "An influential feature stands in for a protected attribute.",
                      ["Drop the proxy or decorrelate it", "Re-run fairness after removal"], "drop_proxy_features"),
    "SENSITIVE_FEATURE_USED": ("Stop using protected attributes as inputs", "Direct use creates legal risk (GDPR, ECOA).",
                               ["Remove the attribute from features", "Keep it only for fairness auditing"], None),
    "NOISE_SENSITIVITY": ("Make the model robust to noisy inputs", "Small perturbations degrade performance.",
                          ["Noise augmentation", "Stronger regularisation", "Input validation & clipping"], "robust_training"),
    "UNSTABLE_PREDICTIONS": ("Stabilise decisions near the boundary", "Decisions flip under tiny changes.",
                             ["Regularise", "Add an abstain/review band around the threshold"], "robust_training"),
    "SINGLE_FEATURE_DEPENDENCY": ("Reduce reliance on a single feature", "An outage of one input breaks the model.",
                                  ["Add fallback features", "Monitor input availability", "Feature dropout during training"], None),
    "MISSING_VALUE_CRASH": ("Handle missing inputs at inference", "The model crashes on nulls.",
                            ["Add imputation to the inference pipeline", "Validate schemas at the API boundary"], None),
    "FEATURE_DRIFT": ("Respond to data drift", "Production inputs differ from training data.",
                      ["Investigate upstream changes", "Retrain on recent data", "Set PSI alerts"], "retrain_recent"),
    "PERFORMANCE_DECAY": ("Retrain: production performance has decayed", "Model is degrading on live data.",
                          ["Retrain with recent labelled data", "Add continuous monitoring and retraining triggers"], "retrain_recent"),
    "PREDICTION_DRIFT": ("Investigate prediction drift", "Output distribution has shifted.", ["Check inputs & label shift"], None),
    "HALLUCINATION": ("Ground answers in retrieved context", "The model asserts facts not in the sources.",
                      ["Grounding/faithfulness filter", "Instruct to cite & abstain", "Improve retrieval"], "grounding_guard"),
    "FAILS_TO_ABSTAIN": ("Teach the model to say 'I don't know'", "It answers unanswerable questions.",
                         ["System prompt: answer only from context", "Grounding filter"], "grounding_guard"),
    "HARMFUL_COMPLIANCE": ("Block harmful requests", "Model provides harmful assistance.",
                           ["Input safety classifier", "Stronger safety system prompt", "Use a safety-tuned model"], "safety_guard"),
    "OVER_REFUSAL": ("Reduce over-refusal", "Legitimate users are refused.", ["Tune refusal policy with benign edge cases"], None),
    "TOXIC_OUTPUT": ("Filter toxic language", "Responses contain abusive language.", ["Output toxicity filter"], "toxicity_filter"),
    "PII_LEAKAGE": ("Prevent personal-data disclosure", "Responses expose PII (GDPR/HIPAA risk).",
                    ["Output PII redaction", "Scope retrieval to the requesting user", "Strip PII from context"], "pii_redaction"),
    "PROMPT_INJECTION": ("Defend against prompt injection", "Attackers can override instructions.",
                         ["Input & retrieved-content screening", "Privilege separation for tools", "Output leak checks"],
                         "injection_guard"),
    "SYSTEM_PROMPT_LEAKAGE": ("Keep secrets out of prompts", "System prompt contents can be extracted.",
                              ["Never put credentials in prompts", "Output leak detection"], "injection_guard"),
    "COUNTERFACTUAL_BIAS": ("Remove demographic bias in generations", "Answers differ by gender/ethnicity.",
                            ["Bias-aware system prompt", "Counterfactual data augmentation", "Neutralising post-filter"],
                            "counterfactual_fairness"),
    "INVALID_STRUCTURED_OUTPUT": ("Enforce the output schema", "Downstream parsers will break.",
                                  ["Use structured-output / JSON mode", "Validate & repair, retry on failure"], "json_repair"),
    "INCONSISTENT_OUTPUTS": ("Make answers deterministic", "Same question, different answers.",
                             ["Lower temperature", "Self-consistency voting", "Cache answers to repeated questions"], "response_cache"),
    "HIGH_LATENCY": ("Reduce latency", "p95 exceeds SLO.", ["Smaller/faster model for easy queries", "Prompt caching",
                                                           "Streaming", "Shorter prompts"], None),
    "GENERATION_ERRORS": ("Fix failing model calls", "API errors/timeouts.", ["Check credentials/quotas", "Retries with back-off"], None),
    "TASK_FAILURES": ("Fix workflow correctness bugs", "Automation produces wrong results.",
                      ["Inspect failing cases & field errors", "Add unit tests per step"], None),
    "FIELD_EXTRACTION_ERRORS": ("Fix field parsing/extraction", "A specific field is often wrong.",
                                ["Locale-aware number/date parsing", "Validate extracted values against schema"], None),
    "WORKFLOW_CRASH": ("Add error handling to the workflow", "Unhandled exceptions stop processing.",
                       ["Catch & classify errors", "Dead-letter queue for failed items"], "workflow_retry"),
    "FAULT_INTOLERANCE": ("Retry transient failures", "Flaky dependencies cause failures.",
                          ["Retry with exponential back-off & jitter", "Circuit breaker"], "workflow_retry"),
    "NON_IDEMPOTENT": ("Make side effects idempotent", "Replays duplicate payments/postings.",
                       ["Idempotency keys on side-effecting calls", "Exactly-once ledger"], "workflow_idempotency"),
    "EXCESSIVE_AGENCY": ("Restrict the agent's permissions", "It performs actions it must not take.",
                         ["Least-privilege tool allow-list", "Approval gate for high-impact actions"], None),
    "HUMAN_OVERSIGHT_BYPASS": ("Enforce human-in-the-loop", "High-impact cases skipped review (EU AI Act Art. 14).",
                               ["Hard policy check before side effects", "Escalate above approval limits"], None),
    "TOOL_SEQUENCE_ERRORS": ("Fix tool orchestration", "Steps are skipped or out of order.", ["Explicit state machine"], None),
    "COMPRESSION_DEGRADATION": ("Use a less aggressive compression setting", "Quality budget exceeded.",
                                ["Higher bit-width / lower sparsity", "Quantisation-aware training or fine-tune after pruning"],
                                "compression_search"),
    "NO_COMPRESSION_WITHIN_BUDGET": ("Search for a compliant compression setting", "No variant met the budget.",
                                     ["Finer-grained search", "Distillation"], "compression_search"),
    "COMPRESSION_DISPARATE_DEGRADATION": ("Check compression impact per group", "Compression hurts some groups more.",
                                          ["Evaluate per-group before shipping compressed models"], None),
    "CHALLENGER_WORSE": ("Keep the champion", "Challenger is significantly worse.", ["Do not promote"], None),
    "CHALLENGER_FAIRNESS_REGRESSION": ("Block challenger promotion", "Fairness regresses.", ["Mitigate bias first"], None),
}

_PRIORITY = {Severity.CRITICAL: "P0", Severity.HIGH: "P1", Severity.MEDIUM: "P2", Severity.LOW: "P3", Severity.INFO: "P3"}


def recommend(findings: list[Finding], compliance: list[ComplianceResult]) -> list[Recommendation]:
    recs: dict[str, Recommendation] = {}
    for f in sorted(findings, key=lambda x: x.severity.rank):
        if f.severity == Severity.INFO and f.code not in PLAYBOOK:
            continue
        title, why, actions, fix = PLAYBOOK.get(f.code, (f.title, f.detail, ["Investigate this finding"], None))
        if f.code in recs:
            recs[f.code].related.append(f.title)
            continue
        recs[f.code] = Recommendation(code=f.code, title=title, priority=_PRIORITY[f.severity],
                                      rationale=f"{why} Evidence: {f.detail}".strip(), actions=list(actions),
                                      auto_fix=fix, related=[f.title])
    for c in compliance:
        if c.status == ControlStatus.FAIL:
            code = f"COMPLIANCE_{c.control_id}"
            recs[code] = Recommendation(code=code, title=f"[{c.framework}] {c.title}", priority="P1",
                                        rationale=f"{c.reference}: {c.evidence}",
                                        actions=[c.remediation] if c.remediation else ["Review requirement"])
    return sorted(recs.values(), key=lambda r: r.priority)
