# Troubleshooting

There are three ways to get help:

1. **Describe the symptom**: `python -m evalassure troubleshoot "accuracy is high but we miss fraud"` searches the built-in knowledge base (16 failure modes).
2. **Diagnose a report**: `python -m evalassure troubleshoot --report reports/<scenario>.json` correlates findings across evaluators into root causes (for example, *near-perfect AUC + duplicate rows → leakage*).
3. **Read the report**: every report has a *Root-cause diagnosis* section and prioritised *Recommendations*.

## AI/ML failure modes: symptom → cause → fix

| Symptom | Likely root cause | Confirm with | Fix (auto-fix id) |
|---|---|---|---|
| Great offline metrics, poor in production | target leakage, train/test contamination, drift, training/serving skew | `data_quality`, `drift` with `production_path` | remove post-outcome features, split before dedup (`data_cleaning`); retrain (`retrain_recent`) |
| 99% accuracy, misses the cases that matter | class imbalance, default 0.5 threshold | per-class recall, balanced accuracy | `class_reweighting`, `threshold_tuning` (with `min_recall`) |
| Approval rates differ by group | historical label bias, proxy features | `fairness` + `explainability` (proxy detection) | `drop_proxy_features`, `fairness_mitigation` (legal review) |
| Probabilities don't match observed rates | miscalibration (often after re-weighting) | ECE in `classification` | `calibration` |
| Train ≫ test | overfitting | `train_test_gap` | `regularization` |
| Predictions flip with tiny changes / crash on nulls | fragile boundary, no imputation at inference | `robustness` | `robust_training`; add imputation and schema validation |
| LLM invents facts | weak grounding, no abstention path | `llm_grounding` with `answerable: false` cases | `grounding_guard`; "answer only from context" prompt; better retrieval |
| LLM follows injected instructions / leaks prompt | instructions and data not separated, secrets in prompt | `llm_security` with canary | `injection_guard`; move secrets out of prompts; least-privilege tools |
| LLM shows emails or phone numbers | over-broad retrieval, no output redaction | `llm_safety` | `pii_redaction`; per-user retrieval scoping |
| JSON parse errors downstream | prose around JSON, trailing commas | `llm_format` | `json_repair`; structured-output mode |
| Different answers to the same question | sampling temperature, ambiguous prompt | `llm_performance.consistency` | `response_cache`, `self_consistency`; lower temperature |
| Refuses legitimate questions | over-broad safety policy | `llm_safety.over_refusal_rate` | tune the policy with benign edge cases (human decision) |
| Automation fails intermittently | no retries | `automation` + `inject_faults` | `workflow_retry` |
| Duplicate payments or postings | no idempotency keys | `check_idempotent: true` | `workflow_idempotency` (add **before** enabling retries) |
| Agent acts without approval | missing policy gate, over-privileged tools | `forbidden_tools`, `requires_human_review` | code change: enforce approval before side effects |
| Compressed model much worse | too few bits / too much sparsity | `compression` with a budget | `compression_search`; per-channel quantisation, QAT, distillation |

## Tool errors

| Error | Meaning / fix |
|---|---|
| `Invalid scenario: missing required key 'model'` | Every scenario needs `name`, `task`, `model` (+ `data` for ML, `llm.cases` for LLM, `automation.cases` for automation). Run `validate`. |
| `unknown evaluator 'x'` | Check spelling: `python -m evalassure list evaluators`. |
| `Cannot find 'datasets/...'` | Paths resolve relative to the scenario folder, then its parent, then the current directory. |
| `target column 'y' not found` | `data.target` must match the CSV header exactly; the error lists available columns. |
| `skipped: model is not retrainable` | Retraining fixes need a built-in or `python:` model with `fit`. For pickles, apply the recommendation in your own pipeline. |
| `skipped: binary classifier with predict_proba required` | Threshold and fairness post-processing need probabilities for 2 classes. |
| `rolled back: no net improvement` | The fix was tried, re-evaluated and undone because overall health did not improve. This is expected behaviour. |
| `not needed: issue already resolved by an earlier fix` | An earlier fix in the chain already removed the targeted finding. |
| `ImportError: Install the Anthropic SDK` | `pip install anthropic`, then set `ANTHROPIC_API_KEY` or run `ant auth login`. |
| `GENERATION_ERRORS` finding with HTTP 401/429 | Credentials or rate limits on the LLM endpoint; reduce `consistency_cases` or add retries in your adapter. |
| `UnicodeEncodeError` on Windows | The CLI already tolerates it; in your own scripts call `sys.stdout.reconfigure(errors="replace")` or set `PYTHONUTF8=1`. |
| Results differ between runs | Built-in data and mocks are seeded. If you use real LLMs, outputs are stochastic; use the `consistency` metric and run more cases. |
