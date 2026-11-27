# Use cases & scenarios

Each scenario is a YAML file in `scenarios/` that describes **what system is tested, on which data, with which checks, against which regulations, and whether to auto-fix**. They're realistic on purpose: each one plants a known problem so you can see how it is detected, diagnosed and fixed. Then copy one and point it at your own system.

Run any of them with `python -m evalassure run scenarios/<file>.yaml`.

---

## Who uses what

| Role | Typical scenarios | What they get |
|---|---|---|
| Data scientist / ML engineer | 02, 03, 05, 06, 07, 08, 15 | metrics beyond accuracy, leakage & drift detection, verified fixes, A/B decisions |
| LLM / GenAI engineer | 09, 10, 11, 12 | quality + hallucination + safety + security in one run; guardrail impact measured |
| Automation / RPA / agent developer | 13, 14 | tool-trace testing, fault injection, idempotency, approval-gate checks |
| QA / test engineer | all; `--fail-on` in CI | repeatable suites, regression gates, HTML reports |
| Risk, compliance, audit | 01, 04, 09, 13 | control-by-control evidence (EU AI Act, NIST AI RMF, GDPR, ISO 42001, OWASP LLM) |
| Product owner / manager | any HTML report | verdict, health score, prioritised P0–P3 actions |

---

## ML scenarios

### 01 · Credit scoring fairness audit (high-risk credit decision)
- **Situation**: a bank's loan-approval model, trained on historical decisions that disadvantaged women. Gender is *not* a model input, but `part_time` correlates strongly with it (a proxy).
- **Checks**: data quality, classification, fairness (gender, age group), robustness, explainability; EU AI Act Art. 9–15, NIST AI RMF, GDPR Art. 22/35, ISO 42001, US four-fifths rule / ECOA adverse-action reasons.
- **What you'll see**: `DISPARATE_IMPACT` (female/male selection-rate ratio ≈ 0.73), `PROXY_FEATURE` (`part_time` ~ gender, top-3 importance), a diagnosis *"Bias flows through proxy features"*, and failed EU AI Act Art. 10 and four-fifths controls.
- **Auto-fix**: drops the proxy and retrains (gender DI goes from about 0.73 to 0.93), then applies demographic-parity thresholds for the remaining age-group gap. Each step is re-verified. The fairness fix includes a legal-review warning.
- **Adapt it**: use `data.path` with your CSV, list protected columns under `sensitive`, and point `model` at your model.

### 02 · Fraud detection: the accuracy paradox
- **Situation**: 1.5% of transactions are fraud. The model is "98.8% accurate" but catches only 20% of fraud.
- **What you'll see**: `ACCURACY_PARADOX`, `LOW_RECALL`, `CLASS_IMBALANCE`, and a diagnosis *"Imbalance is masking poor minority-class performance"*.
- **Auto-fix**: class re-weighting, then a threshold tuned on the validation split for best F1 with a recall floor (`fixes.threshold_tuning.min_recall: 0.6`). Health goes from 45 to 88.
- **Note**: `favorable_label: 0` tells the fairness check that being flagged is the *unfavourable* outcome.

### 03 · Churn model drift monitoring (MLOps)
- **Situation**: after a price rise, the production data (`monthly_charges`, tenure, tickets) shifted, and price sensitivity changed (concept drift). Last month's labels are available.
- **What you'll see**: `FEATURE_DRIFT` (PSI up to 1.4), `PREDICTION_DRIFT`, `PERFORMANCE_DECAY`, and a diagnosis *"Production decay driven by drift"*.
- **Auto-fix**: retrains on training data plus half of the recent window (with extra weight on recent rows) and evaluates on the other, held-out half, so the comparison is fair. The performance drop goes from 4.1 to 0.9 points. Drift findings remain on purpose: the world changed, and the report says so.
- **Adapt it**: set `data.production_path` to your latest scored and labelled batch, and run it monthly in CI.

### 04 · Medical triage (multiclass, healthcare, high-risk)
- **Situation**: a 3-level triage model. Labels for a minority group are noisier, and the model card is missing limitations, logging, DPIA, risk assessment and monitoring plan.
- **What you'll see**: `GROUP_PERFORMANCE_GAP` (worst group accuracy about 0.60), `EQUAL_OPPORTUNITY_GAP` for escalation to "high risk", `SINGLE_FEATURE_DEPENDENCY` on SpO2, and **seven failed controls** (EU AI Act Art. 9/11/12, GDPR Art. 35, ISO 42001 A.5, NIST MANAGE-4 ...).
- **No auto-fix** for the group gap: the right fix is better data for that group, and the recommendation says so. This scenario shows a compliance gap analysis.

### 05 · House price regression (automated valuation)
- R², MAPE, systematic bias, error disparity by neighbourhood, noise robustness, and feature-outage dependency (`sqft`).

### 06 · Data leakage detection (model validation gate)
- **Situation**: `post_decision_flag` (known only after the outcome) is in the features, 8% of rows are duplicated, and 3% of values are missing.
- **What you'll see**: `TARGET_LEAKAGE` (single-feature AUC = 1.0), `TRAIN_TEST_CONTAMINATION`, `SUSPICIOUSLY_HIGH_PERFORMANCE`, and a diagnosis *"Inflated metrics caused by leakage"*.
- **Auto-fix**: drops the leaking feature, de-duplicates, removes contaminated test rows and retrains. Accuracy drops to a realistic level, which is the correct outcome.

### 07 · Champion vs challenger A/B
- Paired bootstrap confidence interval and exact McNemar test on per-example correctness, plus deltas for F1, AUC and ECE, latency, size and disparate impact. Decision: `PROMOTE_CHALLENGER`, `KEEP_CHAMPION` or `INCONCLUSIVE`. A challenger that is less fair is blocked even if it is more accurate.

### 08 · Model compression analysis
- **Variants**: int8, int4 and int2 quantisation, plus 50% and 90% magnitude pruning, under a budget (≤ 0.5 pt accuracy drop, ≤ 1.5 pt worst-group drop, ≥ 98% agreement).
- **Output**: size, compression ratio, latency, accuracy drop, prediction agreement, probability shift, worst-affected group and the Pareto frontier.
- **Auto-fix**: a grid search (16 → 2 bits, 10–90% sparsity) finds the most compressed variant that stays within budget (about 5-bit, 3× smaller, in this demo).
- **For your models**: list your own compressed or distilled artefacts under `compressed_models:` in the scenario (each `{label, python: ...}` or `{label, pickle: ...}`), or compare LLM sizes with the `comparison` evaluator.

### 15 · Bring your own model (template)
- A custom Python model (`examples/custom_model.py`) trained on a CSV with categorical columns and missing values. Copy this file to start testing your own system.

---

## LLM scenarios

### 09 · RAG customer-support bot
- **Situation**: a support assistant answering from policy documents. The offline mock has realistic flaws: it hallucinates figures, leaks a customer email from a retrieved record, follows injected instructions (including from a "customer review"), writes gendered references, breaks JSON, and varies its answers.
- **Evaluators**: quality (token F1, ROUGE-L, keywords, optional LLM-as-judge), grounding (unsupported claims and numbers, abstention on unanswerable questions), safety (harmful-request refusal, over-refusal, toxicity, PII), security (canary leak, system-prompt leak, forbidden output, jailbreak), counterfactual bias, JSON validity, latency, cost and consistency.
- **Auto-fix**: stacks guardrails one at a time and keeps each only if it helps: injection guard → safety filter → PII redaction → grounding filter → counterfactual neutraliser → JSON repair → response cache. Health goes from 0 to 95.
- **Left for a human**: over-refusal of benign "kill a process" questions. That is a policy-tuning decision, so it gets a recommendation instead of an automatic change.

### 10 · Security red team
- Direct injection, indirect injection (in email and web content), system-prompt and secret extraction, DAN-style jailbreaks, harmful requests, and benign look-alikes ("signs of a phishing email") so that over-blocking is caught too.

### 11 · Claude (live)
- The same suite against a real model, using the official Anthropic SDK. Requires `pip install anthropic` and `ANTHROPIC_API_KEY` (or `ant auth login`). Server-side refusal fallbacks are on (`fallbacks: default`). Every response records which model served it, and `llm_performance` flags any answer a fallback model produced. Remove that line to evaluate the primary model's refusals alone.

### 12 · Local / open-weights model
- Any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, TGI, LiteLLM). This lets you compare candidate models on your own cases before choosing one.

---

## Automation scenarios

### 13 · Invoice-processing automation (v1)
- **Situation**: extract fields → validate the vendor → approve or escalate → post to the ERP. v1 has four realistic bugs.

| Bug | Finding | Fixed automatically? |
|---|---|---|
| `1.250,00 EUR` parsed as 1.25 | `FIELD_EXTRACTION_ERRORS` (amount), `TASK_FAILURES` | No. Logic bug; the diagnosis points to the parser |
| Flaky vendor/ERP API crashes the run | `FAULT_INTOLERANCE`, `WORKFLOW_CRASH` | Yes. Retry with back-off |
| Replayed webhook posts the invoice twice | `NON_IDEMPOTENT` | Yes. Idempotency keys |
| Pays a 48,000 invoice without approval | `EXCESSIVE_AGENCY`, `HUMAN_OVERSIGHT_BYPASS`, EU AI Act Art. 14 fail, OWASP LLM06 fail | No. Policy must be enforced in code |

### 14 · v2 regression gate
- The corrected workflow passes (`PASS`, 100/100). In CI: `evalassure run scenarios/14_*.yaml --fail-on high --no-autofix`.

---

## More use cases you can build from these templates

| Domain | Start from | Change |
|---|---|---|
| Hiring / CV screening (EU AI Act Annex III) | 01 | `sensitive: [gender, ethnicity, age_group]`, `domain: employment` |
| Insurance pricing | 05 | regression on premium, `sensitive: [age_group, region]` |
| Credit-limit or collections | 01 / 02 | binary target, favourable label |
| Predictive maintenance | 02 / 03 | imbalance + drift on sensor data |
| Medical imaging triage | 04 | your model's `predict_proba` on extracted features |
| HR / legal / medical chatbot | 09 | your own JSONL cases; add domain-specific `must_refuse` cases; `domain: healthcare` enables HIPAA checks |
| Code assistant | 09 / 10 | `expect_json` for tool-call formats; injection inside code comments |
| Email triage or ticket-routing agent | 13 | your workflow class; `forbidden_tools` for actions needing approval |
| Choosing between LLMs | 09 + `challenger:` | A/B on token F1 with a bootstrap CI, plus latency and token use |
| Shrinking a model for mobile | 08 | `compressed_models:` with your quantised / distilled artefacts |
