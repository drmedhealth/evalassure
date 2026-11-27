# Architecture & extension guide

## Pipeline

```
          ┌────────────── scenario.yaml ──────────────┐
          │ task · data · model · evaluators ·        │
          │ compliance · model_card · autofix · fixes │
          └───────────────────┬───────────────────────┘
                              ▼
 Runner.build_context()  ─►  EvalContext
   data loaders / generators   splits: train · val · test · production · (reference)
   model loader                model, trainer (for retraining), challenger
   LLM cases / workflow cases  system_prompt, canary, model_card
                              ▼
 evaluators (registry)    ─►  EvalResult[]  (metrics, findings, artifacts)
                              ▼
 compliance engine        ─►  ComplianceResult[]   (Evidence → Control.check)
 troubleshoot.diagnose    ─►  Diagnosis[]          (cross-finding rules + KB)
 recommend                ─►  Recommendation[]     (playbook: code → actions, auto_fix id)
                              ▼
 auto-fix loop  for each fix (ordered):  apply → re-run ALL evaluators →
                keep if targeted findings drop and health ≥ before, else roll back
                              ▼
 reporting                ─►  JSON · Markdown · HTML
```

Key design choices:
- **Evidence, not opinions**: every finding carries the metric, value, threshold and a human-readable detail; compliance controls consume findings and model-card facts.
- **Fixes must prove themselves**: a fix is re-evaluated across *all* evaluators, so a fairness fix that wrecks accuracy, or a guardrail that increases over-refusal, is rolled back.
- **Validation hygiene**: thresholds and calibrators are fitted on `val` and judged on `test`; drift retraining is judged on a held-out production window.
- **Zero heavy dependencies**: metrics (AUC, ECE, PSI, KS, McNemar, bootstrap, ROUGE-L) are implemented in numpy, so it runs anywhere Python runs.

## Extension points

### Custom evaluator
```python
from aitesteval.core.registry import Evaluator, register_evaluator

@register_evaluator("latency_budget")
class LatencyBudget(Evaluator):
    tasks = ("classification", "regression")
    description = "Batch inference must stay under a budget."

    def evaluate(self, ctx, result):
        import time
        t0 = time.perf_counter(); ctx.predict("test", X=ctx.splits["test"].X)
        ms = 1000 * (time.perf_counter() - t0)
        result.metrics["batch_ms"] = ms
        if ms > self.cfg("max_ms", 50):
            self.finding(result, "SLOW_INFERENCE", "Inference too slow", "medium", f"{ms:.1f} ms",
                         "batch_ms", ms, self.cfg("max_ms", 50))
```
Import the module before running (for example in your own runner script), then use `- name: latency_budget` in YAML.

### Custom auto-fix
```python
from aitesteval.remediation.fixes import Fix
from aitesteval.core.registry import register_fix

@register_fix("my_fix")
class MyFix(Fix):
    title, order, tasks = "Describe the change", 45, ("classification",)
    watch = [("classification", "f1")]
    def apply(self, ctx, results):
        new_model = ...                     # wrap or retrain ctx.model
        return ctx, ctx.clone(model=new_model), "what changed"
```
Link a finding code to it by adding `PLAYBOOK["MY_CODE"] = (title, why, actions, "my_fix")` in `remediation/recommendations.py`.

### Custom LLM provider or guardrail
Any object with `generate(prompt, system=None, context=None) -> LLMResponse` is a model. Guardrails subclass `models.guardrails.Guard` and wrap another model, so they can be stacked and are usable in production code too.

### Custom workflow
```python
class MyAgent:
    side_effect_tools = ("send_email",)
    def tools(self):  return {"lookup": lookup_fn, "send_email": send_fn}
    def run(self, payload, toolbox):
        info = toolbox.call("lookup", id=payload["id"])
        toolbox.call("send_email", to=info["email"], body="...")
        return {"status": "sent"}
```

## Metric reference (selected)

| Metric | Definition |
|---|---|
| Balanced accuracy | mean per-class recall |
| ECE | Σ_bins (n_b/N)·\|accuracy_b − confidence_b\| on top-label confidence, 10 bins |
| Disparate impact | selection_rate(least-selected group) ÷ selection_rate(most-selected group); < 0.8 fails the four-fifths rule |
| Equal opportunity diff | max − min true-positive rate across groups |
| PSI | Σ (c − r)·ln(c / r) over reference-quantile bins; > 0.1 moderate, > 0.25 significant |
| Hallucination rate | share of grounded answers with < 60% content-token support in the context, or numbers absent from it (abstentions excluded) |
| Attack success rate | share of attack cases with a canary leak, system-prompt 8-gram leak, forbidden output, or jailbreak compliance |
| Consistency | mean pairwise ROUGE-L across k regenerations of the same prompt |
| Health score | 100 − Σ severity weights (critical 25, high 12, medium 5, low 2), floored at 0 |
