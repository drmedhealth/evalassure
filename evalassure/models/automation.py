"""Test harness for AI automation / agent workflows.

A workflow is any object with:
    tools() -> dict[str, callable]          # the tools/APIs the workflow may call
    run(payload: dict, toolbox: ToolBox) -> dict
The ToolBox records every call, injects faults and tracks side effects, so evaluators can
check task success, tool-use correctness, excessive agency, fault tolerance and idempotency.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class TransientError(RuntimeError):
    """Simulated flaky dependency (timeout, 503, rate limit)."""


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class ToolBox:
    tools: dict[str, Callable[..., Any]]
    faults: dict[str, dict[str, Any]] = field(default_factory=dict)  # {tool: {type: transient, times: 1}}
    ledger: list[dict[str, Any]] = field(default_factory=list)  # shared side-effect ledger (persists across runs)
    side_effect_tools: tuple[str, ...] = ()
    calls: list[ToolCall] = field(default_factory=list)
    _fault_counts: dict[str, int] = field(default_factory=dict)

    def call(self, name: str, **args: Any) -> Any:
        if name not in self.tools:
            self.calls.append(ToolCall(name, args, ok=False, error="unknown tool"))
            raise KeyError(f"Tool '{name}' is not available to this workflow")
        t0 = time.perf_counter()
        fault = self.faults.get(name)
        if fault and self._fault_counts.get(name, 0) < int(fault.get("times", 1)):
            self._fault_counts[name] = self._fault_counts.get(name, 0) + 1
            self.calls.append(ToolCall(name, args, ok=False, error="TransientError"))
            raise TransientError(f"{name}: simulated {fault.get('type', 'transient')} failure")
        try:
            out = self.tools[name](**args)
        except Exception as exc:
            self.calls.append(ToolCall(name, args, ok=False, error=str(exc), duration_s=time.perf_counter() - t0))
            raise
        self.calls.append(ToolCall(name, args, ok=True, duration_s=time.perf_counter() - t0))
        if name in self.side_effect_tools:
            self.ledger.append({"tool": name, "args": args})
        return out

    @property
    def tool_sequence(self) -> list[str]:
        return [c.name for c in self.calls if c.ok]


# ------------------------------------------------------------------ middleware (used by fixes)

class RetryToolBox:
    """Retries transient tool failures with exponential back-off (no real sleep in tests)."""

    def __init__(self, inner: Any, retries: int = 3) -> None:
        self.inner, self.retries = inner, retries

    def call(self, name: str, **args: Any) -> Any:
        for attempt in range(self.retries + 1):
            try:
                return self.inner.call(name, **args)
            except TransientError:
                if attempt == self.retries:
                    raise
        return None  # pragma: no cover

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


class IdempotentToolBox:
    """Deduplicates side-effecting calls using an idempotency key (tool + args hash)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def call(self, name: str, **args: Any) -> Any:
        if name in self.inner.side_effect_tools:
            key = hashlib.sha256((name + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()
            for e in self.inner.ledger:
                if e.get("key") == key:
                    return e.get("result", {"status": "duplicate_suppressed"})
            out = self.inner.call(name, **args)
            self.inner.ledger[-1]["key"] = key
            self.inner.ledger[-1]["result"] = out
            return out
        return self.inner.call(name, **args)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


class WorkflowWithMiddleware:
    """Wraps a workflow so its toolbox goes through middleware (retry, idempotency...)."""

    def __init__(self, inner: Any, middleware: list[str]) -> None:
        self.inner, self.middleware = inner, middleware
        self.side_effect_tools = getattr(inner, "side_effect_tools", ())
        self.name = getattr(inner, "name", "workflow") + "+" + "+".join(middleware)

    def tools(self) -> dict[str, Callable[..., Any]]:
        return self.inner.tools()

    def run(self, payload: dict[str, Any], toolbox: Any) -> dict[str, Any]:
        tb = toolbox
        if "idempotency" in self.middleware:
            tb = IdempotentToolBox(tb)
        if "retry" in self.middleware:
            tb = RetryToolBox(tb)
        return self.inner.run(payload, tb)


# ------------------------------------------------------------------ demo workflow

VENDORS = {"ACME Corp": True, "Globex": True, "Initech": True, "Umbrella Ltd": False}


def parse_amount_v1(s: str) -> float:
    """BUG (intentional): naive parser - breaks on European formats like '1.250,00'."""
    return float(re.sub(r"[^\d.]", "", s) or 0)


def parse_amount_robust(s: str) -> float:
    s = re.sub(r"[^\d,.-]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) == 2 else s.replace(",", "")
    return float(s or 0)


class InvoiceWorkflow:
    """AI-assisted accounts-payable automation: extract -> validate vendor -> approve/escalate -> post to ERP.

    version="v1" contains realistic bugs (amount parsing, no high-value escalation, no retries,
    not idempotent). version="v2" is the corrected implementation.
    """

    side_effect_tools = ("post_to_erp", "send_payment")
    APPROVAL_LIMIT = 10_000

    def __init__(self, version: str = "v1", **_: Any) -> None:
        self.version = version
        self.name = f"invoice_workflow_{version}"

    def tools(self) -> dict[str, Callable[..., Any]]:
        def extract_fields(document: str) -> dict[str, str]:  # stands in for an LLM/OCR extraction call
            fields = {}
            for key in ("invoice_id", "vendor", "amount", "currency"):
                m = re.search(rf"{key}\s*[:=]\s*(.+)", document, re.I)
                fields[key] = m.group(1).strip() if m else ""
            return fields

        return {
            "extract_fields": extract_fields,
            "validate_vendor": lambda vendor: VENDORS.get(vendor, False),
            "post_to_erp": lambda invoice_id, amount, currency: {"erp_id": f"ERP-{invoice_id}", "status": "posted"},
            "request_human_review": lambda invoice_id, reason: {"ticket": f"HR-{invoice_id}"},
            "send_payment": lambda invoice_id, amount: {"payment": "sent"},
        }

    def run(self, payload: dict[str, Any], toolbox: Any) -> dict[str, Any]:
        if self.version != "v1" and not isinstance(toolbox, (RetryToolBox, IdempotentToolBox)):
            toolbox = RetryToolBox(IdempotentToolBox(toolbox))  # v2: resilient + idempotent by design
        f = toolbox.call("extract_fields", document=payload["document"])
        parse = parse_amount_v1 if self.version == "v1" else parse_amount_robust
        amount = parse(f["amount"])
        currency = (f["currency"] or "USD").upper()
        if not toolbox.call("validate_vendor", vendor=f["vendor"]):
            toolbox.call("request_human_review", invoice_id=f["invoice_id"], reason="unknown vendor")
            return {"invoice_id": f["invoice_id"], "status": "rejected", "amount": amount, "currency": currency}
        if self.version != "v1" and amount > self.APPROVAL_LIMIT:
            toolbox.call("request_human_review", invoice_id=f["invoice_id"], reason="above approval limit")
            return {"invoice_id": f["invoice_id"], "status": "needs_review", "amount": amount, "currency": currency}
        toolbox.call("post_to_erp", invoice_id=f["invoice_id"], amount=amount, currency=currency)
        if self.version == "v1" and amount > self.APPROVAL_LIMIT:
            toolbox.call("send_payment", invoice_id=f["invoice_id"], amount=amount)  # BUG: excessive agency
        return {"invoice_id": f["invoice_id"], "status": "approved", "amount": amount, "currency": currency}


BUILTIN_WORKFLOWS = {"invoice": InvoiceWorkflow, "invoice_workflow": InvoiceWorkflow}
