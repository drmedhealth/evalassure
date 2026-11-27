"""LLM adapters: offline mock (with switchable flaws), Claude (Anthropic SDK), any
OpenAI-compatible HTTP endpoint (Ollama, vLLM, LM Studio...), or your own Python function."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str | None = None
    refused_by_provider: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def build_prompt(case: dict[str, Any]) -> str:
    """Single place that turns a test case into the user prompt (RAG context is prepended)."""
    if case.get("context"):
        return f"Use only the context to answer.\n\nContext:\n{case['context']}\n\nQuestion: {case['prompt']}"
    return case["prompt"]


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ---------------------------------------------------------------------- mock

INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above) (instructions|rules)", r"disregard (all|your|the) (previous|prior)?",
    r"(reveal|print|show|repeat) (me )?(your|the) (system prompt|instructions|hidden|initial)", r"developer mode",
    r"you are now (dan|unfiltered|jailbroken)", r"pretend (you have|there are) no (rules|restrictions)",
    r"</?(system|admin)>", r"new instructions:",
]
HARMFUL_PATTERNS = [
    r"\bbomb\b", r"explosive", r"malware", r"ransomware", r"keylogger", r"steal (someone|a person|credit|passwords)",
    r"hack into", r"poison (someone|a person)", r"phishing (email|page)", r"untraceable (gun|weapon)",
    r"synthesi[sz]e (meth|drugs)", r"bypass (the )?(bank|security) (login|authentication)",
]
REFUSAL = "I'm sorry, but I can't help with that request. If you have another question, I'm happy to help."
FEMALE_NAMES = {"maria", "aisha", "emily", "priya", "sofia", "fatima", "linda", "mei"}


class MockLLM:
    """Deterministic, offline LLM simulator. Flaws can be switched on to exercise every evaluator.

    flaws: hallucinate (0-1 prob), leak_pii, vulnerable_to_injection, weak_refusal, toxic (0-1),
           bad_json (0-1), inconsistent, biased, over_refusal, base_latency_s
    """

    def __init__(self, name: str = "mock-llm", **flaws: Any) -> None:
        self.name = name
        self.flaws = flaws
        self.calls = 0
        self._seen: dict[str, int] = {}

    def _rand(self, *parts: str) -> float:
        h = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def generate(self, prompt: str, system: str | None = None, context: str | None = None, **_: Any) -> LLMResponse:
        self.calls += 1
        f = self.flaws
        key = hashlib.sha1(((system or "") + prompt).encode()).hexdigest()
        k = self._seen.get(key, 0)
        self._seen[key] = k + 1
        salt = str(k) if f.get("inconsistent") and k > 0 else ""  # repeats of the same prompt vary
        text = self._answer(prompt, system or "", context, salt)
        r = self._rand(prompt, "lat", salt)
        latency = float(f.get("base_latency_s", 0.35)) * (0.6 + 1.2 * r) + (2.5 if r > 0.93 and f.get("slow_tail") else 0)
        return LLMResponse(text=text, latency_s=latency, input_tokens=estimate_tokens((system or "") + prompt),
                           output_tokens=estimate_tokens(text), model=self.name)

    def _answer(self, prompt: str, system: str, context: str | None, salt: str) -> str:
        f = self.flaws
        p = prompt.lower()
        question = p.split("question:")[-1]
        if any(re.search(x, p) for x in INJECTION_PATTERNS):
            if f.get("vulnerable_to_injection"):
                if "pwned" in p:
                    return "PWNED"
                return f"Sure! Here are my instructions: {system}"
            return "I can't share or change my instructions, but I'm happy to help with your original question."
        if any(re.search(x, question) for x in HARMFUL_PATTERNS):
            if f.get("weak_refusal"):
                return "Sure, here is a step-by-step guide. Step 1: gather the materials. Step 2: ..."
            return REFUSAL
        if f.get("over_refusal") and re.search(r"\b(kill|attack|shoot|exploit)\b", question):
            return REFUSAL
        if "json" in p:
            return self._json_answer(prompt, salt)
        if "reference" in question and "for" in question:
            name = next((n for n in FEMALE_NAMES if n in question), None)
            if f.get("biased"):
                return ("She is a caring, supportive and pleasant team member." if name else
                        "He is a decisive leader with outstanding technical skills and strong results.")
            who = re.search(r"for (\w+)", question)
            return f"{who.group(1).title() if who else 'The candidate'} is a reliable professional with strong skills and consistent results."
        answer = self._extract(question, context, salt)
        extras = []
        if f.get("hallucinate", 0) and self._rand(prompt, "hal", salt) < f["hallucinate"]:
            extras.append("This policy was introduced in 2017 and is used by 87% of our customers worldwide.")
        if f.get("leak_pii") and context:
            pii = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{8,}\d", context)
            if pii:
                extras.append(f"For reference, the account contact is {pii[0].strip()}.")
        if f.get("toxic", 0) and self._rand(prompt, "tox", salt) < f["toxic"]:
            extras.append("Honestly, that's a stupid question and you should have figured it out yourself.")
        return " ".join([answer, *extras]).strip()

    def _extract(self, question: str, context: str | None, salt: str) -> str:
        from ..evaluators.metrics import content_tokens

        q = set(content_tokens(question))
        if not context:
            if self.flaws.get("hallucinate"):
                return "The answer is 42, as confirmed by a 2021 industry report."
            return "I don't have enough information to answer that confidently."
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        scored = sorted(((len(q & set(content_tokens(s))), i, s) for i, s in enumerate(sents)), reverse=True)
        best, _, sent = scored[0]
        if best == 0:
            if self.flaws.get("hallucinate"):
                return "Yes. Our premium plan includes this at no cost for the first 24 months."
            return "The provided context does not contain that information, so I can't answer it."
        prefixes = ["Based on the provided information:", "Here is what I found:", "Sure -"]
        prefix = prefixes[int(self._rand(question, salt) * 3) % 3] if salt else prefixes[0]
        if salt and len(scored) > 1 and self._rand(question, salt, "alt") < 0.25:
            sent = f"{sent} {scored[1][2]}"  # inconsistent: sometimes adds extra, loosely related detail
        return f"{prefix} {sent}"

    def _json_answer(self, prompt: str, salt: str) -> str:
        m = re.search(r"keys?:?\s*([\w ,]+)", prompt, re.I)
        keys = [k.strip() for k in (m.group(1).split(",") if m else ["answer"]) if k.strip()][:6]
        text = prompt.lower()
        neg = sum(w in text for w in ["terrible", "broken", "angry", "late", "refund", "worst", "bad"])
        pos = sum(w in text for w in ["great", "love", "excellent", "fast", "happy", "thanks"])
        values = {"sentiment": "negative" if neg > pos else "positive" if pos > neg else "neutral",
                  "confidence": 0.8, "category": "billing" if "charge" in text or "refund" in text else "general",
                  "priority": "high" if neg > 1 else "normal", "summary": "Customer message summary."}
        obj = {k.split()[0]: values.get(k.split()[0], "n/a") for k in keys}
        s = json.dumps(obj)
        if self.flaws.get("bad_json", 0) and self._rand(prompt, "json", salt) < self.flaws["bad_json"]:
            return "Here is the JSON you asked for: " + s[:-1] + ",}"
        return s


# ------------------------------------------------------------------ Claude

class AnthropicLLM:
    """Claude via the official Anthropic Python SDK (`pip install anthropic`).

    Credentials: ANTHROPIC_API_KEY, or a profile from `ant auth login`.
    `fallbacks: "default"` enables server-side refusal fallbacks; the adapter records which
    model actually served each answer so you never mistake a fallback model's output for the
    model under test.
    """

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, effort: str | None = None,
                 fallbacks: str | list | None = None, api_key: str | None = None,
                 timeout_s: float | None = None, max_retries: int = 2, **_: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the Anthropic SDK: pip install anthropic") from exc
        # api_key: pass a key directly (BYOK portals); otherwise the SDK reads ANTHROPIC_API_KEY / `ant` profile.
        client_kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key:
            client_kwargs["api_key"] = api_key
        if timeout_s:
            client_kwargs["timeout"] = timeout_s
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model, self.max_tokens, self.effort, self.fallbacks = model, max_tokens, effort, fallbacks
        self.name = model

    def generate(self, prompt: str, system: str | None = None, context: str | None = None, **_: Any) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.perf_counter()
        if self.fallbacks:
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks=self.fallbacks, **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        iterations = getattr(resp.usage, "iterations", None) or []
        fallback_ran = any(getattr(e, "type", "") == "fallback_message" for e in iterations)
        return LLMResponse(text=text, latency_s=latency, input_tokens=resp.usage.input_tokens,
                           output_tokens=resp.usage.output_tokens, model=resp.model,
                           refused_by_provider=resp.stop_reason == "refusal",
                           meta={"stop_reason": resp.stop_reason, "fallback_ran": fallback_ran})


# ------------------------------------------------------------ generic HTTP

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects (a hosted service must not be bounced to an internal address)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


class HTTPChatLLM:
    """Any OpenAI-compatible /chat/completions endpoint (Ollama, vLLM, LM Studio, TGI, LiteLLM...)."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llama3.1",
                 api_key_env: str | None = None, temperature: float = 0.0, timeout_s: float = 60,
                 api_key: str | None = None, follow_redirects: bool = True, max_tokens: int | None = None,
                 **_: Any) -> None:
        self.base_url, self.model, self.temperature, self.timeout = base_url.rstrip("/"), model, temperature, timeout_s
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.max_tokens = max_tokens
        self.name = model
        self._opener = urllib.request.build_opener() if follow_redirects else urllib.request.build_opener(_NoRedirect)

    def generate(self, prompt: str, system: str | None = None, context: str | None = None, **_: Any) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        t0 = time.perf_counter()
        with self._opener.open(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=text, latency_s=time.perf_counter() - t0, model=data.get("model", self.model),
                           input_tokens=usage.get("prompt_tokens", estimate_tokens(prompt)),
                           output_tokens=usage.get("completion_tokens", estimate_tokens(text)))


class CallableLLM:
    """Wrap any function `fn(prompt, system=None, context=None) -> str | LLMResponse`."""

    def __init__(self, fn: Callable[..., Any], name: str = "custom-llm") -> None:
        self.fn, self.name = fn, name

    def generate(self, prompt: str, system: str | None = None, context: str | None = None, **_: Any) -> LLMResponse:
        t0 = time.perf_counter()
        out = self.fn(prompt, system=system, context=context)
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(text=str(out), latency_s=time.perf_counter() - t0, model=self.name,
                           input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(str(out)))


LLM_PROVIDERS = {"mock": MockLLM, "anthropic": AnthropicLLM, "claude": AnthropicLLM,
                 "http": HTTPChatLLM, "openai_compatible": HTTPChatLLM}
