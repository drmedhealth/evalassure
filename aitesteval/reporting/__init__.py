"""Report writers: JSON (machine-readable, CI), Markdown (PRs/wikis), HTML (stakeholders)."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import ControlStatus, Report


def _default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and o != o:
        return None
    return str(o)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def to_json(report: Report) -> str:
    return json.dumps(_clean(report.to_dict()), indent=2, default=_default)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in list(v.items())[:6])
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v[:6])
    return str(v)


def _scalar_metrics(m: dict) -> list[tuple[str, Any]]:
    return [(k, v) for k, v in m.items() if isinstance(v, (int, float, str, bool)) or v is None][:18]


ICON = {"critical": "🛑", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}
CICON = {"pass": "✅", "fail": "❌", "warn": "⚠️", "manual_review": "📝", "not_applicable": "➖"}


def to_markdown(report: Report) -> str:
    s = report.summary()
    L = [f"# AI Test & Evaluation Report — {report.scenario}", "",
         f"*Use case:* {report.use_case}  ", f"*Task:* `{report.task}` · *Model:* `{report.metadata.get('model')}` · "
         f"*Run:* {report.timestamp}", "",
         "## Summary", "", "| Verdict | Health score | Compliance score | Critical | High | Medium | Low |",
         "|---|---|---|---|---|---|---|",
         f"| **{s['verdict']}** | {s['health_score']}/100 | {s['compliance_score'] if s['compliance_score'] is not None else '–'} | "
         f"{s['findings_by_severity']['critical']} | {s['findings_by_severity']['high']} | "
         f"{s['findings_by_severity']['medium']} | {s['findings_by_severity']['low']} |", ""]
    if "post_fix_health_score" in s:
        cs, pc = s['compliance_score'], s['post_fix_compliance_score']
        comp = f", compliance {cs} → **{pc}**" if cs is not None and pc is not None else ""
        L += [f"**After auto-fix:** health {s['health_score']} → **{s['post_fix_health_score']}**{comp}", ""]
    L += ["## Findings", ""]
    if report.all_findings:
        L += ["| Sev | Code | Finding | Evidence |", "|---|---|---|---|"]
        for f in report.all_findings:
            L.append(f"| {ICON[f.severity.value]} {f.severity.value} | `{f.code}` | {f.title} | {f.detail.replace('|', '/')} |")
    else:
        L.append("No issues found. ✅")
    L += ["", "## Root-cause diagnosis", ""]
    for d in report.diagnoses:
        L += [f"### {d.title} ({d.confidence} confidence)", "", d.root_cause, "", "*Evidence:*"]
        L += [f"- {e}" for e in d.evidence]
        L += ["", "*Next steps:*"] + [f"1. {x}" for x in d.next_steps] + [""]
    L += ["## Recommendations", ""]
    for r in report.recommendations:
        fix = f" · auto-fix: `{r.auto_fix}`" if r.auto_fix else ""
        L += [f"- **{r.priority} — {r.title}**{fix}", f"  - {r.rationale}"] + [f"  - [ ] {a}" for a in r.actions]
    if report.fixes:
        L += ["", "## Auto-fix results", "", "| Fix | Applied | What changed | Before | After |", "|---|---|---|---|---|"]
        for fx in report.fixes:
            L.append(f"| `{fx.fix}` | {'✅' if fx.applied else '—'} {fx.notes} | {fx.description} | {_fmt(fx.before)} | {_fmt(fx.after)} |")
    if report.compliance:
        L += ["", "## Compliance", "", f"> {report.metadata.get('compliance_disclaimer')}", "",
              "| Status | Control | Requirement | Evidence | Remediation |", "|---|---|---|---|---|"]
        post = {c.control_id: c for c in report.post_fix_compliance}
        for c in report.compliance:
            after = f" → {CICON[post[c.control_id].status.value]}" if c.control_id in post and post[c.control_id].status != c.status else ""
            L.append(f"| {CICON[c.status.value]}{after} | {c.framework} {c.control_id} | {c.title} ({c.reference}) | "
                     f"{c.evidence} | {c.remediation} |")
    L += ["", "## Metrics by evaluator", ""]
    for r in report.results:
        status = "skipped: " + r.skipped if r.skipped else ("ERROR: " + r.error if r.error else ("pass" if r.passed else "issues"))
        L += [f"### {r.evaluator} — {status}", ""]
        if r.metrics:
            L += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in _scalar_metrics(r.metrics)]
        L.append("")
    return "\n".join(L)


def to_html(report: Report) -> str:
    """Self-contained HTML (no external assets) rendered from the Markdown sections."""
    md = to_markdown(report)
    body = _md_to_html(md)
    s = report.summary()
    color = {"PASS": "#1a7f37", "PASS_WITH_WARNINGS": "#9a6700", "NEEDS_WORK": "#bc4c00", "BLOCK": "#cf222e"}[s["verdict"]]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report.scenario)} — AI evaluation report</title>
<style>
:root{{color-scheme:light dark;--fg:#1f2328;--bg:#fff;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa}}
@media (prefers-color-scheme:dark){{:root{{--fg:#e6edf3;--bg:#0d1117;--muted:#9198a1;--line:#3d444d;--card:#151b23}}}}
body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:1100px;margin:0 auto;padding:32px 20px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{margin-top:36px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
h3{{font-size:16px;margin:22px 0 6px}} table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}}
th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{background:var(--card)}}
code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:12.5px}} blockquote{{color:var(--muted);margin:0;padding-left:12px;border-left:3px solid var(--line)}}
.verdict{{display:inline-block;padding:4px 12px;border-radius:999px;color:#fff;background:{color};font-weight:600;margin:8px 0}}
ul{{padding-left:22px}} li{{margin:2px 0}}
</style></head><body><div class="verdict">{s['verdict']}</div>{body}</body></html>"""


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<em>\1</em>", t)
    return t


def _md_to_html(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            continue
        if ln.startswith("### "):
            out.append(f"<h3>{_inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("> "):
            out.append(f"<blockquote>{_inline(ln[2:])}</blockquote>")
        elif re.match(r"^\s*(- |\d+\. )", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*(- |\d+\. )", lines[i]):
                txt = re.sub(r"^\s*(- |\d+\. )", "", lines[i])
                indent = "margin-left:20px;" if lines[i].startswith("  ") else ""
                txt = txt.replace("[ ] ", "☐ ")
                items.append(f'<li style="{indent}">{_inline(txt)}</li>')
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def write_reports(report: Report, out_dir: str | Path, formats: tuple[str, ...] = ("json", "md", "html")) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", report.scenario.lower()).strip("_")
    paths = {}
    writers = {"json": to_json, "md": to_markdown, "html": to_html}
    for fmt in formats:
        p = out / f"{slug}.{fmt}"
        p.write_text(writers[fmt](report), encoding="utf-8")
        paths[fmt] = p
    return paths


__all__ = ["to_json", "to_markdown", "to_html", "write_reports", "ControlStatus"]
