"""Background run execution (thread pool). Swap for a Redis/RQ or Celery queue when scaling out."""
from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aitesteval.core.runner import Runner
from aitesteval.reporting import to_html, to_json, to_markdown

from .config import Settings
from .db import DB, loads
from .scenarios import Resolved
from .security import KeyVault, check_outbound_url

log = logging.getLogger("portal.worker")
REPORT_FILES = {"json": "report.json", "md": "report.md", "html": "report.html"}


class Worker:
    def __init__(self, db: DB, settings: Settings, vault: KeyVault) -> None:
        self.db, self.s, self.vault = db, settings, vault
        self.pool = ThreadPoolExecutor(max_workers=max(1, settings.workers), thread_name_prefix="run")

    # ------------------------------------------------------------ lifecycle
    def recover(self) -> None:
        """On startup: runs interrupted mid-flight fail; queued runs are re-queued."""
        self.db.x("UPDATE runs SET status='failed', error='interrupted: the server restarted during this run', "
                  "finished_at=? WHERE status='running'", (time.time(),))
        for r in self.db.q("SELECT id FROM runs WHERE status='queued' ORDER BY created_at"):
            self.submit(r["id"])

    def submit(self, run_id: str) -> None:
        self.pool.submit(self._execute, run_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def run_dir(self, run_id: str) -> Path:
        return self.s.runs_dir / run_id

    # ------------------------------------------------------------ execution
    def _log(self, run_id: str, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg.replace("\n", " ")[:500] + "\n"
        self.db.x("UPDATE runs SET log = log || ? WHERE id=?", (line, run_id))

    def _key(self, key_id: str | None, user_id: str) -> dict[str, Any] | None:
        if not key_id:
            return None
        row = self.db.one("SELECT * FROM provider_keys WHERE id=? AND user_id=?", (key_id, user_id))
        if not row:
            raise ValueError("the provider key used by this run was deleted")
        self.db.x("UPDATE provider_keys SET last_used_at=? WHERE id=?", (time.time(), key_id))
        return {**row, "secret": self.vault.decrypt(row["ciphertext"])}

    def _execute(self, run_id: str) -> None:
        run = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if not run or run["status"] != "queued":
            return
        self.db.x("UPDATE runs SET status='running', started_at=? WHERE id=?", (time.time(), run_id))
        secrets_used: list[str] = []
        try:
            res = Resolved(**loads(run["config"]))
            sc = json.loads(json.dumps(res.scenario))  # deep copy; secrets go only into this copy
            allow_private = self.s.allow_private_urls
            check = lambda u: check_outbound_url(u, allow_private)  # noqa: E731

            if res.llm_key_id:
                k = self._key(res.llm_key_id, run["user_id"])
                secrets_used.append(k["secret"])
                params = sc["model"]["params"]
                params["api_key"] = k["secret"]
                if sc["model"]["provider"] == "http":
                    params["base_url"] = check(k["base_url"])
                    params["follow_redirects"] = allow_private
            if sc.get("model", {}).get("endpoint"):
                sc["model"]["endpoint"] = check(sc["model"]["endpoint"])
                sc["model"]["follow_redirects"] = allow_private
                sc["model"]["url_check"] = check
                if res.endpoint_key_id:
                    k = self._key(res.endpoint_key_id, run["user_id"])
                    secrets_used.append(k["secret"])
                    sc["model"]["api_key"] = k["secret"]

            self._log(run_id, "started")
            runner = Runner(sc, res.base_dir)
            report = runner.run(progress=lambda m: self._log(run_id, m))
            out = self.run_dir(run_id)
            out.mkdir(parents=True, exist_ok=True)
            for fmt, writer in (("json", to_json), ("md", to_markdown), ("html", to_html)):
                (out / REPORT_FILES[fmt]).write_text(_scrub(writer(report), secrets_used), encoding="utf-8")
            summary = report.summary()
            summary["findings"] = [{"code": f.code, "title": f.title, "severity": f.severity.value,
                                    "evaluator": f.evaluator} for f in report.all_findings][:200]
            summary["fixes"] = [{"fix": f.fix, "applied": f.applied, "notes": f.notes} for f in report.fixes]
            summary["post_fix_codes"] = sorted({f.code for r in (report.post_fix_results or []) for f in r.findings})
            self.db.x("UPDATE runs SET status='done', summary=?, finished_at=? WHERE id=?",
                      (_scrub(json.dumps(summary, default=str), secrets_used), time.time(), run_id))
            self._log(run_id, f"finished: {summary['verdict']} (health {summary['health_score']})")
        except Exception as exc:
            msg = _scrub(f"{type(exc).__name__}: {exc}", secrets_used)
            log.warning("run %s failed: %s\n%s", run_id, msg, _scrub(traceback.format_exc(), secrets_used))
            self.db.x("UPDATE runs SET status='failed', error=?, finished_at=? WHERE id=?", (msg[:2000], time.time(), run_id))
            self._log(run_id, "failed: " + msg[:300])
        finally:
            secrets_used.clear()

    def delete_artifacts(self, run_id: str) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)


def _scrub(text: str, secrets_used: list[str]) -> str:
    for s in secrets_used:
        if s and len(s) >= 6:
            text = text.replace(s, "[redacted]")
    return text
