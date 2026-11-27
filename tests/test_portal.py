"""End-to-end tests for the web portal (offline: fake LLM server and model endpoint run locally)."""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from portal.app import create_app  # noqa: E402
from portal.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SECRET = "sk-test-SUPERSECRET-0123456789"
SEEN_AUTH: list[str] = []


class FakeServer(BaseHTTPRequestHandler):
    """/v1/chat/completions -> canned answer; /predict -> rule-based probabilities."""

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN_AUTH.append(self.headers.get("Authorization", ""))
        if self.path.endswith("/chat/completions"):
            out = {"model": body["model"], "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                   "choices": [{"message": {"content": "I can only answer from the provided context."}}]}
        elif self.path == "/predict":
            probs = [[1 - (0.8 if (r[0] or 0) > 40 else 0.2), 0.8 if (r[0] or 0) > 40 else 0.2] for r in body["instances"]]
            out = {"probabilities": probs}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_client(tmp_path, allow_private=False):
    s = Settings(data_dir=tmp_path, secret_key="test-secret-key", allow_private_urls=allow_private, workers=2,
                 max_active_runs_per_user=10)
    return TestClient(create_app(s)), s


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def signup(c, email="ana@example.com"):
    r = c.post("/signup", data={"email": email, "password": "correct-horse-1", "name": "Ana"})
    assert r.status_code == 200 and "Runs" in r.text
    return csrf_of(c.get("/runs").text)


def wait_done(c, run_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/v1/runs/{run_id}").json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def run_id_from(resp) -> str:
    return re.search(r"/runs/([0-9a-f]{32})", str(resp.url)).group(1)


def test_auth_csrf_and_demo_run(tmp_path):
    c, _ = make_client(tmp_path)
    with c:
        assert c.get("/runs", follow_redirects=False).status_code == 303  # login required
        assert "javascript" in c.get("/static/app.js").headers["content-type"]  # nosniff needs a JS type
        tok = signup(c)
        dup = c.post("/signup", data={"email": "ana@example.com", "password": "correct-horse-1", "name": "Ana"})
        assert dup.status_code == 400 and "already exists" in dup.text
        assert c.post("/signup", data={"email": "x@example.com", "password": "short"}).status_code == 400
        assert c.post("/runs", data={"kind": "demo", "demo": "14_automation_invoice_v2_regression"}).status_code == 403
        r = c.post("/runs", data={"csrf": tok, "kind": "demo", "demo": "14_automation_invoice_v2_regression",
                                  "autofix": "on"})
        rid = run_id_from(r)
        done = wait_done(c, rid)
        assert done["status"] == "done" and done["summary"]["verdict"] == "PASS"
        rep = c.get(f"/runs/{rid}/report.html")
        assert rep.status_code == 200 and "sandbox" in rep.headers["content-security-policy"]
        assert "PASS" in c.get(f"/runs/{rid}").text
        # wrong password is rejected; sign-out works
        c.post("/logout", data={"csrf": tok})
        assert c.post("/login", data={"email": "ana@example.com", "password": "nope"}).status_code == 401


def test_ml_csv_run_and_isolation(tmp_path):
    c, s = make_client(tmp_path)
    with c:
        tok = signup(c)
        csv_path = ROOT / "examples" / "data" / "employee_attrition.csv"
        r = c.post("/datasets", data={"csrf": tok, "name": "attrition"},
                   files={"file": ("attrition.csv", csv_path.read_bytes(), "text/csv")})
        assert r.status_code == 200 and "attrition" in r.text
        ds = c.get("/api/v1/datasets").json()[0]
        assert "left" in ds["columns"] or len(ds["columns"]) > 2
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        target = "left" if "left" in header else header[-1]
        sens = [h for h in header if h in ("gender", "age_group")]
        r = c.post("/runs", data={"csrf": tok, "kind": "ml", "name": "attrition audit", "task": "classification",
                                  "dataset_id": ds["id"], "target": target, "sensitive": sens, "model_type": "builtin",
                                  "builtin": "logreg", "compliance": ["eu_ai_act", "gdpr"], "autofix": "on"})
        rid = run_id_from(r)
        done = wait_done(c, rid)
        assert done["status"] == "done", done["error"]
        assert done["summary"]["health_score"] is not None
        for kind in ("demo", "ml", "llm"):
            page = c.get(f"/runs/new?kind={kind}")
            assert page.status_code == 200 and "Start run" in page.text or "Add a provider key" in page.text
        r2 = c.post("/runs", data={"csrf": tok, "kind": "demo", "demo": "05_house_price_regression"})
        wait_done(c, run_id_from(r2))
        cmp = c.get(f"/compare?a={rid}&b={run_id_from(r2)}")
        assert cmp.status_code == 200 and "Health score" in cmp.text
        assert "attrition audit" in c.get("/runs").text

        # another user cannot see it
        c.cookies.clear()
        signup(c, "bo@example.com")
        assert c.get(f"/api/v1/runs/{rid}").status_code == 404
        assert c.get(f"/runs/{rid}/report.json").status_code == 404
        assert c.get("/api/v1/datasets").json() == []


def test_llm_byok_key_never_leaks(tmp_path, fake_server):
    SEEN_AUTH.clear()
    c, s = make_client(tmp_path, allow_private=True)
    with c:
        tok = signup(c)
        r = c.post("/keys", data={"csrf": tok, "provider": "openai_compatible", "label": "local",
                                  "base_url": fake_server + "/v1", "secret": SECRET})
        assert r.status_code == 200 and SECRET not in r.text
        key = c.get("/api/v1/keys").json()[0]
        r = c.post("/runs", data={"csrf": tok, "kind": "llm", "name": "byok", "cases": "redteam", "key_id": key["id"],
                                  "provider": "openai_compatible", "llm_model": "fake-1", "max_tokens": "64",
                                  "consistency_cases": "1", "compliance": ["owasp_llm"]})
        rid = run_id_from(r)
        done = wait_done(c, rid)
        assert done["status"] == "done", done["error"]
        assert SEEN_AUTH and all(a == f"Bearer {SECRET}" for a in SEEN_AUTH)  # key reached the provider
        # ...but nowhere else
        blobs = [p.read_bytes() for p in Path(tmp_path).rglob("*") if p.is_file()]
        assert not any(SECRET.encode() in b for b in blobs)
        for fmt in ("json", "md", "html"):
            assert SECRET not in c.get(f"/runs/{rid}/report.{fmt}").text


def test_remote_endpoint_model_and_ssrf_block(tmp_path, fake_server):
    rows = ["age,tenure,left"] + [f"{20 + i % 40},{i % 10},{int((20 + i % 40) > 40)}" for i in range(300)]
    data = ("\n".join(rows) + "\n").encode()
    # hosted mode: private addresses are refused
    c, _ = make_client(tmp_path / "hosted")
    with c:
        tok = signup(c)
        c.post("/datasets", data={"csrf": tok}, files={"file": ("d.csv", data, "text/csv")})
        ds = c.get("/api/v1/datasets").json()[0]
        r = c.post("/runs", data={"csrf": tok, "kind": "ml", "name": "remote", "dataset_id": ds["id"], "target": "left",
                                  "model_type": "endpoint", "endpoint_url": fake_server + "/predict"})
        assert r.status_code == 400 and "private or internal" in r.text
    # self-hosted mode: allowed
    c, _ = make_client(tmp_path / "selfhost", allow_private=True)
    with c:
        tok = signup(c)
        c.post("/datasets", data={"csrf": tok}, files={"file": ("d.csv", data, "text/csv")})
        ds = c.get("/api/v1/datasets").json()[0]
        r = c.post("/runs", data={"csrf": tok, "kind": "ml", "name": "remote", "dataset_id": ds["id"], "target": "left",
                                  "model_type": "endpoint", "endpoint_url": fake_server + "/predict", "autofix": "on"})
        done = wait_done(c, run_id_from(r))
        assert done["status"] == "done", done["error"]
        assert done["summary"]["health_score"] is not None


def test_api_token_flow(tmp_path):
    c, _ = make_client(tmp_path)
    with c:
        tok = signup(c)
        html = c.post("/settings/tokens", data={"csrf": tok, "label": "ci"}).text
        api_tok = re.search(r"(ate_[A-Za-z0-9_\-]+)", html).group(1)
        c.cookies.clear()
        assert c.post("/api/v1/runs", json={"kind": "demo", "demo": "05_house_price_regression"}).status_code == 401
        h = {"Authorization": f"Bearer {api_tok}"}
        r = c.post("/api/v1/runs", json={"kind": "demo", "demo": "05_house_price_regression"}, headers=h)
        assert r.status_code == 202
        rid = r.json()["id"]
        t0 = time.time()
        while c.get(f"/api/v1/runs/{rid}", headers=h).json()["status"] not in ("done", "failed"):
            assert time.time() - t0 < 240
            time.sleep(0.3)
        rep = c.get(f"/api/v1/runs/{rid}/report?format=json", headers=h)
        assert rep.status_code == 200 and "health_score" in rep.text
        bad = c.post("/api/v1/runs", json={"kind": "ml", "dataset_id": "nope"}, headers=h)
        assert bad.status_code == 400
