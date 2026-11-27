"""evalassure portal: multi-user web app + REST API on top of the evalassure engine (BYOK)."""
from __future__ import annotations

import csv
import io
import json
import mimetypes
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from evalassure import __version__ as ENGINE_VERSION

from . import scenarios as S
from .config import Settings
from .db import DB, loads
from .security import (KeyVault, URLNotAllowed, check_outbound_url, hash_password, mask_key, new_token,
                       token_hash, verify_password)
from .worker import REPORT_FILES, Worker

HERE = Path(__file__).resolve().parent
# Windows can map .js to text/plain via the registry; with `nosniff`, browsers then refuse to run it.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
COOKIE = "ate_session"
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")
MESSAGES = {  # fixed flash messages (never echo user input through the URL)
    "welcome": "Account created. Start with a demo run, or add a provider key for live LLM tests.",
    "run_deleted": "Run deleted.", "ds_uploaded": "Dataset uploaded.", "ds_deleted": "Dataset deleted.",
    "key_saved": "Key saved and encrypted. It will not be shown again.", "key_deleted": "Key deleted.",
    "token_deleted": "API token revoked.", "logged_out": "You are signed out.",
}
ACTIVE = ("queued", "running")


class NeedLogin(Exception):
    pass


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    db = DB(s.db_path)
    vault = KeyVault(s.secret_key)
    worker = Worker(db, s, vault)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["ts"] = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else ""
    templates.env.filters["dur"] = lambda r: (f"{(r['finished_at'] - r['started_at']):.0f}s"
                                              if r.get("finished_at") and r.get("started_at") else "")
    failures: dict[str, deque] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker.recover()
        yield
        worker.shutdown()

    app = FastAPI(title="evalassure portal", version=ENGINE_VERSION, lifespan=lifespan,
                  description="Test, evaluate, compliance-check and auto-fix AI/ML models, LLM apps and automations. "
                              "Authenticate with `Authorization: Bearer <api token>` (create one under Settings).")
    app.state.db, app.state.worker, app.state.settings = db, worker, s
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            resp.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-src 'self'; "
                "frame-ancestors 'self'; form-action 'self'; base-uri 'none'; object-src 'none'")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return resp

    @app.exception_handler(NeedLogin)
    async def _need_login(request: Request, _: NeedLogin):
        return RedirectResponse("/login", status_code=303)

    # ------------------------------------------------------------------ auth helpers
    def session_of(request: Request) -> tuple[dict | None, dict | None]:
        tok = request.cookies.get(COOKIE)
        if not tok:
            return None, None
        sess = db.one("SELECT * FROM sessions WHERE token_hash=? AND expires_at>?", (token_hash(tok), time.time()))
        if not sess:
            return None, None
        user = db.one("SELECT id, email, name, created_at FROM users WHERE id=?", (sess["user_id"],))
        return user, sess

    def page_user(request: Request) -> dict:
        user, sess = session_of(request)
        if not user:
            raise NeedLogin()
        request.state.csrf = sess["csrf"]
        return user

    def api_user(request: Request) -> dict:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            row = db.one("SELECT * FROM api_tokens WHERE token_hash=?", (token_hash(auth[7:].strip()),))
            if row:
                db.x("UPDATE api_tokens SET last_used_at=? WHERE id=?", (time.time(), row["id"]))
                return db.one("SELECT id, email, name, created_at FROM users WHERE id=?", (row["user_id"],))
        user, _ = session_of(request)  # the browser session also works for GETs from the UI
        if user and request.method == "GET":
            return user
        raise HTTPException(401, "send `Authorization: Bearer <api token>` (create one under Settings)")

    async def form_with_csrf(request: Request) -> Any:
        form = await request.form()
        if form.get("csrf") != getattr(request.state, "csrf", None):
            raise HTTPException(403, "invalid or missing CSRF token; reload the page and try again")
        return form

    def render(request: Request, template: str, user: dict | None = None, status: int = 200, **ctx: Any) -> HTMLResponse:
        msg = MESSAGES.get(request.query_params.get("msg", ""))
        return templates.TemplateResponse(request, template, {"user": user, "csrf": getattr(request.state, "csrf", ""),
                                                          "msg": msg, "settings": s, **ctx}, status_code=status)

    def start_session(user_id: str, request: Request) -> Response:
        tok = new_token()
        db.insert("sessions", {"token_hash": token_hash(tok), "user_id": user_id, "csrf": new_token(),
                               "expires_at": time.time() + s.session_days * 86400})
        db.x("DELETE FROM sessions WHERE expires_at<?", (time.time(),))
        resp = RedirectResponse("/runs?msg=welcome" if request.url.path == "/signup" else "/runs", status_code=303)
        resp.set_cookie(COOKIE, tok, httponly=True, samesite="lax", secure=s.secure_cookies,
                        max_age=s.session_days * 86400)
        return resp

    def throttled(key: str) -> bool:
        q = failures[key]
        while q and q[0] < time.time() - 900:
            q.popleft()
        return len(q) >= 10

    # ------------------------------------------------------------------ data helpers
    def user_datasets(uid: str) -> dict[str, dict]:
        rows = db.q("SELECT * FROM datasets WHERE user_id=? ORDER BY created_at DESC", (uid,))
        for r in rows:
            r["columns"] = loads(r["columns"], [])
        return {r["id"]: r for r in rows}

    def user_keys(uid: str) -> dict[str, dict]:
        rows = db.q("SELECT id, provider, label, base_url, masked, created_at, last_used_at FROM provider_keys "
                    "WHERE user_id=? ORDER BY created_at DESC", (uid,))
        return {r["id"]: r for r in rows}

    def get_run(uid: str, run_id: str) -> dict:
        run = db.one("SELECT * FROM runs WHERE id=? AND user_id=?", (run_id, uid))
        if not run:
            raise HTTPException(404, "run not found")
        run["summary"] = loads(run["summary"], {})
        run["config"] = loads(run["config"], {})
        return run

    def public_run(run: dict) -> dict:
        sc = run["config"].get("scenario", {})
        return {"id": run["id"], "name": run["name"], "kind": run["kind"], "task": sc.get("task"),
                "status": run["status"], "created_at": run["created_at"], "started_at": run["started_at"],
                "finished_at": run["finished_at"], "summary": run["summary"], "error": run["error"],
                "estimated_llm_requests": run["config"].get("estimated_llm_requests", 0),
                "reports": {f: f"/api/v1/runs/{run['id']}/report?format={f}" for f in REPORT_FILES}
                if run["status"] == "done" else {}}

    def create_run(user: dict, req: S.RunRequest) -> dict:
        n_active = db.one("SELECT COUNT(*) n FROM runs WHERE user_id=? AND status IN ('queued','running')",
                          (user["id"],))["n"]
        if n_active >= s.max_active_runs_per_user:
            raise ValueError(f"you already have {n_active} runs in progress (limit {s.max_active_runs_per_user}); "
                             "wait for one to finish")
        if req.kind == "ml" and req.model_type == "endpoint" and req.endpoint_url:
            check_outbound_url(req.endpoint_url, s.allow_private_urls)
        resolved = S.build(req, user_datasets(user["id"]), user_keys(user["id"]), s.max_llm_cases,
                           s.max_llm_output_tokens)
        sc = resolved.scenario
        run = db.insert("runs", {"user_id": user["id"], "name": sc["name"], "kind": req.kind, "status": "queued",
                                 "config": resolved.model_dump_json(), "key_id": resolved.llm_key_id})
        worker.submit(run["id"])
        return run

    def form_to_request(form: Any) -> S.RunRequest:
        g, gl = form.get, form.getlist
        data: dict[str, Any] = {k: v for k, v in form.items() if k not in ("csrf",) and not k.startswith("mc_")
                                and k not in ("compliance", "guardrails", "sensitive", "drop", "autofix")}
        data = {k: v for k, v in data.items() if v != ""}
        data.update(compliance=gl("compliance"), guardrails=gl("guardrails"), sensitive=gl("sensitive"),
                    drop=gl("drop"), autofix=g("autofix") == "on")
        card = {k[3:]: v for k, v in form.items() if k.startswith("mc_")}
        for b in ("logging_enabled", "dpia_completed", "ai_disclosure", "automated_decision"):
            card[b] = card.get(b) == "on"
        data["model_card"] = card
        return S.RunRequest(**data)

    def validation_text(exc: ValidationError) -> str:
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())

    async def save_upload(user: dict, file: UploadFile, name: str) -> dict:
        fname = Path(file.filename or "upload").name
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext not in ("csv", "jsonl"):
            raise ValueError("upload a .csv (tabular data) or .jsonl (LLM test cases) file")
        limit = s.max_upload_mb * 1024 * 1024
        buf = bytearray()
        while chunk := await file.read(1024 * 1024):
            buf.extend(chunk)
            if len(buf) > limit:
                raise ValueError(f"file is larger than {s.max_upload_mb} MB")
        text = bytes(buf).decode("utf-8-sig", errors="strict")
        rows, cols = (_inspect_csv(text) if ext == "csv" else _inspect_jsonl(text, s.max_llm_cases))
        ds_id = db.new_id()
        folder = s.uploads_dir / user["id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ds_id}.{ext}"
        path.write_bytes(bytes(buf))
        return db.insert("datasets", {"id": ds_id, "user_id": user["id"], "name": (name or fname)[:120], "kind": ext,
                                      "filename": fname[:200], "path": str(path), "rows": rows,
                                      "columns": json.dumps(cols), "size_bytes": len(buf)})

    # ================================================================== pages
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        user, _ = session_of(request)
        if user:
            return RedirectResponse("/runs", status_code=303)
        return render(request, "landing.html", demos=S.demo_catalogue())

    @app.get("/signup", response_class=HTMLResponse)
    def signup_page(request: Request):
        if not s.allow_signup:
            return render(request, "auth.html", mode="signup", error="Sign-up is disabled on this portal.", status=403)
        return render(request, "auth.html", mode="signup")

    @app.post("/signup")
    def signup(request: Request, email: str = Form(...), password: str = Form(...), name: str = Form("")):
        if not s.allow_signup:
            raise HTTPException(403, "sign-up is disabled")
        email = email.strip().lower()
        err = ("Enter a valid email address." if not EMAIL_RE.match(email) else
               "Use at least 10 characters for the password." if len(password) < 10 else
               "An account with this email already exists." if db.one("SELECT 1 FROM users WHERE email=?", (email,))
               else None)
        if err:
            return render(request, "auth.html", mode="signup", error=err, email=email, name=name, status=400)
        user = db.insert("users", {"email": email, "name": name.strip()[:80] or email.split("@")[0],
                                   "password_hash": hash_password(password)})
        return start_session(user["id"], request)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return render(request, "auth.html", mode="login")

    @app.post("/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...)):
        email = email.strip().lower()
        key = f"{request.client.host if request.client else '?'}|{email}"
        if throttled(key):
            return render(request, "auth.html", mode="login", email=email, status=429,
                          error="Too many failed attempts. Try again in 15 minutes.")
        row = db.one("SELECT * FROM users WHERE email=?", (email,))
        ok = verify_password(password, row["password_hash"]) if row else verify_password(password, _DUMMY_HASH)
        if not (row and ok):
            failures[key].append(time.time())
            return render(request, "auth.html", mode="login", email=email, error="Wrong email or password.", status=401)
        failures.pop(key, None)
        return start_session(row["id"], request)

    @app.post("/logout")
    async def logout(request: Request, user: dict = Depends(page_user)):
        await form_with_csrf(request)
        tok = request.cookies.get(COOKIE, "")
        db.x("DELETE FROM sessions WHERE token_hash=?", (token_hash(tok),))
        resp = RedirectResponse("/login?msg=logged_out", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    # ------------------------------------------------------------------ runs
    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(request: Request, user: dict = Depends(page_user)):
        runs = db.q("SELECT * FROM runs WHERE user_id=? ORDER BY created_at DESC LIMIT 200", (user["id"],))
        for r in runs:
            r["summary"] = loads(r["summary"], {})
            r["task"] = loads(r["config"], {}).get("scenario", {}).get("task", "")
        done = [r for r in runs if r["status"] == "done"]
        stats = {"total": len(runs), "active": sum(r["status"] in ACTIVE for r in runs),
                 "blocked": sum(r["summary"].get("verdict") == "BLOCK" for r in done),
                 "avg_health": round(sum(r["summary"].get("post_fix_health_score", r["summary"].get("health_score", 0))
                                         for r in done) / len(done)) if done else None}
        return render(request, "runs.html", user, runs=runs, stats=stats,
                      refresh=any(r["status"] in ACTIVE for r in runs))

    @app.get("/runs/new", response_class=HTMLResponse)
    def new_run_page(request: Request, kind: str = "demo", user: dict = Depends(page_user)):
        kind = kind if kind in ("demo", "ml", "llm") else "demo"
        return render(request, "new_run.html", user, kind=kind, demos=S.demo_catalogue(),
                      datasets=list(user_datasets(user["id"]).values()), keys=list(user_keys(user["id"]).values()),
                      S=S, form={})

    @app.post("/runs")
    async def create_run_form(request: Request, user: dict = Depends(page_user)):
        form = await form_with_csrf(request)
        try:
            run = create_run(user, form_to_request(form))
        except ValidationError as exc:
            err = validation_text(exc)
        except (ValueError, URLNotAllowed) as exc:
            err = str(exc)
        else:
            return RedirectResponse(f"/runs/{run['id']}", status_code=303)
        kind = form.get("kind", "demo")
        return render(request, "new_run.html", user, kind=kind if kind in ("demo", "ml", "llm") else "demo",
                      demos=S.demo_catalogue(), datasets=list(user_datasets(user["id"]).values()),
                      keys=list(user_keys(user["id"]).values()), S=S, error=err, form=dict(form), status=400)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str, user: dict = Depends(page_user)):
        run = get_run(user["id"], run_id)
        others = db.q("SELECT id, name, created_at FROM runs WHERE user_id=? AND status='done' AND id<>? "
                      "ORDER BY created_at DESC LIMIT 50", (user["id"], run_id))
        return render(request, "run.html", user, run=run, others=others, refresh=run["status"] in ACTIVE,
                      scenario=run["config"].get("scenario", {}))

    @app.get("/runs/{run_id}/report.{fmt}")
    def run_report(run_id: str, fmt: str, request: Request, user: dict = Depends(page_user)):
        return _report_response(get_run(user["id"], run_id), fmt, download=request.query_params.get("download") == "1")

    @app.post("/runs/{run_id}/delete")
    async def delete_run(request: Request, run_id: str, user: dict = Depends(page_user)):
        await form_with_csrf(request)
        run = get_run(user["id"], run_id)
        if run["status"] == "running":
            raise HTTPException(409, "this run is executing; delete it after it finishes")
        db.x("DELETE FROM runs WHERE id=? AND user_id=?", (run_id, user["id"]))
        worker.delete_artifacts(run_id)
        return RedirectResponse("/runs?msg=run_deleted", status_code=303)

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(request: Request, a: str, b: str, user: dict = Depends(page_user)):
        ra, rb = get_run(user["id"], a), get_run(user["id"], b)
        return render(request, "compare.html", user, a=ra, b=rb, diff=_diff(ra, rb))

    def _report_response(run: dict, fmt: str, download: bool = False) -> Response:
        if fmt not in REPORT_FILES:
            raise HTTPException(404, "format must be json, md or html")
        if run["status"] != "done":
            raise HTTPException(409, f"run is {run['status']}; reports exist only for finished runs")
        path = worker.run_dir(run["id"]) / REPORT_FILES[fmt]
        if not path.exists():
            raise HTTPException(404, "report file missing")
        media = {"json": "application/json", "md": "text/markdown; charset=utf-8", "html": "text/html; charset=utf-8"}[fmt]
        slug = re.sub(r"[^a-z0-9]+", "-", run["name"].lower()).strip("-") or "report"
        headers = {"Content-Disposition": f'{"attachment" if download or fmt != "html" else "inline"}; '
                                          f'filename="{slug}.{fmt}"'}
        if fmt == "html":  # report content includes model outputs: render it sandboxed, no scripts
            headers["Content-Security-Policy"] = ("sandbox; default-src 'none'; style-src 'unsafe-inline'; "
                                                  "img-src data:; frame-ancestors 'self'")
        return FileResponse(path, media_type=media, headers=headers)

    # ------------------------------------------------------------------ datasets
    @app.get("/datasets", response_class=HTMLResponse)
    def datasets_page(request: Request, user: dict = Depends(page_user)):
        return render(request, "datasets.html", user, datasets=list(user_datasets(user["id"]).values()))

    @app.post("/datasets")
    async def upload_dataset(request: Request, user: dict = Depends(page_user)):
        form = await form_with_csrf(request)
        file = form.get("file")
        try:
            if not hasattr(file, "read"):
                raise ValueError("choose a file to upload")
            await save_upload(user, file, str(form.get("name", "")))
        except (ValueError, UnicodeDecodeError) as exc:
            err = "file must be UTF-8 text" if isinstance(exc, UnicodeDecodeError) else str(exc)
            return render(request, "datasets.html", user, datasets=list(user_datasets(user["id"]).values()),
                          error=err, status=400)
        return RedirectResponse("/datasets?msg=ds_uploaded", status_code=303)

    @app.post("/datasets/{ds_id}/delete")
    async def delete_dataset(request: Request, ds_id: str, user: dict = Depends(page_user)):
        await form_with_csrf(request)
        ds = db.one("SELECT * FROM datasets WHERE id=? AND user_id=?", (ds_id, user["id"]))
        if not ds:
            raise HTTPException(404, "dataset not found")
        db.x("DELETE FROM datasets WHERE id=? AND user_id=?", (ds_id, user["id"]))
        Path(ds["path"]).unlink(missing_ok=True)
        return RedirectResponse("/datasets?msg=ds_deleted", status_code=303)

    # ------------------------------------------------------------------ BYOK keys
    @app.get("/keys", response_class=HTMLResponse)
    def keys_page(request: Request, user: dict = Depends(page_user)):
        return render(request, "keys.html", user, keys=list(user_keys(user["id"]).values()))

    @app.post("/keys")
    async def add_key(request: Request, user: dict = Depends(page_user)):
        form = await form_with_csrf(request)
        provider, label = str(form.get("provider", "")), str(form.get("label", "")).strip()[:80]
        secret, base_url = str(form.get("secret", "")).strip(), str(form.get("base_url", "")).strip()
        err = None
        if provider not in ("anthropic", "openai_compatible", "model_endpoint"):
            err = "choose a provider"
        elif not (8 <= len(secret) <= 512) or any(c.isspace() for c in secret):
            err = "paste the full key (no spaces)"
        elif provider == "anthropic" and not secret.startswith("sk-ant-"):
            err = "Anthropic API keys start with sk-ant-"
        elif provider == "openai_compatible":
            try:
                base_url = check_outbound_url(base_url, s.allow_private_urls).rstrip("/")
            except URLNotAllowed as exc:
                err = f"base URL: {exc}"
        if err:
            return render(request, "keys.html", user, keys=list(user_keys(user["id"]).values()), error=err,
                          form={"provider": provider, "label": label, "base_url": base_url}, status=400)
        db.insert("provider_keys", {"user_id": user["id"], "provider": provider, "label": label or provider,
                                    "base_url": base_url if provider == "openai_compatible" else None,
                                    "ciphertext": vault.encrypt(secret), "masked": mask_key(secret)})
        return RedirectResponse("/keys?msg=key_saved", status_code=303)

    @app.post("/keys/{key_id}/delete")
    async def delete_key(request: Request, key_id: str, user: dict = Depends(page_user)):
        await form_with_csrf(request)
        db.x("DELETE FROM provider_keys WHERE id=? AND user_id=?", (key_id, user["id"]))
        return RedirectResponse("/keys?msg=key_deleted", status_code=303)

    # ------------------------------------------------------------------ settings: API tokens, account
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, user: dict = Depends(page_user)):
        tokens = db.q("SELECT id, label, prefix, created_at, last_used_at FROM api_tokens WHERE user_id=? "
                      "ORDER BY created_at DESC", (user["id"],))
        return render(request, "settings.html", user, tokens=tokens)

    @app.post("/settings/tokens")
    async def create_token(request: Request, user: dict = Depends(page_user)):
        form = await form_with_csrf(request)
        tok = new_token("ate_")
        db.insert("api_tokens", {"user_id": user["id"], "label": str(form.get("label", "")).strip()[:80] or "token",
                                 "token_hash": token_hash(tok), "prefix": tok[:10]})
        tokens = db.q("SELECT id, label, prefix, created_at, last_used_at FROM api_tokens WHERE user_id=? "
                      "ORDER BY created_at DESC", (user["id"],))
        return render(request, "settings.html", user, tokens=tokens, new_token=tok)

    @app.post("/settings/tokens/{tid}/delete")
    async def delete_token(request: Request, tid: str, user: dict = Depends(page_user)):
        await form_with_csrf(request)
        db.x("DELETE FROM api_tokens WHERE id=? AND user_id=?", (tid, user["id"]))
        return RedirectResponse("/settings?msg=token_deleted", status_code=303)

    @app.post("/settings/delete-account")
    async def delete_account(request: Request, user: dict = Depends(page_user)):
        form = await form_with_csrf(request)
        if form.get("confirm", "").strip().lower() != user["email"]:
            tokens = db.q("SELECT id, label, prefix, created_at, last_used_at FROM api_tokens WHERE user_id=?",
                          (user["id"],))
            return render(request, "settings.html", user, tokens=tokens, status=400,
                          error="Type your email address exactly to confirm account deletion.")
        if db.one("SELECT 1 FROM runs WHERE user_id=? AND status='running'", (user["id"],)):
            raise HTTPException(409, "a run is executing; delete the account after it finishes")
        for r in db.q("SELECT id FROM runs WHERE user_id=?", (user["id"],)):
            worker.delete_artifacts(r["id"])
        for d in db.q("SELECT path FROM datasets WHERE user_id=?", (user["id"],)):
            Path(d["path"]).unlink(missing_ok=True)
        db.x("DELETE FROM users WHERE id=?", (user["id"],))  # cascades: sessions, tokens, keys, datasets, runs
        resp = RedirectResponse("/?msg=logged_out", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    # ================================================================== REST API
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "engine": ENGINE_VERSION}

    @app.get("/api/v1/catalog")
    def catalog(user: dict = Depends(api_user)):
        return {"demos": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in S.demo_catalogue().items()},
                "llm_suites": {k: v[0] for k, v in S.BUILTIN_CASES.items()}, "frameworks": S.FRAMEWORKS,
                "guardrails": S.GUARDRAILS, "ml_models": S.ML_MODELS, "claude_models": S.CLAUDE_MODELS,
                "limits": {"max_llm_cases": s.max_llm_cases, "max_llm_output_tokens": s.max_llm_output_tokens,
                           "max_active_runs": s.max_active_runs_per_user, "max_upload_mb": s.max_upload_mb}}

    @app.get("/api/v1/runs")
    def api_runs(user: dict = Depends(api_user), limit: int = 50):
        rows = db.q("SELECT * FROM runs WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user["id"], min(limit, 200)))
        return [public_run({**r, "summary": loads(r["summary"], {}), "config": loads(r["config"], {})}) for r in rows]

    @app.post("/api/v1/runs", status_code=202)
    def api_create_run(req: S.RunRequest, user: dict = Depends(api_user)):
        try:
            run = create_run(user, req)
        except (ValueError, URLNotAllowed) as exc:
            raise HTTPException(400, str(exc)) from exc
        return public_run(get_run(user["id"], run["id"]))

    @app.get("/api/v1/runs/{run_id}")
    def api_run(run_id: str, user: dict = Depends(api_user)):
        return public_run(get_run(user["id"], run_id))

    @app.get("/api/v1/runs/{run_id}/report")
    def api_report(run_id: str, format: str = "json", user: dict = Depends(api_user)):
        return _report_response(get_run(user["id"], run_id), format)

    @app.delete("/api/v1/runs/{run_id}", status_code=204)
    def api_delete_run(run_id: str, user: dict = Depends(api_user)):
        run = get_run(user["id"], run_id)
        if run["status"] == "running":
            raise HTTPException(409, "run is executing")
        db.x("DELETE FROM runs WHERE id=? AND user_id=?", (run_id, user["id"]))
        worker.delete_artifacts(run_id)
        return Response(status_code=204)

    @app.get("/api/v1/datasets")
    def api_datasets(user: dict = Depends(api_user)):
        return [{k: v for k, v in d.items() if k not in ("path", "user_id")} for d in user_datasets(user["id"]).values()]

    @app.post("/api/v1/datasets", status_code=201)
    async def api_upload(file: UploadFile = File(...), name: str = Form(""), user: dict = Depends(api_user)):
        try:
            d = await save_upload(user, file, name)
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "file must be UTF-8 text") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"id": d["id"], "name": d["name"], "kind": d["kind"], "rows": d["rows"], "columns": loads(d["columns"], [])}

    @app.get("/api/v1/keys")
    def api_keys(user: dict = Depends(api_user)):
        return list(user_keys(user["id"]).values())  # masked only; keys are added in the web UI

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api/") or request.headers.get("accept", "").startswith("application/json"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        user, sess = session_of(request)
        request.state.csrf = sess["csrf"] if sess else ""
        return render(request, "error.html", user, status=exc.status_code, code=exc.status_code, detail=exc.detail)

    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots():
        return "User-agent: *\nDisallow: /\n"

    return app


# ---------------------------------------------------------------------- helpers

_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


def _inspect_csv(text: str) -> tuple[int, list[str]]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("the CSV is empty") from exc
    header = [h.strip() for h in header]
    if len(header) < 2 or any(not h for h in header):
        raise ValueError("the CSV needs a header row with at least 2 named columns")
    if len(set(header)) != len(header):
        raise ValueError("column names in the header must be unique")
    if len(header) > 500:
        raise ValueError("at most 500 columns are supported")
    rows = sum(1 for r in reader if any(c.strip() for c in r))
    if rows < 30:
        raise ValueError(f"the CSV has {rows} data rows; at least 30 are needed for a meaningful evaluation")
    return rows, header


def _inspect_jsonl(text: str, max_cases: int) -> tuple[int, list[str]]:
    ids, keys = set(), set()
    n = 0
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {i} is not valid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("prompt"), str) or "id" not in obj:
            raise ValueError(f"line {i}: each case needs an `id` and a `prompt` string")
        if obj["id"] in ids:
            raise ValueError(f"line {i}: duplicate id {obj['id']!r}")
        ids.add(obj["id"])
        keys.update(obj)
        n += 1
    if n == 0:
        raise ValueError("no test cases found")
    if n > max_cases:
        raise ValueError(f"{n} cases exceeds the limit of {max_cases}")
    return n, sorted(keys)


def _final_codes(run: dict) -> set[str]:
    s = run["summary"]
    if "post_fix_health_score" in s:
        return set(s.get("post_fix_codes", []))
    return {f["code"] for f in s.get("findings", [])}


def _diff(a: dict, b: dict) -> dict[str, Any]:
    ca, cb = _final_codes(a), _final_codes(b)
    title = {f["code"]: f["title"] for r in (a, b) for f in r["summary"].get("findings", [])}

    def final(r: dict, k: str) -> Any:
        s = r["summary"]
        return s.get(f"post_fix_{k}", s.get(k))

    rows = []
    for label, key in (("Health score", "health_score"), ("Compliance score", "compliance_score")):
        va, vb = final(a, key), final(b, key)
        delta = (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        rows.append({"label": label, "a": va, "b": vb, "delta": delta})
    return {"metrics": rows, "resolved": sorted((c, title.get(c, c)) for c in ca - cb),
            "new": sorted((c, title.get(c, c)) for c in cb - ca), "both": sorted((c, title.get(c, c)) for c in ca & cb)}


app = None  # created by portal.__main__ or `uvicorn portal.app:get_app --factory`


def get_app() -> FastAPI:
    return create_app()
