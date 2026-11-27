# Web portal (multi-user, bring-your-own-key)

The portal puts aitesteval behind a website and REST API so a whole team, or your customers, can test AI systems without installing anything. Each user has a private workspace with their own datasets, runs, reports, API tokens and provider keys.

## Run it

```bash
pip install -e ".[portal]"
python -m portal --port 8000          # then open http://127.0.0.1:8000
```

On Windows, if port 8000 is reserved (`WinError 10013`), pick another port, such as `--port 8765`.

With Docker (set `PORTAL_SECRET_KEY` in `.env` first; copy `.env.example`):

```bash
docker compose up -d --build
```

All state lives in `PORTAL_DATA_DIR` (default `./portal_data`, `/data` in the container): the SQLite database, uploaded files and run reports. Back up that folder.

`python -m portal` binds to `$PORT` when that variable is set (as on Render and similar platforms), otherwise to 8000; an explicit `--port` always wins.

### Deploying on Render

The repository includes a Render Blueprint, [`render.yaml`](../render.yaml), so the **Deploy to Render** button in the README creates the portal in one click. The Blueprint defines one Docker web service on the **free** plan with:

- a generated `PORTAL_SECRET_KEY`, `PORTAL_SECURE_COOKIES=true` (Render serves HTTPS), `PORTAL_ALLOW_PRIVATE_URLS=false` and open sign-up;
- `PORTAL_DATA_DIR=/tmp/portal_data`, because the free plan has no disk;
- conservative limits for a 512 MB instance: one worker, one active run per user, 10 MB uploads, 50 LLM cases and 1024 output tokens per run;
- a health check on `/api/v1/health`.

**Data on the free plan is ephemeral.** The filesystem is wiped on every restart and redeploy, and free services sleep after a period of inactivity, so accounts, datasets, runs and saved provider keys all disappear. That is fine for a public demo, but tell your users.

For real use, switch the service to a paid plan, attach a [persistent disk](https://render.com/docs/disks) (for example mounted at `/data`), and set `PORTAL_DATA_DIR` to the disk's mount path. Keep the same `PORTAL_SECRET_KEY`, or saved keys become unreadable, and raise the limits to suit the instance size. The production checklist below still applies.

## What users can do

| Page | Purpose |
|---|---|
| **New run → Demo scenario** | Run any of the bundled offline scenarios (no key needed). Good for onboarding. |
| **New run → ML model (CSV)** | Upload a CSV, then choose the target, sensitive attributes, columns to ignore, optional production data (for drift), and either a built-in baseline model or **your deployed model's HTTP endpoint**. |
| **New run → LLM app (your key)** | Test Claude (Anthropic key) or any OpenAI-compatible endpoint on a built-in suite (support bot, red team) or your own JSONL cases, with an optional system prompt, guardrails and compliance frameworks. Shows the number of requests before you start. |
| **Runs** | Status, verdict, health before and after auto-fix, compliance score; pick two runs to compare. |
| **Run page** | Findings (resolved or remaining after fixes), auto-fix results, the full report (sandboxed), and downloads (HTML, Markdown, JSON). |
| **Compare** | Score changes plus findings resolved, new, or present in both. Use it for model v1 vs v2, or for Claude vs another model. |
| **Datasets** | Upload `.csv` (tabular) or `.jsonl` (LLM cases), which are validated on upload. |
| **Keys** | Save provider keys (BYOK). |
| **Settings** | API tokens for CI; delete the account and all its data. |

## Bring your own key (BYOK)

- Runs that call an LLM provider use the **user's** key and are billed to the user's provider account. The portal operator pays nothing for model usage.
- Keys are encrypted at rest with Fernet (AES-128-CBC plus HMAC-SHA256). The encryption key is derived (HKDF) from `PORTAL_SECRET_KEY`.
- After saving, a key is never shown again; only its last 4 characters are displayed.
- A key is decrypted only in the worker, in memory, while a run that uses it executes. It never enters the stored run config, the log, the database summary or the reports, and is redacted if a provider error message echoes it. The test suite checks every file under the data directory for the key.
- Before starting an LLM run, the form shows the approximate request count. Auto-fix for LLM runs is **off by default**, because each guardrail candidate re-runs every case.
- Limits for each run: `PORTAL_MAX_LLM_CASES` and `PORTAL_MAX_LLM_OUTPUT_TOKENS`. Concurrency for each user: `PORTAL_MAX_ACTIVE_RUNS`.

## Testing a deployed ML model (endpoint)

The portal calls your endpoint with `POST` and JSON:

```json
{"feature_names": ["age", "income", "region=north", "region=south"],
 "instances": [[34, 52000, 1, 0], [51, null, 0, 1]]}
```

Your endpoint returns either or both of:

```json
{"predictions": [0, 1]}
{"probabilities": [[0.8, 0.2], [0.3, 0.7]]}
```

Notes:
- Features are the numeric matrix built from the CSV: text columns are one-hot encoded as `column=value`, and missing values are sent as `null`.
- Probability columns are in sorted class order.
- Probabilities enable AUC, calibration, threshold and fairness checks and fixes.
- Retraining fixes are skipped, because the portal cannot retrain a remote model. Post-processing fixes (threshold, calibration, group thresholds) still run, and the report shows you what to change.
- The robustness and explainability checks send perturbed copies of the test set, so expect a few hundred requests. Requests are batched (500 rows each) and capped per run.

## Security model

| Risk | Control |
|---|---|
| Users running code on the server | Users never submit YAML or code. The portal builds scenarios from allow-listed fields only, so the engine's `python:` and `pickle:` loaders cannot be reached. Custom Python models stay a CLI feature. |
| SSRF through user URLs (LLM base URL, model endpoint) | `http`/`https` only; no credentials in URLs; hostnames resolving to loopback, private, link-local (including cloud metadata), multicast or reserved ranges are refused; redirects are refused. Self-hosted single-team installs can opt out with `PORTAL_ALLOW_PRIVATE_URLS=true`. |
| Cross-tenant access | Every query filters by the owner's `user_id`. Run and dataset IDs are random 128-bit values. Tests confirm that another user gets a 404. |
| XSS through model output in reports | Reports HTML-escape all content and are served with `Content-Security-Policy: sandbox` inside a sandboxed iframe. App pages use a strict CSP (no inline scripts). |
| CSRF | A per-session token on every form POST, plus `SameSite=Lax`, `HttpOnly` cookies. |
| Credential stuffing | Passwords are hashed with scrypt. Login is throttled to 10 failures per 15 minutes per IP and email. Sessions expire. |
| Uploaded files | Extension allow-list, size cap, strict UTF-8, and structure validation. Files are stored under random names in a folder for each user, never under the uploaded file name. |

## REST API

Authenticate with `Authorization: Bearer <token>` (create a token under **Settings**). The interactive reference is at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness check (no auth) |
| GET | `/api/v1/catalog` | Demos, LLM suites, frameworks, guardrails, models, limits |
| POST | `/api/v1/datasets` | Upload (multipart: `file`, optional `name`) |
| GET | `/api/v1/datasets` / `/api/v1/keys` | List datasets / saved keys (masked) |
| POST | `/api/v1/runs` | Start a run (JSON body as in `RunRequest`, `portal/scenarios.py`) → 202 |
| GET | `/api/v1/runs`, `/api/v1/runs/{id}` | Status and summary |
| GET | `/api/v1/runs/{id}/report?format=json\|md\|html` | Report |
| DELETE | `/api/v1/runs/{id}` | Delete a run |

Example CI gate:

```bash
RUN=$(curl -s -X POST "$PORTAL/api/v1/runs" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"kind":"llm","name":"nightly","cases":"redteam","provider":"anthropic","key_id":"<key id>","llm_model":"claude-opus-5-5","autofix":false}' | jq -r .id)
until [ "$(curl -s -H "Authorization: Bearer $TOKEN" "$PORTAL/api/v1/runs/$RUN" | jq -r .status)" != "queued" ] && \
      [ "$(curl -s -H "Authorization: Bearer $TOKEN" "$PORTAL/api/v1/runs/$RUN" | jq -r .status)" != "running" ]; do sleep 10; done
curl -s -H "Authorization: Bearer $TOKEN" "$PORTAL/api/v1/runs/$RUN" | jq -e '.summary.verdict != "BLOCK"'
```

## Production checklist

1. Set `PORTAL_SECRET_KEY` to a long random value and store it in a secrets manager. **If it is lost, saved keys cannot be decrypted.**
2. Serve over HTTPS behind a reverse proxy (Caddy, nginx, a cloud load balancer) and set `PORTAL_SECURE_COOKIES=true`.
3. Keep `PORTAL_ALLOW_PRIVATE_URLS=false` for any deployment open to people outside your team. Also block egress from the container to internal networks at the network layer, because DNS rebinding can get past a hostname check.
4. Back up `PORTAL_DATA_DIR`.
5. Set `PORTAL_ALLOW_SIGNUP=false` for an invite-only or internal portal.
6. Publish your own terms of service and privacy policy before inviting outside users. Uploaded datasets may contain personal data, and you are its processor.

## Scaling beyond one server

This first version uses SQLite and an in-process thread pool, which suits one server and a team or small customer base. To grow:
- move to Postgres (the queries in `portal/db.py` are plain SQL);
- run runs on a queue (Redis with RQ or Celery) with separate worker containers;
- store reports in object storage;
- add organisations and roles, SSO, billing and usage metering.
