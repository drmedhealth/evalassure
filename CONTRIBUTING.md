# Contributing

Thanks for helping. Bug fixes, new evaluators, compliance controls, scenarios
and docs are all welcome.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,portal]"
pytest -q
```

Run the portal locally with `python -m portal --port 8765`.

## Pull requests

- Keep each PR focused on one change, and add or update tests for it.
- `pytest -q` must pass.
- New compliance controls must cite the published requirement they map to
  (article, clause or section) and must not claim legal conformity.
- New red-team cases must be generic test prompts; do not add real exploits
  against third-party systems or content that is harmful in itself.
- Never commit API keys, real personal data, or anything under `portal_data/`.
  Use synthetic data (see `evalassure/data/synthetic.py`).

## Where things live

See "Project layout" in the [README](README.md). Portal internals and the
security model are described in [docs/PORTAL.md](docs/PORTAL.md).

By contributing you agree that your contribution is licensed under the
[MIT License](LICENSE).
