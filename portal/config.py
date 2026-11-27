"""Portal settings, read from environment variables (see docs/PORTAL.md)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

def _find_root() -> Path:
    """Folder holding scenarios/ and datasets/: PORTAL_APP_ROOT, the source checkout, or the working dir."""
    if os.environ.get("PORTAL_APP_ROOT"):
        return Path(os.environ["PORTAL_APP_ROOT"]).resolve()
    for cand in (Path(__file__).resolve().parent.parent, Path.cwd()):
        if (cand / "scenarios").is_dir():
            return cand
    return Path.cwd()


ROOT = _find_root()


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PORTAL_DATA_DIR", ROOT / "portal_data")).resolve())
    secret_key: str = field(default_factory=lambda: os.environ.get("PORTAL_SECRET_KEY", ""))
    allow_signup: bool = field(default_factory=lambda: _bool("PORTAL_ALLOW_SIGNUP", True))
    # Hosted (multi-tenant) deployments must keep this False: user-supplied URLs may not reach
    # localhost / private networks / cloud metadata. Self-hosted single-team installs can enable it
    # to test models on their own network (e.g. Ollama on localhost).
    allow_private_urls: bool = field(default_factory=lambda: _bool("PORTAL_ALLOW_PRIVATE_URLS", False))
    secure_cookies: bool = field(default_factory=lambda: _bool("PORTAL_SECURE_COOKIES", False))
    workers: int = field(default_factory=lambda: int(os.environ.get("PORTAL_WORKERS", "2")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("PORTAL_MAX_UPLOAD_MB", "25")))
    max_active_runs_per_user: int = field(default_factory=lambda: int(os.environ.get("PORTAL_MAX_ACTIVE_RUNS", "3")))
    max_llm_cases: int = field(default_factory=lambda: int(os.environ.get("PORTAL_MAX_LLM_CASES", "200")))
    max_llm_output_tokens: int = field(default_factory=lambda: int(os.environ.get("PORTAL_MAX_LLM_OUTPUT_TOKENS", "4096")))
    session_days: int = field(default_factory=lambda: int(os.environ.get("PORTAL_SESSION_DAYS", "14")))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.secret_key:
            # Development fallback: generate once and keep it next to the data. In production set
            # PORTAL_SECRET_KEY (losing it makes every stored provider key unreadable).
            kf = self.data_dir / "secret.key"
            if not kf.exists():
                kf.write_text(secrets.token_urlsafe(48), encoding="utf-8")
            self.secret_key = kf.read_text(encoding="utf-8").strip()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "portal.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"
