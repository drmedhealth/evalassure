"""Passwords, tokens, encryption of BYOK provider keys, and outbound-URL checks."""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import secrets
import socket
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ------------------------------------------------------------------ passwords

_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, dk_b64 = stored.split("$")
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), **_SCRYPT)
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except Exception:
        return False


# ------------------------------------------------------------------ tokens

def new_token(prefix: str = "") -> str:
    return prefix + secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ------------------------------------------------------------------ BYOK encryption

class KeyVault:
    """Encrypts provider API keys at rest (Fernet = AES-128-CBC + HMAC-SHA256).

    The encryption key is derived from PORTAL_SECRET_KEY with HKDF. Plaintext keys exist only
    in memory while a run that needs them is executing; they are never written to the database,
    run config, logs or reports.
    """

    def __init__(self, secret_key: str) -> None:
        raw = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"aitesteval-portal",
                   info=b"byok-provider-keys").derive(secret_key.encode())
        self._f = Fernet(base64.urlsafe_b64encode(raw))

    def encrypt(self, plaintext: str) -> str:
        return self._f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._f.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("stored key cannot be decrypted (was PORTAL_SECRET_KEY changed?)") from exc


def mask_key(key: str) -> str:
    return ("…" + key[-4:]) if len(key) >= 8 else "…"


# ------------------------------------------------------------------ outbound URL checks (SSRF)

class URLNotAllowed(ValueError):
    pass


def check_outbound_url(url: str, allow_private: bool) -> str:
    """Validate a user-supplied URL the server will call. Returns the URL or raises URLNotAllowed.

    Blocks non-http(s) schemes, embedded credentials and (unless allow_private) any hostname that
    resolves to loopback, private, link-local (incl. cloud metadata 169.254.169.254), multicast,
    reserved or unspecified addresses. Redirects are refused separately by the HTTP adapters.
    """
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        raise URLNotAllowed("URL must start with http:// or https://")
    if not p.hostname:
        raise URLNotAllowed("URL has no host")
    if p.username or p.password:
        raise URLNotAllowed("put credentials in a saved key, not in the URL")
    if allow_private:
        return url.strip()
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise URLNotAllowed(f"cannot resolve host '{p.hostname}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified or (ip.version == 6 and ip.ipv4_mapped and not ip.ipv4_mapped.is_global)):
            raise URLNotAllowed(f"'{p.hostname}' resolves to a private or internal address ({ip}); "
                                "hosted runs can only call public endpoints")
    return url.strip()
