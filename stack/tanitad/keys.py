"""Local secret loading for the TanitAD stack — contains NO secrets itself.

Credentials live in the **git-ignored** ``Keys.txt`` at the repo root (OpenRouter,
Hugging Face, Google AI Studio). This module parses them into standard
environment variables so stack code and the scheduled agents share one entry
point, without any key ever touching tracked files:

    from tanitad.keys import enable_tls, load_keys
    enable_tls()     # route Python TLS through the OS trust store
    load_keys()      # populate os.environ from the git-ignored Keys.txt

Env vars set (only when found; never overwrites an already-set value):
    OPENROUTER_API_KEY
    HF_TOKEN, HUGGING_FACE_HUB_TOKEN      (huggingface_hub reads the latter)
    GOOGLE_API_KEY, GEMINI_API_KEY

Why ``enable_tls()``: the dev machine sits behind an intercepting HTTPS proxy
whose root CA is only in the Windows certificate store, so stdlib/``certifi``
verification fails ("unable to get local issuer certificate"). ``truststore``
routes verification through the OS store (which trusts that CA) — secure, no
verification disabling. Measured 2026-07-06: HF whoami succeeds only with this.

Security contract: this file has NO secrets; it reads a git-ignored file that is
never committed (verified: ``Keys.txt`` absent from all git history). Do not log
the values or write them to tracked files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Match on the *value shape* (robust to however the labels are worded in Keys.txt).
_PATTERNS: dict[str, str] = {
    "OPENROUTER_API_KEY": r"sk-or-v1-[A-Za-z0-9]+",
    "HF_TOKEN": r"hf_[A-Za-z0-9]+",
    "GOOGLE_API_KEY": r"AIza[A-Za-z0-9_\-]+",
}
# Extra env-var names that third-party libs expect for the same secret.
_ALIASES: dict[str, list[str]] = {
    "HF_TOKEN": ["HUGGING_FACE_HUB_TOKEN"],
    "GOOGLE_API_KEY": ["GEMINI_API_KEY"],
}


def _repo_root() -> Path:
    # stack/tanitad/keys.py -> tanitad -> stack -> repo root
    return Path(__file__).resolve().parents[2]


def keys_path() -> Path:
    """Location of the git-ignored keys file (override with ``TANITAD_KEYS_FILE``)."""
    override = os.environ.get("TANITAD_KEYS_FILE")
    return Path(override) if override else _repo_root() / "Keys.txt"


def load_keys(path: str | Path | None = None, *, overwrite: bool = False) -> list[str]:
    """Parse the keys file into ``os.environ``; return the env-var names set.

    Missing file -> returns ``[]`` (no error), so this is safe to call anywhere.
    """
    p = Path(path) if path is not None else keys_path()
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    set_vars: list[str] = []
    for var, pat in _PATTERNS.items():
        m = re.search(pat, txt)
        if not m:
            continue
        for name in (var, *_ALIASES.get(var, [])):
            if overwrite or not os.environ.get(name):
                os.environ[name] = m.group(0)
                set_vars.append(name)
    return set_vars


def enable_tls() -> bool:
    """Route Python TLS through the OS trust store (needed behind the dev-machine
    proxy). Returns True if applied, False if ``truststore`` isn't installed
    (``pip install -e .[net]``). Safe to call more than once."""
    try:
        import truststore
        truststore.inject_into_ssl()
        return True
    except Exception:
        return False
