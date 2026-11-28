"""mdfix_common.py -- PREREG_HSGATE_MDFIX helpers for PHASE B (forward / verdict / bars copies).  NEW file, refcv9-hsgate agent, 2026-10-26.

REGISTERED_MDFIX.sha256 (written by the Master Mind) is a MULTI-LINE pin file whose first lines are '#' comments and whose comment line 2 carries the
PREREG's sha256.  ship_r8_sub_common.read_seal / hex64 read the FIRST whitespace token ('#') and would return '' (fail-safe, but blocking), and a naive
'first 64-hex token' reader would return the PREREG sha -- NOT the gate's.  So every pinned sha is read here by its EXACT path line ('<sha> *<path>'),
exactly one line per path, 64 lower-case hex, else ''.
"""
from __future__ import annotations

import re
from pathlib import Path

import hsgate_common as H

REG_MDFIX = H.WORK / "REGISTERED_MDFIX.sha256"
PREREG_MDFIX = H.WORK / "PREREG_HSGATE_MDFIX.md"
PREREG_MDFIX_SHA = H.WORK / "PREREG_HSGATE_MDFIX.sha256"
PINNED = ("MDFIX_FROZEN.json", "next/MDFIX_POWER_FIT.json", "MDFIX_FIT_INFO.json", "code/mdfix_fit.py")
_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](\S.*)$")


def registered_sha(relpath: str, reg: Path = REG_MDFIX) -> str:
    try:
        lines = Path(reg).read_text(encoding="utf-8-sig").splitlines()
    except Exception:          # noqa: BLE001
        return ""
    hits = []
    for ln in lines:
        if ln.lstrip().startswith("#"):
            continue
        m = _LINE.match(ln.strip())
        if m and m.group(2).strip().replace("\\", "/") == relpath:
            hits.append(m.group(1).lower())
    return hits[0] if len(hits) == 1 else ""


def check_prereg_mdfix() -> dict:
    a, b = H.sha256_file(PREREG_MDFIX), H.first_token(PREREG_MDFIX_SHA)
    if len(a) != 64 or len(b) != 64 or a != b:
        raise SystemExit(f"[mdfix] PREREG_HSGATE_MDFIX seal check failed ({len(a)}/{len(b)} chars, equal={a == b})")
    return {"file": str(PREREG_MDFIX), "sha256": a}


def check_registration() -> dict:
    """REFUSES (SystemExit) unless every file pinned in REGISTERED_MDFIX.sha256 re-hashes NOW to its pin (both sides 64 hex)."""
    out = {"registration_file": str(REG_MDFIX), "registration_file_sha256": H.sha256_file(REG_MDFIX), "pins": {}}
    for p in PINNED:
        reg, now = registered_sha(p), H.sha256_file(H.WORK / p)
        out["pins"][p] = {"registered": reg, "now": now}
        if len(reg) != 64 or len(now) != 64:
            raise SystemExit(f"[mdfix] INCONCLUSIVE registration for {p}: registered {len(reg)} chars, file {len(now)} chars -- refusing")
        if reg != now:
            raise SystemExit(f"[mdfix] {p} hashes to {now}, REGISTERED_MDFIX pins {reg} -- refusing")
    out["ALL_MATCH"] = True
    return out
