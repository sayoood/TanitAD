"""A2 -- the NAVTEST QUARANTINE GATE for the behaviour-label path (and any other label/training path that wants it).

navtest is EVAL ONLY.  This gate refuses a (log_name, token) that is in the navtest list, by TWO independent keys, so a renamed log or a re-keyed token cannot slip through:
  * LOG gate   -- the 147 navtest log names (= nuPlan DB stems); data: gate_data/navtest_logs.txt
  * TOKEN gate -- every frame token and scene token of those logs; data: gate_data/navtest_tokens.txt.gz
Both files are verified against navtest_gate.json's sha256 on load; a missing / altered / empty list REFUSES EVERYTHING (fail closed), never passes everything.

Deliberate-regression arm (the gate's own RED test): REFE_LAB_MUTATE=no_navtest_gate disables the check; tests/test_behaviour_gate.py asserts that, under it, a navtest
log is NOT refused (so the "refused" assertion would go RED) -- i.e. the test is sensitive to the gate being present.
Data dir: env A2_GATE_DATA, else <this dir>/gate_data, else the first of ../gate_data, ../../gate_data holding navtest_gate.json (a2/ layout).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

MUTATE_ENV = "REFE_LAB_MUTATE"
MUTATION = "no_navtest_gate"
_CACHE: dict = {}


class NavtestLeak(RuntimeError):
    """A navtest log / token was offered to a label or training path."""


class GateDataError(RuntimeError):
    """The quarantine lists are missing, empty or do not match their recorded sha256 -- the gate cannot vouch, so it refuses."""


def _data_dir() -> Path:
    if os.environ.get("A2_GATE_DATA"):
        return Path(os.environ["A2_GATE_DATA"])
    here = Path(__file__).resolve().parent
    for d in (here / "gate_data", here.parent / "gate_data", here.parent.parent / "gate_data"):
        if (d / "navtest_gate.json").is_file():
            return d
    raise GateDataError(f"no gate_data/navtest_gate.json beside {here} (set A2_GATE_DATA)")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(force: bool = False) -> tuple:
    """-> (frozenset of log names, frozenset of tokens).  Verified, cached."""
    d = _data_dir()
    if not force and _CACHE.get("dir") == str(d):
        return _CACHE["logs"], _CACHE["toks"]
    meta = json.loads((d / "navtest_gate.json").read_text())
    lp, tp = d / "navtest_logs.txt", d / "navtest_tokens.txt.gz"
    for p, key in ((lp, "logs_sha256"), (tp, "tokens_sha256")):
        if not p.is_file() or _sha(p) != meta[key]:
            raise GateDataError(f"{p} is missing or its sha256 differs from navtest_gate.json:{key} -- refusing (fail closed)")
    logs = frozenset(x for x in lp.read_text().split() if x)
    with gzip.open(tp, "rt") as fh:
        toks = frozenset(x for x in fh.read().split() if x)
    if len(logs) != meta["n_logs"] or len(toks) != meta["n_tokens"] or not logs or not toks:
        raise GateDataError(f"navtest lists have {len(logs)} logs / {len(toks)} tokens, expected {meta['n_logs']} / {meta['n_tokens']} -- refusing")
    _CACHE.update(dir=str(d), logs=logs, toks=toks)
    return logs, toks


def is_navtest(log_name: str, token: str) -> tuple:
    """-> (log_hit, token_hit).  Pure lookup, ignores the mutation switch."""
    logs, toks = load()
    ln = str(log_name or "")
    ln = ln[:-3] if ln.endswith(".db") else ln                  # a DB filename is accepted too
    return (ln in logs, str(token or "").lower() in toks)


def assert_not_navtest(log_name: str, token: str) -> None:
    """Raise NavtestLeak if the log OR the token is navtest.  An empty log name or token is refused as well (an unkeyed row cannot be proven disjoint)."""
    if os.environ.get(MUTATE_ENV) == MUTATION:
        return                                                    # DELIBERATE REGRESSION ARM: the gate is off
    if not log_name or not token:
        raise NavtestLeak(f"unkeyed sample (log_name={log_name!r}, token={token!r}): cannot prove it is not navtest")
    lh, th = is_navtest(log_name, token)
    if lh or th:
        raise NavtestLeak(f"navtest is EVAL ONLY: log {log_name!r} ({'LOG' if lh else '-'}) token {token!r} ({'TOKEN' if th else '-'}) are in the quarantine list")


def assert_batch(metas) -> None:
    """metas: iterable of dicts with log_name / token (the labeller's job meta)."""
    for m in metas:
        assert_not_navtest(m.get("log_name"), m.get("token"))
