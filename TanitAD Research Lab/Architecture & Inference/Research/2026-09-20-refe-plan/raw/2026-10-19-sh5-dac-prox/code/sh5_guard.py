"""The seal guard.  Every stage that can produce a number from REAL data (`prep --labels`, `fit`, `analyze`, `sh5_ctx_build.py run`) calls `require_sealed()` first and REFUSES unless

  * PREREG_SH5_DAC_PROX.md hashes to the sha256 recorded in PREREG_SH5_DAC_PROX.sha256 (and the .sha256 file exists: no seal, no number), and
  * every file in CODE_PINS.json hashes to its pinned sha256 (a pinned file edited after the seal = refuse; the fix is a written Amendment before the stage runs).  An Amendment k is
    PREREG_SH5_DAC_PROX_AMENDMENT<k>.md + its .sha256 + CODE_PINS_AMENDMENT<k>.json ({"from": {file: previous pin}, "to": {file: new pin}}); the chain is read in order and a file can only move from its
    CURRENT pin (Amendment 1 re-pinned the G-LIVE autocast fix).

The guard itself and `seal.py` are the checker, not the checked: they are listed with their own hashes in SEALED_AT.txt.  The CPU tests call the lower-level functions and never need a seal.
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
PREREG = os.path.join(ROOT, "PREREG_SH5_DAC_PROX.md")
PREREG_SHA = os.path.join(ROOT, "PREREG_SH5_DAC_PROX.sha256")
CODE_PINS = os.path.join(ROOT, "CODE_PINS.json")


class NotSealed(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def require_sealed(root: str = None) -> dict:
    r = root or ROOT
    pre, rec, pins = os.path.join(r, "PREREG_SH5_DAC_PROX.md"), os.path.join(r, "PREREG_SH5_DAC_PROX.sha256"), os.path.join(r, "CODE_PINS.json")
    for p in (pre, rec, pins):
        if not os.path.exists(p):
            raise NotSealed(f"{p} is missing: no seal, no number")
    want = open(rec, encoding="utf-8").read().split()[0]
    got = sha256_file(pre)
    if got != want:
        raise NotSealed(f"PREREG sha256 {got} != recorded {want}: every stage refuses")
    merged = dict(json.load(open(pins, encoding="utf-8"))["files"])
    chain = []
    k = 1
    while os.path.exists(os.path.join(r, f"PREREG_SH5_DAC_PROX_AMENDMENT{k}.md")):                # a written Amendment re-pins ONLY files that were already pinned, and only from their CURRENT pin
        md = os.path.join(r, f"PREREG_SH5_DAC_PROX_AMENDMENT{k}.md")
        sh, pj = os.path.join(r, f"PREREG_SH5_DAC_PROX_AMENDMENT{k}.sha256"), os.path.join(r, f"CODE_PINS_AMENDMENT{k}.json")
        for q in (sh, pj):
            if not os.path.exists(q):
                raise NotSealed(f"{q} is missing: an Amendment without its hash and its pins does not count")
        if sha256_file(md) != open(sh, encoding="utf-8").read().split()[0]:
            raise NotSealed(f"AMENDMENT{k} sha256 does not match its record")
        am = json.load(open(pj, encoding="utf-8"))
        for name, new_h in am["to"].items():
            if name not in merged or merged[name] != am["from"].get(name):
                raise NotSealed(f"AMENDMENT{k}: {name} is not pinned at the hash the amendment starts from (chain broken)")
            merged[name] = new_h
        chain.append(k)
        k += 1
    bad = []
    for name, h in merged.items():
        p = os.path.join(r, name)
        if not os.path.exists(p) or sha256_file(p) != h:
            bad.append(name)
    if bad:
        raise NotSealed(f"pinned code changed after the seal (or after the last Amendment): {bad}; write an Amendment before running")
    return {"prereg": got, "pins": len(merged), "amendments": chain}
