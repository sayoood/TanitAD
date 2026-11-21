#!/usr/bin/env python3
"""Derive the small r2 files from r1's generated ones (exact-count replacements; each file's changes are listed here and nowhere else). r1's files (box2_ship/ of dagorc_rebase/chain, installed in C:/Users/Admin/box2_ship) are never touched.

  wake_v8fdagoaccorc_r2.sh          = r1's waker with the r2 log names (chain_v8f_dagoaccorc_r2.out, v8fdagoaccorc_r2_guard.out) and its own gates-seen marker (.dagoaccorc_r2_gates_seen)
  run_chain_v8f_dagoaccorc_r2.cmd   = r1's direct-start runner with the r2 chain name / log and AFTER_OUT = the r1 chain's log (no waiter: the SPEEDREAD pull has ended)
  The guard is r1's INSTALLED v8f_guard_dagoaccorc.sh, unchanged (its anchored end test reads V8F_CHAIN_OUT, which the r2 chain sets to its own log); the asserter is r1's, unchanged.
    python build_r2_small.py          writes ../box2_ship/ (LF; the .cmd keeps CRLF)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP2 = HERE.parent / "box2_ship"
R1 = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/box2_ship")


def rd(p) -> bytes:
    return Path(p).read_bytes()


def rep(b: bytes, old: str, new: str, n=None) -> bytes:
    o = old.encode("utf-8")
    assert (b.count(o) >= 1) if n is None else (b.count(o) == n), (old[:70], b.count(o), n)
    print("  replace x%d: %s" % (b.count(o), old[:60]))
    return b.replace(o, new.encode("utf-8"))


def put(name, data: bytes):
    (SHIP2 / name).write_bytes(data)
    print("wrote", name, len(data), "bytes", "CRLF" if b"\r\n" in data else "LF")


w = rd(R1 / "wake_v8fdagoaccorc.sh")
w = rep(w, "v8fdagoaccorc_guard.out", "v8fdagoaccorc_r2_guard.out", 2)
w = rep(w, "chain_v8f_dagoaccorc.out", "chain_v8f_dagoaccorc_r2.out", 2)
w = rep(w, ".dagoaccorc_gates_seen", ".dagoaccorc_r2_gates_seen", 2)
w = rep(w, "# Waker for PREREG_V8F_DAG_ORC's box-2 chain (chain_v8f_dagoaccorc_r2.out).", "# Waker for r2 (the re-run after the r1 K-REACH stop) of PREREG_V8F_DAG_ORC's box-2 chain (chain_v8f_dagoaccorc_r2.out). The r2 aside's lines (^ASIDE-) come first; an ASIDE-FAIL is a ZZV8FMINI-STOP of the chain and wakes below.", 1)
put("wake_v8fdagoaccorc_r2.sh", w)

r = rd(R1 / "run_chain_v8f_dagoaccorc.cmd")
r = rep(r, "rem PREREG_V8F_DAG_ORC (Amendment 1, DAG-OACC base) on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the DAG-OACC r2 chain's log (already ENDED: ZZV8FMINI-DONE); a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does.",
        "rem PREREG_V8F_DAG_ORC (Amendment 1, DAG-OACC base) r2 on box 2: ALL FOUR RUNS FROM THEIR GATES after the r1 chain's K-REACH stop. NOT RUN: needs the PI's go and the landed trainer fix (STREAM_PATTERN / STREAM_RECORD set). AFTER_OUT = the r1 chain's log: the chain GOes ONLY if it ENDED in exactly ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc with BOX2_BUSY released; anything else HALTs.", 1)
r = rep(r, "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out", "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out", 1)
r = rep(r, "chain_v8f_dagoaccorc.sh", "chain_v8f_dagoaccorc_r2.sh", 1)
r = rep(r, "> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out 2>&1", "> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out 2>&1", 1)
put("run_chain_v8f_dagoaccorc_r2.cmd", r)
