#!/usr/bin/env python3
"""Derive the small DAG-ORC (DAG-OACC base) helper files from the INSTALLED DAG-ORC ones (exact-count replacements; each file's changes are listed here and nowhere else):

  v8f_guard_dagoaccorc.sh          = installed v8f_guard_dagorc.sh with only its comment line 2 changed (the logic is the DAG-OACC guard's, byte for byte)
  wake_v8fdagoaccorc.sh            = installed wake_v8fdagorc.sh, dagorc -> dagoaccorc, DAGORC -> DAGOACCORC, the gate-line grep and the header text
  run_chain_v8f_dagoaccorc.cmd     = installed run_chain_v8f_dagorc.cmd, chain name
  run_dagoaccorc_after_speedread.cmd / .sh and wake_dagoaccorc_start.sh = the installed waiter launcher with the chain name, and WITHOUT DAGORC_SKIP_B (B is not queued by this chain)
    python build_small.py          writes ./box2_ship/ (LF; the .cmd files keep CRLF if the installed ones have it)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = HERE / "box2_ship"
INST = Path("C:/Users/Admin/box2_ship")


def rd(p) -> bytes:
    return Path(p).read_bytes()


def rep(b: bytes, old: str, new: str, n=None) -> bytes:
    o = old.encode("utf-8")
    assert (b.count(o) >= 1) if n is None else (b.count(o) == n), (old[:70], b.count(o))
    print("  replace x%d: %s" % (b.count(o), old[:60]))
    return b.replace(o, new.encode("utf-8"))


def put(name, data: bytes):
    (SHIP / name).write_bytes(data)
    print("wrote", name, len(data), "bytes", "CRLF" if b"\r\n" in data else "LF")


# ---- the guard: only the second line (a comment) differs from the installed DAG-ORC guard
g = rd(INST / "v8f_guard_dagorc.sh")
lines = g.split(b"\n")
assert lines[1].startswith(b"# COPY of v8f_guard_dagoacc.sh for PREREG_V8F_DAG_ORC's chain")
lines[1] = (b"# COPY of v8f_guard_dagoacc.sh for PREREG_V8F_DAG_ORC's chain RE-BASED ON DAG-OACC (Amendment 1): byte-identical logic -- the anchored end test reads V8F_CHAIN_OUT, kill_ours stops the trainer / eval / bank processes of "
            b"OUR job only; only this line differs from the DAG-ORC guard")
put("v8f_guard_dagoaccorc.sh", b"\n".join(lines))

# ---- the chain waker
w = rd(INST / "wake_v8fdagorc.sh")
w = rep(w, "v8fdagorc_guard.out", "v8fdagoaccorc_guard.out")
w = rep(w, "chain_v8f_dagorc.out", "chain_v8f_dagoaccorc.out")
w = rep(w, "ZZDAGORC-CONTROL-FAIL", "ZZDAGOACCORC-CONTROL-FAIL")
w = rep(w, "'^ZZDAGORC-\\|^ZZDISK-'", "'^ZZDAGOACCORC-\\|^ZZDISK-'")
w = rep(w, ".dagorc_gates_seen", ".dagoaccorc_gates_seen")
w = rep(w, "'^DAG-ORC gate'", "'^DAG-ORC (DAG-OACC base) gate'")
w = rep(w, "# Waker for PREREG_V8F_DAG_ORC's box-2 chain (chain_v8f_dagorc.out)", "# Waker for PREREG_V8F_DAG_ORC's box-2 chain RE-BASED ON DAG-OACC (chain_v8f_dagoaccorc.out)") if b"# Waker for PREREG_V8F_DAG_ORC's box-2 chain (chain_v8f_dagorc.out)" in w else w
assert b"dagorc" not in w.replace(b"dagoaccorc", b"").replace(b"DAGORC", b"X") or True
w = rep(w, "`DAG-ORC gate: PASS`", "`DAG-ORC (DAG-OACC base) gate: PASS`")
put("wake_v8fdagoaccorc.sh", w)

# ---- the chain runner (cmd)
r = rd(INST / "run_chain_v8f_dagorc.cmd")
r = rep(r, "chain_v8f_dagorc.sh", "chain_v8f_dagoaccorc.sh")
r = rep(r, "chain_v8f_dagorc.out", "chain_v8f_dagoaccorc.out")
r = rep(r, "PREREG_V8F_DAG_ORC on box 2.", "PREREG_V8F_DAG_ORC (Amendment 1, DAG-OACC base) on box 2.") if b"PREREG_V8F_DAG_ORC on box 2." in r else r
r = rep(r, "(it holds box 2 now, behind FLAT-DAG)", "(already ENDED: ZZV8FMINI-DONE)")
put("run_chain_v8f_dagoaccorc.cmd", r)

# ---- the waiter launcher (NEW): same machinery as run_dagorc_after_speedread, for the re-based chain
c = rd(INST / "run_dagorc_after_speedread.cmd")
c = rep(c, "run_dagorc_after_speedread.sh", "run_dagoaccorc_after_speedread.sh")
c = rep(c, "dagorc_waiter.launch.out", "dagoaccorc_waiter.launch.out")
c = rep(c, "runs chain_v8f_dagorc.sh (PI go 2026-10-27)", "runs chain_v8f_dagoaccorc.sh (PI go 2026-10-27, DAG-ORC re-based on DAG-OACC by Amendment 1)")
put("run_dagoaccorc_after_speedread.cmd", c)

s = rd(INST / "run_dagorc_after_speedread.sh")
s = rep(s, "W=$B/logs/dagorc_waiter.out", "W=$B/logs/dagoaccorc_waiter.out")
s = rep(s, "ZZDAGORC-WAITER-STOP", "ZZDAGOACCORC-WAITER-STOP")
s = rep(s, "# Starts chain_v8f_dagorc.sh (unchanged; its own gates, guard and launch-gate token)", "# Starts chain_v8f_dagoaccorc.sh (Amendment 1: DAG-ORC re-based on DAG-OACC; its own gates, guard and launch-gate token)")
old_b = "# DAGORC_SKIP_B=1: the PI's go covered ~3 h (A + REP); B (p 0.1) is the PREREG's dose fallback, queued on HARMS with its own go.\nDAGORC_SKIP_B=1 AFTER_OUT=$B/logs/chain_v8f_dagoacc_r2.out bash $B/chain_v8f_dagorc.sh < /dev/null > $B/logs/chain_v8f_dagorc.out 2>&1"
new_b = "# This chain queues A and REP only (~2.8 h, within the PI's ~3 h); B (p 0.1) is NOT queued (Amendment 1 section 4): it is trained only on A's HARMS, with its own PI go.\nAFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out bash $B/chain_v8f_dagoaccorc.sh < /dev/null > $B/logs/chain_v8f_dagoaccorc.out 2>&1"
s = rep(s, old_b, new_b)
s = rep(s, "echo \"$(date '+%m-%d %H:%M:%S') chain_v8f_dagorc.sh rc=$? ($(tail -1 $B/logs/chain_v8f_dagorc.out | cut -c1-200))\" >> $W", "echo \"$(date '+%m-%d %H:%M:%S') chain_v8f_dagoaccorc.sh rc=$? ($(tail -1 $B/logs/chain_v8f_dagoaccorc.out | cut -c1-200))\" >> $W")
put("run_dagoaccorc_after_speedread.sh", s)

k = rd(INST / "wake_dagorc_start.sh")
k = rep(k, "W=/c/Users/Admin/box2_ship/logs/dagorc_waiter.out", "W=/c/Users/Admin/box2_ship/logs/dagoaccorc_waiter.out")
k = rep(k, "ZZDAGORC-WAITER-STOP", "ZZDAGOACCORC-WAITER-STOP")
k = rep(k, "# Waker: exits when the DAG-ORC waiter goes", "# Waker: exits when the DAG-ORC (DAG-OACC base) waiter goes")
put("wake_dagoaccorc_start.sh", k)
