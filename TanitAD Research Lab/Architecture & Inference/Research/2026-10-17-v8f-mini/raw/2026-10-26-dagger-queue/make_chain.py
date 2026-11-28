#!/usr/bin/env python3
"""Generate the dev-box files of the DAgger chain into dagq/box2_ship/ from the ORIGINALS in C:/Users/Admin/box2_ship (read-only; never edited):

  chain_v8f_dag.sh      = chain_v8f_dag.sh.in with the banked-checkpoint line COPIED from chain_v8f_orc.sh (its `CKPT_SHA=(...)` line), never retyped
  v8f_guard_dag.sh      = v8f_guard.sh with the bank generators added to the processes a temperature / time stop kills
  run_chain_v8f_dag.cmd = the detached launcher, the form of run_chain_v8f_orc.cmd
  wake_v8fdag.sh        = the waker, the form of wake_v8forc2.sh

LF line endings for the .sh files (bash), CRLF for the .cmd.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = Path("C:/Users/Admin/box2_ship")
OUT = HERE / "box2_ship"


def lf(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def main() -> int:
    orc = (SHIP / "chain_v8f_orc.sh").read_text(encoding="utf-8")
    m = re.findall(r"^CKPT_SHA=\(.*\)\s*$", orc, flags=re.M)
    if len(m) != 1:
        sys.exit(f"chain_v8f_orc.sh: expected exactly one CKPT_SHA=(...) line, found {len(m)}")
    src = (OUT / "chain_v8f_dag.sh.in").read_text(encoding="utf-8")
    assert src.count("@@CKPT_SHA@@") == 1
    (OUT / "chain_v8f_dag.sh").write_bytes(lf(src.replace("@@CKPT_SHA@@", m[0].strip())))
    guard = (SHIP / "v8f_guard.sh").read_text(encoding="utf-8")
    old = "v8f_mini_train|v8f_mini_eval|v8f_lora_smoke"
    if guard.count(old) != 1:
        sys.exit("v8f_guard.sh: the process pattern was not found exactly once")
    g = guard.replace(old, "v8f_mini_train|v8f_mini_eval|v8f_dagger_bank|v8f_dagger_queue_tool|v8f_lora_smoke")
    g = g.replace("# Box 2 guard for the v8F-mini chain", "# COPY of v8f_guard.sh for PREREG_V8F_DAGGER's chain: the ONLY change is that kill_ours also stops the bank generators (v8f_dagger_bank / v8f_dagger_queue_tool), which read cache_full on the USB HDD.\n# Box 2 guard for the v8F-mini chain", 1)
    (OUT / "v8f_guard_dag.sh").write_bytes(lf(g))
    cmd = ["@echo off",
           "rem PREREG_V8F_DAGGER on box 2. NOT RUN: it needs the PI's go (box 2 is the PI's machine). The chain refuses to start without AFTER_OUT (the log of the chain that holds box 2) and, before touching anything, reads PREREG_V8F_ORECOVER's read:",
           "rem ROBUST / PARTIAL park the arm, REFUSED / UNPLACED stop it. Set AFTER_OUT to the ORC chain's log (it ended with ZZV8FMINI-DONE).",
           '"C:\\Program Files\\Git\\bin\\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_orc.out bash /c/Users/Admin/box2_ship/chain_v8f_dag.sh < /dev/null > /c/Users/Admin/box2_ship/logs/chain_v8f_dag.out 2>&1" < NUL']
    (OUT / "run_chain_v8f_dag.cmd").write_bytes(("\r\n".join(cmd) + "\r\n").encode("utf-8"))
    wake = r'''#!/usr/bin/env bash
# Waker for PREREG_V8F_DAGGER's box-2 chain (chain_v8f_dag.out). Exits on: the chain's end / stop marker (a BLOCKED, REFUSED or PARKED stop is a stop), a guard stop, the four gates passing (once; they come first after the banks),
# each bank pulled, or each run's pull landing on E:. Script polling, not model turns.
L=/c/Users/Admin/box2_ship/logs; O=$L/chain_v8f_dag.out
np0=$(grep -ac ' pulled to ' $O 2>/dev/null); np0=${np0:-0}
for i in $(seq 1 1500); do
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O 2>/dev/null && { echo "CHAIN END"; tail -15 $O; exit 0; }
  grep -aq ZZGUARD-STOP $L/v8fdag_guard.out 2>/dev/null && { echo "GUARD: $(grep -a ZZGUARD-STOP $L/v8fdag_guard.out | tail -1)"; tail -8 $O; exit 4; }
  grep -aq '^ZZDAG-BLOCKED\|^ZZDAG-REFUSED' $O 2>/dev/null && { echo "BANK JUDGE: $(grep -a '^ZZDAG-' $O | tail -2)"; tail -6 $O; exit 0; }
  [ "$(grep -ac '^DAG gate: PASS' $O 2>/dev/null)" -ge 4 ] && [ ! -f $L/.dag_gates_seen ] && { touch $L/.dag_gates_seen; echo "GATES PASSED"; grep -a '^DAG gate' $O; tail -4 $O; exit 0; }
  n=$(grep -ac ' pulled to ' $O 2>/dev/null); [ "${n:-0}" -gt "$np0" ] && { echo "NEW PULL"; grep -a ' pulled to ' $O | tail -1; tail -6 $O; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 $O
'''
    (OUT / "wake_v8fdag.sh").write_bytes(lf(wake))
    print("wrote", [p.name for p in sorted(OUT.glob("*")) if p.name != "chain_v8f_dag.sh.in"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
