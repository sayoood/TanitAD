#!/usr/bin/env python3
"""rehearsal_world.py <root> <fail|pass> -- builds a THROWAWAY world on C: for rehearsing the REAL-mode chain's wait / pull-check / commit-gate / rollout-failure paths without a real r2 run:
  <root>/art/        pulled run directories (A s0 only; a FAKE checkpoint, so the first rollout must fail and the chain must STOP -- nothing is ever read)
  <root>/box2.out    a synthetic box-2 chain log
  fail: the box-2 log is the REAL r1 snapshot (A s0 K-REACH FAIL): the chain must STOP on it before touching any checkpoint.
  pass: the log shows A s0 pulled + run record PASS + K-REACH PASS; the pull has all 8 files and a token binding the BASE commit (nothing differs from the base, so the rollout-code gate passes);
        the chain must get through wait, pullcheck, commit gate, then STOP at the rollout of the fake checkpoint ('rollout ... failed')."""
import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dagorct1_lib as L  # noqa: E402
import dagorct1_selftest as S  # noqa: E402

root, mode = sys.argv[1].replace("\\", "/"), sys.argv[2]
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root + "/art")
os.makedirs(root + "/out")
os.makedirs(root + "/logs")
if mode == "fail":
    shutil.copy(S.SNAP, root + "/box2.out")
else:
    d, tg = S.mk_pull(root + "/art", "dagoaccorc", 0, commit=L.BASE_COMMIT)
    S.write_sums(d, tg)
    ck = hashlib.sha256(open(d + "/ckpt_last.pt", "rb").read()).hexdigest()
    pth = f"{root}/art/{L.run_name('dagoaccorc', 0)}"
    open(root + "/box2.out", "w", newline="\n").write(
        "\n".join([
            "10-28 02:00 box 2 code = tip abc",
            f"10-28 02:09 {L.run_name('dagoaccorc', 0)} pulled to {pth} (checkpoint sha256 {ck})",
            "DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorc s0 | trainer seed 0 | argv sha16 b2e18d0d8af33e42 | steps 3000 | ego_perturb_p 0.3 | w_acc 49.6969",
            "DAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorc | swap 14688/145037 = 0.1013 (must lie in 0.1 +- 0.02) | ego 23000/76898 = 0.2991 (A: 0.3 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag True",
        ]) + "\n")
    time.sleep(4)      # the chain refuses a SHA256SUMS.txt younger than 3 s
print("world ready:", root, mode)
