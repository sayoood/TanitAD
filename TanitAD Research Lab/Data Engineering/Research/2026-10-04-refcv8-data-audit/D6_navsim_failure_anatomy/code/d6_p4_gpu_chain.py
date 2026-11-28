"""D6 P4 GPU chain -- runs ONLY as the child of the battery's ``with_gpu_lock.py`` (the dev-box GPU lock is then HELD by the wrapper).

    <tanitad python> with_gpu_lock.py --job refcv8-d6-p4 --log raw/p4_gpu_wait.log --rec raw/p4_gpu_rec.json --max-wait-s 43200 \
        --child-timeout-s 10800 -- <tanitad python> d6_p4_gpu_chain.py

ONE stage: ``d6_p4_export.py`` (= the D6 bridge copy + the mask hook) over ALL 5,912 navhard tokens with EXACTLY the arguments of the P1'
export (``raw/chain_P1X.log`` line 1: the banked step-50,400 navhard bridge command + --export-fan), only the output dirs and the label
differ:  --out raw/p4_bridge  --export-fan raw/p4_fan  mask -> raw/p4_mask.  Resumable row by row.  Writes raw/p4_gpu_done.json.
Refuses to run unless the lock names this job AND its pid is an ancestor of this process.  Never touches the lock.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_p1p2_chain as CH  # noqa: E402  (COMMON_X: the P1' arguments, verbatim)

RAW = CH.RAW
JOB = "refcv8-d6-p4"
CH.JOB = JOB                                       # lock_pid() checks the lock against THIS job
PY = CH.PY
TREE = CH.TREE


def main() -> int:
    def log(m):
        with open(os.path.join(RAW, "p4_gpu_chain.log"), "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")
    if not CH.lock_is_ours():
        log("REFUSED: the GPU lock does not name this job / this wrapper")
        return 3
    tf = os.path.join(RAW, "_tokens_P1X_all.json")
    n = len(json.load(open(tf, encoding="utf-8"))["tokens"])
    if n != 5912:
        log(f"REFUSED: token file has {n} tokens, not 5912")
        return 4
    common = [("D6P4" if x == "D6P1P2" else x) for x in CH.COMMON_X]
    out, fan, mask = (os.path.join(RAW, d) for d in ("p4_bridge", "p4_fan", "p4_mask"))
    log(f"START under the lock job={JOB} wrapper_pid={CH.lock_pid()}")
    t0 = time.time()
    tries = 0
    rc = None
    env = dict(os.environ)
    env.update({"PYTHONPATH": f"{TREE}/stack;{TREE}/taniteval", "TANITAD_REPO": TREE, "OMP_NUM_THREADS": "6",
                "PYTHONIOENCODING": "utf-8"})
    logp = os.path.join(RAW, "p4_chain_export.log")
    while True:
        tries += 1
        if CH.rows_done(out, "R7_A1", n):
            log("rows complete -> nothing to run")
            rc = 0
            break
        cmd = [PY, os.path.join(HERE, "d6_p4_export.py"), "--mask-out", mask, "--",
               "--arms", "R7_A1", "--tokens-file", tf, "--out", out,
               "--gpu-lock-job", JOB, "--gpu-lock-pid", str(CH.lock_pid())] + common + ["--export-fan", fan]
        with open(logp, "a", encoding="utf-8") as fh:
            fh.write(f"CMD (try {tries}): " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        if rc == 0 or tries >= 90 or not CH.lock_is_ours():
            break
        tail = open(logp, encoding="utf-8", errors="replace").read()[-3000:]
        if "other python compute app" not in tail:
            break
        log(f"foreign python compute app on the card (try {tries}); waiting 60 s")
        time.sleep(60)
    res = {"stage": "P4_EXPORT", "rc": rc, "tries": tries, "wall_s": round(time.time() - t0, 1),
           "rows_complete": CH.rows_done(out, "R7_A1", n), "n_expected": n}
    json.dump(res, open(os.path.join(RAW, "p4_gpu_done.json"), "w"), indent=1)
    log("DONE " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
