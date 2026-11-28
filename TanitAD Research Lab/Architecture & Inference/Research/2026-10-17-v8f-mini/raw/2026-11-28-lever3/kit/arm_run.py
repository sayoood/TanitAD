#!/usr/bin/env python3
"""arm_run.py TAG -- train ONE LEVER3 arm (D3K_s0 | D3K_s1 | D9K_s0 | D9K_s1) on the dev-box GPU, only through a PASS gate token (card PREREG_V8F_LEVER3_V2).
Refuses (rc 2, nothing started, no lock taken) unless: (1) runs/TAG/argv.json passes control b4 (l3_argv.py verify_one); (2) runs/TAG/gate/gate_gatetok.json exists (a token is never hand-written; it is minted by
`l3_chk.py mint TAG`, i.e. the study-side checker kit/dscale10k_check_l3.py); (3) the checker's `run` (via `l3_chk.py run TAG`) verifies that token against this exact argv, commit, data, tree, trainer, checker,
template and pair, and only then starts the trainer. The checker's own refusal (rc 2) is passed through and the lock is released.
Then: waits for the shared dev-box GPU lock (l3_gpulock.py: JSON lock file created exclusively, honours devbox_gpu.chain_active, never forces another holder), runs `l3_chk.py run` under it, releases ONLY if the lock
still names this job and pid, writes runs/TAG/ARM_DONE (plain write) after rc 0.
STATUS: wired to the checker; the token-refusal path is tested on the dry root (REPORT_GATE.md); a full run has not been started (no token exists before the full cache, bank and join exist)."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l3_argv as A
import l3_gpulock as L
ROOT = os.environ.get("L3_ROOT", "C:/Users/Admin/lever3")
KIT = A.KIT
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CHECKER = KIT + "/dscale10k_check_l3.py"
WAIT_S = int(os.environ.get("L3_LOCK_WAIT_S", "86400"))


def refuse(msg):
    print("ZZ ARM REFUSED ZZ", msg)
    sys.exit(2)


def main():
    tag = sys.argv[1]
    if tag not in A.ARMS:
        refuse(f"unknown tag {tag}")
    D = f"{ROOT}/runs/{tag}"
    p = f"{D}/argv.json"
    if not os.path.isfile(p):
        refuse(f"no {p} (l3_argv.py render first)")
    bad = A.verify_one(tag, json.load(open(p))["argv"], ROOT)
    if bad:
        refuse(f"control b4: {bad}")
    tok = f"{D}/gate/gate_gatetok.json"
    if not os.path.isfile(tok):
        refuse(f"no gate token {tok} (mint: l3_chk.py mint {tag}; never by hand)")
    if not os.path.isfile(CHECKER):
        refuse(f"gate checker {CHECKER} is missing")
    if os.path.isfile(f"{D}/ARM_DONE"):
        print("ARM_DONE exists; skipped")
        return 0
    job = f"l3_{tag}"
    if not L.take(job, WAIT_S):
        refuse(f"GPU lock not obtained within {WAIT_S} s: {L.holder()}")
    rc = 1
    try:
        os.makedirs(f"{D}/train", exist_ok=True)
        e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_XET="1")
        e.pop("CUDA_VISIBLE_DEVICES", None)
        with open(f"{D}/train.log", "a", encoding="utf-8") as fh:
            rc = subprocess.run([PY, KIT + "/l3_chk.py", "run", tag, "--root", ROOT], env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        L.release(job)
    if rc != 0:
        print(f"ZZ ARM {tag} FAIL ZZ rc={rc}" + (" (token refused)" if rc == 2 else ""))
        return rc
    open(f"{D}/ARM_DONE", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")
    print(f"ZZ ARM {tag} DONE ZZ")
    return 0


sys.exit(main())
