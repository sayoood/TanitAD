#!/usr/bin/env python3
"""post_rep.py (dev box): from the replicate pauses' pod chains (C0r2, S1E1 and, when S1Z reads Z1/Z5, S1Z1; PREREG_REP_P6.md) to the arms' E3 logits and changed-pick scores, unattended.
Usage: post_rep.py <T0 YYYY-MM-DDTHH:MM:SS> [ARM ...]   (default arms C0r2 S1E1 -> pass_rep; any other set -> pass_rep_<arms>).
Launch detached. Steps:
1 wait for "ZZREP_CHAIN_DONE <arm>" for every arm in the pod's chain_<arm>.log, stamped at or after the pause start (a REFUSED / ZZREP_CHAIN_FAIL line ends this script FAILED)
2 pull rep/<arm lower>/model_sft_B.pt from the private HF repo, verify the LFS sha256 (token read in place from Keys.txt, never printed)
3 ONE E3 logit pass for all arms on the dev-box GPU (chain_arm_pass.py takes the shared GPU lock: waits, never forces)
4 each arm's changed picks scored by table lookup (changed_from_table.py; its end-to-end control on Batch 1 is 0 differences)
The read itself (read_rep.py, PREREG_REP_P6) is run after this script by the Master Mind.
Markers: F:/tanitad_scratch/p6arms/post_rep.DONE or post_rep.FAILED (first line = the reason). Log: F:/tanitad_scratch/p6arms/post_rep.log."""
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = "F:/tanitad_scratch/p6arms/"
LOG = ROOT + "post_rep.log"
P6B = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"          # never a bare "bash" from Python on this box: it resolves to WSL's
T0 = sys.argv[1] if len(sys.argv) > 1 else "2026-11-10T04:39:45"     # only chain-log lines at or after the pause start count; pass the start of the pause being read
if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", T0):
    sys.exit(f"T0 {T0!r} is not YYYY-MM-DDTHH:MM:SS")
ARMS = tuple(sys.argv[2:]) or ("C0r2", "S1E1")      # usage: post_rep.py <T0> [ARM ...]; one E3 pass serves every arm given (its cost does not grow with the arm count)
if len(set(ARMS)) != len(ARMS) or not set(ARMS) <= {"C0r2", "S1E1", "S1Z1"}:
    sys.exit(f"arms {ARMS} must be distinct and among C0r2, S1E1, S1Z1")
REPO = "Sayood/tanitad-refe-scorer1p6"
PASS = ROOT + "pass_rep" + ("" if ARMS == ("C0r2", "S1E1") else "_" + "_".join(a.lower() for a in ARMS))
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def mark(ok, msg):
    with open(ROOT + ("post_rep.DONE" if ok else "post_rep.FAILED"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    say(("DONE " if ok else "FAILED ") + msg)
    sys.exit(0 if ok else 1)


def pod(cmd, t=90):
    r = subprocess.run([BASH, "C:/Users/Admin/par18/pd.sh", str(t), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    return "\n".join(l for l in r.stdout.splitlines() if "hf_" not in l)


def wait_chains():
    t0 = time.time()
    while time.time() - t0 < 8 * 3600:
        done = []
        for a in ARMS:
            q = (f"awk 'substr($1,1,19) >= \"{T0}\"' /workspace/data/refe_rep/chain_{a.lower()}.log 2>/dev/null"
                 f" | grep -E 'ZZREP_CHAIN_DONE {a}|ZZREP_CHAIN_FAIL|REFUSED' | tail -1")
            out = pod(q).strip()
            if f"ZZREP_CHAIN_DONE {a}" in out:
                done.append(a)
            elif out:
                mark(False, f"pod chain {a} did not finish: " + out[:300])
        if len(done) == len(ARMS):
            say("pod chains done: " + " ".join(done)); return
        time.sleep(300)
    mark(False, "no chain end within 8 h")


def pull(arm):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    fn = f"rep/{arm.lower()}/model_sft_B.pt"
    tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
    info = HfApi(token=tok).get_paths_info(REPO, [fn], repo_type="model")
    if len(info) != 1 or getattr(info[0], "lfs", None) is None:
        mark(False, f"{fn} not found as an LFS file in {REPO}")
    want = info[0].lfs.sha256
    p = hf_hub_download(REPO, fn, repo_type="model", token=tok, local_dir=ROOT + "hf")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != want:
        mark(False, f"pulled {fn} sha256 {h.hexdigest()} != HF LFS {want}")
    say(f"pulled {p} {os.path.getsize(p)} B sha256 {want}")
    return p


def run(args, cwd, name, env_extra=None, timeout=6 * 3600):
    env = dict(os.environ); env.update(env_extra or {}); env.pop("MSYS_NO_PATHCONV", None)
    say(f"step {name}: start")
    with open(ROOT + f"post_rep_{name}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    say(f"step {name}: rc {rc}")
    return rc


def main():
    for m in ("post_rep.DONE", "post_rep.FAILED"):
        if os.path.exists(ROOT + m):
            print("a terminal marker exists already:", m); return 1
    say("start")
    wait_chains()
    pts = {a: pull(a) for a in ARMS}
    armargs = sum([["--arm", f"{a}={pts[a]}"] for a in ARMS], [])
    if run([PY, P6B + "eval/chain_arm_pass.py", "--out", PASS, *armargs], P6B + "eval", "armpass", timeout=8 * 3600) != 0 \
            or not all(os.path.isfile(f"{PASS}/{a}/e3_logits.npz") for a in ARMS):
        mark(False, f"E3 logit pass failed (see {ROOT}post_rep_armpass.log)")
    for a in ARMS:
        if run([PY, P6B + "eval/changed_from_table.py", "--table", TABLE, "--arm-logits", f"{PASS}/{a}/e3_logits.npz", "--out", f"{PASS}/{a}/e3_changed_scores.csv"],
               P6B + "eval", f"lookup_{a}", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 1800) != 0:
            mark(False, f"table lookup failed for {a}")
    mark(True, f"E3 logits and changed-pick scores for {' '.join(ARMS)} in {PASS}; next: read_rep.py (PREREG_REP_P6)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark(False, "exception: " + traceback.format_exc()[-600:])
