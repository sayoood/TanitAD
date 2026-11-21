#!/usr/bin/env python3
"""post_s1zsm.py (dev box): from S1ZSM's pod chain end to its navtest verdict, unattended (post_s1z.py's flow, PREREG_S1ZSM_P6).  Launch detached.
Usage: post_s1zsm.py <T0 YYYY-MM-DDTHH:MM:SS>   (the pause start: only chain-log lines stamped at or after it count, so the dry-run lines are ignored)
1 wait for ZZS1ZSM_CHAIN_DONE in the pod's /workspace/data/refe_s1zsm/chain.log (a REFUSED / ZZS1ZSM_CHAIN_FAIL line ends this script FAILED)
2 pull s1zsm/model_sft_B.pt from the private HF repo, verify the LFS sha256 (token read in place from Keys.txt, never printed)
3 E3 logit pass on the dev-box GPU (chain_arm_pass.py takes the shared GPU lock: waits, never forces)
4 the changed picks' scores by table lookup (changed_from_table.py; its end-to-end control on Batch 1 is 0 differences)
5 read_s1zsm.py read (the frozen card PREREG_S1ZSM_P6.md, sha256 ac75b289...), against S1Z's pass and Batch 1's C0 / C0r; with PREREG_REP_P6's read if it exists (reported, not deciding)
Markers: F:/tanitad_scratch/p6arms/post_s1zsm.DONE or post_s1zsm.FAILED (first line = the reason). Log: F:/tanitad_scratch/p6arms/post_s1zsm.log."""
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = "F:/tanitad_scratch/p6arms/"
LOG = ROOT + "post_s1zsm.log"
P6B = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"          # never a bare "bash" from Python on this box: it resolves to WSL's
T0 = sys.argv[1] if len(sys.argv) > 1 else ""
if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", T0):
    sys.exit(f"T0 {T0!r} is not YYYY-MM-DDTHH:MM:SS (pass the pause start)")
REPO, FN = "Sayood/tanitad-refe-scorer1p6", "s1zsm/model_sft_B.pt"
PASS = ROOT + "pass_s1zsm"
S1Z_PASS = ROOT + "pass_s1z/S1Z"
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
REP_READ = "F:/tanitad_scratch/repread_real/rep_read.json"
READ_OUT = "F:/tanitad_scratch/s1zsmread/real"


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def mark(ok, msg):
    with open(ROOT + ("post_s1zsm.DONE" if ok else "post_s1zsm.FAILED"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    say(("DONE " if ok else "FAILED ") + msg)
    sys.exit(0 if ok else 1)


def pod(cmd, t=90):
    r = subprocess.run([BASH, "C:/Users/Admin/par18/pd.sh", str(t), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    return "\n".join(l for l in r.stdout.splitlines() if "hf_" not in l)


def wait_chain():
    q = f"awk 'substr($1,1,19) >= \"{T0}\"' /workspace/data/refe_s1zsm/chain.log | grep -E 'ZZS1ZSM_CHAIN_DONE|ZZS1ZSM_CHAIN_FAIL|REFUSED' | tail -1"
    t0 = time.time()
    while time.time() - t0 < 9 * 3600:
        out = pod(q).strip()
        if "ZZS1ZSM_CHAIN_DONE" in out:
            say("pod chain: " + out[:200]); return
        if out:
            mark(False, "pod chain did not finish: " + out[:300])
        time.sleep(300)
    mark(False, "no chain end within 9 h")


def pull():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
    info = HfApi(token=tok).get_paths_info(REPO, [FN], repo_type="model")
    if len(info) != 1 or getattr(info[0], "lfs", None) is None:
        mark(False, f"{FN} not found as an LFS file in {REPO}")
    want = info[0].lfs.sha256
    p = hf_hub_download(REPO, FN, repo_type="model", token=tok, local_dir=ROOT + "hf")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != want:
        mark(False, f"pulled {FN} sha256 {h.hexdigest()} != HF LFS {want}")
    say(f"pulled {p} {os.path.getsize(p)} B sha256 {want}")
    return p


def run(args, cwd, name, env_extra=None, timeout=6 * 3600):
    env = dict(os.environ); env.update(env_extra or {}); env.pop("MSYS_NO_PATHCONV", None)
    say(f"step {name}: start")
    with open(ROOT + f"post_s1zsm_{name}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    say(f"step {name}: rc {rc}")
    return rc


def main():
    for m in ("post_s1zsm.DONE", "post_s1zsm.FAILED"):
        if os.path.exists(ROOT + m):
            print("a terminal marker exists already:", m); return 1
    for p in (ROOT + "pass/C0/e3_logits.npz", ROOT + "pass/C0r/e3_logits.npz", S1Z_PASS + "/e3_logits.npz", TABLE):
        if not os.path.isfile(p):
            mark(False, f"a read input is missing before the wait: {p}")
    say(f"start (T0 {T0})")
    wait_chain()
    pt = pull()
    if run([PY, P6B + "eval/chain_arm_pass.py", "--out", PASS, "--arm", "S1ZSM=" + pt], P6B + "eval", "armpass", timeout=8 * 3600) != 0 or not os.path.isfile(PASS + "/S1ZSM/e3_logits.npz"):
        mark(False, f"E3 logit pass failed (see {ROOT}post_s1zsm_armpass.log)")
    if run([PY, P6B + "eval/changed_from_table.py", "--table", TABLE, "--arm-logits", PASS + "/S1ZSM/e3_logits.npz", "--out", PASS + "/S1ZSM/e3_changed_scores.csv"],
           P6B + "eval", "lookup", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 1800) != 0:
        mark(False, "table lookup failed")
    args = [PY, P6B + "s1zsm/read/read_s1zsm.py", "read", "--c0", ROOT + "pass/C0", "--c0r", ROOT + "pass/C0r", "--s1z", S1Z_PASS, "--s1zsm", PASS + "/S1ZSM", "--out", READ_OUT]
    if os.path.isfile(REP_READ):
        args[-2:-2] = ["--rep-read", REP_READ]
    rc = run(args, P6B + "s1zsm/read", "read", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 3600)
    if rc != 0:
        mark(False, f"read_s1zsm rc {rc} (2 = a gate failed or INCOMPLETE; see post_s1zsm_read.log)")
    mark(True, f"S1ZSM read written to {READ_OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark(False, "exception: " + traceback.format_exc()[-600:])
