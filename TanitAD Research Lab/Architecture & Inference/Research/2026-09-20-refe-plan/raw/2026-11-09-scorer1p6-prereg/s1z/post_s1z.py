#!/usr/bin/env python3
"""post_s1z.py (dev box): from S1Z's pod chain end to its navtest verdict, unattended. Launch detached (launch_post_s1z.py).
1 wait for ZZS1Z_CHAIN_DONE in the pod's chain.log, stamped at or after run 2's start (a REFUSED / ZZS1Z_CHAIN_FAIL line there ends this script FAILED)
2 pull s1z/model_sft_B.pt from the private HF repo, verify the LFS sha256 (token read in place from Keys.txt, never printed)
3 E3 logit pass on the dev-box GPU (chain_arm_pass.py takes the shared GPU lock: waits, never forces)
4 the changed picks' scores by table lookup (changed_from_table.py; its end-to-end control on Batch 1 is 0 differences)
5 read_s1z.py read (the frozen card PREREG_S1Z_P6.md, sha256 8c7b6b69...)
Markers: F:/tanitad_scratch/p6arms/post_s1z.DONE or post_s1z.FAILED (first line = the reason). Log: F:/tanitad_scratch/p6arms/post_s1z.log."""
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = "F:/tanitad_scratch/p6arms/"
LOG = ROOT + "post_s1z.log"
P6B = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"          # never a bare "bash" from Python on this box: it resolves to WSL's
T0 = "2026-11-10T02:38:47"
REPO, FN = "Sayood/tanitad-refe-scorer1p6", "s1z/model_sft_B.pt"
PASS = ROOT + "pass_s1z"
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
READ_OUT = "F:/tanitad_scratch/s1zread/real"


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def mark(ok, msg):
    with open(ROOT + ("post_s1z.DONE" if ok else "post_s1z.FAILED"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    say(("DONE " if ok else "FAILED ") + msg)
    sys.exit(0 if ok else 1)


def pod(cmd, t=90):
    r = subprocess.run([BASH, "C:/Users/Admin/par18/pd.sh", str(t), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    return "\n".join(l for l in r.stdout.splitlines() if "hf_" not in l)


def wait_chain():
    q = f"awk 'substr($1,1,19) >= \"{T0}\"' /workspace/data/refe_s1z/chain.log | grep -E 'ZZS1Z_CHAIN_DONE|ZZS1Z_CHAIN_FAIL|REFUSED' | tail -1"
    t0 = time.time()
    while time.time() - t0 < 8 * 3600:
        out = pod(q).strip()
        if "ZZS1Z_CHAIN_DONE" in out:
            say("pod chain: " + out[:200]); return
        if out:
            mark(False, "pod chain did not finish: " + out[:300])
        time.sleep(300)
    mark(False, "no chain end within 8 h")


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
    with open(ROOT + f"post_s1z_{name}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    say(f"step {name}: rc {rc}")
    return rc


def main():
    for m in ("post_s1z.DONE", "post_s1z.FAILED"):
        if os.path.exists(ROOT + m):
            print("a terminal marker exists already:", m); return 1
    say("start")
    wait_chain()
    pt = pull()
    if run([PY, P6B + "eval/chain_arm_pass.py", "--out", PASS, "--arm", "S1Z=" + pt], P6B + "eval", "armpass") != 0 or not os.path.isfile(PASS + "/S1Z/e3_logits.npz"):
        mark(False, f"E3 logit pass failed (see {ROOT}post_s1z_armpass.log)")
    if run([PY, P6B + "eval/changed_from_table.py", "--table", TABLE, "--arm-logits", PASS + "/S1Z/e3_logits.npz", "--out", PASS + "/S1Z/e3_changed_scores.csv"],
           P6B + "eval", "lookup", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 1800) != 0:
        mark(False, "table lookup failed")
    rc = run([PY, P6B + "s1z/read/read_s1z.py", "read", "--c0", ROOT + "pass/C0", "--c0r", ROOT + "pass/C0r", "--s1z", PASS + "/S1Z", "--out", READ_OUT],
             P6B + "s1z/read", "read", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 3600)
    if rc != 0:
        mark(False, f"read_s1z rc {rc} (2 = a gate failed or INCOMPLETE; see post_s1z_read.log)")
    mark(True, f"S1Z read written to {READ_OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark(False, "exception: " + traceback.format_exc()[-600:])
