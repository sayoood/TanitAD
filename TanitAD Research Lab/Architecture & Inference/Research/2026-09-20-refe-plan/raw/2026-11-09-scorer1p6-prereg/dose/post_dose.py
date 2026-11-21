#!/usr/bin/env python3
"""post_dose.py (dev box): from S1ZD's pod chain end to its navtest read, unattended (post_s1zsm.py's flow, PREREG_DOSE_P6).  NOT RUN by its author.  Launch detached (WMI `cmd /c ... < NUL`).
Usage: post_dose.py <T0 YYYY-MM-DDTHH:MM:SS>   (the pause start: only chain-log lines stamped at or after it count, so dry-run lines are ignored)
1 wait for `ZZDOSE_CHAIN_DONE S1ZD` in the pod's /workspace/data/refe_dose/chain_s1zd.log (a REFUSED / ZZDOSE_CHAIN_FAIL line ends this script FAILED)
2 pull dose/s1zd/model_sft_B.pt from the private HF repo, verify the LFS sha256 (token read in place from Keys.txt, never printed); also pull dose/s1zd/sft.jsonl (the chain's egress list uploads it, dose_chain.sh:264) for read_dose's --sft-log
3 E3 logit pass on the dev-box GPU (chain_arm_pass.py takes the shared GPU lock: waits, never forces) into F:/tanitad_scratch/p6arms/pass_dose, arm S1ZD
4 the changed picks' scores by table lookup (changed_from_table.py; table_p6final_c.npz)
5 p6base/dose/read/read_dose.py read --p6-pass <dir holding P6_control/> --c0 <dir> --c0r <dir> --s1z <dir> --s1z1 <dir> --s1zd <dir> [--sft-log <sft.jsonl>] --out <dir>  (CLI per read/HANDOFF.md and read_dose.py:427-449, checked
  with `read_dose.py read --help`), ONLY IF that file exists.  If it does not exist the script ends DONE with "read step skipped" and the Master Mind runs the read by hand.
  Amendment A1 (PREREG_DOSE_P6_A1.md, 1aa0e702): S1ZD's argv carries --epochs 2.0; nothing in this script changes for it.
Markers: F:/tanitad_scratch/p6arms/post_dose.DONE or post_dose.FAILED (first line = the reason).  Log: F:/tanitad_scratch/p6arms/post_dose.log."""
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = "F:/tanitad_scratch/p6arms/"
LOG = ROOT + "post_dose.log"
P6B = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"          # never a bare "bash" from Python on this box: it resolves to WSL's
T0 = sys.argv[1] if len(sys.argv) > 1 else ""
if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", T0):
    sys.exit(f"T0 {T0!r} is not YYYY-MM-DDTHH:MM:SS (pass the pause start)")
REPO, FN = "Sayood/tanitad-refe-scorer1p6", "dose/s1zd/model_sft_B.pt"
PASS = ROOT + "pass_dose"
P6_PASS = ROOT + "pass_rep_c0r2_s1e1_s1z1"          # --p6-pass is the FOLDER that holds P6_control/e3_logits.npz (read_dose.py:238-239, :471)
C0_PASS = ROOT + "pass/C0"
C0R_PASS = ROOT + "pass/C0r"
S1Z_PASS = ROOT + "pass_s1z/S1Z"
S1Z1_PASS = ROOT + "pass_rep_c0r2_s1e1_s1z1/S1Z1"
FN_SFT = "dose/s1zd/sft.jsonl"
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
READ_PY = P6B + "dose/read/read_dose.py"
READ_OUT = "F:/tanitad_scratch/doseread/real"
CHAIN_LOG = "/workspace/data/refe_dose/chain_s1zd.log"


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def mark(ok, msg):
    with open(ROOT + ("post_dose.DONE" if ok else "post_dose.FAILED"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    say(("DONE " if ok else "FAILED ") + msg)
    sys.exit(0 if ok else 1)


def pod(cmd, t=90):
    r = subprocess.run([BASH, "C:/Users/Admin/par18/pd.sh", str(t), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    return "\n".join(l for l in r.stdout.splitlines() if "hf_" not in l)


def wait_chain():
    q = f"awk 'substr($1,1,19) >= \"{T0}\"' {CHAIN_LOG} | grep -E 'ZZDOSE_CHAIN_DONE|ZZDOSE_CHAIN_FAIL|REFUSED' | tail -1"
    t0 = time.time()
    while time.time() - t0 < 10 * 3600:                  # ESTIMATED 4.1 L40-h on the pod
        out = pod(q).strip()
        if "ZZDOSE_CHAIN_DONE" in out:
            say("pod chain: " + out[:200])
            return
        if out:
            mark(False, "pod chain did not finish: " + out[:300])
        time.sleep(300)
    mark(False, "no chain end within 10 h")


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


def pull_sft():
    """Best effort: the chain uploads sft.jsonl with the arm (dose_chain.sh egress list).  Returns the local path or None (the read then runs without --sft-log)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="replace").read()).group(0)
        info = HfApi(token=tok).get_paths_info(REPO, [FN_SFT], repo_type="model")
        if len(info) != 1:
            say(f"{FN_SFT} not found in {REPO}: the read runs without --sft-log")
            return None
        p = hf_hub_download(REPO, FN_SFT, repo_type="model", token=tok, local_dir=ROOT + "hf")
        say(f"pulled {p} {os.path.getsize(p)} B")
        return p
    except Exception as ex:                                  # never print the token: only the exception class and a short message without it
        say(f"sft.jsonl pull failed ({type(ex).__name__}): the read runs without --sft-log")
        return None


def run(args, cwd, name, env_extra=None, timeout=6 * 3600):
    env = dict(os.environ)
    env.update(env_extra or {})
    env.pop("MSYS_NO_PATHCONV", None)
    say(f"step {name}: start")
    with open(ROOT + f"post_dose_{name}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    say(f"step {name}: rc {rc}")
    return rc


def read_cmd(sft=None):
    """The read_dose.py command line (CLI: read_dose.py:427-449): every required flag, --sft-log only when the log was pulled."""
    cmd = [PY, READ_PY, "read", "--p6-pass", P6_PASS, "--c0", C0_PASS, "--c0r", C0R_PASS, "--s1z", S1Z_PASS, "--s1z1", S1Z1_PASS, "--s1zd", PASS + "/S1ZD"]
    if sft:
        cmd += ["--sft-log", sft]
    return cmd + ["--out", READ_OUT]


def main():
    for m in ("post_dose.DONE", "post_dose.FAILED"):
        if os.path.exists(ROOT + m):
            print("a terminal marker exists already:", m)
            return 1
    for p in (P6_PASS + "/P6_control/e3_logits.npz", C0_PASS + "/e3_logits.npz", C0R_PASS + "/e3_logits.npz", S1Z_PASS + "/e3_logits.npz", S1Z1_PASS + "/e3_logits.npz", TABLE):
        if not os.path.isfile(p):
            mark(False, f"a read input is missing before the wait: {p}")
    say(f"start (T0 {T0})")
    wait_chain()
    pt = pull()
    sft = pull_sft()
    if run([PY, P6B + "eval/chain_arm_pass.py", "--out", PASS, "--arm", "S1ZD=" + pt], P6B + "eval", "armpass", timeout=8 * 3600) != 0 or not os.path.isfile(PASS + "/S1ZD/e3_logits.npz"):
        mark(False, f"E3 logit pass failed (see {ROOT}post_dose_armpass.log)")
    if run([PY, P6B + "eval/changed_from_table.py", "--table", TABLE, "--arm-logits", PASS + "/S1ZD/e3_logits.npz", "--out", PASS + "/S1ZD/e3_changed_scores.csv"],
           P6B + "eval", "lookup", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 1800) != 0:
        mark(False, "table lookup failed")
    if not os.path.isfile(READ_PY):                       # the read script is another agent's; do not invent its CLI
        mark(True, f"S1ZD E3 logits and changed-pick scores in {PASS}/S1ZD; read step SKIPPED: {READ_PY} does not exist (run it by hand: read --p6-pass {P6_PASS} --c0 {C0_PASS} --c0r {C0R_PASS} --s1z {S1Z_PASS} --s1z1 {S1Z1_PASS} --s1zd {PASS}/S1ZD --out {READ_OUT})")
    rc = run(read_cmd(sft),
             os.path.dirname(READ_PY), "read", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 3600)
    if rc != 0:
        mark(False, f"read_dose rc {rc} (see post_dose_read.log)")
    mark(True, f"S1ZD read written to {READ_OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark(False, "exception: " + traceback.format_exc()[-600:])
