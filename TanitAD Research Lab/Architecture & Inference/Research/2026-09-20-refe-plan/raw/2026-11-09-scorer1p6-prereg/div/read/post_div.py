#!/usr/bin/env python3
"""post_div.py (dev box): from the DIV pod chain's end to the DIV read, unattended (post_dose.py's flow, PREREG_DIV_P6).  NOT RUN by its author.  Launch detached (WMI `cmd /c ... < NUL`).

Usage: post_div.py <T0 YYYY-MM-DDTHH:MM:SS> [--chain-log PATH] [--done-pattern TEXT] [--fail-pattern TEXT ...] [--no-ledger] [--no-wait]
  T0 = the pause start: only chain-log lines stamped at or after it count, so dry-run lines are ignored.
  TO CONFIRM against the pod chain (written by another agent in p6base/div/files/): --chain-log default /workspace/data/refe_div/chain_div.log and --done-pattern default ZZDIV_CHAIN_DONE.
1 wait for the DONE line in the pod's chain log (a ZZDIV_CHAIN_FAIL or REFUSED line ends this script FAILED)
2 pull div/dv0/model_sft_B.pt and div/dv1/model_sft_B.pt from the private HF repo Sayood/tanitad-refe-scorer1p6, verify each LFS sha256 (token read in place from Keys.txt, never printed)
3 E3 logit pass on the dev-box GPU: ONE chain_arm_pass.py call with --arm DV0=<pt> --arm DV1=<pt> (it takes several --arm), out F:/tanitad_scratch/p6arms/pass_div.
  chain_arm_pass takes the shared GPU lock itself (waits, never forces): this script never touches the lock file.  The pass resumes from _progress/part_*.npz; a 0-byte
  part_N.npz.tmp.npz left by a crash breaks the resume, so before the pass (and before a retry) such a file is removed ONLY if it is 0 bytes, and the removal is logged.
4 read_div.py read --out F:/tanitad_scratch/divread/real   (members default to the ten dirs of read_div.MEMBER_DIRS; K + 5 in the ledger; add --no-ledger for a dry run)
Markers: F:/tanitad_scratch/p6arms/post_div.DONE or post_div.FAILED (first line = the reason).  Log: F:/tanitad_scratch/p6arms/post_div.log."""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = "F:/tanitad_scratch/p6arms/"
LOG = ROOT + "post_div.log"
P6B = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"          # never a bare "bash" from Python on this box: it resolves to WSL's
REPO = "Sayood/tanitad-refe-scorer1p6"
FNS = {"DV0": "div/dv0/model_sft_B.pt", "DV1": "div/dv1/model_sft_B.pt"}
PASS = ROOT + "pass_div"
READ_PY = P6B + "div/read/read_div.py"
READ_OUT = "F:/tanitad_scratch/divread/real"
TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
CHAIN_LOG_DEFAULT = "/workspace/data/refe_div/chain_div.log"      # TO CONFIRM (the pod chain's log path)
DONE_DEFAULT = "ZZDIV_CHAIN_DONE"                                 # TO CONFIRM (the pod chain's DONE-line text)
FAIL_DEFAULT = ["ZZDIV_CHAIN_FAIL", "REFUSED"]
KEYS = "E:/Projects/TanitAD/Keys.txt"
# read inputs that must exist before the long wait (the ten members minus the two this chain makes, P6, the truth table)
READ_INPUTS = [ROOT + "pass/P6_control/e3_logits.npz", TABLE] + [ROOT + p + "/e3_logits.npz" for p in (
    "pass/C0", "pass/C0r", "pass/S1E", "pass_rep_c0r2_s1e1_s1z1/C0r2", "pass_rep_c0r2_s1e1_s1z1/S1E1", "pass_rep_c0r2_s1e1_s1z1/S1Z1", "pass_s1z/S1Z", "pass_s1zsm/S1ZSM")]


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def mark(ok, msg):
    with open(ROOT + ("post_div.DONE" if ok else "post_div.FAILED"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    say(("DONE " if ok else "FAILED ") + msg)
    sys.exit(0 if ok else 1)


def pod(cmd, t=90):
    r = subprocess.run([BASH, "C:/Users/Admin/par18/pd.sh", str(t), cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    return "\n".join(l for l in r.stdout.splitlines() if "hf_" not in l)


def chain_query(chain_log, t0, done, fails):
    pat = "|".join([re.escape(done)] + [re.escape(x) for x in fails])
    return f"awk 'substr($1,1,19) >= \"{t0}\"' {chain_log} | grep -E '{pat}' | tail -1"


def wait_chain(chain_log, t0, done, fails):
    q = chain_query(chain_log, t0, done, fails)
    start = time.time()
    while time.time() - start < 6 * 3600:                # ESTIMATED about 3 L40-h on the pod (card: one P7 pause)
        out = pod(q).strip()
        if done in out:
            say("pod chain: " + out[:200])
            return
        if out:
            mark(False, "pod chain did not finish: " + out[:300])
        time.sleep(300)
    mark(False, "no chain end within 6 h")


def pull(fn, name):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    tok = re.search(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="replace").read()).group(0)
    info = HfApi(token=tok).get_paths_info(REPO, [fn], repo_type="model")
    if len(info) != 1 or getattr(info[0], "lfs", None) is None:
        mark(False, f"{fn} not found as an LFS file in {REPO}")
    want = info[0].lfs.sha256
    p = hf_hub_download(REPO, fn, repo_type="model", token=tok, local_dir=ROOT + "hf_div")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != want:
        mark(False, f"pulled {fn} sha256 {h.hexdigest()} != HF LFS {want}")
    say(f"{name}: pulled {p} {os.path.getsize(p)} B sha256 {want}")
    return p


def clean_zero_byte_tmp(progress_dir):
    """Lesson of 2026-11-10: a crash can leave a 0-byte part_N.npz.tmp.npz that breaks the resume.  Remove such a file ONLY if it is exactly 0 bytes; log each removal.
    A non-empty tmp file is left alone and logged.  Returns the list of removed paths."""
    removed = []
    for p in sorted(glob.glob(progress_dir.rstrip("/\\") + "/part_*.npz.tmp.npz")):
        sz = os.path.getsize(p)
        if sz == 0:
            os.remove(p)
            removed.append(p)
            say(f"removed 0-byte leftover {p}")
        else:
            say(f"WARNING non-empty tmp file left in place (not removed): {p} {sz} B")
    return removed


def run(args, cwd, name, env_extra=None, timeout=6 * 3600):
    env = dict(os.environ)
    env.update(env_extra or {})
    env.pop("MSYS_NO_PATHCONV", None)
    say(f"step {name}: start")
    with open(ROOT + f"post_div_{name}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    say(f"step {name}: rc {rc}")
    return rc


def armpass_cmd(pts):
    return [PY, P6B + "eval/chain_arm_pass.py", "--out", PASS, "--arm", "DV0=" + pts["DV0"], "--arm", "DV1=" + pts["DV1"]]


def read_cmd(no_ledger=False):
    cmd = [PY, READ_PY, "read", "--out", READ_OUT]
    return cmd + (["--no-ledger"] if no_ledger else [])


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("t0")
    ap.add_argument("--chain-log", default=CHAIN_LOG_DEFAULT)
    ap.add_argument("--done-pattern", default=DONE_DEFAULT)
    ap.add_argument("--fail-pattern", action="append", default=None)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--no-wait", action="store_true", help="skip step 1 (the chain has already ended)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", a.t0):
        ap.error(f"T0 {a.t0!r} is not YYYY-MM-DDTHH:MM:SS (pass the pause start)")
    a.fail_pattern = a.fail_pattern or FAIL_DEFAULT
    return a


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    for m in ("post_div.DONE", "post_div.FAILED"):
        if os.path.exists(ROOT + m):
            print("a terminal marker exists already:", m)
            return 1
    for p in READ_INPUTS + [READ_PY]:
        if not os.path.isfile(p):
            mark(False, f"a read input is missing before the wait: {p}")
    say(f"start (T0 {a.t0}, chain log {a.chain_log}, done pattern {a.done_pattern})")
    if not a.no_wait:
        wait_chain(a.chain_log, a.t0, a.done_pattern, a.fail_pattern)
    pts = {n: pull(fn, n) for n, fn in FNS.items()}
    clean_zero_byte_tmp(PASS + "/_progress")
    ok = lambda: all(os.path.isfile(f"{PASS}/{n}/e3_logits.npz") for n in FNS)      # noqa: E731
    rc = run(armpass_cmd(pts), P6B + "eval", "armpass", timeout=8 * 3600)
    if rc != 0 or not ok():                                  # one resumable retry, after clearing a crash leftover
        say("arm pass did not complete; clearing 0-byte leftovers and retrying once (resumes from _progress)")
        clean_zero_byte_tmp(PASS + "/_progress")
        rc = run(armpass_cmd(pts), P6B + "eval", "armpass", timeout=8 * 3600)
    if rc != 0 or not ok():
        mark(False, f"E3 logit pass failed (see {ROOT}post_div_armpass.log)")
    rc = run(read_cmd(a.no_ledger), os.path.dirname(READ_PY), "read", {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, 3600)
    if rc != 0:
        mark(False, f"read_div rc {rc} (see post_div_read.log; rc 2 = card or gate failed, no verdict)")
    mark(True, f"DV0 and DV1 E3 logits in {PASS}; DIV read written to {READ_OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark(False, "exception: " + traceback.format_exc()[-600:])
