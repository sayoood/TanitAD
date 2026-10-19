#!/usr/bin/env python3
"""The ONE GPU phase of SEARCH-P2-CONFIRM.  Run ONLY through `launch_stepA_ctype.py run -- <driverl-eval python> spc_gpu_stage.py ...` (the dev-box GPU lock: it waits for `chain_active`, >= 8 GiB RAM and a free GPU,
and never touches another owner's lock).  Three steps in ONE lock hold, each the certified L1 / L2-FRESH script ITSELF (byte-identical, sha256-checked here):
  1. BRIDGE  `bridge_g2b.py` on the 3 banked B tokens: IDENTITY (the bundle forward reproduces the banked fan bit for bit) and the one-pixel RED arm (the gate must fire).  Markers ZZBRIDGE_PASS | ZZBRIDGE_FAIL.
  2. CID     `make_stepA_bundle.py` on the 16 carve-B control tokens -> <W>/cid/bundle  (ZZBUNDLE_OK 16)
  3. C1      `make_stepA_bundle.py` on B-REST's 905 tokens           -> <W>/B_REST/bundle (ZZBUNDLE_OK 905)
Stops at the first failing step (exit 3).  Prints ZZSTAGE_<STEP>_OK / ZZSTAGE_FAIL <step> and ZZSTAGE_DONE.  `--dry-run` prints the plan and checks that every input exists (CPU, no model).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
FRESH_CODE = os.environ.get("SPC_FRESH_CODE", "C:/Users/Admin/par18/l2fresh/code")
BUNDLE_SCRIPT_SHA256_LF = "8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237"       # the script the B chain ran and the bridge certified
BRIDGE_SAMPLE = "C:/Users/Admin/par18/l2fresh/raw/bridge3_sample.json"
TABLE = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def plan(work: str, cid_sample: str, brest_sample: str, py: str = None, bundle_script: str = None, bridge_script: str = None, ckpt: str = CKPT, plan_json: str = PLAN, table: str = TABLE):
    py = py or sys.executable
    bundle_script = bundle_script or os.path.join(L1_CODE, "make_stepA_bundle.py")
    bridge_script = bridge_script or os.path.join(FRESH_CODE, "bridge_g2b.py")
    mk = lambda sample, out: [py, "-u", bundle_script, "--ckpt", ckpt, "--plan", plan_json, "--sample", sample, "--table", table, "--out-dir", out]   # noqa: E731
    return [("BRIDGE", [py, "-u", bridge_script, "--ckpt", ckpt, "--plan", plan_json, "--sample", BRIDGE_SAMPLE, "--table", table, "--out-dir", os.path.join(work, "bridge_run"), "--bundle-script", bundle_script],
             r"ZZBRIDGE_PASS"),
            ("CID", mk(cid_sample, os.path.join(work, "cid", "bundle")), r"ZZBUNDLE_OK 16(\s|$)"),
            ("C1", mk(brest_sample, os.path.join(work, "B_REST", "bundle")), r"ZZBUNDLE_OK 905(\s|$)")]


def run(steps, runner=None, out=sys.stdout) -> int:
    import re
    runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, text=True, errors="replace"))
    for name, argv, ok_re in steps:
        r = runner(argv)
        text = (r.stdout or "") + (r.stderr or "")
        for line in text.splitlines():
            if line.startswith("ZZ"):
                print(line, file=out, flush=True)
        if r.returncode != 0 or not re.search(ok_re, text):
            print(f"ZZSTAGE_FAIL {name} rc={r.returncode} marker_found={bool(re.search(ok_re, text))}", file=out, flush=True)
            return 3
        print(f"ZZSTAGE_{name}_OK", file=out, flush=True)
    print("ZZSTAGE_DONE", file=out, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="run dir; bundles land in <work>/cid/bundle and <work>/B_REST/bundle")
    ap.add_argument("--cid-sample", required=True)
    ap.add_argument("--brest-sample", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    steps = plan(a.work, a.cid_sample, a.brest_sample)
    bs = steps[1][1][2]                                       # [python, -u, <bundle script>, ...]
    need = [bs, steps[0][1][2], a.cid_sample, a.brest_sample, BRIDGE_SAMPLE, TABLE, PLAN, CKPT]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        print("ZZSTAGE_FAIL missing inputs", miss, flush=True)
        return 3
    got = sha_lf(bs)
    if got != BUNDLE_SCRIPT_SHA256_LF:
        print(f"ZZSTAGE_FAIL the bundle script is not the certified one: {got[:12]} != {BUNDLE_SCRIPT_SHA256_LF[:12]}", flush=True)
        return 3
    if a.dry_run:
        for n, argv_, _ in steps:
            print("PLAN", n, " ".join(argv_))
        print("ZZSTAGE_DRYRUN_OK")
        return 0
    return run(steps)


if __name__ == "__main__":
    sys.exit(main())
