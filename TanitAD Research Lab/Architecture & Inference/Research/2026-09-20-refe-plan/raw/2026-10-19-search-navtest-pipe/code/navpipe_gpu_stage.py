#!/usr/bin/env python3
"""The GPU phase of ONE navtest chunk.  Run ONLY through `launch_stepA_ctype.py run -- <driverl-eval python> navpipe_gpu_stage.py ...` (the dev-box GPU lock: it waits for `chain_active`, >= 8 GiB RAM and a free
GPU, and never touches another owner's lock).  NOT RUN by the package.

  --bridge-sample S   (first chunk only)  navpipe_bridge.py on 3 navtest tokens: IDENTITY + the one-pixel RED arm.  ZZBRIDGE_PASS | ZZBRIDGE_FAIL.
  --chunk-sample  C   make_navtest_bundle.py on the chunk's tokens -> <out>/bundle (ZZBUNDLE_OK n)
One lock hold per chunk.  Stops at the first failing step (exit 3).  `--dry-run` prints the plan and checks that every input exists (CPU, no model).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
BANKED_SEAM = "D:/Projects/TanitAD/data/refe_navtest/seams/stage2_p2_full.npz"


def ref_args(a) -> list:
    return ["--reference-kind", getattr(a, "reference_kind", "seam") or "seam", "--reference-path", getattr(a, "reference_path", None) or a.banked_seam]


def plan(a, py=None):
    py = py or sys.executable
    steps = []
    if a.bridge_sample:
        steps.append(("BRIDGE", [py, "-u", os.path.join(HERE, "navpipe_bridge.py"), "--ckpt", a.ckpt, "--plan", a.plan, "--sample", a.bridge_sample, *ref_args(a),
                                 "--out-dir", os.path.join(a.out_dir, "bridge_run")], r"ZZBRIDGE_PASS"))
    n = a.n_expected
    steps.append(("BUNDLE", [py, "-u", os.path.join(HERE, "make_navtest_bundle.py"), "--ckpt", a.ckpt, "--plan", a.plan, "--sample", a.chunk_sample, *ref_args(a),
                             "--out-dir", os.path.join(a.out_dir, "bundle")], rf"ZZBUNDLE_OK {n}(\s|$)"))
    return steps


def run(steps, runner=None, out=sys.stdout) -> int:
    runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, text=True, errors="replace"))
    for name, argv, ok_re in steps:
        r = runner(argv)
        text = (r.stdout or "") + (r.stderr or "")
        for line in text.splitlines():
            if line.startswith("ZZ"):
                print(line, file=out, flush=True)
        if r.returncode != 0 or not re.search(ok_re, text):                                 # @GUARD:GS-STEP
            print(f"ZZSTAGE_FAIL {name} rc={r.returncode} marker_found={bool(re.search(ok_re, text))}", file=out, flush=True)
            return 3
        print(f"ZZSTAGE_{name}_OK", file=out, flush=True)
    print("ZZSTAGE_DONE", file=out, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="<out>/bundle and <out>/bridge_run are created")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--chunk-sample", required=True)
    ap.add_argument("--n-expected", type=int, required=True)
    ap.add_argument("--bridge-sample", default=None)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--banked-seam", default=BANKED_SEAM)
    ap.add_argument("--reference-kind", default="seam", choices=("seam", "table", "s0fan"))
    ap.add_argument("--reference-path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    need = [a.plan, a.chunk_sample, a.ckpt, a.reference_path or a.banked_seam] + ([a.bridge_sample] if a.bridge_sample else [])
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        print("ZZSTAGE_FAIL missing inputs", miss, flush=True)
        return 3
    steps = plan(a)
    if a.dry_run:
        for n, argv_, _ in steps:
            print("PLAN", n, " ".join(argv_))
        print("ZZSTAGE_DRYRUN_OK")
        return 0
    return run(steps)


if __name__ == "__main__":
    sys.exit(main())
