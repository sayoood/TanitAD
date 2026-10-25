"""Stage the REAL banked B side of PREREG_V8F_FLAT (M2b s0/s1) exactly as chain_v8f_flat.sh stages it on box 2, but from the tip 18c62306 of the mirror (git show, raw bytes),
into <dest>/{eval,runs}; prints the sha256 of every staged file. Nothing is written into the repo tree.   usage: stage_real_b.py <dest>

FLAT needs fewer B files than MIXO: the standard eval, the absmask eval, the GT-code oracle (+ the all-goal oracle that carries its GT block) and the all-latcons oracle; no dose chain."""
import hashlib
import os
import subprocess
import sys

DEST = os.path.abspath(sys.argv[1])
TIP, MIRROR = "18c623068626d5f9f57d5aeb09d2fddb6b89e1fb", "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
B1 = "TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"


def show(path):
    return subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True, check=True).stdout


def put(path_in_repo, dest_rel):
    b = show(path_in_repo)
    out = os.path.join(DEST, dest_rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b)
    print(f"{hashlib.sha256(b).hexdigest()[:16]}  {len(b):>10}  {dest_rel}  <-  {path_in_repo.replace(PK + '/', '')}")


for s in (0, 1):
    for f in ("summary.json", "argv.json", "selfcheck.json", "train_log.jsonl"):
        put(f"{PK}/raw/full/runs/M2b_s{s}/{f}", f"runs/M2b_s{s}/{f}")
    for e in ("npz", "json"):
        put(f"{PK}/raw/full/eval/eval_M2b_s{s}.{e}", f"eval/eval_M2b_s{s}.{e}")
        put(f"{PK}/raw/2026-10-22-cz/eval/eval_M2b_s{s}-tacabs.{e}", f"eval/eval_M2b_s{s}-tacabs.{e}")
        put(f"{PK}/raw/2026-10-20-tac/eval/eval_M2b_s{s}-tacorc.{e}", f"eval/eval_M2b_s{s}-tacorc.{e}")
        put(f"{PK}/raw/2026-10-21-tacgrp2/eval/eval_M2b_s{s}-tacorc-all-latcons.{e}", f"eval/eval_M2b_s{s}-tacorc-all-latcons.{e}")
    put(f"{PK}/raw/2026-10-21-tacgrp2/eval/eval_M2b_s{s}-tacorc-all-goal.npz", f"eval/eval_M2b_s{s}-tacorc-all-goal.npz")
put(f"{PK}/raw/full/eval/floors_perwindow.npz", "eval/floors_perwindow.npz")
put(B1, "b1/b1_eval_lead_block.npz")
put(f"{PK}/raw/2026-10-25-kind/window_strata_eval141.npz", "strata/window_strata_eval141.npz")
