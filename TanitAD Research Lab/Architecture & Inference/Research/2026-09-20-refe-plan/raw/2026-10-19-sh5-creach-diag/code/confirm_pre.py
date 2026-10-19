"""C-REACH confirmation of an AMENDED code copy (Research Lab, 2026-10-19).  Runs exactly what `sh5_train_eval.py fit --stage PRE --folds 0 --seed S` runs (`fit_main` -> `_fit_locked`: G-LIVE, train,
predict, OOF file, the partial-fold ZZPREFLIGHT report), from the code directory given by --code, for every seed given.  The ONLY difference from `fit_main` is that the seal guard is not called: the
amended copy is a DRAFT (the Master Mind seals).  The GPU is taken through that copy's `sh5_gpulock` (unchanged from the sealed one).

usage: PY confirm_pre.py --code DIR --seeds 0,1 --out OUTDIR
"""
import argparse
import json
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--out", required=True)
ap.add_argument("--max-wait", type=float, default=21600.0)
a = ap.parse_args()
sys.path.insert(0, a.code)
import numpy as np                       # noqa: E402
import sh5_lib as L                      # noqa: E402
import sh5_train_eval as TE              # noqa: E402
import sh5_gpulock as GL                 # noqa: E402

DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
print(f"ZZCONFIRM code {a.code}: LR {L.LR} EPOCHS {L.EPOCHS_TOTAL} = {L.CYCLES_SHARDED} x {L.INNER_SHARDED} FOURIER_K {L.FOURIER_K} WARMUP {L.WARMUP} P_SUB {L.P_SUB}", flush=True)
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
ctx_rows = [row_of[str(t)] for t in D["token"]]
os.makedirs(a.out, exist_ok=True)


def body():
    rcs = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        x = types.SimpleNamespace(ctx=f"{DD}/ctx_local/ctx.npy", out_dir=a.out, stage="PRE", seed=s)
        specs = [TE.Spec("plant", 0, s)]
        rc = TE.RunCfg(seed=s)
        print(f"ZZSH5FIT stage PRE seed {s}: 1 learners, {D['N']} sets, {D['T']} tokens (amended copy)", flush=True)
        rcs[s] = TE._fit_locked(D, ctx_rows, specs, rc, x)
        print(f"ZZCONFIRM_SEED {s} rc {rcs[s]}", flush=True)
    return max(rcs.values())


rc = GL.run_locked_inprocess(body, max_wait=a.max_wait, min_avail_gb=L.MIN_AVAIL_GB_FIT, owner="sh5_diag", job="sh5-creach-confirm", log=lambda m: print(m, flush=True))
print(f"ZZCONFIRM_EXIT {rc}", flush=True)
sys.exit(rc)
