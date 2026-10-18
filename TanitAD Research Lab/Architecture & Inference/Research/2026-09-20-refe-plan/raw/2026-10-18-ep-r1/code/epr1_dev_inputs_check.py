#!/usr/bin/env python3
"""INPUT-PHASE-ONLY check of `epr1_dev.py` against the REAL landed dev inputs.  Spends nothing: it never calls `load_dev_outcomes` / `run_dev`, so no `pdms`, `sub` or `valid` array is opened and no EP-R1 row is applied
(there is none yet).  It proves, on the real files, the things the dev gate needs BEFORE any outcome is read:

  * the three dev files hash to the landed sha256 and load under the pinned shape (1,705 tokens / 24 logs);
  * integrity: P2's real head reproduces the stored logits from the cached pre-head features (all six columns) within 2e-3, and a next-token control reads far above it;
  * C0 (identity): with P2's own EP row (and EP + TTC rows) the logits are bit-identical and the pick equals P2's banked pick on 1,705 / 1,705 tokens;
  * C4 (fan frozen): the dump the features came from holds exactly the table's proposals (max abs diff 0.0 m), same token/log order, same picks.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I epr1_dev_inputs_check.py --out epr1_dev_inputs_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1_dev as D   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-dir", default=D.DEV_DIR)
    ap.add_argument("--p2-ckpt", default=D.R.DEFAULT_P2_CKPT)
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    opts = D.DevOptions(run_dir=out_dir, registered_result_sha256="n/a (inputs-only check)", dev_dir=a.dev_dir, p2_ckpt=a.p2_ckpt, out_dir=out_dir)
    guard = opts.guard()
    W6, b6 = D.load_p2_head(guard, opts.p2_ckpt)
    inp = D.load_dev_inputs(guard, opts)
    rec = {"what": "epr1_dev.py input phase on the REAL landed dev inputs; no outcome array opened; no row applied", "dev": inp.provenance,
           "integrity_featcheck": D.integrity_featcheck(inp, W6, b6, opts.featcheck_tol, opts.chunk), "C0_pick_identity": D.c0_identity(inp, W6, b6, opts.chunk),
           "C4_fan_frozen": D.c4_fan_frozen(inp), "outcomes_opened": False, "elapsed_s": round(time.time() - t0, 1)}
    rec["all_pass"] = bool(all(rec[k]["pass"] for k in ("integrity_featcheck", "C0_pick_identity", "C4_fan_frozen")))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(D._jsonable(rec), f, indent=1)
    print(json.dumps(D._jsonable({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "sha256"}) for k, v in rec.items()}), indent=1))
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
