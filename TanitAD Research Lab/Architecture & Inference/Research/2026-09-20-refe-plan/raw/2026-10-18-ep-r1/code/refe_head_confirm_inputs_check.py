#!/usr/bin/env python3
"""INPUT-PHASE-ONLY check of `refe_head_confirm.py` and `make_navtest_bank_plan.py` against the REAL landed navtest artifacts.  Spends nothing: it never calls `read_outcomes` / `stage_read`, so no `pdms`,
`sub` or `valid` array of the table is opened, no row is applied (there is none yet), and neither the S0 bank nor any GPU is touched.  It proves, on the real files:

  * the landed table hashes to the registered sha256 and loads under the pinned shape (12,146 tokens / 136 logs) through THIS module's guard (the dev guard would refuse these path names);
  * C0's table half: the planner rule on the stored logits reproduces the stored pick on 12,146 / 12,146 tokens;
  * the banked P2 seam `stage2_p2_full.npz`: poses == the table's executed pose of P2's pick, fingerprints equal, on every token (the control the applier runs on every look);
  * the plan converter: the 14 landed chunks partition the table's tokens exactly once, each chunk's sha256 and token_log verified, all 136 nuPlan DBs exist; 14 plans + the command list are written (scratch);
  * what a feature bank would weigh, from the table's own shape.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I refe_head_confirm_inputs_check.py --chunks-dir <chunks> --out-dir <scratch>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1_dev as D            # noqa: E402
import make_navtest_bank_plan as P   # noqa: E402
import refe_head_confirm as C   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--table", default=C.NAVTEST_TABLE)
    ap.add_argument("--banked-p2-seam", default=C.NAVTEST_BANKED_SEAM)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    opts = C.ConfirmOptions(out_dir=a.out_dir, bank_dir=os.path.join(a.out_dir, "bank_not_built"), table=a.table, banked_p2_seam=a.banked_p2_seam)
    guard = opts.guard()
    tab = C.load_table_inputs(guard, opts)                                       # sha256 + shape pins at the REAL defaults
    N = tab.n
    pk = np.concatenate([D.pick_logits(tab.logits[s:s + 1024]) for s in range(0, N, 1024)])
    rec = {"what": "refe_head_confirm.py / make_navtest_bank_plan.py input phase on the REAL landed navtest artifacts; no outcome array opened; no row applied; no GPU; the S0 bank is not touched",
           "table": {"path": a.table, "sha256": tab.sha256, "n_tokens": N, "n_logs": len(tab.log_names), "pins_overridden": opts.pins_overridden()},
           "C0_table_half": {"rule_pick_equals_stored_pick": int((pk == tab.pick).sum()), "n_tokens": N, "pass": bool((pk == tab.pick).all())}}
    z = D._io_npz(guard, a.banked_p2_seam)
    bt = {str(t): i for i, t in enumerate(z["token"])}
    rows = np.array([bt.get(str(t), -1) for t in tab.token])
    ok_tok = bool((rows >= 0).all())
    ar = np.arange(N)
    ours = tab.proposals[ar, tab.pick.astype(np.int64)].astype(np.float64)
    theirs = np.asarray(z["poses"])[rows].astype(np.float64) if ok_tok else None
    fps = bool(ok_tok and all(str(z["fingerprint"][r]) == tab.fingerprint[i] for i, r in enumerate(rows)))
    d = float(np.abs(ours - theirs).max()) if ok_tok else None
    rec["banked_p2_seam"] = {"path": a.banked_p2_seam, "n_seam_tokens": int(len(z["token"])), "all_table_tokens_in_seam": ok_tok, "poses_max_abs_m": d, "fingerprints_equal": fps,
                             "sampling": [float(x) for x in z["sampling"]], "pass": bool(ok_tok and d == 0.0 and fps)}
    z.close()
    man = P.build(a.chunks_dir, a.table, os.path.join(a.out_dir, "plans"), "D:/Projects/TanitAD/data/refe_navtest/bank", check_db=True)
    rec["plan_converter"] = {"n_chunks": man["n_chunks"], "n_tokens": man["n_tokens"], "n_logs": man["n_logs"], "all_nuplan_dbs_exist": True, "chunk_sizes": [e["n"] for e in man["chunks"]],
                             "pass": bool(man["n_chunks"] == 14 and man["n_tokens"] == 12146 and man["n_logs"] == 136)}
    feat_bytes = N * 64 * 256 * 2
    rec["bank_size"] = {"features_fp16_bytes": feat_bytes, "features_MB": round(feat_bytes / 1e6, 1), "note": "arithmetic from the pinned shape [12146,64,256] fp16; the fan dump beside it is ~0.3 GB (ESTIMATED from the 14 landed dump_main chunks: 98 MB without proposals20)"}
    rec["outcomes_opened"] = False
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["all_pass"] = bool(rec["C0_table_half"]["pass"] and rec["banked_p2_seam"]["pass"] and rec["plan_converter"]["pass"])
    out = os.path.join(a.out_dir, "refe_head_confirm_inputs_check.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(D._jsonable(rec), f, indent=1)
    print(json.dumps(D._jsonable(rec), indent=1))
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
