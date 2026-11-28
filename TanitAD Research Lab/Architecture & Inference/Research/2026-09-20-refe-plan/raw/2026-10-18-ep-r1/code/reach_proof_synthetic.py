"""Reach proof for EP-R1 on a SYNTHETIC bank (never the real one): runs epr1.reachability_check and writes the numbers it measured.
    CUDA_VISIBLE_DEVICES=-1 python -I reach_proof_synthetic.py <out.json>
Every number here comes from the synthetic world of test_epr1.make_world; no real label, feature or logit is read.  The claims it banks:
  full / pair_only / anchor_only : EP-row gradient > 0 (weight and bias)
  every frozen row, the features : EXACTLY 0.0   (frozen_rows_w_per_row has six zeros; `features` is 0.0)
  zero_control                   : EXACTLY 0.0 on the EP row too (a probe stuck at non-zero cannot pass)
  C6 (i)  unsafe-label invariance bit-identical ; C6 (ii) all-unsafe batch gradient exactly 0.0 ; C6 (iii) safe-label change changes the gradient
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1 as R            # noqa: E402
import test_epr1 as T       # noqa: E402

tmp = tempfile.mkdtemp(prefix="epr1_reach_")
try:
    w = T.make_world(os.path.join(tmp, "w"))
    o = T.opts_for(w, os.path.join(tmp, "o"))
    parts = R._setup(o)
    W6, b6, data = parts[8], parts[9], parts[10]
    rep = R.reachability_check(W6, b6, R._probe_batch(data, 64), R.Cfg(lr=1e-3, anchor=0.1))
    rep["world"] = "synthetic (test_epr1.make_world, seed 0): 40 logs, unconditional EP labels (violators 0), NC in {0, 0.5, 1}"
    json.dump(rep, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(json.dumps({k: rep[k] for k in ("full", "zero_control", "c6_unsafe_label_invariance_bit_identical", "c6_all_unsafe_ep_row_grad", "c6_safe_label_change_changes_gradient", "update_moved_ep_row_max_abs")}, indent=1))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
