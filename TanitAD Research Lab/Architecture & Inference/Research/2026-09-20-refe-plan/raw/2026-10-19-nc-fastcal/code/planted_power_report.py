"""NC-FASTCAL: print the planted-bank numbers behind the C-NULL power claim (same banks and decision path as test_ncfastcal.py).  Synthetic only; no real label."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                        # noqa: E402
import test_ncfastcal as T                                                                # noqa: E402

L, P = T.L, T.P
for kind in ("speed", "global"):
    b = T.planted_bank(kind)
    S = T.stats_for(b)
    d, params = T.evaluate(b, S=S)
    pdms, pk_p = P.pdms_from_labels(b["lab"]), P.rule_pick(b["z6"])
    print(f"planted bank '{kind}' (N {len(b['z6'])} sets, {b['n_logs']} logs): calibrated composite RECOVERS = {d['recovers']}")
    for sd, v in d["per_seed"].items():
        print(f"  seed {sd}: dNC {v['d_nc']['point']:+.4f} [{v['d_nc']['lo']:+.4f}, {v['d_nc']['hi']:+.4f}]  dPDMS x100 {v['d_pdms_x100']['point']:+.3f} [{v['d_pdms_x100']['lo']:+.3f}, {v['d_pdms_x100']['hi']:+.3f}]")
    print("  fitted a by stratum (S, Q1..Q4, mean over folds):", np.round(params[:, :, 0].mean(0), 3).tolist(), " b:", np.round(params[:, :, 1].mean(0), 3).tolist())
    du = L.arm_decision(P, L.composite(b["z6"], ep=b["ep"]), b["lab"], pdms, pk_p, S)
    print(f"  uncalibrated composite: dNC {du['d_nc']['point']:+.4f} [{du['d_nc']['lo']:+.4f}, {du['d_nc']['hi']:+.4f}]  dPDMS x100 {du['d_pdms_x100']['point']:+.3f} [{du['d_pdms_x100']['lo']:+.3f}, {du['d_pdms_x100']['hi']:+.3f}]")
    null = L.run_null(P, b["z6"], b["lab"], {0: b["ep"], 1: b["ep"]}, b["code"], b["fold"], L.token_groups(b["tokens"]), pdms, pk_p, S)
    print(f"  C-NULL: RECOVERS fires on {null['fires']} of {null['of']}; (i) alone on both seeds {null['cond_i_only_fires']} of {null['of']}; VOID {null['void']}; final label {L.final_label(d['recovers'], null['void'])}")
    for r in null["replicates"]:
        print(f"    rep {r['rep']}: " + "; ".join(f"seed {sd} dNC upper {v['d_nc']['hi']:+.4f}, dPDMS lower {v['d_pdms_x100']['lo']:+.3f}" for sd, v in r["per_seed"].items()))
