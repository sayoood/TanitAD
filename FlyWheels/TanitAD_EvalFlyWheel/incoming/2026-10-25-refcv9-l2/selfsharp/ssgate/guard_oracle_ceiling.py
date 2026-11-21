"""guard_oracle_ceiling.py -- DESCRIPTIVE (no bar, privileged): the factored decode applied to the GT-class ORACLE force, i.e. the
ceiling a more ACCURATE lon head could reach once lat/lon are decoupled.  S6 compliant, stage on, seed 0; episode bootstrap B 2000.

    python guard_oracle_ceiling.py --out <dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssguard_score as SG  # noqa: E402
C, S, G = SG.C, SG.S, SG.G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--bank-ss", default="F:/refcv9_selfsharp")
    a = ap.parse_args()
    PR = C.stack()["PR"]; rows = S.metric_rows(PR); keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    E = C.load_eval_pf(0)
    base = S.join_plan8(S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0), E)
    orc = S.join_plan8(S.load_trace(G.L2 / "bank" / "trace_oracle_s0.npz"), E)
    ss = S.join_plan8(S.load_trace(Path(a.bank_ss) / "trace_selfsharp_s0.npz"), E)
    cells = lambda P: S.arm_cells(E, P)["stage"]
    cb = cells(base)
    arms = {"ORACLE": orc, "GUARD-ORACLE": SG.guard(base, orc).astype(base.dtype), "GUARD-ALL": SG.guard(base, ss).astype(base.dtype)}
    res = {"tool": "guard_oracle_ceiling.py", "script_sha256": G.sha(__file__), "reading": "DESCRIPTIVE, privileged (GT class) -- a ceiling, not a deployable arm",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames", "estimator": G.ESTIMATOR, "arms": {}}
    for nm, P in arms.items():
        c = cells(P); cmp, worse = G.g_worse(PR, E["eid"], c, cb, rows, keys)
        res["arms"][nm] = {"md": S.rate(c, "missed_decel"), "fd": S.rate(c, "false_decel"), "worse": worse,
                           "cells": {k: {f: cmp[k][f] for f in ("delta", "lo", "hi", "separated")} for k in ("missed_decel", "false_decel", "ade_m", "fde_m", "LAT_heading_mae_deg", "LAT_cross_mae_m")}}
    md_b = S.rate(cb, "missed_decel")
    g, o = res["arms"]["GUARD-ALL"]["md"], res["arms"]["GUARD-ORACLE"]["md"]
    res["base_md"], res["share_of_guarded_ceiling_reached_by_GUARD-ALL"] = md_b, (md_b - g) / (md_b - o) if md_b != o else None
    Path(a.out).mkdir(parents=True, exist_ok=True); C.jdump(res, Path(a.out) / "GUARD_ORACLE_CEILING.json")
    for nm, r in res["arms"].items():
        print(nm, f"MD {r['md']:.4f} FD {r['fd']:.4f} worse={r['worse']}", {k: f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in r["cells"].items()})
    print("share", res["share_of_guarded_ceiling_reached_by_GUARD-ALL"])


main()
