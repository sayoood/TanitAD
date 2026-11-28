"""n1dc_strategic.py -- STAGED. The STRATEGIC contrasts of the N1DC read (card PREREG_V8F_N1DC.md sha256 2eae7964, section 3), adapted from n1dw_strategic.py with the SAME ROUTE-COST rule, unchanged.
T0 at the window's first tick, fresh ages.  Cell functions are navrb2_strategic.pair_contrasts / route_cost_cells (frozen, imported): S_CKPT at 30 / 50 / 80 m real / donor / zero, the route-following skill (real - donor),
RC-A route setting with the floor from C0_rep - C0_s0, and the RB-window cells beside.
Pairs per seed: R = N1DC_s{s} (eval_N1DC-N1DC_s{s}_<v>.npz in --t0); comparison B = N1D_s{s} (banked T0 passes eval_NAVRBD-N1D_s{s}_<v>.npz in --n1d-t0; the card's comparator, DECIDING) and, beside, N1R_s{s}
(eval_NAVRB2-N1R_s{s}_<v>.npz in --n1r-t0).  route_cost['N1DC-N1D'] / ['N1DC-N1R'] = the RC-A cells that are a cost at BOTH seeds.
Donor passes of the N1DC arms hold the RB channel at zero beside the donor nav (navrb2_lib.READING r4), exactly as the N1D / N1R donor passes did, so the pairs are like for like.
Usage: python n1dc_strategic.py --out OUT.json [--t0 DIR] [--n1d-t0 DIR] [--n1r-t0 DIR] [--banked-t0 DIR] [--grid GRID.json] [--n-boot 2000] [--hmax 16]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "F:/TanitAD-scratch/strat1/navrb2/read")
import navrb2_lib as NL  # noqa: E402
import navrb2_strategic as N2S  # noqa: E402

for _p in (NL.SR, NL.SR + "/theory/obd/read", NL.SR + "/theory/otband/read"):
    sys.path.insert(0, _p)
import strat1_lib as L  # noqa: E402
import strat1_strategic as SS  # noqa: E402  (frozen)
import obd_read as OR  # noqa: E402

VARIANTS = N2S.VARIANTS
N1R_T0 = "F:/TanitAD-scratch/strat1/navrb2/read/out/t0"
N1D_T0 = os.environ.get("N1DC_N1D_T0", HERE + "/banked_n1d/x/out/t0")
T0_DEFAULT = os.environ.get("N1DC_T0", HERE + "/out/t0")
HMAX = 16                                                                       # cfg.hmax recorded in RESULT_NAVRB2.json strata (the item's history rows)


COMPARATORS = (("N1D", "NAVRBD-N1D"), ("N1R", "NAVRB2-N1R"))     # (name, tag prefix of the banked pass)


def pair_name(c):
    return f"N1DC-{c}"


def route_cost_of(rows_by_seed):
    """RC-A cells that are a cost at BOTH seeds (the NAVRB2 rule, unchanged).  Pure."""
    return N2S.route_cost_cells(rows_by_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default=T0_DEFAULT)
    ap.add_argument("--n1r-t0", default=N1R_T0)
    ap.add_argument("--n1d-t0", default=N1D_T0)
    ap.add_argument("--banked-t0", default=NL.BANKED_T0)
    ap.add_argument("--grid", default="")
    ap.add_argument("--n-boot", type=int, default=NL.NB)
    ap.add_argument("--hmax", type=int, default=HMAX)
    a = ap.parse_args()
    CI = L.load_ci()                                                  # the STRAT1 read tree ci.py, as the frozen readers use
    join, navtok, navtext, ci, t, eid = SS.load_inputs(a.grid or None)
    G = SS.ground(navtok, navtext, join, ci, t)
    rb = NL.strata(NL.rb_windows(a.grid or None), a.hmax)["rb"]

    def load(root, tag, v):
        z = dict(np.load(f"{root}/eval_{tag}_{v}.npz", allow_pickle=True))
        if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
            raise RuntimeError(f"{tag} {v}: the evaluator's windows differ from the grid")
        return z
    c0 = {k: SS.rc_cells(SS.rcA_errors(load(a.banked_t0, f"STRAT1-{k}", "real")["ckpt"], G)) for k in ("C0_rep", "C0_s0")}
    floors = {c: OR.floor_of(c0["C0_rep"][c], c0["C0_s0"][c]) for c in SS.RC_CELLS}
    rep = {"tool": "n1dc_strategic.py", "tier": "T0 at the window's first tick (fresh ages)", "evidence_class": "MEASURED (ours; T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0; 'another draw of EPISODES' only", "reading_r4": NL.READING["r4_T0_donor_channel"],
           "rule": "ROUTE-COST = an RC-A cell that is a cost at BOTH seeds (navrb2_strategic.route_cost_cells, unchanged), read N1DC - N1D (deciding) and N1DC - N1R (beside)", "hmax": a.hmax, "RC-A_floor": floors, "pairs": {}, "route_cost": {}}
    roots = {"N1D": a.n1d_t0, "N1R": a.n1r_t0}
    for cname, ctag in COMPARATORS:
        per_seed = {}
        for s in NL.SEEDS:
            R = {v: load(a.t0, f"N1DC-N1DC_s{s}", v) for v in VARIANTS}
            B = {v: load(roots[cname], f"{ctag}_s{s}", v) for v in VARIANTS}
            per_seed[s] = N2S.pair_contrasts(CI, G, eid, R, B, rb, floors, a.n_boot)
        rep["pairs"][pair_name(cname)] = {str(s): per_seed[s] for s in per_seed}
        rep["route_cost"][pair_name(cname)] = route_cost_of({s: per_seed[s]["route_setting_RC-A_R_minus_banked"] for s in per_seed})
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "| route_cost:", rep["route_cost"])


if __name__ == "__main__":
    main()
