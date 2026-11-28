"""n1dw_strategic.py -- STAGED, NOT RUN. The STRATEGIC contrast N1DW{2,4} - N1R of the N1DW read (adapted from navrbd_strategic.py; the ROUTE-COST rule is unchanged) (card 5.2), adapted from navrb2_strategic.py with the SAME ROUTE-COST rule, unchanged.
T0 at the window's first tick, fresh ages.  The cell functions are navrb2_strategic.pair_contrasts / route_cost_cells (themselves strat1_strategic's own, frozen, imported): S_CKPT at 30 / 50 / 80 m real / donor / zero,
the route-following skill (real - donor), RC-A route setting with the floor from C0_rep - C0_s0, and the RB-window cells beside.
Pair N1D-N1R per seed: R = the N1D arm (eval_NAVRBD-N1D_s{s}_<v>.npz in --t0), the comparison arm = the NAVRB2 N1R pass (eval_NAVRB2-N1R_s{s}_<v>.npz in --n1r-t0, made on this box in the same environment).
route_cost['N1D-N1R'] = the list of RC-A cells that are a cost at BOTH seeds (the card's ROUTE-COST rule).  navrbd_read.py reads bool(route_cost['N1D-N1R']).
Donor passes of the N1D arms hold the RB channel at zero beside the donor nav, exactly as the N1R donor passes did (navrb2_lib.READING r4), so the pair is like for like.
Usage: python navrbd_strategic.py --out OUT.json [--t0 DIR] [--n1r-t0 DIR] [--banked-t0 DIR] [--grid GRID.json] [--n-boot 2000] [--hmax 16]"""
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
T0_DEFAULT = os.environ.get("N1DW_T0", HERE + "/out/t0")
HMAX = 16                                                                       # cfg.hmax recorded in RESULT_NAVRB2.json strata (the item's history rows)


def pair_name(w):
    return f"N1DW{w}-N1R"


def route_cost_of(rows_by_seed):
    """RC-A cells that are a cost at BOTH seeds (the NAVRB2 rule, unchanged).  Pure."""
    return N2S.route_cost_cells(rows_by_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default=T0_DEFAULT)
    ap.add_argument("--n1r-t0", default=N1R_T0)
    ap.add_argument("--banked-t0", default=NL.BANKED_T0)
    ap.add_argument("--grid", default="")
    ap.add_argument("--n-boot", type=int, default=NL.NB)
    ap.add_argument("--hmax", type=int, default=HMAX)
    ap.add_argument("--weights", type=int, nargs="+", default=[2, 4])
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
    rep = {"tool": "n1dw_strategic.py", "tier": "T0 at the window's first tick (fresh ages)", "evidence_class": "MEASURED (ours; T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0; 'another draw of EPISODES' only", "reading_r4": NL.READING["r4_T0_donor_channel"],
           "rule": "ROUTE-COST = an RC-A cell that is a cost at BOTH seeds (navrb2_strategic.route_cost_cells, unchanged), read N1D - N1R", "hmax": a.hmax, "RC-A_floor": floors, "pairs": {}, "route_cost": {}}
    for w in a.weights:
        per_seed = {}
        for s in NL.SEEDS:
            R = {v: load(a.t0, f"N1DW-N1DW{w}_s{s}", v) for v in VARIANTS}
            B = {v: load(a.n1r_t0, f"NAVRB2-N1R_s{s}", v) for v in VARIANTS}
            per_seed[s] = N2S.pair_contrasts(CI, G, eid, R, B, rb, floors, a.n_boot)
        rep["pairs"][pair_name(w)] = {str(s): per_seed[s] for s in per_seed}
        rep["route_cost"][pair_name(w)] = route_cost_of({s: per_seed[s]["route_setting_RC-A_R_minus_banked"] for s in per_seed})
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "| route_cost:", rep["route_cost"])


if __name__ == "__main__":
    main()
