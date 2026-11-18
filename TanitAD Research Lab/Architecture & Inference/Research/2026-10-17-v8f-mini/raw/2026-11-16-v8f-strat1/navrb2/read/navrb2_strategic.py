"""navrb2_strategic.py -- the STRATEGIC contrasts R arm - banked arm that the frozen STRAT1 readers (which read arm - C0) do not print (card 5.2: "Each STRATEGIC cell also prints S1R - S1").  T0 at the window's first tick,
fresh ages; the cell functions are strat1_strategic's own (UNEDITED, imported): route_cells (S_CKPT at 30 / 50 / 80 m), rcA_errors / rc_cells / route_setting (RC-A, floor from C0_rep - C0_s0), contrast_rows (n rule), ground.
Per pair (S1R-S1, N1R-N1) and seed: S_CKPT real, donor and zero contrasts R - banked per arc; the route-following SKILL (real - donor) of the R arm minus the banked arm's; RC-A route setting R - banked with the
cost / gain rule; on the RB windows beside (S_CKPT real - donor of each arm; n printed, descriptive).  route_cost[pair] = an RC-A cell that is a cost at BOTH seeds (the card's ROUTE-COST rule, read R - banked).
READING r4 (navrb2_lib.READING): the R donor pass holds the channel at zero beside the donor nav.  Usage: python navrb2_strategic.py --out OUT.json [--t0 DIR] [--grid GRID.json] [--n-boot 2000]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

for _p in (NL.SR, NL.SR + "/theory/obd/read", NL.SR + "/theory/otband/read"):
    sys.path.insert(0, _p)
import strat1_lib as L  # noqa: E402
import strat1_strategic as SS  # noqa: E402  (frozen)
import obd_read as OR  # noqa: E402

VARIANTS = ("real", "donor", "zero")


def pair_contrasts(CI, G, eid, R, B, rb, floors, nb):
    """R, B: {variant: T0 npz dict} -> the contrasts of one seed of one pair.  Pure given the arrays (the unit tests plant them)."""
    cR = {v: SS.route_cells(R[v]["ckpt"], G) for v in VARIANTS}
    cB = {v: SS.route_cells(B[v]["ckpt"], G) for v in VARIANTS}
    out = {"S_CKPT_R_minus_banked": {v: SS.contrast_rows(CI, cR[v], cB[v], eid) for v in VARIANTS},
           "skill_real_minus_donor_R_minus_banked": SS.contrast_rows(CI, cR["real"] - cR["donor"], cB["real"] - cB["donor"], eid)}
    m = np.asarray(rb, bool)
    out["on_RB_beside"] = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[m]))),
                           "R_real_minus_donor": SS.contrast_rows(CI, cR["real"][m], cR["donor"][m], np.asarray(eid)[m]),
                           "banked_real_minus_donor": SS.contrast_rows(CI, cB["real"][m], cB["donor"][m], np.asarray(eid)[m]),
                           "R_real_minus_banked_real": SS.contrast_rows(CI, cR["real"][m], cB["real"][m], np.asarray(eid)[m])}
    rcR, rcB = SS.rc_cells(SS.rcA_errors(R["real"]["ckpt"], G)), SS.rc_cells(SS.rcA_errors(B["real"]["ckpt"], G))
    out["route_setting_RC-A_R_minus_banked"] = SS.route_setting(CI, rcR, rcB, floors, eid, nb)
    return out


def route_cost_cells(rows_by_seed):
    """RC-A cells that are a cost at BOTH seeds (needs exactly two seeds).  Pure."""
    if len(rows_by_seed) != 2:
        return []
    return [c for c in SS.RC_CELLS if all(r[c].get("cost") for r in rows_by_seed.values())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", default=NL.T0)
    ap.add_argument("--banked-t0", default=NL.BANKED_T0)
    ap.add_argument("--grid", default="")
    ap.add_argument("--n-boot", type=int, default=NL.NB)
    a = ap.parse_args()
    CI = L.load_ci(NL.TREE)
    join, navtok, navtext, ci, t, eid = SS.load_inputs(a.grid or None)
    G = SS.ground(navtok, navtext, join, ci, t)
    w = NL.rb_windows(a.grid or None)
    rb = NL.strata(w, NL.hist_of(f"{NL.CK}/N1R_s0/ckpt_last.pt"))["rb"]

    def load(root, tag, v):
        z = dict(np.load(f"{root}/eval_{tag}_{v}.npz", allow_pickle=True))
        if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
            raise RuntimeError(f"{tag} {v}: the evaluator's windows differ from the grid")
        return z
    c0 = {k: SS.rc_cells(SS.rcA_errors(load(a.banked_t0, f"STRAT1-{k}", "real")["ckpt"], G)) for k in ("C0_rep", "C0_s0")}
    floors = {c: OR.floor_of(c0["C0_rep"][c], c0["C0_s0"][c]) for c in SS.RC_CELLS}
    rep = {"tool": "navrb2_strategic.py", "tier": "T0 at the window's first tick (fresh ages)", "evidence_class": "MEASURED (ours; T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0; 'another draw of EPISODES' only", "reading_r4": NL.READING["r4_T0_donor_channel"],
           "RC-A_floor": floors, "pairs": {}, "route_cost": {}}
    for r_fam, b_fam in (("S1R", "S1"), ("N1R", "N1")):
        pn, per_seed = f"{r_fam}-{b_fam}", {}
        for s in NL.SEEDS:
            R = {v: load(a.t0, f"NAVRB2-{r_fam}_s{s}", v) for v in VARIANTS}
            B = {v: load(a.banked_t0, f"STRAT1-{b_fam}_s{s}", v) for v in VARIANTS}
            per_seed[s] = pair_contrasts(CI, G, eid, R, B, rb, floors, a.n_boot)
        rep["pairs"][pn] = {str(s): per_seed[s] for s in per_seed}
        rc = route_cost_cells({s: per_seed[s]["route_setting_RC-A_R_minus_banked"] for s in per_seed})
        rep["route_cost"][pn] = rc
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "| route_cost:", {k: v for k, v in rep["route_cost"].items()})


if __name__ == "__main__":
    main()
