"""strat1_read.py -- the PREREG_V8F_STRAT1 read: driving cells (24 per seed) and the assembly of RESULT_STRAT1.json (READ_PLAN 2.2, 2.6).  Pattern: obd_read.py (imported: cell_delta, floors, to_cells).
Tier of the 24 cells: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner), 141 episodes / 2,399 windows, per seed.  CPU, zero training.
Evidence class: MEASURED (ours; dumps).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); "another TRAINING run" is the floor F;
"another INFERENCE run" does not apply (deterministic planner, one environment, one thread count).
COMMITTED RULES (card section 4 and 6, verbatim in force)
  Floor F = |C0_rep - C0_s0| per cell, read at seed 0 and applied to both seeds.  COST on a cell at a seed: Delta = S1 - C0 separated on the worse side AND |Delta| > F.  GAIN: separated on the
  better side AND |Delta| > F.  Outcomes (per cell and seed, both seeds):
    KEEPS  no cost on any of the 24 cells at either seed;   ADDS = KEEPS and the same cell a gain at both seeds;
    COSTS-NAV    a cell costs at both seeds and N1 - C0 costs on that cell at both seeds;   COSTS-HEADS  a cell costs at both seeds in S1 - C0 and N1 - C0 does not cost there at both seeds;
    SEEDS-DISAGREE  a cost at exactly one seed (on a cell that does not cost at both).
  READING CHOICE (r1, the card is silent, written before any number): when a cell costs at both seeds AND another cell costs at one seed, both labels are printed (COSTS-* plus SEEDS-DISAGREE);
  a both-seed cost of both kinds prints COSTS-NAV and COSTS-HEADS.  ROUTE-COST: a cost at both seeds on an RC-A cell (strat1_strategic.py) while the 24 read KEEPS or ADDS.
  Every outcome prints its next step (the card's trigger).  No pause string: STRATEGIC is read in full (PI 2026-11-16).
Usage:  python strat1_read.py --dumps DIR --out RESULT_STRAT1.json [--md RESULT_STRAT1.md] [--strategic STRAT.json] [--ctx CTX.json] [--smoke] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("STRAT1_OBD_DIR", HERE + "/theory/obd/read"))
sys.path.insert(0, HERE + "/theory/otband/read")
import strat1_lib as L  # noqa: E402
import obd_read as OR  # noqa: E402  (its module level is definitions only)

CELLS, FAMILY = OR.CELLS, OR.FAMILY
NEXT = {
    "KEEPS": "the route text replaces the oracle nav in the recipe; N2 card (44-class text embedding + distance, an e_nav change) to recover roundabouts, bends and ramps",
    "ADDS": "as KEEPS; the gain is named with its echo-split row",
    "COSTS-NAV": "N2 card next (the 38.1 % FOLLOW-type text on v9-turn windows is the named suspect); no adoption",
    "COSTS-HEADS": "arm S1d = S1 with --ctx-detach on (built, default off), same seeds and reads",
    "SEEDS-DISAGREE": "C0 and S1 at seeds 10 (init 0) and 11 (init 1); a cell is a cost at >= 3 of 4 seeds with one per init",
    "ROUTE-COST": "the N2 card (reported as written)",
}


def cost_sets(cells):
    return {c for c, v in cells.items() if v.get("cost")}


def gain_sets(cells):
    return {c for c, v in cells.items() if v.get("gain")}


def driving_outcome(s1c0, n1c0):
    """s1c0 / n1c0: {seed: {cell: cell_delta}} for seeds 0 and 1 -> (labels [..], detail).  Pure; the unit tests plant every row."""
    if set(s1c0) != {0, 1}:
        return ["NOT-YET"], {"reason": "the card needs S1 - C0 at both seeds"}
    c0, c1 = cost_sets(s1c0[0]), cost_sets(s1c0[1])
    both, one = c0 & c1, c0 ^ c1
    labels, det = [], {"costs_both_seeds": sorted(both), "costs_one_seed": sorted(one)}
    if not c0 and not c1:
        g = gain_sets(s1c0[0]) & gain_sets(s1c0[1])
        det["gains_both_seeds"] = sorted(g)
        return (["ADDS"] if g else ["KEEPS"]), det
    if both:
        if len(n1c0) == 2:
            n0, n1 = cost_sets(n1c0[0]), cost_sets(n1c0[1])
            nav = sorted(both & n0 & n1)
            heads = sorted(both - (n0 & n1))
            det["costs_nav"], det["costs_heads"] = nav, heads
            if nav:
                labels.append("COSTS-NAV")
            if heads:
                labels.append("COSTS-HEADS")
        else:
            labels.append("COSTS (N1 not read: NAV vs HEADS not attributable)")
    if one:
        labels.append("SEEDS-DISAGREE")
    return labels, det


def next_steps(labels, route_cost):
    out = [f"{x}: {NEXT[x.split(' ')[0]]}" for x in labels if x.split(" ")[0] in NEXT]
    if route_cost:
        out.append("ROUTE-COST: " + NEXT["ROUTE-COST"])
    return out or ["no committed next step yet: an outcome is missing (NOT-YET)"]


def pair_rows(CI, arm, ref, eid, floors, nb):
    return OR.compare_cells(CI, arm, ref, eid, floors, nb)


def slim(cells):
    keep = ("status", "delta", "lo", "hi", "separated", "floor", "above_floor", "gain", "cost", "n_windows", "n_episodes")
    return {c: {k: v[k] for k in keep if k in v} for c, v in cells.items()}


def family_blocks(cells_by_pair):
    """per family: the cells with their rows (never pooled into one score)"""
    fam = {}
    for c in CELLS:
        f = FAMILY[c.split(":")[-1]]
        fam.setdefault(f, {})[c] = {p: cells_by_pair[p][c] for p in cells_by_pair}
    return fam


def provenance(dumps, tags):
    P = {"inputs_sha256": {k: L.PINS[k] for k in L.PINS}, "strat_platform_by_clip.npz": L.sha256_file(L.INP + "/strat_platform_by_clip.npz"),
         "navtext_nav_report.json": L.sha256_file(L.INP + "/navtext_nav_report.json"), "read_pins": json.load(open(HERE + "/READ_PINS.json", encoding="utf-8")),
         "checkpoints": {}, "bank_manifests": {}, "code_of_record": L.COMMIT}
    for a in tags:
        p = f"{L.CK}/{a}/ckpt_last.pt"
        if os.path.exists(p):
            P["checkpoints"][a] = L.sha256_file(p)
            P["bank_manifests"][a] = L.sha256_file(f"{L.CK}/{a}/BANK_MANIFEST.json")
    for f in ("strat1_read.py", "strat1_strategic.py", "strat1_ctx_read.py", "strat1_lib.py", "run_dumps_strat1.py", "run_t0_strat1.py"):
        P.setdefault("reader_sha256", {})[f] = L.sha256_file(f"{HERE}/{f}")
    return P


def load_arm_cells(geom, dg, J, glat, glon, d, expect_eps):
    x = OR.load_dump(d) if expect_eps == 141 else load_dump_n(d, expect_eps)
    for k in OR.SAME:
        assert np.array_equal(x[k], dg[k]), f"window or model-free array differs from the reference dump: {k} in {d}"
    return OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)


def load_dump_n(d, n):
    import glob
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == n, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.environ.get("STRAT1_T1", L.HERE + "/t1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--strategic", default=None)
    ap.add_argument("--ctx", default=None)
    ap.add_argument("--smoke", action="store_true", help="a code check on reduced dumps: skips the 141-episode count; never a result")
    ap.add_argument("--smoke-episodes", type=int, default=141)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci()
    dpath = lambda arm: f"{a.dumps}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    ready = {arm: os.path.isdir(dpath(arm)) for arm in L.ARMS}
    rep = {"tool": "strat1_read.py", "card": "Research/2026-10-17-v8f-mini/raw/2026-11-16-v8f-strat1/design/PREREG_V8F_STRAT1.md (frozen 2026-11-16)", "smoke": bool(a.smoke),
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC route": "T0 at the window's first tick",
                    "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = the floor F = |C0_rep - C0_s0|",
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("Usage:")[0].strip(), "arms_ready": ready, "controls": {"problems": [], "notes": {}}}
    prob = rep["controls"]["problems"]
    if not (ready["C0_s0"] and ready["C0_rep"]):
        rep["outcome"] = "NOT-YET"
        rep["reason"] = "C0_s0 and C0_rep (the floor) are required"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, rep["outcome"], rep["reason"])
        return
    import obd_geom as G
    geom = G.build()
    ne = a.smoke_episodes if a.smoke else 141
    dg = load_dump_n(dpath("C0_s0"), ne)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm in L.ARMS:
        if ready[arm]:
            cells[arm] = load_arm_cells(geom, dg, J, glat, glon, dpath(arm), ne)
            print(f"loaded {arm} {time.time() - t0:.0f}s", flush=True)
    eid = dg["eid"]
    F = OR.floors_from(cells["C0_rep"], cells["C0_s0"])
    rep["floors"] = {"F": {c: v for c, v in F.items()}, "definition": "|C0_rep - C0_s0| per cell at seed 0, applied to both seeds (one replicate pair)"}
    rep["controls"]["K1_self_pair"] = OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    pairs, s1c0, n1c0 = {}, {}, {}
    for s in (0, 1):
        if f"S1_s{s}" in cells and f"C0_s{s}" in cells:
            pairs[f"S1-C0_s{s}"] = pair_rows(CI, cells[f"S1_s{s}"], cells[f"C0_s{s}"], eid, F, a.n_boot)
            s1c0[s] = pairs[f"S1-C0_s{s}"]
        if f"N1_s{s}" in cells and f"C0_s{s}" in cells:
            pairs[f"N1-C0_s{s}"] = pair_rows(CI, cells[f"N1_s{s}"], cells[f"C0_s{s}"], eid, F, a.n_boot)
            n1c0[s] = pairs[f"N1-C0_s{s}"]
        if f"S1_s{s}" in cells and f"N1_s{s}" in cells:
            pairs[f"S1-N1_s{s}"] = pair_rows(CI, cells[f"S1_s{s}"], cells[f"N1_s{s}"], eid, F, a.n_boot)
    labels, det = driving_outcome(s1c0, n1c0)
    strat = json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None
    ctx = json.load(open(a.ctx, encoding="utf-8")) if a.ctx else None
    route_cost = bool(strat and strat.get("route_setting_RC-A", {}).get("cost_at_both_seeds") and any(x in ("KEEPS", "ADDS") for x in labels))
    fam = family_blocks({p: slim(v) for p, v in pairs.items()})
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "estimator": "paired episode-cluster bootstrap", "n_windows": int(len(eid)),
                           "n_episodes": int(len(np.unique(eid)))} for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    rep["families"]["STRATEGIC"] = {"tier": "T0 at the window's first tick (route); T0 (detected info)", "route_setting_RC-A": (strat or {}).get("route_setting_RC-A", "NOT-YET"),
                                    "route_following": (strat or {}).get("outcome_route_following", "NOT-YET"), "detected_info": (ctx or {}).get("outcome", "NOT-YET"),
                                    "not_in_arm": {"derived_goals": {"status": "not in this arm", "reason": "later card; goal-input rule"},
                                                   "intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc", "would_have_n": {"6-12 s": 32, "12-30 s": 69, "above 30 s": 20}},
                                                   "visibility": {"status": "not built", "reason": "label 25/40 below the constant 31/40"}}}
    rep["outcome_driving"], rep["outcome_driving_detail"] = labels, det
    rep["route_cost"] = route_cost
    rep["next_step"] = next_steps(labels, route_cost)
    rep["pairs_beside"] = {"C0_vs_box2_OBD-CTL": "not read here (cross-venue beside row; instrument beside_obd.py, MM step)"}
    rep["provenance"] = provenance(a.dumps, L.ARMS)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "|", labels, "| problems:", len(prob), "|", rep["wall_s"], "s")
    for n in rep["next_step"]:
        print("  NEXT:", n)


if __name__ == "__main__":
    main()
