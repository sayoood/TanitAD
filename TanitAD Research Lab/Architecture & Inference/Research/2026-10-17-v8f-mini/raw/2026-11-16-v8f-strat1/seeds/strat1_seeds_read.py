"""strat1_seeds_read.py -- the STRAT1 SEEDS-DISAGREE follow-up read.  Frozen card PREREG_V8F_STRAT1, section 6 trigger, verbatim: "C0 and S1 at seeds 10 (init 0) and 11 (init 1);
a cell is a cost at >= 3 of 4 seeds with one per init".  Driving cells: the 24 cells of strat1_read.py (imported UNEDITED).  STRATEGIC at seeds 10 / 11 comes from
strat1_strategic.py and strat1_ctx_read.py run with --seeds 10 11 (seeds_run.py) and is assembled beside the s0 / s1 STRATEGIC read of RESULT_STRAT1.json.
Tier of the 24 cells: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner), 141 episodes / 2,399 windows per seed.
Evidence class: MEASURED (ours; dumps, zero training).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES");
"another TRAINING run" = the four seeds and the card's floor F; "another INFERENCE run" does not apply (deterministic planner, one environment, one thread count).
RULES (written 2026-11-16 ~21:45Z, before any seed-10 / seed-11 number exists: the four arms are training on the pod; nothing of theirs has been read)
  r0  the card, verbatim: a cell is a cost at >= 3 of 4 seeds with one per init.  Seeds {0, 1, 10, 11}; init 0 (M2b_s0) = {0, 10}, init 1 (M2b_s1) = {1, 11}.
  r1  cost / gain at one seed: the card's rule unchanged (S1 - C0 separated on the worse / better side AND |Delta| > F), F = |C0_rep - C0_s0| per cell at seed 0, applied to all
      four seeds (the card's one replicate pair).
  r2  COST4(cell) = cost at >= 3 of the 4 seeds AND >= 1 of them in each init.  GAIN4(cell) = gain at >= 3 of the 4 seeds AND >= 1 in each init.
  r3  outcome: no COST4 cell -> KEEPS (ADDS when a GAIN4 cell exists).  COST4 cells -> COSTS-NAV for a COST4 cell on which N1 - C0 costs at both seeds 0 and 1 (N1 was trained only
      there), COSTS-HEADS for the other COST4 cells; both labels print when both occur.
  r4  a cell that costs at exactly 2 of the 4 seeds prints as SPLIT-2of4 beside the outcome: not a cost under r0, and never hidden.
  r5  next step per outcome: the card's table (strat1_read.NEXT).
  r6  control K-RE: the seed-0 and seed-1 rows recomputed here must equal RESULT_STRAT1.json's banked rows exactly (same dumps, code and bootstrap seed); a mismatch stops the read.
Usage:  python strat1_seeds_read.py --dumps DIR --out RESULT_STRAT1_SEEDS.json --banked RESULT_STRAT1.json [--strategic S.json] [--ctx C.json] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (HERE, R):
    if p not in sys.path:
        sys.path.insert(0, p)
import seeds_run as SRUN  # noqa: E402

L = SRUN.patch_lib()
import strat1_read as S  # noqa: E402  (UNEDITED; module level is definitions only)

SEEDS = (0, 1, 10, 11)
INIT = {0: 0, 10: 0, 1: 1, 11: 1}
K_OF_4 = 3
READ_ARMS = ("C0_s0", "C0_rep", "S1_s0", "N1_s0", "C0_s1", "S1_s1", "N1_s1", "C0_s10", "S1_s10", "C0_s11", "S1_s11")


def count_rule(per_seed, k=K_OF_4):
    """per_seed: {seed: set(cells)} -> {cell: [seeds]} for the cells present at >= k seeds with >= 1 seed in each init (r2).  Pure."""
    cells = set().union(*per_seed.values()) if per_seed else set()
    out = {}
    for c in sorted(cells):
        ss = sorted(s for s, v in per_seed.items() if c in v)
        if len(ss) >= k and {INIT[s] for s in ss} == {0, 1}:
            out[c] = ss
    return out


def seeds_outcome(s1c0, n1c0):
    """s1c0: {seed: {cell: row}} at the four seeds; n1c0: {seed: {cell: row}} at seeds 0 and 1 -> (labels, detail).  Pure; the unit tests plant every row (r2-r4)."""
    if set(s1c0) != set(SEEDS):
        return ["NOT-YET"], {"reason": f"the rule needs S1 - C0 at seeds {list(SEEDS)}; have {sorted(s1c0)}"}
    costs = {s: S.cost_sets(s1c0[s]) for s in SEEDS}
    gains = {s: S.gain_sets(s1c0[s]) for s in SEEDS}
    by_cell = {c: sorted(s for s in SEEDS if c in costs[s]) for c in sorted(set().union(*costs.values()))}
    det = {"cost_seeds_by_cell": by_cell, "COST4": count_rule(costs),
           "SPLIT-2of4": sorted(c for c, ss in by_cell.items() if len(ss) == 2)}
    if not det["COST4"]:
        det["GAIN4"] = count_rule(gains)
        return (["ADDS"] if det["GAIN4"] else ["KEEPS"]), det
    labels = []
    if set(n1c0) == {0, 1}:
        n = S.cost_sets(n1c0[0]) & S.cost_sets(n1c0[1])
        det["costs_nav"] = sorted(c for c in det["COST4"] if c in n)
        det["costs_heads"] = sorted(c for c in det["COST4"] if c not in n)
        if det["costs_nav"]:
            labels.append("COSTS-NAV")
        if det["costs_heads"]:
            labels.append("COSTS-HEADS")
    else:
        labels.append("COSTS (N1 not read: NAV vs HEADS not attributable)")
    return labels, det


def k_re(pairs, banked):
    """r6: the recomputed seed-0 / seed-1 S1 - C0 and N1 - C0 rows equal the banked ones exactly -> (pass, mismatches)"""
    bad = []
    for fam, blk in banked["families"].items():
        for cell, rows in (blk.get("cells") or {}).items():
            for p in ("S1-C0_s0", "S1-C0_s1", "N1-C0_s0", "N1-C0_s1"):
                if p in rows and p in pairs:
                    for key in ("delta", "lo", "hi", "cost", "gain"):
                        a, b = pairs[p][cell].get(key), rows[p].get(key)
                        if a != b and not (isinstance(a, float) and isinstance(b, float) and np.float64(a) == np.float64(b)):
                            bad.append(f"{p} {cell} {key}: {a} vs banked {b}")
    return (not bad), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banked", required=True, help="RESULT_STRAT1.json (the s0 / s1 read, for K-RE)")
    ap.add_argument("--strategic", default=None)
    ap.add_argument("--ctx", default=None)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci()
    dpath = lambda arm: f"{a.dumps}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    missing = [arm for arm in READ_ARMS if not os.path.isdir(dpath(arm))]
    rep = {"tool": "strat1_seeds_read.py", "card": "Research/2026-10-17-v8f-mini/raw/2026-11-16-v8f-strat1/design/PREREG_V8F_STRAT1.md (frozen 2026-11-16), section 6 SEEDS-DISAGREE",
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)",
                    "STRATEGIC route": "T0 at the window's first tick", "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = the four seeds and F",
           "rules": __doc__.split("RULES")[1].split("Usage:")[0].strip(), "seeds_commit": SRUN.SEEDS_COMMIT, "seeds_commit_diff_vs_code_of_record": list(SRUN.SEEDS_DIFF),
           "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    if missing:
        rep["outcome_driving"] = ["NOT-YET"]
        rep["reason"] = f"dumps missing: {missing}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", missing)
        sys.exit(1)
    import obd_geom as G
    geom = G.build()
    dg = S.load_dump_n(dpath("C0_s0"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm in READ_ARMS:
        cells[arm] = S.load_arm_cells(geom, dg, J, glat, glon, dpath(arm), L.EVAL_N_EPISODES)
        print(f"loaded {arm} {time.time() - t0:.0f}s", flush=True)
    eid = dg["eid"]
    F = S.OR.floors_from(cells["C0_rep"], cells["C0_s0"])
    rep["floors"] = {"F": dict(F), "definition": "|C0_rep - C0_s0| per cell at seed 0, applied to all four seeds (r1)"}
    rep["controls"]["K1_self_pair"] = S.OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    pairs, s1c0, n1c0 = {}, {}, {}
    for s in SEEDS:
        pairs[f"S1-C0_s{s}"] = S.pair_rows(CI, cells[f"S1_s{s}"], cells[f"C0_s{s}"], eid, F, a.n_boot)
        s1c0[s] = pairs[f"S1-C0_s{s}"]
    for s in (0, 1):
        pairs[f"N1-C0_s{s}"] = S.pair_rows(CI, cells[f"N1_s{s}"], cells[f"C0_s{s}"], eid, F, a.n_boot)
        n1c0[s] = pairs[f"N1-C0_s{s}"]
    for s in (10, 11):
        pairs[f"C0_s{s}-C0_s{s - 10}"] = S.pair_rows(CI, cells[f"C0_s{s}"], cells[f"C0_s{s - 10}"], eid, F, a.n_boot)
    ok, bad = k_re({p: S.slim(v) for p, v in pairs.items()}, json.load(open(a.banked, encoding="utf-8")))
    rep["controls"]["K-RE"] = {"pass": ok, "mismatches": bad[:20], "n_mismatches": len(bad)}
    if not ok:
        prob.append("K-RE: the recomputed seed-0 / seed-1 rows differ from RESULT_STRAT1.json")
    labels, det = seeds_outcome(s1c0, n1c0)
    fam = S.family_blocks({p: S.slim(v) for p, v in pairs.items()})
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "estimator": "paired episode-cluster bootstrap", "n_windows": int(len(eid)),
                           "n_episodes": int(len(np.unique(eid)))} for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    strat = json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None
    ctx = json.load(open(a.ctx, encoding="utf-8")) if a.ctx else None
    banked = json.load(open(a.banked, encoding="utf-8"))
    rep["families"]["STRATEGIC"] = {"tier": "T0 at the window's first tick (route); T0 (detected info)",
                                    "seeds_10_11": {"route_following": (strat or {}).get("outcome_route_following", "NOT-YET"),
                                                    "route_setting_RC-A": (strat or {}).get("route_setting_RC-A", "NOT-YET"),
                                                    "detected_info": (ctx or {}).get("outcome", "NOT-YET")},
                                    "seeds_0_1_banked": {"route_following": banked["families"]["STRATEGIC"].get("route_following"),
                                                         "detected_info": banked["families"]["STRATEGIC"].get("detected_info")},
                                    "K9_harness_identity": "not re-run here: K9b PASS on the full grid at S1_s0 and S1_s1 (k9diag/, same harness and tree)"}
    rep["outcome_driving"], rep["outcome_driving_detail"] = labels, det
    rep["outcome_driving_s0_s1_banked"] = banked.get("outcome_driving")
    rep["next_step"] = S.next_steps(labels, False)
    rep["provenance"] = S.provenance(a.dumps, READ_ARMS)
    rep["provenance"]["seeds_reader_sha256"] = {f: L.sha256_file(f"{HERE}/{f}") for f in ("strat1_seeds_read.py", "seeds_run.py", "test_seeds_read.py")}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "|", labels, "| COST4:", list(det.get("COST4", {})), "| SPLIT-2of4:", det.get("SPLIT-2of4"), "| problems:", len(prob), "|", rep["wall_s"], "s")
    for n in rep["next_step"]:
        print("  NEXT:", n)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
