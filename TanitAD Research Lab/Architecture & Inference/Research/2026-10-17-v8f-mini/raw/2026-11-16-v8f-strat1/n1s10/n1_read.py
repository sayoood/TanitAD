"""n1_read.py -- the STRAT1 N1-at-seeds-10/11 read.  Frozen rules: PREREG_ADDENDUM_N1S10.md (written 2026-11-16 ~22:45Z, before either arm was trained; nothing below changes a rule).
Same reader and constants as the SEEDS read: the 24 driving cells of strat1_read.py (imported UNEDITED), F = |C0_rep - C0_s0| per cell, cost / gain as the card (separated on the worse / better side AND |Delta| > F).
Tier of the 24 cells: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner), 141 episodes / 2,399 windows per seed.  Evidence class: MEASURED (ours; dumps and T0, zero training).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 (another draw of EPISODES only; training-run variance = the four seeds and F).
RULES (addendum, verbatim in substance)
  seeds {0, 1, 10, 11}; init 0 (M2b_s0) = {0, 10}, init 1 (M2b_s1) = {1, 11}.
  NAV4(cell)    N1 - C0 costs at >= 3 of the 4 seeds, with >= 1 seed in each init.
  HEADS4(cell)  S1 - N1 costs at >= 3 of the 4 seeds, with >= 1 seed in each init.
  COST4(cell)   S1 - C0 costs at >= 3 of the 4 seeds, with >= 1 in each init (the SEEDS read rule r2).
  attribution of each COST4 cell:  NAV4 and not HEADS4 -> NAV (the nav text; the card COSTS-NAV next step applies in addition);  HEADS4 and not NAV4 -> HEADS (COSTS-HEADS stands, S1d decides);
                                   both -> BOTH (both next steps);  neither -> NEITHER (not attributable at 4 seeds; reported as written, S1d read as the card says).
  beside, no bar: C0_s10 - C0_s0 and C0_s11 - C0_s1 per cell (a second replicate pair; H-ESTIM-SEED-1).
  K-RE: every recomputed pair that also exists in a banked result (RESULT_STRAT1.json for seeds 0 / 1, RESULT_STRAT1_SEEDS.json for seeds 10 / 11) equals it exactly; at least one pair must be compared per banked file.
Usage:  python n1_read.py --dumps DIR --out RESULT_STRAT1_N1S10.json --banked RESULT_STRAT1.json --seeds-result RESULT_STRAT1_SEEDS.json [--strategic S.json] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (HERE, os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import n1_run as NRUN  # noqa: E402

L = NRUN.patch_lib()
import strat1_read as S  # noqa: E402  (UNEDITED)
import strat1_seeds_read as SS  # noqa: E402  (UNEDITED; count_rule, SEEDS, INIT)

SEEDS = SS.SEEDS
READ_ARMS = ("C0_s0", "C0_rep", "C0_s1", "C0_s10", "C0_s11", "S1_s0", "S1_s1", "S1_s10", "S1_s11", "N1_s0", "N1_s1", "N1_s10", "N1_s11")
KEYS = ("delta", "lo", "hi", "cost", "gain")


def attribution(cost4, nav4, heads4):
    """cost4 / nav4 / heads4: {cell: [seeds]} -> {cell: 'NAV' | 'HEADS' | 'BOTH' | 'NEITHER'} for each COST4 cell (addendum table).  Pure."""
    out = {}
    for c in cost4:
        n, h = c in nav4, c in heads4
        out[c] = "BOTH" if (n and h) else "NAV" if n else "HEADS" if h else "NEITHER"
    return out


def n1_outcome(s1c0, n1c0, s1n1, rule=None):
    """each arg {seed: {cell: row with cost / gain}} at the four seeds -> (attribution {cell: label}, detail).  Pure; the unit tests plant every row.
    rule: the counting function (default SS.count_rule); the tests swap it for mutated variants."""
    rule = rule or SS.count_rule
    for name, d in (("S1 - C0", s1c0), ("N1 - C0", n1c0), ("S1 - N1", s1n1)):
        if set(d) != set(SEEDS):
            return {}, {"status": "NOT-YET", "reason": f"the rule needs {name} at seeds {list(SEEDS)}; have {sorted(d)}"}
    cost4 = rule({s: S.cost_sets(s1c0[s]) for s in SEEDS})
    nav4 = rule({s: S.cost_sets(n1c0[s]) for s in SEEDS})
    heads4 = rule({s: S.cost_sets(s1n1[s]) for s in SEEDS})
    by = lambda d: {c: sorted(s for s in SEEDS if c in S.cost_sets(d[s])) for c in sorted(set().union(*(S.cost_sets(d[s]) for s in SEEDS)))}  # noqa: E731
    det = {"status": "OK", "COST4": cost4, "NAV4": nav4, "HEADS4": heads4, "N1-C0_cost_seeds_by_cell": by(n1c0), "S1-N1_cost_seeds_by_cell": by(s1n1)}
    return attribution(cost4, nav4, heads4), det


NEXT_BY_LABEL = {
    "NAV": "the cost is the nav text's: " + S.NEXT["COSTS-NAV"] + " (in addition to S1d)",
    "HEADS": "COSTS-HEADS stands: " + S.NEXT["COSTS-HEADS"] + " (S1d decides)",
    "BOTH": "both next steps: " + S.NEXT["COSTS-NAV"] + "; and " + S.NEXT["COSTS-HEADS"],
    "NEITHER": "not attributable at 4 seeds; reported as written, and S1d is read as the card says: " + S.NEXT["COSTS-HEADS"],
}


def k_re(pairs, banked_list):
    """pairs: {pair_name: {cell: row}}; banked_list: [(name, banked_json)] -> (pass, mismatches, n_compared per file).
    A file with zero compared rows fails (a vacuous pass is not a pass)."""
    bad, counts = [], {}
    for name, banked in banked_list:
        n = 0
        for fam, blk in banked["families"].items():
            for cell, rows in (blk.get("cells") or {}).items():
                for p, row in rows.items():
                    if p in pairs and cell in pairs[p]:
                        n += 1
                        for key in KEYS:
                            a, b = pairs[p][cell].get(key), row.get(key)
                            if a != b and not (isinstance(a, float) and isinstance(b, float) and np.float64(a) == np.float64(b)):
                                bad.append(f"{name} {p} {cell} {key}: {a} vs banked {b}")
        counts[name] = n
        if n == 0:
            bad.append(f"{name}: no recomputed pair matched a banked row (vacuous control)")
    return (not bad), bad, counts


def md_table(fam_cells, pair_names):
    lines = ["| cell | " + " | ".join(pair_names) + " |", "|---|" + "---|" * len(pair_names)]
    for c, rows in fam_cells.items():
        def f(r):
            if r.get("status") != "OK":
                return str(r.get("status"))
            return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" + (" COST" if r.get("cost") else "") + (" GAIN" if r.get("gain") else "")
        lines.append(f"| {c} | " + " | ".join(f(rows[p]) if p in rows else "" for p in pair_names) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banked", required=True)
    ap.add_argument("--seeds-result", required=True)
    ap.add_argument("--strategic", default=None)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci()
    dpath = lambda arm: f"{a.dumps}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    missing = [arm for arm in READ_ARMS if not os.path.isdir(dpath(arm))]
    rep = {"tool": "n1_read.py", "addendum": "PREREG_ADDENDUM_N1S10.md (frozen 2026-11-16 ~22:45Z)", "card": "PREREG_V8F_STRAT1.md section 6 (frozen 2026-11-16)",
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)",
                    "STRATEGIC route": "T0 at the window's first tick", "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); another draw of EPISODES only; training-run variance = the four seeds and F",
           "rules": __doc__.split("RULES")[1].split("Usage:")[0].strip(), "n1_commit": NRUN.SRUN.SEEDS_COMMIT, "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    if missing:
        rep["status"] = "NOT-YET"
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
    rep["floors"] = {"F": dict(F), "definition": "|C0_rep - C0_s0| per cell at seed 0, applied to all four seeds"}
    rep["controls"]["K1_self_pair"] = S.OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    pairs, s1c0, n1c0, s1n1 = {}, {}, {}, {}
    for s in SEEDS:
        for name, x, y, dest in ((f"S1-C0_s{s}", f"S1_s{s}", f"C0_s{s}", s1c0), (f"N1-C0_s{s}", f"N1_s{s}", f"C0_s{s}", n1c0), (f"S1-N1_s{s}", f"S1_s{s}", f"N1_s{s}", s1n1)):
            pairs[name] = S.pair_rows(CI, cells[x], cells[y], eid, F, a.n_boot)
            dest[s] = pairs[name]
    for s in (10, 11):
        pairs[f"C0_s{s}-C0_s{s - 10}"] = S.pair_rows(CI, cells[f"C0_s{s}"], cells[f"C0_s{s - 10}"], eid, F, a.n_boot)
    slim = {p: S.slim(v) for p, v in pairs.items()}
    banked = [("RESULT_STRAT1.json", json.load(open(a.banked, encoding="utf-8"))), ("RESULT_STRAT1_SEEDS.json", json.load(open(a.seeds_result, encoding="utf-8")))]
    ok, bad, counts = k_re(slim, banked)
    rep["controls"]["K-RE"] = {"pass": ok, "mismatches": bad[:20], "n_mismatches": len(bad), "rows_compared": counts}
    if not ok:
        prob.append("K-RE: a recomputed row differs from a banked row (or none was compared)")
    attr, det = n1_outcome(s1c0, n1c0, s1n1)
    rep["attribution"], rep["attribution_detail"] = attr, det
    rep["next_step"] = {c: NEXT_BY_LABEL[lab] for c, lab in attr.items()} or ["no COST4 cell: nothing to attribute"]
    names = [f"{k}_s{s}" for k in ("N1-C0", "S1-N1", "S1-C0") for s in SEEDS] + ["C0_s10-C0_s0", "C0_s11-C0_s1"]
    fam = S.family_blocks({p: slim[p] for p in names})
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "estimator": "paired episode-cluster bootstrap", "n_windows": int(len(eid)),
                           "n_episodes": int(len(np.unique(eid)))} for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    rep["beside_no_bar"] = {"C0_s10-C0_s0_TAC-LAT": slim["C0_s10-C0_s0"].get("TAC-LAT"), "C0_s11-C0_s1_TAC-LAT": slim["C0_s11-C0_s1"].get("TAC-LAT"),
                            "F_TAC-LAT": F.get("TAC-LAT"), "note": "a second replicate pair; the single-pair floor F understates training-run variance on this cell (H-ESTIM-SEED-1); the next STRAT1-family card sets F from at least two replicate pairs"}
    strat = json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None
    st = {"tier": "T0 at the window's first tick (route); T0 (detected info)",
          "route_following_N1_and_S1_per_seed": {f"{k}_s{s}": ((strat or {}).get("arms", {}).get(f"{k}_s{s}", {}).get("block", "NOT-YET")) for k in ("N1", "S1") for s in SEEDS},
          "route_following_note": "the card label (S-FOLLOWS / S-PARTIAL / S-IGNORES) is defined on S1 at two seeds; N1 blocks are read beside it (real, donor, zero nav at T0). The label strat1_strategic.py prints over four seeds is not the card two-seed label and is not quoted here.",
          "route_setting_RC-A_S1-C0": (strat or {}).get("route_setting_RC-A", "NOT-YET"),
          "detected_info": {"N1": "not computable: N1 has no context heads (cfg.ctx_heads off); n = 0 detected-info outputs. S1 detected info at seeds 0 / 1 / 10 / 11 is in RESULT_STRAT1.json and RESULT_STRAT1_SEEDS.json."}}
    rep["families"]["STRATEGIC"] = st
    rep["provenance"] = S.provenance(a.dumps, READ_ARMS)
    rep["provenance"]["frozen_reader_sha256"] = {f: L.sha256_file(os.path.join(R, "seeds", f)) for f in ("strat1_seeds_read.py", "seeds_run.py")}
    rep["provenance"]["n1_reader_sha256"] = {f: L.sha256_file(f"{HERE}/{f}") for f in ("n1_read.py", "n1_run.py", "test_n1_read.py", "after_n1s10.sh", "launch_n1_read.sh")}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = [f"# STRAT1 N1 at seeds 10 / 11 (MEASURED, T1-kin; 141 episodes / 2,399 windows per seed; paired episode-cluster bootstrap B {a.n_boot} seed 0)", "",
          f"Addendum rules: NAV4 / HEADS4 / COST4 as frozen. Problems: {prob or 'none'}. K-RE rows compared: {counts}.", "",
          "## Attribution of COST4 cells", "", "```", json.dumps({"COST4": det.get("COST4"), "NAV4": det.get("NAV4"), "HEADS4": det.get("HEADS4"), "attribution": attr}, indent=1), "```", "",
          "## Next step", ""] + [f"* {c}: {t}" for c, t in (rep["next_step"].items() if isinstance(rep["next_step"], dict) else enumerate(rep["next_step"]))] + \
         ["", "## Beside, no bar", "", "```", json.dumps(rep["beside_no_bar"], indent=1, default=float), "```", ""]
    for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        if f in rep["families"]:
            md += [f"## {f} (T1-kin)", "", md_table(rep["families"][f]["cells"], names), ""]
    md += ["## STRATEGIC (T0)", "", "See the JSON: route following of N1 and S1 per seed, route setting RC-A, and the N1 detected-info reason (no context heads)."]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "| attribution:", attr, "| COST4:", list(det.get("COST4", {})), "| problems:", len(prob), "|", rep["wall_s"], "s")
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
