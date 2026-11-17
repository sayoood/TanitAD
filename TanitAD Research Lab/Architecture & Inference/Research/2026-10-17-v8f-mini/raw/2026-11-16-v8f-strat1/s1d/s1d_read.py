"""s1d_read.py -- the STRAT1 S1d read (S1 with --ctx-detach on, seeds 0 / 1 / 10 / 11).  Card PREREG_V8F_STRAT1 section 6, COSTS-HEADS row, verbatim: arm S1d = S1 with --ctx-detach on (built, default off),
same seeds and reads.  The card names the arm and the reads and no outcome rule for S1d.  The rules below are PROPOSED by the builder (PREREG_ADDENDUM_S1D_READ.md) for the MM to freeze BEFORE any S1d number
exists; the reader prints this docstring into its result so the version that ran is on record.  Same reader and constants as the SEEDS and N1 reads: the 24 driving cells of strat1_read.py (UNEDITED),
F = |C0_rep - C0_s0| per cell, cost / gain as the card.  Tier: T1-kin, 141 episodes / 2,399 windows per seed.  Evidence class: MEASURED (zero training).  Estimator: paired episode-cluster bootstrap B 2,000 seed 0
(another draw of EPISODES only; training-run variance = the four seeds and F).
RULES (proposed)
  seeds {0, 1, 10, 11}; init 0 = {0, 10}, init 1 = {1, 11}; the count rule is the SEEDS read rule r2 (>= 3 of the 4 seeds, >= 1 in each init).
  COST4_S1(cell)  S1 - C0 costs (count rule).   COST4_S1d(cell)  S1d - C0 costs (count rule).   GAIN4_DET(cell)  S1d - S1 gains, i.e. S1d is separated better than S1 and |Delta| > F (count rule).
  For each COST4_S1 cell:  CURED = not COST4_S1d and GAIN4_DET;  PARTIAL = not COST4_S1d and not GAIN4_DET;  REMAINS = COST4_S1d (the cost is not carried by the ctx-loss gradient into the adapter).
  NEW-COST = a COST4_S1d cell that is not a COST4_S1 cell (listed, never hidden).   GAIN4 of S1d - C0 is listed.
  next step (PROPOSED): CURED -> S1d replaces S1 as the route-text arm and the card KEEPS / ADDS next step (N2 card) is read on S1d;  PARTIAL -> reported as written, S1d is the cost-free arm but the detach gain is not
  separated: S1d and the N1 attribution are read together;  REMAINS -> reported as written; the cost is not the ctx-loss gradient, the next lever is named from the N1 read attribution (NAV4 -> the N2 card).
  STRATEGIC (card sections 4.4 / 4.5, read on S1d at the pairs {0, 1} and {10, 11}): route following label, route setting RC-A, detected info per variable with K4; a variable DETECTS in S1 and not in S1d is listed as DETECT-LOST.
  K-RE: every recomputed pair that exists in a banked result (RESULT_STRAT1.json, RESULT_STRAT1_SEEDS.json, RESULT_STRAT1_N1S10.json when present) equals it exactly; each file must have >= 1 compared row.
Usage:  python s1d_read.py --dumps DIR --out RESULT_STRAT1_S1D.json --banked A.json B.json [C.json] [--strategic-01 X --strategic-1011 Y --ctx-01 P --ctx-1011 Q] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (HERE, os.path.join(R, "n1s10"), os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import s1d_run as SRUN  # noqa: E402

L = SRUN.patch_lib()
import strat1_read as S  # noqa: E402  (UNEDITED)
import strat1_seeds_read as SS  # noqa: E402  (UNEDITED)
import n1_read as N  # noqa: E402  (this study family; k_re, md_table)

SEEDS = SS.SEEDS
READ_ARMS = ("C0_s0", "C0_rep", "C0_s1", "C0_s10", "C0_s11", "S1_s0", "S1_s1", "S1_s10", "S1_s11", "N1_s0", "N1_s1", "N1_s10", "N1_s11") + SRUN.S1D
NEXT_BY_LABEL = {
    "CURED": "PROPOSED: S1d replaces S1 as the route-text arm; the card KEEPS / ADDS next step (the N2 card) is read on S1d",
    "PARTIAL": "PROPOSED: reported as written; the S1 cost is gone under S1d but the detach gain over S1 is not separated; S1d and the N1 read attribution are read together",
    "REMAINS": "PROPOSED: reported as written; the cost is not carried by the ctx-loss gradient into the adapter; the next lever is named from the N1 read attribution (NAV4 -> the N2 card)",
}


def s1d_outcome(s1c0, s1dc0, s1ds1, rule=None):
    """each arg {seed: {cell: row}} at the four seeds -> (labels {cell: CURED | PARTIAL | REMAINS}, detail).  Pure; the unit tests plant every row."""
    rule = rule or SS.count_rule
    for name, d in (("S1 - C0", s1c0), ("S1d - C0", s1dc0), ("S1d - S1", s1ds1)):
        if set(d) != set(SEEDS):
            return {}, {"status": "NOT-YET", "reason": f"the rule needs {name} at seeds {list(SEEDS)}; have {sorted(d)}"}
    cost_s1 = rule({s: S.cost_sets(s1c0[s]) for s in SEEDS})
    cost_s1d = rule({s: S.cost_sets(s1dc0[s]) for s in SEEDS})
    gain_det = rule({s: S.gain_sets(s1ds1[s]) for s in SEEDS})
    labels = {}
    for c in cost_s1:
        labels[c] = "REMAINS" if c in cost_s1d else ("CURED" if c in gain_det else "PARTIAL")
    det = {"status": "OK", "COST4_S1": cost_s1, "COST4_S1d": cost_s1d, "GAIN4_DET": gain_det, "NEW-COST": {c: v for c, v in cost_s1d.items() if c not in cost_s1},
           "GAIN4_S1d_vs_C0": rule({s: S.gain_sets(s1dc0[s]) for s in SEEDS})}
    return labels, det


def detect_lost(ctx_s1_by_pair, ctx_s1d_by_pair):
    """{pair: {variable: label}} for S1 and for S1d -> {pair: [variables that DETECT in S1 and not in S1d]}.  Pure."""
    out = {}
    for pair, s1 in ctx_s1_by_pair.items():
        d = ctx_s1d_by_pair.get(pair, {})
        out[pair] = sorted(v for v, lab in s1.items() if str(lab).startswith("DETECTS") and not str(d.get(v, "")).startswith("DETECTS"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banked", nargs="+", required=True)
    ap.add_argument("--strategic-01", default=None)
    ap.add_argument("--strategic-1011", default=None)
    ap.add_argument("--ctx-01", default=None)
    ap.add_argument("--ctx-1011", default=None)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci()
    dpath = lambda arm: f"{a.dumps}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    missing = [arm for arm in READ_ARMS if not os.path.isdir(dpath(arm))]
    rep = {"tool": "s1d_read.py", "card": "PREREG_V8F_STRAT1.md section 6, COSTS-HEADS row (frozen 2026-11-16); outcome rules PROPOSED, see PREREG_ADDENDUM_S1D_READ.md",
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)",
                    "STRATEGIC route": "T0 at the window's first tick", "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); another draw of EPISODES only; training-run variance = the four seeds and F",
           "rules": __doc__.split("RULES (proposed)")[1].split("Usage:")[0].strip(), "s1d_commit": SRUN.S1D_COMMIT, "s1d_diff_vs_code_of_record": list(SRUN.S1D_DIFF_VS_RECORD),
           "controls": {"problems": []}}
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
    pairs, s1c0, s1dc0, s1ds1 = {}, {}, {}, {}
    for s in SEEDS:
        for name, x, y, dest in ((f"S1d-C0_s{s}", f"S1d_s{s}", f"C0_s{s}", s1dc0), (f"S1d-S1_s{s}", f"S1d_s{s}", f"S1_s{s}", s1ds1), (f"S1d-N1_s{s}", f"S1d_s{s}", f"N1_s{s}", {}),
                                 (f"S1-C0_s{s}", f"S1_s{s}", f"C0_s{s}", s1c0), (f"N1-C0_s{s}", f"N1_s{s}", f"C0_s{s}", {})):
            pairs[name] = S.pair_rows(CI, cells[x], cells[y], eid, F, a.n_boot)
            dest[s] = pairs[name]
    slim = {p: S.slim(v) for p, v in pairs.items()}
    banked = [(os.path.basename(b), json.load(open(b, encoding="utf-8"))) for b in a.banked if os.path.exists(b)]
    ok, bad, counts = N.k_re(slim, banked)
    rep["controls"]["K-RE"] = {"pass": ok, "mismatches": bad[:20], "n_mismatches": len(bad), "rows_compared": counts}
    if not ok:
        prob.append("K-RE: a recomputed row differs from a banked row (or none was compared)")
    labels, det = s1d_outcome(s1c0, s1dc0, s1ds1)
    rep["detach_outcome"], rep["detach_outcome_detail"] = labels, det
    rep["next_step"] = {c: NEXT_BY_LABEL[lab] for c, lab in labels.items()} or ["no COST4_S1 cell: nothing for the detach to cure"]
    names = [f"{k}_s{s}" for k in ("S1d-C0", "S1d-S1", "S1d-N1", "S1-C0") for s in SEEDS]
    fam = S.family_blocks({p: slim[p] for p in names})
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "estimator": "paired episode-cluster bootstrap", "n_windows": int(len(eid)),
                           "n_episodes": int(len(np.unique(eid)))} for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    ld = lambda p: json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None  # noqa: E731
    st = {"tier": "T0 at the window's first tick (route); T0 (detected info)", "S1d_route_following": {}, "S1d_route_setting_RC-A": {}, "S1d_detected_info": {}, "S1_detected_info_beside": {}}
    for pair, sp, cp in (("0_1", a.strategic_01, a.ctx_01), ("10_11", a.strategic_1011, a.ctx_1011)):
        sj, cj = ld(sp), ld(cp)
        st["S1d_route_following"][pair] = (sj or {}).get("outcome_route_following", "NOT-YET")
        st["S1d_route_setting_RC-A"][pair] = (sj or {}).get("route_setting_RC-A", "NOT-YET")
        st["S1d_detected_info"][pair] = (cj or {}).get("outcome", "NOT-YET")
    banked_by = dict(banked)
    s1_b = banked_by.get("RESULT_STRAT1.json", {}).get("families", {}).get("STRATEGIC", {}).get("detected_info")
    s1_s = ((banked_by.get("RESULT_STRAT1_SEEDS.json", {}).get("families", {}).get("STRATEGIC", {}).get("seeds_10_11") or {}).get("detected_info"))
    st["S1_detected_info_beside"] = {"0_1": s1_b, "10_11": s1_s}
    s1d_det = {k: (v if isinstance(v, dict) else {}) for k, v in st["S1d_detected_info"].items()}
    st["DETECT-LOST"] = detect_lost({"0_1": s1_b or {}, "10_11": s1_s or {}}, s1d_det)
    rep["families"]["STRATEGIC"] = st
    rep["provenance"] = S.provenance(a.dumps, READ_ARMS)
    rep["provenance"]["frozen_reader_sha256"] = {f: L.sha256_file(os.path.join(R, "seeds", f)) for f in ("strat1_seeds_read.py", "seeds_run.py")}
    rep["provenance"]["study_reader_sha256"] = {f: L.sha256_file(f"{HERE}/{f}") for f in ("s1d_read.py", "s1d_run.py", "s1d_view.py", "test_s1d_read.py", "after_s1d.sh", "launch_s1d_read.sh")}
    rep["provenance"]["n1_reader_sha256"] = {f: L.sha256_file(os.path.join(R, "n1s10", f)) for f in ("n1_read.py", "n1_run.py")}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = [f"# STRAT1 S1d (S1 with --ctx-detach on) at seeds 0 / 1 / 10 / 11 (MEASURED, T1-kin; 141 episodes / 2,399 windows per seed; B {a.n_boot} seed 0)", "",
          f"Outcome rules are PROPOSED (PREREG_ADDENDUM_S1D_READ.md). Problems: {prob or 'none'}. K-RE rows compared: {counts}.", "", "## Detach outcome per S1 COST4 cell", "", "```",
          json.dumps({"labels": labels, "detail": det}, indent=1), "```", "", "## Next step", ""] + [f"* {c}: {t}" for c, t in (rep["next_step"].items() if isinstance(rep["next_step"], dict) else enumerate(rep["next_step"]))] + [""]
    for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        if f in rep["families"]:
            md += [f"## {f} (T1-kin)", "", N.md_table(rep["families"][f]["cells"], names), ""]
    md += ["## STRATEGIC (T0)", "", "See the JSON: route following, route setting RC-A, detected info with K4, and DETECT-LOST per seed pair."]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "| detach:", labels, "| COST4_S1d:", list(det.get("COST4_S1d", {})), "| problems:", len(prob), "|", rep["wall_s"], "s")
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
