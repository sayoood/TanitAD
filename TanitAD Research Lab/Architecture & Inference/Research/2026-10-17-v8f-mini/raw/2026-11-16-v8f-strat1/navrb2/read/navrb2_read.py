"""navrb2_read.py -- the PREREG_V8F_NAVRB2 read (card FROZEN, sha256 e238ba8d; sections 5, 6, 7).  CPU, zero training.  Evidence class of every number: MEASURED (ours; dumps and T0 evals).
Tiers: LONGITUDINAL / LATERAL / TACTICAL = T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav with nav_rb held from t0, logged scene; 141 episodes / 2,399 windows per arm);
STRATEGIC = T0 at the window's first tick (read by the frozen STRAT1 readers on a view, see launch_navrb2_read.sh).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0:
'another draw of EPISODES' only; 'another TRAINING run' is the floor F2 = max(F, F_rep) (card 5.3; F from C0_rep - C0_s0, F_rep from the two C0REP same-seed pairs); 'another INFERENCE run' does not apply
(deterministic planner, one environment, one thread count).  Optimistic by construction (card 3): the route is supplied along the driven path.
COMMITTED RULES (card section 5, 6, 7 verbatim in force; the reading choices r1-r5 in navrb2_lib.READING were written before any number)
  D1 (deciding): TAC-LAT accuracy on the RB windows (235 windows / 16 episodes), R arm - banked arm, paired, per seed; gain = separated better AND |delta| > F2_rb (F2 recomputed on RB from the same replicate pairs).
  D2 (beside): TURN recall on RB-TURN (43 windows / 6 episodes, descriptive) and the false-turn share on RB minus RB-TURN.  All 24 cells on RB: beside, no bar.
  Guard: the 24 all-window cells, N1R - N1 and S1R - S1, cost / gain by the STRAT1 rule under F2.  STRATEGIC in full by the frozen readers (+ R - banked contrasts in navrb2_strategic.py).
  Z-EVAL: real - zero (the R checkpoint dumped with the channel zeroed) on the windows whose nav_rb frames are all 0 must read EXACTLY 0.0 on every cell.  Z-USE (beside): real - zero on RB.
  Outcomes per pair at seeds 0 and 1: RB-CLOSES / RB-NAV-ONLY / RB-SEEDS-DISAGREE / RB-READ-NO-GAIN / RB-UNREAD / RB-COSTS (each prints its next lever).
Usage:  python navrb2_read.py --out RESULT_NAVRB2.json [--md RESULT_NAVRB2.md] [--t1 DIR] [--strategic J --ctx J --ctx-nofog J --contrast J] [--smoke] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

for _p in (NL.SR, NL.SR + "/theory/obd/read", NL.SR + "/theory/otband/read"):
    sys.path.insert(0, _p)
import strat1_lib as L  # noqa: E402
import strat1_read as S  # noqa: E402  (frozen, UNEDITED)
import obd_read as OR  # noqa: E402

CELLS, ACC = OR.CELLS, OR.ACCURACY
DECIDING = "TAC-LAT"
PAIRS = (("N1R", "N1"), ("S1R", "S1"))
NEXT = {
    "RB-CLOSES": "the channel enters the route-text recipe (S1R replaces S1 as the route-text arm); next lever on the remaining RB gap vs C0 (v9 recall): the RB-DIST card (exit number + countdown distance channels)",
    "RB-NAV-ONLY": "S1R's heads compete with the channel through the shared adapter: S1dR at s0 / s1 (detached heads, 2 arms, PI go on cost)",
    "RB-SEEDS-DISAGREE": "E4 (card 4.2): N1R / S1R at s10 / s11 (PI go on cost); COST4 / GAIN4 under F2",
    "RB-READ-NO-GAIN": "the state is read but carries no side: RB-DIST card (exit number, countdown distance), zero-training first: the navecho2 nav-only classifier with the state + exit channels",
    "RB-UNREAD": "dose: the same arms with --nav-rb-lr-mult 30 at s0 (2 arms, PI go on cost); first, zero-training, the Adam-state reachability gate on e_nav_rb.weight (zerograd_gate.py) and its final norm",
    "RB-COSTS": "not adopted; N1Z_s0 (card 4.2) separates the channel from training noise before any other arm",
    "NOT-YET": "an input is missing",
}


# ------------------------------------------------------------------------------------------------------------------------------ pure logic (unit-tested; the tests plant every row)
def floor2(F, Frep):
    """F2 = max(F, F_rep) per cell; a cell with no card floor (None) keeps None; a missing F_rep counts as 0 (the C0REP rule)"""
    return {c: (None if F.get(c) is None else float(max(F[c], (Frep or {}).get(c) or 0.0))) for c in CELLS}


def rb_floors(cells, mask):
    """-> (F_rb, F_rep_rb, F2_rb) for TAC-LAT on the RB windows from the C0 cells: F = |C0_rep - C0_s0|, F_rep = max(|C0_s10r - C0_s10|, |C0_s11r - C0_s11|), the same estimator as the all-window floors."""
    f = OR.floor_of(cells["C0_rep"][DECIDING][mask], cells["C0_s0"][DECIDING][mask])
    reps = [OR.floor_of(cells[r][DECIDING][mask], cells[o][DECIDING][mask]) for r, o in (("C0_s10r", "C0_s10"), ("C0_s11r", "C0_s11"))]
    reps = [x for x in reps if x is not None]
    frep = max(reps) if reps else None
    f2 = None if f is None else float(max(f, frep or 0.0))
    return f, frep, f2


def d1_row(CI, arm, ref, eid, mask, floor, nb=NL.NB):
    """D1: TAC-LAT accuracy on the stratum, arm - ref, paired episode-cluster bootstrap; gain / cost by the card rule with the floor"""
    m = np.asarray(mask, bool)
    return OR.cell_delta(CI, np.asarray(arm[DECIDING])[m], np.asarray(ref[DECIDING])[m], np.asarray(eid)[m], True, floor, nb)


def d2_rows(CI, arm, arm_pred, ref, ref_pred, eid, turn, rb, nb=NL.NB):
    """D2: TURN recall on RB-TURN (share where TAC-LAT is correct) and the false-turn share on RB minus RB-TURN (share of windows whose predicted lateral decision is a turn); arm, ref and arm - ref paired"""
    turn, rb = np.asarray(turn, bool), np.asarray(rb, bool)
    non = rb & ~turn
    out = {"turn": {"n_windows": int(turn.sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[turn])))},
           "non_turn_rb": {"n_windows": int(non.sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[non])))}}
    if turn.any():
        a, b = np.asarray(arm[DECIDING])[turn], np.asarray(ref[DECIDING])[turn]
        out["turn_recall"] = {"arm": float(a.mean()), "ref": float(b.mean()), "paired": L.paired(CI, a, b, np.asarray(eid)[turn], nb)}
    if non.any():
        a = np.isin(np.asarray(arm_pred)[non], (1, 2)).astype(np.float64)
        b = np.isin(np.asarray(ref_pred)[non], (1, 2)).astype(np.float64)
        out["false_turn_share"] = {"arm": float(a.mean()), "ref": float(b.mean()), "paired": L.paired(CI, a, b, np.asarray(eid)[non], nb)}
    return out


def z_eval(real, zero, mask):
    """Z-EVAL: real - zero on the windows in `mask` (no nav_rb frame on in the item's history) must be EXACTLY 0.0 on every cell, window by window (NaN equal to NaN).  Pure."""
    m = np.asarray(mask, bool)
    cells, ok = {}, bool(m.any())
    for c in CELLS:
        a, b = np.asarray(real[c], np.float64)[m], np.asarray(zero[c], np.float64)[m]
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        both = np.isfinite(a) & np.isfinite(b)
        mx = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        cells[c] = {"max_abs_window_diff": mx, "nan_mismatch": nan_mismatch, "mean_diff": float((a[both] - b[both]).mean()) if both.any() else 0.0}
        ok = ok and mx == 0.0 and nan_mismatch == 0
    return {"pass": bool(ok), "n_windows": int(m.sum()), "must_read": "exactly 0.0 on every cell", "cells": cells}


def z_use(CI, real, zero, eid, rb, nb=NL.NB):
    """Z-USE (beside): real - zero on the RB windows, all 24 cells, paired.  'separated' (reading r1) = at least one cell separated, either side.  TAC-LAT alone is printed beside."""
    rows = OR.compare_cells(CI, {c: np.asarray(real[c], np.float64)[np.asarray(rb, bool)] for c in CELLS}, {c: np.asarray(zero[c], np.float64)[np.asarray(rb, bool)] for c in CELLS},
                            np.asarray(eid)[np.asarray(rb, bool)], None, nb)
    sep = sorted(c for c, r in rows.items() if r.get("status") == "OK" and r.get("separated"))
    return {"separated": bool(sep), "separated_cells": sep, "tac_lat": {k: rows[DECIDING].get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")}, "rows": S.slim(rows)}


def cost_cells(rows):
    return {c for c, v in rows.items() if v.get("cost")}


def gain_cells(rows):
    return {c for c, v in rows.items() if v.get("gain")}


def pair_labels(pair, d1, guard, zuse, route_cost=False):
    """card section 7 for ONE pair ('N1R-N1' / 'S1R-S1'): d1 {seed: row}, guard {seed: set of cost cells}, zuse {seed: bool} at seeds 0 and 1 -> (labels [...], detail).  RB-NAV-ONLY needs both pairs: see outcomes().  Pure."""
    if set(d1) != {0, 1} or set(guard) != {0, 1} or set(zuse) != {0, 1}:
        return ["NOT-YET"], {"reason": "the card needs both seeds"}
    gain = [bool(d1[s].get("gain")) for s in (0, 1)]
    cost = [bool(d1[s].get("cost")) for s in (0, 1)]
    both_cost, one_cost = sorted(guard[0] & guard[1]), sorted(guard[0] ^ guard[1])
    guard_ok = not (guard[0] or guard[1]) and not route_cost
    labels = []
    if all(cost) or both_cost:
        labels.append("RB-COSTS")
    if sum(gain) == 1 or one_cost:
        labels.append("RB-SEEDS-DISAGREE")
    if not any(zuse.values()):
        labels.append("RB-UNREAD")
    if all(zuse.values()) and not any(gain):
        labels.append("RB-READ-NO-GAIN")
    if pair == "S1R-S1" and all(gain) and guard_ok and all(zuse.values()):
        labels.append("RB-CLOSES")
    det = {"d1_gain_by_seed": gain, "d1_cost_by_seed": cost, "guard_cost_cells_both_seeds": both_cost, "guard_cost_cells_one_seed": one_cost, "guard_holds": guard_ok,
           "route_cost": bool(route_cost), "z_use_separated_by_seed": [bool(zuse[0]), bool(zuse[1])]}
    return labels, det


def outcomes(per_pair):
    """per_pair {pair: (labels, detail)} -> {"per_pair": {...}, "cross": [RB-NAV-ONLY?]}"""
    n, s = per_pair.get("N1R-N1"), per_pair.get("S1R-S1")
    cross = []
    if n and s and "NOT-YET" not in n[0] + s[0]:
        if all(n[1]["d1_gain_by_seed"]) and not all(s[1]["d1_gain_by_seed"]):
            cross.append("RB-NAV-ONLY")
    return {"per_pair": {p: {"labels": v[0], "detail": v[1]} for p, v in per_pair.items()}, "cross_pair": cross}


def k_re(slim_pairs, banked):
    """K-RE: every recomputed banked pair row equals RESULT_STRAT1.json exactly; at least one row compared (a vacuous compare is a failure).  slim_pairs {pair: {cell: slim row}}; banked = RESULT_STRAT1.json dict"""
    bad, n = [], 0
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY"):
        for cell, rows in (banked.get("families", {}).get(fam, {}).get("cells") or {}).items():
            for pair, row in rows.items():
                if pair in slim_pairs and cell in slim_pairs[pair]:
                    n += 1
                    mine = slim_pairs[pair][cell]
                    for k, v in row.items():
                        if k in mine and mine[k] != v:
                            bad.append(f"{pair} {cell} {k}: {mine[k]!r} vs {v!r}")
    return (n > 0 and not bad), bad, n


# ------------------------------------------------------------------------------------------------------------------------------ driver
def load_arm(geom, dg, J, glat, glon, d, expect_eps=141):
    """-> (cells {24: [N]}, pred_lat [N]) of one dump; the window / model-free arrays must equal the reference dump's (the frozen loader's assertion)"""
    x = OR.load_dump(d) if expect_eps == 141 else S.load_dump_n(d, expect_eps)
    for k in OR.SAME:
        assert np.array_equal(x[k], dg[k]), f"window or model-free array differs from the reference dump: {k} in {d}"
    A = geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"])
    return OR.to_cells(A, glat, glon), np.asarray(A["_lat"])


def md_table(rows_by_col, cells):
    cols = list(rows_by_col)
    out = ["| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in cells:
        line = []
        for k in cols:
            r = rows_by_col[k].get(c, {})
            line.append("n/a" if r.get("status") not in (None, "OK") or "delta" not in r else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" + ("*" if r.get("separated") else "") + ("G" if r.get("gain") else "") + ("C" if r.get("cost") else ""))
        out.append(f"| {c} | " + " | ".join(line) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--t1", default=NL.T1)
    ap.add_argument("--banked-t1", default=NL.BANKED_DUMPS)
    ap.add_argument("--strategic", default=None, help="frozen strat1_strategic.py output on the R view (S1R / N1R presented as S1 / N1)")
    ap.add_argument("--ctx", default=None)
    ap.add_argument("--ctx-nofog", default=None)
    ap.add_argument("--contrast", default=None, help="navrb2_strategic.py output (R - banked STRATEGIC contrasts, RC-A ROUTE-COST)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=NL.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci(NL.TREE)
    rep = {"tool": "navrb2_read.py", "card": "PREREG_V8F_NAVRB2.md (FROZEN 2026-11-17)", "card_sha256": NL.CARD_SHA256, "smoke": bool(a.smoke), "reading_choices": NL.READING,
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav with nav_rb held from t0, logged scene)", "STRATEGIC route": "T0 at the window's first tick",
                    "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = the floor F2; inference variance does not apply",
           "limit": "optimistic by construction: the route text is GPS matched to the drive's own trace (card 3); every number reads 'with the route supplied along the driven path'",
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("Usage:")[0].strip(), "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    dR = lambda arm: f"{a.t1}/dump_NAVRB2-{arm}_m10_causal_heads"  # noqa: E731
    dB = lambda arm: f"{a.banked_t1}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    banked_arms = ["C0_s0", "C0_rep", "C0_s1", "C0_s10", "C0_s10r", "C0_s11", "C0_s11r", "N1_s0", "S1_s0", "N1_s1", "S1_s1"]
    r_arms = list(NL.R_ARMS) + [x + NL.Z for x in NL.R_ARMS]
    missing = [d for d in [dB(x) for x in banked_arms] + [dR(x) for x in r_arms] if not os.path.isdir(d)]
    if missing:
        rep["outcome"], rep["reason"] = "NOT-YET", f"dumps missing: {[os.path.basename(m) for m in missing]}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", rep["reason"])
        sys.exit(1)
    import obd_geom as G
    geom = G.build()
    ne = 141
    dg = S.load_dump_n(dB("C0_s0"), ne)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    eid = dg["eid"]
    w = NL.rb_windows()
    if not (np.array_equal(w["ci"], dg["clip_index"]) and np.array_equal(w["t"], dg["ws"])):
        raise RuntimeError("the grid windows differ from the reference dump's (clip_index, ws)")
    hmax = NL.hist_of(f"{NL.CK}/N1R_s0/ckpt_last.pt")
    st = NL.strata(w, hmax)
    rb, zero_hist = st["rb"], st["zero_hist"]
    nv = json.load(open(NL.SR.rsplit("/", 1)[0] + "/navrb/out/RESULT_NAVRB.json", encoding="utf-8"))
    want_turn = {(p["episode"], p["step"]) for p in nv["per_window"]}
    turn = np.array([bool(rb[i]) and (int(eid[i]), int(dg["ws"][i])) in want_turn for i in range(len(eid))])
    rep["strata"] = {"RB": NL.count_stratum(rb, eid), "RB-TURN": NL.count_stratum(turn, eid), "RB-minus-TURN": NL.count_stratum(rb & ~turn, eid), "zero_hist_Z-EVAL": NL.count_stratum(zero_hist, eid), "hmax": hmax,
                     "all": NL.count_stratum(np.ones(len(eid), bool), eid)}
    if (rep["strata"]["RB"]["n_windows"], rep["strata"]["RB"]["n_episodes"]) != (NL.RB_EXPECT_WINDOWS, NL.RB_EXPECT_EPISODES) and not a.smoke:
        prob.append(f"RB count {rep['strata']['RB']} != the card's {NL.RB_EXPECT_WINDOWS} / {NL.RB_EXPECT_EPISODES}")
    if (rep["strata"]["RB-TURN"]["n_windows"], rep["strata"]["RB-TURN"]["n_episodes"]) != (NL.RB_TURN_EXPECT_WINDOWS, NL.RB_TURN_EXPECT_EPISODES) and not a.smoke:
        prob.append(f"RB-TURN count {rep['strata']['RB-TURN']} != the card's 43 / 6")
    cells, pred = {}, {}
    for arm in banked_arms:
        cells[arm], pred[arm] = load_arm(geom, dg, J, glat, glon, dB(arm), ne)
        print(f"loaded {arm} {time.time() - t0:.0f}s", flush=True)
    for arm in r_arms:
        cells[arm], pred[arm] = load_arm(geom, dg, J, glat, glon, dR(arm), ne)
        print(f"loaded {arm} {time.time() - t0:.0f}s", flush=True)
    # floors: the all-window F2 from the banked C0REP read, and F2_rb recomputed on RB
    c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
    F, Frep = c0rep["floors"]["F"], c0rep["floors"]["F_rep"]
    F2 = floor2(F, Frep)
    f_rb, frep_rb, f2_rb = rb_floors(cells, rb)
    rep["floors"] = {"F": F, "F_rep": Frep, "F2": F2, "n_pairs_used_for_F_rep": c0rep["floors"].get("n_pairs_used"), "source": "RESULT_STRAT1_C0REP.json (banked, sha256 in provenance)",
                     "RB_TAC-LAT": {"F_rb": f_rb, "F_rep_rb": frep_rb, "F2_rb": f2_rb, "definition": "the same estimator as F / F_rep, restricted to the 235 RB windows"}}
    rep["controls"]["K1_self_pair"] = OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    # K-RE: banked pairs recomputed with the STRAT1 floor F equal RESULT_STRAT1.json
    banked = json.load(open(NL.STRAT1_RESULT, encoding="utf-8"))
    F_s1 = banked["floors"]["F"]
    kre_pairs = {}
    for s in NL.SEEDS:
        for fam in ("S1", "N1"):
            kre_pairs[f"{fam}-C0_s{s}"] = S.slim(S.pair_rows(CI, cells[f"{fam}_s{s}"], cells[f"C0_s{s}"], eid, F_s1, a.n_boot))
    ok, bad, n_re = k_re(kre_pairs, banked)
    rep["controls"]["K-RE"] = {"pass": ok, "n_rows_compared": n_re, "mismatches": bad[:20], "against": "RESULT_STRAT1.json (S1-C0, N1-C0 at s0 / s1, floor F)"}
    if not ok:
        prob.append("K-RE: a recomputed banked pair differs from RESULT_STRAT1.json (or none was compared)")
    # Z-EVAL / Z-USE
    ze, zu = {}, {}
    for arm in NL.R_ARMS:
        ze[arm] = z_eval(cells[arm], cells[arm + NL.Z], zero_hist)
        zu[arm] = z_use(CI, cells[arm], cells[arm + NL.Z], eid, rb, a.n_boot)
        if not ze[arm]["pass"]:
            prob.append(f"Z-EVAL {arm}: real - zero is not exactly 0.0 on the zero-history windows (n {ze[arm]['n_windows']})")
    rep["controls"]["Z-EVAL"] = ze
    rep["z_use_beside"] = {arm: {k: v for k, v in zu[arm].items() if k != "rows"} for arm in zu}
    # D1 / D2 / guard
    per_pair, rows_out = {}, {}
    for r_fam, b_fam in PAIRS:
        pn = f"{r_fam}-{b_fam}"
        d1, d2, guard, zus, allw, onrb = {}, {}, {}, {}, {}, {}
        for s in NL.SEEDS:
            R_, B_ = f"{r_fam}_s{s}", f"{b_fam}_s{s}"
            d1[s] = d1_row(CI, cells[R_], cells[B_], eid, rb, f2_rb, a.n_boot)
            d2[s] = d2_rows(CI, cells[R_], pred[R_], cells[B_], pred[B_], eid, turn, rb, a.n_boot)
            allw[s] = S.pair_rows(CI, cells[R_], cells[B_], eid, F2, a.n_boot)
            guard[s] = cost_cells(allw[s])
            zus[s] = zu[R_]["separated"]
            onrb[s] = OR.compare_cells(CI, {c: cells[R_][c][rb] for c in CELLS}, {c: cells[B_][c][rb] for c in CELLS}, eid[rb], None, a.n_boot)
        route_cost = False
        if a.contrast and os.path.exists(a.contrast):
            route_cost = bool(json.load(open(a.contrast, encoding="utf-8")).get("route_cost", {}).get(pn))
        per_pair[pn] = pair_labels(pn, d1, guard, zus, route_cost)
        rows_out[pn] = {"D1_TAC-LAT_on_RB": {str(s): d1[s] for s in d1}, "D2_beside": {str(s): d2[s] for s in d2}, "all_window_24_cells_under_F2": {str(s): S.slim(allw[s]) for s in allw},
                        "all_24_cells_on_RB_beside_no_bar": {str(s): S.slim(onrb[s]) for s in onrb}, "guard_cost_cells": {str(s): sorted(guard[s]) for s in guard},
                        "guard_gain_cells": {str(s): sorted(gain_cells(allw[s])) for s in allw}}
    rep["pairs"] = rows_out
    out = outcomes(per_pair)
    rep["outcome"] = out
    labels_all = sorted({x for v in out["per_pair"].values() for x in v["labels"]} | set(out["cross_pair"]))
    rep["next_step"] = {x: NEXT[x] for x in labels_all} or {"none": "no committed label fired: print the rows and name the cheapest next arm (RULE ZERO)"}
    fam = S.family_blocks({p: {s: S.slim(v) for s, v in {}.items()} for p in ()}) if False else None  # noqa: F841
    rep["families"] = {}
    for fam_name in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        cs = [c for c in CELLS if OR.FAMILY[c.split(":")[-1]] == fam_name]
        rep["families"][fam_name] = {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "estimator": "paired episode-cluster bootstrap",
                                     "cells": {c: {f"{pn}_s{s}": rows_out[pn]["all_window_24_cells_under_F2"][str(s)][c] for pn in rows_out for s in NL.SEEDS} for c in cs}}
    ld = lambda p: json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None  # noqa: E731
    rep["families"]["STRATEGIC"] = {"tier": "T0 at the window's first tick (route); T0 (detected info); unpaused 2026-11-16, read in full",
                                    "route_following_and_setting_frozen_reader_on_R_view": (ld(a.strategic) or {"status": "NOT-YET"}) if not ld(a.strategic) else {k: ld(a.strategic).get(k) for k in ("outcome_route_following", "route_setting_RC-A", "controls")},
                                    "R_minus_banked_contrasts": ld(a.contrast) or "NOT-YET",
                                    "detected_info_S1R_frozen_reader": (ld(a.ctx) or {}).get("outcome", "NOT-YET"), "detected_info_S1R_nofog_PI_ruling_2026-11-17": (ld(a.ctx_nofog) or {}).get("outcome", "NOT-YET"),
                                    "not_in_arm": {"derived_goals": "not in this arm", "intent_6_30s": "not in this arm (no output beyond 80 m of arc)"}}
    rep["provenance"] = {"card_sha256": NL.CARD_SHA256, "code_of_record": NL.COMMIT, "navtext_nav_rb.npz": NL.RB_SHA256, "navtext_nav_rb_report.json": NL.sha256_file(NL.RB_REPORT),
                         "banked_results_sha256": {os.path.basename(p): NL.sha256_file(p) for p in (NL.C0REP_RESULT, NL.STRAT1_RESULT, NL.SR.rsplit("/", 1)[0] + "/navrb/out/RESULT_NAVRB.json")},
                         "checkpoints": {x: NL.sha256_file(f"{NL.CK}/{x}/ckpt_last.pt") for x in NL.R_ARMS if os.path.exists(f"{NL.CK}/{x}/ckpt_last.pt")},
                         "reader_sha256": {f: NL.sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sh", ".cmd", ".ps1"))},
                         "frozen_reader_sha256": {f: NL.sha256_file(f"{NL.SR}/{f}") for f in ("strat1_read.py", "strat1_strategic.py", "strat1_ctx_read.py", "strat1_lib.py", "strat1_k4_eval.py")}}
    if prob:
        rep["outcome_status"] = "INVALID (control problems; labels printed unvalidated)"
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = [f"# NAVRB2: the roundabout nav channel (MEASURED, T1-kin; 141 episodes / 2,399 windows per arm; B {a.n_boot} seed 0; card sha256 {NL.CARD_SHA256[:8]})", "",
          f"Problems: {prob or 'none'}. K-RE rows: {n_re}. Strata: {json.dumps(rep['strata'])}.", "", "## Outcome", "", "```", json.dumps(out, indent=1, default=float), "```", "", "## Next step", ""]
    md += [f"* {k}: {v}" for k, v in rep["next_step"].items()] + ["", "## D1 (TAC-LAT on RB) and floors", "", "```", json.dumps({pn: rows_out[pn]["D1_TAC-LAT_on_RB"] for pn in rows_out}, indent=1, default=float),
                                                                       json.dumps(rep["floors"]["RB_TAC-LAT"]), "```", ""]
    cols = {f"{pn}_s{s}": rows_out[pn]["all_window_24_cells_under_F2"][str(s)] for pn in rows_out for s in NL.SEEDS}
    for fam_name in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        md += [f"## {fam_name} (T1-kin; * separated, G gain, C cost under F2)", "", md_table(cols, [c for c in CELLS if OR.FAMILY[c.split(':')[-1]] == fam_name]), ""]
    md += ["## STRATEGIC (T0)", "", "See the JSON: route following (real / donor / zero), RC-A route setting, R - banked contrasts, detected info (frozen reader and no-fog)."]
    open(a.md or a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", json.dumps({p: v[0] for p, v in per_pair.items()}), out["cross_pair"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    for p in prob:
        print("  PROBLEM:", p)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
