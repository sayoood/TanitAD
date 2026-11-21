"""c0rep_read.py -- the STRAT1 C0 same-seed replicate read.  Frozen rules: PREREG_ADDENDUM_C0REP.md (sha256 prefix 2f41cf5c, written 2026-11-17 ~19:45Z before either arm trained; nothing below changes a rule).
Same reader and constants as the SEEDS / N1S10 / S1D reads: the 24 driving cells of strat1_read.py (imported UNEDITED), F = |C0_rep - C0_s0| per cell, cost / gain as the card (separated on the worse / better side AND |Delta| > floor).
Tier of the 24 cells: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), 141 episodes / 2,399 windows per arm.  STRATEGIC: T0 at the window's first tick.
Evidence class: MEASURED (ours; dumps and T0, zero training).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); the floor answers "another training run at the same seed" with 2 pairs (a weak floor).
RULES (addendum, verbatim in substance)
  arms C0_s10r / C0_s11r: argv byte-identical to C0_s10 / C0_s11 except the run root (K-ARGV; on a mismatch the pair is left out before any number is read).
  same-seed pairs: C0_s10r - C0_s10 and C0_s11r - C0_s11 on every cell (delta, CI, separated).
  F_rep(cell) = max(|C0_s10r - C0_s10|, |C0_s11r - C0_s11|) from the full-set point deltas;  F2(cell) = max(F(cell), F_rep(cell)), F(cell) = |C0_rep - C0_s0| (the card floor).
  re-read: COST4 and GAIN4 for S1 - C0, N1 - C0, S1d - C0 and S1d - S1 at seeds {0, 1, 10, 11} with the card count rule (>= 3 of 4 seeds, >= 1 in each init) and F2 in place of F; nothing else changes.
  label (TAC-LAT is the deciding cell; every cell is printed):
    COST-HOLDS             TAC-LAT is still COST4_S1 under F2.
    COST-WITHIN-FLOOR      TAC-LAT is COST4_S1 under F and not under F2.
    DETERMINISTIC-AT-SEED  both same-seed pairs read exactly 0.0 on all 24 cells: F_rep = 0, F2 = F, COST4 stands as read.
  controls: K1 self-pair exactly 0.0; K-RE every recomputed pair that exists in a banked result equals it exactly (each banked file needs >= 1 compared row); K-ARGV as above.
  builder additions that decide nothing: K-REFLAG (re-deriving cost / gain under F reproduces the recomputed rows exactly), K-PIN (frozen reader sha256 prefixes and the addendum prefix).
Usage:  python c0rep_read.py --dumps DIR --out RESULT_STRAT1_C0REP.json --banked A.json B.json C.json D.json [--t0 DIR] [--n-boot 2000]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (HERE, os.path.join(R, "s1d"), os.path.join(R, "n1s10"), os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import c0rep_run as CRUN  # noqa: E402

L = CRUN.patch_lib()
import strat1_read as S  # noqa: E402  (UNEDITED)
import strat1_seeds_read as SS  # noqa: E402  (UNEDITED)
import n1_read as N  # noqa: E402  (study family: k_re, md_table, n1_outcome)
import s1d_read as SD  # noqa: E402  (study family: s1d_outcome)

SEEDS = SS.SEEDS
CELLS = S.CELLS
ACCURACY = S.OR.ACCURACY
DECIDING = "TAC-LAT"
RUN_ROOT_REP, RUN_ROOT_ORIG = "/root/strat1/runs_c0rep/", "/root/strat1/runs/"
REP_OF = {"C0_s10r": "C0_s10", "C0_s11r": "C0_s11"}
READ_ARMS = ("C0_s0", "C0_rep", "C0_s1", "C0_s10", "C0_s11") + tuple(REP_OF) + tuple(f"{k}_s{s}" for k in ("S1", "N1", "S1d") for s in SEEDS)
PINS = {"strat1_seeds_read.py": "2d30dd83", "seeds_run.py": "237e8655", "strat1_read.py": "fec13926", "strat1_lib.py": "538edf86",
        "strat1_strategic.py": "f27c005d", "run_dumps_strat1.py": "1ebc3692", "run_t0_strat1.py": "f0072c63"}
ADDENDUM_PREFIX = "2f41cf5c"
FIVE_PAIR = (("S1-C0", "S1", "C0"), ("N1-C0", "N1", "C0"), ("S1d-C0", "S1d", "C0"), ("S1d-S1", "S1d", "S1"), ("S1-N1", "S1", "N1"))
NEXT = {
    "COST-HOLDS": "S1's TAC-LAT cost is beyond same-seed training noise. The S1D label is re-read under F2 (CURED / PARTIAL / REMAINS as written); if it reads REMAINS, the next lever is the N2 card's attribution with the floor in hand.",
    "COST-WITHIN-FLOOR": "S1's TAC-LAT cost is not distinguishable from same-seed training noise: S1 has no measured driving cost on that cell. The S1D question (does the detach cure it) is not decidable and is closed as such; S1 stays the route-text arm.",
    "DETERMINISTIC-AT-SEED": "The pod trainer is bit-deterministic at a fixed seed, so a same-seed replicate cannot measure training noise: F_rep = 0, F2 = F, COST4 stands as read. The floor lever then needs a different-seed design, a new addendum.",
    "PREMISE-ABSENT": "TAC-LAT is not COST4_S1 under F in this recomputation, so the label question does not arise; reported as written, no step is attached.",
    "NOT-COMPUTABLE": "fewer than two same-seed pairs survived K-ARGV: no label is read.",
}


# ----------------------------------------------------------------------------------------------------------------------------------- pure logic (unit-tested)
def k_argv(rep_argv, orig_argv):
    """K-ARGV: the replicate's argv equals the original's token for token once the run root /root/strat1/runs_c0rep/ is replaced by /root/strat1/runs/.  -> (pass, [reason])"""
    a = json.dumps(rep_argv, sort_keys=True).replace(RUN_ROOT_REP, RUN_ROOT_ORIG)
    b = json.dumps(orig_argv, sort_keys=True)
    if a == b:
        return True, []
    ta, tb = json.loads(a), json.loads(b)
    if isinstance(ta, list) and isinstance(tb, list):
        diff = [f"token {i}: {x!r} vs {y!r}" for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        if len(ta) != len(tb):
            diff.append(f"length {len(ta)} vs {len(tb)}")
        return False, diff[:10] or ["differs"]
    keys = sorted(set(ta) | set(tb)) if isinstance(ta, dict) and isinstance(tb, dict) else []
    return False, [f"key {k}: {ta.get(k)!r} vs {tb.get(k)!r}" for k in keys if ta.get(k) != tb.get(k)][:10] or ["differs"]


def f_rep(pair_list):
    """pair_list: [{cell: row}] (the surviving same-seed pairs) -> {cell: max |delta| over the pairs, or None when no pair has the cell}.  Full-set point deltas."""
    out = {}
    for c in CELLS:
        ds = [abs(p[c]["delta"]) for p in pair_list if p[c].get("status") == "OK"]
        out[c] = float(max(ds)) if ds else None
    return out


def floor2(F, Frep):
    """F2 = max(F, F_rep) per cell.  A cell with no card floor (None: never scorable as cost / gain) keeps None; a missing F_rep counts as 0."""
    return {c: (None if F.get(c) is None else float(max(F[c], Frep.get(c) or 0.0))) for c in CELLS}


def reflag(row, higher, floor):
    """cost / gain of one pair row under another floor: the card rule exactly as obd_read.cell_delta states it (separated on the worse / better side AND |delta| > floor).  Pure."""
    if row.get("status") != "OK":
        return dict(row, floor=floor, above_floor=False, cost=False, gain=False)
    pt, sep = row["delta"], bool(row["separated"])
    better = (pt > 0) if higher else (pt < 0)
    worse = (pt < 0) if higher else (pt > 0)
    above = False if floor is None else abs(pt) > floor
    return dict(row, floor=floor, above_floor=bool(above), gain=bool(sep and better and above), cost=bool(sep and worse and above))


def reflag_pair(rows, floors):
    return {c: reflag(rows[c], c in ACCURACY, None if floors is None else floors.get(c)) for c in CELLS}


def read_under(per_seed_rows, floors, rule=None):
    """per_seed_rows {seed: {cell: row}} -> {"COST4": {cell: [seeds]}, "GAIN4": {...}} under `floors`.  rule: the counting function (default SS.count_rule); the tests swap mutated variants."""
    rule = rule or SS.count_rule
    re = {s: reflag_pair(per_seed_rows[s], floors) for s in per_seed_rows}
    return {"COST4": rule({s: S.cost_sets(re[s]) for s in re}), "GAIN4": rule({s: S.gain_sets(re[s]) for s in re})}, re


def c0rep_label(pair_list, cost4_f, cost4_f2, deciding=DECIDING):
    """pair_list: [{cell: row}] same-seed pairs (>= 2 needed); cost4_f / cost4_f2: COST4 of S1 - C0 under F / F2 ({cell: [seeds]}) -> (label, reason).  Pure."""
    if len(pair_list) < 2:
        return "NOT-COMPUTABLE", f"{len(pair_list)} same-seed pair(s) survived K-ARGV; the addendum needs 2"
    ok_rows = [r for p in pair_list for r in p.values() if r.get("status") == "OK"]
    if ok_rows and all(r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in ok_rows):
        return "DETERMINISTIC-AT-SEED", f"both pairs read exactly 0.0 (delta, lo, hi) on all {len(ok_rows) // len(pair_list)} scorable cells"
    if deciding in cost4_f2:
        return "COST-HOLDS", f"{deciding} is COST4_S1 under F2 at seeds {cost4_f2[deciding]}"
    if deciding in cost4_f:
        return "COST-WITHIN-FLOOR", f"{deciding} is COST4_S1 under F (seeds {cost4_f[deciding]}) and not under F2"
    return "PREMISE-ABSENT", f"{deciding} is not COST4_S1 under F"


def pin_problems(read_dir, pins=None, addendum=None):
    """frozen reader sha256 prefixes and the addendum prefix -> list of problems"""
    out = []
    for f, pre in (pins or PINS).items():
        h = hashlib.sha256(open(os.path.join(read_dir, "seeds" if f in ("strat1_seeds_read.py", "seeds_run.py") else "", f), "rb").read()).hexdigest()
        if not h.startswith(pre):
            out.append(f"{f}: sha256 {h[:8]} != {pre}")
    if addendum:
        h = hashlib.sha256(open(addendum, "rb").read()).hexdigest()
        if not h.startswith(ADDENDUM_PREFIX):
            out.append(f"addendum: sha256 {h[:8]} != {ADDENDUM_PREFIX}")
    return out


def md_floor_table(F, Frep, F2):
    lines = ["| cell | F | F_rep | F2 |", "|---|---|---|---|"]
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    for c in CELLS:
        lines.append(f"| {c} | {fmt(F.get(c))} | {fmt(Frep.get(c))} | {fmt(F2.get(c))} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------------------- STRATEGIC (T0)
def strategic_pairs(t0, CI, n_boot):
    """route following and route setting RC-A of each replicate pair, through the frozen strat1_strategic functions (imported, unedited).  C0 has no context heads: no detected info."""
    import strat1_strategic as S2
    join, navtok, navtext, ci, t, eid = S2.load_inputs(None)
    G = S2.ground(navtok, navtext, join, ci, t)
    VARS = S2.VARIANTS

    def load(tag):
        zs = {}
        for v in VARS:
            z = dict(np.load(f"{t0}/eval_STRAT1-{tag}_{v}.npz", allow_pickle=True))
            if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
                raise RuntimeError(f"{tag} {v}: the evaluator's windows differ from the grid")
            zs[v] = z
        cells = {v: S2.route_cells(zs[v]["ckpt"], G) for v in VARS}
        return cells, S2.rc_cells(S2.rcA_errors(zs["real"]["ckpt"], G))
    base = {a: load(a) for a in ("C0_rep", "C0_s0")}
    fl = {c: S.OR.floor_of(base["C0_rep"][1][c], base["C0_s0"][1][c]) for c in S2.RC_CELLS}
    out = {"tier": "T0 at the window's first tick (real / donor / zero nav, fresh ages)", "RC-A_floor_F": fl, "pairs": {}}
    rc_delta = {}
    for rep, orig in REP_OF.items():
        cr, rr = load(rep)
        co, ro = load(orig)
        pair = {"route_following_original": S2.arm_block(CI, co, G, eid, None), "route_following_replicate": S2.arm_block(CI, cr, G, eid, None),
                "replicate_minus_original_real_nav": S2.contrast_rows(CI, cr["real"], co["real"], eid),
                "route_setting_RC-A_replicate_minus_original": S2.route_setting(CI, rr, ro, fl, eid, n_boot)}
        out["pairs"][f"{rep}-{orig}"] = pair
        rc_delta[rep] = {c: pair["route_setting_RC-A_replicate_minus_original"][c] for c in S2.RC_CELLS}
    out["RC-A_F_rep"] = {c: f_rep_row([rc_delta[r][c] for r in rc_delta]) for c in S2.RC_CELLS}
    out["detected_info"] = "not computable: C0 has no context heads (cfg.ctx_heads off); n = 0 detected-info outputs"
    return out


def f_rep_row(rows):
    ds = [abs(r["delta"]) for r in rows if r.get("status") == "OK"]
    return float(max(ds)) if ds else None


# ----------------------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banked", nargs="+", required=True)
    ap.add_argument("--t0", default=None, help="T0 folder holding the C0_s10r / C0_s11r passes (STRATEGIC)")
    ap.add_argument("--addendum", default=os.path.join(HERE, "PREREG_ADDENDUM_C0REP.md"))
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = L.load_ci()
    alias = os.environ.get("C0REP_SMOKE_ALIAS") == "1"      # smoke only: each replicate reads its original's dump (a bit-identical pair), to exercise the reader before the real dumps exist
    dpath = lambda arm: f"{a.dumps}/dump_STRAT1-{REP_OF.get(arm, arm) if alias else arm}_m10_causal_heads"  # noqa: E731
    rep = {"tool": "c0rep_read.py", "smoke_alias": alias, "addendum": f"PREREG_ADDENDUM_C0REP.md (sha256 prefix {ADDENDUM_PREFIX}, frozen 2026-11-17 ~19:45Z before either arm trained)",
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC route": "T0 at the window's first tick",
                    "STRATEGIC detected info": "T0"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; the floor answers 'another training run at the same seed' with 2 pairs (a weak floor)",
           "rules": __doc__.split("RULES")[1].split("Usage:")[0].strip(), "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    pins = pin_problems(R, addendum=a.addendum)
    rep["controls"]["K-PIN"] = {"pass": not pins, "problems": pins}
    if pins:
        prob.append("K-PIN: " + "; ".join(pins))
    # K-ARGV first: a failing pair is left out before any number is read
    valid = []
    rep["controls"]["K-ARGV"] = {}
    for r_tag, o_tag in REP_OF.items():
        ra = json.load(open(f"{L.CK}/{r_tag}/argv.json", encoding="utf-8"))
        oa = json.load(open(f"{L.CK}/{o_tag}/argv.json", encoding="utf-8"))
        ok, why = k_argv(ra, oa)
        rep["controls"]["K-ARGV"][r_tag] = {"pass": ok, "reasons": why}
        if ok:
            valid.append(r_tag)
        else:
            prob.append(f"K-ARGV {r_tag}: {why[:3]}")
    missing = [arm for arm in READ_ARMS if not os.path.isdir(dpath(arm))]
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
    rep["controls"]["K1_self_pair"] = S.OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    pairs, by_kind = {}, {k: {} for k, _, _ in FIVE_PAIR}
    for s in SEEDS:
        for name, x, y in FIVE_PAIR:
            pairs[f"{name}_s{s}"] = S.pair_rows(CI, cells[f"{x}_s{s}"], cells[f"{y}_s{s}"], eid, F, a.n_boot)
            by_kind[name][s] = pairs[f"{name}_s{s}"]
    rep_pairs = {}
    for r_tag, o_tag in REP_OF.items():
        rep_pairs[f"{r_tag}-{o_tag}"] = S.pair_rows(CI, cells[r_tag], cells[o_tag], eid, F, a.n_boot)
    pairs.update(rep_pairs)
    for s in (10, 11):                                   # the earlier replicate pairs C0_s10 - C0_s0 / C0_s11 - C0_s1 (banked in the N1S10 read): K-RE only
        pairs[f"C0_s{s}-C0_s{s - 10}"] = S.pair_rows(CI, cells[f"C0_s{s}"], cells[f"C0_s{s - 10}"], eid, F, a.n_boot)
    slim = {p: S.slim(v) for p, v in pairs.items()}
    banked = [(os.path.basename(b), json.load(open(b, encoding="utf-8"))) for b in a.banked]
    ok, bad, counts = N.k_re(slim, banked)
    rep["controls"]["K-RE"] = {"pass": ok, "mismatches": bad[:20], "n_mismatches": len(bad), "rows_compared": counts}
    if not ok:
        prob.append("K-RE: a recomputed row differs from a banked row (or none was compared)")
    # K-REFLAG: re-deriving cost / gain under F reproduces the recomputed rows
    rf_bad = [f"{p} {c}" for p, rows in pairs.items() for c, v in reflag_pair(rows, F).items() if (v["cost"], v["gain"]) != (rows[c].get("cost"), rows[c].get("gain"))]
    rep["controls"]["K-REFLAG"] = {"pass": not rf_bad, "mismatches": rf_bad[:10], "rows": sum(len(r) for r in pairs.values())}
    if rf_bad:
        prob.append("K-REFLAG: re-deriving cost / gain under F differs from the recomputed rows")
    # floors
    surviving = [rep_pairs[f"{t}-{REP_OF[t]}"] for t in valid]
    Frep = f_rep(surviving) if surviving else {c: None for c in CELLS}
    F2 = floor2(F, Frep)
    rep["floors"] = {"F": dict(F), "F_rep": Frep, "F2": F2, "n_pairs_used": len(surviving),
                     "definition": "F = |C0_rep - C0_s0|; F_rep = max over the surviving same-seed pairs of |replicate - original| (full-set point deltas); F2 = max(F, F_rep); applied to all four seeds",
                     "n_cells_F_none": sum(v is None for v in F.values())}
    # same-seed pairs, every cell
    rep["same_seed_pairs"] = {p: S.slim(v) for p, v in rep_pairs.items()}
    # re-read under F and F2
    rep["reread"] = {}
    for kind, _, _ in FIVE_PAIR:
        rf, _ = read_under(by_kind[kind], F)
        rf2, rows2 = read_under(by_kind[kind], F2)
        rep["reread"][kind] = {"under_F": rf, "under_F2": rf2, "changed_by_F2": {"COST4_dropped": sorted(set(rf["COST4"]) - set(rf2["COST4"])), "GAIN4_dropped": sorted(set(rf["GAIN4"]) - set(rf2["GAIN4"])),
                                                                                  "COST4_added": sorted(set(rf2["COST4"]) - set(rf["COST4"])), "GAIN4_added": sorted(set(rf2["GAIN4"]) - set(rf["GAIN4"]))}}
        rep["reread"][kind]["rows_under_F2"] = {str(s): {c: {k: v for k, v in r.items() if k in ("delta", "lo", "hi", "separated", "floor", "above_floor", "cost", "gain")} for c, r in rows2[s].items()} for s in SEEDS}
    label, reason = c0rep_label(surviving, rep["reread"]["S1-C0"]["under_F"]["COST4"], rep["reread"]["S1-C0"]["under_F2"]["COST4"])
    rep["label"], rep["label_reason"], rep["next_step"] = label, reason, NEXT[label]
    rep["deciding_cell"] = {"cell": DECIDING, "S1-C0_per_seed_under_F2": {str(s): {k: v for k, v in rep["reread"]["S1-C0"]["rows_under_F2"][str(s)][DECIDING].items()} for s in SEEDS},
                            "F": F.get(DECIDING), "F_rep": Frep.get(DECIDING), "F2": F2.get(DECIDING)}
    # the S1D label and the N1 attribution under F and F2 (labels as the earlier reads wrote them)
    def rows_under(kind, fl):
        return {s: reflag_pair(by_kind[kind][s], fl) for s in SEEDS}
    rep["s1d_label"] = {}
    rep["n1_attribution"] = {}
    for tag, fl in (("under_F", F), ("under_F2", F2)):
        lab, det = SD.s1d_outcome(rows_under("S1-C0", fl), rows_under("S1d-C0", fl), rows_under("S1d-S1", fl))
        rep["s1d_label"][tag] = {"labels": lab, "detail": det}
        att, det2 = N.n1_outcome(rows_under("S1-C0", fl), rows_under("N1-C0", fl), rows_under("S1-N1", fl))
        rep["n1_attribution"][tag] = {"attribution": att, "detail": det2}
    names = ["C0_s10r-C0_s10", "C0_s11r-C0_s11"]
    fam = S.family_blocks({p: slim[p] for p in names})
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "estimator": "paired episode-cluster bootstrap", "n_windows": int(len(eid)),
                           "n_episodes": int(len(np.unique(eid)))} for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    rep["n_cells_nonzero_by_pair"] = {p: sum(1 for r in v.values() if r.get("status") == "OK" and r["delta"] != 0.0) for p, v in rep_pairs.items()}
    if a.t0 and all(os.path.exists(f"{a.t0}/eval_STRAT1-{t}_{v}.npz") for t in list(REP_OF) + list(REP_OF.values()) + ["C0_rep", "C0_s0"] for v in ("real", "donor", "zero")):
        rep["families"]["STRATEGIC"] = strategic_pairs(a.t0, CI, a.n_boot)
        print(f"strategic done {time.time() - t0:.0f}s", flush=True)
    else:
        rep["families"]["STRATEGIC"] = {"status": "NOT-YET", "reason": "T0 passes for the replicates (or their references) are missing"}
        prob.append("STRATEGIC: T0 passes missing")
    rep["provenance"] = S.provenance(a.dumps, READ_ARMS)
    rep["provenance"]["frozen_reader_sha256"] = {f: L.sha256_file(os.path.join(R, "seeds", f)) for f in ("strat1_seeds_read.py", "seeds_run.py")}
    rep["provenance"]["study_reader_sha256"] = {f: L.sha256_file(f"{HERE}/{f}") for f in ("c0rep_read.py", "c0rep_run.py", "test_c0rep_read.py", "after_c0rep.sh", "launch_c0rep_read.sh", "strat1_bank_pod_c0rep.py")}
    rep["provenance"]["family_reader_sha256"] = {f: L.sha256_file(os.path.join(R, d, f)) for d, f in (("n1s10", "n1_read.py"), ("n1s10", "n1_run.py"), ("s1d", "s1d_read.py"), ("s1d", "s1d_run.py"))}
    rep["provenance"]["addendum_sha256"] = L.sha256_file(a.addendum)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = [f"# STRAT1 C0 same-seed replicates (MEASURED; T1-kin driving cells, T0 STRATEGIC; 141 episodes / 2,399 windows; paired episode-cluster bootstrap B {a.n_boot} seed 0, 'another draw of EPISODES')", "",
          f"Addendum PREREG_ADDENDUM_C0REP.md (sha256 prefix {ADDENDUM_PREFIX}). Problems: {prob or 'none'}. K-RE rows compared: {counts}. Pairs used for F_rep: {len(surviving)} of 2.", "",
          f"## Label: {label}", "", reason, "", f"Next step: {NEXT[label]}", "", "## Floors", "", md_floor_table(F, Frep, F2), "",
          "## COST4 / GAIN4 under F and F2", "", "```", json.dumps({k: {"under_F": v["under_F"], "under_F2": v["under_F2"], "changed_by_F2": v["changed_by_F2"]} for k, v in rep["reread"].items()}, indent=1), "```", "",
          "## S1D label and N1 attribution under F and F2", "", "```", json.dumps({"s1d_label": {k: v["labels"] for k, v in rep["s1d_label"].items()}, "n1_attribution": {k: v["attribution"] for k, v in rep["n1_attribution"].items()}}, indent=1), "```", ""]
    for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        if f in rep["families"]:
            md += [f"## {f} (T1-kin), replicate minus original", "", N.md_table(rep["families"][f]["cells"], names), ""]
    md += ["## STRATEGIC (T0)", "", "See the JSON: route following of the replicate and the original, the paired replicate-minus-original contrast per arc, RC-A route setting with F and F_rep; detected info is not computable for C0 (no context heads)."]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "| label:", label, "|", reason, "| problems:", len(prob), "|", rep["wall_s"], "s")
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
