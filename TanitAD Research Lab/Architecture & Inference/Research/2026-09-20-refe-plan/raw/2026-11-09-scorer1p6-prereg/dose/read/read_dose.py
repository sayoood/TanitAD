#!/usr/bin/env python3
"""read_dose.py -- the reader of PREREG_DOSE_P6 (frozen 2026-11-10, sha256 037c1272...ebb52), sections 2 to 5. CPU only. It judges ONE arm named S1ZD (S1Z's recipe, 1,600 updates).

    python read_dose.py read --p6-pass DIR --c0 DIR --c0r DIR --s1z DIR --s1z1 DIR --s1zd DIR [--sft-log PATH] --out OUT [--no-ledger | --ledger PATH]

DIR holds e3_logits.npz (the dev-box arm pass); --p6-pass is the folder that holds P6_control/e3_logits.npz.

REUSE, NOT A FORK: read_rep.py (the replicate reader: Table, changed_rows, run_vector, slot_inputs, make_inputs, bar_proven, goal_label), which imports the Batch 1 reader
read_scorer1p6 (gates, paired log-cluster bootstrap, family tables) and the S1Z reader read_s1z.s1z_read (G1-G4, M2) unchanged. Each of S1Z, S1Z1 and S1ZD is read exactly as read_rep
reads an S1Z-recipe run. What this file ADDS: D1, D2 (the card's two bars), the A/B/D/C/Unnamed outcome map, the regret split of S1ZD next to P6's (split.py of the P6 navtest read, on
the card's truth table), the trainer held-out curve from --sft-log, the P6-control gate and the DOSE K ledger key.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
P6BASE = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base"
REP_DIR = os.environ.get("DOSE_REP_DIR", P6BASE + "/rep/read")
if REP_DIR not in sys.path:
    sys.path.insert(0, REP_DIR)
import read_rep as RR  # noqa: E402  (the replicate reader, imported unchanged; it puts eval/ and s1z/read on sys.path)

R, S = RR.R, RR.S
SEL_DIR = P6BASE + "/selectors"
CARD = P6BASE + "/PREREG_DOSE_P6.md"
CARD_SHA256 = "037c127235b51c374780f31ad2e93855ac7c141a4051f89498f40f1beff52ebb"
REAL_LEDGER = SEL_DIR + "/ledger_K.json"
DEFAULT_LEDGER = REAL_LEDGER
CONTROL_ROOT = "F:/tanitad_scratch/dose_read_work"

# ------------------------------------------------------------------------------------------------ frozen constants (each with its card line; a test compares them to the card text)
S_P = 0.1791                                  # section 0/2: "s_p = 0.1791", fixed, an input from the REP read
T_975_DF4 = 2.776                             # section 0: "t(0.975, 4) = 2.776"
D1_THR = T_975_DF4 * S_P                      # section 3 D1: "t * s_p = 2.776 * 0.1791 = 0.4972"
D2_THR = T_975_DF4 * S_P * math.sqrt(1.0 + 1.0 / 2.0)   # section 3 D2: "t * s_p * sqrt(1 + 1/2) = 0.6089"
D1_PRINTED, D2_PRINTED = 0.4972, 0.6089
GOAL = 94.55                                  # section 3 goal
BARS = {"bootstrap_draws": R.B_DRAWS, "bootstrap_seed": R.SEED, "s_p": S_P, "t": T_975_DF4, "D1_threshold": D1_THR, "D1_threshold_printed": D1_PRINTED,
        "D2_threshold": D2_THR, "D2_threshold_printed": D2_PRINTED, "goal": GOAL}

NEXT = {   # card section 4, each outcome's next lever
    "DOSE-A": "more updates is a real lever. Next: S1ZD's seed-1 replicate (adoptability, as the REP card), then a 6,400-update dose, each with a PI check before pod time",
    "DOSE-B": "the dose clears noise against P6 but not against the 400-update recipe. Next: S1ZD's seed-1 replicate, so D2 is read on two runs",
    "DOSE-D": "not adoptable. Next: the failing guard's family split on E2 (Shapley over heads), under its own card",
    "DOSE-C": ("a longer fine-tune of the scorer's loss does not buy PDMS. Next: leave the fine-tune line and attack the largest regret term, zero-multiplier picks (2.81 of 6.70 points), "
               "with a lever outside the scorer-loss family, under its own card"),
    "Unnamed": "matches none of DOSE-A, B, D, C (an input is undefined): labelled as such, the card names no lever",
}
NAME = {"DOSE-A": "DOSE WORKS", "DOSE-B": "BEATS P6 ONLY", "DOSE-D": "GUARD FAIL", "DOSE-C": "SATURATED", "Unnamed": "UNNAMED"}
ORDER = ("DOSE-A", "DOSE-B", "DOSE-D", "DOSE-C")        # section 4: "Ordering of labels when several match: A, B, D, C"

AMBIGUITIES = [
    "D1 Runs: S1Z, S1Z1 (the D2 reference) and S1ZD each go through read_s1z.s1z_read in its S1Z slot, as read_rep does for S1Z-recipe runs; E3 scores of changed picks come from "
    "table_p6final_c.npz (sha 3bda2080...), never from an arm csv. C0 and C0r are required because s1z_read needs them (P1's zero point and the floor F = |C0r - C0|); the card's CLI lists only "
    "the S1Z recipe runs, so --c0 and --c0r are ADDED inputs. They decide nothing in this card.",
    "D2 'Guards' = G1, G2, G3 (Batch 1, vs P6) and G4 (CRUISE slow-pick share) -- all four must pass, exactly as read_rep.py's guards_ok for an S1Z-recipe run. M2 (failure share, S1Z card) is REPORTED "
    "next to them (as read_rep reports it) and does not decide. The S1Z/S1Z1 reference runs' guards are printed, not deciding.",
    "D3 Thresholds are computed from the card's printed s_p and t (2.776 * 0.1791 = 0.49718, times sqrt(1.5) = 0.60892) and compared strictly; the card's rounded 0.4972 / 0.6089 are printed "
    "next to them (a test pins that they round to the card's values). They differ by < 2e-5.",
    "D4 D2's reference is the mean of the S1Z and S1Z1 per-token E3 PDMS vectors inside each draw (card section 2), on the same draws as D1 and every other statistic. PROVEN iff CI-lo > 0 and d > 0.6089, both strict.",
    "D5 Outcome labels follow section 4 literally: A = D1 and D2 PROVEN and all guards pass; B = D1 PROVEN, D2 not; D = D1 PROVEN and a guard fails; C = D1 NOT PROVEN. Every match is listed; the "
    "primary label follows A, B, D, C. B carries no guard clause on the card, so B and D can both match (B first). Unnamed only when D1 or D2 is undefined (non-finite CI).",
    "D6 Regret split = split.py of the P6 navtest read (raw/2026-11-09-p6-navtest-read/selection/split.py) on the card's truth table table_p6final_c.npz, with the arm's pick (navsim_v1 rule on its logits, "
    "torch float32, first index on ties) in place of the stored pick; zero-multiplier = pick PDMS 0 while a non-zero proposal existed, TTC-0 = non-zero pick with TTC sub-score 0, rest = every other token; "
    "contribution = mean over ALL tokens of (best of 64 - pick) x 100 restricted to the class. Intervals: the shared 10,000 log-cluster draws of this read (split.py used its own multinomial draws, so its "
    "printed CIs differ in the later digits; the POINT values are the same).",
    "D7 --sft-log: a JSON-lines file (sft.jsonl or the stdout log); every row with event 'eval' is listed (tag, n, base and B held-out pick x100, B - base with its CI, B zero picks). The card says every 200 "
    "updates; the trainer's default --eval-every is 400, so the tags present are printed as found. A monitor only (PREREG_SCORER1P6 section 2): in-sample for P6, decides nothing. Not given: 'not given'.",
    "D8 --p6-pass: the dev-box arm pass's P6_control/e3_logits.npz must reproduce the banked P6 pick on 100 % of tokens and carry bit-identical proposals (gate G0-P6PASS); its logits' bit-equality with the "
    "banked P6 logits is reported. A failing gate is READ STOPPED (the pass pipeline would not be the banked P6's).",
    "D9 K ledger: one key per S1ZD E3 read, DOSE:S1ZD:<sha256 of e3_logits.npz>:E3 (family DOSE; style of REP:<run>:<sha>:E3 and S1ZSM:<sha>:E3); S1Z and S1Z1 are re-read as references and NOT keyed (their "
    "REP keys exist). D1 and D2 are this card's own pre-registered tests; the Bonferroni interval of d vs P6 at K is for best-of-K statements only.",
    "D10 A real read refuses an S1ZD whose e3_logits.npz is bit-identical to S1Z's, S1Z1's, C0's, C0r's or P6_control's (a 1,600-update arm is never identical to them); --control-standin allows it, stamps "
    "CONTROL_NOT_A_RESULT, writes only under " + CONTROL_ROOT + " and never touches the real ledger.",
]


# ------------------------------------------------------------------------------------------------ pure functions (tested on hand-made numbers)
def d1_proven(lo, d):
    return RR.bar_proven(lo, d, D1_THR)


def d2_proven(lo, d):
    return RR.bar_proven(lo, d, D2_THR)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def decide_outcome(d1, d2, guards_ok, defined=True):
    """d1, d2, guards_ok bool; defined False when a CI is not finite -> dict(matches, primary, name, next_levers)"""
    m = []
    if defined:
        if d1 and d2 and guards_ok:
            m.append("DOSE-A")
        if d1 and not d2:
            m.append("DOSE-B")
        if d1 and not guards_ok:
            m.append("DOSE-D")
        if not d1:
            m.append("DOSE-C")
    if not m:
        m = ["Unnamed"]
    prim = next((o for o in ORDER if o in m), "Unnamed")
    return {"matches": m, "primary": prim, "name": NAME[prim], "next_levers": {o: NEXT[o] for o in m}}


CLASSES = ("zero_pick_selection", "zero_pick_proposal", "nonzero_pick_TTC0", "rest")
CLASS_LABEL = {"zero_pick_selection": "zero-multiplier picks (pick PDMS 0, a non-zero proposal existed)", "zero_pick_proposal": "zero pick, all 64 proposals score 0",
               "nonzero_pick_TTC0": "TTC-0 picks (non-zero pick, TTC sub-score 0)", "rest": "rest (EP, comfort)"}
TTC_IX = R.SUBN.index("TTC")


def regret_masks(P, S_, pick):
    """split.py of the P6 navtest read, same logic. P [N,64] pdms, S_ [N,64,6], pick [N] -> (reg [N], masks, best, pk)"""
    N = len(pick)
    ix = np.arange(N)
    best = np.nanmax(P, 1)
    pk = P[ix, pick]
    reg = best - pk
    zero = pk == 0
    prop = zero & (best == 0)
    sel = zero & (best > 0)
    ttc0 = (~zero) & (S_[ix, pick, TTC_IX] == 0)
    rest = (~zero) & (~ttc0)
    return reg, {"zero_pick_selection": sel, "zero_pick_proposal": prop, "nonzero_pick_TTC0": ttc0, "rest": rest}, best, pk


def regret_split(table, tix, pick, B3):
    """-> dict: mean pick, mean best, regret (x100) and the four classes with n, share, contribution and the shared-draw CI; '_per_token' for the pairing"""
    P, S_ = table.pdms[tix], table.sub[tix]
    reg, masks, best, pk = regret_masks(P, S_, np.asarray(pick))
    out = {"n_tokens": int(len(pick)), "mean_pick_x100": float(100 * pk.mean()), "mean_best_x100": float(100 * best.mean()), "regret_x100": float(100 * reg.mean()), "components_x100": {}}
    for k in CLASSES:
        m = masks[k]
        pt, r, n = B3.stat(100.0 * reg * m)
        rr = R.rec(pt, r, n)
        out["components_x100"][k] = {"label": CLASS_LABEL[k], "n_tokens": int(m.sum()), "share_of_tokens": float(m.mean()), "regret_contribution": float(100 * (reg * m).mean()),
                                     "ci95": [rr["lo"], rr["hi"]]}
    out["sum_of_components_x100"] = float(sum(v["regret_contribution"] for v in out["components_x100"].values()))
    out["_per_token"] = {k: 100.0 * reg * masks[k] for k in CLASSES}
    return out


def regret_pair(table, tix, pick_arm, pick_p6, B3):
    a = regret_split(table, tix, pick_arm, B3)
    b = regret_split(table, tix, pick_p6, B3)
    diff = {}
    for k in CLASSES:
        pt, r, n = B3.diff(a["_per_token"][k], b["_per_token"][k])
        diff[k] = R.rec(pt, r, n)
    pt, r, n = B3.diff(sum(a["_per_token"].values()), sum(b["_per_token"].values()))
    diff["total_regret"] = R.rec(pt, r, n)
    a.pop("_per_token"), b.pop("_per_token")
    return {"S1ZD": a, "P6": b, "S1ZD_minus_P6_contribution_x100": diff,
            "note": "reported, not deciding; negative = less regret for S1ZD (paired log-cluster bootstrap on the same draws as D1)"}


def parse_sft_log(path):
    """every JSON row with event 'eval' of a trainer log -> list of dicts (tag, n, base/B pick, B - base, B zero picks)"""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            i = ln.find("{")
            if i < 0:
                continue
            try:
                r = json.loads(ln[i:])
            except ValueError:
                continue
            if not isinstance(r, dict) or r.get("event") != "eval":
                continue
            b, base = r.get("B") or {}, r.get("base") or {}
            pm = b.get("pick_minus_base") or {}
            rows.append({"tag": r.get("tag"), "n": r.get("n"), "base_pick_x100": base.get("pick"), "B_pick_x100": b.get("pick"), "B_minus_base_mean": pm.get("mean"),
                         "B_minus_base_ci95": pm.get("ci95"), "B_zero_picks": b.get("zero_picks"), "base_zero_picks": base.get("zero_picks"), "at": r.get("at")})
    return rows


def sha256_file(p):
    return R.sha256_file(p)


# ------------------------------------------------------------------------------------------------ the read
def dose_read(inp, table, p6_pass_dir=None, sft_log=None, ledger_path=None, standin=False, card=None):
    """inp.arms: C0, C0r, S1Z, S1Z1, S1ZD -> result dict, status READ / READ STOPPED / INCOMPLETE / REFUSED."""
    t0 = time.time()
    card = card or CARD
    res = {"tool": "read_dose.py", "card": {"path": card, "sha256_expected": CARD_SHA256, "sha256_actual": sha256_file(card) if os.path.exists(card) else None},
           "tier": R.TIER, "evidence_class": R.EVIDENCE, "estimator": R.ESTIMATOR, "bars": BARS, "strategic": R.STRATEGIC, "ambiguities_dose": AMBIGUITIES,
           "ambiguities_batch1": R.AMBIGUITIES, "inputs": inp.provenance}
    if standin:
        res["stamp"] = "CONTROL_NOT_A_RESULT"
    if res["card"]["sha256_actual"] != CARD_SHA256:
        res.update(status="READ STOPPED", why="the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    runs = ["S1Z", "S1Z1", "S1ZD"]
    need = ["C0", "C0r"] + runs
    miss = [k for k in need if k not in inp.arms]
    if miss:
        res.update(status="REFUSED", why=f"runs {miss} are required (C0 and C0r for s1z_read's zero point and floor; S1Z and S1Z1 for D2; S1ZD is the arm)")
        return res
    shas = {k: inp.provenance["arms"][k][1] for k in need}
    if inp.provenance.get("p6_control_sha"):
        shas["P6_control"] = inp.provenance["p6_control_sha"]
    dup = sorted({k for a_, b_ in itertools.combinations(shas, 2) if shas[a_] == shas[b_] for k in (a_, b_)})
    if dup and not standin:
        res.update(status="REFUSED", why=f"inputs {dup} have bit-identical e3_logits.npz (a real S1ZD is never identical to another arm); only --control-standin allows it (D10)")
        return res
    if not table.sha256.startswith(RR.TABLE_SHA256_PREFIX):
        res.update(status="REFUSED", why=f"the truth table sha256 {table.sha256} does not start with {RR.TABLE_SHA256_PREFIX}")
        return res
    res["truth_table"] = {"path": table.path, "sha256": table.sha256}
    res["code_sha256"] = {"read_dose.py": sha256_file(HERE / "read_dose.py"), "read_rep.py": sha256_file(Path(REP_DIR) / "read_rep.py"),
                          "read_scorer1p6.py": sha256_file(P6BASE + "/eval/read_scorer1p6.py"), "read_s1z.py": sha256_file(P6BASE + "/s1z/read/read_s1z.py")}
    W, N = inp.world, len(inp.world.token)
    p6_pick = R.pick_logits(inp.p6_logits)
    try:
        tix = table.align(list(W.token))
        p6_score = np.array([inp.p6_rows[t][1] for t in W.token], np.float64)
    except (RR.Refused, KeyError) as ex:
        res.update(status="READ STOPPED", why=f"P6 or truth-table alignment: {ex}")
        return res

    # ---- G0-P6PASS
    if p6_pass_dir is not None:
        g = {"name": "G0-P6PASS", "path": str(Path(p6_pass_dir) / "P6_control" / "e3_logits.npz")}
        try:
            Z = np.load(g["path"], allow_pickle=False)
            ix, err = R._align(Z["token"], list(W.token), "P6_control")
            if err:
                g.update(PASS=False, why=err)
            else:
                pk_ = "proposals" if "proposals" in Z.files else "props8"
                Lc, Pc = np.asarray(Z["logits"])[ix], np.asarray(Z[pk_])[ix]
                g["n_pick_equal_banked_p6"] = int((R.pick_logits(Lc) == p6_pick).sum())
                g["n_tokens"] = int(N)
                g["proposals_bit_identical"] = bool(R.bits_equal(Pc, inp.p6_proposals))
                g["logits_bit_identical_reported"] = bool(R.bits_equal(Lc, inp.p6_logits))
                g["max_abs_logit_diff_reported"] = float(np.abs(Lc.astype(np.float64) - np.asarray(inp.p6_logits, np.float64)).max())
                g["PASS"] = bool(g["n_pick_equal_banked_p6"] == N and g["proposals_bit_identical"])
        except Exception as ex:      # noqa: BLE001
            g.update(PASS=False, why=f"cannot read P6_control: {ex}")
        res["p6_pass_gate"] = g
        if not g["PASS"]:
            res.update(status="READ STOPPED", why=f"G0-P6PASS failed: {g}")
            return res

    # ---- picks, in-memory changed-pick scores (table route)
    pick = {}
    try:
        for nm in need:
            a = inp.arms[nm]
            ix, err = R._align(a.e3_token, list(W.token), nm)
            if err:
                res.update(status="READ STOPPED", why=err)
                return res
            pick[nm] = R.pick_logits(np.asarray(a.e3_logits)[ix])
            a.rows = RR.changed_rows(table, tix, W, pick[nm], p6_pick)
    except RR.Refused as ex:
        res.update(status="READ STOPPED", why=str(ex))
        return res

    # ---- one s1z_read per S1Z-recipe run (S1Z, S1Z1, S1ZD)
    B3 = R.BootSet(W.log)
    per, gates = {}, {}
    for nm in runs:
        r1_ = S.s1z_read(RR.slot_inputs(inp, {"C0": "C0", "C0r": "C0r", "S1Z": nm}), smoke=False, ledger_path=None)
        for g_, v in (r1_.get("gates") or {}).items():
            gates[f"{nm}:{g_}"] = v
        if r1_.get("status") != "READ":
            res.update(status=r1_.get("status"), why=f"run {nm}: {r1_.get('why')}", gates=gates, missing_tokens=r1_.get("missing_tokens"))
            return res
        per[nm] = r1_
    res["gates"] = gates

    vec_p6 = p6_score * 100.0
    vec = {nm: RR.run_vector(table, tix, pick[nm], p6_pick, p6_score) for nm in runs}
    runs_out = {}
    for nm in runs:
        r1_ = per[nm]
        ep = r1_["batch1_endpoints_S1Z_slot"]
        lvl_b1, vs_b1 = ep["E3_pdms"], ep["E3_pdms_vs_P6"]
        ok_g = {k: bool(r1_["guards"][k]["ok"]) for k in ("G1", "G2", "G3", "G4")}
        fam = r1_["family_clause"]
        pt, r, n = B3.stat(vec[nm])
        lvl = R.rec(pt, r, n)
        pt, r, n = B3.diff(vec[nm], vec_p6)
        vs = R.rec(pt, r, n)
        chk = abs(lvl["point"] - lvl_b1["point"])
        if chk > 1e-9 or vs["lo"] != vs_b1["lo"] or vs["hi"] != vs_b1["hi"]:
            res.update(status="READ STOPPED", why=f"run {nm}: the table-vector level/CI disagrees with the Batch 1 reader's (point diff {chk}, CI {vs} vs {vs_b1})")
            return res
        m2 = r1_["M2"]
        runs_out[nm] = {"logits_sha256": shas[nm], "E3_pdms": lvl, "E3_pdms_vs_P6": vs, "goal_status_run": ep["goal_status"], "E3_pick_changed_vs_P6": r1_["E3_pick_changed_vs_P6"],
                        "guards_by_id": ok_g, "guards_pass": all(ok_g.values()),
                        "guards": {"G1": r1_["guards"]["G1"], "G2": r1_["guards"]["G2"], "G3": r1_["guards"]["G3"], "G4": r1_["guards"]["G4"], "M2_reported": m2},
                        "M2_clears": bool(m2["clears"]) if isinstance(m2, dict) and "clears" in m2 else None,
                        "four_families_vs_P6": {"STRATEGIC": R.STRATEGIC, "components": fam["components"] if "components" in fam else fam.get("components_vs_P6"),
                                                "adverse_components": fam["adverse_components"]},
                        "artifact": f"{inp.provenance['arms'][nm][0]} + truth table {table.sha256[:8]}", "tier": R.TIER["E3"],
                        "estimator": "paired log-cluster bootstrap, 10,000 draws, seed 20260927 (every run on the same draws)"}
    res["runs"] = runs_out

    # ---- D1, D2
    d_vec = vec["S1ZD"]
    x_s1z = np.mean([vec["S1Z"], vec["S1Z1"]], axis=0)
    pt, r_d1, n = B3.diff(d_vec, vec_p6)
    d1 = R.rec(pt, r_d1, n)
    pt, r_d2, n = B3.diff(d_vec, x_s1z)
    d2 = R.rec(pt, r_d2, n)
    pt, r, n = B3.stat(x_s1z)
    ref_lvl = R.rec(pt, r, n)
    defined = all(finite(v) for v in (d1["point"], d1["lo"], d2["point"], d2["lo"]))
    p1, p2 = d1_proven(d1["lo"], d1["point"]), d2_proven(d2["lo"], d2["point"])
    guards_ok = runs_out["S1ZD"]["guards_pass"]
    gid = runs_out["S1ZD"]["guards_by_id"]
    res["bars_read"] = {
        "S1ZD_E3_pdms": runs_out["S1ZD"]["E3_pdms"], "S1Z_recipe_mean_E3_pdms": ref_lvl,
        "D1": {"d_S1ZD_minus_P6": d1, "threshold": D1_THR, "threshold_printed": D1_PRINTED, "ci_lo_gt_0": bool(finite(d1["lo"]) and d1["lo"] > 0),
               "d_gt_threshold": bool(finite(d1["point"]) and d1["point"] > D1_THR), "PROVEN": p1},
        "D2": {"d_S1ZD_minus_S1Z_recipe_mean": d2, "reference_runs": ["S1Z", "S1Z1"], "threshold": D2_THR, "threshold_printed": D2_PRINTED,
               "ci_lo_gt_0": bool(finite(d2["lo"]) and d2["lo"] > 0), "d_gt_threshold": bool(finite(d2["point"]) and d2["point"] > D2_THR), "PROVEN": p2},
        "guards": {**gid, "all_pass": guards_ok, "M2_reported_not_deciding": runs_out["S1ZD"]["M2_clears"]},
        "goal": {"bar": GOAL, "S1ZD": R.fin(runs_out["S1ZD"]["E3_pdms"]["point"]), "status": RR.goal_label(runs_out["S1ZD"]["E3_pdms"]["point"])}}
    out = decide_outcome(p1, p2, guards_ok, defined)
    out["inputs"] = {"D1": p1, "D2": p2, "guards_ok": guards_ok, "defined": defined}
    res["outcome"] = out

    # ---- reported, not deciding
    res["regret_split"] = regret_pair(table, tix, pick["S1ZD"], p6_pick, B3)
    res["trainer_heldout_curve"] = ({"path": str(sft_log), "sha256": sha256_file(sft_log), "rows": parse_sft_log(sft_log),
                                     "note": "trainer monitor only; in-sample for P6 (PREREG_SCORER1P6 section 2), decides nothing"} if sft_log else "not given")

    # ---- K ledger and the Bonferroni interval (a successful read only)
    kinfo = {"ledger": None, "K": None, "note": "no ledger path given: K NOT COUNTED"}
    if ledger_path:
        if SEL_DIR not in sys.path:
            sys.path.insert(0, SEL_DIR)
        import selcore
        led = selcore.Ledger(ledger_path)
        pre = "CONTROL:" if standin else ""
        key = f"{pre}DOSE:S1ZD:{shas['S1ZD']}:E3"
        added = [key] if led.add(key, nominee="S1ZD", family="DOSE", set="E3", smoke=bool(standin)) else []
        K = led.K
        lo, hi = selcore.interval(r_d1[np.isfinite(r_d1)], K)
        kinfo = {"ledger": ledger_path, "K": K, "key": key, "keys_added": added, "bonferroni_S1ZD_minus_P6_at_K": {"point": d1["point"], "lo": lo, "hi": hi},
                 "note": "K counts every navtest read in the programme's ledger; D1 and D2 are this card's own pre-registered tests (card section 5)"}
    res["K_ledger"] = kinfo
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "READ"
    return res


# ------------------------------------------------------------------------------------------------ markdown
def to_markdown(res):
    L = []
    if res.get("stamp"):
        L += [f"**{res['stamp']}** -- a control run: S1ZD is fed another arm's logits. No number here is a result and none may be quoted as one.", ""]
    L += ["# PREREG_DOSE_P6 read", "", f"**Status: {res.get('status')}**", "", f"- Card `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)",
          f"- Tier: E3 = {R.TIER['E3']}", f"- Evidence class: {R.EVIDENCE}", f"- Estimator: {R.ESTIMATOR}", ""]
    if res.get("status") != "READ":
        L += [f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    ci, b, o = R.ci_s, res["bars_read"], res["outcome"]
    gl = ("all pass" if b["guards"]["all_pass"] else "FAIL " + ",".join(k for k in ("G1", "G2", "G3", "G4") if not b["guards"][k]))
    L += ["## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})", f"- Next lever: {o['next_levers'][o['primary']]}",
          f"- D1 (S1ZD - P6): d {ci(b['D1']['d_S1ZD_minus_P6'])}, bar CI-lo > 0 and d > {b['D1']['threshold_printed']}: **{'PROVEN' if b['D1']['PROVEN'] else 'NOT PROVEN'}**",
          f"- D2 (S1ZD - S1Z recipe mean of S1Z, S1Z1): d {ci(b['D2']['d_S1ZD_minus_S1Z_recipe_mean'])}, bar CI-lo > 0 and d > {b['D2']['threshold_printed']}: **{'PROVEN' if b['D2']['PROVEN'] else 'NOT PROVEN'}**",
          f"- Guards G1 to G4: {gl}; M2 clears (reported): {b['guards']['M2_reported_not_deciding']}",
          f"- Goal {b['goal']['bar']}: S1ZD {b['goal']['S1ZD']:.3f} -> **{b['goal']['status']}**", f"- S1ZD E3 PDMS x100: {R.lvl_s(b['S1ZD_E3_pdms'])}; S1Z recipe mean {R.lvl_s(b['S1Z_recipe_mean_E3_pdms'])}",
          f"- K ledger: K = {res['K_ledger'].get('K')} ({res['K_ledger'].get('note')})", "", "## Runs (E3, paired CI vs P6)", "", "| run | PDMS x100 | vs P6 [95 %] | G1 | G2 | G3 | G4 | changed picks |", "|---|---|---|---|---|---|---|---|"]
    for nm, v in res["runs"].items():
        g = v["guards_by_id"]
        L.append(f"| {nm} | {R.lvl_s(v['E3_pdms'])} | {ci(v['E3_pdms_vs_P6'])} | " + " | ".join("ok" if g[k] else "FAIL" for k in ("G1", "G2", "G3", "G4")) + f" | {v['E3_pick_changed_vs_P6']} |")
    rg = res["regret_split"]
    L += ["", "## Regret split (reported, not deciding; x100 PDMS points)", "", f"S1ZD: pick {rg['S1ZD']['mean_pick_x100']:.3f}, best of 64 {rg['S1ZD']['mean_best_x100']:.3f}, regret {rg['S1ZD']['regret_x100']:.3f}; "
          f"P6: pick {rg['P6']['mean_pick_x100']:.3f}, best {rg['P6']['mean_best_x100']:.3f}, regret {rg['P6']['regret_x100']:.3f}", "",
          "| class | P6 n | P6 contribution | S1ZD n | S1ZD contribution | S1ZD - P6 [95 %] |", "|---|---|---|---|---|---|"]
    for k in CLASSES:
        a, p = rg["S1ZD"]["components_x100"][k], rg["P6"]["components_x100"][k]
        L.append(f"| {CLASS_LABEL[k]} | {p['n_tokens']} | {p['regret_contribution']:.3f} | {a['n_tokens']} | {a['regret_contribution']:.3f} | {ci(rg['S1ZD_minus_P6_contribution_x100'][k])} |")
    L += [f"| total regret | | {rg['P6']['regret_x100']:.3f} | | {rg['S1ZD']['regret_x100']:.3f} | {ci(rg['S1ZD_minus_P6_contribution_x100']['total_regret'])} |", "", "## Trainer held-out curve (monitor, not deciding)", ""]
    tc = res["trainer_heldout_curve"]
    if isinstance(tc, str):
        L.append(f"- {tc}")
    else:
        L += ["| tag | n | base pick | B pick | B - base [95 %] | B zero picks |", "|---|---|---|---|---|---|"]
        for r in tc["rows"]:
            c_ = r["B_minus_base_ci95"] or [None, None]
            L.append(f"| {r['tag']} | {r['n']} | {R.fmt(r['base_pick_x100'], 3)} | {R.fmt(r['B_pick_x100'], 3)} | {R.fmt(r['B_minus_base_mean'], 3)} [{R.fmt(c_[0], 3)}, {R.fmt(c_[1], 3)}] | {r['B_zero_picks']} |")
    L += ["", "## Families (vs P6; STRATEGIC PAUSED)", "", f"- STRATEGIC: {R.STRATEGIC}", "- longitudinal / lateral / tactical components are in the json under runs.S1ZD.four_families_vs_P6; adverse components:"]
    for nm, v in res["runs"].items():
        L.append(f"  - {nm}: {v['four_families_vs_P6']['adverse_components'] or 'none'}")
    if res.get("p6_pass_gate"):
        g = res["p6_pass_gate"]
        L += ["", f"## G0-P6PASS: {'PASS' if g['PASS'] else 'FAIL'} (picks equal {g.get('n_pick_equal_banked_p6')}/{g.get('n_tokens')}; proposals bit-identical {g.get('proposals_bit_identical')}; logits bit-identical {g.get('logits_bit_identical_reported')})"]
    L += ["", "## Ambiguities (this reader)", ""] + [f"- {a}" for a in res["ambiguities_dose"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, "dose_read.json"), os.path.join(out_dir, "dose_read.md")
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=R.jdefault)
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_markdown(res))
    return jp, mp


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read"])
    ap.add_argument("--p6-pass", required=True, help="folder holding P6_control/e3_logits.npz (the dev-box pass)")
    ap.add_argument("--c0", required=True, help="ADDED input: s1z_read needs C0 (P1's zero point)")
    ap.add_argument("--c0r", required=True, help="ADDED input: s1z_read needs C0r (the floor F = |C0r - C0|)")
    ap.add_argument("--s1z", required=True)
    ap.add_argument("--s1z1", required=True)
    ap.add_argument("--s1zd", required=True)
    ap.add_argument("--sft-log", default=None, help="the S1ZD trainer's sft.jsonl / stdout log (held-out curve)")
    ap.add_argument("--table", default=RR.DEFAULT_TABLE)
    ap.add_argument("--p6-proposals", default=R.DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--e2-tokens", default=S.DEFAULT_E2_TOKENS)
    ap.add_argument("--e2-table", default=S.DEFAULT_E2_TABLE)
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--world-dump", default=None)
    ap.add_argument("--c3-tokens", default=None)
    ap.add_argument("--no-tactical", action="store_true")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--control-standin", action="store_true", help="S1ZD is another arm's dir (a control); output only under " + CONTROL_ROOT + "; never the real ledger")
    ap.add_argument("--card", default=CARD, help="for the deliberate-regression control only: a different card file is refused by its sha256")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    return ap


def _under(path, root):
    return path == root or path.startswith(root + "/")


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = os.path.abspath(a.out).replace("\\", "/")
    if a.control_standin:
        if not _under(out, CONTROL_ROOT):
            raise SystemExit(f"REFUSING: --control-standin writes only under {CONTROL_ROOT}")
        if not a.no_ledger and os.path.abspath(a.ledger).replace("\\", "/") == REAL_LEDGER:
            raise SystemExit("REFUSING: a control never touches the real K ledger (use --no-ledger or a COPY)")
    elif _under(out, CONTROL_ROOT):
        raise SystemExit("REFUSING: a real read must not write under the control folder")
    S.wait_free_ram(a.min_free_ram_gb)
    ns = argparse.Namespace(p6_proposals=a.p6_proposals, p6_csv=a.p6_csv, e2_tokens=a.e2_tokens, e2_table=a.e2_table, export=a.export, world_dump=a.world_dump,
                            run=[f"C0={a.c0}", f"C0r={a.c0r}", f"S1Z={a.s1z}", f"S1Z1={a.s1z1}", f"S1ZD={a.s1zd}"], c3_tokens=a.c3_tokens, no_tactical=a.no_tactical)
    inp = RR.make_inputs(ns)
    pc = Path(a.p6_pass) / "P6_control" / "e3_logits.npz"
    inp.provenance["p6_control_sha"] = sha256_file(pc) if pc.exists() else None
    table = RR.Table(a.table)
    ledger = None if a.no_ledger else a.ledger
    res = dose_read(inp, table, p6_pass_dir=a.p6_pass, sft_log=a.sft_log, ledger_path=ledger, standin=a.control_standin, card=a.card)
    jp, mp = write_outputs(res, a.out)
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "outcome": res.get("outcome", {}).get("primary"), "K": res.get("K_ledger", {}).get("K"),
                      "stamp": res.get("stamp"), "json": jp, "md": mp}, indent=1, default=R.jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
