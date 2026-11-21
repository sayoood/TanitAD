"""ds_read_b3.py -- the DSCALE10K BLOCK-3 read: the H-DATA-SCALE-1 sub-read (PREREG_V8F_DSCALE10K section 4 "H-DATA-SCALE-1 sub-read", section 6 label line; frozen card; MM rulings 2026-11-16 / 11-17 / 11-18).
CPU, zero training.  Writes RESULT_DSCALE10K_B3.json + .md.  STAGE A ONLY, SEED 0 ONLY: UE_s0 (U set, 4 union epochs = 61,568 steps) and BL_s0 (B set, the same 61,568 steps) against the block-1 stage-A arms B_s0, U_s0, B_rep.
CARD TEXT (CARD_DATASCALE_TRAIN.md lines 195-197, quoted as the rule):
  Delta_data = U_s0 - B_s0 (fixed steps), Delta_ep = BL_s0 - B_s0 (fixed data, about 3.2x steps), and UE_s0 - BL_s0.  Read on ADE6 against 2F_A (F_A = the stage-A ADE6 floor from B_rep), with T's latcons beside.
  Labels: DATA > EPOCHS (U - BL better by more than 2F_A); EPOCHS > DATA (the reverse); NEITHER.   Line 276: a label set on one seed is reported as one seed.
READING CHOICES (the card is silent; each is a named choice for the MM, never a moved bar)
  1. ADE6 = the per-window `ade6` of the T0 E0 eval of every arm (the DSCALE instrument's own ADE6, the quantity named at card line 27 / 196); LOWER is better.  F_A = abs(mean ADE6(B_rep) - mean ADE6(B_s0)) over the same
     141 episodes / 2,399 windows (one replicate pair, seed 2 against seed 0; a weak floor).  The bar is magnitude only, as written: "better by more than 2F_A", strict; the paired interval is printed beside, not added as a condition.
     A floor that is not a positive finite number (a bit-identical replicate) gives NOT-YET: it is not a floor.
  2. "U - BL" (line 197) read LITERALLY = U_s0 - BL_s0 (ADE6 at U's own 19,164 steps against BL's 61,568): DATA > EPOCHS iff ADE6(U_s0) is lower than ADE6(BL_s0) by more than 2F_A (= Delta_data - Delta_ep).
     The same rule is also run on the line-195 third quantity UE_s0 - BL_s0 (data at the larger fixed compute).  If the two readings give different labels the label is READING-SENSITIVE: both are printed, neither is quoted
     alone (the UNIT-SENSITIVE precedent of MM_RULING_GTFIX); the literal reading is kept in `label_literal`.
  3. The four families are read at stage A, T1-kin (LONGITUDINAL / LATERAL / TACTICAL) and T0 (STRATEGIC, route passes made for the five stage-A arms, with K-EVALX), for every pair, with the card's cell rule and F_cell = abs(B_rep - B_s0) per
     cell at stage A; they are reported beside the label and never pooled.  T's latcons (T0, beside) per pair with the card's T-DATA-LIMITED rule.
  4. Controls: K1 self-pair, T0 identity, K-RE (B1) of the block-1 stage-A rows (T1 cells, mechanism rows, ADE6 means), K-RE (B2) of the block-2 copy of the same rows, K-EVALX at stage A.  The seed-1 rows of block 2 are not inputs.
TIERS: ADE6 / mechanism = T0 (world-model diagnostic, never driving performance); LONGITUDINAL / LATERAL / TACTICAL = T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene);
STRATEGIC = T0 at the window's first tick.  ESTIMATOR: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); training-run variance = F (one replicate pair); ONE SEED.
Usage:  python ds_read_b3.py --out RESULT_DSCALE10K_B3.json [--b1-result ...] [--b2-result ...] [--n-boot 2000]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R  # noqa: E402  (block-1 module, unedited: pure logic, loaders)
import ds_read_b2 as B2  # noqa: E402  (block-2 module, unedited: K-RE helpers, UNIT-SENSITIVE, md_cell)

OR, FAMILIES, HEADLINE, FAMILY = R.OR, R.FAMILIES, R.HEADLINE, R.FAMILY
LABELS = ("DATA > EPOCHS", "EPOCHS > DATA", "NEITHER")
ARMS_A = ("B_s0", "U_s0", "B_rep", "UE_s0", "BL_s0")
PAIRS = {                                    # pair name -> (arm, reference); stage A, seed 0 (the card's three quantities, the literal "U - BL", and two beside)
    "U-B_s0.A": ("U_s0", "B_s0"),            # Delta_data (fixed steps); K-RE against block 1
    "BL-B_s0.A": ("BL_s0", "B_s0"),          # Delta_ep (fixed data, 3.2x steps)
    "UE-BL.A": ("UE_s0", "BL_s0"),           # the line-195 third quantity: data at the larger fixed compute
    "U-BL.A": ("U_s0", "BL_s0"),             # "U - BL" read literally (Delta_data - Delta_ep)
    "UE-B_s0.A": ("UE_s0", "B_s0"),          # beside: data plus compute
    "UE-U_s0.A": ("UE_s0", "U_s0"),          # beside: compute on the union
}
LABEL_PAIRS = ("U-BL.A", "UE-BL.A")          # the two readings of the label rule
NEXT = {"DATA > EPOCHS": "Updates the H-DATA-SCALE-1 register row: more unique clips beat more steps at fixed data (one seed).",
        "EPOCHS > DATA": "Updates the H-DATA-SCALE-1 register row: more steps at fixed data beat more unique clips (one seed).",
        "NEITHER": "Updates the H-DATA-SCALE-1 register row: neither lever separates from the other by more than 2F_A on ADE6 (one seed).  Card section 6 DATA-NEUTRAL: read `ue_minus_bl_better_than_2FA`.",
        "READING-SENSITIVE": "The two readings of 'U - BL' disagree: the MM rules which one the register row takes (both are printed, neither is quoted alone)."}


# ------------------------------------------------------------------------------------------------------------------------------------------ pure logic
def verdict(delta, f_a):
    """delta = ADE6(first) - ADE6(second) in metres (LOWER ADE is better); f_a = F_A.  -> 'DATA > EPOCHS' when first is better by MORE than 2F_A (strict), 'EPOCHS > DATA' when worse by more than 2F_A, else 'NEITHER'.
    A floor that is not a positive finite number reads NOT-YET (a bit-identical replicate is not a floor); a NaN delta reads NOT-YET."""
    if f_a is None or not math.isfinite(f_a) or f_a <= 0.0 or delta is None or not math.isfinite(delta):
        return "NOT-YET"
    if delta < -2.0 * f_a:
        return "DATA > EPOCHS"
    if delta > 2.0 * f_a:
        return "EPOCHS > DATA"
    return "NEITHER"


def data_scale_label(d_u_bl, d_ue_bl, f_a):
    """the H-DATA-SCALE-1 label from the two readings (choice 2).  -> {label, label_literal, label_ue_bl, reading_sensitive, two_f_a, ...}.  Any NOT-YET reading gives NOT-YET."""
    lit, alt = verdict(d_u_bl, f_a), verdict(d_ue_bl, f_a)
    out = {"label_literal": lit, "label_ue_bl": alt, "delta_ade6_U_minus_BL": d_u_bl, "delta_ade6_UE_minus_BL": d_ue_bl, "F_A": f_a, "two_F_A": None if f_a is None else 2.0 * f_a,
           "ue_minus_bl_better_than_2FA": bool(alt == "DATA > EPOCHS")}
    if "NOT-YET" in (lit, alt):
        out.update({"label": "NOT-YET", "reading_sensitive": False})
    elif lit != alt:
        out.update({"label": "READING-SENSITIVE", "reading_sensitive": True, "labels_by_reading": {"U_s0 - BL_s0 (literal)": lit, "UE_s0 - BL_s0": alt}})
    else:
        out.update({"label": lit, "reading_sensitive": False})
    return out


def outcome_b3(lab, blocked=None, ade_given=True, problems=None):
    """the outcome of the read: BLOCKED (a refused bank item) first, then NOT-YET (ADE6 unread), then the label; a failed control makes the label not quotable"""
    if blocked:
        return ["BLOCKED"]
    if not ade_given or lab is None:
        return ["NOT-YET"]
    if problems:
        return [f"{lab['label']} (CONTROL FAILED: {len(problems)} problem(s); not quotable)"]
    return [lab["label"]]


def kevalx_problem_b3(t0route, arms=("B_s0", "U_s0", "B_rep", "UE_s0", "BL_s0")):
    """-> reason or None: out/t0route/KEVALX.json is a PASS over the real and donor passes of every stage-A arm of this read (50c70ae8 vs 008f48b7)"""
    p = f"{t0route}/KEVALX.json"
    if not os.path.isfile(p):
        return "KEVALX.json missing"
    k = json.load(open(p, encoding="utf-8"))
    want = {f"{a}.A {v}" for a in arms for v in ("real", "donor")}
    if not k.get("pass") or set(k.get("items", {})) != want or not all(v["identical"] for v in k["items"].values()):
        return f"KEVALX.json is not a PASS over {sorted(want)}"
    return None


def route_problem_b3(t0route, arms=ARMS_A):
    """-> reason or None: every stage-A route pass (real, donor, zero, realx, donorx) of every arm exists, and K-EVALX passed"""
    miss = [f"{a}.A {v}" for a in arms for v in B2.ROUTE_VARIANTS if not os.path.exists(f"{t0route}/eval_DS-{a}-A_{v}.npz")]
    if miss:
        return f"T0 route passes missing: {miss}"
    why = kevalx_problem_b3(t0route, arms)
    return f"K-EVALX: {why} (the zero passes are not admissible)" if why else None


def pair_family_status(rows):
    """one pair's U-style cell rows -> {family: family_status} (the card's family rule, headline cells, unedited)"""
    return {f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}


def ade_means(Z, names):
    return {n: float(np.asarray(Z[(n, 0)]["ade6"], np.float64).mean()) for n in names}


def kre_ade6(recomputed_means, banked_beside):
    """K-RE of the block-1 ADE6 means: U_s0 (A) and B_s0 (B) equal the banked mechanism block's beside.ade6 exactly"""
    n, bad = R.kre_compare({"ade6": {"A": recomputed_means["U_s0"], "B": recomputed_means["B_s0"]}}, {"ade6": {"A": banked_beside["ade6"]["A"], "B": banked_beside["ade6"]["B"]}}, ("A", "B"))
    return {"pass": bool(n == 2 and not bad), "rows_compared": n, "mismatches": bad}


def fam_rows_all(pairs, strat):
    """per pair the cell rows of the four families (stage A): T1-kin cells from `pairs`, STRATEGIC cells from `strat` when it was read"""
    out = {}
    for p, rows in pairs.items():
        r = dict(rows)
        if strat.get("status") == "OK" and p in strat["pairs"]:
            r.update(strat["pairs"][p])
        out[p] = r
    return out


def md_pair_rows(rows, cells):
    out = ["| cell | Delta [95 % interval] sep F verdict | n windows |", "|---|---|---|"]
    for c in cells:
        v = rows.get(c)
        out.append(f"| {c} | {B2.md_cell(v)} | {(v or {}).get('n_windows', 0)} |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1-result", default=L.B1_ROOT + "/out/RESULT_DSCALE10K_B1.json")
    ap.add_argument("--b2-result", default="F:/TanitAD-scratch/datascale/read2/out/RESULT_DSCALE10K_B2.json")
    ap.add_argument("--dumps0", default=L.B1_ROOT + "/t1", help="block-1 T1 dumps (B_s0.A, U_s0.A, B_rep.A)")
    ap.add_argument("--dumps3", default=L.T1, help="block-3 T1 dumps (UE_s0.A, BL_s0.A)")
    ap.add_argument("--t0mech", default=L.OUT + "/t0mech", help="ONE folder holding every arm's stage-A mechanism evals (the block-1 files copied in by sync_b1.py, the block-3 ones written here)")
    ap.add_argument("--t0route", default=L.OUT + "/t0route")
    ap.add_argument("--init-record", default="F:/TanitAD-scratch/datascale/gate/pod_out/gate/init.json")
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args(argv)
    t_start = time.time()
    CI = L.load_ci()
    dpath = lambda arm, st="A": f"{a.dumps3 if arm in L.ARMS_B3 else a.dumps0}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    ready = {L.item(arm, "A"): os.path.isdir(dpath(arm)) for arm in ARMS_A}
    rep = {"tool": "ds_read_b3.py", "card": "PREREG_V8F_DSCALE10K (frozen; sha256 48a74901...; MM rulings 2026-11-16 / 11-17 / 11-18)", "block": 3, "seed": "0 (ONE SEED: reported as one seed)", "code_of_record": L.COMMIT,
           "hypothesis": "H-DATA-SCALE-1 (stage A): DATA > EPOCHS / EPOCHS > DATA / NEITHER", "steps": {"UE_s0": L.STEPS_ARM["UE_s0"], "BL_s0": L.STEPS_ARM["BL_s0"], "B_s0 / U_s0 / B_rep": L.STEPS["A"]},
           "tier": {"ADE6 label / mechanism (T's latcons)": "T0 (world-model and tactical diagnostic, never driving performance)",
                    "LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC": "T0 at the window's first tick (--ages fresh)"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training); block-1 stage-A rows reproduced under K-RE (B1)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = F (one replicate pair, B_rep); ONE seed",
           "committed_rules": __doc__.split("CARD TEXT")[1].split("Usage:")[0].strip(), "items_ready": ready, "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    miss = [k for k, v in ready.items() if not v]
    if miss:
        rep["outcome"], rep["reason"] = ["NOT-YET"], f"T1 dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", miss)
        return 1
    b1 = json.load(open(a.b1_result, encoding="utf-8")) if os.path.isfile(a.b1_result) else None
    b2 = json.load(open(a.b2_result, encoding="utf-8")) if os.path.isfile(a.b2_result) else None
    if b1 is None:
        prob.append(f"K-RE (B1) NOT-GIVEN: {a.b1_result} is missing, so no block-1 stage-A row can be checked")
    import obd_geom as G
    geom = G.build()
    dg = R.load_dump_n(dpath("B_s0"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm in ARMS_A:
        cells[arm] = R.load_arm_cells(geom, dg, J, glat, glon, dpath(arm), L.EVAL_N_EPISODES)
        print(f"loaded {arm}.A {time.time() - t_start:.0f}s", flush=True)
    eid = dg["eid"]
    rep["controls"]["K1_self_pair"] = OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    F = OR.floors_from(cells["B_rep"], cells["B_s0"])
    rep["floors"] = {"definition": "F_cell = |B_rep - B_s0| per cell at stage A (one replicate pair, B_rep seed 2)", "stage_A": F}
    pairs = {p: R.pair_cells(CI, cells[x], cells[y], eid, F, a.n_boot) for p, (x, y) in PAIRS.items()}
    pairs["B_rep-B_s0.A"] = R.pair_cells(CI, cells["B_rep"], cells["B_s0"], eid, None, a.n_boot)
    # K-RE (B1) and (B2): the stage-A T1 rows of U - B_s0 recomputed through this pipeline equal the banked ones, every field exactly
    rows_rec = R.slim(pairs["U-B_s0.A"])
    kre = {}
    if b1 is not None:
        kre["B1 U-B_s0.A (T1 cells)"] = B2.kre_b1(rows_rec, b1["pairs"]["U-B_s0.A"])
    if b2 is not None and "U-B_s0.A" in b2.get("pairs", {}):
        kre["B2 U-B_s0.A (T1 cells)"] = B2.kre_b1(rows_rec, b2["pairs"]["U-B_s0.A"])
    else:
        rep["controls"]["K-RE_B2_note"] = "the block-2 result is not present: K-RE (B2) NOT-GIVEN (block-1 rows are the reference)"
    # STRATEGIC at stage A (T0 route passes)
    SS, Gr, ci, t = R.load_strategic_helpers()
    strat = {"tier": rep["tier"]["STRATEGIC"], "headline": list(HEADLINE["STRATEGIC"]), "status": "NOT-YET", "not_in_this_card": {
        "detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"},
        "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}},
        "note": "the C0-type v9 nav shares its source with RC-A, so S1 is echo-prone for every arm alike: the Delta between arms is read, never the level; block 1 / 2 made no stage-A route pass, so there is no banked stage-A STRATEGIC row (K-RE does not apply; K-EVALX is the control)"}
    why = route_problem_b3(a.t0route)
    if why:
        strat["reason"] = why
    else:
        R_ = {arm: R.route_arrays(a.t0route, arm, "A", SS, Gr, ci, t) for arm in ARMS_A}
        sf = R.strat_floors(R_["B_rep"], R_["B_s0"])
        sp = {p: R.strat_pair(CI, R_[x], R_[y], eid, sf, a.n_boot) for p, (x, y) in PAIRS.items()}
        strat.update({"status": "OK", "floors": sf, "n_scorable_80m": {arm: int(np.isfinite(R_[arm]["S_CKPT_real@80"]).sum()) for arm in R_}, "best_constant_80m_reference": 0.794,
                      "pairs": {p: R.slim(v) for p, v in sp.items()}})
        for p in PAIRS:
            pairs[p].update({c: v for c, v in sp[p].items() if c in HEADLINE["STRATEGIC"]})
    # T0: the ADE6 label, T's latcons beside
    import ds_t0 as T
    nm = lambda arm: T.name_of(arm, "A")  # noqa: E731
    HR, GR, PGT, WC, pairfn = T.modules()
    mech, lab, ade_given = {"tier": rep["tier"]["ADE6 label / mechanism (T's latcons)"]}, None, False
    ade = {"tier": mech["tier"], "definition": "ADE6 = the per-window `ade6` of each arm's T0 E0 eval (DSCALE instrument); lower is better; F_A = |mean ADE6(B_rep) - mean ADE6(B_s0)|"}
    try:
        Zn, eid0 = PGT.load_all(a.t0mech, (0,), tuple(nm(x) for x in ARMS_A), WC)
        Z = {(x, 0): Zn[(nm(x), 0)] for x in ARMS_A}
        means = ade_means(Z, ARMS_A)
        f_a = float(abs(means["B_rep"] - means["B_s0"]))
        ade.update({"mean_ade6_m": means, "F_A": f_a, "n_windows": int(len(eid0)), "n_episodes": int(len(np.unique(eid0)))})
        ade["pairs"] = {p: dict(PGT.paired(pairfn, Z[(x, 0)]["ade6"], Z[(y, 0)]["ade6"], eid0, a.n_boot), pair=f"{x} - {y}") for p, (x, y) in PAIRS.items()}
        lab = data_scale_label(ade["pairs"]["U-BL.A"]["point"], ade["pairs"]["UE-BL.A"]["point"], f_a)
        lab["interval_beside"] = {p: {"point": ade["pairs"][p]["point"], "lo": ade["pairs"][p]["lo"], "hi": ade["pairs"][p]["hi"], "separated": bool(ade["pairs"][p]["separated"])} for p in LABEL_PAIRS}
        lab["delta_data_U_minus_B"], lab["delta_ep_BL_minus_B"] = ade["pairs"]["U-B_s0.A"]["point"], ade["pairs"]["BL-B_s0.A"]["point"]
        ade_given = True
        if b1 is not None and "U_vs_B_stage_A" in b1.get("mechanism_stage_A_T0", {}):
            ade["K-RE_B1"] = kre_ade6(means, b1["mechanism_stage_A_T0"]["U_vs_B_stage_A"]["beside"])
            kre["B1 ADE6 means (U_s0, B_s0)"] = ade["K-RE_B1"]
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        ade["status"] = "NOT-YET"
        ade["reason"] = f"{type(e).__name__}: {e}"
    rep["ade6_stage_A_T0"] = ade
    # T's latcons beside (card: "with T's latcons beside"); the deciding frame of each pair is its reference arm's
    mech_ref = {"U-B_s0.A": "B_s0", "BL-B_s0.A": "B_s0", "UE-BL.A": "BL_s0", "U-BL.A": "BL_s0", "UE-B_s0.A": "B_s0", "UE-U_s0.A": "U_s0"}
    try:
        fl = T.mech_floor(a.t0mech, nm("B_rep"), nm("B_s0"))
        mech["stage_A_T_floor"] = fl
        mech["pairs"] = {}
        for p, (x, y) in PAIRS.items():
            m = T.mech_pair(a.t0mech, nm(x), nm(y), 0, a.n_boot, ref_name=nm(mech_ref[p]))
            mech["pairs"][p] = {"r": m["r"], "rmse_slot": m["rmse_slot"], "mse_delta": m["mse_delta"], "t_rule": B2.label_by_units(T.decide_t, m, fl["floor_mse"]), "frame": m.get("frame"), "beside": m["beside"]}
        r_u = T.mech_pair(a.t0mech, nm("R_s0"), nm("U_s0"), 0, a.n_boot, ref_name=nm("U_s0"))
        mech["identity_control"] = T.identity_control(a.t0mech, nm("BL_s0"))
        if not mech["identity_control"]["pass"]:
            prob.append("K1-T0: an arm against itself did not read r exactly 0 [0, 0]")
        if b1 is not None and "reach" in b1.get("mechanism_stage_A_T0", {}):
            u_b = T.mech_pair(a.t0mech, nm("U_s0"), nm("B_s0"), 0, a.n_boot)
            kre["B1 mechanism rows (U vs B, R vs U, floor)"] = B2.kre_mech({"U_vs_B_stage_A": u_b, "R_vs_U_stage_A": r_u, "stage_A_T_floor": fl}, b1["mechanism_stage_A_T0"])
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        mech["status"] = "NOT-YET"
        mech["reason"] = f"{type(e).__name__}: {e}"
    rep["mechanism_stage_A_T0"] = mech
    rep["controls"]["K-RE"] = {"pass": bool(kre) and all(v["pass"] for v in kre.values()), "per_check": kre}
    if not rep["controls"]["K-RE"]["pass"]:
        prob.append("K-RE: a recomputed block-1 stage-A row differs from the banked row, or no banked row was available")
    blocked = {L.item(x, "A"): L.bank_problem(x, "A") for x in ARMS_A if L.bank_problem(x, "A")}
    rep["blocked_items"] = blocked
    labels = outcome_b3(lab, list(blocked) or None, ade_given, prob)
    rep["outcome"] = labels
    rep["h_data_scale_1"] = lab
    rep["next_step"] = [f"{x.split(' (')[0]}: {NEXT[x.split(' (')[0]]}" for x in labels if x.split(" (")[0] in NEXT] or ["no label: see the reason fields"]
    # the four families per pair (stage A), never pooled
    fam_rows = fam_rows_all(pairs, strat)
    rep["family_status"] = {p: pair_family_status(rows) for p, rows in fam_rows.items() if p in PAIRS}
    n_w, n_e = int(len(eid)), int(len(np.unique(eid)))
    rep["families"] = {}
    for f in FAMILIES:
        rep["families"][f] = {"tier": rep["tier"]["STRATEGIC"] if f == "STRATEGIC" else rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "headline": list(HEADLINE[f]),
                              "cells_stage_A": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(rows, f).items()} for p, rows in fam_rows.items() if p in PAIRS},
                              "status_stage_A": {p: rep["family_status"][p][f] for p in PAIRS}, "n_windows": n_w, "n_episodes": n_e, "estimator": "paired episode-cluster bootstrap", "evidence_class": "MEASURED"}
    rep["families"]["STRATEGIC"]["block"] = strat
    rep["families"]["PRIMARY (ADE / FDE, beside, decide nothing)"] = {"cells_stage_A": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(rows, "PRIMARY").items()} for p, rows in fam_rows.items() if p in PAIRS}}
    rep["pairs"] = {p: R.slim(v) for p, v in pairs.items()}
    rep["not_built_beside_rows"] = {"DK-VIOL counts": "NOT COMPUTABLE here: beside row (OBD o4), outside the 24 cells; routed to the W-FAM-STRAT1-TTC-TACCONF instrument (MM ruling, block 2)",
                                    "T-LAT-F1 / T-LON-F1, goal macro-F1 by provenance, nav-echo split": f"NOT COMPUTABLE here (n={n_w} windows, heads are in the --dump-heads dumps): same instrument, same ruling",
                                    "per-country / platform eval breakdown (card 8)": "named work item, not in the cell set",
                                    "stage B": "UE_s0 and BL_s0 have no stage B (card 2.3: stage A only); every number here is stage A"}
    rep["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    tl = lambda p: B2._lab(mech.get("pairs", {}).get(p, {}).get("t_rule"))  # noqa: E731
    md = ["# DSCALE10K block 3: the H-DATA-SCALE-1 sub-read (UE_s0, BL_s0 against B_s0, U_s0, B_rep), stage A, ONE SEED (seed 0), MEASURED", "",
          f"Outcome: **{labels}**. Controls: problems {prob or 'none'}.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B {a.n_boot} seed 0; tiers: ADE6 and T's latcons T0, LONGITUDINAL / LATERAL / TACTICAL T1-kin, STRATEGIC T0.", ""]
    if lab:
        md += ["## H-DATA-SCALE-1 label (card line 195-197: ADE6 against 2F_A; lower ADE is better)", "",
               f"* F_A = {lab['F_A']:.4f} m (|ADE6(B_rep) - ADE6(B_s0)|), 2F_A = {lab['two_F_A']:.4f} m.  One replicate pair: a weak floor.",
               f"* Delta_data U_s0 - B_s0 = {lab['delta_data_U_minus_B']:+.4f} m; Delta_ep BL_s0 - B_s0 = {lab['delta_ep_BL_minus_B']:+.4f} m.",
               f"* U_s0 - BL_s0 (literal 'U - BL') = {lab['delta_ade6_U_minus_BL']:+.4f} m -> **{lab['label_literal']}**; UE_s0 - BL_s0 = {lab['delta_ade6_UE_minus_BL']:+.4f} m -> **{lab['label_ue_bl']}**.",
               f"* Label: **{lab['label']}**" + (f" {lab.get('labels_by_reading')}" if lab.get("reading_sensitive") else "") + ".  Intervals beside: " + json.dumps(lab["interval_beside"]), ""]
    md += ["T's latcons (T0, beside; T-DATA-LIMITED rule of the card, per pair): " + "; ".join(f"{p}: {tl(p)}" for p in PAIRS), "",
           "Family status at stage A per pair: " + json.dumps(rep["family_status"] and {p: {f: v['status'] for f, v in s.items()} for p, s in rep["family_status"].items() if p in LABEL_PAIRS + ("U-B_s0.A", "BL-B_s0.A")}), ""]
    for f in FAMILIES:
        for p in ("U-B_s0.A", "BL-B_s0.A", "U-BL.A", "UE-BL.A"):
            md += [f"## {f} ({rep['families'][f]['tier']}): {p} = {PAIRS[p][0]} - {PAIRS[p][1]}, n = {n_w} windows / {n_e} episodes", "", md_pair_rows(rep["families"][f]["cells_stage_A"][p], list(rep["families"][f]["cells_stage_A"]["U-B_s0.A"])), ""]
    md += ["## STRATEGIC members not in this card", "", json.dumps(strat["not_in_this_card"])]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", labels, "| problems:", len(prob), "|", rep["wall_s"], "s")
    return 1 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
