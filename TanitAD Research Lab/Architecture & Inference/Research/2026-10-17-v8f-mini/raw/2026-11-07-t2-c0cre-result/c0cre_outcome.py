"""c0cre_outcome.py: PREREG_T2_RENC.md section 6, applied mechanically to the stage-5 reads of C0C-RE.  Standard library only; run as `python -I c0cre_outcome.py ...`.

Inputs (all written by c0cre_read.sh; nothing is recomputed here):
  --dom         score_dom/DOMINANCE_T2.json            the pinned read (pin 4, sha256 7fd65ffd...) of C0C-RE (stage 5) against the floor P10
  --pair        pair_C0C-RE_C0C/PAIR_T2.json           the pair read C0C-RE - C0C (paired scene-cluster bootstrap, B 2000, seed 0, all 19 rows, both seeds)
  --c0c-dom     stage4/score_q_identity/DOMINANCE_T2.json   the pin-4 read of C0C against P10 (the reference for "no NEW separated-worse row against P10")
  --blind-pair  pair_BLIND_C0C-RE/PAIR_T2.json         OPTIONAL, beside, decides nothing: C0C-RE-BLIND - C0C-RE
  --card        PREREG_T2_RENC.md                      its first 5,544 bytes must hash to 425d3496...
  --out-dir     where RESULT_T2_RENC.json and RESULT_T2_RENC.md are written

Section 6, in the card's order (the first that holds names the outcome):
  1. RE-DOMINANT  the pinned read gives DOMINANT-T2 (both seeds).  Reported as written; NOT counted as the scale signal until a replicate (folds swapped by the parity of the
                  second hex digit, encoders retrained) reads DOMINANT too.
  2. RE-MOVES     C0C-RE - C0C is separated-better on ADE6 at both seeds, AND M1 crawl share of that pair is not separated-worse at either seed, AND no row is separated-worse
                  against P10 for C0C-RE that is not separated-worse against P10 for C0C (same seed).
  3. RE-NULL or RE-HARMS  anything else.  The card does not split the two; this script labels RE-HARMS when ANY of the three RE-MOVES clauses fails on its WORSE side
                  (ADE6 separated-worse at either seed; M1 separated-worse in the pair at either seed; a new separated-worse row against P10 at either seed) and RE-NULL otherwise.
                  The split is a label only: both carry the same committed next lever.
  RE-FAILS-MECH (section 4) is not reachable from these inputs: M-RENC and G-KEEP passed on 2026-11-07 12:24 (card log).
A row's direction is never re-derived here: the verdict strings (separated-better / separated-worse / unseparated) are the pinned reader's own `family_table`.
Reading choices the card leaves open, fixed here: (i) "M1" in clause 2 is the M1 crawl share row of the PAIR C0C-RE - C0C (the sentence's subject); M1 against P10 is covered by the
third clause and is shown beside as a sensitivity; (ii) the RE-HARMS / RE-NULL split above.  Neither moves a bar: each only labels what the card already calls "anything else".
Integrity: every cross-check that fails makes the outcome INPUT-INVALID (exit 3) instead of a guess; a failed gate in a pinned read gives NOT-READ (exit 2).
STRATEGIC is PAUSED (PI ruling item 25): shown, never scored as a pass.  Tier T2.  The interval answers another draw of SCENES only.
"""
import argparse
import hashlib
import json
import os
import sys

DL = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
CARD = f"{DL}/t2_prereg/PREREG_T2_RENC.md"
CARD_BYTES = 5544
CARD_SHA = "425d3496eb52e30025c11c2609409ea1478aa2fe70ba6609c9f9477c128fd722"
PIN4_SHA = "7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9"
ARM, BLIND, BASE = "C0C-RE", "C0C-RE-BLIND", "C0C"
F_T1KIN_M = 0.12074
SEEDS = ("s0", "s1")
FAMILY_ORDER = ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "crawl")
ROWS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE",
        "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")
VERDICTS = ("separated-better", "separated-worse", "unseparated")
TIER = "T2 re-perception (agents replayed at log time; not a reactive-traffic result)"
ESTIMATOR = "paired scene-cluster bootstrap, B 2000, seed 0: answers another draw of SCENES only"
SB, SW, UN = VERDICTS
NEXT = {
    "RE-DOMINANT": "Reported as written, with renders beside P10's. NOT counted as the scale signal until a replicate reads DOMINANT-T2 too: the folds swapped by the parity of the second hex digit of the "
                   "scene id, and the encoders retrained (card 6.1).",
    "RE-MOVES": "The recipe joins the programme as a render-robustness step for any vision arm, the LoRA arm L included (trained on top of L's merged encoder, same folds). If the T1-kin read of C0 on "
                f"eval-141 re-encoded through E' also holds |ADE6 delta| < F = {F_T1KIN_M} m at both seeds, it is reported as a candidate encoder change for the driving model, under its own card (card 6.2).",
    "RE-NULL": "The next lever is the planner rather than the encoder: post-training C0 on rendered on-policy states of one fold (DAgger at T2), read on the other fold, under its own card (card 6.3).",
    "RE-HARMS": "The next lever is the planner rather than the encoder: post-training C0 on rendered on-policy states of one fold (DAgger at T2), read on the other fold, under its own card (card 6.3).",
}
DOES_NOT_SAY = [
    "The interval answers another draw of SCENES. It does not answer another TRAINING run (one adaptation seed; H-ESTIM-SEED-1): a separated row is necessary, not sufficient, for a lever claim. Inference is deterministic.",
    "STRATEGIC is PAUSED (PI ruling item 25). Its rows are shown and are not a pass; no strategic claim follows from this read. LONGITUDINAL, LATERAL and TACTICAL keep full force.",
    "Tier T2 on the refusal-surviving scenes of the 62 (selected on the human's logs, launch scenes over-represented). Not a reactive-traffic result and not a statement about PhysicalAI-AV at large.",
    "It says nothing about real-frame driving: the T1-kin read of C0 on eval-141 re-encoded through E' (card section 5, beside) is a separate read and is not part of this script.",
    "C0C-RE-BLIND - C0C-RE (beside) decides nothing; the reading rule of the LoRA card's open item 5(b) is not applied here.",
    "Render panels beside P10's (card 6.1) are not produced by this script.",
]


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.load(open(path, encoding="utf-8"))


def card_ok(path):
    if not os.path.isfile(path):
        return False, None
    b = open(path, "rb").read()
    got = hashlib.sha256(b[:CARD_BYTES]).hexdigest()
    return len(b) >= CARD_BYTES and got == CARD_SHA, got


def table(read, seed):
    """{row: {family, verdict, delta, lo, hi, separated}} of one seed of a pinned-read or pair JSON, from its `family_table` and `rows`."""
    ft, rows = read["by_seed"][seed]["family_table"], read["by_seed"][seed]["rows"]
    out = {}
    for fam, lst in ft.items():
        for e in lst:
            out[e["row"]] = {"family": fam, "verdict": e["verdict"], "delta": e["delta"], "lo": e["lo"], "hi": e["hi"], "separated": rows.get(e["row"], {}).get("separated")}
    return out


def sw_rows(tab):
    return sorted(r for r, v in tab.items() if v["verdict"] == SW)


def table_problems(tag, tab, read_a=None):
    """Internal consistency of one seed's table: the 19 rows, a legal verdict, `separated` agreeing with the verdict, and (pinned reads) read_a's row lists agreeing with the verdicts."""
    p = []
    if sorted(tab) != sorted(ROWS):
        p.append(f"{tag}: row set differs from the 19 rows (missing {sorted(set(ROWS) - set(tab))}, extra {sorted(set(tab) - set(ROWS))})")
    for r, v in tab.items():
        if v["verdict"] not in VERDICTS:
            p.append(f"{tag}: row {r!r} has verdict {v['verdict']!r}")
        elif v["separated"] is not (v["verdict"] != UN):
            p.append(f"{tag}: row {r!r} verdict {v['verdict']!r} but rows[...].separated is {v['separated']!r}")
    if read_a is not None:
        if sorted(read_a.get("separated_worse_rows", [])) != sw_rows(tab):
            p.append(f"{tag}: read_a.separated_worse_rows {sorted(read_a.get('separated_worse_rows', []))} differ from the family_table verdicts {sw_rows(tab)}")
        sb = sorted(r for r, v in tab.items() if v["verdict"] == SB)
        if sorted(read_a.get("separated_better_rows", [])) != sb:
            p.append(f"{tag}: read_a.separated_better_rows differ from the family_table verdicts")
    return p


def gates_state(read):
    g = read.get("gates") or {}
    return {k: bool(v.get("PASS")) for k, v in g.items()}, {k: "; ".join(v.get("why", []))[:300] for k, v in g.items() if not v.get("PASS")}


# ------------------------------------------------------------------------------------------------ the evaluation (pure)
def evaluate(dom, pair, c0c_dom, blind_pair=None, card=(True, CARD_SHA), names=None):
    """-> result dict.  `names` = {input name: path/sha} for provenance only."""
    prob, warn = [], []
    res = {"card": "PREREG_T2_RENC", "card_prefix_bytes": CARD_BYTES, "card_prefix_sha256_expected": CARD_SHA, "card_prefix_sha256_now": card[1], "card_frozen_ok": bool(card[0]),
           "tier": TIER, "estimator": ESTIMATOR, "inputs": names or {}, "arm_checked": ARM, "rehearsal_on_other_arm": ARM != "C0C-RE", "strategic": "PAUSED (PI ruling item 25): shown, never scored as a pass"}
    if not card[0]:
        prob.append("the first 5,544 bytes of PREREG_T2_RENC.md do not hash to 425d3496...")
    # ---- identity of the inputs
    if dom.get("arm") != ARM or dom.get("arm_stage") != 5:
        prob.append(f"dom: arm {dom.get('arm')!r} stage {dom.get('arm_stage')!r}, want {ARM!r} stage 5")
    if dom.get("script_sha256") != PIN4_SHA:
        prob.append(f"dom: script_sha256 {str(dom.get('script_sha256'))[:12]} is not pin 4 ({PIN4_SHA[:8]})")
    if "B 2000, seed 0" not in str(dom.get("estimator")):
        prob.append("dom: estimator is not 'B 2000, seed 0'")
    if pair.get("pair") != f"{ARM} - {BASE}" or (pair.get("arm_a") or {}).get("arm") != ARM or (pair.get("arm_a") or {}).get("stage") != 5 \
            or (pair.get("arm_b") or {}).get("arm") != BASE or (pair.get("arm_b") or {}).get("stage") != 4:
        prob.append(f"pair: {pair.get('pair')!r} (a={pair.get('arm_a', {}).get('arm')}@{pair.get('arm_a', {}).get('stage')}, b={pair.get('arm_b', {}).get('arm')}@{pair.get('arm_b', {}).get('stage')}), want '{ARM} - {BASE}' with {ARM}@5 and {BASE}@4")
    if not (pair.get("pin4_sha256") == pair.get("pin4_sha256_now") == PIN4_SHA):
        prob.append("pair: the pinned dominance read was not pin 4 when the pair ran")
    if "B 2000, seed 0" not in str(pair.get("estimator")):
        prob.append("pair: estimator is not 'B 2000, seed 0'")
    if c0c_dom.get("arm") != BASE or c0c_dom.get("arm_stage") != 4 or c0c_dom.get("script_sha256") != PIN4_SHA:
        prob.append(f"c0c-dom: arm {c0c_dom.get('arm')!r} stage {c0c_dom.get('arm_stage')!r} script {str(c0c_dom.get('script_sha256'))[:8]}, want {BASE} stage 4 read by pin 4")
    # ---- gates: a failed gate in a pinned read means no row was read
    dg, dwhy = gates_state(dom)
    pg, pwhy = gates_state(pair)
    res["gates"] = {"dominance": dg, "pair": pg, "failed": {"dominance": dwhy, "pair": pwhy}}
    not_read = dom.get("outcome") == "GATE-FAIL" or pair.get("outcome") != "READ" or not all(dg.values()) or not all(pg.values()) or "by_seed" not in dom or "by_seed" not in pair
    if not_read:
        res.update(outcome="NOT-READ", terms=None, integrity={"ok": not prob, "problems": prob, "warnings": warn},
                   why="a gate of a pinned read failed, so no row was read; no section-6 outcome is applied. Failed gates: " + json.dumps({"dominance": dwhy, "pair": pwhy})[:900],
                   next_lever="None applied. Fix what the failed gate names and read again (Rule Zero: this is a blocker, not a result).")
        return finish(res, prob)
    # ---- tables
    T = {s: {"dom": table(dom, s), "pair": table(pair, s), "c0c": table(c0c_dom, s)} for s in SEEDS}
    for s in SEEDS:
        prob += table_problems(f"dom/{s}", T[s]["dom"], dom["by_seed"][s].get("read_a"))
        prob += table_problems(f"pair/{s}", T[s]["pair"])
        prob += table_problems(f"c0c-dom/{s}", T[s]["c0c"], c0c_dom["by_seed"][s].get("read_a"))
    if any("row set differs" in x for x in prob):
        res.update(outcome="INPUT-INVALID", terms=None, integrity={"ok": False, "problems": prob, "warnings": warn}, why="a table does not hold the 19 rows; no section-6 outcome is applied",
                   next_lever="None applied. Resolve the problems listed under integrity, then read again.")
        return finish(res, prob)
    rs_d, rs_p = (dom.get("r_scn") or {}), (pair.get("r_scn") or {})
    if rs_d.get("problems") or rs_p.get("problems"):
        prob.append(f"R-SCN problems: dom {rs_d.get('problems')}, pair {rs_p.get('problems')}")
    sc_d, sc_p, sc_c = (rs_d.get("scenes") or {}).get("read_on"), (rs_p.get("scenes") or {}).get("read_on"), (c0c_dom.get("r_scn") or {}).get("scenes", {}).get("read_on")
    if sc_d != sc_p:
        prob.append(f"the pinned read and the pair were read on different scenes ({sc_d} vs {sc_p})")
    if sc_d != 48:
        warn.append(f"scenes read on: {sc_d}; the card names 48. R-SCN removes a scene refused in ANY stage from every arm, so this is by rule, but it is not the card's 48.")
    # the C0C reference must be the SAME numbers the pair holds for C0C against P10 (same scenes, same rows)
    ref = (pair.get("arm_vs_P10") or {}).get(BASE)
    if not ref:
        prob.append("pair: no arm_vs_P10 block for C0C to check the reference against")
    else:
        for s in SEEDS:
            for lab in ROWS:
                a, b = ref[s].get(lab), c0c_dom["by_seed"][s]["rows"].get(lab)
                if not a or not b or any(a[k] != b[k] for k in ("delta", "lo", "hi", "separated")):
                    prob.append(f"the pair's C0C-against-P10 row {s}/{lab} differs from the pin-4 read of C0C (stage4/score_q_identity): the reference is on other scenes or another read")
                    break
        ref_re = (pair.get("arm_vs_P10") or {}).get(ARM)
        if ref_re:
            for s in SEEDS:
                for lab in ROWS:
                    a, b = ref_re[s].get(lab), dom["by_seed"][s]["rows"].get(lab)
                    if not a or not b or any(a[k] != b[k] for k in ("delta", "lo", "hi", "separated")):
                        prob.append(f"the pair's C0C-RE-against-P10 row {s}/{lab} differs from the pinned read of C0C-RE")
                        break
    # ---- the terms of section 6
    pinned = {s: bool(dom["decision"]["per_seed"][s]["DOMINANT"]) for s in SEEDS} if (dom.get("decision") or {}).get("per_seed") else {s: False for s in SEEDS}
    claims_dom = dom.get("outcome") == "DOMINANT-T2"
    if claims_dom and not all(pinned.values()):
        prob.append("dom: outcome DOMINANT-T2 but decision.per_seed does not say DOMINANT at both seeds")
    if claims_dom:
        for s in SEEDS:
            if sw_rows(T[s]["dom"]):
                prob.append(f"dom: outcome DOMINANT-T2 but {s} holds separated-worse rows {sw_rows(T[s]['dom'])} (D2)")
    terms = {"pinned_outcome": dom.get("outcome"), "pinned_dominant_both_seeds": bool(claims_dom and all(pinned.values())), "per_seed": {}}
    for s in SEEDS:
        pa, pm = T[s]["pair"]["ADE6"], T[s]["pair"]["M1 crawl share"]
        new_sw = sorted(set(sw_rows(T[s]["dom"])) - set(sw_rows(T[s]["c0c"])))
        terms["per_seed"][s] = {
            "pair_ADE6": {"delta": pa["delta"], "lo": pa["lo"], "hi": pa["hi"], "verdict": pa["verdict"], "separated_better": pa["verdict"] == SB, "separated_worse": pa["verdict"] == SW},
            "pair_M1": {"delta": pm["delta"], "lo": pm["lo"], "hi": pm["hi"], "verdict": pm["verdict"], "not_separated_worse": pm["verdict"] != SW},
            "separated_worse_vs_P10": {"C0C-RE": sw_rows(T[s]["dom"]), "C0C": sw_rows(T[s]["c0c"]), "new_in_C0C-RE": new_sw, "none_new": not new_sw},
            "M1_vs_P10": {"C0C-RE": T[s]["dom"]["M1 crawl share"]["verdict"], "C0C": T[s]["c0c"]["M1 crawl share"]["verdict"]},
        }
    ps = terms["per_seed"]
    t_ade = all(ps[s]["pair_ADE6"]["separated_better"] for s in SEEDS)
    t_m1 = all(ps[s]["pair_M1"]["not_separated_worse"] for s in SEEDS)
    t_new = all(ps[s]["separated_worse_vs_P10"]["none_new"] for s in SEEDS)
    terms.update(clause_ADE6_separated_better_both_seeds=t_ade, clause_M1_not_separated_worse_either_seed=t_m1, clause_no_new_separated_worse_vs_P10=t_new, re_moves_holds=bool(t_ade and t_m1 and t_new))
    harms = []
    for s in SEEDS:
        if ps[s]["pair_ADE6"]["separated_worse"]:
            harms.append(f"{s}: ADE6 of C0C-RE - C0C is separated-worse")
        if not ps[s]["pair_M1"]["not_separated_worse"]:
            harms.append(f"{s}: M1 crawl share of C0C-RE - C0C is separated-worse")
        if ps[s]["separated_worse_vs_P10"]["new_in_C0C-RE"]:
            harms.append(f"{s}: new separated-worse rows against P10: {ps[s]['separated_worse_vs_P10']['new_in_C0C-RE']}")
    terms["harm_triggers"] = harms
    # sensitivity (decides nothing): the other reading of "M1" in clause 2 (M1 against P10)
    m1_p10_ok = all(ps[s]["M1_vs_P10"]["C0C-RE"] != SW for s in SEEDS)
    terms["sensitivity_M1_read_against_P10"] = {"clause_holds": m1_p10_ok, "would_change_re_moves": bool(t_ade and t_new and m1_p10_ok) != terms["re_moves_holds"]}
    if prob:
        res.update(outcome="INPUT-INVALID", terms=terms, integrity={"ok": False, "problems": prob, "warnings": warn}, why="a cross-check of the inputs failed; no section-6 outcome is applied",
                   next_lever="None applied. Resolve the problems listed under integrity, then read again.")
        return finish(res, prob)
    if terms["pinned_dominant_both_seeds"]:
        out = "RE-DOMINANT"
    elif terms["re_moves_holds"]:
        out = "RE-MOVES"
    else:
        out = "RE-HARMS" if harms else "RE-NULL"
    res.update(outcome=out, terms=terms, integrity={"ok": True, "problems": [], "warnings": warn}, next_lever=NEXT[out],
               replicate_required=bool(out == "RE-DOMINANT"), counted_as_scale_signal=False,
               also_satisfies_re_moves=bool(out == "RE-DOMINANT" and terms["re_moves_holds"]),
               three_family_form_beside={s: bool(((dom.get("three_family_form_beside") or {}).get(s) or {}).get("DOMINANT")) for s in SEEDS},
               t1_kin_beside="NOT READ HERE: needs the T1-kin read of C0 on eval-141 re-encoded through E' (|ADE6 delta| against F = %s m)" % F_T1KIN_M)
    res["families"] = {s: family_summary(T[s]["pair"]) for s in SEEDS}
    res["families_vs_P10"] = {s: family_summary(T[s]["dom"]) for s in SEEDS}
    res["seed_floor_beside"] = {lab: {"gap_C0C-RE": ((pair.get("seed_floor") or {}).get("a") or {}).get(lab, {}).get("gap"), "gap_C0C": ((pair.get("seed_floor") or {}).get("b") or {}).get(lab, {}).get("gap"),
                                      "abs_delta_lt_floor": (pair.get("abs_delta_below_seed_floor_beside") or {}).get(lab)} for lab in ("ADE6", "M1 crawl share")}
    res["pair_rows"] = {s: T[s]["pair"] for s in SEEDS}
    res["vs_P10_rows"] = {s: {"C0C-RE": T[s]["dom"], "C0C": T[s]["c0c"]} for s in SEEDS}
    res["n_scenes"] = sc_d
    res["starts_read_on"] = ((rs_d.get("starts") or {}).get("read_on") or {}).get("total")
    if blind_pair is not None:
        res["beside_blind"] = blind_beside(blind_pair)
    return finish(res, prob)


def family_summary(tab):
    out = {}
    for fam in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "crawl"):
        rows = {r: v for r, v in tab.items() if v["family"] == fam}
        out[fam] = {"n_rows": len(rows), "separated_better": sorted(r for r, v in rows.items() if v["verdict"] == SB), "separated_worse": sorted(r for r, v in rows.items() if v["verdict"] == SW),
                    "unseparated": sorted(r for r, v in rows.items() if v["verdict"] == UN)}
    out["STRATEGIC"]["status"] = "PAUSED (PI item 25): not a pass"
    return out


def blind_beside(bp):
    out = {"decides_nothing": True, "pair": bp.get("pair"), "outcome": bp.get("outcome"), "problems": []}
    if bp.get("pair") != f"{BLIND} - {ARM}":
        out["problems"].append(f"pair is {bp.get('pair')!r}, want '{BLIND} - {ARM}'")
    if bp.get("outcome") != "READ" or "by_seed" not in bp:
        out["problems"].append("the pair read did not produce rows: " + json.dumps({k: v.get('why') for k, v in (bp.get('gates') or {}).items() if not v.get('PASS')})[:400])
        return out
    out["rows"] = {s: {r: {k: v[k] for k in ("family", "verdict", "delta", "lo", "hi")} for r, v in table(bp, s).items() if r in ("ADE6", "speed MAE", "along-track MAE", "M1 crawl share")} for s in SEEDS}
    out["families"] = {s: family_summary(table(bp, s)) for s in SEEDS}
    return out


def finish(res, prob):
    res["exit_code"] = {"NOT-READ": 2, "INPUT-INVALID": 3}.get(res["outcome"], 0)
    return res


# ------------------------------------------------------------------------------------------------ the .md, generated from the JSON
def fmt(x):
    return "n/a" if x is None else f"{x:+.4f}"


def row_line(r, v):
    return f"| {v['family']} | {r} | {fmt(v['delta'])} | {fmt(v['lo'])} | {fmt(v['hi'])} | {v['verdict']}{' (PAUSED)' if v['family'] == 'STRATEGIC' else ''} |"


def render_md(res):
    L = [f"# RESULT_T2_RENC: {res['outcome']}", "",
         f"Card `PREREG_T2_RENC.md` (frozen prefix 5,544 bytes, sha256 {res['card_prefix_sha256_expected'][:8]}..., frozen ok: {res['card_frozen_ok']}). Tier: {res['tier']}. Estimator: {res['estimator']}.",
         f"Generated by `c0cre_outcome.py` from `RESULT_T2_RENC.json`; section 6 applied as written. Scenes read on: {res.get('n_scenes')}; starts: {res.get('starts_read_on')}. STRATEGIC: {res['strategic']}.", ""]
    if res.get("rehearsal_on_other_arm"):
        L[2:2] = [f"> REHEARSAL: the inputs carry the arm `{res['arm_checked']}`, not C0C-RE. This file shows the wiring of the read; it is not a result of C0C-RE.", ""]
    if res["outcome"] in ("NOT-READ", "INPUT-INVALID"):
        L += [f"## {res['outcome']}", "", res.get("why", ""), ""]
        for p in res["integrity"]["problems"]:
            L.append(f"* {p}")
        L += ["", f"Next: {res['next_lever']}"]
        return "\n".join(L) + "\n"
    t = res["terms"]
    L += ["## Outcome", "", f"**{res['outcome']}**.", "", f"* Pinned read (pin 4) of C0C-RE against P10: **{t['pinned_outcome']}** (DOMINANT at both seeds: {t['pinned_dominant_both_seeds']}). Three-family form beside (STRATEGIC paused): {res['three_family_form_beside']}.",
          f"* RE-MOVES clauses (card 6.2): ADE6 of C0C-RE - C0C separated-better at both seeds: **{t['clause_ADE6_separated_better_both_seeds']}**; M1 crawl share of the pair not separated-worse at either seed: **{t['clause_M1_not_separated_worse_either_seed']}**; "
          f"no new separated-worse row against P10: **{t['clause_no_new_separated_worse_vs_P10']}**.", "", "| term | s0 | s1 |", "|---|---|---|"]
    p0, p1 = t["per_seed"]["s0"], t["per_seed"]["s1"]
    L += [f"| ADE6, C0C-RE - C0C | {fmt(p0['pair_ADE6']['delta'])} [{fmt(p0['pair_ADE6']['lo'])}, {fmt(p0['pair_ADE6']['hi'])}] {p0['pair_ADE6']['verdict']} | {fmt(p1['pair_ADE6']['delta'])} [{fmt(p1['pair_ADE6']['lo'])}, {fmt(p1['pair_ADE6']['hi'])}] {p1['pair_ADE6']['verdict']} |",
          f"| M1 crawl share, C0C-RE - C0C | {fmt(p0['pair_M1']['delta'])} [{fmt(p0['pair_M1']['lo'])}, {fmt(p0['pair_M1']['hi'])}] {p0['pair_M1']['verdict']} | {fmt(p1['pair_M1']['delta'])} [{fmt(p1['pair_M1']['lo'])}, {fmt(p1['pair_M1']['hi'])}] {p1['pair_M1']['verdict']} |",
          f"| separated-worse vs P10, C0C-RE | {', '.join(p0['separated_worse_vs_P10']['C0C-RE']) or 'none'} | {', '.join(p1['separated_worse_vs_P10']['C0C-RE']) or 'none'} |",
          f"| separated-worse vs P10, C0C | {', '.join(p0['separated_worse_vs_P10']['C0C']) or 'none'} | {', '.join(p1['separated_worse_vs_P10']['C0C']) or 'none'} |",
          f"| new in C0C-RE | {', '.join(p0['separated_worse_vs_P10']['new_in_C0C-RE']) or 'none'} | {', '.join(p1['separated_worse_vs_P10']['new_in_C0C-RE']) or 'none'} |", ""]
    if t["harm_triggers"]:
        L += ["RE-HARMS label triggers (label only; the card's outcome 3 is one class): " + "; ".join(t["harm_triggers"]), ""]
    sens = t["sensitivity_M1_read_against_P10"]
    L += [f"Sensitivity, decides nothing: reading clause 2's M1 as M1 against P10 gives clause {sens['clause_holds']}; it would change the RE-MOVES term: {sens['would_change_re_moves']}.", "",
          "## Committed next lever (card section 6)", "", res["next_lever"], ""]
    if res["outcome"] == "RE-DOMINANT":
        L += [f"Counted as the scale signal: **{res['counted_as_scale_signal']}**; replicate required: **{res['replicate_required']}**. Also satisfies the RE-MOVES clauses: {res['also_satisfies_re_moves']}.", ""]
    L += [f"T1-kin read beside: {res['t1_kin_beside']}.", ""]
    for s in SEEDS:
        L += [f"## Pair C0C-RE - C0C, seed {s}: the four families", "",
              "Error rows: a negative delta is C0C-RE better. The verdict is the pinned reader's own direction rule.", "", "| family | row | delta | lo | hi | verdict |", "|---|---|---|---|---|---|"]
        tab = res["pair_rows"][s]
        for fam in FAMILY_ORDER:
            for r in ROWS:
                if tab[r]["family"] == fam:
                    L.append(row_line(r, tab[r]))
        fs = res["families"][s]
        L += ["", "Family counts (separated-better / separated-worse / unseparated): " + "; ".join(f"{f} {len(v['separated_better'])}/{len(v['separated_worse'])}/{len(v['unseparated'])}" for f, v in fs.items() if f != "STRATEGIC")
              + f"; STRATEGIC PAUSED, shown only: {len(fs['STRATEGIC']['separated_better'])}/{len(fs['STRATEGIC']['separated_worse'])}/{len(fs['STRATEGIC']['unseparated'])}.", ""]
    L += ["## Verdicts against P10, C0C-RE beside C0C (pinned read, pin 4)", "", "| family | row | C0C-RE s0 | C0C s0 | C0C-RE s1 | C0C s1 |", "|---|---|---|---|---|---|"]
    for fam in FAMILY_ORDER:
        for r in ROWS:
            v = res["vs_P10_rows"]["s0"]["C0C-RE"][r]
            if v["family"] == fam:
                g = lambda s, a: res["vs_P10_rows"][s][a][r]["verdict"]  # noqa: E731
                L.append(f"| {fam}{' (PAUSED)' if fam == 'STRATEGIC' else ''} | {r} | {g('s0', 'C0C-RE')} | {g('s0', 'C0C')} | {g('s1', 'C0C-RE')} | {g('s1', 'C0C')} |")
    L += ["", "## Seed floor beside (|delta| against the arm's own seed gap; decides nothing)", ""]
    for lab, v in res["seed_floor_beside"].items():
        L.append(f"* {lab}: seed gap C0C-RE {v['gap_C0C-RE']}, C0C {v['gap_C0C']}; |delta| below floor: {v['abs_delta_lt_floor']}")
    if "beside_blind" in res:
        b = res["beside_blind"]
        L += ["", "## Beside, deciding nothing: C0C-RE-BLIND - C0C-RE", ""]
        if b["problems"]:
            L += [f"Not tabulated: {'; '.join(b['problems'])}"]
        else:
            L += ["| row | s0 | s1 |", "|---|---|---|"]
            for r in ("ADE6", "speed MAE", "along-track MAE", "M1 crawl share"):
                c = lambda s: f"{fmt(b['rows'][s][r]['delta'])} [{fmt(b['rows'][s][r]['lo'])}, {fmt(b['rows'][s][r]['hi'])}] {b['rows'][s][r]['verdict']}"  # noqa: E731
                L.append(f"| {r} | {c('s0')} | {c('s1')} |")
    L += ["", "## What this does not say", ""] + [f"* {x}" for x in DOES_NOT_SAY]
    if res["integrity"]["warnings"]:
        L += ["", "## Warnings", ""] + [f"* {w}" for w in res["integrity"]["warnings"]]
    L += ["", "## Provenance", ""] + [f"* {k}: `{v.get('path')}` sha256 `{v.get('sha256')}`" for k, v in res["inputs"].items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    global ARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--c0c-dom", required=True)
    ap.add_argument("--blind-pair", default=None)
    ap.add_argument("--card", default=CARD)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default=ARM, help="REHEARSAL ONLY: the arm name the inputs must carry (default C0C-RE). Anything else is recorded in the JSON as arm_checked and the result is not a C0C-RE result.")
    a = ap.parse_args(argv)
    ARM = a.arm
    names = {}
    for k, p in (("dominance_read", a.dom), ("pair_read", a.pair), ("c0c_reference_read", a.c0c_dom), ("blind_pair_read", a.blind_pair), ("card", a.card), ("c0cre_outcome.py", os.path.abspath(__file__))):
        if p and os.path.isfile(p):
            names[k] = {"path": p.replace("\\", "/"), "sha256": sha256_file(p)}
        elif p and k != "blind_pair_read":
            names[k] = {"path": p, "sha256": None}
    res = evaluate(load(a.dom), load(a.pair), load(a.c0c_dom), load(a.blind_pair) if a.blind_pair and os.path.isfile(a.blind_pair) else None, card_ok(a.card), names)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/RESULT_T2_RENC.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    with open(f"{a.out_dir}/RESULT_T2_RENC.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(res))
    print(json.dumps({"outcome": res["outcome"], "exit_code": res["exit_code"], "integrity_problems": res["integrity"]["problems"][:3]}))
    return res["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
