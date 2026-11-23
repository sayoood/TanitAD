"""s3_name.py -- the SEED3 name (card PREREG_V8F_SEED3 sha256 75f3b191, section 4, exactly).  Pure python; it reads the JSON of the read and decides nothing else.
Names: SEED3-BLOCKED | SEED3-T0-FAILS | SEED3-HARMS | SEED3-REPLICATES | SEED3-NO-EFFECT.  Precedence: BLOCKED, then T0-FAILS, then HARMS, then REPLICATES, then NO-EFFECT.
  BLOCKED    a gate / provenance / lever-acts check fails (or its record is absent: not read)
  T0-FAILS   G2 fails: the upper interval bound of RH_s2 - B0_s2 at T0 ADE6 is > +0.1174 m
  HARMS      G2 holds and P1-s2 (RH_s2 - B0_s2 under C* = R0C10XHb20) has a family WORSE
  REPLICATES G2 holds, P1-s2 has at least one family BETTER and none WORSE
  NO-EFFECT  G2 holds, P1-s2 has no family BETTER and none WORSE
Family verdicts are the reader's (rh_read.py, the family rule as written: BETTER / WORSE only if EVERY named cell is, else MIXED / NOT-SEPARATED); a MIXED family is neither BETTER nor WORSE, and is printed.
-STRADDLE: if a decisive P1-s2 cell straddles its floor (Amendment 2: interval excludes 0 while abs(D) <= F_c) and reading it by its sign changes the name, the name carries -STRADDLE, both names are printed and it goes to the PI.
A P1 family NOT-COMPUTABLE (or G2 missing) when the name needs it: NOT-DECIDABLE (outside the table; never a made-up name).
Lever-acts record (JSON, written by the pod side / MM, every key must be true): see LEVER_KEYS.  The 8th item of the card (the read's checkpoints sha-equal to the pod's) is computed here from both results' provenance.
Usage: python s3_name.py --rh RESULT_S3_RH.json --ki RESULT_S3_KI.json --gt0-rh GT0_S3_RH.json --gt0-ki GT0_S3_KI.json --mech MECH4_S3.json --lever LEVER_ACTS_S3.json --out RESULT_SEED3.json"""
import argparse
import json
import os
import sys

BAR = 0.1174          # F_T0^4, card section 3; the full value
BESIDE_BAR = 0.0705   # printed beside, decides nothing
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
LEVER_KEYS = ("gate_records_minted_verified", "zerograd_pass", "post_every_tensor_moved", "bank_generator_names_UE_s2A_coverage_ge_0p5", "dagger_probe_RH_s2_KI_s2_pass", "dagger_rows_sum_gt_0", "KI_s2_kinput_items")


def family_of(vs):
    """same rule as rh_read.family_of"""
    if any(v == "NOT-COMPUTABLE" for v in vs):
        return "NOT-COMPUTABLE"
    if all(v == "BETTER" for v in vs):
        return "BETTER"
    if all(v == "WORSE" for v in vs):
        return "WORSE"
    if not any(v in ("BETTER", "WORSE") for v in vs):
        return "NOT-SEPARATED"
    return "MIXED"


def sign_resolved(p1):
    """families of the P1 pair with every straddling cell read BETTER / WORSE by its sign (rh_read.flipped_families)"""
    out = {}
    for f in FAMS[:3]:
        vs = [(v.get("straddle_sign_verdict") or v["verdict"]) for cell, v in p1["cells"].items() if isinstance(v, dict) and (cell.split(":")[0] == f or (f == "LONGITUDINAL" and cell == "HEADWAY:U"))]
        out[f] = family_of(vs)
    sv = [(v.get("straddle_sign_verdict") or v["verdict"]) for v in p1["cells"].get("STRATEGIC:pos_tick2", [])]
    out["STRATEGIC"] = family_of(sv) if sv else p1["families"]["STRATEGIC"]
    return out


def straddling(p1):
    s = [c for c, v in p1["cells"].items() if isinstance(v, dict) and v.get("straddles_floor")]
    return s + [f"STRATEGIC:pos{int(v['arc_m'])}" for v in p1["cells"].get("STRATEGIC:pos_tick2", []) if v.get("straddles_floor")]


def name_from(blocked, g2_hi, fams):
    """-> (name, why).  fams = {family: verdict} or None."""
    if blocked:
        return "SEED3-BLOCKED", "; ".join(blocked)
    if g2_hi is None:
        return "NOT-DECIDABLE", "G2 not computable (T0 eval or its file missing)"
    if g2_hi > BAR:
        return "SEED3-T0-FAILS", f"G2: upper bound {g2_hi:+.4f} > +{BAR}"
    if fams is None or any(fams.get(f) in (None, "NOT-COMPUTABLE") for f in FAMS):
        return "NOT-DECIDABLE", f"P1-s2 family not computable: {fams}"
    if any(fams[f] == "WORSE" for f in FAMS):
        return "SEED3-HARMS", f"P1-s2 WORSE families {[f for f in FAMS if fams[f] == 'WORSE']}"
    if any(fams[f] == "BETTER" for f in FAMS):
        return "SEED3-REPLICATES", f"G2 holds; P1-s2 BETTER families {[f for f in FAMS if fams[f] == 'BETTER']}, none WORSE"
    return "SEED3-NO-EFFECT", "G2 holds; P1-s2 has no family BETTER and none WORSE"


def decide_name(lever, prov_ok, g2, p1, g2_bar_in_file=BAR):
    """lever: dict or None; prov_ok: {'RH': bool|None, 'KI': bool|None}; g2: {'point','lo','hi'} or None; p1: the P1 pair read of the RH_s2 - B0_s2 result or None -> dict"""
    blocked = []
    if lever is None:
        blocked.append("lever-acts record missing (LEVER_ACTS_S3.json): not read")
    else:
        blocked += [f"lever-acts check not true: {k}" for k in LEVER_KEYS if lever.get(k) is not True]
    for arm, ok in prov_ok.items():
        if ok is not True:
            blocked.append(f"provenance of the {arm} read: checkpoints not sha-equal to the pod's (ok={ok})")
    if g2 is not None and abs(float(g2_bar_in_file) - BAR) > 1e-12:
        blocked.append(f"the G-T0 file was written under bar {g2_bar_in_file}, not {BAR}")
    hi = None if g2 is None else float(g2["hi"])
    fams = None if p1 is None else dict(p1["families"])
    name, why = name_from(blocked, hi, fams)
    res = {"name_as_read": name, "reason": why, "g2_upper": hi, "g2_holds": None if hi is None else bool(hi <= BAR), "p1_families_as_read": fams,
           "mixed_families_printed": [] if fams is None else [f for f in FAMS if fams.get(f) == "MIXED"]}
    if p1 is not None and name not in ("SEED3-BLOCKED", "SEED3-T0-FAILS"):
        st = straddling(p1)
        res["straddling_decisive_cells"] = st
        if st:
            sr = sign_resolved(p1)
            n2, w2 = name_from(blocked, hi, sr)
            res["p1_families_sign_resolved"], res["name_by_sign"] = sr, n2
            if n2 != name:
                res["both_names"] = [name, n2]
                name = name + "-STRADDLE"
                why = f"{why}; a straddling decisive cell changes the name by its sign ({res['both_names']}): to the PI"
    res["name"], res["reason"] = name, why
    return res


def _j(p):
    return json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def fam_line(pr):
    return None if pr is None else {"families": pr.get("families"), "straddling": straddling(pr)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("rh", "ki", "gt0-rh", "gt0-ki", "mech", "lever", "out"):
        ap.add_argument("--" + k, required=(k == "out"), default="")
    a = ap.parse_args()
    rh, ki, g_rh, g_ki, mech, lever = (_j(x) for x in (a.rh, a.ki, a.gt0_rh, a.gt0_ki, a.mech, a.lever))
    prov = {"RH": None if rh is None else bool((rh.get("provenance") or {}).get("ok")), "KI": None if ki is None else bool((ki.get("provenance") or {}).get("ok"))}
    for arm, r in (("RH", rh), ("KI", ki)):
        if r is not None and (r.get("blocked") or {}).get("blocked") is True:
            prov[arm] = False
    g2 = ((g_rh or {}).get("per_seed") or {}).get("2")
    g2 = g2 if isinstance(g2, dict) and "hi" in g2 else None
    p1 = ((rh or {}).get("pairs") or {}).get("P1")
    res = decide_name(lever, prov, g2, p1, (g_rh or {}).get("F_T0_m", BAR))
    res["card"] = "PREREG_V8F_SEED3.md sha256 75f3b191 section 4 (one-seed reads of s2 decide the name)"
    res["inputs"] = {"G2": {"RH_s2_minus_B0_s2_T0_ADE6": g2, "bar_m": BAR, "beside_bar_m": BESIDE_BAR, "beside_holds": None if g2 is None else bool(g2["hi"] <= BESIDE_BAR)},
                     "P1_s2": {"pair": None if p1 is None else [p1["A"], p1["B"]], "tier": "T1-kin/hb0.2", "cells": None if p1 is None else p1["cells"],
                               "floors_file": (rh or {}).get("floors_file")},
                     "lever_acts": lever, "provenance_ok": prov}
    P = (rh or {}).get("P") or {}
    k2 = ((g_ki or {}).get("per_seed") or {}).get("2") or {}
    pairs_ki, pairs_rh = (ki or {}).get("pairs") or {}, (rh or {}).get("pairs") or {}
    res["beside_printed_no_bar"] = {
        "KI_s2_minus_B0_s2_G_T0": None if not g_ki else {"upper": k2.get("hi"), "point_m": k2.get("point_m"), "under_0.1174": k2.get("status"), "under_0.0705": k2.get("beside_status")},
        "KI_s2_minus_B0_s2_P1": fam_line(pairs_ki.get("P1")),
        "MECH4_s2_and_three_seed_list": None if not mech else {"MECH4_s2": mech.get("MECH4_s2"), "three_seed_list": mech.get("MECH4_three_seed_list")},
        "KI_s2_minus_RH_s2_C_star": fam_line(pairs_ki.get("BESIDE_KI_minus_RH_R0C10XHb20")),
        "KI_s2_minus_RH_s2_R0Hb20": fam_line(pairs_ki.get("BESIDE_KI_minus_RH_R0Hb20")),
        "RH_s2_minus_B0_s2_R0Hb20": fam_line(pairs_rh.get("BESIDE_RH_minus_B0_R0Hb20")),
        "P2_P3_P4_on_RH_s2": {"P2": (P.get("P2") or {}).get("pass"), "P3": (P.get("P3") or {}).get("pass"), "P4": (P.get("P4") or {}).get("pass")},
        "GT0_s0_s1_under_0.1174_already_seen_decides_nothing": None if not mech else mech.get("GT0_s0_s1_under_0p1174_PRINTED_ONLY"),
        "three_seed_pooled_reads": "NOT-COMPUTED: the card defines F_c for two seeds only (max of the two arms' seed spreads) and rh_read.floor_of handles one or two seeds; a three-seed floor is not frozen, so the frozen reader cannot take three seeds unchanged"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("NAME:", res["name"], "|", res["reason"])
    print("G2 upper", res["g2_upper"], "holds", res["g2_holds"], "(beside 0.0705:", res["inputs"]["G2"]["beside_holds"], ") | P1-s2 families", res["p1_families_as_read"], "| mixed", res["mixed_families_printed"])
    if res.get("both_names"):
        print("STRADDLE: both names", res["both_names"])
    print("wrote", a.out)
    print("ZZ SEED3-NAME DONE ZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
