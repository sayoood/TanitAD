"""rh_read.py (R2) -- the ROLLHIST reader (card PREREG_V8F_ROLLHIST.md sha256 781eea87, sections 3.2 to 3.4; Amendment 1 sha256 87928487; Amendment 2 sha256 15004dc9, items 2, 3, 4, 6).  numpy only; it reads the JSON of the frozen readers and decides nothing else.
Inputs (all produced by rh_read_chain.sh):  RESULT_Q3.json (families LONGITUDINAL / LATERAL / TACTICAL cells, q3_read_r3.py), RESULT_HEADWAY.json (headway U, headway_read_r2.py),
READ_STRAT_ROLLHIST.json (STRATEGIC, rh_strat.py over the frozen ADD66 reader), GT0_ROLLHIST.json (guard G-T0, rh_t0.py), refs (stage-A UE.A under the form of record: RESULT_Q3.json / RESULT_HEADWAY.json read over out/refs dumps, MEASURED; ADD68's JSON is printed beside, INHERITED), pod/bank record (blocked flags).
It prints EXACTLY ONE committed outcome name (section 3.4, precedence BLOCKED, HARMS, SEED-DISAGREE, then the table order); on one seed the same name prefixed PROVISIONAL.  Outside the committed table
it prints OUTSIDE-TABLE and says which cell broke, never a name made up.  Cells are per family, never pooled.
Cell verdict (ADD63's family reader + the card): BETTER / WORSE when the paired pooled interval excludes 0 on that side, both seed means agree in sign and abs(D) > F_c; else NOT-SEPARATED.
F_c (two seeds) = max(abs(A_s1 - A_s0), abs(B_s1 - B_s0)) over the two arms of the pair under the same form (RH - B0: the card's max(abs(RH_s1-RH_s0), abs(B0_s1-B0_s0))); UE.B's term is NOT COMPUTABLE (STAGEB not run).
F_U,c (one seed) = max over the two arms' forms of abs(UE_s1.A - UE_s0.A) from the stage-A references (our rollouts under the form of record, MEASURED; ADD68 arm_levels printed beside as floor_ADD68_inherited); a form without a reference makes the cell NOT COMPUTABLE.
Family: BETTER if every named cell BETTER, WORSE if every named cell WORSE, NOT-SEPARATED if none BETTER/WORSE, else MIXED.  Named cells: LONGITUDINAL speed_mae_mps, along_mae_m and headway U; LATERAL heading_mae_deg,
yaw_rate_mae_degps; TACTICAL ade_tac_m; STRATEGIC pos at 30/50/80 m at the third tick (ROLL2 pair).
Usage:  python rh_read.py --q3 .. --hw .. --strat .. --gt0 .. --refs-q3 .. --seeds 0,1 --out RESULT_ROLLHIST.json      |      python rh_read.py --selftest (K-SELF, K-PLANT, mutation, on synthetic JSON)"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rh_common as C  # noqa: E402

FAMS = C.FAMILIES
REG = set()      # deliberate-regression switches used ONLY by --selftest: g1_off, g2_off, straddle_off, any_straddle, attr_off
LAT_LONG_TAC = ("LONGITUDINAL", "LATERAL", "TACTICAL")
CELL_ORDER = {f: list(C.NAMED[f]) for f in C.NAMED}
CELL_ORDER["LONGITUDINAL"] = CELL_ORDER["LONGITUDINAL"] + ["HEADWAY:U"]
BESIDE_CELLS = ["LATERAL:cross_mae_m", "LATERAL:curvature_mae_1pm", "TACTICAL:lon_err_rate", "TACTICAL:lat_err_rate", "TACTICAL:joint_err_rate"]
NAMES = ("ROLLHIST-REPAIRS-K-UNNEEDED", "ROLLHIST-REPAIRS", "ROLLHIST-PARTIAL", "ROLLHIST-NO-EFFECT", "ROLLHIST-HARMS", "ROLLHIST-SEED-DISAGREE", "ROLLHIST-BLOCKED")


def pk(A, B):
    return f"I10{A}_minus_I10{B}"


def lvl_q3(q3, run, s, cell):
    d = (q3.get("arm_levels") or {}).get(f"I10{run}_s{s}")
    return None if d is None else d.get(cell.split(":")[1] + "|all")


def lvl_hw(hw, run, s):
    d = (hw.get("levels") or {}).get(f"I10{run}")
    if not d:
        return None
    try:
        return d["cells"]["U"]["per_seed"][s]
    except Exception:                                    # noqa: BLE001
        return None


def spread(getter, run, seeds):
    """abs(level_s1 - level_s0) of one arm, or None"""
    a, b = getter(run, seeds[0]), getter(run, seeds[1])
    return None if a is None or b is None else abs(float(a) - float(b))


def floor_of(getter, A, B, seeds, refgetter=None, forms=None):
    """-> (F, source).  Two seeds: max of the two arms' seed spreads.  One seed: stage-A floor from the references, same forms."""
    if len(seeds) == 2:
        sp = [spread(getter, r, seeds) for r in (A, B)]
        return (None, "arm levels missing") if any(x is None for x in sp) else (max(sp), "two-seed F_c")
    if refgetter is None:
        return None, "no stage-A reference"
    sp = []
    for f in sorted({x.split("_", 1)[1] for x in (A, B)}):
        v = spread(refgetter, f, (0, 1))
        if v is None:
            return None, f"stage-A reference for form {f} missing (UE.A under {f} not found)"
        sp.append(v)
    return max(sp), "one-seed F_U,c (stage-A, INHERITED)"


def cell_verdict(c, F):
    """c = {d_s, D, A_lo, A_hi}; F float or None -> dict(verdict, ...)"""
    if c is None:
        return {"verdict": "NOT-COMPUTABLE", "why": "cell missing"}
    d_s, D, lo, hi = [float(x) for x in c["d_s"]], float(c["D"]), float(c["A_lo"]), float(c["A_hi"])
    out = {"D": D, "interval": [lo, hi], "per_seed": d_s, "F": F}
    if F is None:
        out["verdict"] = "NOT-COMPUTABLE"
        out["why"] = "floor missing"
        return out
    better = bool(hi < 0 and all(x < 0 for x in d_s) and abs(D) > F)
    worse = bool(lo > 0 and all(x > 0 for x in d_s) and abs(D) > F)
    out["verdict"] = "BETTER" if better else "WORSE" if worse else "NOT-SEPARATED"
    out["straddles_floor"] = bool((hi < 0 or lo > 0) and (all(x < 0 for x in d_s) or all(x > 0 for x in d_s)) and abs(D) <= F)     # Amendment 2 item 3: interval excludes 0 while abs(D) <= F_c
    out["straddle_sign_verdict"] = ("BETTER" if hi < 0 else "WORSE") if out["straddles_floor"] else None
    out["seeds_opposite_beyond_floor"] = bool(len(d_s) == 2 and d_s[0] * d_s[1] < 0 and min(abs(d_s[0]), abs(d_s[1])) > F)
    return out


def family_of(vs):
    if any(v == "NOT-COMPUTABLE" for v in vs):
        return "NOT-COMPUTABLE"
    if all(v == "BETTER" for v in vs):
        return "BETTER"
    if all(v == "WORSE" for v in vs):
        return "WORSE"
    if not any(v in ("BETTER", "WORSE") for v in vs):
        return "NOT-SEPARATED"
    return "MIXED"


def strat_family(sj, pair, seeds, refl=None):
    """STRATEGIC family of one pair: the ADD66 pair read at the third tick (tick 2), `pos` at 30/50/80 m; floor per arc = max of the two arms' seed spreads of the tick-2 pos level."""
    P = (sj or {}).get("pairs", {}).get(pair)
    if P is None:
        return {"family": "NOT-COMPUTABLE", "why": "strategic read missing", "cells": []}
    A, B = C.PAIRS[pair]
    cells = []
    for m, arc in enumerate(C.ARCS):
        b = P["tick2"]["pos"][m]
        if len(seeds) == 2:
            la, lb = (sj["levels_tick2_pos"].get(f"{r}_s{s}") for r in (A, B) for s in seeds), None
        F = None
        lv = sj.get("levels_tick2_pos", {})
        if len(seeds) == 2:
            try:
                F = max(abs(lv[f"{A}_s{seeds[1]}"][m] - lv[f"{A}_s{seeds[0]}"][m]), abs(lv[f"{B}_s{seeds[1]}"][m] - lv[f"{B}_s{seeds[0]}"][m]))
            except KeyError:
                F = None
        else:
            F = (refl or {}).get(f"pos{int(arc)}")
        if b.get("mean") is None or F is None:
            cells.append({"arc_m": arc, "verdict": "NOT-COMPUTABLE", "F": F})
            continue
        v = cell_verdict({"d_s": b["per_seed"], "D": b["mean"], "A_lo": b["lo"], "A_hi": b["hi"]}, F)
        v["arc_m"] = arc
        cells.append(v)
    return {"family": family_of([c["verdict"] for c in cells]), "cells": cells}


def pair_read(name, q3, hw, sj, seeds, refget_q3, refget_hw, refl):  # refget_q3(form, cell, s), refget_hw(form, s): stage-A levels
    A, B = C.PAIRS[name]
    key = pk(A, B)
    P = (q3.get("pairs") or {}).get(key)
    H = (hw.get("pairs") or {}).get(key)
    out = {"A": A, "B": B, "families": {}, "cells": {}, "beside": {}}
    allc = {}
    for f in LAT_LONG_TAC:
        cells = {}
        for cell in CELL_ORDER[f]:
            if cell == "HEADWAY:U":
                F, src = floor_of(lambda r, s: lvl_hw(hw, r, s), A, B, seeds, refget_hw)
                cells[cell] = dict(cell_verdict(None if H is None else H["cells"].get("U"), F), floor_source=src)
            else:
                F, src = floor_of(lambda r, s, c=cell: lvl_q3(q3, r, s, c), A, B, seeds, (lambda r, s, c=cell: refget_q3(r, c, s)) if refget_q3 else None)
                cells[cell] = dict(cell_verdict(None if P is None else P["families"].get(cell), F), floor_source=src)
        out["cells"].update(cells)
        out["families"][f] = family_of([cells[c]["verdict"] for c in cells])
    for cell in BESIDE_CELLS:
        out["beside"][cell] = None if P is None else {k: P["families"][cell][k] for k in ("D", "A_lo", "A_hi", "d_s")}
    sf = strat_family(sj, name, seeds, refl)
    out["cells"]["STRATEGIC:pos_tick2"] = sf["cells"]
    out["families"]["STRATEGIC"] = sf["family"]
    out["headway_beside"] = None if H is None else {c: {k: H["cells"][c][k] for k in ("D", "A_lo", "A_hi", "d_s")} for c in ("C", "F") if c in H["cells"]}
    return out


def ref_getters(refq3, refhw=None):
    """stage-A references (ADD68 RESULT_Q3.json / RESULT_HEADWAY.json, INHERITED): level of UE.A under a form at model seed s -> (q3 getter(form, cell, s), hw getter(form, s))"""
    lev = (refq3 or {}).get("arm_levels") or {}

    def q(form, cell, s):
        d = lev.get(f"I10UEA_{form}_s{s}") or lev.get(f"I10{form}_s{s}")       # our rollout under the form of record, else ADD67's R0Hb20 / R0KHb20 dump (study wrapper = this form, c-EXEC)
        return None if d is None else d.get(cell.split(":")[1] + "|all")

    def h(form, s):
        v = lvl_hw(refhw or {}, f"UEA_{form}", s)
        return v if v is not None else lvl_hw(refhw or {}, form, s)
    return q, h


def flipped_families(pr):
    """Amendment 2 item 3: the families of one pair with every straddling named cell read BETTER / WORSE by its sign (abs(D) <= F_c but the pooled interval excludes 0)."""
    out = {}
    for f in LAT_LONG_TAC:
        vs = []
        for cell, v in pr["cells"].items():
            if isinstance(v, dict) and (cell.split(":")[0] == f or (f == "LONGITUDINAL" and cell == "HEADWAY:U")):
                vs.append(v.get("straddle_sign_verdict") or v["verdict"])
        out[f] = family_of(vs)
    sv = [(v.get("straddle_sign_verdict") or v["verdict"]) for v in pr["cells"].get("STRATEGIC:pos_tick2", [])]
    out["STRATEGIC"] = family_of(sv) if sv else pr["families"]["STRATEGIC"]
    return out


def straddles_of(pr):
    out = [cell for cell, v in pr["cells"].items() if isinstance(v, dict) and v.get("straddles_floor")]
    out += [f"STRATEGIC:pos{int(v['arc_m'])}" for v in pr["cells"].get("STRATEGIC:pos_tick2", []) if v.get("straddles_floor")]
    return out


def table(fams, p3, g, seeds, blocked, dis):
    """-> (base name, reason, P dict).  fams = {pair: {family: verdict}}.  Precedence: BLOCKED, NOT-DECIDABLE, HARMS, seed-opposite SEED-DISAGREE, then the table (+ gaps G1, G2 of Amendment 2 item 2)."""
    one = len(seeds) == 1
    p1 = fams["P1"]
    better = [f for f in FAMS if p1[f] == "BETTER"]
    worse = [f for f in FAMS if p1[f] == "WORSE"]
    nc = [f for f in FAMS if p1[f] == "NOT-COMPUTABLE"]
    nw = lambda n: [f for f in FAMS if fams[n][f] == "WORSE"]       # noqa: E731
    rh_ok = {n: (len(nw(n)) == 0 and not any(fams[n][f] == "NOT-COMPUTABLE" for f in FAMS)) for n in ("P2_ACC_RH", "P2_D1_RH", "P2_D2_RH")}
    b0_worse = [f for f in LAT_LONG_TAC if fams["P2_ACC_B0"][f] == "WORSE"]
    attr2 = len(b0_worse) >= 2 or ("attr_off" in REG)                                       # Amendment 2 item 4: the attribution control; failing it fails P2
    p2 = bool(all(rh_ok.values()) and attr2)
    p4 = bool(len(nw("P4")) == 0 and not any(fams["P4"][f] == "NOT-COMPUTABLE" for f in FAMS))
    gps = {k: v.get("status") for k, v in (g.get("per_seed") or {}).items()}
    g_fail_one = bool(len(gps) == 2 and sorted(gps.values()) == ["FAILS", "HOLDS"])
    g0 = g.get("holds") is True
    P = {"P1": {"families": p1, "better": better, "worse": worse, "not_computable": nc},
         "P2": {"RH_not_worse_on_every_family": rh_ok, "attribution_B0_ACC_worse_on_at_least_two_of_three": attr2, "B0_ACC_worse_families": b0_worse, "pass": p2,
                "note": "Amendment 2 item 4: P2 holds only if RH reads NOT WORSE on every family AND the B0 attribution control holds; if the control fails, P2 fails"},
         "P3": p3, "P4": {"families": fams["P4"], "pass": p4}, "G_T0": {"holds": g.get("holds"), "fails_at_all_seeds": g.get("fails_at_all_seeds"), "per_seed": gps, "fails_at_exactly_one_seed": g_fail_one}}
    p3ok = bool(p3["pass"])
    if blocked and (blocked.get("blocked") is True):
        return "ROLLHIST-BLOCKED", blocked.get("why", "blocked"), P
    if nc:
        return "NOT-DECIDABLE", f"P1 family not computable: {nc}", P
    if worse or g.get("fails_at_all_seeds") is True:
        return "ROLLHIST-HARMS", f"P1 WORSE families {worse}; G-T0 fails at all seeds: {g.get('fails_at_all_seeds')}", P
    if (not one) and dis:
        return "ROLLHIST-SEED-DISAGREE", f"seeds of opposite sign beyond the floor on {dis}", P
    if g.get("holds") is None:
        return "NOT-DECIDABLE", "G-T0 not computable", P
    p1ok2 = len(better) >= 2
    if p1ok2 and p2 and p3ok and p4 and g0:
        return "ROLLHIST-REPAIRS-K-UNNEEDED", "", P
    if p1ok2 and p2 and p3ok and g0 and not p4:
        return "ROLLHIST-REPAIRS", "", P
    if p1ok2 and p2 and p3ok and g_fail_one and "g2_off" not in REG:
        return "ROLLHIST-SEED-DISAGREE", "G2 (Amendment 2): the REPAIRS conditions hold except that G-T0 fails at exactly one seed", P
    if len(better) == 1 and not worse and p2 and p3ok and "g1_off" not in REG:
        return "ROLLHIST-PARTIAL", f"G1 (Amendment 2): exactly one P1 family BETTER ({better}), none WORSE, P2 and P3 hold", P
    if len(better) >= 1 and (not p2 or not p3ok):
        return "ROLLHIST-PARTIAL", "", P
    if not better:
        return "ROLLHIST-NO-EFFECT", "", P
    return "OUTSIDE-TABLE", f"P1 BETTER {better}; P2 {p2}; P3 {p3ok}; P4 {p4}; G-T0 holds {g0}, fails at exactly one seed {g_fail_one}: a combination no row names (Amendment 2 item 2: unclassified, any next lever chosen post hoc and labelled so; per-family verdicts {p1})", P


def reference_pairs(refq3, refhw):
    """P2's reference on UE.A (Amendment 2 item 5): D1, D2 and ACC on UE.A from our stage-A rollouts; printed beside, decides no name.  Raw paired cells D [lo, hi] (no floor applied)."""
    out = {}
    for lab, (a, b) in {"D1": ("UEA_R0G50Hb20", "UEA_R0G0Hb20"), "D2": ("UEA_R0C10Hb20", "UEA_R0G50Hb20"), "ACC": ("UEA_R0C10XHb20", "R0KHb20")}.items():
        key = f"I10{a}_minus_I10{b}"
        P = ((refq3 or {}).get("pairs") or {}).get(key)
        H = ((refhw or {}).get("pairs") or {}).get(key)
        o = {"pair": key, "cells": {}}
        for cell in [c for f in C.NAMED for c in C.NAMED[f]]:
            c = None if P is None else P["families"].get(cell)
            o["cells"][cell] = None if c is None else {"D": c["D"], "interval": [c["A_lo"], c["A_hi"]]}
        u = None if H is None else H["cells"].get("U")
        o["cells"]["HEADWAY:U"] = None if u is None else {"D": u["D"], "interval": [u["A_lo"], u["A_hi"]]}
        out[lab] = o
    return out


def decide(q3, hw, sj, gt0, refq3, seeds, blocked, provenance=None, refhw=None, add68_q3=None, add68_hw=None, strat_floor=None):
    seeds = tuple(seeds)
    one = len(seeds) == 1
    res = {"card": "PREREG_V8F_ROLLHIST.md sha256 781eea87 + Amendment 1 sha256 87928487 + Amendment 2 sha256 15004dc9", "tier": "T1-kin/hb0.2 (executor of record); G-T0 at T0", "evidence_class": "MEASURED (ours; zero training on the dev box; checkpoints from the pod chain)",
           "estimator": "paired episode-cluster bootstrap B 2,000 seed 0 (taniteval/ci.py via the frozen readers); answers 'another draw of EPISODES' only; the second seed answers 'another TRAINING run'; inference is deterministic",
           "seeds": list(seeds), "one_seed": one, "beside_UE_B": "NOT-COMPUTABLE (STAGEB not run: UE_s0.B / UE_s1.B do not exist; RH - UE.B and UE.B's term in F_c are not printed)",
           "pairs": {}, "P": {}, "G_T0": gt0, "blocked": blocked, "provenance": provenance}
    rq, rh = ref_getters(refq3, refhw)
    aq, ah = ref_getters(add68_q3, add68_hw)
    refl = (dict(strat_floor or {}) if one else None)                                   # strategic one-seed floor: no stage-A tick-2 pos level is banked locally -> NOT COMPUTABLE
    for name in C.PAIRS:
        pr = pair_read(name, q3 or {}, hw or {}, sj or {}, seeds, rq if one else None, rh if one else None, refl)
        if one and add68_q3 is not None:                         # ADD68's floors (INHERITED) beside ours (MEASURED)
            A, B = C.PAIRS[name]
            for cell, v in pr["cells"].items():
                if isinstance(v, dict) and cell != "HEADWAY:U":
                    v["floor_ADD68_inherited"] = floor_of(None, A, B, seeds, lambda r, s, c=cell: aq(r, c, s))[0]
                elif isinstance(v, dict) and cell == "HEADWAY:U":
                    v["floor_ADD68_inherited"] = floor_of(None, A, B, seeds, ah)[0]
        res["pairs"][name] = pr
    rn = (sj or {}).get("roll_name") or {}
    r_rh_p, r_rh_c, r_b0_p = rn.get(C.run_name(C.A_KIND, "R0Hb20")), rn.get(C.run_name(C.A_KIND, "R0C10XHb20")), rn.get(C.run_name("B0", "R0Hb20"))      # P03: the treated arm is C.A_KIND (RH by default)
    p3 = {"RH_R0Hb20": r_rh_p, "RH_R0C10XHb20": r_rh_c, "B0_R0Hb20_must_read_DEGRADES": r_b0_p, "pass": bool(r_rh_p == "HOLDS" and r_rh_c == "HOLDS" and r_b0_p == "DEGRADES")}
    g = gt0 or {}
    dis = [cell for cell, v in res["pairs"]["P1"]["cells"].items() if isinstance(v, dict) and v.get("seeds_opposite_beyond_floor")]
    dis += [f"STRATEGIC:pos{int(v['arc_m'])}" for v in res["pairs"]["P1"]["cells"].get("STRATEGIC:pos_tick2", []) if v.get("seeds_opposite_beyond_floor")]
    asread = {n: res["pairs"][n]["families"] for n in C.PAIRS}
    base, why, P = table(asread, p3, g, seeds, blocked, dis)
    res["P"] = P
    strad = {n: straddles_of(res["pairs"][n]) for n in C.PAIRS}
    res["P"]["seed_disagree_cells"] = dis
    res["P"]["straddling_cells_by_pair"] = {n: v for n, v in strad.items() if v}
    res["P"]["straddle_floor_cells"] = strad["P1"]
    res["name_as_read"] = base
    if any(strad.values()) and "any_straddle" in REG and base not in ("ROLLHIST-BLOCKED", "NOT-DECIDABLE", "ROLLHIST-HARMS"):
        base, why = "ROLLHIST-SEED-DISAGREE", "REGRESSION arm: any straddling cell"
    elif any(strad.values()) and "straddle_off" not in REG:
        flipped = {n: flipped_families(res["pairs"][n]) for n in C.PAIRS}
        base2, why2, P2 = table(flipped, p3, g, seeds, blocked, dis)
        res["name_straddles_by_sign"] = base2
        res["P_straddles_by_sign"] = P2
        if base2 != base and base not in ("ROLLHIST-BLOCKED", "NOT-DECIDABLE", "ROLLHIST-HARMS"):
            why = f"Amendment 2 item 3: the floor decides the name: {base} with the straddling cells NOT-SEPARATED, {base2} with them BETTER/WORSE by sign; straddling cells {res['P']['straddling_cells_by_pair']}"
            base = "ROLLHIST-SEED-DISAGREE"
            res["both_names"] = [res["name_as_read"], base2]
    res["P2_reference_UE_A"] = {"from": "our stage-A rollouts under the form of record, MEASURED; the card expects WORSE; beside, decides no name", "pairs": reference_pairs(refq3, refhw)}
    name = ("PROVISIONAL " + base) if (one and base in NAMES) else base
    res["outcome"], res["outcome_reason"] = name, why
    res["next_lever"] = (NEXT_P03 if C.A_KIND == "RHp3" else NEXT_K if C.A_KIND == "RHK" else NEXT_RL if C.A_KIND == "RL" else NEXT).get(base)
    res["treated_arm"] = C.A_KIND
    return res


NEXT = {"ROLLHIST-REPAIRS-K-UNNEEDED": "to the coordinator and the PI: the integrated candidate specified with the causal handover and without readout K; register the base as UE.A + ROLLHIST",
        "ROLLHIST-REPAIRS": "the candidate keeps K beside; DAgger iteration 2 from RH's own states (~2.3 pod-h) only if P4's gap is above floor on two families",
        "ROLLHIST-PARTIAL": "share p = 0.3 on the SAME bank (stage B only, K-B2 holds, ~0.27 pod-h per seed)",
        "ROLLHIST-NO-EFFECT": "p = 0.3 on the same bank, then a 9,000-step stage B (~0.8 pod-h)",
        "ROLLHIST-HARMS": "localise with the beside UE.B (needs STAGEB); next: a bank under readout K or the recovery label (H-ST-RECOVERYLABEL-1)",
        "ROLLHIST-SEED-DISAGREE": "third seed pair: needs UE_s2.A (3.3-3.9 pod-h) plus RH_s2 and B0_s2: back to the PI",
        "ROLLHIST-BLOCKED": "no number; if coverage: bank of tick 1 only or the recovery label; if gate: fix and re-gate"}
# P03 (Amendment 7 DRAFT section 5): the same names for RHp3 (share p = 0.3, same bank); p = 0.3 has now been tried, so the card's PARTIAL / NO-EFFECT lever "p = 0.3" is spent.  A lever the card does not name is a MM pre-registration, never decided here.
NEXT_P03 = dict(NEXT)
NEXT_P03["ROLLHIST-PARTIAL"] = "p = 0.3 is spent and the card names no lever after it: MM pre-registers one before any run (candidates, post hoc until frozen: 9,000-step stage B at p = 0.3 ~0.8 pod-h ESTIMATED; DAgger iteration 2 ~2.3 pod-h; the recovery label)"
NEXT_P03["ROLLHIST-NO-EFFECT"] = "the card's row: p = 0.3 on the same bank (now done), then a 9,000-step stage B at p = 0.3 (~0.8 pod-h ESTIMATED from 0.282 s/step)"

# RHK (Amendment 8 DRAFT section 5): the same names for RHK (RH's dose p = 0.1, a bank generated under readout K, token causal_g0x).  The rows are the MM's pre-registration, committed before any RHK number; a lever the card does not name
# is never decided here.
NEXT_K = dict(NEXT)
NEXT_K["ROLLHIST-REPAIRS-K-UNNEEDED"] = "to the coordinator and the PI: the integrated candidate with the causal handover and without readout K at inference; register the base as UE.A + ROLLHIST on the readout-K bank"
NEXT_K["ROLLHIST-REPAIRS"] = "the candidate keeps K beside, trained on the K bank; DAgger iteration 2 from RHK's own states (~2.3 pod-h ESTIMATED) only if P4's gap is above floor on two families"
NEXT_K["ROLLHIST-PARTIAL"] = "both banks (C*, K) are spent at p = 0.1 and the card names no lever after them: the MM pre-registers one before any run (candidates, post hoc until frozen: the recovery label H-ST-RECOVERYLABEL-1; a 9,000-step stage B on the K bank at p = 0.1 ~0.8 pod-h ESTIMATED; DAgger iteration 2 from RHK's own states ~2.3 pod-h)"
NEXT_K["ROLLHIST-NO-EFFECT"] = "a 9,000-step stage B on the K bank at p = 0.1 (~0.8 pod-h ESTIMATED from 0.282 s/step); p = 0.3 on the K bank only after that, because RHp3 failed G-T0 at both seeds"
NEXT_K["ROLLHIST-HARMS"] = "own rolled history harms under BOTH bank readouts (C*, K): the recovery label for off-route states (H-ST-RECOVERYLABEL-1); the beside UE.B localisation stays NOT-COMPUTABLE until STAGEB runs"


# ---------------------------------------------------------------- synthetic controls (K-SELF, K-PLANT, mutation)
def synth(levels_spread=0.01, plant=None, seeds=(0, 1), self_pair=False, attr=True):
    """synthetic q3 / hw / strat / gt0 / refs of the right shape.  plant = {cell: (D, per_seed list)} for the pair P1 only (cells of LONGITUDINAL etc.)"""
    cells_q3 = [c for f in C.NAMED for c in C.NAMED[f]] + BESIDE_CELLS
    q3 = {"pairs": {}, "arm_levels": {}}
    hw = {"pairs": {}, "levels": {}}
    for run in C.ARM_FORMS_ALL:
        for s in seeds:
            q3["arm_levels"][f"I10{run}_s{s}"] = {c.split(":")[1] + "|all": 1.0 + levels_spread * s for c in cells_q3}
        hw["levels"][f"I10{run}"] = {"cells": {"U": {"per_seed": [0.1 + levels_spread * s for s in seeds]}}}
    for name, (A, B) in C.PAIRS.items():
        fam = {}
        for c in cells_q3:
            d = (0.0, [0.0] * len(seeds), 0.0, 0.0)
            if plant and name == "P1" and c in plant:
                D, ds = plant[c]
                d = (D, ds, D - 0.05, D + 0.05)
            fam[c] = {"D": d[0], "d_s": d[1], "A_lo": d[2], "A_hi": d[3]}
        q3["pairs"][pk(A, B)] = {"families": fam}
        u = (0.0, [0.0] * len(seeds), 0.0, 0.0)
        if plant and name == "P1" and "HEADWAY:U" in plant:
            D, ds = plant["HEADWAY:U"]
            u = (D, ds, D - 0.03, D + 0.03)
        hw["pairs"][pk(A, B)] = {"cells": {"U": {"D": u[0], "d_s": u[1], "A_lo": u[2], "A_hi": u[3]}, "C": {"D": 0.0, "d_s": [0.0] * len(seeds), "A_lo": 0.0, "A_hi": 0.0}}}
    sj = {"roll_name": {r: "HOLDS" for r in C.ARM_FORMS_ALL}, "pairs": {}, "levels_tick2_pos": {}}
    sj["roll_name"][C.run_name("B0", "R0Hb20")] = "DEGRADES"
    for run in C.ARM_FORMS_ALL:
        for s in seeds:
            sj["levels_tick2_pos"][f"{run}_s{s}"] = [10.0 + levels_spread * s] * 3
    for name, (A, B) in C.PAIRS.items():
        z = {"mean": 0.0, "per_seed": [0.0] * len(seeds), "lo": 0.0, "hi": 0.0}
        sj["pairs"][name] = {"tick1": {"pos": [dict(z)] * 3}, "tick2": {"pos": [dict(z) for _ in range(3)]}}
        if plant and name == "P1" and "STRATEGIC:pos" in plant:
            D, ds = plant["STRATEGIC:pos"]
            sj["pairs"][name]["tick2"]["pos"] = [{"mean": D, "per_seed": ds, "lo": D - 0.05, "hi": D + 0.05} for _ in range(3)]
    # P2 attribution control: B0's ACC pair reads WORSE on three families (known value planted in the synthetic read)
    key = pk(*C.PAIRS["P2_ACC_B0"])
    for c in (cells_q3[:6] + ["TACTICAL:ade_tac_m"]) if attr else []:
        q3["pairs"][key]["families"][c] = {"D": 0.5, "d_s": [0.5] * len(seeds), "A_lo": 0.4, "A_hi": 0.6}
    gt0 = {"holds": True, "fails_at_all_seeds": False}
    refq3 = {"arm_levels": {}}
    return q3, hw, sj, gt0, refq3


def selftest():
    rep, ok = {}, True
    q3, hw, sj, gt0, rq = synth()
    r = decide(q3, hw, sj, gt0, rq, (0, 1), None)
    cells_all = [v for p in ("P1",) for v in r["pairs"][p]["cells"].values() if isinstance(v, dict)]
    self_ok = all(v.get("verdict") == "NOT-SEPARATED" and v.get("D") == 0.0 for v in cells_all) and r["outcome"] == "ROLLHIST-NO-EFFECT" and all(r["pairs"]["P1"]["families"][f] == "NOT-SEPARATED" for f in FAMS)
    rep["K-SELF"] = {"pass": bool(self_ok), "must_read": "a self-pair reads exactly 0 on every cell, NOT-SEPARATED, outcome NO-EFFECT", "outcome": r["outcome"]}
    plant = {"LONGITUDINAL:speed_mae_mps": (-0.5, [-0.5, -0.5]), "LONGITUDINAL:along_mae_m": (-0.5, [-0.5, -0.5]), "HEADWAY:U": (-0.2, [-0.2, -0.2]), "TACTICAL:ade_tac_m": (-0.5, [-0.5, -0.5])}
    r2 = decide(q3 := synth(plant=plant)[0], hw := synth(plant=plant)[1], sj, gt0, rq, (0, 1), None)
    f2 = r2["pairs"]["P1"]["families"]
    moved = {f for f in FAMS if f2[f] != "NOT-SEPARATED"}
    rep["K-PLANT"] = {"pass": bool(moved == {"LONGITUDINAL", "TACTICAL"} and f2["LONGITUDINAL"] == "BETTER" and f2["TACTICAL"] == "BETTER"), "must_read": "a plant on LONGITUDINAL (3 cells) and TACTICAL moves exactly those two families to BETTER",
                           "moved": sorted(moved), "outcome": r2["outcome"]}
    # a plant inside the floor must NOT move a cell: spread 0.3 > |D| 0.2
    q3b, hwb, sjb, gtb, rqb = synth(levels_spread=0.3, plant={"TACTICAL:ade_tac_m": (-0.2, [-0.2, -0.2])})
    r3 = decide(q3b, hwb, sjb, gtb, rqb, (0, 1), None)
    rep["K-FLOOR"] = {"pass": bool(r3["pairs"]["P1"]["families"]["TACTICAL"] == "NOT-SEPARATED" and r3["pairs"]["P1"]["cells"]["TACTICAL:ade_tac_m"].get("straddles_floor") is True),
                      "must_read": "an effect below F_c (0.3) is NOT-SEPARATED and flagged straddles_floor"}
    # seed disagreement
    q3c, hwc, sjc, gtc, rqc = synth(plant={"TACTICAL:ade_tac_m": (0.0, [-0.5, 0.5])})
    r4 = decide(q3c, hwc, sjc, gtc, rqc, (0, 1), None)
    rep["K-SEEDS"] = {"pass": bool(r4["outcome"] == "ROLLHIST-SEED-DISAGREE"), "outcome": r4["outcome"]}
    # HARMS
    q3d, hwd, sjd, gtd, rqd = synth(plant={"TACTICAL:ade_tac_m": (0.5, [0.5, 0.5])})
    r5 = decide(q3d, hwd, sjd, gtd, rqd, (0, 1), None)
    rep["K-HARMS"] = {"pass": bool(r5["outcome"] == "ROLLHIST-HARMS"), "outcome": r5["outcome"]}
    rep.update(selftest_r2())
    rep.update(selftest_r3())
    rep["MUTATION_other_seed_arm"] = mutation_provenance()
    rep["ALL_PASS"] = bool(all(v["pass"] for v in rep.values() if isinstance(v, dict)))
    return rep


def _name(kw, gt0=None, reg=(), seeds=(0, 1)):
    q3, hw, sj, g, rq = synth(**kw)
    REG.clear()
    REG.update(reg)
    try:
        r = decide(q3, hw, sj, gt0 if gt0 is not None else g, rq, seeds, None)
    finally:
        REG.clear()
    return r


def selftest_r2():
    """Amendment 2 items 2, 3, 4, 6: each rule has a case that must read as committed and a deliberate-regression arm (the rule switched off) that must read DIFFERENTLY (RED)."""
    rep = {}
    two = {"LONGITUDINAL:speed_mae_mps": (-0.5, [-0.5, -0.5]), "LONGITUDINAL:along_mae_m": (-0.5, [-0.5, -0.5]), "HEADWAY:U": (-0.2, [-0.2, -0.2]), "TACTICAL:ade_tac_m": (-0.5, [-0.5, -0.5])}
    one_fam = {"TACTICAL:ade_tac_m": (-0.5, [-0.5, -0.5])}
    # G1
    a, b = _name({"plant": one_fam}), _name({"plant": one_fam}, reg={"g1_off"})
    rep["R2-G1"] = {"pass": bool(a["outcome"] == "ROLLHIST-PARTIAL" and b["outcome"] != "ROLLHIST-PARTIAL"), "must_read": "exactly one family BETTER, none WORSE, P2 and P3 hold -> ROLLHIST-PARTIAL; with G1 off it must NOT",
                    "read": a["outcome"], "regression_arm_read": b["outcome"]}
    # G2
    g_one = {"holds": False, "fails_at_all_seeds": False, "per_seed": {"0": {"status": "HOLDS"}, "1": {"status": "FAILS"}}}
    a, b = _name({"plant": two}, gt0=g_one), _name({"plant": two}, gt0=g_one, reg={"g2_off"})
    ctl = _name({"plant": two})
    rep["R2-G2"] = {"pass": bool(a["outcome"] == "ROLLHIST-SEED-DISAGREE" and b["outcome"] != "ROLLHIST-SEED-DISAGREE" and ctl["outcome"] == "ROLLHIST-REPAIRS-K-UNNEEDED"),
                    "must_read": "REPAIRS conditions hold but G-T0 fails at exactly one seed -> SEED-DISAGREE; the same plant with G-T0 holding reads REPAIRS-K-UNNEEDED; G2 off must NOT read SEED-DISAGREE",
                    "read": a["outcome"], "regression_arm_read": b["outcome"], "control_G_T0_holds": ctl["outcome"]}
    # straddling cell that flips the name (floor 0.3 > |D| 0.2 on every decisive cell of LONGITUDINAL and TACTICAL)
    st = {"LONGITUDINAL:speed_mae_mps": (-0.2, [-0.2, -0.2]), "LONGITUDINAL:along_mae_m": (-0.2, [-0.2, -0.2]), "HEADWAY:U": (-0.2, [-0.2, -0.2]), "TACTICAL:ade_tac_m": (-0.2, [-0.2, -0.2])}
    a, b = _name({"plant": st, "levels_spread": 0.3}), _name({"plant": st, "levels_spread": 0.3}, reg={"straddle_off"})
    ctl = _name({"plant": st, "levels_spread": 0.01})
    names2 = sorted(a.get("both_names") or [])
    rep["R2-STRADDLE-FLIPS"] = {"pass": bool(a["outcome"] == "ROLLHIST-SEED-DISAGREE" and names2 == ["ROLLHIST-NO-EFFECT", "ROLLHIST-REPAIRS-K-UNNEEDED"] and b["outcome"] == "ROLLHIST-NO-EFFECT" and ctl["outcome"] == "ROLLHIST-REPAIRS-K-UNNEEDED"),
                                "must_read": "straddling cells that move the name -> SEED-DISAGREE and BOTH names printed; with the rule off it reads NO-EFFECT; a floor below the effect (no straddle) reads REPAIRS-K-UNNEEDED",
                                "read": a["outcome"], "both_names": a.get("both_names"), "regression_arm_read": b["outcome"], "control_no_straddle": ctl["outcome"]}
    # straddling cell that does NOT flip the name: one LONGITUDINAL cell straddles, its siblings are NOT-SEPARATED -> LONGITUDINAL MIXED either way, no family BETTER
    nf = {"LONGITUDINAL:speed_mae_mps": (-0.2, [-0.2, -0.2])}
    a, b = _name({"plant": nf, "levels_spread": 0.3}), _name({"plant": nf, "levels_spread": 0.3}, reg={"any_straddle"})
    rep["R2-STRADDLE-KEEPS"] = {"pass": bool(a["outcome"] == "ROLLHIST-NO-EFFECT" and a.get("name_straddles_by_sign") == "ROLLHIST-NO-EFFECT" and not a.get("both_names") and b["outcome"] == "ROLLHIST-SEED-DISAGREE"),
                                "must_read": "a straddling cell that does not change the name keeps the first name (NO-EFFECT); the regression arm (any straddling cell fires) reads SEED-DISAGREE",
                                "read": a["outcome"], "name_with_straddles_by_sign": a.get("name_straddles_by_sign"), "regression_arm_read": b["outcome"]}
    # P2 attribution control
    two_fam = {"LONGITUDINAL:speed_mae_mps": (-0.5, [-0.5, -0.5]), "LONGITUDINAL:along_mae_m": (-0.5, [-0.5, -0.5]), "HEADWAY:U": (-0.2, [-0.2, -0.2]), "TACTICAL:ade_tac_m": (-0.5, [-0.5, -0.5])}
    a, b = _name({"plant": two_fam, "attr": False}), _name({"plant": two_fam, "attr": False}, reg={"attr_off"})
    ctl = _name({"plant": two_fam, "attr": True})
    rep["R2-P2-ATTRIBUTION"] = {"pass": bool(a["P"]["P2"]["pass"] is False and a["outcome"] == "ROLLHIST-PARTIAL" and b["P"]["P2"]["pass"] is True and ctl["P"]["P2"]["pass"] is True),
                                "must_read": "the B0 ACC pair NOT worse on two of three families -> P2 fails (two families BETTER, P2 fails -> PARTIAL); control (B0 ACC WORSE on three) -> P2 holds; the check off -> P2 would hold",
                                "read": a["outcome"], "P2_pass": a["P"]["P2"]["pass"], "regression_arm_P2_pass": b["P"]["P2"]["pass"], "control_P2_pass": ctl["P"]["P2"]["pass"]}
    # item 6: one seed -> PROVISIONAL on every committed name; the table of names it can print
    q3, hw, sj, g, rq = synth(seeds=(0,), plant=two)
    ref_q3 = {"arm_levels": {}}
    for f in C.FORMS:
        for s in (0, 1):
            ref_q3["arm_levels"][f"I10UEA_{f}_s{s}"] = {c.split(":")[1] + "|all": 1.0 + 0.01 * s for c in [c for f2 in C.NAMED for c in C.NAMED[f2]]}
    ref_hw = {"levels": {f"I10UEA_{f}": {"cells": {"U": {"per_seed": [0.1, 0.11]}}} for f in C.FORMS}}
    sfl = {"pos30": 0.01, "pos50": 0.01, "pos80": 0.01}
    r = decide(q3, hw, sj, g, ref_q3, (0,), None, None, ref_hw, strat_floor=sfl)
    r_nf = decide(q3, hw, sj, g, ref_q3, (0,), None, None, ref_hw)
    rep["R2-PROVISIONAL"] = {"pass": bool(r["outcome"].startswith("PROVISIONAL ROLLHIST-") and r_nf["outcome"] == "NOT-DECIDABLE"), "must_read": "a one-seed read prefixes PROVISIONAL to every committed name; without a STRATEGIC one-seed floor it reads NOT-DECIDABLE, never a name", "read": r["outcome"], "no_strategic_floor_read": r_nf["outcome"]}
    q3, hw, sj, g, rq = synth(seeds=(0,))
    sj["pairs"] = {n: {"tick1": {}, "tick2": {"pos": [{"mean": 0.0, "per_seed": [0.0], "lo": 0.0, "hi": 0.0}] * 3}} for n in C.PAIRS}
    names1 = []
    for plant in (None, one_fam):
        q3, hw, sj1, g, rq = synth(seeds=(0,), plant=plant)
        r = decide(q3, hw, sj1, g, ref_q3, (0,), None, None, ref_hw, strat_floor=sfl)
        names1.append(r["outcome"])
    rep["R2-PROVISIONAL-NAMES"] = {"pass": bool(all(n.startswith("PROVISIONAL ROLLHIST-") for n in names1)), "names": names1}
    return rep


def load_strat_floor(path):
    """the one-seed STRATEGIC floor file of rh_strat_floor.py -> {pos30, pos50, pos80}; None if absent or its controls did not pass."""
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return d.get("floor") if d.get("controls_all_pass") is True and d.get("floor") else None


def selftest_r3():
    """R3: a one-seed read WITH the floor file prints a STRATEGIC verdict; the regression arm WITHOUT it prints NOT-DECIDABLE."""
    import tempfile
    plant = {"TACTICAL:ade_tac_m": (-0.5, [-0.5, -0.5])}
    ref_q3 = {"arm_levels": {}}
    for f in C.FORMS:
        for s in (0, 1):
            ref_q3["arm_levels"][f"I10UEA_{f}_s{s}"] = {c.split(":")[1] + "|all": 1.0 + 0.01 * s for c in [c for f2 in C.NAMED for c in C.NAMED[f2]]}
    ref_hw = {"levels": {f"I10UEA_{f}": {"cells": {"U": {"per_seed": [0.1, 0.11]}}} for f in C.FORMS}}
    d = tempfile.mkdtemp(dir=C.HERE + "/tmp")
    fp, bad = os.path.join(d, "floor.json"), os.path.join(d, "floor_bad.json")
    json.dump({"controls_all_pass": True, "floor": {"pos30": 0.3, "pos50": 1.5, "pos80": 1.4}}, open(fp, "w"))
    json.dump({"controls_all_pass": False, "floor": None}, open(bad, "w"))
    q3, hw, sj, g, rq = synth(seeds=(0,), plant=plant)
    with_f = decide(q3, hw, sj, g, ref_q3, (0,), None, None, ref_hw, strat_floor=load_strat_floor(fp))
    reg = decide(q3, hw, sj, g, ref_q3, (0,), None, None, ref_hw, strat_floor=load_strat_floor(""))
    bad_r = decide(q3, hw, sj, g, ref_q3, (0,), None, None, ref_hw, strat_floor=load_strat_floor(bad))
    sf = with_f["pairs"]["P1"]["families"]["STRATEGIC"]
    for x in (fp, bad):
        os.remove(x)
    os.rmdir(d)
    ok = bool(sf != "NOT-COMPUTABLE" and with_f["outcome"] != "NOT-DECIDABLE" and reg["outcome"] == "NOT-DECIDABLE" and bad_r["outcome"] == "NOT-DECIDABLE")
    return {"R3-STRAT-FLOOR-WIRED": {"pass": ok, "must_read": "one-seed read with the floor file prints a STRATEGIC verdict; without it (regression arm), or with a floor file whose controls failed, it prints NOT-DECIDABLE",
                                     "strategic_family_with_floor": sf, "outcome_with_floor": with_f["outcome"], "regression_arm_outcome": reg["outcome"], "failed_controls_floor_outcome": bad_r["outcome"]}}


def provenance(w, shas, seeds=None):
    """every rollout's recorded checkpoint sha256 equals the arm's own (RH_s0 != RH_s1); -> (ok, problems).  shas = {"RH_s0": sha256, ...}"""
    bad = []
    if len({shas[k] for k in shas}) != len(shas):
        bad.append("two arms share one checkpoint sha256")
    for run in C.ARM_FORMS_ALL:
        kind = run.split("_", 1)[0]
        for s in (C.SEEDS if seeds is None else seeds):          # READ FIX 3: only the read's seeds (a one-seed read has no seed-1 run records)
            f = f"{w}/t1/I10{run}_s{s}/ticks/run_I10{run}_s{s}.txt"
            if not os.path.isfile(f):
                bad.append(f"run record missing {run}_s{s}")
                continue
            t = open(f, encoding="utf-8").read()
            i = t.find("ckpt sha256 ")
            got = t[i + 12:i + 76] if i >= 0 else ""
            if got != shas.get(f"{kind}_s{s}"):
                bad.append(f"{run}_s{s}: rollout used checkpoint {got[:12]}, arm {kind}_s{s} is {str(shas.get(f'{kind}_s{s}'))[:12]}")
    return (not bad), bad


def mutation_provenance():
    """feed the reader the OTHER seed's arm: a run record of RH_s1 naming RH_s0's checkpoint must make the provenance check RED; the unmutated tree must read green"""
    import tempfile
    d = tempfile.mkdtemp(prefix="rhmut_", dir=os.environ.get("TMP") or None)
    shas = {f"{k}_s{s}": (f"{i}{s}" * 32)[:64] for i, k in enumerate(C.RUN_KINDS) for s in C.SEEDS}
    for run in C.ARM_FORMS_ALL:
        kind = run.split("_", 1)[0]
        for s in C.SEEDS:
            p = f"{d}/t1/I10{run}_s{s}/ticks"
            os.makedirs(p)
            open(f"{p}/run_I10{run}_s{s}.txt", "w").write(f"arm {kind}_s{s} ckpt sha256 {shas[f'{kind}_s{s}']} executor\n")
    green, _ = provenance(d, shas)
    p = f"{d}/t1/I10{C.run_name('RH', 'R0C10XHb20')}_s1/ticks/run_I10{C.run_name('RH', 'R0C10XHb20')}_s1.txt"
    open(p, "w").write(f"arm RH_s1 ckpt sha256 {shas['RH_s0']} executor\n")
    red, why = provenance(d, shas)
    for root, dirs, files in os.walk(d, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for x in dirs:
            os.rmdir(os.path.join(root, x))
    os.rmdir(d)
    return {"pass": bool(green and not red), "unmutated_green": bool(green), "mutated_red": bool(not red), "reason_seen": why[:2]}


def p03_answer(res, rh):
    """ONE line: did p = 0.3 repair P2 / P3?  RHp3's P2 / P3 (pooled over the read's seeds, as the card's table reads them) against RH's from the finished PARTIAL read (beside, read only)."""
    P = res.get("P") or {}
    p2, p3 = (P.get("P2") or {}).get("pass"), (P.get("P3") or {}).get("pass")
    rp = (rh or {}).get("P") or {}
    r2, r3 = (rp.get("P2") or {}).get("pass"), (rp.get("P3") or {}).get("pass")
    word = "REPAIRS both P2 and P3" if (p2 and p3) else "repairs P2 but NOT P3" if p2 else "repairs P3 but NOT P2" if p3 else "repairs NEITHER P2 nor P3"
    return (f"p = 0.3 {word}: RHp3 P2 {p2}, P3 {p3} (outcome {res.get('outcome')}); RH at p = 0.1 (the PARTIAL read): P2 {r2}, P3 {r3}"
            + ("" if rh else " [the RH result file was not found]"))


def k_answer(res, rh, rhp3=None):
    """ONE line: does the readout-K bank repair P2 / P3 and hold G-T0?  RHK's P2 / P3 (pooled over the read's seeds) and G-T0 against RH's (C* bank, p = 0.1, the PARTIAL read; both False, G-T0 failed at seed 0), read only, beside."""
    P = res.get("P") or {}
    p2, p3 = (P.get("P2") or {}).get("pass"), (P.get("P3") or {}).get("pass")
    g = P.get("G_T0") or {}
    gword = "G-T0 HOLDS" if g.get("holds") else ("G-T0 FAILS at all seeds" if g.get("fails_at_all_seeds") else "G-T0 FAILS at one seed" if g.get("fails_at_exactly_one_seed") else f"G-T0 reads {g.get('holds')!r}")
    rp = (rh or {}).get("P") or {}
    r2, r3 = (rp.get("P2") or {}).get("pass"), (rp.get("P3") or {}).get("pass")
    rg = ((rp.get("G_T0") or {}).get("per_seed"))
    word = "REPAIRS both P2 and P3" if (p2 and p3) else "repairs P2 but NOT P3" if p2 else "repairs P3 but NOT P2" if p3 else "repairs NEITHER P2 nor P3"
    return (f"K bank {word} and {gword}: RHK P2 {p2}, P3 {p3} (outcome {res.get('outcome')}); RH on the C* bank (the PARTIAL read): P2 {r2}, P3 {r3}, G-T0 per seed {rg}"
            + ("" if rh else " [the RH result file was not found]"))

# RL (Amendment 9 DRAFT): the same names for RL (RH's argv + exactly `--dagger-label speed_v1`: the SPEED-consistent label on matcher-ok own states, the RH bank, p = 0.1).  The rows are the MM's pre-registration, committed before any RL
# number; a lever the card does not name is never decided here.
NEXT_RL = dict(NEXT_K)
NEXT_RL["ROLLHIST-REPAIRS"] = "the speed-consistent label repairs the card's P2 / P3 on the RH bank: DAgger iteration 2 from RL's own states only if P4's gap is above floor on two families (the MM pre-registers it)"
NEXT_RL["ROLLHIST-PARTIAL"] = "the speed label is spent at p = 0.1 and the card names no lever after it: the MM pre-registers one before any run (candidates, post hoc until frozen: the label at p = 0.3; DAgger iteration 2 from RL's own states)"
NEXT_RL["ROLLHIST-NO-EFFECT"] = "the speed label moved nothing at p = 0.1: check the label's reach on the arm's own selfcheck census (relabelled share of ok records) BEFORE reading it as a refutation; the MM pre-registers the next lever"
NEXT_RL["ROLLHIST-HARMS"] = "own rolled history harms even with the speed-consistent label on matcher-ok states: the MM pre-registers the next lever (candidates, post hoc until frozen: pose-offset recovery on un-re-anchored inputs; DAgger iteration 2)"


def rl_answer(res, rh):
    """ONE line: does the speed-consistent label repair P2 / P3 and hold G-T0?  RL's P2 / P3 and G-T0 against RH's (same bank, same dose, human label; the PARTIAL read), read only, beside."""
    P = res.get("P") or {}
    p2, p3 = (P.get("P2") or {}).get("pass"), (P.get("P3") or {}).get("pass")
    g = P.get("G_T0") or {}
    gword = "G-T0 HOLDS" if g.get("holds") else ("G-T0 FAILS at all seeds" if g.get("fails_at_all_seeds") else "G-T0 FAILS at one seed" if g.get("fails_at_exactly_one_seed") else f"G-T0 reads {g.get('holds')!r}")
    rp = (rh or {}).get("P") or {}
    r2, r3 = (rp.get("P2") or {}).get("pass"), (rp.get("P3") or {}).get("pass")
    rg = ((rp.get("G_T0") or {}).get("per_seed"))
    word = "REPAIRS both P2 and P3" if (p2 and p3) else "repairs P2 but NOT P3" if p2 else "repairs P3 but NOT P2" if p3 else "repairs NEITHER P2 nor P3"
    return (f"speed label {word} and {gword}: RL P2 {p2}, P3 {p3} (outcome {res.get('outcome')}); RH on the same bank with the human label (the PARTIAL read): P2 {r2}, P3 {r3}, G-T0 per seed {rg}"
            + ("" if rh else " [the RH result file was not found]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q3")
    ap.add_argument("--hw")
    ap.add_argument("--strat")
    ap.add_argument("--strat-floor", default="")
    ap.add_argument("--gt0")
    ap.add_argument("--refs-q3", default="")
    ap.add_argument("--refs-hw", default="")
    ap.add_argument("--add68-q3", default="")
    ap.add_argument("--add68-hw", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--w", default=C.OUT)
    ap.add_argument("--blocked", default="")
    ap.add_argument("--rh-result", default=C.OUT + "/read/RESULT_ROLLHIST.json", help="P03: the finished PARTIAL read of RH at p = 0.1 (read only), printed beside")
    ap.add_argument("--out", default=C.OUT + "/read/RESULT_ROLLHIST.json")
    a = ap.parse_args()
    if a.selftest:
        rep = selftest()
        print(json.dumps(rep, indent=1))
        print("ZZ RH-READ-SELFTEST all hold: " + str(rep["ALL_PASS"]))
        return 0 if rep["ALL_PASS"] else 1
    seeds = tuple(int(x) for x in a.seeds.split(","))
    J = lambda p: json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None      # noqa: E731
    shas = {}
    for k in C.RUN_KINDS:
        for s in seeds:
            f = f"{C.CK}/{k}_s{s}/ckpt_last.pt"
            shas[f"{k}_s{s}"] = C.sha256_file(f) if os.path.isfile(f) else None
    prov_ok, prov = provenance(a.w, {k: v for k, v in shas.items() if v}, seeds) if all(shas.values()) else (False, ["checkpoint missing"])
    blocked = J(a.blocked) or {"blocked": False}
    if not prov_ok:
        blocked = {"blocked": True, "why": "provenance: " + "; ".join(prov[:3])}
    res = decide(J(a.q3), J(a.hw), J(a.strat), J(a.gt0), J(a.refs_q3), seeds, blocked, {"ok": prov_ok, "problems": prov, "checkpoint_sha256": shas}, J(a.refs_hw), J(a.add68_q3), J(a.add68_hw), strat_floor=load_strat_floor(a.strat_floor))
    if C.A_KIND == "RHp3":
        res["p03_answer"] = p03_answer(res, J(a.rh_result))
        print("P03:", res["p03_answer"])
    if C.A_KIND == "RHK":
        res["k_answer"] = k_answer(res, J(a.rh_result))
        print("K:", res["k_answer"])
    if C.A_KIND == "RL":
        res["rl_answer"] = rl_answer(res, J(a.rh_result))
        print("RL:", res["rl_answer"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("OUTCOME:", res["outcome"], "|", res["outcome_reason"])
    print("P1 families:", res["P"]["P1"]["families"], "| P2", res["P"]["P2"]["pass"], "| P3", res["P"]["P3"]["pass"], "| P4", res["P"]["P4"]["pass"], "| G-T0", res["P"]["G_T0"])
    print("wrote", a.out)
    print("ZZ RH-READ DONE ZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
