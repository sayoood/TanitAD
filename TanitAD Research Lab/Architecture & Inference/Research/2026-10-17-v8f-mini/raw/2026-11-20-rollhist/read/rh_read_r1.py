"""rh_read.py -- the ROLLHIST reader (card PREREG_V8F_ROLLHIST.md sha256 781eea87, sections 3.2 to 3.4; Amendment 1 sha256 87928487).  numpy only; it reads the JSON of the frozen readers and decides nothing else.
Inputs (all produced by rh_read_chain.sh):  RESULT_Q3.json (families LONGITUDINAL / LATERAL / TACTICAL cells, q3_read_r3.py), RESULT_HEADWAY.json (headway U, headway_read_r2.py),
READ_STRAT_ROLLHIST.json (STRATEGIC, rh_strat.py over the frozen ADD66 reader), GT0_ROLLHIST.json (guard G-T0, rh_t0.py), refs (stage-A: ADD68 RESULT_Q3.json, INHERITED), pod/bank record (blocked flags).
It prints EXACTLY ONE committed outcome name (section 3.4, precedence BLOCKED, HARMS, SEED-DISAGREE, then the table order); on one seed the same name prefixed PROVISIONAL.  Outside the committed table
it prints OUTSIDE-TABLE and says which cell broke, never a name made up.  Cells are per family, never pooled.
Cell verdict (ADD63's family reader + the card): BETTER / WORSE when the paired pooled interval excludes 0 on that side, both seed means agree in sign and abs(D) > F_c; else NOT-SEPARATED.
F_c (two seeds) = max(abs(A_s1 - A_s0), abs(B_s1 - B_s0)) over the two arms of the pair under the same form (RH - B0: the card's max(abs(RH_s1-RH_s0), abs(B0_s1-B0_s0))); UE.B's term is NOT COMPUTABLE (STAGEB not run).
F_U,c (one seed) = max over the two arms' forms of abs(UE_s1.A - UE_s0.A) from the stage-A references (ADD68 RESULT_Q3.json arm_levels, INHERITED); a form without a reference makes the cell NOT COMPUTABLE.
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
    out["straddles_floor"] = bool((hi < 0 or lo > 0) and (all(x < 0 for x in d_s) or all(x > 0 for x in d_s)) and abs(D) <= F)
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
        d = lev.get(f"I10{form}_s{s}")
        return None if d is None else d.get(cell.split(":")[1] + "|all")

    def h(form, s):
        return lvl_hw(refhw or {}, form, s)
    return q, h


def decide(q3, hw, sj, gt0, refq3, seeds, blocked, provenance=None, refhw=None):
    seeds = tuple(seeds)
    one = len(seeds) == 1
    res = {"card": "PREREG_V8F_ROLLHIST.md sha256 781eea87 + Amendment 1 sha256 87928487", "tier": "T1-kin/hb0.2 (executor of record); G-T0 at T0", "evidence_class": "MEASURED (ours; zero training on the dev box; checkpoints from the pod chain)",
           "estimator": "paired episode-cluster bootstrap B 2,000 seed 0 (taniteval/ci.py via the frozen readers); answers 'another draw of EPISODES' only; the second seed answers 'another TRAINING run'; inference is deterministic",
           "seeds": list(seeds), "one_seed": one, "beside_UE_B": "NOT-COMPUTABLE (STAGEB not run: UE_s0.B / UE_s1.B do not exist; RH - UE.B and UE.B's term in F_c are not printed)",
           "pairs": {}, "P": {}, "G_T0": gt0, "blocked": blocked, "provenance": provenance}
    # stage-A references for the one-seed floor and the reference line of P2
    rq, rh = ref_getters(refq3, refhw)
    refl = None
    if one:
        refl = {}                                                # strategic one-seed floor: no stage-A tick-2 pos level is banked locally -> NOT COMPUTABLE
    for name in C.PAIRS:
        res["pairs"][name] = pair_read(name, q3 or {}, hw or {}, sj or {}, seeds, rq if one else None, rh if one else None, refl)
    fam = lambda n: res["pairs"][n]["families"]       # noqa: E731
    p1 = fam("P1")
    better = [f for f in FAMS if p1[f] == "BETTER"]
    worse = [f for f in FAMS if p1[f] == "WORSE"]
    nc = [f for f in FAMS if p1[f] == "NOT-COMPUTABLE"]
    res["P"]["P1"] = {"families": p1, "better": better, "worse": worse, "not_computable": nc}
    # P2
    nw = lambda n: [f for f in FAMS if fam(n)[f] == "WORSE"]       # noqa: E731
    rh_ok = {n: (len(nw(n)) == 0 and not any(fam(n)[f] == "NOT-COMPUTABLE" for f in FAMS)) for n in ("P2_ACC_RH", "P2_D1_RH", "P2_D2_RH")}
    b0_worse = [f for f in LAT_LONG_TAC if fam("P2_ACC_B0")[f] == "WORSE"]
    attr2 = len(b0_worse) >= 2
    res["P"]["P2"] = {"RH_not_worse_on_every_family": rh_ok, "attribution_B0_ACC_worse_on_at_least_two_of_three": attr2, "B0_ACC_worse_families": b0_worse,
                      "pass": bool(all(rh_ok.values()) and attr2),
                      "reference_UE_A": "ADD68 pair ACC on UE.A at the 2026-11 read: expected WORSE (INHERITED, read_boundtier/pod/RESULT_Q3.json); D1/D2 on UE.A under hb0.2 have NO reference (ADD63 ran the old executor)"}
    # P3
    rn = (sj or {}).get("roll_name") or {}
    r_rh_p, r_rh_c, r_b0_p = rn.get(C.run_name("RH", "R0Hb20")), rn.get(C.run_name("RH", "R0C10XHb20")), rn.get(C.run_name("B0", "R0Hb20"))
    res["P"]["P3"] = {"RH_R0Hb20": r_rh_p, "RH_R0C10XHb20": r_rh_c, "B0_R0Hb20_must_read_DEGRADES": r_b0_p, "pass": bool(r_rh_p == "HOLDS" and r_rh_c == "HOLDS" and r_b0_p == "DEGRADES")}
    # P4
    res["P"]["P4"] = {"families": fam("P4"), "pass": bool(len(nw("P4")) == 0 and not any(fam("P4")[f] == "NOT-COMPUTABLE" for f in FAMS))}
    # G-T0
    g = gt0 or {}
    res["P"]["G_T0"] = {"holds": g.get("holds"), "fails_at_all_seeds": g.get("fails_at_all_seeds")}
    # seed disagreement on the decisive cells of P1
    dis, strad = [], []
    for f in LAT_LONG_TAC:
        for cell, v in res["pairs"]["P1"]["cells"].items():
            if cell.split(":")[0] != f and not (f == "LONGITUDINAL" and cell == "HEADWAY:U"):
                continue
            if isinstance(v, dict):
                if v.get("seeds_opposite_beyond_floor"):
                    dis.append(cell)
                if v.get("straddles_floor"):
                    strad.append(cell)
    for v in res["pairs"]["P1"]["cells"].get("STRATEGIC:pos_tick2", []):
        if v.get("seeds_opposite_beyond_floor"):
            dis.append(f"STRATEGIC:pos{int(v['arc_m'])}")
        if v.get("straddles_floor"):
            strad.append(f"STRATEGIC:pos{int(v['arc_m'])}")
    res["P"]["seed_disagree_cells"], res["P"]["straddle_floor_cells"] = dis, strad
    # outcome
    base = None
    why = ""
    if blocked and (blocked.get("blocked") is True):
        base, why = "ROLLHIST-BLOCKED", blocked.get("why", "blocked")
    elif nc:
        base, why = "NOT-DECIDABLE", f"P1 family not computable: {nc}"
    elif worse or g.get("fails_at_all_seeds") is True:
        base, why = "ROLLHIST-HARMS", f"P1 WORSE families {worse}; G-T0 fails at all seeds: {g.get('fails_at_all_seeds')}"
    elif (not one) and (dis or strad):
        base, why = "ROLLHIST-SEED-DISAGREE", f"seed-opposite cells {dis}; decisive cells straddling the floor {strad}"
    elif g.get("holds") is None:
        base, why = "NOT-DECIDABLE", "G-T0 not computable"
    else:
        p1ok2, p1ok1 = len(better) >= 2, len(better) >= 1
        p2, p3, p4, g0 = res["P"]["P2"]["pass"], res["P"]["P3"]["pass"], res["P"]["P4"]["pass"], g.get("holds") is True
        if p1ok2 and p2 and p3 and p4 and g0:
            base = "ROLLHIST-REPAIRS-K-UNNEEDED"
        elif p1ok2 and p2 and p3 and g0 and not p4:
            base = "ROLLHIST-REPAIRS"
        elif p1ok1 and (not p2 or not p3):
            base = "ROLLHIST-PARTIAL"
        elif not better:
            base = "ROLLHIST-NO-EFFECT"
        else:
            base, why = "OUTSIDE-TABLE", f"P1 BETTER {better}; P2 {p2}; P3 {p3}; P4 {p4}; G-T0 holds {g0}: a combination the card's table does not name (not invented)"
    name = ("PROVISIONAL " + base) if (one and base in NAMES) else base
    res["outcome"], res["outcome_reason"] = name, why
    res["next_lever"] = NEXT.get(base)
    return res


NEXT = {"ROLLHIST-REPAIRS-K-UNNEEDED": "to the coordinator and the PI: the integrated candidate specified with the causal handover and without readout K; register the base as UE.A + ROLLHIST",
        "ROLLHIST-REPAIRS": "the candidate keeps K beside; DAgger iteration 2 from RH's own states (~2.3 pod-h) only if P4's gap is above floor on two families",
        "ROLLHIST-PARTIAL": "share p = 0.3 on the SAME bank (stage B only, K-B2 holds, ~0.27 pod-h per seed)",
        "ROLLHIST-NO-EFFECT": "p = 0.3 on the same bank, then a 9,000-step stage B (~0.8 pod-h)",
        "ROLLHIST-HARMS": "localise with the beside UE.B (needs STAGEB); next: a bank under readout K or the recovery label (H-ST-RECOVERYLABEL-1)",
        "ROLLHIST-SEED-DISAGREE": "third seed pair: needs UE_s2.A (3.3-3.9 pod-h) plus RH_s2 and B0_s2: back to the PI",
        "ROLLHIST-BLOCKED": "no number; if coverage: bank of tick 1 only or the recovery label; if gate: fix and re-gate"}


# ---------------------------------------------------------------- synthetic controls (K-SELF, K-PLANT, mutation)
def synth(levels_spread=0.01, plant=None, seeds=(0, 1), self_pair=False):
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
    for c in cells_q3[:6] + ["TACTICAL:ade_tac_m"]:
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
    rep["MUTATION_other_seed_arm"] = mutation_provenance()
    rep["ALL_PASS"] = bool(all(v["pass"] for v in rep.values() if isinstance(v, dict)))
    return rep


def provenance(w, shas):
    """every rollout's recorded checkpoint sha256 equals the arm's own (RH_s0 != RH_s1); -> (ok, problems).  shas = {"RH_s0": sha256, ...}"""
    bad = []
    if len({shas[k] for k in shas}) != len(shas):
        bad.append("two arms share one checkpoint sha256")
    for run in C.ARM_FORMS_ALL:
        kind = run.split("_", 1)[0]
        for s in C.SEEDS:
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
    shas = {f"{k}_s{s}": (f"{i}{s}" * 32)[:64] for i, k in enumerate(C.KINDS) for s in C.SEEDS}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q3")
    ap.add_argument("--hw")
    ap.add_argument("--strat")
    ap.add_argument("--gt0")
    ap.add_argument("--refs-q3", default="")
    ap.add_argument("--refs-hw", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--w", default=C.OUT)
    ap.add_argument("--blocked", default="")
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
    for k in C.KINDS:
        for s in seeds:
            f = f"{C.CK}/{k}_s{s}/ckpt_last.pt"
            shas[f"{k}_s{s}"] = C.sha256_file(f) if os.path.isfile(f) else None
    prov_ok, prov = provenance(a.w, {k: v for k, v in shas.items() if v}) if all(shas.values()) else (False, ["checkpoint missing"])
    blocked = J(a.blocked) or {"blocked": False}
    if not prov_ok:
        blocked = {"blocked": True, "why": "provenance: " + "; ".join(prov[:3])}
    res = decide(J(a.q3), J(a.hw), J(a.strat), J(a.gt0), J(a.refs_q3), seeds, blocked, {"ok": prov_ok, "problems": prov, "checkpoint_sha256": shas}, J(a.refs_hw))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("OUTCOME:", res["outcome"], "|", res["outcome_reason"])
    print("P1 families:", res["P"]["P1"]["families"], "| P2", res["P"]["P2"]["pass"], "| P3", res["P"]["P3"]["pass"], "| P4", res["P"]["P4"]["pass"], "| G-T0", res["P"]["G_T0"])
    print("wrote", a.out)
    print("ZZ RH-READ DONE ZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
