"""ds_read_b2.py -- the DSCALE10K BLOCK-2 DECIDING READ (PREREG_V8F_DSCALE10K sections 4 and 6; frozen card; MM rulings 2026-11-16 / 11-17).  CPU, zero training.  Writes RESULT_DSCALE10K_B2.json + .md.
Arms read: seed 0 = B_s0, U_s0, B_rep, R_s0 (block 1, read from the block-1 read's folder) and seed 1 = B_s1, U_s1 (block 2, this folder), each at stage A (19,164 steps, mechanism, beside) and stage B (3,000 steps, deciding).
Block-3 arms (UE_s0, BL_s0, X_s0, seeds 3 / 4) print NOT-YET.
Everything of block 1 is imported UNEDITED from ds_read.py (pure logic, loaders, K-RE helpers); this file adds only what a two-seed read needs.
TIERS, ESTIMATOR, EVIDENCE CLASS: as the block-1 read.  LONGITUDINAL / LATERAL / TACTICAL: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), 141 episodes / 2,399 windows.
STRATEGIC: T0 at the window's first tick (--ages fresh; real / donor nav).  Mechanism / reach: T0 diagnostics, never driving performance.  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000,
seed 0 ("another draw of EPISODES"); training-run variance = the floor F (one replicate pair, B_rep seed 2 against B_s0 seed 0, per stage).  Evidence class: MEASURED (ours; zero training).
WHAT BLOCK 2 ADDS (each a reading choice where the card is silent; listed in BUILD_DSCALE10K_B2_READ.md)
  1. The label is computed from the family statuses at BOTH seeds at the deciding stage B: Delta_s = U_s - B_s (seed 0: U_s0 - B_s0; seed 1: U_s1 - B_s1), the card's cell rule, headline cells and family rule unchanged.
  2. The floor F_cell is the seed-0 replicate pair's |B_rep - B_s0| per cell and stage, applied at both seeds (there is one replicate pair, B_rep; the card names no seed-1 replicate).
  3. Seed-0 rows are REUSED from RESULT_DSCALE10K_B1.json; the pipeline recomputes them from the dumps and K-RE (B1) requires every field to equal the banked row exactly.  No seed-0 number is recomputed silently.
  4. 'At both seeds' = the same family at both seeds (MM deviation 4, accepted).  A family whose headline cells are all missing (NOT-YET) at either seed blocks the label: NOT-YET, never read as 'no gain'.
  5. Cross-venue beside at seed 1: abs(B_s1.B - box-2 OBD-CTL_s1) and the paired rows B_s1.B and U_s1.B against OBD-CTL_s1 (seed 0 is reused from block 1).
  6. The zero-nav route pass runs at BOTH seeds (S_CKPT_zero@80 beside S1 / S2): zero, realx and donorx run on the STRAT1 tree (the only evaluator with --nav-withheld); K-EVALX must pass at BOTH seeds (seed 0 from the block-1 read's
     KEVALX.json, seed 1 from this folder's), else STRATEGIC is NOT-YET.  The block-1 GT-block unit fix (ds_t0.to_common_frame) is taken as it is.
  7. UNIT-SENSITIVE: a reach or T label that changes with the unit the latcons RMSE is expressed in (the reference arm's frame = deciding; the arm's own frame and physical units, beside) is reported as UNIT-SENSITIVE.
     A UNIT-SENSITIVE reach is neither REACHED nor NO-REACH: the outcome is NOT-YET (reach UNIT-SENSITIVE).  A UNIT-SENSITIVE T label is beside (it never enters the outcome).  Only the point r is given in the other units.
Usage:  python ds_read_b2.py --out RESULT_DSCALE10K_B2.json [--b1-result ...] [--dumps0 DIR --t0mech0 DIR --t0route0 DIR] [--dumps1 DIR --t0mech1 DIR --t0route1 DIR] [--n-boot 2000]"""
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
import ds_read as R  # noqa: E402  (block-1 module, unedited: pure logic, loaders, K-RE helpers)

OR, FAMILIES, HEADLINE, CELLS, FAMILY = R.OR, R.FAMILIES, R.HEADLINE, R.CELLS, R.FAMILY
SEED_ARMS = {0: ("B_s0", "U_s0"), 1: ("B_s1", "U_s1")}
NEXT = R.NEXT


# ------------------------------------------------------------------------------------------------------------------------------------------ pure logic
def outcome_b2(per_seed, reach_ok, blocked=None, reach_read=True, reach_unit_sensitive=False):
    """per_seed {0: {family: family_status}, 1: {...}} (deciding stage B) -> [label].  Order: BLOCKED, (reach unread -> NOT-YET), (reach UNIT-SENSITIVE -> NOT-YET), NO-REACH, a NOT-YET family at either seed -> NOT-YET,
    then the card's table (ds_read.outcome_label, unedited).  A family with every headline cell missing is an unread family, never 'no gain'.
    UNIT-SENSITIVE (block-2 rule): a reach label that flips with the unit the latcons RMSE is expressed in (reference frame / the arm's own frame / physical units) is neither REACHED nor NO-REACH: no driving label is read on it."""
    if blocked:
        return ["BLOCKED"]
    if not reach_read:
        return ["NOT-YET"]
    if reach_unit_sensitive:
        return ["NOT-YET (reach UNIT-SENSITIVE)"]
    if not reach_ok:
        return ["NO-REACH"]
    if set(per_seed) != {0, 1}:
        return ["NOT-YET"]
    if any(v.get("status") == "NOT-YET" for s in per_seed.values() for v in s.values()):
        return ["NOT-YET"]
    return R.outcome_label(per_seed, True, None)                      # BLOCKED, the reach precondition and NO-REACH were decided above


def outcome_from_mech(per_seed, mech, blocked, mech_given):
    """the outcome of the read: the reach precondition (deciding unit) and its UNIT-SENSITIVE flag are taken from the mechanism block"""
    reach = mech.get("reach", {})
    return outcome_b2(per_seed, reach.get("deciding_label") == "REACHED", blocked, reach_read=mech_given, reach_unit_sensitive=bool(reach.get("unit_sensitive")))


def seed_families(rows):
    """{family: family_status} of one seed at the deciding stage from its U - B cell rows"""
    return {f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}


def kevalx_problem_b2(t0route, arms=("B_s1", "U_s1")):
    """-> reason or None: the seed-1 K-EVALX record (written by ds_run.py route in THIS folder) is a PASS over the real and donor passes of the seed-1 stage-B arms (50c70ae8 vs 008f48b7).  The seed-0 record is read by ds_read.kevalx_problem."""
    p = f"{t0route}/KEVALX.json"
    if not os.path.isfile(p):
        return "KEVALX.json missing"
    k = json.load(open(p, encoding="utf-8"))
    want = {f"{a}.B {v}" for a in arms for v in ("real", "donor")}
    if not k.get("pass") or set(k.get("items", {})) != want or not all(v["identical"] for v in k["items"].values()):
        return f"KEVALX.json is not a PASS over {sorted(want)}"
    return None


ROUTE_VARIANTS = ("real", "donor", "zero", "realx", "donorx")


def route_problem(rdirs, arms_route, kx):
    """-> reason or None: every stage-B route pass (real, donor, zero, realx, donorx) of every arm exists, and K-EVALX passed at both seeds (kx {seed: reason or None})"""
    miss = [f"{arm}.B {v}" for arm, s in arms_route for v in ROUTE_VARIANTS if not os.path.exists(f"{rdirs[s]}/eval_DS-{arm}-B_{v}.npz")]
    if miss:
        return f"T0 route passes missing: {miss}"
    if any(kx.values()):                                     # K-EVALX: the zero / realx / donorx passes ran on the 008f48b7 evaluator; admissible only if its real and donor output equals the 50c70ae8 output (both seeds)
        return f"K-EVALX: {kx} (the zero passes are not admissible)"
    return None


def kre_strat_b1(recomputed, banked_block):
    """K-RE (B1) of the STRATEGIC seed-0 cells: every banked cell, S_CKPT_zero@80 included (the zero pass runs at both seeds)"""
    return kre_b1(recomputed, banked_block)


def units_of(m):
    """r of one mech_pair result in each unit it carries: the deciding (reference-frame) r always; the arm's own frame and physical units when the arm was re-expressed (GTFIX r_beside)"""
    out = {"deciding": m["r"]}
    out.update({k: v for k, v in (m.get("frame", {}).get("r_beside") or {}).items() if k in ("A_frame", "physical_units")})
    return out


def label_by_units(decide, m, *extra):
    """UNIT-SENSITIVE rule (block 2): decide(r, delta, *extra) is run on r in EVERY unit the pair carries (the paired MSE delta and its interval are the deciding frame's: only the point r is given beside in the others).
    -> the deciding decision, plus labels_by_unit; the label is 'UNIT-SENSITIVE' when the labels in the units differ, else the deciding label."""
    rs = units_of(m)
    by = {u: decide(r, m["mse_delta"], *extra) for u, r in rs.items()}
    labels = {u: d["label"] for u, d in by.items()}
    sens = len(set(labels.values())) > 1
    d = dict(by["deciding"])
    d.update({"deciding_label": by["deciding"]["label"], "unit_sensitive": sens, "labels_by_unit": labels, "r_by_unit": rs, "label": "UNIT-SENSITIVE" if sens else by["deciding"]["label"]})
    return d


def _nn(x):
    """NaN -> None recursively (so a banked NaN equals a recomputed NaN in the exact comparison)"""
    if isinstance(x, dict):
        return {k: _nn(v) for k, v in x.items()}
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def kre_b1(recomputed, banked):
    """K-RE (B1): every recomputed seed-0 row equals the banked block-1 row, every field exactly.  Zero rows compared, a banked cell not recomputed, or any mismatch fails it."""
    rc, bk = _nn(json.loads(json.dumps(recomputed, default=float))), _nn(banked)
    n, bad = R.kre_compare(rc, bk)
    absent = sorted(set(bk) - set(rc))
    return {"pass": bool(n > 0 and not bad and not absent), "rows_compared": n, "n_banked_cells": len(bk), "cells_banked_not_recomputed": absent[:12], "mismatches": bad[:12], "n_mismatches": len(bad)}


def flat_mech(m):
    return {"r": m["r"], "rmse_A": m["rmse_slot"]["A"], "rmse_B": m["rmse_slot"]["B"], "delta": m["mse_delta"]["point"], "lo": m["mse_delta"]["lo"], "hi": m["mse_delta"]["hi"], "separated": m["mse_delta"]["separated"]}


def kre_mech(recomputed, banked):
    rows = {"U_vs_B": flat_mech(recomputed["U_vs_B_stage_A"]), "R_vs_U": flat_mech(recomputed["R_vs_U_stage_A"]), "floor": {"floor_mse": recomputed["stage_A_T_floor"]["floor_mse"]}}
    bk = {"U_vs_B": flat_mech(banked["U_vs_B_stage_A"]), "R_vs_U": flat_mech(banked["R_vs_U_stage_A"]), "floor": {"floor_mse": banked["stage_A_T_floor"]["floor_mse"]}}
    n, bad = R.kre_compare(rows, bk, tuple(rows["U_vs_B"]) + ("floor_mse",))
    return {"pass": bool(n >= 15 and not bad), "rows_compared": n, "mismatches": bad[:12]}


def _lab(d):
    """a label_by_units result as text: the label, and per unit when UNIT-SENSITIVE"""
    if not d:
        return "NOT-YET"
    return d["label"] + (f" {d['labels_by_unit']} (r by unit {{{', '.join(f'{u}: {r:+.4f}' for u, r in d['r_by_unit'].items())}}})" if d.get("unit_sensitive") else "")


def md_cell(v):
    if not v or v.get("status") != "OK":
        return "n/a"
    f = v.get("floor")
    return f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']} F={'-' if f is None else round(f, 4)} {'GAIN' if v['gain'] else ('COST' if v['cost'] else '.')}"


def md_two(rows0, rows1, cells):
    out = ["| cell | seed 0 (U_s0 - B_s0): Delta [95 % interval] sep F verdict | seed 1 (U_s1 - B_s1) | n windows |", "|---|---|---|---|"]
    for c in cells:
        a, b = rows0.get(c), rows1.get(c)
        out.append(f"| {c} | {md_cell(a)} | {md_cell(b)} | {(a or b or {}).get('n_windows', 0)} |")
    return "\n".join(out)


def fam_cells(rows, f):
    return {c: v for c, v in rows.items() if (FAMILY.get(c.split(":")[-1]) == f or (f == "STRATEGIC" and c.split(":")[0] in ("S1", "S2", "RC-A")))}


# ------------------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1-result", default=L.B1_ROOT + "/out/RESULT_DSCALE10K_B1.json")
    ap.add_argument("--dumps0", default=L.B1_ROOT + "/t1")
    ap.add_argument("--t0mech0", default=L.B1_ROOT + "/out/t0mech")
    ap.add_argument("--t0route0", default=L.B1_ROOT + "/out/t0route")
    ap.add_argument("--ck0", default=L.B1_ROOT + "/ckpts")
    ap.add_argument("--dumps1", default=L.T1)
    ap.add_argument("--t0mech1", default=L.OUT + "/t0mech")
    ap.add_argument("--t0route1", default=L.OUT + "/t0route")
    ap.add_argument("--ck1", default=L.CK)
    ap.add_argument("--init-record", default="F:/TanitAD-scratch/datascale/gate/pod_out/gate/init.json")
    ap.add_argument("--box2-dumps", default="F:/TanitAD-scratch/otband_read/t1")
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args(argv)
    t_start = time.time()
    CI = L.load_ci()
    d0 = lambda arm, st: f"{a.dumps0}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    d1 = lambda arm, st: f"{a.dumps1}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    dpath = lambda arm, st: (d1 if L.ARMS[arm] == 1 else d0)(arm, st)  # noqa: E731
    items0, items1 = [x for x in L.READ_ITEMS], list(L.B2_ITEMS)
    need = [(x, dpath(*x)) for x in items0 + items1]
    ready = {L.item(*x): os.path.isdir(p) for x, p in need}
    rep = {"tool": "ds_read_b2.py", "card": "PREREG_V8F_DSCALE10K (frozen; sha256 48a74901...; MM rulings 2026-11-16 / 11-17)", "block": 2, "code_of_record": L.COMMIT,
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC": "T0 at the window's first tick (--ages fresh)",
                    "mechanism / reach": "T0 (world-model and tactical diagnostic, never driving performance)"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training); seed-0 rows REUSED from RESULT_DSCALE10K_B1.json under K-RE (B1)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = F (one replicate pair per stage)",
           "committed_rules": __doc__.split("WHAT BLOCK 2 ADDS")[1].split("Usage:")[0].strip(), "items_ready": ready, "controls": {"problems": []}, "not_yet": {k: "NOT-YET (block 3)" for k in L.NOT_YET_B2}}
    prob = rep["controls"]["problems"]
    miss = [k for k, v in ready.items() if not v]
    if miss:
        rep["outcome"], rep["reason"] = ["NOT-YET"], f"T1 dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", miss)
        return 1
    b1 = json.load(open(a.b1_result, encoding="utf-8")) if os.path.isfile(a.b1_result) else None
    if b1 is None:
        prob.append(f"K-RE (B1) NOT-GIVEN: {a.b1_result} is missing, so no seed-0 row can be reused")
    import obd_geom as G
    geom = G.build()
    dg = R.load_dump_n(dpath("B_s0", "A"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for (arm, st), _p in need:
        cells[(arm, st)] = R.load_arm_cells(geom, dg, J, glat, glon, dpath(arm, st), L.EVAL_N_EPISODES)
        print(f"loaded {arm}.{st} {time.time() - t_start:.0f}s", flush=True)
    eid = dg["eid"]
    rep["controls"]["K1_self_pair"] = OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    F = {st: OR.floors_from(cells[("B_rep", st)], cells[("B_s0", st)]) for st in ("A", "B")}
    rep["floors"] = {"definition": "F_cell = |B_rep - B_s0| per cell, per stage (one replicate pair, B_rep seed 2), applied at BOTH seeds", "stage_A": F["A"], "stage_B": F["B"]}
    pairs = {}
    for s, (b_, u_) in SEED_ARMS.items():
        for st in ("A", "B"):
            pairs[f"U-B_s{s}.{st}"] = R.pair_cells(CI, cells[(u_, st)], cells[(b_, st)], eid, F[st], a.n_boot)
    # STRATEGIC (T0, stage B), both seeds; the floor is the seed-0 replicate pair's, as for every cell
    SS, Gr, ci, t = R.load_strategic_helpers()
    rdir = {0: a.t0route0, 1: a.t0route1}
    arms_route = [("B_s0", 0), ("U_s0", 0), ("B_rep", 0), ("B_s1", 1), ("U_s1", 1)]
    strat = {"tier": rep["tier"]["STRATEGIC"], "headline": list(HEADLINE["STRATEGIC"]), "not_in_this_card": {
        "detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"},
        "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}},
        "note": "the C0-type v9 nav shares its source with RC-A, so S1 is echo-prone for every arm alike: Delta (U - B) is read, never the level", "status": "NOT-YET"}
    why = route_problem(rdir, arms_route, {0: R.kevalx_problem(a.t0route0), 1: kevalx_problem_b2(a.t0route1)})
    if why:
        strat["reason"] = why
    else:
        R_ = {arm: R.route_arrays(rdir[s], arm, "B", SS, Gr, ci, t) for arm, s in arms_route}
        sf = R.strat_floors(R_["B_rep"], R_["B_s0"])
        sp = {s: R.strat_pair(CI, R_[SEED_ARMS[s][1]], R_[SEED_ARMS[s][0]], eid, sf, a.n_boot) for s in (0, 1)}
        strat.update({"status": "OK", "floors": sf, "n_scorable_80m": {arm: int(np.isfinite(R_[arm]["S_CKPT_real@80"]).sum()) for arm in R_}, "best_constant_80m_reference": 0.794,
                      "U-B_s0.B": R.slim(sp[0]), "U-B_s1.B": R.slim(sp[1])})
        for s in (0, 1):
            pairs[f"U-B_s{s}.B"].update({c: v for c, v in sp[s].items() if c in HEADLINE["STRATEGIC"]})
    # K-RE (B1): seed-0 rows recomputed through this pipeline equal the banked block-1 rows exactly; the banked rows are then the rows of record
    rows_rec = {"A": R.slim(pairs["U-B_s0.A"]), "B": R.slim(pairs["U-B_s0.B"])}
    seed0_src = "recomputed (no banked block-1 result)"
    rows0 = {"A": pairs["U-B_s0.A"], "B": pairs["U-B_s0.B"]}
    if b1 is not None:
        kre = {f"U-B_s0.{st}": kre_b1(rows_rec[st], b1["pairs"][f"U-B_s0.{st}"]) for st in ("A", "B")}
        if strat["status"] == "OK" and "U-B_s0.B" in strat:
            kre["STRATEGIC U-B_s0.B (S1 / S2 / RC-A)"] = kre_strat_b1(strat["U-B_s0.B"], b1["families"]["STRATEGIC"]["block"]["U-B_s0.B"])
        rep["controls"]["K-RE_B1"] = {"pass": all(v["pass"] for v in kre.values()), "per_pair": kre}
        if not rep["controls"]["K-RE_B1"]["pass"]:
            prob.append("K-RE (B1): a recomputed seed-0 row differs from the banked block-1 row (or a banked cell was not recomputed)")
        else:
            rows0 = {st: b1["pairs"][f"U-B_s0.{st}"] for st in ("A", "B")}
            seed0_src = "reused from RESULT_DSCALE10K_B1.json (K-RE B1: every recomputed field equals the banked row exactly)"
    rows1 = {st: pairs[f"U-B_s1.{st}"] for st in ("A", "B")}
    fam = {0: {st: seed_families(rows0[st]) for st in ("A", "B")}, 1: {st: seed_families(rows1[st]) for st in ("A", "B")}}
    for s in (0, 1):                                                       # stage A carries no route passes (card deviation 8 of block 1)
        fam[s]["A"]["STRATEGIC"] = {"status": "NOT-YET", "gains": [], "costs": [], "missing": list(HEADLINE["STRATEGIC"]), "reason": "T0 route passes are made for the stage-B (deciding) checkpoints only"}
    # T0: reach (seed 0, R_s0 vs U_s0), mechanism at both seeds (stage A, beside)
    import ds_t0 as T
    nm = lambda arm: T.name_of(arm, "A")  # noqa: E731
    mech = {"tier": rep["tier"]["mechanism / reach"]}
    mech_given = False
    try:
        u_b0 = T.mech_pair(a.t0mech0, nm("U_s0"), nm("B_s0"), 0, a.n_boot)
        r_u = T.mech_pair(a.t0mech0, nm("R_s0"), nm("U_s0"), 0, a.n_boot, ref_name=nm("U_s0"))
        fl = T.mech_floor(a.t0mech0, nm("B_rep"), nm("B_s0"))
        mech.update({"U_vs_B_stage_A_seed0": u_b0, "R_vs_U_stage_A": r_u, "stage_A_T_floor": fl, "reach": label_by_units(T.decide_reach, r_u), "T_label_seed0": label_by_units(T.decide_t, u_b0, fl["floor_mse"])})
        mech_given = True
        if b1 is not None and "reach" in b1.get("mechanism_stage_A_T0", {}):
            bm = b1["mechanism_stage_A_T0"]
            rep["controls"]["K-RE_mech_B1"] = kre_mech({"U_vs_B_stage_A": u_b0, "R_vs_U_stage_A": r_u, "stage_A_T_floor": fl}, bm)
            if not rep["controls"]["K-RE_mech_B1"]["pass"]:
                prob.append("K-RE (B1 mechanism): the recomputed seed-0 reach / mechanism row differs from the banked one")
        else:
            prob.append("K-RE (B1 mechanism) NOT-GIVEN: the block-1 result carries no mechanism block")
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        mech["status"] = "NOT-YET"
        mech["reason"] = f"{type(e).__name__}: {e}"
    try:
        u_b1 = T.mech_pair(a.t0mech1, nm("U_s1"), nm("B_s1"), 0, a.n_boot)
        mech["U_vs_B_stage_A_seed1"] = u_b1
        if "stage_A_T_floor" in mech:
            mech["T_label_seed1"] = label_by_units(T.decide_t, u_b1, mech["stage_A_T_floor"]["floor_mse"])
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        mech["seed1_status"] = "NOT-YET"
        mech["seed1_reason"] = f"{type(e).__name__}: {e}"
    rep["mechanism_stage_A_T0"] = mech
    blocked_items = {}
    for (arm, st) in items0:
        blocked_items[L.item(arm, st)] = L.bank_problem(arm, st, a.ck0)
    for (arm, st) in items1:
        blocked_items[L.item(arm, st)] = L.bank_problem(arm, st, a.ck1)
    blocked = {k: v for k, v in blocked_items.items() if v}
    rep["blocked_items"] = blocked
    labels = outcome_from_mech({0: fam[0]["B"], 1: fam[1]["B"]}, mech, list(blocked) or None, mech_given)
    if prob:
        labels = ["BLOCKED"] if "BLOCKED" in labels else [f"{x} (CONTROL FAILED: {len(prob)} problem(s); not quotable)" for x in labels]
    rep["outcome"] = labels
    rep["seed_rows"] = {"seed_0": seed0_src, "seed_1": "computed by this read (U_s1 - B_s1)"}
    rep["family_status"] = {f"seed_{s}": {"stage_B_deciding": fam[s]["B"], "stage_A_beside": fam[s]["A"]} for s in (0, 1)}
    rep["next_step"] = [f"{x.split(' (')[0]}: {NEXT[x.split(' (')[0]]}" for x in labels if x.split(" (")[0] in NEXT] or ["no label: see the reason fields"]
    rep["init_control_beside"] = R.init_control(json.load(open(a.init_record, encoding="utf-8")) if os.path.exists(a.init_record) else None)
    # cross-venue beside: seed 0 reused from block 1, seed 1 computed (OBD-CTL_s1 against B_s1.B and U_s1.B)
    cv = {"definition": "abs(B_s.B - box-2 OBD-CTL_s) per cell: venue plus nondeterminism; beside only, never the control", "seed_0": (b1 or {}).get("cross_venue_beside", {"status": "NOT-YET", "reason": "no block-1 result"})}
    xv = f"{a.box2_dumps}/dump_OBD-CTL_s1_m10_causal_heads"
    if os.path.isdir(xv):
        cx = R.load_arm_cells(geom, dg, J, glat, glon, xv, L.EVAL_N_EPISODES)
        cv["seed_1"] = {"abs_diff": OR.floors_from(cells[("B_s1", "B")], cx), "B_s1.B-OBD-CTL_s1": R.slim(R.pair_cells(CI, cells[("B_s1", "B")], cx, eid, None, a.n_boot)),
                        "U_s1.B-OBD-CTL_s1": R.slim(R.pair_cells(CI, cells[("U_s1", "B")], cx, eid, None, a.n_boot))}
    else:
        cv["seed_1"] = {"status": "NOT-YET", "reason": f"the banked box-2 dump {xv} is not present"}
    rep["cross_venue_beside"] = cv
    n_w, n_e = int(len(eid)), int(len(np.unique(eid)))
    rows_b = {0: rows0["B"], 1: rows1["B"]}
    rep["families"] = {}
    srow = {0: strat.get("U-B_s0.B", {}), 1: strat.get("U-B_s1.B", {})}                # STRATEGIC: all RC-A cells (30 / 50 / 80 m), S1, S2
    for f in FAMILIES:
        rb = {s: (srow[s] if f == "STRATEGIC" else rows_b[s]) for s in (0, 1)}
        rep["families"][f] = {"tier": rep["tier"]["STRATEGIC"] if f == "STRATEGIC" else rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "headline": list(HEADLINE[f]),
                              "cells_stage_B": {"seed_0": {c: R.slim({c: v})[c] for c, v in fam_cells(rb[0], f).items()}, "seed_1": {c: R.slim({c: v})[c] for c, v in fam_cells(rb[1], f).items()}},
                              "cells_stage_A_beside": ({"seed_0": {c: R.slim({c: v})[c] for c, v in fam_cells(rows0["A"], f).items()}, "seed_1": {c: R.slim({c: v})[c] for c, v in fam_cells(rows1["A"], f).items()}} if f != "STRATEGIC" else "NOT-YET (route passes are stage B only)"),
                              "status_stage_B": {"seed_0": fam[0]["B"][f], "seed_1": fam[1]["B"][f]}, "n_windows": n_w, "n_episodes": n_e, "estimator": "paired episode-cluster bootstrap",
                              "evidence_class": "MEASURED"}
    rep["families"]["STRATEGIC"]["block"] = strat
    rep["families"]["PRIMARY (ADE / FDE, beside, decide nothing)"] = {"cells_stage_B": {"seed_0": {c: R.slim({c: v})[c] for c, v in fam_cells(rows0["B"], "PRIMARY").items()},
                                                                                       "seed_1": {c: R.slim({c: v})[c] for c, v in fam_cells(rows1["B"], "PRIMARY").items()}}}
    rep["pairs"] = {f"U-B_s1.{st}": R.slim(pairs[f"U-B_s1.{st}"]) for st in ("A", "B")}
    rep["pairs"].update({f"U-B_s0.{st}": R.slim(rows0[st]) for st in ("A", "B")})
    rep["not_built_beside_rows"] = {"DK-VIOL counts": "NOT COMPUTABLE here: beside row (OBD o4), outside the 24 cells; the MM ruling routes it to the W-FAM-STRAT1-TTC-TACCONF instrument once it lands",
                                    "T-LAT-F1 / T-LON-F1, goal macro-F1 by provenance, nav-echo split": f"NOT COMPUTABLE here (n={n_w} windows, heads are in the --dump-heads dumps): same instrument, same ruling",
                                    "per-country / platform eval breakdown (card 8)": "named work item, not in the cell set"}
    rep["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = ["# DSCALE10K block 2: the deciding read (B_s0 / U_s0 and B_s1 / U_s1), MEASURED; 141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2,000 seed 0", "",
          f"Outcome: **{labels}**. Controls: problems {prob or 'none'}. Reach (R_s0, T0): {_lab(mech.get('reach'))}. T mechanism (T0, beside) seed 0: {_lab(mech.get('T_label_seed0'))}; seed 1: {_lab(mech.get('T_label_seed1'))}.",
          f"Seed 0: {seed0_src}. Seed 1: computed here. Floor F: the seed-0 replicate pair (B_rep), at both seeds. Evidence class MEASURED; tiers: LONGITUDINAL / LATERAL / TACTICAL T1-kin, STRATEGIC T0.", "",
          "Family status at the deciding stage B: " + json.dumps({f"seed_{s}": {f: v['status'] for f, v in fam[s]['B'].items()} for s in (0, 1)}), ""]
    for f in FAMILIES:
        md += [f"## {f} ({rep['families'][f]['tier']}), n = {n_w} windows / {n_e} episodes", "", md_two(rep["families"][f]["cells_stage_B"]["seed_0"], rep["families"][f]["cells_stage_B"]["seed_1"], list(rep["families"][f]["cells_stage_B"]["seed_0"])), ""]
    md += ["## STRATEGIC members not in this card", "", json.dumps(strat["not_in_this_card"]), "", "## Cross-venue beside (seed 1, OBD-CTL_s1)", "", json.dumps({k: v for k, v in cv["seed_1"].items() if k == "status" or k == "reason"}) or ""]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", labels, "| problems:", len(prob), "|", rep["wall_s"], "s")
    return 1 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
