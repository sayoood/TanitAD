"""ds_read.py -- the DSCALE10K BLOCK-1 read (PREREG_V8F_DSCALE10K sections 4 and 6; frozen card; MM rulings 2026-11-16 / 11-17).  CPU, zero training.  Writes RESULT_DSCALE10K_B1.json + .md.
Arms read: B_s0, U_s0, B_rep (stage A = 19,164 steps, stage B = OBD-CTL fine-tune 3,000 steps) and R_s0 (stage A).  Everything else (B_s1, U_s1, UE_s0, BL_s0, X_s0, seeds 3 / 4) prints NOT-YET.
TIERS.  LONGITUDINAL / LATERAL / TACTICAL: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner), 141 episodes / 2,399 windows.
STRATEGIC: T0 at the window's first tick (--ages fresh; real / donor / zero nav).  Mechanism (T's latcons RMSE), GT-oracle and absmask evals: T0, a world-model and tactical diagnostic, never driving performance.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES" only); training-run variance = the floor F (one replicate pair, B_rep seed 2 against B_s0 seed 0, per stage).
COMMITTED RULES (card 4 and 6, in force; nothing below changes them)
  cell: Delta = U - B at the same stage; gain = separated on the better side AND |Delta| > F_cell; cost = separated on the worse side AND |Delta| > F_cell; F_cell = |B_rep - B_s0| per cell, per stage.
  headline cells decide the family label: LONGITUDINAL L1 op speed MAE, L2 op headway abs error; LATERAL A1 op yaw-rate MAE, A2 op curvature MAE; TACTICAL T1 TAC-LAT acc, T2 TAC-LON acc;
  STRATEGIC S1 RC-A checkpoint position error at 80 m, S2 S_CKPT skill (real - donor) at 80 m.  Family gain at a seed: >= 1 headline gain and no headline cost; family cost: >= 1 headline cost.
  PRECONDITION (read first): R_s0 against U_s0 at stage A on T's latcons: r_R = RMSE(R)/RMSE(U) - 1 >= +5 %, separated (sign rule) -> REACHED; else NO-REACH (U's read is not interpretable; no driving label).
  Mechanism label (stage A, T0, beside): T-DATA-LIMITED = r = RMSE(U)/RMSE(B) - 1 <= -5 %, separated (sign rule), abs(MSE delta) > the stage-A T floor.
  Outcome label (needs BOTH seeds; block 1 has seed 0 only, so the label prints NOT-YET and the seed-0 family read prints beside it): DATA-HELPS (>= 2 families gain at both seeds, no cost at either seed),
  DATA-HELPS-PARTIAL (exactly 1 family gain at both seeds, no cost), DATA-NEUTRAL (no gain at either seed, no cost, reach passed), DATA-HURTS (a family cost at both seeds), MIXED (a family gain and a family cost, both at
  both seeds), SEEDS-DISAGREE (a gain or cost at one seed only, not covered above), BLOCKED (a gate FAIL, coverage < 0.5, a refused arm).  NO-REACH precedes all.  Each label prints its committed next step.
  STRATEGIC members not in this card, per member with the reason: detected information n/a (no ctx head), derived goals not in this arm, strategic intent 6-30 s not computable (no output beyond 80 m of arc).
Usage:  python ds_read.py --dumps DIR --t0mech DIR --t0route DIR --out RESULT_DSCALE10K_B1.json [--init-record init.json] [--kre-t1] [--kre-t0] [--n-boot 2000] [--smoke]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, HERE + "/theory/obd/read", HERE + "/theory/otband/read"):
    if p not in sys.path:
        sys.path.insert(0, p)
import ds_lib as L  # noqa: E402
import obd_read as OR  # noqa: E402  (numpy only at module level; the 24 cells, cell_delta, floors_from, compare_cells)

CELLS, FAMILY, ACCURACY = OR.CELLS, OR.FAMILY, OR.ACCURACY
OP = OR.OP
HEADLINE = {"LONGITUDINAL": (f"{OP}:speed_mae_mps", f"{OP}:dk_hw_abs_err_m"), "LATERAL": (f"{OP}:yaw_rate_mae_degps", f"{OP}:curvature_mae_1pm"),
            "TACTICAL": ("TAC-LAT", "TAC-LON"), "STRATEGIC": ("S1:pos_err_m@80", "S2:skill@80")}
FAMILIES = tuple(HEADLINE)
STRAT_HIGHER = {"S2:skill@80": True}                         # every other strategic cell is an error (lower is better)
ARCS = (30, 50, 80)
NEXT = {
    "NO-REACH": "U's read is not interpretable and no driving label is given. Next: the G-LIVE reach table re-run on the trained U checkpoint, a count of 10k windows per batch from the logged sampler, a fix, and a re-run of R and U.",
    "DATA-HELPS": "1. The PI's rebase ruling: U as the base of record for STRAT1 / ACTRSP descendants (Q3). 2. The next rung, 2x the union (about 29k clips, inside the 2x extrapolation limit), encode plus storage go (Q4). 3. The VLM-CoT pass on the new clips as the tactical-coverage lever.",
    "DATA-HELPS-PARTIAL": "Item 2 of DATA-HELPS for that family. For each family without a gain, its named lever: LONGITUDINAL -> LON-PLAN / HWM-PLAN (AMEND_HWMPLAN_3); LATERAL -> adapter training (LAT-TOK's FAIL branch); TACTICAL -> the VLM-CoT pass on the 10k; STRATEGIC -> STRAT1's nav text on the U base.",
    "DATA-NEUTRAL": "Corpus size is not the binding term at 3.2x for this recipe. Next: if UE - BL is better by more than 2F_A, the lever is compute on the union (UE through stage B, both seeds); else LoRA on DINOv3 on the U set (PREREG_V8F_LORA.md), plus the VLM-CoT pass.",
    "DATA-HURTS": "Diagnose dilution. Run X_s0 (10k only) and the per-goal-class F1 (VLM vs geometry bits). Read eval-141 per country and platform. A mixture arm needs a sampler-weight flag, so it goes in an amendment with its G-DVB entry.",
    "MIXED": "The trade-off is reported per family, never pooled. Next: the mixture arm (amendment) and X_s0.",
    "SEEDS-DISAGREE": "Seeds 3 and 4 for B and U (about 2.6 pod-h, E). Re-read with F from the replicate pairs; the label is set then.",
    "BLOCKED": "Back to the MM with the failing check. The arm is not launched.",
}


# ------------------------------------------------------------------------------------------------------------------------------------------ pure logic
def family_status(rows, heads):
    """rows {cell: cell_delta row}; heads: the family's headline cells -> {status gain|cost|none|NOT-YET, gains, costs, missing}.  Cost wins over gain; a missing / not-applicable headline cell is listed."""
    gains = [c for c in heads if rows.get(c, {}).get("gain")]
    costs = [c for c in heads if rows.get(c, {}).get("cost")]
    missing = [c for c in heads if c not in rows or rows[c].get("status") != "OK"]
    if len(missing) == len(heads):
        return {"status": "NOT-YET", "gains": [], "costs": [], "missing": missing}
    return {"status": "cost" if costs else ("gain" if gains else "none"), "gains": gains, "costs": costs, "missing": missing}


def seed_families(rows_by_family):
    """{family: family_status} -> ({families with a family gain}, {families with a family cost})"""
    return ({f for f, v in rows_by_family.items() if v["status"] == "gain"}, {f for f, v in rows_by_family.items() if v["status"] == "cost"})


def outcome_label(per_seed, reach_ok, blocked=None):
    """per_seed {0: {family: status}, 1: {...}} -> list of labels.  Pure; the unit tests plant every row of the card's section 6 table.
    Reading choices (the card is silent; listed for the MM): a family cost / gain 'at both seeds' is the SAME family at both seeds; MIXED needs a both-seed gain AND a both-seed cost; a cost at one seed only,
    or a cost in different families at the two seeds, is SEEDS-DISAGREE."""
    if blocked:
        return ["BLOCKED"]
    if not reach_ok:
        return ["NO-REACH"]
    if set(per_seed) != {0, 1}:
        return ["NOT-YET"]
    g0, c0 = seed_families(per_seed[0])
    g1, c1 = seed_families(per_seed[1])
    gb, cb = g0 & g1, c0 & c1
    if gb and cb:
        return ["MIXED"]
    if cb:
        return ["DATA-HURTS"]
    if c0 | c1:
        return ["SEEDS-DISAGREE"]
    if len(gb) >= 2:
        return ["DATA-HELPS"]
    if len(gb) == 1:
        return ["DATA-HELPS-PARTIAL"]
    if not (g0 | g1):
        return ["DATA-NEUTRAL"]
    return ["SEEDS-DISAGREE"]


def seed0_reading(fam0, reach_ok, blocked=None):
    """what the seed-0 family read would label if seed 1 mirrored it: PROVISIONAL, one seed, never a label"""
    return {"label": "NOT-YET (seed 1 absent)", "provisional_if_seed_1_agrees": outcome_label({0: fam0, 1: fam0}, reach_ok, blocked), "families_gain": sorted(seed_families(fam0)[0]),
            "families_cost": sorted(seed_families(fam0)[1])}


def pair_cells(CI, arm, ref, eid, floors, nb):
    return OR.compare_cells(CI, arm, ref, eid, floors, nb)


def slim(cells):
    keep = ("status", "delta", "lo", "hi", "separated", "floor", "above_floor", "gain", "cost", "n_windows", "n_episodes")
    return {c: {k: v[k] for k in keep if k in v} for c, v in cells.items()}


def strat_pair(CI, arm, ref, eid, floors, nb):
    """S1 / S2 (and the other RC-A cells) of arm minus ref with the cell rule; floors {cell: float|None}"""
    return {c: OR.cell_delta(CI, arm[c], ref[c], eid, STRAT_HIGHER.get(c, False), None if floors is None else floors.get(c), nb) for c in arm}


def strat_floors(rep, base):
    return {c: OR.floor_of(rep[c], base[c]) for c in base}


def kre_compare(recomputed, banked, keys=("delta", "lo", "hi", "separated", "floor", "gain", "cost", "n_windows", "n_episodes")):
    """exact equality of every field of every recomputed cell against the banked row; -> (n_compared, [mismatch strings]).  Zero compared rows is itself a failure."""
    bad, n = [], 0
    for c, row in recomputed.items():
        b = banked.get(c)
        if b is None:
            continue
        for k in keys:
            if k in b or k in row:
                n += 1
                if b.get(k) != row.get(k) and not (isinstance(b.get(k), float) and isinstance(row.get(k), float) and abs(b[k] - row[k]) <= 1e-12 * max(1.0, abs(b[k]))):
                    bad.append(f"{c}.{k}: recomputed {row.get(k)!r} != banked {b.get(k)!r}")
    return n, bad


def init_control(rec):
    """C-INIT (card 2.3), read beside: B_s0 and U_s0 start from the same step-0 parameters -> the lever is data alone; else Delta is data plus init"""
    if not rec:
        return {"status": "NOT-GIVEN", "reading": "no init record was passed"}
    r = (rec.get("record") or rec).get("B_s0_vs_U_s0_params_equal")
    return {"status": "OK", "params_equal": r, "reading": ("the same step-0 parameters: Delta is the data lever alone" if r else "the step-0 parameters differ: Delta is read as data plus init")}


# ------------------------------------------------------------------------------------------------------------------------------------------ loaders (need torch via obd_geom)
def load_dump_n(d, n):
    import glob
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == n, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def load_arm_cells(geom, dg, J, glat, glon, d, n):
    x = load_dump_n(d, n)
    for k in OR.SAME:
        assert np.array_equal(x[k], dg[k]), f"window or model-free array differs from the reference dump: {k} in {d}"
    return OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)


def kevalx_problem(t0route):
    """-> reason or None: out/t0route/KEVALX.json (written by ds_run.py route) says every stage-B `real` == `realx` (50c70ae8 vs 008f48b7) on every array and number"""
    p = f"{t0route}/KEVALX.json"
    if not os.path.isfile(p):
        return "KEVALX.json missing"
    k = json.load(open(p, encoding="utf-8"))
    if not k.get("pass") or set(k.get("items", {})) != {f"{a}.B {v}" for a in ("B_s0", "U_s0", "B_rep") for v in ("real", "donor")} or not all(v["identical"] for v in k["items"].values()):
        return "KEVALX.json is not a PASS over the real and donor passes of B_s0.B, U_s0.B, B_rep.B"
    return None


def route_arrays(t0, arm, stage, SS, G, ci, t):
    """the STRATEGIC cells of one stage-B arm from its three T0 passes: S1 / RC-A errors from the real pass, S2 = S_CKPT(real) - S_CKPT(donor) at 80 m (paired per window)."""
    zs = {}
    for v, f in (("real", "realx"), ("donor", "donorx"), ("zero", "zero")):      # all three from the 008f48b7 evaluator (the only one that records win_ci / win_t); K-EVALX proves real == realx and donor == donorx against the 50c70ae8 passes
        z = dict(np.load(f"{t0}/eval_DS-{arm}-{stage}_{f}.npz", allow_pickle=True))
        if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
            raise RuntimeError(f"{arm}.{stage} {v}: the evaluator's windows differ from the grid")
        zs[v] = z
    cells = {v: SS.route_cells(zs[v]["ckpt"], G) for v in zs}
    rc = SS.rc_cells(SS.rcA_errors(zs["real"]["ckpt"], G))
    out = {f"S1:{k}" if k == "pos_err_m@80" else f"RC-A:{k}": v for k, v in rc.items()}
    out["S2:skill@80"] = np.where(np.isfinite(cells["real"][:, 2]) & np.isfinite(cells["donor"][:, 2]), cells["real"][:, 2] - cells["donor"][:, 2], np.nan)
    out["S_CKPT_real@80"], out["S_CKPT_donor@80"], out["S_CKPT_zero@80"] = cells["real"][:, 2], cells["donor"][:, 2], cells["zero"][:, 2]
    return out


def load_strategic_helpers():
    """the unedited STRAT1 helpers (ground truth tables, S_CKPT, RC-A checkpoint errors), pointed at the read tree and the pinned inputs"""
    os.environ["STRAT1_TREE"], os.environ["STRAT1_INPUTS"] = L.TREE, L.INP
    if L.S1R not in sys.path:
        sys.path.append(L.S1R)
    import strat1_strategic as SS
    join, navtok, navtext, ci, t, eid = SS.load_inputs()
    return SS, SS.ground(navtok, navtext, join, ci, t), ci, t


# ------------------------------------------------------------------------------------------------------------------------------------------ K-RE
def kre_t1(CI, geom, S1_dumps, banked_json, nb):
    """K-RE (T1): recompute STRAT1's S1_s0 - C0_s0 on the 24 cells from the STRAT1 dumps through THIS pipeline (loader, floors from C0_rep - C0_s0, cell rule) and compare with RESULT_STRAT1.json exactly."""
    dp = lambda arm: f"{S1_dumps}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
    dg = load_dump_n(dp("C0_s0"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    c = {a: load_arm_cells(geom, dg, J, glat, glon, dp(a), L.EVAL_N_EPISODES) for a in ("C0_s0", "C0_rep", "S1_s0")}
    F = OR.floors_from(c["C0_rep"], c["C0_s0"])
    rows = slim(pair_cells(CI, c["S1_s0"], c["C0_s0"], dg["eid"], F, nb))
    banked = {cell: v["S1-C0_s0"] for fam in json.load(open(banked_json, encoding="utf-8"))["families"].values() if isinstance(fam, dict) and "cells" in fam for cell, v in fam["cells"].items() if "S1-C0_s0" in v}
    n, bad = kre_compare(rows, banked)
    return {"pass": bool(n > 0 and not bad and len(banked) == len(CELLS)), "rows_compared": n, "n_banked_cells": len(banked), "mismatches": bad[:12], "n_mismatches": len(bad)}


def kre_t0(eval_dir, banked_json, n_boot=2000):
    """K-RE (T0): the DSCALE instrument's banked read (M2b-df50 against M2b, seed 0) recomputed through ds_t0 from the banked eval files: r, the slot RMSEs and the MSE delta equal exactly."""
    import ds_t0 as T
    b = json.load(open(banked_json, encoding="utf-8"))["rows"]["s0"]["dt"]
    m = T.mech_pair(eval_dir, "M2b-df50", "M2b", 0, n_boot)
    n, bad = kre_compare({"dt": {"r": m["r"], "q": m["rmse_slot"]["A"], "b": m["rmse_slot"]["B"], "delta": m["mse_delta"]["point"], "separated": m["mse_delta"]["separated"]}},
                         {"dt": {"r": b["r"], "q": b["rmse_slot"]["Q"], "b": b["rmse_slot"]["B"], "delta": b["mse_delta"]["point"], "separated": b["mse_delta"]["separated"]}}, ("r", "q", "b", "delta", "separated"))
    return {"pass": bool(n == 5 and not bad), "rows_compared": n, "mismatches": bad}


# ------------------------------------------------------------------------------------------------------------------------------------------ main
def md_table(fam_rows):
    out = ["| cell | Delta | 95 % interval | sep | floor F | gain | cost | n windows |", "|---|---|---|---|---|---|---|---|"]
    for c, v in fam_rows.items():
        if v.get("status") != "OK":
            out.append(f"| {c} | n/a | | | | | | {v.get('n_windows', 0)} |")
        else:
            out.append(f"| {c} | {v['delta']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['separated']} | {v['floor'] if v['floor'] is None else round(v['floor'], 4)} | {v['gain']} | {v['cost']} | {v['n_windows']} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=L.T1)
    ap.add_argument("--t0mech", default=L.OUT + "/t0mech")
    ap.add_argument("--t0route", default=L.OUT + "/t0route")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-record", default="F:/TanitAD-scratch/datascale/gate/pod_out/gate/init.json")
    ap.add_argument("--box2-dumps", default="F:/TanitAD-scratch/otband_read/t1", help="the banked box-2 OBD-CTL dumps (same tree, same threads): read beside, never the control")
    ap.add_argument("--kre-t1", action="store_true")
    ap.add_argument("--kre-t0", action="store_true")
    ap.add_argument("--strat1-dumps", default=L.S1R + "/t1")
    ap.add_argument("--strat1-banked", default=L.S1R + "/out/RESULT_STRAT1.json")
    ap.add_argument("--kre-t0-dir", default=HERE + "/kre")
    ap.add_argument("--dscale-banked", default=HERE + "/ref/dscale_read.json")
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args(argv)
    t_start = time.time()
    CI = L.load_ci()
    dpath = lambda arm, st: f"{a.dumps}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    ready = {L.item(arm, st): os.path.isdir(dpath(arm, st)) for arm, st in L.READ_ITEMS}
    rep = {"tool": "ds_read.py", "card": "PREREG_V8F_DSCALE10K (frozen; sha256 48a74901...; MM rulings 2026-11-16 / 11-17)", "block": 1, "code_of_record": L.COMMIT,
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC": "T0 at the window's first tick (--ages fresh)",
                    "mechanism / GT-oracle / absmask": "T0 (world-model and tactical diagnostic, never driving performance)"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = F (one replicate pair per stage)",
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("Usage:")[0].strip(), "items_ready": ready, "controls": {"problems": []}, "not_yet": {k: "NOT-YET (block 2 / 3)" for k in L.NOT_YET_ARMS}}
    prob = rep["controls"]["problems"]
    miss = [k for k, v in ready.items() if not v]
    if miss:
        rep["outcome"], rep["reason"] = ["NOT-YET"], f"T1 dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", miss)
        return 1
    import obd_geom as G
    geom = G.build()
    dg = load_dump_n(dpath("B_s0", "A"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm, st in L.READ_ITEMS:
        cells[(arm, st)] = load_arm_cells(geom, dg, J, glat, glon, dpath(arm, st), L.EVAL_N_EPISODES)
        print(f"loaded {arm}.{st} {time.time() - t_start:.0f}s", flush=True)
    eid = dg["eid"]
    rep["controls"]["K1_self_pair"] = OR.identity_control(CI, eid)
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    F = {st: OR.floors_from(cells[("B_rep", st)], cells[("B_s0", st)]) for st in ("A", "B")}
    rep["floors"] = {"definition": "F_cell = |B_rep - B_s0| per cell, per stage (one replicate pair; B_rep is seed 2)", "stage_A": F["A"], "stage_B": F["B"]}
    pairs = {f"U-B_s0.{st}": pair_cells(CI, cells[("U_s0", st)], cells[("B_s0", st)], eid, F[st], a.n_boot) for st in ("A", "B")}
    pairs["R-U_s0.A"] = pair_cells(CI, cells[("R_s0", "A")], cells[("U_s0", "A")], eid, F["A"], a.n_boot)
    pairs["B_rep-B_s0.A"] = pair_cells(CI, cells[("B_rep", "A")], cells[("B_s0", "A")], eid, None, a.n_boot)
    pairs["B_rep-B_s0.B"] = pair_cells(CI, cells[("B_rep", "B")], cells[("B_s0", "B")], eid, None, a.n_boot)
    # cross-venue beside (card 2.3): abs(B_s0 - box-2 OBD-CTL_s0), venue plus nondeterminism; the banked box-2 dump is read in this environment
    xv = f"{a.box2_dumps}/dump_OBD-CTL_s0_m10_causal_heads"
    if os.path.isdir(xv):
        cx = load_arm_cells(geom, dg, J, glat, glon, xv, L.EVAL_N_EPISODES)
        rep["cross_venue_beside"] = {"definition": "abs(B_s0.B - box-2 OBD-CTL_s0) per cell: venue plus nondeterminism; beside only, never the control",
                                     "abs_diff": OR.floors_from(cells[("B_s0", "B")], cx), "B_s0.B-OBD-CTL_s0": slim(pair_cells(CI, cells[("B_s0", "B")], cx, eid, None, a.n_boot)),
                                     "U_s0.B-OBD-CTL_s0": slim(pair_cells(CI, cells[("U_s0", "B")], cx, eid, None, a.n_boot))}
    else:
        rep["cross_venue_beside"] = {"status": "NOT-YET", "reason": f"the banked box-2 dump {xv} is not present"}
    # STRATEGIC (T0, stage B)
    SS, Gr, ci, t = load_strategic_helpers()
    miss0 = [f"{arm}.B {v}" for arm in ("B_s0", "U_s0", "B_rep") for v in ("real", "donor", "zero", "realx", "donorx") if not os.path.exists(f"{a.t0route}/eval_DS-{arm}-B_{v}.npz")]
    strat, sfam = {"tier": rep["tier"]["STRATEGIC"], "headline": list(HEADLINE["STRATEGIC"]), "not_in_this_card": {
        "detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"},
        "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}},
        "note": "the C0-type v9 nav shares its source with RC-A, so S1 is echo-prone for every arm alike: Delta (U - B) is read, never the level"}, {"status": "NOT-YET"}
    if miss0:
        strat["status"] = "NOT-YET"
        strat["reason"] = f"T0 route passes missing: {miss0}"
    elif kevalx_problem(a.t0route):                       # K-EVALX: the zero passes ran on the 008f48b7 evaluator; they are admissible only if its `real` output equals the banked 50c70ae8 `real` output
        strat["status"] = "NOT-YET"
        strat["reason"] = f"K-EVALX: {kevalx_problem(a.t0route)} (the zero passes are not admissible)"
    else:
        R_ = {arm: route_arrays(a.t0route, arm, "B", SS, Gr, ci, t) for arm in ("B_s0", "U_s0", "B_rep")}
        sf = strat_floors(R_["B_rep"], R_["B_s0"])
        sp = strat_pair(CI, R_["U_s0"], R_["B_s0"], eid, sf, a.n_boot)
        strat.update({"status": "OK", "floors": sf, "U-B_s0.B": slim(sp), "n_scorable_80m": {arm: int(np.isfinite(R_[arm]["S_CKPT_real@80"]).sum()) for arm in R_},
                      "best_constant_80m_reference": 0.794})
        pairs["U-B_s0.B"].update({c: v for c, v in sp.items() if c in HEADLINE["STRATEGIC"]})
    # families
    fam = {}
    for st in ("A", "B"):
        rows = pairs[f"U-B_s0.{st}"]
        fam[st] = {f: family_status(rows, HEADLINE[f]) for f in FAMILIES if not (f == "STRATEGIC" and st == "A")}
        if st == "A":
            fam[st]["STRATEGIC"] = {"status": "NOT-YET", "gains": [], "costs": [], "missing": list(HEADLINE["STRATEGIC"]), "reason": "T0 route passes are made for the stage-B (deciding) checkpoints only"}
    # T0 mechanism, precondition, mechanism label
    import ds_t0 as T
    nm = lambda arm: T.name_of(arm, "A")  # noqa: E731
    mech = {"tier": rep["tier"]["mechanism / GT-oracle / absmask"]}
    try:
        u_b = T.mech_pair(a.t0mech, nm("U_s0"), nm("B_s0"), 0, a.n_boot)
        r_u = T.mech_pair(a.t0mech, nm("R_s0"), nm("U_s0"), 0, a.n_boot, ref_name=nm("U_s0"))
        fl = T.mech_floor(a.t0mech, nm("B_rep"), nm("B_s0"))
        idc = T.identity_control(a.t0mech, nm("B_s0"))
        mech.update({"U_vs_B_stage_A": u_b, "R_vs_U_stage_A": r_u, "stage_A_T_floor": fl, "identity_control": idc, "reach": T.decide_reach(r_u["r"], r_u["mse_delta"]),
                     "T_label": T.decide_t(u_b["r"], u_b["mse_delta"], fl["floor_mse"])})
        if not idc["pass"]:
            prob.append("K1-T0: an arm against itself did not read r exactly 0 [0, 0]")
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        mech["status"] = "NOT-YET"
        mech["reason"] = f"{type(e).__name__}: {e}"
    rep["mechanism_stage_A_T0"] = mech
    reach_ok = mech.get("reach", {}).get("label") == "REACHED"
    mech_given = "reach" in mech
    blocked = [k for k, v in {L.item(a_, s_): L.bank_problem(a_, s_) for a_, s_ in L.READ_ITEMS}.items() if v]
    rep["blocked_items"] = {k: L.bank_problem(*k.split(".")) for k in blocked}
    labels = outcome_label({0: fam["B"]}, reach_ok if mech_given else True, blocked or None)
    if not mech_given:
        labels = ["NOT-YET (the T0 mechanism evals are missing: the R_s0 precondition is unread)"]
    rep["outcome"] = labels
    rep["seed_0_read"] = {"deciding_stage": "B", "stage_B_families": fam["B"], "stage_A_families_beside": fam["A"], **seed0_reading(fam["B"], reach_ok if mech_given else True, blocked or None)}
    rep["next_step"] = [f"{x}: {NEXT[x]}" for x in labels if x in NEXT] or ["no label until seed 1 (block 2: B_s1, U_s1) is read; the seed-0 read, its floor and the reach control are above"]
    rep["init_control_beside"] = init_control(json.load(open(a.init_record, encoding="utf-8")) if os.path.exists(a.init_record) else None)
    rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"] if f != "STRATEGIC" else rep["tier"]["STRATEGIC"], "headline": list(HEADLINE.get(f, ())),
                           "cells_U-B_s0.B": {c: slim({c: v})[c] for c, v in pairs["U-B_s0.B"].items() if (FAMILY.get(c.split(":")[-1]) == f or (f == "STRATEGIC" and c.split(":")[0] in ("S1", "S2")))},
                           "cells_U-B_s0.A": {c: slim({c: v})[c] for c, v in pairs["U-B_s0.A"].items() if FAMILY.get(c.split(":")[-1]) == f}, "status_stage_B": fam["B"].get(f),
                           "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "estimator": "paired episode-cluster bootstrap"} for f in FAMILIES}
    rep["families"]["PRIMARY (ADE / FDE, beside, decide nothing)"] = {"cells_U-B_s0.B": {c: slim({c: v})[c] for c, v in pairs["U-B_s0.B"].items() if FAMILY.get(c.split(":")[-1]) == "PRIMARY"}}
    rep["families"]["STRATEGIC"]["block"] = strat
    rep["pairs"] = {p: slim(v) for p, v in pairs.items()}
    rep["not_built_beside_rows"] = {"DK-VIOL counts": "beside row (OBD o4), not part of the 24-cell set", "T-LAT-F1 / T-LON-F1, goal macro-F1 by provenance (VLM vs geometry bits), nav-echo split":
                                    "no instrument in the STRAT1 / OBD readers; the heads are in the --dump-heads dumps; named work item", "per-country / platform eval breakdown (card 8)": "named work item, not in the cell set"}
    if a.kre_t1:
        rep["controls"]["K-RE_T1"] = kre_t1(CI, geom, a.strat1_dumps, a.strat1_banked, L.NB)   # the banked rows are B 2,000: K-RE recomputes at B 2,000 whatever --n-boot says
        if not rep["controls"]["K-RE_T1"]["pass"]:
            prob.append("K-RE (T1): the recomputed STRAT1 S1-C0_s0 row differs from the banked one")
    if a.kre_t0:
        rep["controls"]["K-RE_T0"] = kre_t0(a.kre_t0_dir, a.dscale_banked, L.NB)
        if not rep["controls"]["K-RE_T0"]["pass"]:
            prob.append("K-RE (T0): the recomputed DSCALE D-T row differs from the banked one")
    rep["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = ["# DSCALE10K block 1 (B_s0, U_s0, B_rep, R_s0), MEASURED; 141 episodes / 2,399 windows; B 2,000 seed 0", "",
          f"Outcome: **{labels}**. Controls: problems {prob or 'none'}. Reach (R_s0, T0): {mech.get('reach', 'NOT-YET')}. T mechanism (T0, beside): {mech.get('T_label', 'NOT-YET')}.", "",
          "Seed 0 only: the label needs seed 1 (block 2). Seed-0 family read at the deciding stage B: " + json.dumps(rep["seed_0_read"]["stage_B_families"]), ""]
    for f in FAMILIES:
        md += [f"## {f} ({rep['families'][f]['tier']})", "", md_table(rep["families"][f]["cells_U-B_s0.B"]), ""]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", labels, "| problems:", len(prob), "|", rep["wall_s"], "s")
    return 1 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
