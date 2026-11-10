#!/usr/bin/env python3
"""read_rep.py -- the reader of PREREG_REP_P6 (frozen 2026-11-10, sha256 4f59ae7c...c4131), sections 2 to 6. CPU only.

    python read_rep.py read --run C0=DIR --run C0r=DIR --run C0r2=DIR --run S1E=DIR --run S1E1=DIR [--run S1Z=DIR --run S1Z1=DIR] --out OUT
           [--smoke-standin] [--control-verdict scorer1p6_read.json] [--no-ledger]

Run names are fixed by the card: recipe C0 = {C0, C0r, C0r2}, recipe S1E = {S1E, S1E1}, recipe S1Z = {S1Z, S1Z1}. DIR holds e3_logits.npz (the dev-box arm pass).

REUSE, NOT A FORK: every per-run number (gates G0-*, E3 PDMS with its paired log-cluster CI vs P6, G1 to G3, the family tables) comes from the Batch 1 reader read_scorer1p6.run_read, called
unchanged once per run (C0-recipe runs in its C0 slot; S1E-recipe runs in its S1E slot next to the real C0, so G3 is also read against C0 as Batch 1 does). S1Z-recipe runs go through
read_s1z.s1z_read (G1 to G4, M2 reported). The E3 score of a run's pick is the full truth table's pdms at that proposal (CHANGED_FROM_TABLE_CONTROL.md: 0 differences); the harness csv of the
arm dir is NOT used. What this file ADDS: s_p, df, t, R1, R2, F3, recipe means inside each bootstrap draw (same draws as every other statistic), the seed ensemble ENS_R, the REP-A/B/C
outcome map and the K ledger keys.
"""
from __future__ import annotations

import argparse
import hashlib
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
for _p in (P6BASE + "/eval", P6BASE + "/s1z/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import read_scorer1p6 as R  # noqa: E402  (the Batch 1 reader, unchanged)
import read_s1z as S  # noqa: E402  (the S1Z reader, unchanged)

SEL_DIR = P6BASE + "/selectors"
CARD = P6BASE + "/PREREG_REP_P6.md"
CARD_SHA256 = "4f59ae7cec6c812dea93991af349ded67aa04c6fb915b08acc8322c6805c4131"
DEFAULT_TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
TABLE_SHA256_PREFIX = "3bda2080"
DEFAULT_LEDGER = SEL_DIR + "/ledger_K.json"
SMOKE_ROOT = "F:/tanitad_scratch/repread/smoke"

# ------------------------------------------------------------------------------------------------ frozen constants (each with its card line)
RECIPE_OF = {"C0": "C0", "C0r": "C0", "C0r2": "C0", "S1E": "S1E", "S1E1": "S1E", "S1Z": "S1Z", "S1Z1": "S1Z"}   # card section 3 "Recipe sets"
CARD_SIZE = {"C0": 3, "S1E": 2, "S1Z": 2}
T_TABLE = {2: 4.303, 3: 3.182, 4: 2.776}      # card section 3: "df 2 -> 4.303; df 3 -> 3.182; df 4 -> 2.776"
F_BATCH1 = 0.3796                             # card section 5 floor clause "2 * 0.3796 = 0.759"
FLOOR_FACTOR = 2.0
GOAL = 94.55                                  # card section 4 "goal"
BARS = {"bootstrap_draws": R.B_DRAWS, "bootstrap_seed": R.SEED, "t_table": T_TABLE, "floor_F_batch1": F_BATCH1, "floor_factor": FLOOR_FACTOR, "goal": GOAL}

NEXT = {   # card section 5, each outcome's next lever
    "REP-A": ("the recipe's effect is real and the 2*F bars were not hiding it. Next: (i) its seed ensemble ENS_R, read under its own card with ENS_R - best single run CI-lo > 0 (0 GPU); "
              "(ii) the 1,600-update dose of that recipe, Batch 2, with a PI check before pod time"),
    "REP-B": ("a 400-update scorer fine-tune is not distinguishable from training noise on E3 for the recipes read. Next: the dose arm (1,600 updates) of the recipe with the largest mean, "
              "with a PI check before pod time. The cards' 2*F bars are re-stated with F3 next to F (reported; no verdict is re-decided)"),
    "REP-C": "that auxiliary loss is a lever beyond plain fine-tuning. Next: S1E and S1Z losses stacked in one arm, under its own card",
    "Unnamed": ("matches none of REP-A, REP-B, REP-C (for example a recipe is PROVEN on R1 but a run fails a guard, so it is not adoptable and REP-B's 'no recipe PROVEN' is false): "
                "labelled as such, the card names no lever"),
}
FLOOR_TEXT = ("F3 > 2 * 0.3796 = 0.759: the single-pair F understated training noise, and that is written next to every result that used 2*F")
OUTCOME_NAME = {"REP-A": "REAL GAIN", "REP-B": "NOISE", "REP-C": "LOSS BEATS PLAIN", "Unnamed": "UNNAMED"}
PRIMARY_ORDER = ("REP-A", "REP-C", "REP-B")

AMBIGUITIES = [
    "P1 The recipe mean inside each draw is the mean of the runs' per-token E3 PDMS vectors; its bootstrap is the bootstrap of that mean vector on the same draws as every run (identical to taking the mean of per-run draw statistics because every run shares the token set and cluster weights).",
    "P2 t: the card's printed values for df 2, 3, 4 are used as written; any other df is computed with scipy.stats.t.ppf(0.975, df) if scipy is present and otherwise REFUSED. A partial set of runs (for example the control's C0, C0r, S1E: df 1) is therefore read with scipy and stamped PARTIAL_NOT_THE_CARD_READ.",
    "P3 R2 uses sqrt(1/n_R + 1/n_C0) with the C0 recipe's actual run count; the card writes 1/3, which equals this for n_C0 = 3 (flagged when n_C0 is not 3).",
    "P4 G3 of an S1E-recipe run is read against P6 and against the real C0 run (as Batch 1 does for S1E); G3 of a C0-recipe run against P6; G3 of an S1Z-recipe run against P6 only (the S1Z card). G1 to G3 are Batch 1's, G4 and M2 are read_s1z's.",
    "P5 The floor clause threshold is 2 * 0.3796 = 0.7592 exactly (the card writes it as '= 0.759').",
    "P6 Outcome overlap: every match is listed; the primary label follows REP-A, REP-C, REP-B, Unnamed. REP-B is 'no recipe PROVEN on R1' among the recipes read; a recipe PROVEN but not adoptable matches none of A/B (REP-C may still hold).",
    "P7 Goal status per recipe: GOAL MET iff the recipe mean >= 94.55 (Batch 1 EPS slack in the arm's disfavour), else BELOW GOAL.",
    "P8 K ledger: one key per run E3 read (REP:<run>:<logits sha>:E3) and one per recipe-mean read (REP:<recipe>:mean:<sha of the run shas>:E3). ENS_R reads are reported and are NOT keyed (the card keys runs and recipe means only). A real read re-reads C0, C0r and S1E, which Batch 1 already read; the REP keys are separate from any Batch 1 key.",
    "P9 A duplicate arm (two run names with the same e3_logits sha256) is refused unless --smoke-standin: a real seed replicate is never bit-identical.",
]


class Refused(Exception):
    pass


# ------------------------------------------------------------------------------------------------ pure statistics (tested on hand-made numbers)
def t_value(df):
    """t(0.975, df): the card's printed value for df 2, 3, 4; scipy otherwise; refuse if neither"""
    if df in T_TABLE:
        return T_TABLE[df]
    if df < 1:
        raise Refused(f"df = {df}: s_p needs at least one recipe with two or more runs")
    try:
        from scipy import stats
    except Exception as ex:      # noqa: BLE001
        raise Refused(f"df = {df} is not in the card's t table and scipy is not available: {ex}")
    return float(stats.t.ppf(0.975, df))


def seed_stats(points):
    """points: {recipe: [run points]} -> dict(s_R per recipe with n>=2, s_p, df). s_p = sqrt(sum (n-1) s_R^2 / sum (n-1)), n-1 denominators."""
    s, num, df = {}, 0.0, 0
    for rc, v in points.items():
        n = len(v)
        if n < 2:
            continue
        sd = float(np.std(np.asarray(v, np.float64), ddof=1))
        s[rc] = sd
        num += (n - 1) * sd * sd
        df += n - 1
    if df < 1:
        raise Refused("df = 0: no recipe has two runs, so the pooled seed SD s_p is undefined; R1 and R2 are not computed")
    return {"s_R": s, "s_p": math.sqrt(num / df), "df": df}


def r1_threshold(t, s_p, n_R):
    return t * s_p / math.sqrt(n_R)


def r2_threshold(t, s_p, n_R, n_c0):
    return t * s_p * math.sqrt(1.0 / n_R + 1.0 / n_c0)


def bar_proven(lo, d, thr):
    """PROVEN iff CI-lo > 0 AND d > thr; both strict, no slack (card section 4)"""
    if lo is None or d is None:
        return False
    return bool(lo > 0 and d > thr)


def f3_of(points_c0):
    """largest pairwise |difference| among the C0 recipe's runs (None for fewer than 2)"""
    v = list(points_c0)
    if len(v) < 2:
        return None
    return float(max(abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]))


def floor_clause(f3):
    if f3 is None:
        return {"f3": None, "threshold": FLOOR_FACTOR * F_BATCH1, "triggered": None, "text": None}
    thr = FLOOR_FACTOR * F_BATCH1
    return {"f3": f3, "threshold": thr, "triggered": bool(f3 > thr), "text": FLOOR_TEXT if f3 > thr else None}


def decide_outcome(adoptable, r1, r2):
    """adoptable {recipe: bool}, r1 {recipe: bool}, r2 {recipe: bool} (S1E, S1Z only) -> dict(matches, primary, name, next_levers)"""
    m = []
    if any(adoptable.values()):
        m.append("REP-A")
    if any(r2.values()):
        m.append("REP-C")
    if r1 and not any(r1.values()):
        m.append("REP-B")
    if not m:
        m = ["Unnamed"]
    prim = next((o for o in PRIMARY_ORDER if o in m), "Unnamed")
    return {"matches": m, "primary": prim, "name": OUTCOME_NAME[prim], "next_levers": {o: NEXT[o] for o in m}}


def goal_label(mean):
    return "GOAL MET" if R.ge(mean, GOAL) else "BELOW GOAL"


# ------------------------------------------------------------------------------------------------ table lookup (the validated route)
class Table:
    def __init__(self, path):
        self.path = str(path)
        self.sha256 = R.sha256_file(path)
        with np.load(path, allow_pickle=False) as T:
            self.token = [str(t) for t in T["token"]]
            self.pdms, self.sub, self.valid = T["pdms"], T["sub"], T["valid"]
        self.pos = {t: i for i, t in enumerate(self.token)}
        if len(self.pos) != len(self.token):
            raise Refused("the truth table has duplicate tokens")

    def align(self, world_tokens):
        miss = [t for t in world_tokens if t not in self.pos]
        if miss:
            raise Refused(f"the truth table lacks {len(miss)} evaluation tokens")
        return np.array([self.pos[t] for t in world_tokens])


def changed_rows(table, tix, world, pick, p6_pick):
    """e3_changed_scores rows in memory: token -> (valid, score, six sub-scores) for every token whose pick differs from P6's (changed_from_table.build's content)"""
    rows = {}
    for i in np.where(pick != p6_pick)[0]:
        k, ti = int(pick[i]), int(tix[i])
        if not np.isfinite(table.pdms[ti, k]):
            raise Refused(f"token {world.token[i]} proposal {k}: no table value")
        rows[str(world.token[i])] = (bool(table.valid[ti, k]), float(table.pdms[ti, k]), np.asarray(table.sub[ti, k], np.float64))
    return rows


def run_vector(table, tix, pick, p6_pick, p6_score):
    """per-token E3 PDMS x100 of a run's pick: P6's banked csv score where the pick equals P6's, else the table's pdms at the pick (the Batch 1 pool recipe)"""
    n = len(pick)
    v = np.where(pick == p6_pick, p6_score, table.pdms[tix, pick])
    return v * 100.0


# ------------------------------------------------------------------------------------------------ inputs
def make_inputs(a):
    ns = argparse.Namespace(p6_proposals=a.p6_proposals, p6_csv=a.p6_csv, e2_tokens=a.e2_tokens, e2_table=a.e2_table, export=a.export, world_dump=a.world_dump,
                            arm=list(a.run), ens3_csv=None, c3_tokens=a.c3_tokens, no_tactical=a.no_tactical)
    return R.inputs_from_files(ns)


def slot_inputs(inp, mapping):
    """a Batch 1 Inputs whose arms are {slot: inp.arms[run]} (the same Arm objects, nothing else changed)"""
    arms = {slot: inp.arms[run] for slot, run in mapping.items()}
    return R.Inputs(inp.world, inp.p6_logits, inp.p6_proposals, inp.p6_rows, inp.e2_tokens, inp.e2_table, arms, inp.p6_pick_banked, inp.p6_fingerprint, inp.c3_strata,
                    inp.tactical, inp.provenance, inp.p6_token)


def _pub(d):
    return json.loads(json.dumps(d, default=R.jdefault))


# ------------------------------------------------------------------------------------------------ the read
def rep_read(inp, table, smoke=False, ledger_path=None, control_ref=None):
    """inp.arms: run name -> Arm. -> result dict, status READ / READ STOPPED / INCOMPLETE / REFUSED."""
    t0 = time.time()
    res = {"tool": "read_rep.py", "card": {"path": CARD, "sha256_expected": CARD_SHA256, "sha256_actual": R.sha256_file(CARD) if os.path.exists(CARD) else None},
           "tier": R.TIER, "evidence_class": R.EVIDENCE, "estimator": R.ESTIMATOR, "bars": BARS, "strategic": R.STRATEGIC, "ambiguities_rep": AMBIGUITIES,
           "ambiguities_batch1": R.AMBIGUITIES, "inputs": inp.provenance}
    if smoke:
        res["stamp"] = "SMOKE_NOT_A_RESULT"
    if res["card"]["sha256_actual"] != CARD_SHA256:
        res.update(status="READ STOPPED", why="the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    runs = [k for k in inp.arms if not k.startswith("_")]
    unknown = [k for k in runs if k not in RECIPE_OF]
    if unknown or not runs:
        res.update(status="REFUSED", why=f"run names must be among {sorted(RECIPE_OF)}; got {runs}")
        return res
    if any(RECIPE_OF[k] != "C0" for k in runs) and "C0" not in runs:
        res.update(status="REFUSED", why="run C0 is required next to any S1E or S1Z run (G3 against C0, and the S1Z reader's zero point)")
        return res
    if any(RECIPE_OF[k] == "S1Z" for k in runs) and "C0r" not in runs:
        res.update(status="REFUSED", why="run C0r is required next to an S1Z run (read_s1z needs the floor F = |C0r - C0|)")
        return res
    shas = {k: inp.provenance["arms"][k][1] for k in runs}
    dup = sorted({k for k in runs for j in runs if j != k and shas[k] == shas[j]})
    if dup and not smoke:
        res.update(status="REFUSED", why=f"runs {dup} have bit-identical e3_logits.npz (a real seed replicate is never identical); only --smoke-standin allows it")
        return res
    if not table.sha256.startswith(TABLE_SHA256_PREFIX):
        res.update(status="REFUSED", why=f"the truth table sha256 {table.sha256} does not start with {TABLE_SHA256_PREFIX}")
        return res
    res["truth_table"] = {"path": table.path, "sha256": table.sha256}
    res["code_sha256"] = {"read_rep.py": R.sha256_file(HERE / "read_rep.py"), "read_scorer1p6.py": R.sha256_file(P6BASE + "/eval/read_scorer1p6.py"), "read_s1z.py": R.sha256_file(P6BASE + "/s1z/read/read_s1z.py")}

    W, N = inp.world, len(inp.world.token)
    p6_pick = R.pick_logits(inp.p6_logits)
    try:
        tix = table.align(list(W.token))
        p6_score = np.array([inp.p6_rows[t][1] for t in W.token], np.float64)
    except (Refused, KeyError) as ex:
        res.update(status="READ STOPPED", why=f"P6 or truth-table alignment: {ex}")
        return res
    # ---- picks, in-memory changed-pick scores (table route), aligned logits
    L3, pick = {}, {}
    try:
        for nm in runs:
            a = inp.arms[nm]
            ix, err = R._align(a.e3_token, list(W.token), nm)
            if err:
                res.update(status="READ STOPPED", why=err)
                return res
            L3[nm] = np.asarray(a.e3_logits)[ix]
            pick[nm] = R.pick_logits(L3[nm])
            a.rows = changed_rows(table, tix, W, pick[nm], p6_pick)
    except Refused as ex:
        res.update(status="READ STOPPED", why=str(ex))
        return res

    # ---- one Batch 1 read per run
    B3 = R.BootSet(W.log)
    per, gates = {}, {}
    for nm in runs:
        rc = RECIPE_OF[nm]
        if rc == "C0":
            r1_ = R.run_read(slot_inputs(inp, {"C0": nm}), need_ens3=False)
            slot = "C0"
        elif rc == "S1E":
            r1_ = R.run_read(slot_inputs(inp, {"C0": "C0", "S1E": nm}), need_ens3=False)
            slot = "S1E"
        else:
            r1_ = S.s1z_read(slot_inputs(inp, {"C0": "C0", "C0r": "C0r", "S1Z": nm}), smoke=smoke, ledger_path=None)
            slot = "S1Z"
        for g, v in (r1_.get("gates") or {}).items():
            gates[f"{nm}:{g}"] = v
        if r1_.get("status") != "READ":
            res.update(status=r1_.get("status"), why=f"run {nm}: {r1_.get('why')}", gates=gates, missing_tokens=r1_.get("missing_tokens"))
            return res
        per[nm] = (r1_, slot)
    res["gates"] = gates

    # ---- vectors, levels, CIs vs P6 on the shared draws
    vec = {nm: run_vector(table, tix, pick[nm], p6_pick, p6_score) for nm in runs}
    vec_p6 = p6_score * 100.0
    runs_out, pts, ctl = {}, {}, {}
    for nm in runs:
        r1_, slot = per[nm]
        if slot == "S1Z":
            ep = r1_["batch1_endpoints_S1Z_slot"]
            lvl_b1, vs_b1 = ep["E3_pdms"], ep["E3_pdms_vs_P6"]
            ok_g = {k: bool(r1_["guards"][k]["ok"]) for k in ("G1", "G2", "G3", "G4")}
            guards_ok = all(ok_g.values())
            guard_detail = {"G1": r1_["guards"]["G1"], "G2": r1_["guards"]["G2"], "G3": r1_["guards"]["G3"], "G4": r1_["guards"]["G4"], "M2_reported": r1_["M2"]}
            fam = r1_["family_clause"]
            changed = r1_["E3_pick_changed_vs_P6"]
            goal_run = ep["goal_status"]
        else:
            e = r1_["endpoints"][slot]
            lvl_b1, vs_b1 = e["E3_pdms"], e["E3_pdms_vs_P6"]
            guard_detail = e["guards"]
            guards_ok = bool(e["guards"]["all_guards_ok"])
            ok_g = None
            fam = r1_["family_clause"]["per_arm_vs_P6"][slot]
            changed = e["E3_pick_changed_vs_P6"]
            goal_run = e["goal_status"]
        pt, r, n = B3.stat(vec[nm])
        lvl = R.rec(pt, r, n)
        pt, r, n = B3.diff(vec[nm], vec_p6)
        vs = R.rec(pt, r, n)
        chk = abs(lvl["point"] - lvl_b1["point"])
        if chk > 1e-9 or vs["lo"] != vs_b1["lo"] or vs["hi"] != vs_b1["hi"]:
            res.update(status="READ STOPPED", why=f"run {nm}: the table-vector level/CI disagrees with the Batch 1 reader's (point diff {chk}, CI {vs} vs {vs_b1})")
            return res
        pts[nm] = lvl["point"]
        if slot != "S1Z":
            ctl[nm] = _pub({**{f: e[f] for f in CONTROL_FIELDS}, "vec_E3_pdms": lvl, "vec_E3_pdms_vs_P6": vs})
        runs_out[nm] = {"recipe": RECIPE_OF[nm], "slot_in_batch1_reader": slot, "logits_sha256": shas[nm], "E3_pdms": lvl, "E3_pdms_vs_P6": vs, "goal_status_run": goal_run,
                        "E3_pick_changed_vs_P6": changed, "guards_pass": guards_ok, "guards": guard_detail, "guards_by_id": ok_g,
                        "four_families_vs_P6": {"STRATEGIC": R.STRATEGIC, "components": fam["components"] if "components" in fam else fam.get("components_vs_P6"),
                                               "adverse_components": fam["adverse_components"]},
                        "artifact": f"{inp.provenance['arms'][nm][0]} + truth table {table.sha256[:8]}", "tier": R.TIER["E3"],
                        "estimator": "paired log-cluster bootstrap, 10,000 draws, seed 20260927 (every run on the same draws)"}
    res["runs"] = runs_out

    # ---- recipes
    rec_runs = {}
    for nm in runs:
        rec_runs.setdefault(RECIPE_OF[nm], []).append(nm)
    points = {rc: [pts[n] for n in v] for rc, v in rec_runs.items()}
    partial = any(len(rec_runs.get(rc, [])) != CARD_SIZE[rc] for rc in ("C0", "S1E")) or any(rc == "S1Z" and len(v) != 2 for rc, v in rec_runs.items())
    res["card_complete"] = not partial
    if partial:
        res["stamp_partial"] = "PARTIAL_NOT_THE_CARD_READ"
    try:
        ss = seed_stats(points)
        t = t_value(ss["df"])
    except Refused as ex:
        res.update(status="REFUSED", why=str(ex), seed={"points": points})
        return res
    s_p, df = ss["s_p"], ss["df"]
    f3 = f3_of(points.get("C0", []))
    res["seed"] = {"s_R": ss["s_R"], "s_p": s_p, "df": df, "t_0.975": t, "points": points, "F3_largest_pairwise_C0": f3, "floor_clause": floor_clause(f3),
                   "definition": "s_p = sqrt(sum (n_R-1) s_R^2 / sum (n_R-1)), n-1 denominators, over recipes with n_R >= 2"}
    mean_vec = {rc: np.mean([vec[n] for n in v], axis=0) for rc, v in rec_runs.items()}
    rec_out, r1, r2, adoptable, r_draws = {}, {}, {}, {}, {}
    for rc, v in rec_runs.items():
        n_R = len(v)
        pt, r, n = B3.stat(mean_vec[rc])
        lvl = R.rec(pt, r, n)
        pt_d, r_d, n_d = B3.diff(mean_vec[rc], vec_p6)
        d1 = R.rec(pt_d, r_d, n_d)
        r_draws[rc] = r_d
        thr1 = r1_threshold(t, s_p, n_R)
        r1[rc] = bar_proven(d1["lo"], d1["point"], thr1)
        o = {"runs": v, "n_R": n_R, "mean_E3_pdms": lvl, "R1_d_vs_P6": d1, "R1_threshold_t_sp_over_sqrt_nR": thr1, "R1_ci_lo_gt_0": bool(d1["lo"] > 0), "R1_d_gt_threshold": bool(d1["point"] > thr1),
             "R1_PROVEN": r1[rc], "goal_status": goal_label(lvl["point"]), "all_runs_pass_guards": all(runs_out[n]["guards_pass"] for n in v),
             "failed_guard_runs": [n for n in v if not runs_out[n]["guards_pass"]]}
        if rc in ("S1E", "S1Z") and "C0" in mean_vec:
            n_c0 = len(rec_runs["C0"])
            pt_c, r_c, n_cc = B3.diff(mean_vec[rc], mean_vec["C0"])
            d2 = R.rec(pt_c, r_c, n_cc)
            thr2 = r2_threshold(t, s_p, n_R, n_c0)
            r2[rc] = bar_proven(d2["lo"], d2["point"], thr2)
            o.update(R2_d_vs_C0_recipe=d2, R2_threshold=thr2, R2_n_C0=n_c0, R2_card_literal_one_third_equals_1_over_nC0=bool(n_c0 == 3), R2_ci_lo_gt_0=bool(d2["lo"] > 0),
                     R2_d_gt_threshold=bool(d2["point"] > thr2), R2_PROVEN=r2[rc])
            r_draws[rc + "-C0"] = r_c
        adoptable[rc] = bool(r1[rc] and o["all_runs_pass_guards"])
        o["adoptable"] = adoptable[rc]
        rec_out[rc] = o
    res["recipes"] = rec_out

    # ---- seed ensembles (reported; read after the verdict inputs are fixed)
    ens = {}
    for rc, v in rec_runs.items():
        if len(v) < 2:
            ens[rc] = {"status": "NOT READ: one run only"}
            continue
        Le = np.mean([L3[n].astype(np.float32) for n in v], axis=0, dtype=np.float32)
        pe = R.pick_logits(Le)
        ve = run_vector(table, tix, pe, p6_pick, p6_score)
        pt, r, n = B3.stat(ve)
        pt_d, r_d, n_d = B3.diff(ve, vec_p6)
        d = R.rec(pt_d, r_d, n_d)
        best = max(v, key=lambda n_: pts[n_])
        pt_b, r_b, n_b = B3.diff(ve, vec[best])
        ens[rc] = {"status": "READ", "runs": v, "E3_pdms": R.rec(pt, r, n), "vs_P6": d, "vs_best_single_run": {"run": best, **R.rec(pt_b, r_b, n_b)},
                   "n_changed_picks_vs_P6": int((pe != p6_pick).sum()), "rule": "float32 logit mean over the recipe's runs, navsim_v1 pick (torch float32, first index on ties)",
                   "note": "reported, not deciding; the deployable form of an adoptable recipe is decided under its own card"}
    res["seed_ensembles"] = ens

    # ---- outcome
    out = decide_outcome(adoptable, r1, r2)
    out["inputs"] = {"R1": r1, "R2": r2, "adoptable": adoptable, "F3": f3, "floor_clause": res["seed"]["floor_clause"]}
    if partial:
        out["stamp"] = "PARTIAL_NOT_THE_CARD_READ: not all of the card's runs are present; the bars are shown for information"
    res["outcome"] = out

    # ---- K ledger and Bonferroni intervals (a successful read only)
    kinfo = {"ledger": None, "K": None, "note": "no ledger path given: K NOT COUNTED"}
    if ledger_path:
        if SEL_DIR not in sys.path:
            sys.path.insert(0, SEL_DIR)
        import selcore
        led = selcore.Ledger(ledger_path)
        pre = "SMOKE:" if smoke else ""
        added = []
        for nm in runs:
            k = f"{pre}REP:{nm}:{shas[nm]}:E3"
            if led.add(k, nominee=nm, family="REP", set="E3", smoke=bool(smoke)):
                added.append(k)
        for rc, v in rec_runs.items():
            k = f"{pre}REP:{rc}:mean:{hashlib.sha256('|'.join(shas[n] for n in sorted(v)).encode()).hexdigest()}:E3"
            if led.add(k, nominee=rc + " mean", family="REP", set="E3", smoke=bool(smoke)):
                added.append(k)
        K = led.K
        bon = {}
        for rc in rec_runs:
            lo, hi = selcore.interval(r_draws[rc][np.isfinite(r_draws[rc])], K)
            bon[rc] = {"point": rec_out[rc]["R1_d_vs_P6"]["point"], "lo": lo, "hi": hi}
        kinfo = {"ledger": ledger_path, "K": K, "keys_added": added, "bonferroni_recipe_mean_minus_P6_at_K": bon,
                 "note": "K counts every navtest read in the programme's ledger; best-of-K statements use these intervals; R1 and R2 are this card's own pre-registered tests (card section 6)"}
    res["K_ledger"] = kinfo
    if control_ref:
        res["control"] = compare_control(control_ref, runs_out, ctl)
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "READ"
    return res


# ------------------------------------------------------------------------------------------------ control against the Batch 1 verdict
CONTROL_FIELDS = ("E3_pdms", "E3_pdms_vs_P6", "E3_pick_changed_vs_P6", "goal_status", "E3_subscores_vs_P6", "G2_delta_regret_vs_P6", "G3_overspeed_diff_vs_P6",
                  "slow_pick_share_cruise", "guards")


def _flat(o, path=""):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _flat(o[k], f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, o


def compare_control(verdict_path, runs_out, ctl):
    """each of C0, C0r, S1E present in the read: its Batch 1 endpoint fields and its table-vector level and CI vs P6 must equal scorer1p6_read.json leaf by leaf"""
    ref = json.load(open(verdict_path, encoding="utf-8"))["endpoints"]
    n, bad, runs = 0, [], []
    for nm in ("C0", "C0r", "S1E"):
        if nm not in runs_out:
            continue
        runs.append(nm)
        mine = ctl[nm]
        for f in list(CONTROL_FIELDS) + ["vec_E3_pdms", "vec_E3_pdms_vs_P6"]:
            refv = ref[nm]["E3_pdms"] if f == "vec_E3_pdms" else ref[nm]["E3_pdms_vs_P6"] if f == "vec_E3_pdms_vs_P6" else ref[nm][f]
            a, b = dict(_flat(mine[f], f)), dict(_flat(refv, f))
            for k in sorted(set(a) | set(b)):
                n += 1
                if a.get(k, "MISSING") != b.get(k, "MISSING"):
                    bad.append((nm + k, a.get(k, "MISSING"), b.get(k, "MISSING")))
    return {"verdict_path": verdict_path, "runs_compared": runs, "n_leaves_compared": n, "n_mismatch": len(bad), "mismatches": bad[:12], "EQUAL": bool(n > 0 and not bad)}


# ------------------------------------------------------------------------------------------------ markdown
def to_markdown(res):
    sm = res.get("stamp") == "SMOKE_NOT_A_RESULT"
    L = []
    if sm:
        L += ["**SMOKE_NOT_A_RESULT** -- stand-in runs (copies of existing arms) fill the replicate slots; no number here is a result and none may be quoted as one.", ""]
    if res.get("stamp_partial"):
        L += [f"**{res['stamp_partial']}**", ""]
    L += ["# PREREG_REP_P6 read", "", f"**Status: {res.get('status')}**", "", f"- Card `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)",
          f"- Tier: E3 = {R.TIER['E3']}", f"- Evidence class: {R.EVIDENCE}", f"- Estimator: {R.ESTIMATOR}", ""]
    if res.get("status") != "READ":
        L += [f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    ci = R.ci_s
    o, sd = res["outcome"], res["seed"]
    L += ["## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})", f"- Next lever: {o['next_levers'][o['primary']]}"]
    if sd["floor_clause"]["triggered"]:
        L.append(f"- Floor clause: **{FLOOR_TEXT}** (F3 = {sd['F3_largest_pairwise_C0']:.4f})")
    L += [f"- F3 = {R.fmt(sd['F3_largest_pairwise_C0'])}; s_p = {sd['s_p']:.4f}; df = {sd['df']}; t = {sd['t_0.975']}", f"- K ledger: K = {res['K_ledger'].get('K')} ({res['K_ledger'].get('note')})", "",
          "## Recipes (mean of runs vs P6, R1; S1E and S1Z also R2 vs the C0 recipe)", "", "| recipe | runs | mean E3 PDMS | d vs P6 [95 %] | R1 thr | R1 | R2 d vs C0 [95 %] | R2 thr | R2 | guards | adoptable | goal |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rc, v in res["recipes"].items():
        r2 = (f"{ci(v['R2_d_vs_C0_recipe'])} | {v['R2_threshold']:.4f} | **{'PROVEN' if v['R2_PROVEN'] else 'NOT PROVEN'}**" if "R2_PROVEN" in v else "n/a | n/a | n/a")
        L.append(f"| {rc} | {', '.join(v['runs'])} | {R.lvl_s(v['mean_E3_pdms'])} | {ci(v['R1_d_vs_P6'])} | {v['R1_threshold_t_sp_over_sqrt_nR']:.4f} | **{'PROVEN' if v['R1_PROVEN'] else 'NOT PROVEN'}** | {r2} | "
                 f"{'all pass' if v['all_runs_pass_guards'] else 'FAIL ' + ','.join(v['failed_guard_runs'])} | {v['adoptable']} | {v['goal_status']} |")
    L += ["", "## Runs (E3, paired CI vs P6)", "", "| run | recipe | PDMS x100 | vs P6 [95 %] | guards | changed picks |", "|---|---|---|---|---|---|"]
    for nm, v in res["runs"].items():
        L.append(f"| {nm} | {v['recipe']} | {R.lvl_s(v['E3_pdms'])} | {ci(v['E3_pdms_vs_P6'])} | {'pass' if v['guards_pass'] else 'FAIL'} | {v['E3_pick_changed_vs_P6']} |")
    L += ["", "## Seed ensembles ENS_R (reported, not deciding)", ""]
    for rc, v in res["seed_ensembles"].items():
        L.append(f"- {rc}: {v['status']}" + (f"; {R.lvl_s(v['E3_pdms'])}; vs P6 {ci(v['vs_P6'])}; vs best single run {v['vs_best_single_run']['run']} {ci(v['vs_best_single_run'])}" if v["status"] == "READ" else ""))
    L += ["", "## Families", "", f"- STRATEGIC: {R.STRATEGIC}", "- longitudinal / lateral / tactical components per run are in the json under runs.<run>.four_families_vs_P6; adverse components:"]
    for nm, v in res["runs"].items():
        L.append(f"  - {nm}: {v['four_families_vs_P6']['adverse_components'] or 'none'}")
    if "control" in res:
        c = res["control"]
        L += ["", "## Control (against the Batch 1 verdict)", "", f"- runs {c['runs_compared']}: leaves compared {c['n_leaves_compared']}, mismatches {c['n_mismatch']}: **{'EQUAL' if c['EQUAL'] else 'NOT EQUAL'}**"]
    L += ["", "## Ambiguities (this reader)", ""] + [f"- {a}" for a in res["ambiguities_rep"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, "rep_read.json"), os.path.join(out_dir, "rep_read.md")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, default=R.jdefault)
    open(mp, "w", encoding="utf-8", newline="\n").write(to_markdown(res))
    return jp, mp


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read"])
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR, NAME in C0 C0r C0r2 S1E S1E1 S1Z S1Z1; DIR holds e3_logits.npz")
    ap.add_argument("--table", default=DEFAULT_TABLE)
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
    ap.add_argument("--smoke-standin", action="store_true", help="some runs are copies of other arms' dirs; outputs only under " + SMOKE_ROOT + ", stamped SMOKE_NOT_A_RESULT")
    ap.add_argument("--control-verdict", default=None, help="Batch 1 scorer1p6_read.json: C0, C0r, S1E per-run numbers must equal its endpoints exactly")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = os.path.abspath(a.out).replace("\\", "/")
    if a.smoke_standin and not out.startswith(SMOKE_ROOT):
        raise SystemExit(f"REFUSING: --smoke-standin writes only under {SMOKE_ROOT}")
    if not a.smoke_standin and out.startswith(SMOKE_ROOT):
        raise SystemExit("REFUSING: a real read must not write under the smoke folder")
    if a.smoke_standin and not a.no_ledger:
        raise SystemExit("REFUSING: a smoke read runs with --no-ledger")
    S.wait_free_ram(a.min_free_ram_gb)
    inp = make_inputs(a)
    table = Table(a.table)
    ledger = None if a.no_ledger else a.ledger
    res = rep_read(inp, table, smoke=a.smoke_standin, ledger_path=ledger, control_ref=a.control_verdict)
    jp, mp = write_outputs(res, a.out)
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "outcome": res.get("outcome", {}).get("primary"), "K": res.get("K_ledger", {}).get("K"),
                      "control": res.get("control"), "stamp": res.get("stamp"), "json": jp, "md": mp}, indent=1, default=R.jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
