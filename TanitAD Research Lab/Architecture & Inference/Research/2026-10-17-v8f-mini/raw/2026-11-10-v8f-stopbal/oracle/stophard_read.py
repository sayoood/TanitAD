"""stophard_read.py -- PREREAD_V8F_STOPHARD (theory/stopbal/oracle/PREREAD_V8F_STOPHARD.md, frozen; sha256 in PREREAD_V8F_STOPHARD.sha256).  CPU, zero training.

T's OWN stop decision written into the code as a hard value (no GT, no new input).  Per arm, per window of the STOPDEC first-tick batches:
  OWN    = forward_infer plan (T's own code)
  HARD   = OWN code with (a) goal[STOP_POINT] := 1.0 where own p(STOP_POINT) >= 0.5 and (b) lon part := one-hot STOP where own lon argmax == STOP
  HGOAL  = (a) only;  HLON = (b) only;  UNREACH = the IDENT transform with an unreachable threshold (goal 2.0, lon never): code equals OWN bit for bit.
The model pass is stoporc_read.run_arm, imported and NOT edited: its two module hooks (MODES, force_codes) are pointed at this file's transform before it is called.
Stage `run` banks one npz per arm to scratch; stage `analyse` reads them, runs the controls and writes RESULT_STOPHARD.json.
Usage:  python stophard_read.py --stage run [--tags ...] [--smoke N]
        python stophard_read.py --stage analyse --out RESULT_STOPHARD.json
The pure logic (hard_code, decisions, check_hard, outcome_lever) needs no torch and no model: the unit tests call it directly.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stoporc_read as SO  # noqa: E402  (imported, not copied; its READ dir is put on sys.path on import)

THEORY = SO.THEORY
READ = SO.READ
STOPDEC_OUT = SO.STOPDEC_OUT
SCRATCH = os.environ.get("STHARD_SCRATCH", "F:/TanitAD-scratch/stophard_build")
CARD = os.path.join(HERE, "PREREAD_V8F_STOPHARD.md")
CARD_SHA = "bc9ac9110e5ff43a49c4e8cee215285bea6cb665284e23734acb14e1c703fbe9"

N_G, N_LAT, N_LON, N_CONS, CODE_DIM = SO.N_G, SO.N_LAT, SO.N_LON, SO.N_CONS, SO.CODE_DIM
STOP_GI, LON_STOP_I, LON_SL = SO.STOP_GI, SO.LON_STOP_I, SO.LON_SL
P_STOP_MIN = 0.5                                           # STOPBAL M rule for BAL-G (stopbal_read.py P_STOP_MIN)
MODES = ("unreach", "hard", "hgoal", "hlon")
VARIANTS = ("hard", "hgoal", "hlon")
#            goal threshold, lon rule enabled, write goal, write lon
SPEC = {"unreach": (2.0, False, True, True), "hard": (P_STOP_MIN, True, True, True), "hgoal": (P_STOP_MIN, True, True, False), "hlon": (P_STOP_MIN, True, False, True)}
LEVER_VARIANT = {"BAL-G": "hgoal", "BAL-L": "hlon"}
LEVER_DECISION = {"BAL-G": "goal", "BAL-L": "lon"}
SEEDS = ("s0", "s1")
DECIDE_TAGS = {"BAL-G": ("BAL-G_s0", "BAL-G_s1"), "BAL-L": ("BAL-L_s0", "BAL-L_s1")}
NULL_TAGS = ("BAL-NULL_s0", "BAL-NULL_s1")
FLOOR_ARM = "BAL-NULL_rep"
ALL_TAGS = (*NULL_TAGS, FLOOR_ARM, "BAL-G_s0", "BAL-G_s1", "BAL-L_s0", "BAL-L_s1")
CELLS = SO.CELLS
S_CELL = SO.PRIMARY_CELL                                   # tactical ADE (knots 11-30)
TACTICAL_FAMILY = {"status": "n = 0", "reason": "T's decision is the transformed input of this read"}
STRATEGIC_FAMILY = SO.STRATEGIC_FAMILY


def sha256(p):
    return SO.sha256(p)


# ------------------------------------------------------------------------------------------------------------------ the transform (numpy, float32)
def decisions(own, goal_thr=P_STOP_MIN):
    """T's own decisions from its code: (goal [N] bool: p(STOP_POINT) >= thr ; lon [N] bool: lon argmax == STOP)."""
    own = np.asarray(own)
    return own[:, STOP_GI] >= goal_thr, own[:, LON_SL].argmax(1) == LON_STOP_I


def hard_code(own, mode):
    """own [N, 60] float32 -> a copy with T's own decision written as a hard value.  Uses no label and no other input."""
    assert mode in SPEC, mode
    thr, lon_on, w_goal, w_lon = SPEC[mode]
    own = np.asarray(own)
    c = own.copy()
    g_fire = own[:, STOP_GI] >= thr
    l_fire = (own[:, LON_SL].argmax(1) == LON_STOP_I) & lon_on
    if w_goal:
        c[g_fire, STOP_GI] = 1.0
    if w_lon:
        oh = np.zeros((int(l_fire.sum()), N_LON), own.dtype)
        oh[:, LON_STOP_I] = 1.0
        blk = c[:, LON_SL]
        blk[l_fire] = oh
        c[:, LON_SL] = blk
    return c


def _force_hook(own, lon_al, mode):
    """the hook stoporc_read.run_arm calls (it passes the GT allowed set; this transform ignores it by construction)."""
    return hard_code(own, mode)


def check_hard(own, code, mode, sel=None):
    """Control 2 on [sel] windows.  Independent of hard_code's implementation: the structural properties are stated column by column.  -> (counts, problems)."""
    own, code = np.asarray(own), np.asarray(code)
    if sel is not None:
        own, code = own[sel], code[sel]
    prob = []
    if own.shape != code.shape or own.dtype != code.dtype:
        return {}, [f"{mode}: shape/dtype {own.shape}/{own.dtype} vs {code.shape}/{code.dtype}"]
    thr, lon_on, w_goal, w_lon = SPEC[mode]
    gf, lf = own[:, STOP_GI] >= thr, (own[:, LON_SL].argmax(1) == LON_STOP_I) & lon_on
    gf_eff, lf_eff = gf & w_goal, lf & w_lon
    differ = own.view(np.uint32) != code.view(np.uint32)
    named = np.zeros(CODE_DIM, bool)
    if w_goal:
        named[STOP_GI] = True
    if w_lon:
        named[LON_SL] = True
    n_out = int(differ[:, ~named].sum())
    if n_out:
        prob.append(f"{mode}: {n_out} entries differ outside the named entries")
    if differ[~gf_eff, STOP_GI].any():
        prob.append(f"{mode}: goal entry changed on a window where the goal decision does not fire")
    if differ[:, LON_SL][~lf_eff].any():
        prob.append(f"{mode}: lon part changed on a window where the lon decision does not fire")
    if gf_eff.any() and not (code[gf_eff, STOP_GI] == 1.0).all():
        prob.append(f"{mode}: goal entry is not 1.0 where the goal decision fires")
    if lf_eff.any():
        want = np.zeros((int(lf_eff.sum()), N_LON), own.dtype)
        want[:, LON_STOP_I] = 1.0
        if not np.array_equal(code[lf_eff][:, LON_SL], want):
            prob.append(f"{mode}: lon part is not one-hot STOP where the lon decision fires")
    rows_changed = differ.any(1)
    if rows_changed[~(gf_eff | lf_eff)].any():
        prob.append(f"{mode}: a window with no firing decision changed")
    cnt = {"n_windows": int(len(own)), "goal_fires": int(gf.sum()), "lon_fires": int(lf.sum()), "both_fire": int((gf & lf).sum()), "neither_fires": int((~gf & ~lf).sum()),
           "goal_entry_changed_bits": int(differ[:, STOP_GI].sum()), "lon_part_changed_windows": int(differ[:, LON_SL].any(1).sum()),
           "windows_changed": int(rows_changed.sum()), "entries_outside_named_changed": n_out}
    return cnt, prob


# ------------------------------------------------------------------------------------------------------------------ reading logic (numpy)
def is_cost(d, floor):
    """separated, worse side (higher error), above the floor.  No floor => never a cost."""
    if floor is None or not np.isfinite(floor) or "point" not in d:
        return False
    return bool(d["separated"] and d["point"] > 0 and abs(d["point"]) > floor)


def outcome_lever(per_seed):
    """per_seed = {'s0': {'S_gain': bool, 'A_cost': bool}, 's1': {...}} -> label of the card's table (frozen)."""
    if any(s not in per_seed or per_seed[s].get("S_gain") is None or per_seed[s].get("A_cost") is None for s in SEEDS):
        return "NOT-EVALUATED"
    sg = [bool(per_seed[s]["S_gain"]) for s in SEEDS]
    ac = [bool(per_seed[s]["A_cost"]) for s in SEEDS]
    if all(sg) and not any(ac):
        return "HARD-HELPS"
    if all(sg) and any(ac):
        return "HARD-TRADES"
    if not any(sg):
        return "HARD-NULL"
    return "SEEDS-DISAGREE"


def window_sets_hard(z):
    ok, sp, ctl, S = SO.window_sets(z)
    return ok, sp, ctl, S


def arm_cells(z, BL):
    cells = {}
    for v in ("own", "ident") + MODES:
        plan = z[f"plan_{v}"].astype(np.float64)
        cells[v] = {}
        for bn, (a, b) in SO.BANDS.items():
            r = BL.band_rows(plan, z["g"], a, b)
            for m in SO.METRICS:
                cells[v][f"{bn}:{m}"] = r[m]
    return cells


def stopdec_counts(sd, ok, sp):
    """decision counts of the STOPDEC npz of the arm: (goal, lon) x (stop-point set, control set = ok minus stop-point)."""
    ctl = ok & ~sp
    return {"goal": {"stop_point": int(np.asarray(sd["dec_goal_stop"], bool)[sp].sum()), "control": int(np.asarray(sd["dec_goal_stop"], bool)[ctl].sum())},
            "lon": {"stop_point": int(np.asarray(sd["dec_lon_stop"], bool)[sp].sum()), "control": int(np.asarray(sd["dec_lon_stop"], bool)[ctl].sum())}}


def load_stopbal_counts():
    """RESULT_STOPBAL.json: (hit.n_pos, fpr.n_pos) per lever arm and per NULL arm of that lever's decision.  -> {tag: {decision: (hit, fpr)}}"""
    r = json.load(open(os.path.join(READ, "RESULT_STOPBAL.json"), encoding="utf-8"))
    out = {}
    for lever, lv in r["levers"].items():
        dec = LEVER_DECISION[lever]
        for s, sd in lv["seeds"].items():
            M = sd["M"]
            out.setdefault(f"{lever}_{s}", {})[dec] = (M["hit"]["n_pos"], M["fpr"]["n_pos"])
            out.setdefault(f"BAL-NULL_{s}", {})[dec] = (M["hit_NULL"]["n_pos"], M["fpr_NULL"]["n_pos"])
    return out


def analyse_arm(tag, z, CI, BL, sb_counts):
    ok, sp, ctl, S = window_sets_hard(z)
    sd = dict(np.load(os.path.join(STOPDEC_OUT, f"stopdec_{tag}.npz")))
    own = z["own_code"]
    g_dec, l_dec = decisions(own)
    cx = {"windows_equal_stopdec": bool(np.array_equal(sd["clip_index"], z["ci"]) and np.array_equal(sd["ws"], z["ws"]) and np.array_equal(sd["has"], z["has"])),
          "stop_point_set_equal_stopdec": bool(np.array_equal(sp, (sd["strat_tlstop"] == SO.TLSTOP_SP) & sd["has"] & (np.isfinite(sd["err_mps"]).sum(1) > 0))),
          "own_goal_prob_max_abs_vs_stopdec": float(np.abs(own[:, :N_G].astype(np.float64) - sd["goal_prob"]).max()),
          "own_lon_prob_max_abs_vs_stopdec": float(np.abs(own[:, LON_SL].astype(np.float64) - sd["lon_prob"]).max()),
          "goal_decision_per_window_equals_stopdec": bool(np.array_equal(g_dec, np.asarray(sd["dec_goal_stop"], bool))),
          "lon_decision_per_window_equals_stopdec": bool(np.array_equal(l_dec, np.asarray(sd["dec_lon_stop"], bool)))}
    mine = {"goal": {"stop_point": int(g_dec[sp].sum()), "control": int(g_dec[ctl].sum())}, "lon": {"stop_point": int(l_dec[sp].sum()), "control": int(l_dec[ctl].sum())},
            "both_stop_point": int((g_dec & l_dec)[sp].sum()), "both_control": int((g_dec & l_dec)[ctl].sum()), "either_stop_point": int((g_dec | l_dec)[sp].sum()),
            "n_stop_point": int(sp.sum()), "n_control": int(ctl.sum())}
    sdc = stopdec_counts(sd, ok, sp)
    ctrl = {"n_windows": int(len(z["ci"])), "n_stop_point": int(sp.sum()), "n_control_set": int(ctl.sum()), "n_all_has_present": int(ok.sum()), "n_stop_point_episodes": int(len(np.unique(z["eid"][sp]))),
            "ckpt_sha256": str(z["ckpt_sha256"]), "manifest_ok": bool(z["manifest_ok"]), "tanitad_file": str(z["tanitad_file"]),
            "own_plan_vs_dump_ol_max_abs_m": float(z["ol_vs_own_max_abs_m"]), "stopdec_cross_check": cx,
            "control3_counts": {"mine": mine, "stopdec_npz": sdc, "stopbal_result": {}}, "forced_code_checks": {}, "problems": []}
    P = ctrl["problems"]
    own_plan = z["plan_own"].astype(np.float64)
    # control 1: IDENT (unreachable threshold) equals OWN exactly, plan and code, on every cell
    unreach_code_same = bool(np.array_equal(z["code_unreach"].view(np.uint32), own.view(np.uint32)))
    unreach_plan_max = float(np.abs(z["plan_unreach"].astype(np.float64) - own_plan).max())
    ident_plan_max = float(np.abs(z["plan_ident"].astype(np.float64) - own_plan).max())
    cells = arm_cells(z, BL)
    unreach_cells_zero = bool(all(np.array_equal(np.nan_to_num(cells["unreach"][c], nan=-7.0), np.nan_to_num(cells["own"][c], nan=-7.0)) for c in CELLS))
    ctrl["control1_ident_unreachable"] = {"code_bit_equal_own": unreach_code_same, "plan_vs_own_max_abs_m": unreach_plan_max, "forward_diag_own_code_vs_own_plan_max_abs_m": ident_plan_max, "all_cells_equal_own": unreach_cells_zero}
    if not (unreach_code_same and unreach_plan_max == 0.0 and ident_plan_max == 0.0 and unreach_cells_zero):
        P.append(f"control 1: IDENT with an unreachable threshold differs from OWN (code {unreach_code_same}, plan {unreach_plan_max}, ident {ident_plan_max}, cells {unreach_cells_zero})")
    if ctrl["own_plan_vs_dump_ol_max_abs_m"] == ctrl["own_plan_vs_dump_ol_max_abs_m"] and ctrl["own_plan_vs_dump_ol_max_abs_m"] > 1e-4:
        P.append(f"own plan vs dump ol {ctrl['own_plan_vs_dump_ol_max_abs_m']}")
    if (ctrl["n_stop_point"], ctrl["n_control_set"]) != (SO.EXPECT["n_stop"], SO.EXPECT["n_ctl"]):
        P.append(f"window counts {ctrl['n_stop_point']} / {ctrl['n_control_set']} != {SO.EXPECT['n_stop']} / {SO.EXPECT['n_ctl']}")
    if not (cx["windows_equal_stopdec"] and cx["stop_point_set_equal_stopdec"]):
        P.append("windows / strata differ from the STOPDEC npz")
    if cx["own_goal_prob_max_abs_vs_stopdec"] > 1e-5 or cx["own_lon_prob_max_abs_vs_stopdec"] > 1e-5:
        P.append("own code differs from the STOPDEC npz probabilities")
    # control 2: HARD differs from OWN only in the named entries, only where the decision fires
    for m in MODES:
        for sn, msk in (("stop", sp), ("ctl", ctl)):
            cnt, pr = check_hard(own, z[f"code_{m}"], m, msk)
            ctrl["forced_code_checks"][f"{m}:{sn}"] = cnt
            P.extend(f"control 2 [{sn}]: {p}" for p in pr)
        # plan: a window whose code equals OWN bit for bit has exactly the IDENT plan
        same = (z[f"code_{m}"].view(np.uint32) == own.view(np.uint32)).all(1)
        pm = float(np.abs(z[f"plan_{m}"].astype(np.float64)[same] - own_plan[same]).max()) if same.any() else 0.0
        ctrl["forced_code_checks"][f"{m}:unchanged_windows_plan_max_abs_m"] = {"n_unchanged_windows": int(same.sum()), "max_abs_m": pm}
        if pm != 0.0:
            P.append(f"control 2: {m}: a window with an unchanged code has a plan that differs from OWN by {pm}")
    # control 3: decision counts equal STOPDEC's and STOPBAL M's
    for d in ("goal", "lon"):
        for sn in ("stop_point", "control"):
            if mine[d][sn] != sdc[d][sn]:
                P.append(f"control 3: {d} decision count on {sn} {mine[d][sn]} != STOPDEC npz {sdc[d][sn]}")
    if not (cx["goal_decision_per_window_equals_stopdec"] and cx["lon_decision_per_window_equals_stopdec"]):
        P.append("control 3: per-window decisions differ from the STOPDEC npz")
    for d, (h, f) in (sb_counts.get(tag) or {}).items():
        ctrl["control3_counts"]["stopbal_result"][d] = {"hit_n_pos": h, "fpr_n_pos": f}
        if (mine[d]["stop_point"], mine[d]["control"]) != (h, f):
            P.append(f"control 3: {d} decision counts ({mine[d]['stop_point']}, {mine[d]['control']}) != RESULT_STOPBAL.json M hit/fpr n_pos ({h}, {f})")
    ctrl["pass"] = not P
    # the cells (HARD minus OWN) on S (stop-point windows) and A (all has-and-present windows)
    eid = z["eid"]
    sets = {"S_stop_point": sp, "A_all": ok}
    delta = {}
    for v in VARIANTS:
        delta[v] = {}
        for sn, msk in sets.items():
            idx = np.where(msk)[0]
            delta[v][sn] = {c: SO.paired(CI, cells[v][c][idx], cells["own"][c][idx], np.asarray(eid)[idx]) for c in CELLS}
    own_tac = cells["own"][S_CELL]
    return {"controls": ctrl, "decision_counts": mine, "delta_variant_minus_own": delta,
            "own_tactical_ade_mean": {"S_stop_point": float(np.nanmean(own_tac[sp])), "A_all": float(np.nanmean(own_tac[ok]))}}


def floors_from(arms):
    """per variant, set and cell: |delta(NULL_rep) - delta(NULL_s0)| (point), read at seed 0 and applied to both seeds."""
    out = {}
    if FLOOR_ARM not in arms or NULL_TAGS[0] not in arms:
        return out
    for v in VARIANTS:
        out[v] = {}
        for sn in ("S_stop_point", "A_all"):
            out[v][sn] = {}
            for c in CELLS:
                a, b = arms[FLOOR_ARM]["delta_variant_minus_own"][v][sn][c], arms[NULL_TAGS[0]]["delta_variant_minus_own"][v][sn][c]
                out[v][sn][c] = float(abs(a["point"] - b["point"])) if "point" in a and "point" in b else None
    return out


def flags_for(arm, floors):
    """gain / cost per variant, set, cell for one arm (rule of the card: separated, side, above the floor)."""
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for sn in ("S_stop_point", "A_all"):
            out[v][sn] = {}
            for c in CELLS:
                d = arm["delta_variant_minus_own"][v][sn][c]
                fl = floors.get(v, {}).get(sn, {}).get(c)
                out[v][sn][c] = {"gain": is_gain_cell(d, fl), "cost": is_cost(d, fl), "floor_m": fl}
    return out


def is_gain_cell(d, floor):
    return SO.is_gain(d, floor)


def analyse_stage(out_json, tags):
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    BL = SS.BL
    CI = BL.CI
    sb = load_stopbal_counts()
    arms, banked = {}, {}
    for tag in tags:
        p = os.path.join(SCRATCH, f"stophard_{tag}.npz")
        if not os.path.exists(p):
            banked[tag] = "per-arm npz absent"
            continue
        z = dict(np.load(p, allow_pickle=False))
        arms[tag] = analyse_arm(tag, z, CI, BL, sb)
        banked[tag] = {"scratch_npz_sha256": sha256(p)}
        print(f"analysed {tag}: controls pass {arms[tag]['controls']['pass']}", flush=True)
    floors = floors_from(arms)
    ctl_ok = all(a["controls"]["pass"] for a in arms.values()) and len(arms) == len(tags)
    sp_counts = {a["controls"]["n_stop_point"] for a in arms.values()}
    cross = {"stop_point_counts_equal_across_arms": len(sp_counts) == 1, "counts": sorted(sp_counts)}
    ctl_ok = ctl_ok and cross["stop_point_counts_equal_across_arms"]
    flags = {t: flags_for(a, floors) for t, a in arms.items()}
    outcomes = {}
    for lever, v in LEVER_VARIANT.items():
        per_seed = {}
        for s, tag in zip(SEEDS, DECIDE_TAGS[lever]):
            if tag in flags:
                per_seed[s] = {"S_gain": flags[tag][v]["S_stop_point"][S_CELL]["gain"], "A_cost": flags[tag][v]["A_all"][S_CELL]["cost"], "arm": tag}
        o = outcome_lever(per_seed)
        outcomes[lever] = {"variant_as_decision": v, "outcome_if_controls_held": o, "outcome": "INVALID-CONTROLS" if not ctl_ok else o, "per_seed": per_seed}
    res = {"tool": "stophard_read.py", "card": "theory/stopbal/oracle/PREREAD_V8F_STOPHARD.md", "card_sha256_expected": CARD_SHA,
           "tier": "T1-kin FIRST TICK, zero training, inference-time transform of T's own output; never driving performance",
           "estimator": f"taniteval.ci paired episode-cluster bootstrap, B={SO.NB}, seed {SO.SEED}; answers 'another draw of episodes' only (not another training run, not another inference run: the planner is deterministic)",
           "ci_source": getattr(CI, "SOURCE", getattr(CI, "__file__", "?")), "tree": SO.TREE, "evidence_class": "MEASURED (this file; per-arm rows were banked to scratch npz, sha256 in `banked`)",
           "definitions": {"decision_goal": f"own p(STOP_POINT) (goal index {STOP_GI}) >= {P_STOP_MIN}", "decision_lon": f"own lon argmax == STOP (LON9 index {LON_STOP_I}; code[27:34])",
                           "HARD": "goal[STOP_POINT] := 1.0 where decision_goal, lon part := one-hot STOP where decision_lon", "HGOAL": "goal entry only", "HLON": "lon part only",
                           "S": f"{S_CELL} on the stop-point windows (STOPDEC set)", "A": f"{S_CELL} on all has-and-present windows",
                           "delta": "variant minus OWN (negative = lower error)", "floor": "per variant, set and cell: |delta(NULL_rep) - delta(NULL_s0)| (point), read at seed 0, applied to both seeds",
                           "lever_to_variant": LEVER_VARIANT},
           "families": {"PRIMARY/ADE-FDE, LONGITUDINAL, LATERAL": "in delta_variant_minus_own per cell and set", "TACTICAL": TACTICAL_FAMILY, "STRATEGIC": STRATEGIC_FAMILY},
           "banked": banked, "cross_arm": cross, "controls_ok": ctl_ok, "outcome": outcomes, "floors_m": floors, "flags": flags, "arms": arms}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({"outcome": outcomes, "controls_ok": ctl_ok}, default=float))


# ------------------------------------------------------------------------------------------------------------------ the model pass (stoporc_read.run_arm, unedited)
def run_stage(tags, smoke):
    os.makedirs(SCRATCH, exist_ok=True)
    SO.MODES = MODES                    # run_arm builds and banks code_<m> / plan_<m> for exactly these modes
    SO.force_codes = _force_hook
    for tag in tags:
        print(f"arm {tag}", flush=True)
        o = SO.run_arm(tag, smoke)
        np.savez(os.path.join(SCRATCH, f"{'smoke_' if smoke else ''}stophard_{tag}.npz"), **{k: np.asarray(v) for k, v in o.items()})
        print(f"banked {tag}", flush=True)
        if smoke:                       # a code check on N windows, never a result
            pr = [p for m in MODES for p in check_hard(o["own_code"], o[f"code_{m}"], m)[1]]
            print("SMOKE-CONTROLS", json.dumps({"unreach_plan_vs_own_max_abs_m": float(np.abs(o["plan_unreach"].astype(np.float64) - o["plan_own"]).max()),
                                                "hard_plan_vs_own_max_abs_m": float(np.abs(o["plan_hard"].astype(np.float64) - o["plan_own"]).max()), "code_problems": pr}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "analyse"), required=True)
    ap.add_argument("--tags", nargs="+", default=list(ALL_TAGS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_STOPHARD.json"))
    a = ap.parse_args()
    assert sha256(CARD) == CARD_SHA, "PREREAD_V8F_STOPHARD.md hash differs from the frozen sha256"
    if a.stage == "run":
        run_stage(a.tags, a.smoke)
    else:
        analyse_stage(a.out, a.tags)


if __name__ == "__main__":
    sys.path.insert(0, READ)
    main()
