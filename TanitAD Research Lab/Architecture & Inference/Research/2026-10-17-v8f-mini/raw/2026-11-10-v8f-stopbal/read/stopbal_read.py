"""stopbal_read.py -- the PREREG_V8F_STOPBAL read (frozen card; code of record 5ecdd7e605109e7cda3cad58fcfcebcf6174a946, AMENDMENT_1_CODE.md).
Tier T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; the planner is deterministic), eval-141, per seed.
CPU, zero training, no GPU.  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000, seed 0 ("another draw of episodes" only; "another
training run" is the floor from BAL-NULL_rep).

Usage:  python stopbal_read.py --out <json> [--map <arms.json>] [--smoke] [--n-boot 2000]

Inputs per arm (default layout, see README_READ.md; --map overrides every path):
  dump       <DUMPS>/dump_<tag>_m10_causal_heads/ep*.npz   (141 episode dumps of v8f_t1_eval.py: g, cl, ol, fl, v0, eid, clip_index, ws)
  stopdec    <SDEC>/stopdec_<tag>.npz                      (stopdec_arm.py = theory/stopdec/stopdec_read.py on the arm: T's goal / lon probabilities, strata)
  stopdec_json   the STOPDEC.json written beside it (instrument controls of control 3)
  bank       C:/Users/Admin/t2v8f_pulls/stopbal_<tag>/     (selfcheck.json: controls 1 and 2)

READING CHOICES, committed in README_READ.md BEFORE any STOPBAL number exists (the card's rules are quoted, never changed):
  (r1) Stop-point windows = the STOPDEC set: strat_tlstop == STOP_POINT (level 1) AND has-label AND at least one present speed cell (117 at the banked dump).
       All windows = every eval window with a finite value in the cell (the four-family reader's set).  The P cell is tactical (knots 11-30) ADE on the
       stop-point windows; the NR cell is tactical ADE on all windows.
  (r2) A cell's delta is the unrounded point difference of the arm minus BAL-NULL of the same seed (the interval is the paired bootstrap's).  Its floor is
       |mean(BAL-NULL_rep) - mean(BAL-NULL_s0)| over the same finite windows of the same window set, read at seed 0 and applied to both seeds.
       Gain: separated AND the delta on the better side AND |delta| > floor.  Cost: separated AND on the worse side AND |delta| > floor.  A cell with no
       floor (a missing replicate) is never a gain or a cost.  Lower is better for every cell except the TAC accuracies (higher is better).
  (r3) M: hit rate = share of stop-point windows on which the lever's decision fires (BAL-G: p(STOP_POINT) >= 0.5; BAL-L: lon argmax == STOP, LON9 index 2);
       false-positive rate = share of the other windows (the has-and-present set minus the stop-point windows) on which it fires.  M = hit >= 0.30 AND fpr <= 0.05.
  (r4) The seven outcomes are the card's predicates, evaluated literally; where more than one holds, the headline follows this precedence (most specific first):
       BAL-REGRESSES > BAL-ADOPT > BAL-GAIN-OTHER-PATH > BAL-DECISION-ONLY > BAL-FP-COSTS > BAL-RECALL-SHORT; every predicate that holds is listed.
         REGRESSES: NR is a cost at either seed.            ADOPT: P, NR, M at both seeds.
         GAIN-OTHER-PATH: P and NR at both seeds, M fails at BOTH seeds.     DECISION-ONLY: M and NR at both seeds, P fails at BOTH seeds.
         FP-COSTS: hit >= 0.30 at both seeds and fpr > 0.05 at either.       RECALL-SHORT: hit < 0.30 at either seed.
       No predicate holds => SEEDS-DISAGREE (per-seed flags printed; a third training seed for the arm before any outcome is quoted).
  (r5) RECALL-SHORT branch (card trigger a / b): recall (hit rate) up = hit(arm) - hit(NULL) > the hit floor |hit(REP) - hit(NULL_s0)| at both seeds;
       unchanged = |difference| <= that floor at both seeds; AUROC within the floor = |AUROC(arm) - AUROC(NULL)| <= |AUROC(REP) - AUROC(NULL_s0)| at both
       seeds (AUROC of the lever's score over the has-and-present windows, stop-point windows positive).  (a) up and AUROC within the floor; (b) unchanged;
       else "neither (a) nor (b)": named, decided by the Master Mind.
  (r6) The read is INVALID if control 1, 2, 3 or 4 fails or cannot be read (fail closed); control 5 (suite) is the chain's and is reported as not checked here.
       --smoke skips the selfcheck controls 1 and 2 (the DAG-OACC selfcheck predates the levers) and labels the output a code check, never a result.
STRATEGIC: PAUSED (PI item 25), reported as paused, never as a pass, never omitted.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402

CARD_SHA_PREFIX = "see PREREG_V8F_STOPBAL.sha256"
CODE_OF_RECORD = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"   # AMENDMENT_2_CODE.md; read paths byte-identical to 5ecdd7e6 (READ_NOTE_COMMIT.md)
NB, SEED = 2000, 0
# ---- the card's constants ----
HIT_MIN, FPR_MAX = 0.30, 0.05
P_STOP_MIN = 0.5
STOP_GI = 6                 # GOAL22 index of STOP_POINT (stopcue_read.py)
LON_STOP_I = 2              # LON9 = HOLD CREEP STOP ...
TLSTOP_SP = 1               # strat_tlstop level index of STOP_POINT (speedsrc STRATA_VARS["tlstop"] = TL, STOP_POINT, NO_TL_STOP)
EXPECT = {"n_windows": 2399, "n_has": 2365, "n_stop_point": 117}
CARD_WEIGHTS = {"BAL-NULL": (1.0, 1.0), "BAL-G": (4.5616, 1.0), "BAL-L": (1.0, 4.6991)}   # (goal_pos_weight_stop, lon_class_weight_stop)
LEVERS = {"BAL-G": {"decision": "goal_stop", "score": "goal_score"}, "BAL-L": {"decision": "lon_stop", "score": "lon_score"}}
OP, TAC = "operative_0_2s", "tactical_2_6s"
CELL_P = ("sp", f"{TAC}:ade_m")
CELL_NR = ("all", f"{TAC}:ade_m")
HIGHER_BETTER = ("TAC-LAT", "TAC-LON")
ACCURACY_CELLS = HIGHER_BETTER
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL",
          "dk_hw_abs_err_m": "LONGITUDINAL", "dk_tg_abs_err_s": "LONGITUDINAL", "cross_mae_m": "LATERAL", "heading_mae_deg": "LATERAL",
          "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"}
OUTCOMES = ("BAL-REGRESSES", "BAL-ADOPT", "BAL-GAIN-OTHER-PATH", "BAL-DECISION-ONLY", "BAL-FP-COSTS", "BAL-RECALL-SHORT")   # precedence (r4)
TRIGGERS = {
    "BAL-ADOPT": "the flag joins the recipe of record. Both adopt: one stacked arm (both flags, seeds 0, 1) is the next card, read against the better single arm. "
                 "One adopts: it is the recipe lever; the other is closed with its measured row.",
    "BAL-GAIN-OTHER-PATH": "the third-seed repeat (--seed 10 / 11 per init), plus a read of which T cells moved (the CONSORC decomposition on the arm). Not adopted until the repeat reads P.",
    "BAL-DECISION-ONLY": "a read of O's use of T's stop signal on existing dumps (no training), then the stacked arm.",
    "BAL-FP-COSTS": "W = sqrt of the arm's W (goal 2.14 / lon 2.17), seeds 0, 1.",
    "BAL-RECALL-SHORT": "(a) recall up but short and AUROC within the floor: the second dose (full balance, 20.81 / 22.08). (b) recall unchanged: check the lever's read-back and "
                        "gradient for that run first, then the perception lever (LoRA arm L's read adds STOPCUE; a stop-line auxiliary) goes to the PI as a named blocker.",
    "BAL-REGRESSES": "the arm is not adopted; the next arm halves W (square root) and re-reads.",
    "SEEDS-DISAGREE": "repeated with a third training seed for the arm concerned before any outcome is quoted.",
}


# ------------------------------------------------------------------------------------------------------------------------------------------ estimator access
def load_ci():
    """taniteval/ci.py is pure numpy: loaded from its file, so the unit tests need no torch.  Tree: env STOPBAL_TREE, then the STOPBAL export, then the OTBAND export."""
    cands = [os.environ.get("STOPBAL_TREE"), "F:/TanitAD-scratch/stopbal_read/tree", "F:/TanitAD-scratch/otband_read/tree"]
    for t in cands:
        if t and os.path.exists(t + "/taniteval/taniteval/ci.py"):
            spec = importlib.util.spec_from_file_location("stopbal_ci", t + "/taniteval/taniteval/ci.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.SOURCE = t + "/taniteval/taniteval/ci.py"
            return m
    raise RuntimeError("taniteval/ci.py not found in any read tree (set STOPBAL_TREE)")


# ------------------------------------------------------------------------------------------------------------------------------------------ pure reading logic
def _fin(*xs):
    m = np.ones(len(xs[0]), bool)
    for x in xs:
        m &= np.isfinite(np.asarray(x, np.float64))
    return m


def cell_delta(CI, a, b, eid, mask, higher, floor, nb=NB):
    """One cell of arm a minus reference b over the windows in `mask`; gain / cost per rule (r2)."""
    a, b, eid, mask = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid), np.asarray(mask, bool)
    m = mask & _fin(a, b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0, "gain": False, "cost": False}
    aa, bb, ee = a[m], b[m], eid[m]
    pt = float(aa.mean() - bb.mean())
    r = CI.paired_episode_cluster_bootstrap(aa, bb, ee, n_boot=nb, seed=SEED)
    sep = bool(r["separated"])
    better = (pt > 0) if higher else (pt < 0)
    worse = (pt < 0) if higher else (pt > 0)
    if floor is None:
        above = False
    else:
        above = abs(pt) > floor    # FLOORCMP  (the mutation check drops this comparison)
    return {"status": "OK", "delta": pt, "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": sep, "floor": floor, "above_floor": bool(above),
            "gain": bool(sep and better and above), "cost": bool(sep and worse and above), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(ee)))}


def floor_delta(rep, null0, mask):
    """|mean(rep) - mean(null0)| over the finite windows of `mask` (rule r2); None when nothing is comparable."""
    rep, null0 = np.asarray(rep, np.float64), np.asarray(null0, np.float64)
    m = np.asarray(mask, bool) & _fin(rep, null0)
    return float(abs(rep[m].mean() - null0[m].mean())) if m.any() else None


def auroc(score, label):
    """Mann-Whitney AUROC, ties one half (stopcue_read.auroc); NaN when a class is empty."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def decision_rates(CI, dec, eid, sp, ok, nb=NB):
    """hit rate on the stop-point windows, false-positive rate on the other ok windows (rule r3), each with an episode-cluster interval."""
    dec, eid, sp, ok = np.asarray(dec, bool), np.asarray(eid), np.asarray(sp, bool), np.asarray(ok, bool)
    out = {}
    for name, sel in (("hit", sp & ok), ("fpr", ok & ~sp)):
        if not sel.any():
            out[name] = {"point": None, "n": 0}
            continue
        r = CI.episode_cluster_bootstrap(dec[sel].astype(np.float64), eid[sel], reduce="mean", n_boot=nb, seed=SEED)
        out[name] = {"point": float(dec[sel].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "n": int(sel.sum()), "n_pos": int(dec[sel].sum())}
    return out


def compute_floors(CI, rep, null0, eid, sp, ok):
    """Floors from BAL-NULL_rep minus BAL-NULL_s0: per (set, cell), plus hit / AUROC per decision."""
    sets = {"all": np.ones(len(eid), bool), "sp": np.asarray(sp, bool) & np.asarray(ok, bool)}
    cells = {}
    for sname, msk in sets.items():
        for cname in null0["cells"]:
            cells[(sname, cname)] = floor_delta(rep["cells"][cname], null0["cells"][cname], msk)
    dec = {}
    for lv, spec in LEVERS.items():
        dr, d0 = decision_rates(CI, rep[spec["decision"]], eid, sp, ok, nb=50), decision_rates(CI, null0[spec["decision"]], eid, sp, ok, nb=50)
        a_r, a_0 = auroc(rep[spec["score"]][ok], sp[ok]), auroc(null0[spec["score"]][ok], sp[ok])
        dec[lv] = {"hit": abs(dr["hit"]["point"] - d0["hit"]["point"]), "fpr": abs(dr["fpr"]["point"] - d0["fpr"]["point"]), "auroc": abs(a_r - a_0)}
    return {"cells": cells, "dec": dec}


def read_seed(CI, lever, null, arm, floors, eid, sp, ok, nb=NB):
    """One seed of one lever arm against BAL-NULL of the same seed.  null / arm: {"cells": {name: [N]}, "goal_stop", "lon_stop": bool [N], "goal_score", "lon_score": [N]}."""
    spec = LEVERS[lever]
    sets = {"all": np.ones(len(eid), bool), "sp": np.asarray(sp, bool) & np.asarray(ok, bool)}
    cells = {}
    for (sname, cname) in [(s, c) for s in ("sp", "all") for c in arm["cells"]]:
        fl = None if floors is None else floors["cells"].get((sname, cname))
        cells[f"{sname}|{cname}"] = cell_delta(CI, arm["cells"][cname], null["cells"][cname], eid, sets[sname], cname in HIGHER_BETTER, fl, nb)
    P, NR = cells[f"{CELL_P[0]}|{CELL_P[1]}"], cells[f"{CELL_NR[0]}|{CELL_NR[1]}"]
    rates = decision_rates(CI, arm[spec["decision"]], eid, sp, ok, nb)
    rates0 = decision_rates(CI, null[spec["decision"]], eid, sp, ok, nb)
    auc, auc0 = auroc(arm[spec["score"]][ok], sp[ok]), auroc(null[spec["score"]][ok], sp[ok])
    hit, fpr = rates["hit"]["point"], rates["fpr"]["point"]
    fd = None if floors is None else floors["dec"][lever]
    flags = {"P": bool(P.get("gain")), "NR_cost": bool(NR.get("cost")), "NR_ok": not bool(NR.get("cost")), "hit_ok": hit is not None and hit >= HIT_MIN,
             "fpr_ok": fpr is not None and fpr <= FPR_MAX}
    flags["M"] = flags["hit_ok"] and flags["fpr_ok"]
    costs = [k for k, v in cells.items() if v.get("cost")]
    gains = [k for k, v in cells.items() if v.get("gain")]
    return {"flags": flags, "P_cell": P, "NR_cell": NR, "M": {"hit": rates["hit"], "fpr": rates["fpr"], "hit_NULL": rates0["hit"], "fpr_NULL": rates0["fpr"],
            "auroc": auc, "auroc_NULL": auc0, "hit_floor": None if fd is None else fd["hit"], "fpr_floor": None if fd is None else fd["fpr"],
            "auroc_floor": None if fd is None else fd["auroc"], "decision": spec["decision"], "rule": f"hit >= {HIT_MIN} AND fpr <= {FPR_MAX}"},
            "cells": cells, "named_costs": costs, "named_gains": gains}


def lever_outcome(per_seed):
    """per_seed: list of two read_seed() results (seed 0, seed 1).  Rule r4, literally.  -> {label, holding, per_seed_flags, trigger, recall_short_branch}"""
    f = [s["flags"] for s in per_seed]
    allp = lambda k: all(x[k] for x in f)  # noqa: E731
    nonep = lambda k: not any(x[k] for x in f)  # noqa: E731
    hold = {
        "BAL-REGRESSES": any(x["NR_cost"] for x in f),
        "BAL-ADOPT": allp("P") and allp("NR_ok") and allp("M"),
        "BAL-GAIN-OTHER-PATH": allp("P") and allp("NR_ok") and nonep("M"),
        "BAL-DECISION-ONLY": allp("M") and allp("NR_ok") and nonep("P"),
        "BAL-FP-COSTS": allp("hit_ok") and any(not x["fpr_ok"] for x in f),
        "BAL-RECALL-SHORT": any(not x["hit_ok"] for x in f),
    }
    holding = [o for o in OUTCOMES if hold[o]]
    label = holding[0] if holding else "SEEDS-DISAGREE"
    out = {"label": label, "holding": holding, "per_seed_flags": f, "trigger": TRIGGERS[label], "recall_short_branch": None}
    if label == "BAL-RECALL-SHORT" or "BAL-RECALL-SHORT" in holding:
        up, same, auc_in = [], [], []
        for s in per_seed:
            m = s["M"]
            if m["hit_floor"] is None or m["auroc_floor"] is None or m["hit_NULL"]["point"] is None:
                up.append(False), same.append(False), auc_in.append(False)
                continue
            d = m["hit"]["point"] - m["hit_NULL"]["point"]
            up.append(d > m["hit_floor"])
            same.append(abs(d) <= m["hit_floor"])
            auc_in.append(abs(m["auroc"] - m["auroc_NULL"]) <= m["auroc_floor"])
        a, b = all(up) and all(auc_in), all(same)
        out["recall_short_branch"] = {"recall_up_both": all(up), "recall_unchanged_both": all(same), "auroc_within_floor_both": all(auc_in),
                                      "branch": "(a) second dose (full balance 20.81 / 22.08)" if a else ("(b) check read-back and gradient first, then the perception lever to the PI" if b else "neither (a) nor (b): named for the Master Mind")}
    return out


def both_levers_line(lv):
    """The card's rule for the two lever arms together (ADOPT branch)."""
    ad = [k for k, v in lv.items() if v.get("label") == "BAL-ADOPT"]
    if len(ad) == 2:
        return "both adopt: one stacked arm (both flags, seeds 0, 1) is the next card, read against the better single arm"
    if len(ad) == 1:
        other = [k for k in lv if k not in ad][0]
        return f"{ad[0]} adopts: it is the recipe lever; {other} is closed with its measured row"
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------ controls
def check_levers(tag_kind, sc):
    """Control 1 from one selfcheck.json dict: the lever read-back equals the card's value for the arm, G-DVB passes.  -> problem strings (empty = pass)."""
    bad = []
    want_g, want_l = CARD_WEIGHTS[tag_kind]
    lv = (sc or {}).get("levers") or {}
    for key, want in (("goal_pos_weight_stop", want_g), ("lon_class_weight_stop", want_l)):
        v = lv.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad.append(f"control 1: levers.{key} missing or not a number ({v!r})")
        elif v != want:
            bad.append(f"control 1: levers.{key} reads {v}, the card wants {want}" + (" (an arm whose selfcheck reads 1.0 for its own lever did not test it)" if v == 1.0 else ""))
    if (sc or {}).get("declared_vs_built") != "PASS":
        bad.append(f"control 1: G-DVB declared_vs_built is {(sc or {}).get('declared_vs_built')!r}, not 'PASS'")
    return bad


def check_selfcheck(tag_kind, sc):
    """Controls 1 and 2 from one selfcheck.json dict.  -> list of problem strings (empty = pass)."""
    bad = check_levers(tag_kind, sc)
    live = (sc or {}).get("live") or {}
    terms = live.get("terms") or {}
    for k in ("goal", "lon"):
        v = terms.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            bad.append(f"control 2: live.terms.{k} is not finite ({v!r})")
    dead = live.get("dead_params")
    if not isinstance(dead, list):
        bad.append("control 2: live.dead_params missing")
    else:
        hit = [d for d in dead if str(d).startswith(("h_goal", "h_lon")) or "h_goal" in str(d) or "h_lon" in str(d)]
        if hit:
            bad.append(f"control 2: dead T head parameters {hit}")
    g = (live.get("groups_grad") or {}).get("T")
    if not isinstance(g, (int, float)) or not g > 0:
        bad.append(f"control 2: live.groups_grad.T is not > 0 ({g!r})")
    return bad


def check_instrument(ctl):
    """Control 3, from the STOPDEC.json `controls[tag]` block."""
    bad = []
    for k in ("top1_vs_dump_ol_max_abs_m", "g_vs_dump_max_abs_m", "v0_vs_dump_max_abs_mps", "ident_vs_top1_max_abs_m", "own_all_vs_ident_max_abs_m"):
        if ctl.get(k) != 0.0:
            bad.append(f"control 3: {k} = {ctl.get(k)!r}, must read exactly 0.0")
    if ctl.get("goal_prob_in_0_1") is not True:
        bad.append("control 3: goal probabilities not in [0, 1]")
    dev = ctl.get("lon_prob_rows_sum_to_1_max_dev")
    if not isinstance(dev, (int, float)) or not dev < 1e-5:
        bad.append(f"control 3: lon probability rows do not sum to 1 ({dev!r})")
    if ctl.get("manifest_ok") is False:
        bad.append("control 3: checkpoint sha256 differs from its bank manifest")
    if ctl.get("n_windows") != EXPECT["n_windows"] or ctl.get("n_has") != EXPECT["n_has"]:
        bad.append(f"control 3: windows {ctl.get('n_windows')}/{ctl.get('n_has')}, banked dump {EXPECT['n_windows']}/{EXPECT['n_has']}")
    if (ctl.get("group_counts_has") or {}).get("STOP_POINT") != EXPECT["n_stop_point"]:
        bad.append(f"control 3: STOP_POINT windows {(ctl.get('group_counts_has') or {}).get('STOP_POINT')}, expected {EXPECT['n_stop_point']}")
    return bad


def planted_auroc_controls():
    """Control 3, instrument self-test: planted AUROC 1.0 / 0.5 / 0.0 and probabilities handled in [0, 1]."""
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    r = {"perfect": auroc(np.where(y, 0.9, 0.1), y), "constant": auroc(np.full(8, 0.3), y), "inverted": auroc(np.where(y, 0.1, 0.9), y)}
    r["pass"] = r["perfect"] == 1.0 and r["constant"] == 0.5 and r["inverted"] == 0.0
    return r


# ------------------------------------------------------------------------------------------------------------------------------------------ loading
def default_map(dumps, sdec, pull):
    m = {}
    for tag in ("BAL-NULL_s0", "BAL-NULL_s1", "BAL-NULL_rep", "BAL-G_s0", "BAL-G_s1", "BAL-L_s0", "BAL-L_s1"):
        m[tag] = {"dump": f"{dumps}/dump_{tag}_m10_causal_heads", "stopdec": f"{sdec}/stopdec_{tag}.npz", "stopdec_json": f"{sdec}/STOPDEC_{tag}.json",
                  "stopdec_tag": tag, "bank": f"{pull}/stopbal_{tag}"}
    return m


def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == 141, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def align_stopdec(z, dump):
    """stopdec npz rows -> dump row order, keyed by (clip_index, ws); both must hold the same window set."""
    kd = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(dump["clip_index"], dump["ws"]))}
    km = [(int(c), int(w)) for c, w in zip(z["clip_index"], z["ws"])]
    assert len(set(km)) == len(km) == len(kd) and set(km) == set(kd), "stopdec windows differ from the dump windows"
    perm = np.empty(len(km), np.int64)
    for i, k in enumerate(km):
        perm[kd[k]] = i
    out = {k: z[k][perm] for k in z}
    assert np.array_equal(out["clip_index"], dump["clip_index"]) and np.array_equal(out["ws"], dump["ws"]) and np.array_equal(out["eid"], dump["eid"])
    return out


def window_sets(sd):
    has = np.asarray(sd["has"], bool)
    ok = has & (np.isfinite(sd["err_mps"]).sum(1) > 0)
    sp = (np.asarray(sd["strat_tlstop"]) == TLSTOP_SP) & ok
    return ok, sp


def arm_from_files(spec, ref, geom, cache):
    """Load one arm: dump + stopdec + geometry -> the dict read_seed wants (+ the raw pieces the controls need)."""
    d = load_dump(spec["dump"])
    for k in ("eid", "clip_index", "ws", "g", "fl", "v0"):
        assert np.array_equal(d[k], ref[k]), f"window or model-free array differs from the banked dump: {k} in {spec['dump']}"
    sd = align_stopdec(dict(np.load(spec["stopdec"])), d)
    key = spec["dump"]
    if key not in cache:
        A = geom["arm_arrays"](d["cl"], d["g"], geom["J"])
        cells = {f"{b}:{m}": np.asarray(A[b][m], np.float64) for b in geom["bands"] for m in geom["metrics"]}
        cache[key] = (cells, A)
    cells, A = cache[key]
    cells = dict(cells)
    for nm, gl, k in (("TAC-LAT", geom["glat"], "_lat"), ("TAC-LON", geom["glon"], "_lon")):
        cells[nm] = (A[k] == gl).astype(np.float64)
    gp, lp = sd["goal_prob"], sd["lon_prob"]
    arm = {"cells": cells, "goal_score": gp[:, STOP_GI], "lon_score": lp[:, LON_STOP_I], "goal_stop": gp[:, STOP_GI] >= P_STOP_MIN, "lon_stop": lp.argmax(1) == LON_STOP_I}
    ctl = {"dec_goal_matches_npz": bool(np.array_equal(arm["goal_stop"], sd["dec_goal_stop"])), "dec_lon_matches_npz": bool(np.array_equal(arm["lon_stop"], sd["dec_lon_stop"])),
           "probs_in_0_1": bool(((gp >= 0) & (gp <= 1)).all() and ((lp >= 0) & (lp <= 1)).all())}
    return arm, sd, ctl


def build_geom(ref0):
    import stopbal_geom as G
    J = G.join(ref0)
    return {"arm_arrays": G.arm_arrays, "J": J, "bands": list(G.BANDS), "metrics": G.METRICS, "G": G}


def kind_of(tag):
    return tag.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", help="arms JSON: {arm_tag: {dump, stopdec, stopdec_json, stopdec_tag, bank}}; default layout otherwise")
    ap.add_argument("--smoke", action="store_true", help="code check on banked DAG-OACC dumps: skips controls 1 and 2; never a result")
    ap.add_argument("--n-boot", type=int, default=NB)
    ap.add_argument("--dumps", default="F:/TanitAD-scratch/stopbal_read/t1")
    ap.add_argument("--stopdec-dir", default=os.path.join(HERE, "out", "stopdec"))
    ap.add_argument("--pull", default="C:/Users/Admin/t2v8f_pulls")
    ap.add_argument("--ref", default="E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads", help="banked DAG-OACC dump: the window-identity reference per seed")
    a = ap.parse_args()
    t0 = time.time()
    CI = load_ci()
    arms = json.load(open(a.map, encoding="utf-8")) if a.map else default_map(a.dumps, a.stopdec_dir, a.pull)
    rep = {"tool": "stopbal_read.py", "card": "theory/stopbal/PREREG_V8F_STOPBAL.md (FROZEN)", "code_of_record": CODE_OF_RECORD,
           "tier": "T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner)", "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed {SEED}; ci.py loaded from {CI.SOURCE}. It answers 'another draw of EPISODES'; "
                        "'another training run' is the BAL-NULL_rep floor",
           "smoke": bool(a.smoke), "reading_choices": __doc__.split("READING CHOICES")[1].split("STRATEGIC: PAUSED")[0].strip(),
           "STRATEGIC": {"status": "PAUSED", "ruling": "PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, never as a pass, never omitted."},
           "controls": {"problems": [], "notes": {}}, "levers": {}}
    prob = rep["controls"]["problems"]
    rep["controls"]["notes"]["control_5_suite"] = "NOT CHECKED HERE: the chain's G-SUITE and the Master Mind's pre-launch suite step (card control 5)"
    pa = planted_auroc_controls()
    rep["controls"]["planted_auroc"] = pa
    if not pa["pass"]:
        prob.append("control 3: planted AUROC controls failed")
    # ---- control 4: floor present
    need = ("BAL-NULL_s0", "BAL-NULL_rep")
    missing4 = [t for t in need if t not in arms]
    rep["controls"]["control_4_floor_present"] = not missing4
    if missing4:
        prob.append(f"control 4: {missing4} not present: no gain or cost can be read")
    seeds_ready = {}
    for s in (0, 1):
        need_s = [f"BAL-NULL_s{s}"] + [f"{lv}_s{s}" for lv in LEVERS]
        seeds_ready[s] = all(t in arms for t in need_s)
    if missing4 or not any(seeds_ready.values()):
        rep["outcome"] = "INVALID" if missing4 else "NOT-YET"
        rep["reason"] = "control 4 (floor) not present" if missing4 else "no seed has BAL-NULL and both lever arms"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, rep["outcome"])
        return
    # ---- geometry + arms
    refs = {s: load_dump(a.ref.format(s=s)) for s in (0, 1) if seeds_ready[s]}
    ref_any = refs[min(refs)]
    geom = build_geom(ref_any)
    G = geom["G"]
    cache, data = {}, {}
    for tag, spec in arms.items():
        s = 0 if tag.endswith("_rep") else int(tag[-1])
        if s not in refs:
            refs[s] = load_dump(a.ref.format(s=s))
        if tag.endswith("_s1") and not seeds_ready[1]:
            continue
        ref = refs[s]
        geom["glat"], geom["glon"] = G.labels(ref["g"])
        geom["J"] = G.join(ref)
        arm, sd, ctl = arm_from_files(spec, ref, geom, cache)
        data[tag] = (arm, sd, ref)
        rep["controls"]["notes"][f"instrument_{tag}"] = ctl
        if not all(ctl.values()):
            prob.append(f"control 3: {tag} decision / probability arrays inconsistent {ctl}")
        # control 1/2: selfcheck.  control 3: STOPDEC.json controls block
        if not a.smoke:
            p = os.path.join(spec["bank"], "selfcheck.json")
            if not os.path.exists(p):
                prob.append(f"control 1/2: {p} missing")
            else:
                prob += [f"{tag}: {x}" for x in check_selfcheck(kind_of(tag), json.load(open(p, encoding="utf-8")))]
        sj = spec.get("stopdec_json")
        if sj and os.path.exists(sj):
            ctlj = json.load(open(sj, encoding="utf-8"))["controls"][spec.get("stopdec_tag", tag)]
            prob += [f"{tag}: {x}" for x in check_instrument(ctlj)]
        elif not a.smoke:
            prob.append(f"control 3: {tag}: STOPDEC json missing")
    # window sets must be identical across arms and equal the banked counts (control 3)
    sets = {t: window_sets(d[1]) for t, d in data.items()}
    ok0, sp0 = sets["BAL-NULL_s0"]
    rep["controls"]["window_counts"] = {"n_windows": int(len(ok0)), "n_has": int(np.asarray(data["BAL-NULL_s0"][1]["has"]).sum()), "n_ok": int(ok0.sum()), "n_stop_point": int(sp0.sum())}
    for t, (ok, sp) in sets.items():
        if not (np.array_equal(ok, ok0) and np.array_equal(sp, sp0)):
            prob.append(f"control 3: {t} stop-point / ok window set differs from BAL-NULL_s0")
    c = rep["controls"]["window_counts"]
    if (c["n_windows"], c["n_has"], c["n_stop_point"]) != (EXPECT["n_windows"], EXPECT["n_has"], EXPECT["n_stop_point"]):
        prob.append(f"control 3: window counts {c} differ from the banked dump {EXPECT}")
    eid = ref_any["eid"]
    floors = compute_floors(CI, data["BAL-NULL_rep"][0], data["BAL-NULL_s0"][0], eid, sp0, ok0)
    rep["floors"] = {"rule": "(r2): |BAL-NULL_rep - BAL-NULL_s0| per cell, read at seed 0, applied to both seeds", "dec": floors["dec"],
                     "cells": {f"{k[0]}|{k[1]}": v for k, v in floors["cells"].items()}}
    for lv in LEVERS:
        per = []
        rep["levers"][lv] = {"seeds": {}}
        for s in (0, 1):
            if not seeds_ready[s]:
                rep["levers"][lv]["seeds"][f"s{s}"] = {"status": "NOT-YET"}
                continue
            r = read_seed(CI, lv, data[f"BAL-NULL_s{s}"][0], data[f"{lv}_s{s}"][0], floors, eid, sp0, ok0, a.n_boot)
            rep["levers"][lv]["seeds"][f"s{s}"] = r
            per.append(r)
            print(f"{lv} s{s} {time.time() - t0:.0f}s", flush=True)
        if len(per) == 2:
            rep["levers"][lv].update(lever_outcome(per))
        else:
            rep["levers"][lv].update(label=f"NOT-YET (one seed read; decides nothing)", holding=[], trigger=None)
    rep["both_levers"] = both_levers_line({k: v for k, v in rep["levers"].items()})
    if prob:
        rep["outcome"] = "INVALID"
    elif a.smoke:
        rep["outcome"] = "SMOKE (code check, never a result): " + "; ".join(f"{k} {v['label']}" for k, v in rep["levers"].items())
    else:
        rep["outcome"] = "; ".join(f"{k} {v['label']}" for k, v in rep["levers"].items())
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "| outcome:", rep["outcome"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    for p in prob:
        print("  INVALID:", p)


if __name__ == "__main__":
    main()
