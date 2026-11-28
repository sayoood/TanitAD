r"""STOPPERC stage-1 READ (theory/stopperc/PREREG_V8F_STOPPERC.md section 5 and section 8 / Q3, FROZEN; AMENDMENT_3_A_READINGS.md).  Arm A (stop-location auxiliary) against NULL, per seed.
Tier T1-kin (self-action open loop, m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141.  CPU, zero training.  Estimator: the paired
episode-cluster bootstrap (taniteval.ci), B 2,000, seed 0: it answers "another draw of EPISODES" only; "another training run" is the floor.  The first dose's package
(theory/stopbal/read/, NOT edited, imported by path) supplies the dump driver, the geometry, the instrument controls and the cell estimator; this file adds the stage-1 rules.

  python -I stopperc_s1_read.py refs   [--state D] [--pull D]            the NULL-reuse state and every bank this read needs, re-asserted (rc 0 / 2)
  python -I stopperc_s1_read.py dumps  [TAG ...] [--state D]             T1-kin dumps (CPU) of the arms whose banks are stage-1 runs (default: all of them); refuses without a clean reuse state
  python -I stopperc_s1_read.py read   --out F --verdict F [--state D] [--smoke --map arms.json] [--n-boot 2000]
  python -I stopperc_s1_read.py tags   [--state D]                       the stage-1 arm tags to dump / STOPDEC (space-separated; refuses without a clean reuse state)
  python -I stopperc_s1_read.py identity DUMP_DIR_A DUMP_DIR_B           bit-for-bit compare of two dump folders (the OFF-checkpoint tree-move proof)
The STOPDEC step of the stage-1 arms is job_stopdec_s1.cmd (stopdec_arm.py of the first dose, on the 3e264bbf tree).  README_READ_S1.md has the order.

THE RULES (copied from the card, never reinterpreted; every reading choice a rule needed is numbered s1..s8 in README_READ_S1.md and printed in the result):
  delta = arm minus the same-seed NULL.  A cell's floor = max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|)  (card section 8, Q3).  Gain = separated on the better side AND above the floor;
  cost = separated on the worse side AND above the floor.  Both seeds must agree, else SEEDS-DISAGREE.
  P: tactical ADE (knots 11-30) on the 117 stop-point windows is a gain at both seeds.   G-AUROC: STOPCUE AUROC of p(STOP_POINT) is a gain at both seeds; the lon-STOP AUROC is not a cost.
  G-FP: the false-positive rate at 0.5 on the non-stop windows is not a cost.            NR: all-window tactical ADE is not a cost at either seed.
  STRATEGIC: PAUSED (PI item 25), always printed, never a pass, never omitted."""
import glob
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.environ.get("STOPPERC_THEORY") or os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # .../motion_c/theory (env: the mutation runner reads a copy from elsewhere)
FIRST_READ = os.path.join(THEORY, "stopbal", "read")
DOSE2 = os.path.join(THEORY, "stopbal", "dose2")
sys.path.insert(0, FIRST_READ)
sys.path.insert(0, DOSE2)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import stopbal_read as SR  # noqa: E402   (numpy only; the first dose's cell estimator, controls and loaders)
import null_banks as NB  # noqa: E402    (the first dose's bank re-assertion, reused for the BAL-NULL_s1 / BAL-NULL_rep banks)

COMMIT_S1 = "3e264bbf3fa2f6f17bea448de09aa972faaf7725"                    # STOPPERC arm A code of record (branch agent/v8f-stopaux-20261112, child of 715ef8fc)
BAL_COMMIT = SR.CODE_OF_RECORD                                             # 715ef8fc: the commit the reused BAL-NULL banks name
BAL_NULL_S0_PIN = "5169ea9554bf74004f801b2d594358355363ba47fe07aa8cac052d1da9aa2fdf"   # BAL-NULL_s0's checkpoint file sha256 (AMENDMENT_3 section 4)
TREE = os.environ.get("STOPPERC_TREE", "F:/TanitAD-scratch/stopperc_read/tree")
T1_NEW = os.environ.get("STOPPERC_DUMPS", "F:/TanitAD-scratch/stopperc_read/t1")
T1_BAL = "F:/TanitAD-scratch/stopbal_read/t1"                              # the first dose's dumps (BAL-NULL_s1 / _rep, reused when NULL_REUSE.json reads EQUAL)
SDEC_NEW = os.path.join(HERE, "out", "stopdec")
SDEC_BAL = os.path.join(FIRST_READ, "out", "stopdec")
PULL = "C:/Users/Admin/t2v8f_pulls"
STATE = "C:/Users/Admin/stopperc_state"
REF = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads"

ARMS_ALWAYS = ("NULL_s0", "NULL_rep2", "A_s0", "A_s1")                    # the chain's unconditional runs (their banks are stopperc_<tag>)
ARMS_COND = ("NULL_s1", "NULL_rep")                                         # stopperc_<tag> only when the reuse rule reads DIFFERS; else BAL-NULL_s1 / BAL-NULL_rep
LABELS = ("PERC-REGRESSES", "PERC-ADOPT", "PERC-P-OTHER-PATH", "PERC-FP-COSTS", "PERC-AUROC-ONLY", "PERC-NULL")      # headline precedence (s4)
CEILING_M = 1.5                                                             # "the fraction of the 1.5 m ceiling recovered" (card section 5, P)
CEILING_STOPORC = {"s0": 1.5392291009873544, "s1": 1.476844845746562}       # STOPORC's F-GT tactical-ADE points (card section 8), beside
TRIGGERS = {
    "PERC-ADOPT": "the flag joins the recipe; then the stacked arm, and BAL-L at W 4.6991 re-read on the new base, because a better ROC makes the weighting lever's operating point worth moving",
    "PERC-AUROC-ONLY": "STOPORC and STOPDEC on the arm, zero training; then BAL-L stacked on it",
    "PERC-P-OTHER-PATH": "third seed, plus the CONSORC decomposition",
    "PERC-FP-COSTS": "report; that arm plus the training-window threshold rule of STOPTAU, under its own card",
    "PERC-NULL": "the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)",
    "PERC-REGRESSES": "not adopted. For A, W is halved and the arm re-read; for R, the row is reported",
    "SEEDS-DISAGREE": "a third seed (--seed 10 / 11) for the arm before any outcome is quoted (card section 5 rules)",
}
TIER = "T1-kin (self-action open loop; m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141; a planner diagnostic, never driving performance"
ESTIMATOR = ("taniteval.ci.paired_episode_cluster_bootstrap (cells, rates) and paired_stat_cluster_bootstrap (AUROC), B={nb}, seed 0, episodes resampled with both arms on the same draw. "
             "It answers 'another draw of EPISODES' only; 'another training run' is the floor max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|).")


# ------------------------------------------------------------------------------------------------------------------------------------------ small pieces
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auroc_fast(score, label):
    """Mann-Whitney AUROC, ties one half; vectorised (equals stopbal_read.auroc, pinned by a test).  NaN when a class is empty."""
    s, y = np.asarray(score, np.float64), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    rank_of = ends - (cnt - 1) / 2.0                 # average 1-based rank of each tie group
    ranks = rank_of[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _auc_stat(items):
    a = np.asarray(items, np.float64)
    return auroc_fast(a[:, 0], a[:, 1] > 0.5)


def floor_max(gap_rep, gap_rep2):
    """Card section 8 (Q3): a cell's floor is the larger of the two replicate gaps.  None when either replicate is missing: such a cell is never a gain or a cost."""
    if gap_rep is None or gap_rep2 is None:
        return None
    return max(gap_rep, gap_rep2)  # FLOORMAX  (the mutation check replaces this with the NULL_rep gap alone)


def judge(pt, separated, floor, higher):
    """rule: gain = separated AND better side AND |pt| > floor; cost = separated AND worse side AND |pt| > floor.  No floor => neither."""
    if floor is None:
        return False, False, False
    above = abs(pt) > floor
    better = (pt > 0) if higher else (pt < 0)
    worse = (pt < 0) if higher else (pt > 0)
    return bool(separated and better and above), bool(separated and worse and above), bool(above)


def cue_arrays(sd):
    """The STOPCUE definition (theory/stopcue/stopcue_read.py read_arm) from one STOPDEC npz (rows already in dump order)."""
    has = np.asarray(sd["has"], bool)
    return {"has": has, "pos": np.asarray(sd["stop_hit"], bool), "reg_stop": np.asarray(sd["v_end_gt"], np.float64) < 0.5,
            "goal_p": np.asarray(sd["goal_prob"], np.float64)[:, SR.STOP_GI], "lon_p": np.asarray(sd["lon_prob"], np.float64)[:, SR.LON_STOP_I]}


# ------------------------------------------------------------------------------------------------------------------------------------------ floors
def compute_floors_s1(rep, rep2, null0, eid, sp, ok):
    """Every floor = floor_max(gap to NULL_rep, gap to NULL_rep2), both read against NULL_s0 over the same windows; applied to both seeds (the first dose's reading r2)."""
    eid = np.asarray(eid)
    sets = {"all": np.ones(len(eid), bool), "sp": np.asarray(sp, bool) & np.asarray(ok, bool)}
    cells, detail = {}, {}
    for sname, msk in sets.items():
        for cname in null0["cells"]:
            g1, g2 = SR.floor_delta(rep["cells"][cname], null0["cells"][cname], msk), SR.floor_delta(rep2["cells"][cname], null0["cells"][cname], msk)
            cells[(sname, cname)] = floor_max(g1, g2)
            detail[f"{sname}|{cname}"] = {"gap_rep": g1, "gap_rep2": g2, "floor": cells[(sname, cname)]}
    ok_b, sp_b = np.asarray(ok, bool), np.asarray(sp, bool)
    masks = {"hit": sp_b & ok_b, "fpr": ok_b & ~sp_b}
    out = {"cells": cells, "detail": detail}

    def rate_gap(r, key, m):
        a, b = np.asarray(r["goal_stop"], np.float64)[masks[m]], np.asarray(null0["goal_stop"], np.float64)[masks[m]]
        return float(abs(a.mean() - b.mean())) if len(a) else None
    for name in ("hit", "fpr"):
        g1, g2 = rate_gap(rep, "goal_stop", name), rate_gap(rep2, "goal_stop", name)
        out[name] = floor_max(g1, g2)
        detail[name] = {"gap_rep": g1, "gap_rep2": g2, "floor": out[name]}

    def auc_gap(r, key, which):
        c, n = r["cue"], null0["cue"]
        sel = c["has"] if which != "goal_sp" else (np.asarray(ok, bool))
        ya = c["pos"] if which != "goal_sp" else np.asarray(sp, bool)
        sc = (lambda d: d["lon_p"]) if which == "lon" else (lambda d: d["goal_p"])
        yl = c["reg_stop"] if which == "lon" else ya
        return float(abs(auroc_fast(sc(c)[sel], yl[sel]) - auroc_fast(sc(n)[sel], yl[sel])))
    for which in ("goal", "lon", "goal_sp"):
        g1, g2 = auc_gap(rep, "cue", which), auc_gap(rep2, "cue", which)
        out["auroc_" + which] = None if (math.isnan(g1) or math.isnan(g2)) else floor_max(g1, g2)
        detail["auroc_" + which] = {"gap_rep": g1, "gap_rep2": g2, "floor": out["auroc_" + which]}
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ one seed of arm A
def auc_cell(CI, sa, sb, y, eid, mask, floor, nb, higher=True):
    """Paired AUROC delta (arm - NULL) over `mask`; interval from the paired episode-cluster bootstrap of the statistic; gain / cost by rule."""
    sa, sb, y, eid, mask = np.asarray(sa, np.float64), np.asarray(sb, np.float64), np.asarray(y, bool), np.asarray(eid), np.asarray(mask, bool)
    m = mask & np.isfinite(sa) & np.isfinite(sb)
    if not m.any() or y[m].sum() == 0 or (~y[m]).sum() == 0:
        return {"status": "NOT-APPLICABLE", "n_windows": int(m.sum()), "gain": False, "cost": False}
    a_arm, a_null = auroc_fast(sa[m], y[m]), auroc_fast(sb[m], y[m])
    pt = float(a_arm - a_null)
    yy = y[m].astype(np.float64)
    r = CI.paired_stat_cluster_bootstrap(list(zip(sa[m], yy)), list(zip(sb[m], yy)), eid[m], _auc_stat, n_boot=nb, seed=SR.SEED)
    sep = bool(r["separated"]) and not r.get("undefined", False)
    gain, cost, above = judge(pt, sep, floor, higher)
    return {"status": "OK", "auroc_arm": float(a_arm), "auroc_NULL": float(a_null), "delta": pt, "lo": r.get("lo"), "hi": r.get("hi"), "separated": sep, "floor": floor, "above_floor": above,
            "gain": gain, "cost": cost, "n_windows": int(m.sum()), "n_pos": int(y[m].sum()), "n_episodes": int(len(np.unique(eid[m])))}


def read_seed_s1(CI, null, arm, floors, eid, sp, ok, nb=SR.NB, ceiling_stoporc=None):
    """One seed of arm A against NULL of the same seed.  null / arm: {"cells", "goal_stop", "lon_stop", "goal_score", "lon_score", "cue"} (SR.arm_from_files + cue_arrays)."""
    eid, sp, ok = np.asarray(eid), np.asarray(sp, bool), np.asarray(ok, bool)
    sets = {"all": np.ones(len(eid), bool), "sp": sp & ok}
    cells = {}
    for (sname, cname) in [(s, c) for s in ("sp", "all") for c in arm["cells"]]:
        fl = floors["cells"].get((sname, cname))
        cells[f"{sname}|{cname}"] = SR.cell_delta(CI, arm["cells"][cname], null["cells"][cname], eid, sets[sname], cname in SR.HIGHER_BETTER, fl, nb)
    P, NR = cells[f"{SR.CELL_P[0]}|{SR.CELL_P[1]}"], cells[f"{SR.CELL_NR[0]}|{SR.CELL_NR[1]}"]
    # G-AUROC (STOPCUE definition, s1): the has-label windows, p(STOP_POINT) against stop_hit; lon-STOP score against the regime stop (stopcue_read's auroc_lon_stop_vs_regime_stop)
    ca, cn = arm["cue"], null["cue"]
    goal = auc_cell(CI, ca["goal_p"], cn["goal_p"], ca["pos"], eid, ca["has"], floors.get("auroc_goal"), nb, True)
    lon = auc_cell(CI, ca["lon_p"], cn["lon_p"], ca["reg_stop"], eid, ca["has"], floors.get("auroc_lon"), nb, True)
    goal_sp = auc_cell(CI, ca["goal_p"], cn["goal_p"], sp, eid, ok, floors.get("auroc_goal_sp"), nb, True)      # beside: the first dose's AUROC window set (has-and-present, 117 positives)
    # G-FP: the decision p(STOP_POINT) >= 0.5 on the non-stop has-and-present windows (the first dose's M gate fpr set); the hit rate beside
    ga, gn = np.asarray(arm["goal_stop"], np.float64), np.asarray(null["goal_stop"], np.float64)
    fpr = SR.cell_delta(CI, ga, gn, eid, ok & ~sp, False, floors.get("fpr"), nb)
    hit = SR.cell_delta(CI, ga, gn, eid, ok & sp, True, floors.get("hit"), nb)
    for d, (m_) in ((fpr, ok & ~sp), (hit, ok & sp)):
        if d.get("status") == "OK":
            d["rate_arm"] = float(ga[m_ & np.isfinite(ga)].mean())
            d["rate_NULL"] = float(gn[m_ & np.isfinite(gn)].mean())
    lon_ok = bool(not lon.get("cost"))
    flags = {"P": bool(P.get("gain")), "NR_cost": bool(NR.get("cost")), "NR_ok": not bool(NR.get("cost")), "G_AUROC_goal_gain": bool(goal.get("gain")), "G_AUROC_lon_cost": bool(lon.get("cost")),
             "G_AUROC": bool(goal.get("gain")) and lon_ok, "G_FP_cost": bool(fpr.get("cost")), "G_FP_ok": not bool(fpr.get("cost"))}
    named_costs = [k for k, v in cells.items() if v.get("cost")] + [f"cue|{n}" for n, v in (("auroc_goal", goal), ("auroc_lon", lon), ("fpr", fpr), ("hit", hit)) if v.get("cost")]
    named_gains = [k for k, v in cells.items() if v.get("gain")] + [f"cue|{n}" for n, v in (("auroc_goal", goal), ("auroc_lon", lon), ("fpr", fpr), ("hit", hit)) if v.get("gain")]
    rec = None
    if P.get("status") == "OK":
        rec = {"ceiling_m": CEILING_M, "fraction_of_1p5m_recovered": float(-P["delta"] / CEILING_M)}
        if ceiling_stoporc:
            rec["stoporc_ceiling_m"] = ceiling_stoporc
            rec["fraction_of_stoporc_ceiling_recovered"] = float(-P["delta"] / ceiling_stoporc)
    return {"flags": flags, "P_cell": P, "NR_cell": NR, "P_ceiling_recovered": rec, "G_AUROC_goal": goal, "G_AUROC_lon": lon, "G_AUROC_goal_beside_first_dose_windows": goal_sp, "G_FP": fpr, "hit_beside": hit,
            "cells": cells, "named_costs": named_costs, "named_gains": named_gains}


# ------------------------------------------------------------------------------------------------------------------------------------------ the outcome (card section 5 table, s4 precedence)
def arm_outcome(per_seed):
    """per_seed: [seed 0 read, seed 1 read] of read_seed_s1.  The card's six predicates, literally; 'holds' = at BOTH seeds, 'fails' = at NEITHER seed; any mixture holds no predicate (SEEDS-DISAGREE)."""
    f = [s["flags"] for s in per_seed]
    allp = lambda k: all(x[k] for x in f)  # noqa: E731
    nonep = lambda k: not any(x[k] for x in f)  # noqa: E731
    hold = {
        "PERC-REGRESSES": any(x["NR_cost"] for x in f),
        "PERC-ADOPT": allp("P") and allp("G_AUROC") and allp("G_FP_ok") and allp("NR_ok"),
        "PERC-P-OTHER-PATH": allp("P") and allp("NR_ok") and nonep("G_AUROC"),
        "PERC-FP-COSTS": allp("G_AUROC") and any(x["G_FP_cost"] for x in f),
        "PERC-AUROC-ONLY": allp("G_AUROC") and allp("NR_ok") and nonep("P"),
        "PERC-NULL": nonep("P") and nonep("G_AUROC"),
    }
    holding = [o for o in LABELS if hold[o]]
    label = holding[0] if holding else "SEEDS-DISAGREE"
    return {"label": label, "holding": holding, "per_seed_flags": f, "trigger": TRIGGERS[label], "triggers_of_all_holding": {o: TRIGGERS[o] for o in holding}}


# ------------------------------------------------------------------------------------------------------------------------------------------ whole read on loaded arms
def read_all(CI, arms, eid, sp0, ok0, nb=SR.NB, seeds=(0, 1)):
    """arms: {"NULL_s0", "NULL_s1", "NULL_rep", "NULL_rep2", "A_s0", "A_s1": arm dicts}.  -> {"outcome", "floors", "A": {...}} or an INVALID record.  Control 4: all three of NULL_s0, NULL_rep, NULL_rep2 must be present."""
    need = ("NULL_s0", "NULL_rep", "NULL_rep2")
    missing = [t for t in need if t not in arms]
    if missing:
        return {"outcome": "INVALID", "reason": f"control 4: {missing} not present: the floor is max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|), so no gain or cost can be read"}
    ready = {s: (f"NULL_s{s}" in arms and f"A_s{s}" in arms) for s in seeds}
    if not any(ready.values()):
        return {"outcome": "NOT-YET", "reason": "no seed has both its NULL and its A arm"}
    floors = compute_floors_s1(arms["NULL_rep"], arms["NULL_rep2"], arms["NULL_s0"], eid, sp0, ok0)
    out = {"floors": {"rule": "card section 8: max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|) per cell, read at seed 0, applied to both seeds (the first dose's reading r2)",
                      "detail": floors["detail"], "cells": {f"{k[0]}|{k[1]}": v for k, v in floors["cells"].items()},
                      "hit": floors["hit"], "fpr": floors["fpr"], "auroc_goal": floors["auroc_goal"], "auroc_lon": floors["auroc_lon"], "auroc_goal_beside": floors["auroc_goal_sp"]},
           "A": {"seeds": {}}}
    per = []
    for s in seeds:
        if not ready[s]:
            out["A"]["seeds"][f"s{s}"] = {"status": "NOT-YET"}
            continue
        r = read_seed_s1(CI, arms[f"NULL_s{s}"], arms[f"A_s{s}"], floors, eid, sp0, ok0, nb, CEILING_STOPORC.get(f"s{s}"))
        out["A"]["seeds"][f"s{s}"] = r
        per.append(r)
    if len(per) == len(seeds) == 2:
        out["A"].update(arm_outcome(per))
        out["outcome"] = out["A"]["label"]
    else:
        out["A"].update(label="NOT-YET (one seed read; decides nothing)", holding=[], trigger=None)
        out["outcome"] = out["A"]["label"]
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ banks, reuse state
def bank_dir(tag, pull=PULL):
    return f"{pull}/stopperc_{tag}"


def _json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _stop_keys(x, path=""):
    hits = []
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).startswith(("stop_", "STOPAUX", "w_stop_aux")):
                hits.append(f"{path}/{k}")
            hits += _stop_keys(v, f"{path}/{k}")
    return hits


def argv_flag(argv_text, flag):
    t = argv_text.split()
    return t[t.index(flag) + 1] if flag in t else None


def selfcheck_problems(tag, sc, argv_text):
    """Controls 1 / 2 of one stage-1 bank (stopbal_read.check_selfcheck: levers 1.0 / 1.0, G-DVB PASS, T live) plus the A-specific read-backs.  [] = pass."""
    bad = SR.check_selfcheck("BAL-NULL", sc)
    lv = (sc or {}).get("levers") or {}
    arm_a = tag.startswith("A_")
    if arm_a:
        if lv.get("stop_aux") != "on":
            bad.append(f"control 1: levers.stop_aux reads {lv.get('stop_aux')!r}, arm A is built with 'on' (an arm that reads off did not test the lever)")
        w, wa = lv.get("w_stop_aux"), argv_flag(argv_text, "--w-stop-aux")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
            bad.append(f"control 1: levers.w_stop_aux = {w!r}, wanted finite > 0")
        elif wa is None or abs(float(wa) - w) > 1e-9:
            bad.append(f"control 1: levers.w_stop_aux {w} differs from the argv's --w-stop-aux {wa!r}")
        if argv_flag(argv_text, "--stop-aux") != "on":
            bad.append("control 1: the bank's argv.txt has no '--stop-aux on'")
    else:
        hits = _stop_keys({k: v for k, v in (sc or {}).items()})
        if hits or "--stop-aux" in argv_text.split() or "--w-stop-aux" in argv_text.split():
            bad.append(f"control 1: a NULL arm carries stop-aux entries {hits[:3]} / flags (an off arm must carry none)")
    return bad


def bank_problems(tag, pull=PULL):
    """[] iff the stage-1 bank of `tag` is whole and unchanged: files present, ckpt sha256 equals BANK_MANIFEST.json and RUN.done, tag and commit as expected, selfcheck passes."""
    d = bank_dir(tag, pull)
    out = []
    if not os.path.isdir(d):
        return [f"{tag}: the bank folder {d} is missing"]
    for f in ("BANK.done", "BANK_MANIFEST.json", "RUN.done", "selfcheck.json", "ckpt_last.pt", "argv.txt"):
        if not os.path.isfile(os.path.join(d, f)):
            out.append(f"{tag}: {f} is missing in {d}")
    if out:
        return out
    try:
        got, man, run, sc = sha_file(d + "/ckpt_last.pt"), _json(d + "/BANK_MANIFEST.json"), _json(d + "/RUN.done"), _json(d + "/selfcheck.json")
        argv_text = open(d + "/argv.txt", encoding="utf-8").read()
    except (OSError, ValueError) as e:
        return [f"{tag}: a bank file is unreadable ({type(e).__name__})"]
    m_sha = (man.get("files", {}).get("ckpt_last.pt") or {}).get("sha256")
    r_sha = (run.get("evidence") or {}).get("ckpt_sha256")
    if got != m_sha:
        out.append(f"{tag}: ckpt_last.pt hashes to {got[:12]}, BANK_MANIFEST.json says {str(m_sha)[:12]} (the checkpoint changed)")
    if got != r_sha:
        out.append(f"{tag}: ckpt_last.pt hashes to {got[:12]}, RUN.done says {str(r_sha)[:12]} (the checkpoint changed)")
    if man.get("tag") not in (None, tag, "stopperc_" + tag):
        out.append(f"{tag}: the manifest names tag {man.get('tag')!r}")
    if man.get("commit") not in (None, COMMIT_S1):
        out.append(f"{tag}: the bank was trained at commit {man.get('commit')}, the code of record is {COMMIT_S1}")
    out += [f"{tag}: {x}" for x in selfcheck_problems(tag, sc, argv_text)]
    return out


def reuse_state(state=STATE, pull=PULL):
    """-> (mode, problems, info).  mode in {"EQUAL", "DIFFERS"}; problems non-empty => the read refuses.  An absent / UNKNOWN / unreadable NULL_REUSE.json is a refusal, never a default."""
    p = os.path.join(state, "NULL_REUSE.json")
    if not os.path.isfile(p):
        return None, [f"NULL_REUSE.json is absent ({p}): the card's NULL reuse rule has not been run; no NULL arm can be chosen"], {}
    try:
        j = _json(p)
    except (OSError, ValueError):
        return None, [f"NULL_REUSE.json is unreadable ({p})"], {}
    mode = j.get("null_reuse")
    if mode not in ("EQUAL", "DIFFERS"):
        return None, [f"NULL_REUSE.json reads null_reuse={mode!r} (status {j.get('status')!r}): not EQUAL or DIFFERS, so the read refuses"], j
    pr = []
    if j.get("status") != "computed":
        pr.append(f"NULL_REUSE.json status is {j.get('status')!r}, not 'computed'")
    n0 = bank_dir("NULL_s0", pull) + "/ckpt_last.pt"
    if not os.path.isfile(n0):
        pr.append(f"NULL_s0's checkpoint {n0} is missing")
    elif j.get("ckpt_sha256") not in (None, sha_file(n0)):
        pr.append("NULL_REUSE.json was computed on another NULL_s0 checkpoint than the banked one")
    if j.get("bal_pinned_sha256") != BAL_NULL_S0_PIN:
        pr.append(f"NULL_REUSE.json pins BAL-NULL_s0 at {str(j.get('bal_pinned_sha256'))[:12]}, the read's pin is {BAL_NULL_S0_PIN[:12]}")
    if mode == "EQUAL":
        if j.get("tensor_equal") is not True:
            pr.append("NULL_REUSE.json says EQUAL but tensor_equal is not true")
        bp = NB.problems(pull, tags=("BAL-NULL_s1", "BAL-NULL_rep"))
        pr += [f"reused first-dose bank: {x}" for x in bp]
        b0 = os.path.join(NB.bank_dir("BAL-NULL_s0", pull), "ckpt_last.pt")
        if not os.path.isfile(b0) or sha_file(b0) != BAL_NULL_S0_PIN:
            pr.append("BAL-NULL_s0's checkpoint is missing or does not hash to its pin")
    else:
        if j.get("tensor_equal") is not False:
            pr.append("NULL_REUSE.json says DIFFERS but tensor_equal is not false")
    return mode, pr, j


def stage_tags(mode):
    """The arms whose bank is a stage-1 run: always NULL_s0, NULL_rep2, A_s0, A_s1; NULL_s1 and NULL_rep too when the reuse rule failed."""
    return list(ARMS_ALWAYS) + (list(ARMS_COND) if mode == "DIFFERS" else [])


def arm_map(mode, pull=PULL, dumps_new=T1_NEW, dumps_bal=T1_BAL, sdec_new=SDEC_NEW, sdec_bal=SDEC_BAL):
    """arm name -> {dump, stopdec, stopdec_json, stopdec_tag, bank} (stopbal_read's per-arm record)."""
    m = {}
    for t in stage_tags(mode):
        m[t] = {"dump": f"{dumps_new}/dump_{t}_m10_causal_heads", "stopdec": f"{sdec_new}/stopdec_{t}.npz", "stopdec_json": f"{sdec_new}/STOPDEC_{t}.json", "stopdec_tag": t, "bank": bank_dir(t, pull)}
    if mode == "EQUAL":
        for t, b in (("NULL_s1", "BAL-NULL_s1"), ("NULL_rep", "BAL-NULL_rep")):
            m[t] = {"dump": f"{dumps_bal}/dump_{b}_m10_causal_heads", "stopdec": f"{sdec_bal}/stopdec_{b}.npz", "stopdec_json": f"{sdec_bal}/STOPDEC_{b}.json", "stopdec_tag": b, "bank": f"{pull}/stopbal_{b}"}
    return m


def dump_done_problems(tag, spec, bank_ckpt_sha):
    """The dump's done file (written by the driver after its checks) names the bank's checkpoint sha256."""
    full = os.path.basename(spec["dump"])[len("dump_"):]
    done = os.path.join(os.path.dirname(spec["dump"]), f"done_{full}.txt")
    if not os.path.isfile(done):
        return [f"{tag}: the dump's done file {done} is missing"]
    try:
        d = _json(done)
    except (OSError, ValueError):
        return [f"{tag}: the dump's done file is unreadable"]
    return [] if d.get("ckpt_sha256") == bank_ckpt_sha and d.get("n_dumps") == 141 else [f"{tag}: the dump's done file does not match the bank's checkpoint sha256 / 141 episode dumps"]


# ------------------------------------------------------------------------------------------------------------------------------------------ the dump driver (first dose's, re-pointed)
def _stage_refusal(tag, pull):
    """run_dumps_stopbal.refusal for the stage-1 banks: (reason or None, expected ckpt sha256 or None).  Reads files only."""
    d = bank_dir(tag, pull)
    ck = d + "/ckpt_last.pt"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})", None
    pr = bank_problems(tag, pull)
    want = None
    try:
        want = _json(d + "/BANK_MANIFEST.json")["files"]["ckpt_last.pt"]["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    return ("REFUSED: " + " | ".join(pr)) if pr else None, want


def cmd_dumps(tags, state, pull=PULL):
    mode, pr, _ = reuse_state(state, pull)
    if pr:
        for x in pr:
            print("REFUSED: " + x)
        raise SystemExit(2)
    allow = stage_tags(mode)
    tags = tags or allow
    bad = [t for t in tags if t not in allow]
    if bad:
        raise SystemExit(f"REFUSED: dumps of {bad}: with the reuse state {mode} this package dumps {allow} only")
    import run_dumps_stopbal as RD
    RD.ARMS = tuple(allow)
    RD.COMMIT = COMMIT_S1
    RD.TREE, RD.OUT, RD.PULL, RD.HERE = TREE, T1_NEW, pull, HERE
    RD.bank_dir = lambda tag: bank_dir(tag, pull)
    RD.refusal = lambda tag: _stage_refusal(tag, pull)
    sys.argv = ["run_dumps_stopbal.py"] + list(tags)
    RD.main()


def _same(x, y):
    try:
        return bool(np.array_equal(x, y, equal_nan=True))
    except TypeError:                                  # non-numeric arrays (strings)
        return bool(np.array_equal(x, y))


def compare_dumps(da, db):
    """Bit-for-bit comparison of two dump folders (141 ep*.npz each): every array equal, NaN-aware.  -> {"equal", "n_files", "max_abs_diff", "differing"}"""
    fa, fb = sorted(glob.glob(os.path.join(da, "ep*.npz"))), sorted(glob.glob(os.path.join(db, "ep*.npz")))
    res = {"n_files_a": len(fa), "n_files_b": len(fb), "differing": [], "max_abs_diff": 0.0}
    if [os.path.basename(x) for x in fa] != [os.path.basename(x) for x in fb] or not fa:
        res.update(equal=False, reason="the two folders hold different file names or none")
        return res
    for a, b in zip(fa, fb):
        za, zb = np.load(a), np.load(b)
        if sorted(za.files) != sorted(zb.files):
            res["differing"].append(os.path.basename(a) + ": keys")
            continue
        for k in za.files:
            x, y = za[k], zb[k]
            if x.shape != y.shape or not _same(x, y):
                res["differing"].append(f"{os.path.basename(a)}:{k}")
                if x.shape == y.shape and x.dtype.kind in "fiu":
                    res["max_abs_diff"] = max(res["max_abs_diff"], float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    res["equal"] = not res["differing"]
    return res


# ------------------------------------------------------------------------------------------------------------------------------------------ the read
def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def render_verdict(rep):
    """VERDICT_STOPPERC_S1.md from the result record.  Every number is the JSON's; tier and estimator on top; STRATEGIC paused."""
    L = [f"# VERDICT STOPPERC stage 1, arm A: {rep['outcome']}" + ("  (SMOKE: a code check with stand-in arms, never a result)" if rep.get("smoke") else ""), ""]
    L.append(f"Tier: **{TIER}**. Estimator: {rep['estimator']}")
    L.append(f"Evidence class: MEASURED (ours; dumps, zero training). Code of record {COMMIT_S1[:8]}. NULL reuse state: **{rep.get('null_reuse')}**. STRATEGIC: **PAUSED** (PI item 25), not a pass, not omitted.")
    pr = rep["controls"]["problems"]
    L.append(f"Controls: {len(pr)} problem(s)." + ("" if not pr else " The outcome is INVALID:"))
    L += [f"* {p}" for p in pr]
    L.append("Control 5 (the chain's G-SUITE / the Master Mind's pre-launch suite step) is NOT checked here.")
    if "A" not in rep or "seeds" not in rep.get("A", {}):
        L.append(f"\n{rep.get('reason', '')}")
        return "\n".join(L) + "\n"
    A = rep["A"]
    L += ["", f"## Outcome: **{A['label']}**", f"* holding predicates: {A.get('holding')}", f"* committed next step: {A.get('trigger')}"]
    for o, t in (A.get("triggers_of_all_holding") or {}).items():
        L.append(f"  * {o}: {t}")
    L += ["", "## The four gates, per seed (delta = A minus the same-seed NULL; floor = max of the two replicate gaps)", "",
          "| gate | seed | NULL -> A | delta [CI] | floor | separated | verdict |", "|---|---|---|---|---|---|---|"]
    for s in ("s0", "s1"):
        r = A["seeds"].get(s, {})
        if r.get("status") == "NOT-YET":
            L.append(f"| all | {s} | NOT-YET | | | | |")
            continue
        P, NR, ga, lo, fp = r["P_cell"], r["NR_cell"], r["G_AUROC_goal"], r["G_AUROC_lon"], r["G_FP"]
        for nm, c, kind in (("P (tactical ADE, 117 stop-point windows, m)", P, "ade"), ("NR (tactical ADE, all windows, m)", NR, "ade"), ("G-AUROC goal (STOPCUE)", ga, "auc"), ("lon-STOP AUROC (not a cost)", lo, "auc"),
                            ("G-FP (fpr at 0.5, non-stop windows)", fp, "rate"), ("hit rate beside", r["hit_beside"], "rate")):
            if c.get("status") != "OK":
                L.append(f"| {nm} | {s} | {c.get('status')} | | | | |")
                continue
            v = "GAIN" if c["gain"] else ("COST" if c["cost"] else "-")
            pair = f"{c['auroc_NULL']:.4f} -> {c['auroc_arm']:.4f}" if kind == "auc" else (f"{c['rate_NULL']:.4f} -> {c['rate_arm']:.4f}" if kind == "rate" else "")
            L.append(f"| {nm} | {s} | {pair} | {_fmt(c['delta'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}] | {_fmt(c['floor'])} | {c['separated']} | {v} |")
        rc = r.get("P_ceiling_recovered")
        if rc:
            L.append(f"| P, fraction of the 1.5 m ceiling recovered (beside) | {s} | | {rc['fraction_of_1p5m_recovered']:+.3f} (of STOPORC's {rc.get('stoporc_ceiling_m', float('nan')):.4f} m: {rc.get('fraction_of_stoporc_ceiling_recovered', float('nan')):+.3f}) | | | |")
    L += ["", "Per-seed flags: " + json.dumps(A.get("per_seed_flags")), ""]
    L += ["## Four families (48 cells per seed: 2 window sets x (2 bands x 11 metrics + 2 TAC accuracies)); named costs and gains are separated, on their side and above the floor", ""]
    for s in ("s0", "s1"):
        r = A["seeds"].get(s, {})
        if "named_costs" in r:
            L.append(f"* {s}: costs {r['named_costs'] or 'none'}; gains {r['named_gains'] or 'none'}")
    L += ["* LONGITUDINAL, LATERAL and TACTICAL are read per cell in `RESULT_STOPPERC_S1.json` (`A.seeds.sN.cells`, `G_FP`, `hit_beside`). STRATEGIC: **PAUSED** (PI item 25).", "",
          "## Floors (card section 8: max of the two replicate gaps)", ""]
    fl = rep.get("floors", {})
    L.append(f"* P cell floor {_fmt(fl.get('cells', {}).get('sp|' + SR.CELL_P[1]))} m, NR cell floor {_fmt(fl.get('cells', {}).get('all|' + SR.CELL_NR[1]))} m; hit {_fmt(fl.get('hit'))}, fpr {_fmt(fl.get('fpr'))}, AUROC goal {_fmt(fl.get('auroc_goal'))}, lon {_fmt(fl.get('auroc_lon'))}")
    d = fl.get("detail", {}).get("sp|" + SR.CELL_P[1])
    if d:
        L.append(f"* P floor from the two replicates: |NULL_rep - NULL_s0| {_fmt(d['gap_rep'])}, |NULL_rep2 - NULL_s0| {_fmt(d['gap_rep2'])}")
    return "\n".join(L) + "\n"


def main_read(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdict", default=None)
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--pull", default=PULL)
    ap.add_argument("--map", help="arms JSON (smoke only): {NULL_s0, NULL_s1, NULL_rep, NULL_rep2, A_s0, A_s1: {dump, stopdec, stopdec_json, stopdec_tag, bank}}")
    ap.add_argument("--smoke", action="store_true", help="code check with stand-in arms: skips the bank / selfcheck / reuse controls; never a result")
    ap.add_argument("--n-boot", type=int, default=SR.NB)
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args(argv)
    t0 = time.time()
    CI = SR.load_ci()
    rep = {"tool": "stopperc_s1_read.py", "card": "theory/stopperc/PREREG_V8F_STOPPERC.md sections 5 and 8 (FROZEN) + AMENDMENT_3_A_READINGS.md", "code_of_record": COMMIT_S1, "tier": TIER,
           "evidence_class": "MEASURED (ours; dumps, zero training)", "estimator": ESTIMATOR.format(nb=a.n_boot) + f" ci.py loaded from {CI.SOURCE}.", "smoke": bool(a.smoke),
           "reading_choices": "README_READ_S1.md section 'Reading choices' (s1-s8)",
           "STRATEGIC": {"status": "PAUSED", "ruling": "PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, never as a pass, never omitted."},
           "controls": {"problems": [], "notes": {"control_5_suite": "NOT CHECKED HERE: the chain's G-SUITE and the Master Mind's pre-launch suite step"}}}
    prob = rep["controls"]["problems"]
    pa = SR.planted_auroc_controls()
    rep["controls"]["planted_auroc"] = pa
    if not pa["pass"]:
        prob.append("control 3: planted AUROC controls failed")
    if not auroc_fast(np.array([.9, .9, .1, .1, .1]), np.array([1, 1, 0, 0, 0], bool)) == 1.0:
        prob.append("control 3: the vectorised AUROC fails its planted case")
    mode = None
    if a.smoke:
        if not a.map:
            raise SystemExit("REFUSED: --smoke needs --map (stand-in arms)")
        arms_spec = json.load(open(a.map, encoding="utf-8"))
        rep["null_reuse"] = "SMOKE (stand-ins)"
    else:
        mode, pr, info = reuse_state(a.state, a.pull)
        rep["null_reuse"] = mode
        rep["null_reuse_record"] = info
        if pr:
            rep["outcome"], rep["reason"] = "REFUSED", "; ".join(pr)
            prob += [f"control 4: {x}" for x in pr]
            json.dump(rep, open(a.out, "w"), indent=1, default=float)
            print("WROTE", a.out, "REFUSED:", *pr, sep="\n  ")
            raise SystemExit(2)
        arms_spec = arm_map(mode, a.pull)
    rep["arm_paths"] = arms_spec
    # ---- control 4 (floors present) before any heavy load
    need = ("NULL_s0", "NULL_rep", "NULL_rep2")
    missing = [t for t in need if t not in arms_spec]
    if missing:
        rep["outcome"], rep["reason"] = "INVALID", f"control 4: {missing} not in the arm map"
        prob.append(rep["reason"])
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        raise SystemExit(2)
    ready = {s: (f"NULL_s{s}" in arms_spec and f"A_s{s}" in arms_spec) for s in (0, 1)}
    if not a.smoke:
        for t in arms_spec:                                   # control 6: every bank whole (stage-1 banks by bank_problems; reused first-dose banks were re-asserted by reuse_state)
            if t in stage_tags(mode):
                prob += bank_problems(t, a.pull)
    # ---- load arms
    refs = {s: SR.load_dump(a.ref.format(s=s)) for s in (0, 1) if ready[s]}
    ref_any = refs[min(refs)]
    geom = SR.build_geom(ref_any)
    G = geom["G"]
    cache, data = {}, {}
    for tag, spec in arms_spec.items():
        s = 0 if tag in ("NULL_rep", "NULL_rep2") else int(tag[-1])
        if s not in refs:
            refs[s] = SR.load_dump(a.ref.format(s=s))
        if tag.endswith("_s1") and not ready[1]:
            continue
        ref = refs[s]
        geom["glat"], geom["glon"] = G.labels(ref["g"])
        geom["J"] = G.join(ref)
        arm, sd, ctl = SR.arm_from_files(spec, ref, geom, cache)
        arm["cue"] = cue_arrays(sd)
        data[tag] = (arm, sd, ref)
        rep["controls"]["notes"][f"instrument_{tag}"] = ctl
        if not all(ctl.values()):
            prob.append(f"control 3: {tag} decision / probability arrays inconsistent {ctl}")
        if not a.smoke:
            ck = os.path.join(spec["bank"], "ckpt_last.pt")
            if os.path.isfile(ck):
                prob += dump_done_problems(tag, spec, sha_file(ck))
        sj = spec.get("stopdec_json")
        if sj and os.path.exists(sj):
            prob += [f"{tag}: {x}" for x in SR.check_instrument(json.load(open(sj, encoding="utf-8"))["controls"][spec.get("stopdec_tag", tag)])]
        elif not a.smoke:
            prob.append(f"control 3: {tag}: STOPDEC json missing")
    sets = {t: SR.window_sets(d[1]) for t, d in data.items()}
    ok0, sp0 = sets["NULL_s0"]
    rep["controls"]["window_counts"] = {"n_windows": int(len(ok0)), "n_has": int(np.asarray(data["NULL_s0"][1]["has"]).sum()), "n_ok": int(ok0.sum()), "n_stop_point": int(sp0.sum())}
    for t, (ok, sp) in sets.items():
        if not (np.array_equal(ok, ok0) and np.array_equal(sp, sp0)):
            prob.append(f"control 3: {t} stop-point / ok window set differs from NULL_s0")
    c = rep["controls"]["window_counts"]
    if (c["n_windows"], c["n_has"], c["n_stop_point"]) != (SR.EXPECT["n_windows"], SR.EXPECT["n_has"], SR.EXPECT["n_stop_point"]):
        prob.append(f"control 3: window counts {c} differ from the banked dump {SR.EXPECT}")
    n_pos_cue = int(data["NULL_s0"][0]["cue"]["pos"][data["NULL_s0"][0]["cue"]["has"]].sum())
    rep["controls"]["window_counts"]["n_stopcue_positives"] = n_pos_cue
    if n_pos_cue != 131:
        prob.append(f"control 3: STOPCUE positives {n_pos_cue}, the banked STOPCUE read has 131")
    eid = ref_any["eid"]
    res = read_all(CI, {t: d[0] for t, d in data.items()}, eid, sp0, ok0, a.n_boot)
    rep.update({k: v for k, v in res.items() if k != "outcome"})
    if prob:
        rep["outcome"] = "INVALID"
    elif a.smoke:
        rep["outcome"] = "SMOKE (code check, never a result): " + str(res.get("outcome"))
    else:
        rep["outcome"] = res.get("outcome")
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    if a.verdict:
        open(a.verdict, "w", encoding="utf-8").write(render_verdict(rep))
    print("WROTE", a.out, "| outcome:", rep["outcome"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    for p in prob:
        print("  INVALID:", p)
    return 0


def main(argv):
    if not argv or argv[0] not in ("refs", "dumps", "read", "identity", "tags"):
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    state = rest[rest.index("--state") + 1] if "--state" in rest else STATE
    pull = rest[rest.index("--pull") + 1] if "--pull" in rest else PULL
    if cmd == "refs":
        mode, pr, info = reuse_state(state, pull)
        if not pr:
            pr = [x for t in stage_tags(mode) for x in bank_problems(t, pull)]
        for x in pr:
            print("REFUSED: " + x)
        if not pr:
            print("NULL reuse", mode, "| stage-1 banks whole:", stage_tags(mode))
        return 2 if pr else 0
    if cmd == "tags":
        mode, pr, _ = reuse_state(state, pull)
        if pr:
            for x in pr:
                print("REFUSED: " + x, file=sys.stderr)
            return 2
        print(" ".join(stage_tags(mode)))
        return 0
    if cmd == "dumps":
        tags = [t for i, t in enumerate(rest) if not t.startswith("--") and (i == 0 or not rest[i - 1].startswith("--"))]
        return cmd_dumps(tags, state, pull)
    if cmd == "identity":
        r = compare_dumps(rest[0], rest[1])
        print(json.dumps(r))
        return 0 if r["equal"] else 1
    return main_read(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
