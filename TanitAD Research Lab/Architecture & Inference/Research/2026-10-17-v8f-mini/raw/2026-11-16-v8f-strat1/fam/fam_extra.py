"""fam_extra.py -- the two binding family members the STRAT1-family readers print NOT COMPUTABLE for (MM ruling 12, W-FAM-STRAT1-TTC-TACCONF).

LONGITUDINAL distance keeping (headway / time-gap / min-TTC / DK-VIOL) and TACTICAL decision quality (the lat / lon head decision, its confusion matrix, goal-token
selection, the executed manoeuvre and the declared-vs-driven agreement).  CPU, zero training, no GPU; reads banked T1-kin dumps in place, writes nothing but the result.

Defines NO new distance-keeping metric and NO new bootstrap: every DK number is `taniteval.lead_metrics` (distance_keeping, lead_viol, paired_lead_viol, lead_viol_verdict, path_arc_length)
on the B1 lead block, joined by the OTBAND read's own `join` (theory/obd/read/obd_geom.py extracts it unedited); every interval is `taniteval.ci`
(episode_cluster_bootstrap, paired_episode_cluster_bootstrap, stat_cluster_bootstrap, paired_stat_cluster_bootstrap), B 2,000, seed 0.

TIERS (stamped on every block):
  DK-*            T1-kin  (self-action open loop, m = 10 replan, causal bucket, dead-reckoned nav, logged scene; the lead follows its LOGGED future -- it does not react to the arm)
  TAC-SEL-*       T1-kin tick 0 (the head's decision at the window's first tick, dead-reckoned nav; NOT the T0 real-nav read, which is printed beside; control K-T0 reproduces the T0 evaluator's own confusion from its own heads)
  TAC-EXEC-*      T1-kin  (the manoeuvre read off the arm's rolled plan by refc_tactical.factor_from_kinematics, vs the one read off the human path)
  TAC-SELEXE-*    T1-kin  (head decision at tick 0 vs the manoeuvre the same window's plan executes)
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 over the 141 episodes.  It answers "another draw of EPISODES" only; "another TRAINING run" is the floor F = |C0_rep - C0_s0|.
Labels MAY use ego (CLAUDE.md): the v9 lat / lon / goal labels are ground truth; the heads are vision-only at inference.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
READ = os.environ.get("STRAT1_READ_DIR", "F:/TanitAD-scratch/strat1/read")
for _p in (READ, READ + "/theory/obd/read", READ + "/theory/otband/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import strat1_lib as L  # noqa: E402  (pure numpy; the landed lib, never edited)

NB, SEED = L.NB, L.SEED
DT = 0.2
BANDS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30), "full_0_6s": (0, 30)}
DK_KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")
SEEDS = ("s0", "s1", "s10", "s11")
LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")            # tanitad.data.v9_labels.LAT9 (e81bde01)
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")          # LON9
KIN_LAT = ("lane_keep", "turn_left", "turn_right")                                       # refc_tactical.LAT_CLASSES
KIN_LON = ("brake_stop", "steady", "accelerate")                                         # refc_tactical.LON_CLASSES
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT", "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR",
          "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
#: the one head-lateral -> kin-lateral mapping that has no judgement in it; LANE_CHANGE_* have no kinematic-gate counterpart and are EXCLUDED (counted), never folded into lane_keep.
LAT_HEAD_TO_KIN = {0: 0, 1: 1, 2: 2}
IGNORE = -100
SHIFT = 24                                                                                 # item code = (window index << SHIFT) | per-window payload (<= 22 bits)
MASK = (1 << SHIFT) - 1
TIER_DK = "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; the lead follows its logged future and does not react to the arm)"
TIER_SEL = "T1-kin tick 0 (head decision at the window's first tick under the T1 feed: dead-reckoned nav, logged scene; vision-only heads scored against v9 labels that may use ego)"
TIER_EXE = "T1-kin (manoeuvre read off the arm's rolled plan by refc_tactical.factor_from_kinematics vs the human path's)"
TIER_SELEXE = "T1-kin (declared decision at tick 0 vs the manoeuvre the plan executes)"
EST = f"paired episode-cluster bootstrap (taniteval.ci), B={NB}, seed {SEED}, 141 episodes; 'another draw of EPISODES' only; training-run variance = floor F"


# ---------------------------------------------------------------------------------------------------------------------------------------------- loading
def dump_dir(dumps, arm):
    return f"{dumps}/dump_STRAT1-{arm}_m10_causal_heads"


def ticks_path(dumps, arm):
    return f"{dumps}/ticks/ticks_STRAT1-{arm}_m10_causal_heads.npz"


def arms_present(dumps):
    out = []
    for fam in ("C0", "N1", "S1", "S1d"):
        for s in SEEDS + ("rep",):
            a = f"{fam}_{s}"
            if os.path.isdir(dump_dir(dumps, a)) and os.path.isfile(ticks_path(dumps, a)) and os.path.isfile(f"{dumps}/done_STRAT1-{a}_m10_causal_heads.txt"):
                out.append(a)
    return out


def load_dump(d, n_eps=141):
    import glob
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if len(fs) != n_eps:
        raise RuntimeError(f"{d}: {len(fs)} episode files, expected {n_eps}")
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "v0", "eid", "clip_index", "ws")}


def tick0_heads(ticks_npz, clip_index, ws):
    """-> {'lat','lon','goal'} head outputs at each window's FIRST tick, in the dump's window order.  Refuses a window that has no tick 0 or a duplicate."""
    z = np.load(ticks_npz)
    m = np.asarray(z["tick"]) == 0
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(z["ci"])[m], np.asarray(z["t0"])[m]))}
    if len(key) != int(m.sum()):
        raise RuntimeError(f"{ticks_npz}: duplicate (ci, t0) among tick-0 rows")
    try:
        rows = np.asarray([key[(int(c), int(t))] for c, t in zip(clip_index, ws)])
    except KeyError as e:
        raise RuntimeError(f"{ticks_npz}: a dump window has no tick-0 row: {e}") from None
    if len(rows) != int(m.sum()):
        raise RuntimeError(f"{ticks_npz}: {int(m.sum())} tick-0 rows for {len(rows)} dump windows")
    pick = lambda k: np.asarray(z[k])[m][rows]  # noqa: E731
    return {"lat": pick("t_lat_logits").astype(np.float64), "lon": pick("t_lon_logits").astype(np.float64), "goal": pick("t_goal_prob").astype(np.float64)}


def v9_rows(join, clip_index, ws):
    """v9 / join labels of each dump window: row = off[ci] + t (strat1_lib.rows_of, the NOW frame)."""
    r = L.rows_of(join["off"], clip_index, ws)
    has = np.asarray(join["has"])[r].astype(bool)
    lat, lon = np.asarray(join["lat_cls"])[r].astype(np.int64), np.asarray(join["lon_cls"])[r].astype(np.int64)
    return {"has": has, "lat": np.where(has & (lat >= 0), lat, IGNORE), "lon": np.where(has & (lon >= 0), lon, IGNORE),
            "goal_y": np.asarray(join["goal_y"])[r].astype(np.int64), "goal_w": np.asarray(join["goal_w"])[r].astype(np.int64), "rows": r}


# ---------------------------------------------------------------------------------------------------------------------------------------------- DK family
def _lvl(CI, x, eid, nb):
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return {"status": "NOT COMPUTABLE", "reason": "no finite window", "n_windows": 0}
    r = CI.episode_cluster_bootstrap(x[ok], np.asarray(eid)[ok], reduce="mean", n_boot=nb, seed=SEED)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(r["n_windows"]), "n_episodes": int(r["n_episodes"])}


def dk_scores(LM, track, g, J, band):
    """(plan dk, human dk, plan arc, human arc) of one band.  track / g [N,30,2]; J = the OTBAND join (leads, lead_lens, speeds, is_lead)."""
    a, b = band
    p = np.asarray(track, np.float64)[:, a:b]
    h = np.asarray(g, np.float64)[:, a:b]
    dk = LM.distance_keeping(p, J["leads"][:, a:b], J["lead_lens"], J["speeds"], DT)
    hk = LM.distance_keeping(h, J["leads"][:, a:b], J["lead_lens"], J["speeds"], DT)
    return dk, hk, LM.path_arc_length(p), LM.path_arc_length(h)


def dk_arm(LM, CI, track, g, J, eid, band, nb=NB):
    """One arm, one band -> the DK block.  Only LEAD windows enter a mean; NO_LEAD / NOT_STRAIGHT / NO_LABEL are counted by the caller, never scored as free flow."""
    dk, hk, pa, ha = dk_scores(LM, track, g, J, band)
    out = {"tier": TIER_DK, "evidence_class": "MEASURED (ours; banked T1-kin dumps, zero training)", "n_lead_windows": int(np.sum(J["is_lead"])),
           "n_windows": int(len(eid)), "levels": {}, "human_levels": {}}
    for k in DK_KEYS:
        out["levels"][k] = _lvl(CI, dk[k], eid, nb)
        out["human_levels"][k] = _lvl(CI, hk[k], eid, nb)
    out["n_headway"] = int(dk.get("n", 0))
    out["n_time_gap"] = int(dk.get("n_time_gap", 0))
    out["n_closing"] = int(dk.get("n_closing", 0))
    out["n_lead_lost"] = int(np.sum(J["is_lead"] & ~np.isfinite(np.asarray(dk["headway_min_m"], np.float64))))
    out["ttc_note"] = dk.get("censoring_note", "no window with a lead in the corridor")
    try:
        lv = LM.lead_viol(dk, hk, eid, lead_valid=J["is_lead"], n_boot=nb, seed=SEED, plan_arc=pa, human_arc=ha)
        ver = LM.lead_viol_verdict(lv)
        out["DK-VIOL"] = {k: lv.get(k) for k in ("status", "reason", "n_lead_windows", "n_viol", "share", "n_at_risk", "share_of_at_risk", "share_of_at_risk_ci",
                                                  "lead_progress_ratio_clipped", "n_plan_lost_lead")}
        out["DK-VIOL"]["progress"] = {k: (lv.get("progress") or {}).get(k) for k in ("status", "lead_progress_ratio", "lead_progress_ratio_clipped", "n_at_risk", "n")}
        out["DK-VIOL"]["verdict"] = {k: ver.get(k) for k in ("verdict", "bar_decision", "reason")}
    except Exception as e:  # noqa: BLE001  (a family that raises is recorded, never dropped)
        lv = None
        out["DK-VIOL"] = {"status": "NOT COMPUTABLE", "reason": f"{type(e).__name__}: {e}", "n": 0}
    return out, {"dk": dk, "lv": lv, "human_dk": hk}


def dk_pair(LM, CI, A, B, eid, names, F=None, nb=NB):
    """A - B on the DK levels (finite in both) and on DK-VIOL (the instrument's paired reading).  F = {key: floor} (|C0_rep - C0_s0|) when known."""
    out = {"estimator": EST, "levels": {}, "DK-VIOL": None}
    for k in DK_KEYS:
        r = L.paired(CI, np.asarray(A["dk"][k]), np.asarray(B["dk"][k]), eid, nb)
        if r.get("status") == "OK":
            f = (F or {}).get(k)
            r["floor"] = f
            r["above_floor"] = None if f is None else bool(abs(r["delta"]) > f)
            r["cost_or_gain_note"] = "a positive delta is MORE gap / time / TTC than the comparator; better-or-worse is the distance to the human and DK-VIOL"
        out["levels"][k] = r
    if A["lv"] is not None and B["lv"] is not None:
        try:
            p = LM.paired_lead_viol(A["lv"], B["lv"], eid, names=names, n_boot=nb, seed=SEED)
            ar = p.get("at_risk") or {}
            out["DK-VIOL"] = {"at_risk": {k: ar.get(k) for k in ("status", "n_at_risk", "n_viol_a", "n_viol_b", "delta", "lo", "hi", "separated")},
                              "x7_direction_met_at_risk": p.get("x7_direction_met_at_risk")}
            f = (F or {}).get("dk_viol_share_of_at_risk")
            out["DK-VIOL"]["floor"] = f
            d = ar.get("delta")
            out["DK-VIOL"]["above_floor"] = None if (f is None or d is None) else bool(abs(d) > f)
        except Exception as e:  # noqa: BLE001
            out["DK-VIOL"] = {"status": "NOT COMPUTABLE", "reason": f"{type(e).__name__}: {e}"}
    else:
        out["DK-VIOL"] = {"status": "NOT COMPUTABLE", "reason": "one arm's DK-VIOL block did not compute", "n": 0}
    return out


def human_control(LM, g, J, eid, band, nb=200):
    """the instrument's own known-value control: the HUMAN path scored as the plan reads n_viol exactly 0 and progress exactly 1.0"""
    a, b = band
    hk = LM.distance_keeping(np.asarray(g, np.float64)[:, a:b], J["leads"][:, a:b], J["lead_lens"], J["speeds"], DT)
    ha = LM.path_arc_length(np.asarray(g, np.float64)[:, a:b])
    lv = LM.lead_viol(hk, hk, eid, lead_valid=J["is_lead"], n_boot=nb, seed=SEED, plan_arc=ha, human_arc=ha)
    return {"n_viol": lv.get("n_viol"), "lead_progress_ratio": lv.get("lead_progress_ratio"), "n_at_risk": lv.get("n_at_risk"),
            "PASS": bool(lv.get("n_viol") == 0 and lv.get("lead_progress_ratio") == 1.0)}


# ---------------------------------------------------------------------------------------------------------------------------------------------- TAC family
def confusion(yt, yp, K, ignore=IGNORE):
    yt, yp = np.asarray(yt).ravel(), np.asarray(yp).ravel()
    m = (yt != ignore) & (yt >= 0) & (yt < K) & (yp >= 0) & (yp < K)
    return np.bincount(yt[m] * K + yp[m], minlength=K * K).reshape(K, K).astype(np.int64)


def conf_stats(cm):
    """accuracy, kappa, macro-F1 over classes PRESENT among the labels (the evaluator's v8f_consistency.macro_f1 definition), per-class recall / precision / F1"""
    cm = np.asarray(cm, np.int64)
    K = cm.shape[0]
    n = int(cm.sum())
    sup = cm.sum(1)
    pred = cm.sum(0)
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(pred > 0, tp / pred, np.nan)
        f1 = np.where(sup > 0, np.where(tp == 0, 0.0, 2.0 * tp / (sup + pred)), np.nan)
    acc = float(tp.sum() / n) if n else float("nan")
    pe = float((sup * pred).sum() / (n * n)) if n else float("nan")
    kap = float((acc - pe) / (1 - pe)) if n and pe < 1 else float("nan")
    return {"n": n, "accuracy": acc, "kappa": kap, "macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
            "recall": rec, "precision": prec, "f1": f1, "support": sup}


def _conf_stat_fn(gt, K, names, ignore=IGNORE):
    """-> stat(list of item codes) -> {accuracy, kappa, macro_f1, recall_<name>...}.  Item = (window idx << SHIFT) | predicted class; gt is the closure."""
    gt = np.asarray(gt)

    def stat(items):
        v = np.asarray(items, np.int64)
        idx, p = v >> SHIFT, v & MASK
        s = conf_stats(confusion(gt[idx], p, K, ignore))
        out = {"accuracy": s["accuracy"], "kappa": s["kappa"], "macro_f1": s["macro_f1"]}
        out.update({f"recall_{names[c]}": float(s["recall"][c]) for c in range(K)})
        return out
    return stat


def _items(pred):
    return [(i << SHIFT) | int(p) for i, p in enumerate(np.asarray(pred).astype(np.int64))]


def conf_block(CI, gt, pred, K, names, eid, nb=NB, ignore=IGNORE):
    """one arm's decision block: confusion matrix (rows = GT, cols = predicted), per-class recall / support, accuracy, kappa, macro-F1 with episode-cluster CIs, majority baseline"""
    gt = np.asarray(gt)
    cm = confusion(gt, pred, K, ignore)
    s = conf_stats(cm)
    m = (gt != ignore) & (gt >= 0) & (gt < K)
    if not m.any():
        return {"status": "NOT COMPUTABLE", "reason": "no labelled window", "n": 0}
    stat = _conf_stat_fn(gt, K, names, ignore)
    ci = CI.stat_cluster_bootstrap(_items(pred), np.asarray(eid), stat, n_boot=nb, seed=SEED)
    maj = int(np.argmax(cm.sum(1)))
    maj_cm = confusion(gt, np.full(len(gt), maj), K, ignore)
    ms = conf_stats(maj_cm)
    return {"n": s["n"], "n_episodes": int(len(np.unique(np.asarray(eid)[m]))), "classes": list(names), "confusion_rows_GT_cols_pred": cm.tolist(),
            "support": s["support"].tolist(), "accuracy": ci["accuracy"], "kappa": ci["kappa"], "macro_f1": ci["macro_f1"],
            "recall": {names[c]: (None if not np.isfinite(s["recall"][c]) else ci[f"recall_{names[c]}"]) for c in range(K)},
            "precision": {names[c]: (None if not np.isfinite(s["precision"][c]) else float(s["precision"][c])) for c in range(K)},
            "majority_baseline": {"class": names[maj], "accuracy": ms["accuracy"], "macro_f1": ms["macro_f1"]},
            "estimator": "episode_cluster_bootstrap (stat), B=%d, seed %d" % (nb, SEED)}


def conf_pair(CI, gt, pred_a, pred_b, K, names, eid, F=None, nb=NB, ignore=IGNORE):
    gt = np.asarray(gt)
    r = CI.paired_stat_cluster_bootstrap(_items(pred_a), _items(pred_b), np.asarray(eid), _conf_stat_fn(gt, K, names, ignore), n_boot=nb, seed=SEED)
    out = {}
    for k, v in r.items():
        if k.startswith("recall_") and not np.isfinite(conf_stats(confusion(gt, pred_a, K, ignore))["recall"][names.index(k[7:])]):
            continue
        v = dict(v)
        f = (F or {}).get(k)
        v["floor"] = f
        v["above_floor"] = None if (f is None or not np.isfinite(v.get("delta", np.nan))) else bool(abs(v["delta"]) > f)
        out[k] = v
    return out


def goal_arrays(goal_prob, goal_y, goal_w, has):
    P = np.asarray(goal_prob) > 0.5
    sh = np.arange(22, dtype=np.int64)
    Y = ((np.asarray(goal_y, np.int64)[:, None] >> sh) & 1).astype(bool)
    W = (((np.asarray(goal_w, np.int64)[:, None] >> sh) & 1).astype(bool)) & np.asarray(has, bool)[:, None]
    return P, Y, W


def _goal_f1(P, Y, W):
    """per token F1 on windows where the token is weighted (the evaluator's v8f_mini_eval.tactical_metrics definition; NaN where no weighted positive); -> (f1[22], tp, fp, fn)"""
    tp = (P & Y & W).sum(0).astype(np.float64)
    fp = (P & ~Y & W).sum(0).astype(np.float64)
    fn = (~P & Y & W).sum(0).astype(np.float64)
    npos = (Y & W).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(npos > 0, np.where(tp == 0, 0.0, 2 * tp / (2 * tp + fp + fn)), np.nan)
    return f1, tp, fp, fn


def _top1(goal_prob, Y, W):
    """windows with >= 1 weighted positive token: is the highest-probability token (among the tokens weighted in that window) a positive?  -> (hit[N] float, NaN where undefined)"""
    gp = np.where(W, np.asarray(goal_prob, np.float64), -np.inf)
    top = gp.argmax(1)
    has_pos = (Y & W).any(1)
    hit = np.where(has_pos, (Y & W)[np.arange(len(top)), top].astype(np.float64), np.nan)
    return hit


def goal_stat_fn(Y, W):
    """item = (window idx << 24) | predicted token bits"""
    sh = np.arange(22, dtype=np.int64)

    def stat(items):
        v = np.asarray(items, np.int64)
        idx, bits = v >> SHIFT, v & MASK
        P = ((bits[:, None] >> sh) & 1).astype(bool)
        f1, tp, fp, fn = _goal_f1(P, Y[idx], W[idx])
        wsum = W[idx].sum()
        return {"goal_macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
                "goal_micro_token_accuracy": float(((P == Y[idx]) & W[idx]).sum() / wsum) if wsum else float("nan")}
    return stat


def goal_block(CI, goal_prob, goal_y, goal_w, has, eid, nb=NB):
    P, Y, W = goal_arrays(goal_prob, goal_y, goal_w, has)
    if not (Y & W).any():
        return {"status": "NOT COMPUTABLE", "reason": "no weighted positive goal token", "n": 0}, (P, Y, W)
    bits = (P.astype(np.int64) << np.arange(22, dtype=np.int64)).sum(1)
    items = [(i << SHIFT) | int(b) for i, b in enumerate(bits)]
    st = goal_stat_fn(Y, W)
    ci = CI.stat_cluster_bootstrap(items, np.asarray(eid), st, n_boot=nb, seed=SEED)
    f1, tp, fp, fn = _goal_f1(P, Y, W)
    hit = _top1(goal_prob, Y, W)
    ok = np.isfinite(hit)
    top = CI.episode_cluster_bootstrap(hit[ok], np.asarray(eid)[ok], reduce="mean", n_boot=nb, seed=SEED) if ok.any() else None
    return {"n_windows_labelled": int(np.asarray(has).sum()), "n_weighted_cells": int(W.sum()), "tokens": list(GOAL22),
            "f1_per_token": {GOAL22[g]: (None if not np.isfinite(f1[g]) else float(f1[g])) for g in range(22)},
            "n_positive_per_token": {GOAL22[g]: int((Y & W)[:, g].sum()) for g in range(22)},
            "macro_f1": ci["goal_macro_f1"], "micro_token_accuracy": ci["goal_micro_token_accuracy"],
            "top1_in_positives": ({"mean": float(top["mean"]), "lo": float(top["lo"]), "hi": float(top["hi"]), "n_windows": int(ok.sum())} if top else
                                  {"status": "NOT COMPUTABLE", "reason": "no window with a weighted positive", "n": 0}),
            "majority_macro_f1": 0.0, "estimator": "episode_cluster_bootstrap (stat), B=%d, seed %d" % (nb, SEED)}, (P, Y, W)


def goal_pair(CI, bits_a, bits_b, Y, W, eid, F=None, nb=NB):
    ia = [(i << SHIFT) | int(b) for i, b in enumerate(bits_a)]
    ib = [(i << SHIFT) | int(b) for i, b in enumerate(bits_b)]
    r = CI.paired_stat_cluster_bootstrap(ia, ib, np.asarray(eid), goal_stat_fn(Y, W), n_boot=nb, seed=SEED)
    for k, v in r.items():
        f = (F or {}).get(k)
        v["floor"] = f
        v["above_floor"] = None if (f is None or not np.isfinite(v.get("delta", np.nan))) else bool(abs(v["delta"]) > f)
    return r


def selexe_lat(sel_lat, exe_lat):
    """declared lateral decision (head, 5 classes) vs the manoeuvre the plan executes (kin, 3 classes), on the unambiguous classes only; LANE_CHANGE_* excluded and counted"""
    sel, exe = np.asarray(sel_lat), np.asarray(exe_lat)
    m = np.isin(sel, list(LAT_HEAD_TO_KIN))
    mapped = np.vectorize(LAT_HEAD_TO_KIN.get)(sel[m]) if m.any() else np.array([], np.int64)
    agree = np.zeros(len(sel), np.float64) * np.nan
    agree[m] = (mapped == exe[m]).astype(np.float64)
    cm = confusion(np.where(m, np.where(m, sel, 0), IGNORE), exe, 3)
    return agree, {"n_mapped": int(m.sum()), "n_excluded_lane_change": int((~m).sum()), "mapping": "LANE_KEEP->lane_keep, TURN_L->turn_left, TURN_R->turn_right; LANE_CHANGE_L/R excluded (no kinematic-gate class)",
                   "cross_rows_head_cols_executed": cm.tolist()}


def selexe_lon_table(sel_lon, exe_lon):
    """descriptive cross-table, head class (7) x executed kin class (3); no scalar agreement because the two vocabularies have no one-to-one map"""
    cm = np.zeros((7, 3), np.int64)
    np.add.at(cm, (np.asarray(sel_lon), np.asarray(exe_lon)), 1)
    return {"rows_head_classes": list(LON9), "cols_executed_kin": list(KIN_LON), "counts": cm.tolist(), "n": int(cm.sum()),
            "note": "descriptive; the head vocabulary (HOLD/CREEP/STOP/FOLLOW/DECELERATE/ACCELERATE/KEEP) has no one-to-one map onto brake_stop/steady/accelerate, so no agreement scalar is formed"}


# ---------------------------------------------------------------------------------------------------------------------------------------------- driver
def build_geometry():
    """OTBAND read's join / labels, extracted unedited (needs torch; loads tanitad + taniteval from the e81bde01 read tree through read_lib)"""
    import obd_geom as G
    return G.build()


def arm_state(dumps, arm):
    d = load_dump(dump_dir(dumps, arm))
    h = tick0_heads(ticks_path(dumps, arm), d["clip_index"], d["ws"])
    return d, h


def run(dumps, out_dir, nb=NB, arms=None, t0_dir=None, log=print):
    t_start = time.time()
    geom = build_geometry()
    BL = sys.modules["read_lib"]
    LM, CI = BL.LM, BL.CI
    arms = arms or arms_present(dumps)
    if "C0_s0" not in arms:
        raise RuntimeError("C0_s0 is the reference dump and is required")
    join9 = dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False))
    ref, _ = arm_state(dumps, "C0_s0")
    eid = ref["eid"]
    gci, gt_ = L.grid_windows(L.pinned("eval_grid_windows.json"), join9["clip"])
    grid_ok = bool(np.array_equal(gci, ref["clip_index"]) and np.array_equal(gt_, ref["ws"]))
    J = geom.join(ref)
    V = v9_rows(join9, ref["clip_index"], ref["ws"])
    kin_lat_g, kin_lon_g = geom.labels(ref["g"])
    res = {"tool": "fam_extra.py", "work_item": "W-FAM-STRAT1-TTC-TACCONF", "evidence_class": "MEASURED (ours; banked T1-kin dumps, zero training, CPU)", "estimator": EST,
           "tiers": {"DK": TIER_DK, "TAC-SEL": TIER_SEL, "TAC-EXEC": TIER_EXE, "TAC-SELEXE": TIER_SELEXE},
           "arms_read": arms, "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "lead_window_states": J["counts"],
           "controls": {"grid_matches_dump": grid_ok, "problems": []}, "families": {"LONGITUDINAL-DK": {}, "TACTICAL": {}}, "pairs": {}}
    prob = res["controls"]["problems"]
    if not grid_ok:
        prob.append("K-GRID: the dump's (clip_index, ws) differ from the eval grid's windows")
    res["controls"]["K-HUMAN"] = {b: human_control(LM, ref["g"], J, eid, BANDS[b]) for b in BANDS}
    if not all(v["PASS"] for v in res["controls"]["K-HUMAN"].values()):
        prob.append("K-HUMAN: the human path scored as the plan did not read n_viol 0 / progress 1.0")
    S = {}
    for a in arms:
        d, h = arm_state(dumps, a)
        for k in ("eid", "clip_index", "ws", "g", "v0"):
            if not np.array_equal(d[k], ref[k]):
                raise RuntimeError(f"{a}: window or model-free array differs from the reference dump: {k}")
        exe_lat, exe_lon = geom.labels(d["cl"])
        S[a] = {"d": d, "h": h, "exe_lat": exe_lat, "exe_lon": exe_lon, "sel_lat": h["lat"].argmax(1), "sel_lon": h["lon"].argmax(1)}
        dkb, raw = {}, {}
        for b, band in BANDS.items():
            dkb[b], raw[b] = dk_arm(LM, CI, d["cl"], d["g"], J, eid, band, nb)
        S[a]["dk_raw"] = raw
        gb, gparts = goal_block(CI, h["goal"], V["goal_y"], V["goal_w"], V["has"], eid, nb)
        S[a]["goal_parts"] = gparts
        S[a]["goal_bits"] = (gparts[0].astype(np.int64) << np.arange(22, dtype=np.int64)).sum(1)
        tac = {"tier": TIER_SEL,
               "TAC-SEL-LAT": conf_block(CI, V["lat"], S[a]["sel_lat"], 5, LAT9, eid, nb),
               "TAC-SEL-LON": conf_block(CI, V["lon"], S[a]["sel_lon"], 7, LON9, eid, nb),
               "TAC-GOAL": gb,
               "TAC-EXEC-LAT": dict(conf_block(CI, kin_lat_g, exe_lat, 3, KIN_LAT, eid, nb), tier=TIER_EXE),
               "TAC-EXEC-LON": dict(conf_block(CI, kin_lon_g, exe_lon, 3, KIN_LON, eid, nb), tier=TIER_EXE)}
        agree, sx = selexe_lat(S[a]["sel_lat"], exe_lat)
        ok = np.isfinite(agree)
        sx["agreement"] = dict(_lvl(CI, agree, eid, nb), n_windows=int(ok.sum())) if ok.any() else {"status": "NOT COMPUTABLE", "reason": "no mapped window", "n": 0}
        sx["tier"] = TIER_SELEXE
        S[a]["agree"] = agree
        tac["TAC-SELEXE-LAT"] = sx
        tac["TAC-SELEXE-LON"] = dict(selexe_lon_table(S[a]["sel_lon"], exe_lon), tier=TIER_SELEXE)
        tac["TAC-ANCHOR"] = {"status": "NOT COMPUTABLE", "reason": "the arm commits to ONE path per window: there is no candidate fan and no selector, so no anchor selection exists to score "
                             "(four_families.tactical_goal's own reason; closing it needs a fan + selector surface)", "n": int(len(eid))}
        res["families"]["LONGITUDINAL-DK"][a] = dkb
        res["families"]["TACTICAL"][a] = tac
        log(f"read {a} {time.time() - t_start:.0f}s", flush=True)
    # ---- floors F = |C0_rep - C0_s0| on each cell (one replicate pair; applied to every seed)
    F = {"DK": {}, "TAC": {}}
    if "C0_rep" in S:
        for b in BANDS:
            f = {}
            for k in DK_KEYS:
                x, y = res["families"]["LONGITUDINAL-DK"]["C0_rep"][b]["levels"][k], res["families"]["LONGITUDINAL-DK"]["C0_s0"][b]["levels"][k]
                f[k] = abs(x["mean"] - y["mean"]) if "mean" in x and "mean" in y else None
            vx, vy = (res["families"]["LONGITUDINAL-DK"][t][b]["DK-VIOL"].get("share_of_at_risk") for t in ("C0_rep", "C0_s0"))
            f["dk_viol_share_of_at_risk"] = None if vx is None or vy is None else abs(vx - vy)
            F["DK"][b] = f
        tr, t0_ = res["families"]["TACTICAL"]["C0_rep"], res["families"]["TACTICAL"]["C0_s0"]
        for blk in ("TAC-SEL-LAT", "TAC-SEL-LON", "TAC-EXEC-LAT", "TAC-EXEC-LON"):
            F["TAC"][blk] = {k: abs(tr[blk][k]["point"] - t0_[blk][k]["point"]) for k in ("accuracy", "kappa", "macro_f1") if "point" in tr[blk].get(k, {}) and "point" in t0_[blk].get(k, {})}
        F["TAC"]["TAC-GOAL"] = {k: abs(tr["TAC-GOAL"][k]["point"] - t0_["TAC-GOAL"][k]["point"]) for k in ("macro_f1", "micro_token_accuracy")}
        F["TAC"]["TAC-GOAL"] = {("goal_macro_f1" if k == "macro_f1" else "goal_micro_token_accuracy"): v for k, v in F["TAC"]["TAC-GOAL"].items()}
    res["floors"] = {"F": F, "definition": "|C0_rep - C0_s0| per cell at seed 0, applied to every seed (one replicate pair); a delta needs |delta| > F to be read as a lever effect"}
    # ---- pairs
    pairs = []
    for s in SEEDS:
        for x, y in (("S1", "C0"), ("N1", "C0"), ("S1", "N1"), ("S1d", "C0"), ("S1d", "S1")):
            if f"{x}_{s}" in S and f"{y}_{s}" in S:
                pairs.append((f"{x}_{s}", f"{y}_{s}"))
    for a, b in pairs:
        pr = {"DK": {}, "TAC": {}}
        for bn in BANDS:
            pr["DK"][bn] = dk_pair(LM, CI, S[a]["dk_raw"][bn], S[b]["dk_raw"][bn], eid, (a, b), (F["DK"] or {}).get(bn), nb)
        for blk, gt, K, nm, pa, pb in (("TAC-SEL-LAT", V["lat"], 5, LAT9, S[a]["sel_lat"], S[b]["sel_lat"]), ("TAC-SEL-LON", V["lon"], 7, LON9, S[a]["sel_lon"], S[b]["sel_lon"]),
                                       ("TAC-EXEC-LAT", kin_lat_g, 3, KIN_LAT, S[a]["exe_lat"], S[b]["exe_lat"]), ("TAC-EXEC-LON", kin_lon_g, 3, KIN_LON, S[a]["exe_lon"], S[b]["exe_lon"])):
            pr["TAC"][blk] = conf_pair(CI, gt, pa, pb, K, list(nm), eid, (F["TAC"] or {}).get(blk), nb)
        _, Y, W = S[a]["goal_parts"]
        pr["TAC"]["TAC-GOAL"] = goal_pair(CI, S[a]["goal_bits"], S[b]["goal_bits"], Y, W, eid, (F["TAC"] or {}).get("TAC-GOAL"), nb)
        ag_a, ag_b = S[a]["agree"], S[b]["agree"]
        pr["TAC"]["TAC-SELEXE-LAT"] = L.paired(CI, ag_a, ag_b, eid, nb)
        res["pairs"][f"{a}-{b}"] = pr
    # ---- controls K1 (identity: a self-pair reads exactly 0.0 [0, 0]) and K-T0 (the tick-0 head confusion reproduces the T0 evaluator's own)
    k1 = {}
    a0 = "C0_s0"
    k1["DK"] = {k: L.paired(CI, np.asarray(S[a0]["dk_raw"]["full_0_6s"]["dk"][k]), np.asarray(S[a0]["dk_raw"]["full_0_6s"]["dk"][k]), eid, nb) for k in DK_KEYS}
    k1["TAC"] = conf_pair(CI, V["lat"], S[a0]["sel_lat"], S[a0]["sel_lat"], 5, list(LAT9), eid, None, nb)
    rd = lambda v: (v.get("delta") == 0.0 and v.get("lo") == 0.0 and v.get("hi") == 0.0)  # noqa: E731
    k1["pass"] = bool(all(rd(v) for v in k1["DK"].values()) and all(rd(v) for v in k1["TAC"].values() if "delta" in v and np.isfinite(v["delta"])))
    res["controls"]["K1_self_pair"] = {"pass": k1["pass"], "reading": "every self-pair cell reads exactly 0.0 [0.0, 0.0]"}
    if not k1["pass"]:
        prob.append("K1: a self-pair did not read exactly 0.0")
    kt0, beside = {}, {}
    for a in arms:
        p = f"{t0_dir}/eval_STRAT1-{a}_real.json" if t0_dir else None
        if not p or not os.path.isfile(p) or not os.path.isfile(p[:-5] + ".npz"):
            kt0[a] = {"status": "not run", "reason": "no T0 real-nav eval json + npz for this arm"}
            continue
        tj = json.load(open(p, encoding="utf-8"))["raw"]["tactical"]
        z = np.load(p[:-5] + ".npz")
        # K-T0: the T0 evaluator's OWN heads (its npz) scored by THIS module's confusion / goal code against the same v9 join must equal the evaluator's own tactical block exactly
        same_win = bool(np.array_equal(z["win_ci"], ref["clip_index"]) and np.array_equal(z["win_t"], ref["ws"]))
        cl, co = np.asarray(tj["lat"]["confusion"], np.int64), np.asarray(tj["lon"]["confusion"], np.int64)
        mine_l = confusion(V["lat"], np.asarray(z["lat_logits"]).argmax(1), 5)
        mine_o = confusion(V["lon"], np.asarray(z["lon_logits"]).argmax(1), 7)
        Pz, Yz, Wz = goal_arrays(z["goal_prob"], V["goal_y"], V["goal_w"], V["has"])
        f1z = _goal_f1(Pz, Yz, Wz)[0]
        gm = float(np.nanmean(f1z)) if np.isfinite(f1z).any() else float("nan")
        kt0[a] = {"same_windows": same_win, "lat_confusion_equal": bool(np.array_equal(cl, mine_l)), "lon_confusion_equal": bool(np.array_equal(co, mine_o)),
                  "goal_macro_f1_abs_diff": abs(float(tj["goal_macro_f1"]) - gm)}
        kt0[a]["PASS"] = bool(same_win and kt0[a]["lat_confusion_equal"] and kt0[a]["lon_confusion_equal"] and kt0[a]["goal_macro_f1_abs_diff"] < 1e-9)
        if not kt0[a]["PASS"]:
            prob.append(f"K-T0 {a}: this module's confusion / goal code on the T0 evaluator's heads differs from the evaluator's own tactical block")
        # BESIDE (not a control): the T1 tick-0 read vs the T0 real-nav read of the same window.  They differ because T1 feeds dead-reckoned nav, T0 the real nav.
        t = res["families"]["TACTICAL"][a]
        beside[a] = {"lat_accuracy_t1_tick0": t["TAC-SEL-LAT"]["accuracy"]["point"], "lat_accuracy_t0_real_nav": float(np.trace(cl) / cl.sum()),
                     "lon_accuracy_t1_tick0": t["TAC-SEL-LON"]["accuracy"]["point"], "lon_accuracy_t0_real_nav": float(np.trace(co) / co.sum()),
                     "goal_macro_f1_t1_tick0": t["TAC-GOAL"]["macro_f1"]["point"], "goal_macro_f1_t0_real_nav": float(tj["goal_macro_f1"]),
                     "why_they_differ": "T1 tick 0 is fed the dead-reckoned nav (--nav deadreckon), the T0 evaluator the real nav; both are vision-only heads"}
    res["controls"]["K-T0"] = kt0
    res["beside_rows"] = {"T1_tick0_vs_T0_real_nav": beside}
    res["provenance"] = {"reader_sha256": {f: L.sha256_file(f"{HERE}/{f}") for f in ("fam_extra.py",)},
                         "lead_metrics": os.path.abspath(LM.__file__), "ci": os.path.abspath(CI.__file__), "code_of_record": L.COMMIT,
                         "inputs_sha256": {k: L.PINS[k] for k in ("join_eval141.npz", "eval_grid_windows.json")}, "lead_block_sha256": geom.ns.get("LEAD_SHA"), "lead_block": geom.ns.get("LEAD")}
    res["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/RESULT_FAM_EXTRA.json", "w", encoding="utf-8"), indent=1, default=_j)
    return res


def _j(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def reader_rows(res, pair):
    """compact lines for a reader to print for one pair name like 'S1_s0-C0_s0': the DK and TAC cells with delta, CI, floor and n (nothing pooled)"""
    p = res["pairs"][pair]
    rows = []
    for bn, blk in p["DK"].items():
        for k, v in blk["levels"].items():
            if v.get("status") == "OK":
                rows.append(("LONGITUDINAL-DK", f"{bn}:{k}", v["delta"], v["lo"], v["hi"], v.get("above_floor"), v["n_windows"]))
        dv = blk.get("DK-VIOL") or {}
        ar = dv.get("at_risk") or {}
        if ar.get("delta") is not None:
            rows.append(("LONGITUDINAL-DK", f"{bn}:DK-VIOL share_of_at_risk", ar["delta"], ar["lo"], ar["hi"], dv.get("above_floor"), ar.get("n_at_risk")))
    for blk, cells in p["TAC"].items():
        if blk == "TAC-SELEXE-LAT":
            if cells.get("status") == "OK":
                rows.append(("TACTICAL", blk, cells["delta"], cells["lo"], cells["hi"], None, cells["n_windows"]))
            continue
        for k, v in cells.items():
            if "delta" in v and np.isfinite(v["delta"]):
                rows.append(("TACTICAL", f"{blk}:{k}", v["delta"], v["lo"], v["hi"], v.get("above_floor"), v["n_windows"]))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=READ + "/t1")
    ap.add_argument("--t0", default=READ + "/out/t0")
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--n-boot", type=int, default=NB)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    r = run(a.dumps, a.out, a.n_boot, [x for x in a.arms.split(",") if x] or None, a.t0)
    print("WROTE", a.out + "/RESULT_FAM_EXTRA.json", "| problems:", r["controls"]["problems"], "|", r["wall_s"], "s")
