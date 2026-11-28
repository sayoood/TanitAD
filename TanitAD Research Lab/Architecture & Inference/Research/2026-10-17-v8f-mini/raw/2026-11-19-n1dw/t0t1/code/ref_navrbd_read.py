"""navrbd_read.py -- the PREREG_V8F_NAVRBD read (card FROZEN, sha256 881d7f3d; sections 0a, 5, 6, 7).  CPU, zero training.  Evidence class of every number: MEASURED (ours; T1-kin dumps).
Tier: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, nav_rb held from t0, c13 dead-reckoned); STRATEGIC = T0 (read by navrb2_strategic.py style files, passed with --contrast).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0: 'another draw of EPISODES' only.  Training-run variance = the floor F_HO (card 5.3); inference variance does not apply
(deterministic planner, one environment, one thread count).  Optimistic by construction: the route is supplied along the driven path (card 3).

COMMITTED RULES (the card verbatim in force)
  D1-HO (deciding): TAC-LAT accuracy on the T1-feasible held-out grid (5,736 windows / 497 episodes), the KINEMATIC label read_otband.labels (3 classes: lane_keep / turn_left / turn_right) of the
     plan track against the same label of the GT track (obd_read.to_cells).  NOT the 5-way v9 lat_cls.  Delta = N1D_sK - N1R_sK, paired, per seed (K = 0, 1; N1Dx_s0 against N1R_s0).
  GAIN at a seed: delta >= BAR = max(0.0383, F_HO) (= 0.0424) AND the paired CI lower bound above 0.  COST at a seed: delta <= -BAR AND the CI upper bound below 0.
  F_HO = |N1R_s1 - N1R_s0| on the same windows (MEASURED 0.042364 at step 0).  If the N1R pair is re-dumped beside the N1D dumps and F_HO differs, the larger is the bar.
  D1 on eval-141 RB windows and D2-HO (TURN recall, side given turn, false-turn share) are beside: no bar.
  Guard: the 24 eval-141 cells (LONGITUDINAL / LATERAL / TACTICAL / ADE-FDE) N1D - N1R under F2, no cost at either seed, and no ROUTE-COST; STRATEGIC read by the frozen readers (fog unscored).
  Controls: K1 (self-pair exactly 0.0), K-CONST (majority predictor accuracy == majority share), K-FLOOR (live F_HO == banked HO_FLOOR.json), K-RE (N1R - N1 recomputed == RESULT_NAVRB2.json),
     Z-EVAL (real - zero exactly 0.0 on windows whose channel history is all 0, every cell), Z-USE (real - zero on RB windows), N1Dx (not a gain).
  Outcomes (card 7): RBD-GAIN / RBD-ONE-SEED / RBD-NO-GAIN / RBD-UNREAD / RBD-COSTS / RBD-FLOOR-HIGH; every label that holds is printed with its next lever.  A missing input is NOT-YET, never a pass.
READING CHOICES (written before any N1D number; printed into the result)  see READING.
Usage:  python navrbd_read.py --out RESULT_NAVRBD.json [--md ...] [--alias N1D_s0=N1R_s0 ...] [--skip-141] [--contrast J] [--n-boot 2000]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                                   # .../strat1/navrbd
N2R = "F:/TanitAD-scratch/strat1/navrb2/read"
SR = "F:/TanitAD-scratch/strat1/read"
for _p in (N2R, SR, SR + "/theory/obd/read", SR + "/theory/otband/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CARD_SHA256_PREFIX = "881d7f3d"
CARD_PATH = ROOT + "/freeze/PREREG_V8F_NAVRBD.md"
BAR_NAVRB2 = 0.0383
F_HO_BANKED_FILE = ROOT + "/step0/out/HO_FLOOR.json"
GRID_SHA256 = "ca289d0f40a8f71043796fe626b26ecaa141853f843acdb1eecbf5648b9ab148"
HO_WINDOWS, HO_EPISODES = 5736, 497
NB, SEED = 2000, 0
DIRS = {                                           # every location overridable by environment for the stand-in runs
    "n1r_ho": os.environ.get("NAVRBD_N1R_HO", ROOT + "/step0/t1"),
    "d_ho": os.environ.get("NAVRBD_D_HO", HERE + "/dumps"),
    "d_141": os.environ.get("NAVRBD_D_141", HERE + "/dumps141"),
    "n1r_141": os.environ.get("NAVRBD_N1R_141", N2R + "/t1"),
    "banked_141": os.environ.get("NAVRBD_BANKED_141", SR + "/t1"),
}
HO_ARMS = ("N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1Dx_s0", "N1D_s0Z", "N1D_s1Z")
CL = ("lane_keep", "turn_left", "turn_right")

READING = {
    "r1_Z-USE_separated": "Z-USE at a seed is 'separated' when real - zero on the held-out TAC-LAT (D1-HO windows, kinematic label) has a paired CI that excludes 0, either side; the eval-141 24-cell real - zero on RB windows is printed beside (needs the N1D zero dumps on eval-141)",
    "r2_guard": "the guard holds when no eval-141 cell is a cost (separated worse and |delta| > F2) at either seed and there is no ROUTE-COST; a missing contrast file leaves STRATEGIC NOT-YET and RBD-GAIN is then printed as RBD-GAIN-PENDING-STRATEGIC",
    "r3_bar": "the bar is max(0.0383, F_HO) with F_HO the UNROUNDED value (0.042364...); GAIN is delta >= bar; COST is delta <= -bar with the CI above / below 0",
    "r4_floor_high": "RBD-FLOOR-HIGH fires when F_HO > 0.0383 and F_HO > the signed N1D - N1R delta at every read seed",
    "r5_N1Dx": "N1Dx_s0 is read against N1R_s0; if it is a gain, no N1D label stands (printed as N1DX-REGRESSION-GAINS) and the would-be labels are printed beside as labels_if_N1Dx_ignored",
    "r6_Z-EVAL": "Z-EVAL on eval-141: windows whose nav_rb frames over the item's history (cfg.hmax rows) are all 0 in the NAVRB2 1-column table (c1..c14 are 0 whenever c0 is 0, so the same mask is the 15-column all-zero mask); on the held-out grid only when --ho-zero-hist-mask (npy bool [5736]) is given",
    "r7_outcome_status": "FINAL only with every input present and no control problem; otherwise NOT-YET / INVALID, and any labels are printed as unvalidated",
}

NEXT = {
    "RBD-GAIN": "the channel enters the recipe: seeds s10 / s11 (PI go) and S1D, the route-text arm with the exit + distance channel (4 arms); then the vision-geometry lever on the remaining RB gap vs the v9 recall",
    "RBD-ONE-SEED": "N1D at s10 / s11 (2 arms), COST4 / GAIN4 under F2",
    "RBD-NO-GAIN": "the stacking-combiner check (CPU, zero-training): RB-JOINT vision probe out-of-fold probabilities stacked into the ED tree, on the same CV and on the held-out windows; then the vision lever: an RB window weight in the loss and/or the 8x20 grid input (RAW8 - S was +0.0266 separated, below the bar)",
    "RBD-UNREAD": "dose: --nav-rb-lr-mult 30 at s0 (PI go); first the Adam-state reachability read on e_nav_rb.weight, per column group",
    "RBD-COSTS": "not adopted; N1 replicate with the new code (the zeroed-channel arm, 1 arm) to separate the channel from training noise",
    "RBD-FLOOR-HIGH": "the cell cannot resolve this lever at these seeds: more seeds of the pair (PI go), never a lower bar",
    "RBD-GAIN-PENDING-STRATEGIC": "run navrb2_strategic.py style contrast for N1D and pass it with --contrast; RBD-GAIN needs the STRATEGIC guard",
    "N1DX-REGRESSION-GAINS": "the arm's rule, not the channel, produced the gain: no N1D label stands; audit the shuffle variant (identity?) and the dump",
    "NOT-YET": "an input is missing",
}


# ------------------------------------------------------------------------------------------------------------------ pure logic (unit-tested on planted rows)
def pair_ref(seed):
    """the N1R arm an N1D arm at `seed` is paired with (card 4: N1D_s0 / N1R_s0, N1D_s1 / N1R_s1)"""
    return f"N1R_s{int(seed)}"


def d_arm(seed):
    return f"N1D_s{int(seed)}"


def bar_of(f_ho, *others):
    """GAIN bar: max(0.0383, F_HO [, every re-dumped F_HO])  (card 5.3; the larger is the bar)"""
    return float(max(BAR_NAVRB2, f_ho, *others))


def f_ho_of(acc_s0, acc_s1):
    """F_HO = |N1R_s1 - N1R_s0| of the per-window TAC-LAT accuracies on the same windows"""
    return float(abs(np.mean(np.asarray(acc_s1, np.float64)) - np.mean(np.asarray(acc_s0, np.float64))))


def check_three_class(lab):
    """the D1-HO label is the 3-class kinematic factor; a 5-way v9 lat_cls vector (values above 2) is refused"""
    lab = np.asarray(lab)
    assert lab.min() >= 0 and lab.max() <= 2, f"not a 3-class kinematic label (max {lab.max()}): the 5-way lat_cls is not the D1-HO label"
    return lab


def kin_acc(g_lab, p_lab):
    """per-window TAC-LAT accuracy from the KINEMATIC 3-class labels of the GT track and the plan track (read_otband.labels)"""
    return (np.asarray(p_lab) == np.asarray(g_lab)).astype(np.float64)


def d1_ho(CI, acc_arm, acc_ref, eid, bar, nb=NB):
    """D1-HO row: arm - ref accuracy, paired episode-cluster bootstrap; gain = delta >= bar and lo > 0; cost = delta <= -bar and hi < 0"""
    a, b, e = np.asarray(acc_arm, np.float64), np.asarray(acc_ref, np.float64), np.asarray(eid)
    r = CI.paired_episode_cluster_bootstrap(a, b, e, n_boot=nb, seed=SEED)
    d = float(a.mean() - b.mean())
    lo, hi = float(r["lo"]), float(r["hi"])
    return {"status": "OK", "delta": d, "lo": lo, "hi": hi, "separated": bool(r["separated"]), "bar": float(bar), "gain": bool(d >= bar and lo > 0), "cost": bool(d <= -bar and hi < 0),
            "acc_arm": float(a.mean()), "acc_ref": float(b.mean()), "n_windows": int(len(a)), "n_episodes": int(len(np.unique(e)))}


def z_use_ho(CI, acc_real, acc_zero, eid, nb=NB):
    """Z-USE on the held-out windows: real - zero TAC-LAT, paired; separated either side (reading r1)"""
    a, b, e = np.asarray(acc_real, np.float64), np.asarray(acc_zero, np.float64), np.asarray(eid)
    r = CI.paired_episode_cluster_bootstrap(a, b, e, n_boot=nb, seed=SEED)
    return {"status": "OK", "delta": float(a.mean() - b.mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": int(len(a)), "n_episodes": int(len(np.unique(e)))}


def d2_ho(CI, g_lab, p_arm, p_ref, eid, nb=NB):
    """D2-HO beside: TURN recall (GT turn windows where the plan is any turn), side given turn (plan turn windows: same side), false-turn share (GT lane_keep windows where the plan is a turn); arm, ref, paired"""
    g, pa, pr, e = (np.asarray(x) for x in (g_lab, p_arm, p_ref, eid))
    out = {}
    turn, lk = g != 0, g == 0
    out["n_turn_windows"] = {"n_windows": int(turn.sum()), "n_episodes": int(len(np.unique(e[turn])))}
    out["n_lane_keep_windows"] = {"n_windows": int(lk.sum()), "n_episodes": int(len(np.unique(e[lk])))}
    for name, m, f in (("turn_recall", turn, lambda p: (p != 0)), ("false_turn_share", lk, lambda p: (p != 0))):
        if m.any():
            a, b = f(pa)[m].astype(np.float64), f(pr)[m].astype(np.float64)
            r = CI.paired_episode_cluster_bootstrap(a, b, e[m], n_boot=nb, seed=SEED)
            out[name] = {"arm": float(a.mean()), "ref": float(b.mean()), "delta": float(a.mean() - b.mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"])}
    for name, p in (("arm", pa), ("ref", pr)):
        both = turn & (p != 0)
        out[f"side_given_turn_{name}"] = {"value": float((p[both] == g[both]).mean()) if both.any() else None, "n_windows": int(both.sum())}
    return out


def k_const(g_lab, pred=None):
    """K-CONST: the majority-class predictor's accuracy equals the majority share exactly (pred is only for the test's planted non-constant predictor)"""
    g = np.asarray(g_lab)
    maj = int(np.bincount(g, minlength=3).argmax())
    acc = float(((np.full(len(g), maj) if pred is None else np.asarray(pred)) == g).mean())
    share = float((g == maj).mean())
    return {"majority_class": CL[maj], "majority_share": share, "constant_accuracy": acc, "pass": bool(acc == share)}


def k1_self(CI, acc, eid, nb=NB, ref=None):
    """K1: a self-pair reads exactly 0.0 with a degenerate interval (ref is only for the test's planted non-self pair)"""
    ref = acc if ref is None else ref
    r = CI.paired_episode_cluster_bootstrap(np.asarray(acc, np.float64), np.asarray(ref, np.float64), np.asarray(eid), n_boot=nb, seed=SEED)
    return {"delta": float(np.mean(acc) - np.mean(ref)), "lo": float(r["lo"]), "hi": float(r["hi"]), "pass": bool(float(np.mean(acc) - np.mean(ref)) == 0.0 and float(r["lo"]) == 0.0 and float(r["hi"]) == 0.0)}


def k_floor(f_live, f_banked):
    """K-FLOOR: the live F_HO from the dumps equals the banked step-0 value exactly (same dumps, same reader)"""
    return {"live": float(f_live), "banked": float(f_banked), "pass": bool(float(f_live) == float(f_banked))}


def z_eval(real, zero, mask, cells):
    """Z-EVAL: real - zero on the mask windows must be EXACTLY 0.0 on every cell, window by window (NaN equal to NaN); an empty mask fails"""
    m = np.asarray(mask, bool)
    out, ok = {}, bool(m.any())
    for c in cells:
        a, b = np.asarray(real[c], np.float64)[m], np.asarray(zero[c], np.float64)[m]
        nm = int((np.isnan(a) != np.isnan(b)).sum())
        both = np.isfinite(a) & np.isfinite(b)
        mx = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        out[c] = {"max_abs_window_diff": mx, "nan_mismatch": nm}
        ok = ok and mx == 0.0 and nm == 0
    return {"pass": bool(ok), "n_windows": int(m.sum()), "must_read": "exactly 0.0 on every cell", "cells": out}


def outcome(f_ho, d1, zuse, guard_cost, n1dx, route_cost=None, f_ho_live=None):
    """Card section 7, mechanically.  d1 {seed: row or None}, zuse {seed: bool or None}, guard_cost {seed: set of cost cells or None},
    n1dx row or None, route_cost bool or None (None = STRATEGIC NOT-YET).  -> {status, missing, labels, labels_if_N1Dx_ignored, detail}.  Pure."""
    bar = bar_of(f_ho, *( [f_ho_live] if f_ho_live is not None else []))
    miss = []
    for s in (0, 1):
        if d1.get(s) is None:
            miss.append(f"D1-HO N1D_s{s}")
        if zuse.get(s) is None:
            miss.append(f"Z-USE N1D_s{s}")
        if guard_cost.get(s) is None:
            miss.append(f"guard eval-141 N1D_s{s}")
    if n1dx is None:
        miss.append("N1Dx_s0")
    det = {"bar": bar, "F_HO": float(f_ho), "bar_rule": "max(0.0383, F_HO)", "route_cost": route_cost}
    if any(d1.get(s) is None for s in (0, 1)):
        return {"status": "NOT-YET", "missing": miss, "labels": ["NOT-YET"], "labels_if_N1Dx_ignored": [], "detail": det}
    gain = [bool(d1[s]["gain"]) for s in (0, 1)]
    cost = [bool(d1[s]["cost"]) for s in (0, 1)]
    delta = [float(d1[s]["delta"]) for s in (0, 1)]
    zs = [zuse.get(s) for s in (0, 1)]
    gc = [guard_cost.get(s) for s in (0, 1)]
    guard_known = all(x is not None for x in gc)
    both_cost = sorted(gc[0] & gc[1]) if guard_known else []
    guard_ok = bool(guard_known and not gc[0] and not gc[1] and route_cost is False)
    det.update({"d1_gain_by_seed": gain, "d1_cost_by_seed": cost, "d1_delta_by_seed": delta, "z_use_separated_by_seed": zs, "guard_cost_cells_by_seed": [sorted(x) if x is not None else None for x in gc],
                "guard_cost_cells_both_seeds": both_cost, "guard_holds": guard_ok, "N1Dx_gain": None if n1dx is None else bool(n1dx["gain"])})
    labels = []
    zk = all(z is not None for z in zs)
    gain_label = None
    if all(gain) and zk and all(zs) and guard_known and not gc[0] and not gc[1]:
        gain_label = "RBD-GAIN" if route_cost is False else ("RBD-GAIN-PENDING-STRATEGIC" if route_cost is None else None)
    if gain_label and n1dx is not None and not n1dx["gain"]:
        labels.append(gain_label)
    if sum(gain) == 1:
        labels.append("RBD-ONE-SEED")
    if zk and all(zs) and not any(gain):
        labels.append("RBD-NO-GAIN")
    if zk and not any(zs):
        labels.append("RBD-UNREAD")
    if all(cost) or (guard_known and both_cost):
        labels.append("RBD-COSTS")
    if f_ho > BAR_NAVRB2 and all(float(f_ho) > d for d in delta):
        labels.append("RBD-FLOOR-HIGH")
    ignored = ([gain_label] if gain_label else []) + list(labels)
    status = "FINAL" if not miss and route_cost is not None else "NOT-YET"
    if n1dx is not None and n1dx["gain"]:
        labels = ["N1DX-REGRESSION-GAINS"]
    if not labels:
        labels = ["NO-COMMITTED-LABEL"]
    return {"status": status, "missing": miss + (["STRATEGIC contrast (ROUTE-COST)"] if route_cost is None else []), "labels": labels, "labels_if_N1Dx_ignored": ignored, "detail": det}


# ------------------------------------------------------------------------------------------------------------------ IO
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def parse_alias(items):
    """['N1D_s0=N1R_s0', ...] -> {'N1D_s0': 'N1R_s0'}: the stand-in dump of an arm is another arm's dump (a code check, never a result)"""
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = v
    return out


def ho_dir(arm, alias=None):
    arm = (alias or {}).get(arm, arm)
    root = DIRS["n1r_ho"] if arm.startswith("N1R") else DIRS["d_ho"]
    tag = arm[:-1] + "_zero" if (arm.startswith("N1D") and arm.endswith("Z")) else arm      # dump_rbd.sh names the zero variant TAG_zero
    return f"{root}/dump_NAVRBD-HO-{tag}_m10_causal_heads"


def e141_dir(arm, alias=None):
    arm = (alias or {}).get(arm, arm)
    if arm.startswith("N1R"):
        z = arm.endswith("Z")
        return f"{DIRS['n1r_141']}/dump_NAVRB2-{arm}_m10_causal_heads"
    return f"{DIRS['d_141']}/dump_NAVRBD-{arm}_m10_causal_heads"


def card_check():
    got = sha256_file(CARD_PATH)
    return got.startswith(CARD_SHA256_PREFIX), got


def md_table(rows_by_col, cells):
    cols = list(rows_by_col)
    out = ["| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in cells:
        line = []
        for k in cols:
            r = rows_by_col[k].get(c, {})
            line.append("n/a" if r.get("status") not in (None, "OK") or "delta" not in r else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" + ("*" if r.get("separated") else "") + ("G" if r.get("gain") else "") + ("C" if r.get("cost") else ""))
        out.append(f"| {c} | " + " | ".join(line) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--alias", action="append", default=[], help="stand-in: ARM=OTHERARM (dump of OTHERARM read as ARM); a code check, never a result")
    ap.add_argument("--skip-141", action="store_true", help="skip the eval-141 guard cells and Z-EVAL / K-RE (they read NOT-YET)")
    ap.add_argument("--contrast", default=None, help="navrb2_strategic.py style output for N1D - N1R: {'route_cost': {'N1D-N1R': bool}}")
    ap.add_argument("--ho-zero-hist-mask", default=None, help="npy bool [5736]: held-out windows whose channel rows are all 0 (Z-EVAL on the held-out grid)")
    ap.add_argument("--hmax", type=int, default=None, help="cfg.hmax of the N1D checkpoints (default: the value recorded in RESULT_NAVRB2.json)")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    t0 = time.time()
    alias = parse_alias(a.alias)
    import navrb2_lib as NL
    import navrb2_read as N2
    import strat1_lib as L
    import strat1_read as S
    import obd_read as OR
    import obd_geom as G
    CI = L.load_ci()
    ok_card, card_sha = card_check()
    rep = {"tool": "navrbd_read.py", "card": "PREREG_V8F_NAVRBD.md (FROZEN 2026-11-18)", "card_sha256": card_sha, "stand_in_aliases": alias, "reading_choices": READING,
           "tier": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, nav_rb held from t0, c13 dead-reckoned); STRATEGIC T0",
           "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = the floor F_HO (held-out) / F2 (eval-141); inference variance does not apply",
           "limit": "optimistic by construction: the route is supplied along the driven path (card 3)", "committed_rules": __doc__.split("COMMITTED RULES")[1].split("Usage:")[0].strip(), "controls": {"problems": [], "not_yet": []}}
    prob, ny = rep["controls"]["problems"], rep["controls"]["not_yet"]
    if not ok_card:
        prob.append(f"the card sha256 {card_sha[:8]} != {CARD_SHA256_PREFIX}")
    geom = G.build()
    # ---- held-out grid
    have = {arm: os.path.isdir(ho_dir(arm, alias)) for arm in HO_ARMS}
    ho = {}
    for arm in HO_ARMS:
        if have[arm]:
            ho[arm] = S.load_dump_n(ho_dir(arm, alias), HO_EPISODES)
    if "N1R_s0" not in ho or "N1R_s1" not in ho:
        rep["outcome"] = {"status": "NOT-YET", "labels": ["NOT-YET"], "missing": ["N1R_s0 / N1R_s1 held-out dumps"]}
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET: N1R held-out dumps")
        sys.exit(1)
    ref = ho["N1R_s0"]
    eid = ref["eid"]
    if len(eid) != HO_WINDOWS or len(np.unique(eid)) != HO_EPISODES:
        prob.append(f"held-out grid count {len(eid)} / {len(np.unique(eid))} != {HO_WINDOWS} / {HO_EPISODES}")
    for arm, x in ho.items():
        for k in OR.SAME:
            if not np.array_equal(x[k], ref[k]):
                prob.append(f"held-out window / model-free array differs from N1R_s0: {k} in {arm}")
    gl = check_three_class(geom.labels(ref["g"])[0])
    pred = {arm: check_three_class(geom.labels(x["cl"])[0]) for arm, x in ho.items()}
    acc = {arm: kin_acc(gl, p) for arm, p in pred.items()}
    f_live = f_ho_of(acc["N1R_s0"], acc["N1R_s1"])
    banked = json.load(open(F_HO_BANKED_FILE, encoding="utf-8"))
    f_banked = float(banked["F_HO"])
    rep["floor"] = {"F_HO_live": f_live, "F_HO_banked": f_banked, "bar": bar_of(f_banked, f_live), "bar_rule": "max(0.0383, F_HO)", "F_HO_paired_bootstrap_s1_minus_s0": d1_ho(CI, acc["N1R_s1"], acc["N1R_s0"], eid, bar_of(f_banked, f_live), a.n_boot)}
    rep["controls"]["K-FLOOR"] = k_floor(f_live, f_banked)
    if not rep["controls"]["K-FLOOR"]["pass"]:
        prob.append("K-FLOOR: the live F_HO differs from the banked step-0 value")
    rep["controls"]["K-CONST"] = k_const(gl)
    rep["controls"]["K1_self_pair"] = {arm: k1_self(CI, acc[arm], eid, a.n_boot) for arm in ("N1R_s0",) + tuple(x for x in ("N1D_s0",) if x in acc)}
    if not all(v["pass"] for v in rep["controls"]["K1_self_pair"].values()):
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    if not rep["controls"]["K-CONST"]["pass"]:
        prob.append("K-CONST: the majority predictor's accuracy != the majority share")
    bar = bar_of(f_banked, f_live)
    d1, d2, zuse = {}, {}, {}
    rep["gt_class_share"] = {c: float((gl == i).mean()) for i, c in enumerate(CL)}
    for s in (0, 1):
        D, R = d_arm(s), pair_ref(s)
        if D in acc:
            d1[s] = d1_ho(CI, acc[D], acc[R], eid, bar, a.n_boot)
            d2[s] = d2_ho(CI, gl, pred[D], pred[R], eid, a.n_boot)
        else:
            d1[s] = None
            ny.append(f"D1-HO {D} (held-out dump missing)")
        if D in acc and D + "Z" in acc:
            zuse[s] = z_use_ho(CI, acc[D], acc[D + "Z"], eid, a.n_boot)
        else:
            zuse[s] = None
            ny.append(f"Z-USE {D} (held-out zero dump missing)")
    n1dx = d1_ho(CI, acc["N1Dx_s0"], acc[pair_ref(0)], eid, bar, a.n_boot) if "N1Dx_s0" in acc else None
    if n1dx is None:
        ny.append("N1Dx_s0 (held-out dump missing)")
    rep["D1_HO"] = {f"N1D_s{s}-N1R_s{s}": d1[s] for s in (0, 1)}
    rep["D1_HO"]["N1Dx_s0-N1R_s0"] = n1dx
    rep["D1_HO_each_acc"] = {arm: float(v.mean()) for arm, v in acc.items()}
    rep["D2_HO_beside"] = {f"N1D_s{s}-N1R_s{s}": d2.get(s) for s in (0, 1)}
    rep["Z_USE_HO"] = {f"N1D_s{s}": zuse[s] for s in (0, 1)}
    if a.ho_zero_hist_mask and os.path.exists(a.ho_zero_hist_mask):
        zm = np.load(a.ho_zero_hist_mask)
        rep["controls"]["Z-EVAL_HO"] = {}
        for s in (0, 1):
            D = d_arm(s)
            if D in acc and D + "Z" in acc:
                rep["controls"]["Z-EVAL_HO"][D] = z_eval({"TAC-LAT": acc[D]}, {"TAC-LAT": acc[D + "Z"]}, zm, ["TAC-LAT"])
                if not rep["controls"]["Z-EVAL_HO"][D]["pass"]:
                    prob.append(f"Z-EVAL held-out {D}: real - zero is not exactly 0.0 on the all-zero-channel windows")
    else:
        ny.append("Z-EVAL on the held-out grid (no --ho-zero-hist-mask); read on eval-141 below")
    # ---- eval-141 guard, Z-EVAL, Z-USE, K-RE
    guard, allw, zu141, route_cost = {0: None, 1: None}, {}, {}, None
    if not a.skip_141:
        d141 = {arm: e141_dir(arm, alias) for arm in ("N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1D_s0Z", "N1D_s1Z", "N1R_s0Z", "N1R_s1Z")}
        bk = lambda arm: f"{DIRS['banked_141']}/dump_STRAT1-{arm}_m10_causal_heads"  # noqa: E731
        c0 = ["C0_s0", "C0_rep", "C0_s10", "C0_s10r", "C0_s11", "C0_s11r"]
        need = [bk(x) for x in c0 + ["N1_s0", "N1_s1"]] + [d141["N1R_s0"], d141["N1R_s1"]]
        if any(not os.path.isdir(p) for p in need):
            ny.append("eval-141 reference dumps missing: " + str([os.path.basename(p) for p in need if not os.path.isdir(p)]))
        else:
            dg = S.load_dump_n(bk("C0_s0"), 141)
            J = geom.join(dg)
            glat, glon = geom.labels(dg["g"])
            e141 = dg["eid"]
            cells, predl = {}, {}
            for arm, d in [(x, bk(x)) for x in c0 + ["N1_s0", "N1_s1"]] + [(k, v) for k, v in d141.items() if os.path.isdir(v)]:
                cells[arm], predl[arm] = N2.load_arm(geom, dg, J, glat, glon, d, 141)
            c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
            F2 = N2.floor2(c0rep["floors"]["F"], c0rep["floors"]["F_rep"])
            rep["floors_eval141"] = {"F": c0rep["floors"]["F"], "F_rep": c0rep["floors"]["F_rep"], "F2": F2, "source": "RESULT_STRAT1_C0REP.json (banked)"}
            # K-RE: N1R - N1 recomputed equals RESULT_NAVRB2.json (24 cells, both seeds)
            b2 = json.load(open(N2R + "/out/RESULT_NAVRB2.json", encoding="utf-8"))
            bad, nre = [], 0
            for s in (0, 1):
                mine = S.slim(S.pair_rows(CI, cells[f"N1R_s{s}"], cells[f"N1_s{s}"], e141, F2, a.n_boot))
                for c, row in b2["pairs"]["N1R-N1"]["all_window_24_cells_under_F2"][str(s)].items():
                    nre += 1
                    for k, v in row.items():
                        if k in mine.get(c, {}) and mine[c][k] != v:
                            bad.append(f"s{s} {c} {k}: {mine[c][k]!r} vs {v!r}")
            rep["controls"]["K-RE"] = {"pass": bool(nre > 0 and not bad), "n_rows_compared": nre, "mismatches": bad[:20], "against": "RESULT_NAVRB2.json (N1R - N1, 24 cells, s0 / s1)"}
            if not rep["controls"]["K-RE"]["pass"]:
                prob.append("K-RE: a recomputed banked pair differs from RESULT_NAVRB2.json (or none was compared)")
            w = NL.rb_windows()
            hmax = int(a.hmax or b2["strata"]["hmax"])                      # the item history rows: cfg.hmax of the N1R checkpoints (16), recorded in RESULT_NAVRB2.json; --hmax overrides
            st = NL.strata(w, hmax)
            rb, zh = st["rb"], st["zero_hist"]
            rep["strata_141"] = {"RB": NL.count_stratum(rb, e141), "zero_hist_Z-EVAL": NL.count_stratum(zh, e141), "hmax": hmax}
            for s in (0, 1):
                D, R = d_arm(s), pair_ref(s)
                if D in cells:
                    allw[s] = S.pair_rows(CI, cells[D], cells[R], e141, F2, a.n_boot)
                    guard[s] = {c for c, v in allw[s].items() if v.get("cost")}
                    rep.setdefault("D1_eval141_beside", {})[f"N1D_s{s}-N1R_s{s}"] = N2.d1_row(CI, cells[D], cells[R], e141, rb, None, a.n_boot)
                else:
                    ny.append(f"guard eval-141 {D} (dump missing)")
                if D in cells and D + "Z" in cells:
                    rep["controls"].setdefault("Z-EVAL", {})[D] = z_eval(cells[D], cells[D + "Z"], zh, list(OR.CELLS))
                    zu141[D] = N2.z_use(CI, cells[D], cells[D + "Z"], e141, rb, a.n_boot)
                    if not rep["controls"]["Z-EVAL"][D]["pass"]:
                        prob.append(f"Z-EVAL {D}: real - zero is not exactly 0.0 on the zero-history windows (n {rep['controls']['Z-EVAL'][D]['n_windows']})")
                else:
                    ny.append(f"Z-EVAL / Z-USE eval-141 {D} (zero dump missing)")
            rep["Z_USE_eval141_beside"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in zu141.items()}
            rep["guard_eval141"] = {"cells_under_F2": {f"N1D_s{s}-N1R_s{s}": S.slim(allw[s]) for s in allw}, "cost_cells": {str(s): sorted(guard[s]) for s in guard if guard[s] is not None},
                                    "gain_cells": {str(s): sorted(c for c, v in allw[s].items() if v.get("gain")) for s in allw}}
    else:
        ny.append("eval-141 guard, Z-EVAL, K-RE (--skip-141)")
    if a.contrast and os.path.exists(a.contrast):
        route_cost = bool(json.load(open(a.contrast, encoding="utf-8")).get("route_cost", {}).get("N1D-N1R"))
    else:
        ny.append("STRATEGIC contrast (ROUTE-COST); fog stays unscored")
    out = outcome(f_banked, d1, {s: (None if zuse[s] is None else zuse[s]["separated"]) for s in (0, 1)}, guard, n1dx, route_cost, f_live)
    if prob:
        out["status"] = "INVALID (control problems; labels unvalidated)"
    rep["outcome"] = out
    rep["next_step"] = {x: NEXT[x] for x in out["labels"] if x in NEXT} or {"none": "no committed label fired: print the rows and name the cheapest next arm (RULE ZERO)"}
    rep["provenance"] = {"card_sha256": card_sha, "grid_sha256": GRID_SHA256, "HO_FLOOR.json": sha256_file(F_HO_BANKED_FILE),
                         "held_out_dump_provenance": {arm: {"dir": ho_dir(arm, alias)} for arm in ho},
                         "reader_sha256": {f: sha256_file(f"{HERE}/{f}") for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".sh", ".cmd"))}}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = [f"# NAVRBD read (MEASURED, T1-kin; D1-HO {HO_WINDOWS} windows / {HO_EPISODES} episodes; B {a.n_boot} seed 0; card sha256 {card_sha[:8]})", "",
          f"Stand-in aliases: {alias or 'none (real dumps)'}", f"Problems: {prob or 'none'}. NOT-YET: {ny or 'none'}.", "", f"## Outcome: {out['labels']} ({out['status']})", "", "```", json.dumps(out, indent=1, default=float), "```", "", "## Next", ""]
    md += [f"* {k}: {v}" for k, v in rep["next_step"].items()] + ["", "## D1-HO (kinematic TAC-LAT; bar " + f"{bar:.6f})", "", "```", json.dumps(rep["D1_HO"], indent=1, default=float), "```", ""]
    if "guard_eval141" in rep:
        cols = rep["guard_eval141"]["cells_under_F2"]
        md += ["## eval-141 guard (T1-kin; * separated, G gain, C cost under F2)", "", md_table(cols, list(OR.CELLS)), ""]
    open(a.md or a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", out["labels"], out["status"], "| problems:", len(prob), "| not-yet:", len(ny), "|", rep["wall_s"], "s")
    print("D1-HO deltas:", {k: (None if v is None else round(v["delta"], 6)) for k, v in rep["D1_HO"].items()}, "bar", round(bar, 6))
    for p in prob:
        print("  PROBLEM:", p)
    for p in ny:
        print("  NOT-YET:", p)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
