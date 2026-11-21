#!/usr/bin/env python
"""Zero-GPU: what the banked goal-head decisions actually look like.

Reads the `decisions/ep*.npz` of a refav1 dump and reports, for the LATERAL
goal head:

  * the decoded-token histogram (the thing the brief quotes: LANE_KEEP 244 ...)
  * the v7.2 label histogram on the same windows
  * the full confusion  label x decoded
  * turn RECALL and turn PRECISION and direction accuracy, as reproduced from
    the tokens (control: must match the banked 0.205 / 0.778)

⛔ No weights, no model, no GPU. Arithmetic on already-banked integers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

LAT = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
       "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
# ⛔ THIS TUPLE WAS SHORT BY TWO AND SILENTLY DROPPED 133 OF 282 WINDOWS.
# MEASURED 2026-09-05: v7.0's lon vocabulary has EIGHT tokens; this listed six,
# so `hist()` (which iterates `enumerate(names)`) never counted indices 6 and 7
# and `hist_goal_lon` in raw/confusion_cos_ext.json read
# {CRUISE:116, BRAKE_TO:33} = 149 instead of 282 — a well-formed-looking
# histogram missing ADAPT_SPEED_FOR_CURVE (78) and ACCELERATE (55). The banked
# dump's own goal_lon_cl carries all four, and the logits argmax agrees with it
# 282/282. ⇒ never hardcode a vocabulary that the model owns.
LON = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
       "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
# the lateral tokens that MOVE the car sideways -- the "turn" set of the brief
TURN_TOKENS = {"LANE_CHANGE_L", "LANE_CHANGE_R", "NUDGE_L", "NUDGE_R",
               "TURN_L", "TURN_R"}
LEFT_TOKENS = {"LANE_CHANGE_L", "NUDGE_L", "TURN_L"}
RIGHT_TOKENS = {"LANE_CHANGE_R", "NUDGE_R", "TURN_R"}
IGNORE = -100


def load(dump: str) -> dict:
    files = sorted(glob.glob(os.path.join(dump, "decisions", "ep*.npz")))
    if not files:
        raise SystemExit(f"no decisions/ep*.npz under {dump}")
    acc: dict[str, list] = {}
    eid: list[int] = []
    for i, f in enumerate(files):
        d = np.load(f)
        n = len(d["ws"])
        for k in d.files:
            acc.setdefault(k, []).append(d[k])
        eid.append(np.full(n, i))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["_eid"] = np.concatenate(eid)
    out["_n_ep"] = len(files)
    return out


def hist(idx: np.ndarray, names) -> dict:
    """⛔ ACCOUNTS FOR EVERY ELEMENT, BY ASSERTION.

    The bug this guard ends: `names` shorter than the model's vocabulary made
    this function skip the tail classes and return a histogram that still
    looked well-formed — 149 of 282 windows, with nothing to say the other 133
    had been dropped. A count that silently omits part of its input is worse
    than a crash, because it is quotable. The totals must reconcile."""
    h = {}
    for i, nm in enumerate(names):
        c = int((idx == i).sum())
        if c:
            h[nm] = c
    n_ign = int((idx == IGNORE).sum())
    if n_ign:
        h["<ignore>"] = n_ign
    counted = sum(h.values())
    if counted != int(idx.size):
        missing = sorted(set(int(v) for v in np.unique(idx)
                             if v != IGNORE and int(v) >= len(names)))
        raise ValueError(
            f"hist() accounted for {counted} of {idx.size} entries; "
            f"vocabulary has {len(names)} names but the data carries "
            f"index/indices {missing}. Fix the vocabulary — do NOT relax "
            f"this check; a partial histogram reads as a complete one.")
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = load(a.dump)
    n = len(d["ws"])
    print(f"windows n = {n}   episodes = {d['_n_ep']}")
    print(f"keys: {sorted(k for k in d if not k.startswith('_'))}\n")

    goal = d["goal_lat_cl"]
    pred = d["lat_pred_nav_true"]
    lab = d["lat_label"]

    print("== LATERAL token histograms ==")
    print(f"  decoded GOAL  (goal_lat_cl)      : {hist(goal, LAT)}")
    print(f"  declared HEAD (lat_pred_nav_true): {hist(pred, LAT)}")
    print(f"  v7.2 LABEL    (lat_label)        : {hist(lab, LAT)}")
    same = int((goal == pred).sum())
    print(f"  goal == declared head argmax     : {same}/{n} "
          f"= {same / n:.4f}   (1.0 = one decode path)\n")

    print(f"  LON decoded goal : {hist(d['goal_lon_cl'], LON)}")
    print(f"  LON label        : {hist(d['lon_label'], LON)}\n")

    # ---- confusion, label x decoded, on labelled windows -------------------
    ok = lab != IGNORE
    print(f"== CONFUSION (label rows x decoded cols), labelled n = {int(ok.sum())} ==")
    hdr = "  " + "label\\dec".ljust(15) + "".join(t[:9].rjust(10) for t in LAT)
    print(hdr)
    conf = np.zeros((len(LAT), len(LAT)), dtype=int)
    for li in range(len(LAT)):
        row = ok & (lab == li)
        for gi in range(len(LAT)):
            conf[li, gi] = int((row & (goal == gi)).sum())
        if row.sum():
            print("  " + LAT[li].ljust(15)
                  + "".join(str(conf[li, gi]).rjust(10) for gi in range(len(LAT)))
                  + f"   | n={int(row.sum())}")
    print()

    # ---- recall / precision on the TURN set --------------------------------
    lab_turn = np.array([lab[i] != IGNORE and LAT[lab[i]] in TURN_TOKENS
                         for i in range(n)])
    goal_turn = np.array([LAT[goal[i]] in TURN_TOKENS for i in range(n)])

    def sgn(idx, arr):
        s = []
        for i in idx:
            t = LAT[arr[i]]
            s.append(-1 if t in LEFT_TOKENS else (1 if t in RIGHT_TOKENS else 0))
        return np.array(s)

    both = np.where(lab_turn & goal_turn)[0]
    dir_ok = int((sgn(both, lab) == sgn(both, goal)).sum())
    print("== TOKEN-SPACE recall / precision (v7.2 label as truth) ==")
    print(f"  label turns                        : {int(lab_turn.sum())}/{int(ok.sum())}"
          f" = {lab_turn.sum() / max(ok.sum(), 1):.4f}")
    print(f"  goal proposes a turn               : {int(goal_turn.sum())}/{n}"
          f" = {goal_turn.sum() / n:.4f}")
    print(f"  RECALL  goal turns | label turns   : "
          f"{int((lab_turn & goal_turn).sum())}/{int(lab_turn.sum())}"
          f" = {(lab_turn & goal_turn).sum() / max(lab_turn.sum(), 1):.4f}")
    print(f"  PRECIS. label turns | goal turns   : "
          f"{int((lab_turn & goal_turn).sum())}/{int(goal_turn.sum())}"
          f" = {(lab_turn & goal_turn).sum() / max(goal_turn.sum(), 1):.4f}")
    print(f"  DIRECTION correct | both turn      : {dir_ok}/{len(both)}"
          f" = {dir_ok / max(len(both), 1):.4f}\n")

    # ---- the head's own accuracy (what a fine-tune would target) -----------
    acc = int((pred[ok] == lab[ok]).sum())
    print(f"== HEAD accuracy vs v7.2 label ==  {acc}/{int(ok.sum())}"
          f" = {acc / max(ok.sum(), 1):.4f}")
    for t in ("TURN_L", "TURN_R"):
        li = LAT.index(t)
        m = ok & (lab == li)
        if m.sum():
            print(f"   per-class recall {t:<14}: "
                  f"{int((pred[m] == li).sum())}/{int(m.sum())}")
    print()

    rec = {
        "dump": a.dump, "n_windows": int(n), "n_episodes": int(d["_n_ep"]),
        "n_labelled": int(ok.sum()),
        "hist_goal_lat": hist(goal, LAT), "hist_pred_lat": hist(pred, LAT),
        "hist_label_lat": hist(lab, LAT),
        "hist_goal_lon": hist(d["goal_lon_cl"], LON),
        "hist_label_lon": hist(d["lon_label"], LON),
        "goal_eq_head_frac": same / n,
        "confusion_label_x_decoded": conf.tolist(),
        "lat_vocab": list(LAT),
        "token_recall": float((lab_turn & goal_turn).sum() / max(lab_turn.sum(), 1)),
        "token_precision": float((lab_turn & goal_turn).sum() / max(goal_turn.sum(), 1)),
        "direction_acc": float(dir_ok / max(len(both), 1)),
        "n_both_turn": int(len(both)),
        "head_acc": float(acc / max(ok.sum(), 1)),
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
