#!/usr/bin/env python
"""The refav1 TACTICAL DECODER audit — 0 GPU, from BANKED artifacts only.

WHAT THIS IS. The cost-surface panel (2026-09-03, `D-REFAV1-COST-SURFACE`)
established that refav1's planner chases a goal that IS the constant-velocity
rollout, because the tactical decoder emits ``LANE_KEEP`` everywhere. This tool
reads the class-recall table off the two banked step-1,000 dumps and separates
the four candidate mechanisms WITHOUT a forward pass:

  1. MAJORITY-CLASS COLLAPSE  — the head reads the base rate and nothing else.
  2. UNSUPERVISED HEAD        — the term exists but never shaped the weights.
  3. REPRESENTATION           — the tactical latent cannot separate turns.
  4. PLUMBING                 — the decision never reaches the goal.

(1)/(2) are read here; (4) is read here (goal_lat vs lat_pred agreement);
(3) needs `intent_probe.py` (one forward pass).

⛔ CONTROLS, per `TanitAD_ValidateAIDesign` §4, EVERY panel carries its controls:
  * ``constant_only``  — always predict the in-band MAJORITY class. Its accuracy
    MUST read the base rate EXACTLY (assert, not hope), and its macro-recall
    MUST read 1/K exactly for the K classes present.
  * ``shuffled_label`` — labels permuted WITHIN episode-clusters, n_perm draws;
    the model's macro-recall is scored against that null.
  * printed n and the number of classes present.

TIER: T0 — banked open-loop dumps, no closed-loop simulator.
EVIDENCE CLASS: MEASURED (re-read of banked artifacts written 2026-09-03).

INPUTS (all banked; nothing is recomputed):
  <dump>/decisions/ep*.npz   `taniteval/tools/refav1_arm.py` sidecar:
      ws, lat_label, lon_label, route_label (-100 = out-of-band),
      {lat,lon,route}_pred_{nav_true,nav_shuffled,nav_zero},
      goal_{source,lat,lon}_<arm>, nav_cmd, nav_valid
  cost_surface_<arm>.json    for `gt_turn_deg` (the GEOMETRIC "human turns"
      stratum, >= 5 deg over the plan horizon — NOT the v7.2 label).

USAGE
  python decode_audit.py --dump A=<dir> --dump B=<dir> \
      --cost A=<json> --cost B=<json> --labels <s2_labels_v7.2_eval.jsonl.gz> \
      --out decode_audit.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

IGNORE_ID = -100
#: the geometric stratum used by `cost_surface_probe.py` (its `GT_TURN_DEG`).
GT_TURN_DEG = 5.0
#: v7.0 FROZEN vocabulary (`tanitad/models/vocab_v7.py:290/:298`). Hard-copied
#: here ONLY as a fallback: `--vocab-from-stack` imports the real tuples and
#: refuses on any mismatch, so a drift cannot pass silently.
LAT_V7 = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
          "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON_V7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
          "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")


def _md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _check_vocab() -> dict:
    """Second probe on the vocabulary: import the live tuples and compare."""
    try:
        from tanitad.models.v6 import (tactical_lat_actions,
                                       tactical_lon_actions_v)
        lat = tuple(tactical_lat_actions("v7.0"))
        lon = tuple(tactical_lon_actions_v("v7.0"))
    except Exception as e:                       # noqa: BLE001
        return {"source": "hardcopy", "import_error": repr(e),
                "lat": list(LAT_V7), "lon": list(LON_V7)}
    if lat != LAT_V7 or lon != LON_V7:
        raise SystemExit(f"⛔ vocabulary drift: stack says lat={lat} lon={lon}, "
                         f"this tool has lat={LAT_V7} lon={LON_V7}")
    return {"source": "stack-verified", "lat": list(lat), "lon": list(lon)}


# --------------------------------------------------------------------------- #
# banked readers
# --------------------------------------------------------------------------- #
def load_dump(dump_dir: str) -> dict:
    """Concatenate the per-episode sidecars, keeping the episode id per row."""
    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        raise SystemExit(f"⛔ no decisions/ep*.npz under {dump_dir}")
    cols: dict[str, list] = collections.defaultdict(list)
    eid: list[int] = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = int(np.asarray(d["ws"]).size)
        fi = int(os.path.basename(f)[2:5])
        eid += [fi] * n
        for k in d.files:
            a = np.asarray(d[k])
            if a.ndim >= 1 and a.shape[0] == n:
                cols[k].append(a)
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()
           if len(v) == len(files)}
    out["_eid"] = np.asarray(eid, dtype=np.int64)
    out["_n_files"] = len(files)
    return out


def load_gt_turn(cost_json: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.load(open(cost_json, encoding="utf-8"))
    rows = d["rows"]
    return (np.array([r.get("gt_turn_deg", np.nan) for r in rows], float),
            np.array([r.get("ep", -1) for r in rows], np.int64))


def label_census(path: str) -> dict:
    lat = collections.Counter()
    lon = collections.Counter()
    n = 0
    bands = collections.Counter()
    t0s = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            a = r.get("a_tac") or {}
            lat[a.get("lat")] += 1
            lon[a.get("lon")] += 1
            b = (r.get("bands") or {}).get("tactical_s")
            bands[tuple(b) if b else None] += 1
            t0s[r.get("t0_s")] += 1
    return {"path": path, "md5": _md5(path), "n_records": n,
            "lat_counts": dict(lat), "lon_counts": dict(lon),
            "lat_fracs": {k: round(v / n, 6) for k, v in lat.items()},
            "lon_fracs": {k: round(v / n, 6) for k, v in lon.items()},
            "tactical_bands": {str(k): v for k, v in bands.items()},
            "t0_s": {str(k): v for k, v in t0s.items()}}


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(y, p):
        m[int(a), int(b)] += 1
    return m


def recall_table(y, p, names) -> dict:
    k = len(names)
    cm = confusion(y, p, k)
    sup = cm.sum(axis=1)
    hit = np.diag(cm)
    pred_n = cm.sum(axis=0)
    rec = np.where(sup > 0, hit / np.maximum(sup, 1), np.nan)
    prec = np.where(pred_n > 0, hit / np.maximum(pred_n, 1), np.nan)
    present = sup > 0
    return {
        "n": int(sup.sum()),
        "n_classes_present": int(present.sum()),
        "classes_present": [names[i] for i in range(k) if present[i]],
        "support": {names[i]: int(sup[i]) for i in range(k)},
        "predicted": {names[i]: int(pred_n[i]) for i in range(k)},
        "recall": {names[i]: (None if not present[i] else round(float(rec[i]), 6))
                   for i in range(k)},
        "precision": {names[i]: (None if pred_n[i] == 0
                                 else round(float(prec[i]), 6))
                      for i in range(k)},
        "accuracy": round(float(hit.sum() / max(sup.sum(), 1)), 6),
        "macro_recall_over_present": round(float(np.nanmean(rec[present])), 6),
        "confusion": cm.tolist(),
    }


def constant_only(y, names) -> dict:
    """⭐ THE CONTROL THAT MUST READ A KNOWN VALUE EXACTLY.

    Predict the majority class for every row. Accuracy MUST equal the base rate
    of that class; macro-recall over the K present classes MUST equal 1/K.
    Both are asserted, not reported and hoped over."""
    k = len(names)
    cnt = np.bincount(y, minlength=k)
    maj = int(cnt.argmax())
    p = np.full_like(y, maj)
    t = recall_table(y, p, names)
    base = float(cnt[maj] / cnt.sum())
    kp = int((cnt > 0).sum())
    exp_macro = 1.0 / kp
    ok_acc = abs(t["accuracy"] - base) < 1e-12
    ok_macro = abs(t["macro_recall_over_present"] - exp_macro) < 1e-12
    if not (ok_acc and ok_macro):
        raise SystemExit(
            f"⛔ constant-only control did NOT read its known value: "
            f"acc {t['accuracy']} vs base rate {base}; macro "
            f"{t['macro_recall_over_present']} vs 1/K {exp_macro}")
    t.update({"majority_class": names[maj], "base_rate": round(base, 6),
              "expected_macro_recall_1_over_K": round(exp_macro, 6),
              "reads_known_value_exactly": True})
    return t


def shuffled_label_null(y, p, eid, names, n_perm: int, seed: int) -> dict:
    """Permute the LABELS and score the model's own predictions against each.

    ⚠️ TWO NULLS, because the rows are CLUSTERED and the naive one is
    anti-conservative here. The v7.2 label is emitted ONCE PER CLIP, so an
    episode's in-band rows all carry the SAME label (MEASURED: 2 in-band rows
    per episode on this panel). A free permutation therefore breaks a
    dependence the data really has and makes chance look tighter than it is.

      * ``free``    — ``rng.permutation(y)``, every row independent.
      * ``clustered`` — the EPISODE BLOCKS are permuted, each block's labels
        travelling together. This is the honest null for per-clip labels and
        it is the one to quote.

    The null answers: how good does this prediction vector look against a label
    vector with the same marginal and no relation to the input?
    """
    rng = np.random.default_rng(seed)
    eid = np.asarray(eid)
    uniq = np.unique(eid)
    blocks = [np.where(eid == e)[0] for e in uniq]
    obs = recall_table(y, p, names)
    out = {"n_perm": int(n_perm), "n_episode_blocks": int(len(blocks)),
           "observed_accuracy": obs["accuracy"],
           "observed_macro": obs["macro_recall_over_present"]}
    for mode in ("free", "clustered"):
        acc, macro = [], []
        for _ in range(n_perm):
            if mode == "free":
                ys = rng.permutation(y)
            else:
                order = rng.permutation(len(blocks))
                ys = np.empty_like(y)
                for dst, src in zip(blocks, [blocks[i] for i in order]):
                    n = min(len(dst), len(src))
                    ys[dst[:n]] = y[src[:n]]
                    if len(dst) > n:                     # ragged blocks
                        ys[dst[n:]] = y[src[-1]]
            t = recall_table(ys, p, names)
            acc.append(t["accuracy"])
            macro.append(t["macro_recall_over_present"])
        acc, macro = np.asarray(acc), np.asarray(macro)
        out[mode] = {
            "null_accuracy_mean": round(float(acc.mean()), 6),
            "null_accuracy_p95": round(float(np.percentile(acc, 95)), 6),
            "null_macro_mean": round(float(macro.mean()), 6),
            "null_macro_p95": round(float(np.percentile(macro, 95)), 6),
            "p_accuracy_ge_observed":
                round(float((acc >= obs["accuracy"]).mean()), 6),
            "p_macro_ge_observed":
                round(float((macro >= obs["macro_recall_over_present"]).mean()), 6)}
    out["quote"] = "clustered"
    return out


def boot_frac(v, eid, n_boot: int, seed: int) -> dict:
    """Episode-cluster bootstrap on a 0/1 per-window vector (CLAUDE.md: the
    decision-grade interval). Point estimate is the FULL-SET mean."""
    v = np.asarray(v, dtype=np.float64)
    eid = np.asarray(eid)
    uniq = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in uniq}
    rng = np.random.default_rng(seed)
    pt = float(v.mean()) if v.size else float("nan")
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        boots.append(float(v[sel].mean()))
    b = np.asarray(boots)
    return {"point": round(pt, 6), "lo": round(float(np.percentile(b, 2.5)), 6),
            "hi": round(float(np.percentile(b, 97.5)), 6),
            "n": int(v.size), "n_episodes": int(uniq.size), "n_boot": int(n_boot),
            "estimator": "episode-cluster bootstrap (percentile, full-set point)"}


def arm_panel(d: dict, gt_deg, names_lat, names_lon, n_perm, n_boot, seed) -> dict:
    eid = d["_eid"]
    out: dict = {"n_windows": int(eid.size), "n_episodes": int(d["_n_files"]),
                 "window_index_values": sorted(set(int(x) for x in d["ws"])),
                 "nav_valid_frac": round(float(np.asarray(d["nav_valid"]).mean()), 6)}

    # ---- the in-band mask: the ONLY rows that ever carried a gradient ----- #
    lat_y = np.asarray(d["lat_label"])
    lon_y = np.asarray(d["lon_label"])
    m = lat_y != IGNORE_ID
    out["labelled"] = {
        "n_in_band": int(m.sum()),
        "frac_in_band": round(float(m.mean()), 6),
        "in_band_window_indices": sorted(set(int(x) for x in
                                             np.asarray(d["ws"])[m])),
        "lat_and_lon_masks_identical":
            bool((lat_y != IGNORE_ID).sum() == (lon_y != IGNORE_ID).sum()
                 and bool(((lat_y != IGNORE_ID) == (lon_y != IGNORE_ID)).all())),
    }

    for axis, y_all, names in (("lat", lat_y, names_lat),
                               ("lon", lon_y, names_lon)):
        blk: dict = {}
        for cond in ("nav_true", "nav_shuffled", "nav_zero"):
            key = f"{axis}_pred_{cond}"
            if key not in d:
                continue
            p_all = np.asarray(d[key])
            blk[cond] = {
                "prediction_histogram_ALL_windows": {
                    names[i]: int(c) for i, c in
                    enumerate(np.bincount(p_all, minlength=len(names))) if c},
                "n_distinct_predictions_ALL": int(np.unique(p_all).size),
                "in_band": recall_table(y_all[m], p_all[m], names),
            }
            if cond == "nav_true":
                blk[cond]["controls"] = {
                    "constant_only": constant_only(y_all[m], names),
                    "shuffled_label": shuffled_label_null(
                        y_all[m], p_all[m], eid[m], names, n_perm, seed),
                }
                # is the model's prediction vector LITERALLY the constant?
                blk[cond]["is_constant_predictor_ALL_windows"] = \
                    bool(np.unique(p_all).size == 1)
                blk[cond]["constant_value"] = (
                    names[int(p_all[0])] if np.unique(p_all).size == 1 else None)
        out[axis] = blk

    # ---- plumbing: does the decoded token reach the goal? ----------------- #
    plumb = {}
    for arm in ("cl", "cl_navshuf"):
        gk = f"goal_lat_{arm}"
        if gk not in d:
            continue
        gl = np.asarray(d[gk])
        gn = np.asarray(d[f"goal_lon_{arm}"])
        cond = "nav_true" if arm == "cl" else "nav_shuffled"
        pl = np.asarray(d[f"lat_pred_{cond}"])
        pn = np.asarray(d[f"lon_pred_{cond}"])
        ok = gl >= 0
        plumb[arm] = {
            "n_with_goal_action": int(ok.sum()),
            "goal_source_histogram": {int(k): int(v) for k, v in
                                      zip(*np.unique(np.asarray(
                                          d[f"goal_source_{arm}"]),
                                          return_counts=True))},
            "goal_lat_histogram": {names_lat[i]: int(c) for i, c in
                                   enumerate(np.bincount(gl[ok],
                                                         minlength=len(names_lat)))
                                   if c},
            "goal_lon_histogram": {names_lon[i]: int(c) for i, c in
                                   enumerate(np.bincount(gn[ok],
                                                         minlength=len(names_lon)))
                                   if c},
            "goal_lat_equals_head_argmax":
                round(float((gl[ok] == pl[ok]).mean()), 6) if ok.any() else None,
            "goal_lon_equals_head_argmax":
                round(float((gn[ok] == pn[ok]).mean()), 6) if ok.any() else None,
        }
    out["plumbing"] = plumb

    # ---- the GEOMETRIC "human turns" stratum ------------------------------ #
    if gt_deg is not None and gt_deg.size == eid.size:
        turns = np.isfinite(gt_deg) & (gt_deg >= GT_TURN_DEG)
        pl = np.asarray(d["lat_pred_nav_true"])
        turn_ids = [names_lat.index(t) for t in ("TURN_L", "TURN_R")]
        lat_ids = [i for i, n in enumerate(names_lat) if n != "LANE_KEEP"]
        asked_turn = np.isin(pl, turn_ids)
        asked_nonkeep = np.isin(pl, lat_ids)
        out["human_turn_stratum"] = {
            "gt_turn_deg_threshold": GT_TURN_DEG,
            "n_human_turns": int(turns.sum()),
            "n_total": int(turns.size),
            "head_asks_TURN_on_human_turns":
                int(asked_turn[turns].sum()),
            "head_asks_ANY_non_LANE_KEEP_on_human_turns":
                int(asked_nonkeep[turns].sum()),
            "recall_TURN_on_geometric_turns":
                boot_frac(asked_turn[turns].astype(float), eid[turns],
                          n_boot, seed) if turns.any() else None,
            "label_says_TURN_on_geometric_turns": int(
                np.isin(lat_y, turn_ids)[turns & m].sum()),
            "n_geometric_turns_that_are_in_band": int((turns & m).sum()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    help="NAME=<dump dir> (repeatable)")
    ap.add_argument("--cost", action="append", default=[],
                    help="NAME=<cost_surface_*.json> (repeatable)")
    ap.add_argument("--labels", default=None, help="eval labels .jsonl.gz")
    ap.add_argument("--train-labels", default=None, help="train labels .jsonl.gz")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vocab = _check_vocab()
    names_lat, names_lon = vocab["lat"], vocab["lon"]
    dumps = dict(x.split("=", 1) for x in a.dump)
    costs = dict(x.split("=", 1) for x in a.cost)

    res: dict = {
        "tool": "decode_audit.py",
        "tier": "T0",
        "tier_note": ("banked open-loop step-1,000 dumps; no simulator, no "
                      "training. A re-read of artifacts, not a new measurement "
                      "of the world."),
        "evidence_class": "MEASURED",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocabulary": vocab,
        "inputs": {"dumps": dumps, "costs": costs, "labels": a.labels,
                   "train_labels": a.train_labels},
        "estimators": {
            "interval": "episode-cluster bootstrap, percentile, full-set point",
            "null": "label permutation within the in-band panel",
        },
        "arms": {},
    }
    if a.labels:
        res["label_census_eval"] = label_census(a.labels)
    if a.train_labels:
        res["label_census_train"] = label_census(a.train_labels)

    for name, dd in dumps.items():
        d = load_dump(dd)
        gt = None
        if name in costs:
            gt, ep = load_gt_turn(costs[name])
            if gt.size != d["_eid"].size:
                print(f"⚠️  {name}: cost rows {gt.size} != dump rows "
                      f"{d['_eid'].size} — gt_turn stratum SKIPPED",
                      file=sys.stderr)
                gt = None
        res["arms"][name] = arm_panel(d, gt, names_lat, names_lon,
                                      a.n_perm, a.n_boot, a.seed)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1)[:2000])
    for nm, blk in res["arms"].items():
        print(f"\n=== {nm} ===")
        print(f"  in-band {blk['labelled']['n_in_band']}/{blk['n_windows']}")
        for axis in ("lat", "lon"):
            t = blk[axis]["nav_true"]
            print(f"  {axis}: pred_hist={t['prediction_histogram_ALL_windows']}")
            print(f"       in-band acc={t['in_band']['accuracy']} "
                  f"macro={t['in_band']['macro_recall_over_present']} "
                  f"(const-only base={t['controls']['constant_only']['base_rate']} "
                  f"macro={t['controls']['constant_only']['expected_macro_recall_1_over_K']})")
            print(f"       recall={t['in_band']['recall']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
