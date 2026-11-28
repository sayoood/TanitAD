#!/usr/bin/env python3
"""Read the nav manipulation sweep and separate the three candidate mechanisms.

  (a) THE NAV INPUT IS IGNORED       -> the plan does not move under the sweep.
  (b) NAV IS USED, TURN OUT OF REACH -> it moves, but no candidate spans the required
                                        curvature.
  (c) NAV IS USED AND REACHABLE      -> the right candidate exists and loses selection.

The discriminator is NOT one number. A single "does it move" statistic cannot tell a
LATERAL response from a LONGITUDINAL one, and this sweep's headline is exactly that
distinction — so every delta is reported per CHANNEL (`lat` = ego +y = left, `lon` = ego
+x = forward), never pooled into a distance.

⚠️ THE NOISE FLOOR IS REPORTED NEXT TO EVERY EFFECT. `repro` is the disagreement between
the re-run plan at the BANKED nav and the banked plan itself — i.e. what a JPEG q90
round-trip of the observation does to the 2 s waypoint. It is not the sweep's own noise
(the sweep is deterministic and that is asserted), it is the scale of this model's
sensitivity to an input change a human cannot see. An effect below it is real but is not
a control authority.

Intervals: `taniteval.ci.paired_episode_cluster_bootstrap`, clusters = the openloop run's
own 9 DISJOINT contiguous segments of the clip (`cluster_note` in the rollout payload).
⚠️ Nine segments of ONE clip is a weaker unit than 40 independent episodes; that travels
with every interval here and is not laundered into an "episode bootstrap".
⛔ `overlapping_holdout_se` is never used — it biases the point estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAV_NAMES = ("follow", "left", "right", "straight")
FOLLOW, LEFT, RIGHT, STRAIGHT = "0", "1", "2", "3"
TURN_LAT_M = 2.0          # |GT lateral @2s| above which the tick is "in the turn"


def _ci():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval"))
    from taniteval.ci import (episode_cluster_bootstrap,          # noqa: E402
                              paired_episode_cluster_bootstrap)
    return episode_cluster_bootstrap, paired_episode_cluster_bootstrap


def segments(rollout_json: Path):
    """k -> cluster id, using the run's OWN segment definition."""
    r = json.loads(Path(rollout_json).read_text())
    out = {}
    for i, seg in enumerate(r["rollouts"]):
        for s in seg["steps"]:
            out[int(s["k"])] = i
    return out


def wp(t, nav, i=3):
    return np.asarray(t["sweep"][nav]["traj"][i], float)


def analyse(sweep_path: Path, rollout_path: Path) -> dict:
    boot1, boot2 = _ci()
    d = json.loads(Path(sweep_path).read_text())
    seg = segments(rollout_path)
    T = d["ticks"]
    eid = np.array([seg.get(t["k"], -1) for t in T])

    # --- the noise floor: JPEG round-trip at the BANKED nav -------------------------
    repro = np.array([np.linalg.norm(np.asarray(t["plan_banked"][3], float)
                                     - wp(t, str(t["nav_banked"]))) for t in T])

    lat = {nv: np.array([wp(t, nv)[1] for t in T]) for nv in (FOLLOW, LEFT, RIGHT,
                                                              STRAIGHT)}
    lon = {nv: np.array([wp(t, nv)[0] for t in T]) for nv in (FOLLOW, LEFT, RIGHT,
                                                              STRAIGHT)}
    gt = np.array([[np.nan, np.nan] if t["gt_ego_wp"] is None
                   else t["gt_ego_wp"][3] for t in T], float)
    v = np.array([t["v"] for t in T])
    in_turn = np.abs(gt[:, 1]) >= TURN_LAT_M

    res = {"arm": d["arm"], "condition": d["condition"], "scene": d["scene"],
           "ckpt": d["ckpt"], "step": d["step"], "n_ticks": len(T),
           "n_clusters": int(len(set(eid.tolist()))),
           "dropped_ticks": d["dropped_ticks"],
           "has_trajectory_fan": d["anchors"] is not None,
           "repro_jpeg_2s_m": {"median": round(float(np.median(repro)), 4),
                               "p90": round(float(np.percentile(repro, 90)), 4),
                               "max": round(float(repro.max()), 4)},
           "nav_banked_hist": {NAV_NAMES[i]: int(sum(1 for t in T
                                                     if t["nav_banked"] == i))
                               for i in range(4)},
           "nav_valid_true": int(sum(1 for t in T if t["nav_valid"])),
           "n_in_turn": int(in_turn.sum()),
           "turn_definition": f"|GT lateral @2 s| >= {TURN_LAT_M} m"}

    # --- (a) does the plan move at all? --------------------------------------------
    def blk(mask, tag):
        m = mask if mask is not None else np.ones(len(T), bool)
        e = eid[m]
        o = {}
        for nv, nm in ((LEFT, "left"), (RIGHT, "right")):
            o[f"dlat_{nm}_minus_follow"] = boot2(lat[nv][m], lat[FOLLOW][m], e)
            o[f"dlon_{nm}_minus_follow"] = boot2(lon[nv][m], lon[FOLLOW][m], e)
        # the LATERAL SEPARATION a route command must produce: left must plan
        # further left than right. This is the single number that says whether nav
        # has lateral authority at all.
        o["lateral_separation_left_minus_right"] = boot2(lat[LEFT][m], lat[RIGHT][m], e)
        o["longitudinal_separation_left_minus_right"] = boot2(lon[LEFT][m],
                                                              lon[RIGHT][m], e)
        # sign coherence — a rate, not a mean, so one huge tick cannot carry it
        o["sign_coherence"] = {
            "left_plans_more_left_than_follow":
                round(float((lat[LEFT][m] > lat[FOLLOW][m]).mean()), 4),
            "right_plans_more_right_than_follow":
                round(float((lat[RIGHT][m] < lat[FOLLOW][m]).mean()), 4),
            "left_plans_more_left_than_right":
                round(float((lat[LEFT][m] > lat[RIGHT][m]).mean()), 4),
            "n": int(m.sum())}
        # magnitude of the response per channel, and the ratio that names the defect
        ml = float(np.abs(lat[LEFT][m] - lat[FOLLOW][m]).mean())
        mo = float(np.abs(lon[LEFT][m] - lon[FOLLOW][m]).mean())
        o["mean_abs_response_m"] = {"lateral": round(ml, 4), "longitudinal": round(mo, 4),
                                    "lon_over_lat": round(mo / max(ml, 1e-9), 3)}
        o["response_above_jpeg_floor_rate"] = round(float(
            (np.abs(lat[LEFT][m] - lat[RIGHT][m]) > np.median(repro)).mean()), 4)
        return {tag: o}

    res.update(blk(None, "all_ticks"))
    if in_turn.sum() >= 3:
        res.update(blk(in_turn, "in_turn_only"))

    # --- (b) reach: is the REQUIRED lateral inside what the arm can emit? -----------
    ok = np.isfinite(gt[:, 1])
    best = np.max(np.stack([lat[FOLLOW], lat[LEFT], lat[RIGHT]]), 0)
    res["reach"] = {
        "n_scored": int(ok.sum()),
        "gt_lat_2s_max_m": round(float(np.nanmax(gt[:, 1])), 4),
        "best_over_nav_lat_2s_max_m": round(float(best[ok].max()), 4),
        "deficit_at_gt_peak_m": round(float(np.nanmax(gt[:, 1])
                                            - best[ok][np.nanargmax(gt[ok, 1])]), 4),
        "note": "best_over_nav is the most-left 2 s waypoint ANY in-vocabulary nav "
                "produced at that tick — the arm's realised lateral reach, not a "
                "candidate set (see has_trajectory_fan)."}
    if in_turn.sum() >= 3:
        res["reach"]["in_turn_mean_gt_lat_m"] = round(float(gt[in_turn, 1].mean()), 4)
        res["reach"]["in_turn_mean_best_lat_m"] = round(float(best[in_turn].mean()), 4)
        res["reach"]["in_turn_lat_shortfall"] = boot2(gt[in_turn, 1], best[in_turn],
                                                      eid[in_turn])

    # --- (c) selection: only meaningful where a fan exists --------------------------
    if d["anchors"] is not None:
        A = np.asarray(d["anchors"], float)               # [N, S, 2]
        res["fan"] = {"n_anchors": int(A.shape[0]),
                      "lat_2s_min_m": round(float(A[:, 3, 1].min()), 4),
                      "lat_2s_max_m": round(float(A[:, 3, 1].max()), 4),
                      "lon_2s_min_m": round(float(A[:, 3, 0].min()), 4),
                      "lon_2s_max_m": round(float(A[:, 3, 0].max()), 4)}
        if ok.sum():
            # rank of the anchor NEAREST the GT 2 s point, under each nav
            ranks = {}
            for nv in (FOLLOW, LEFT, RIGHT):
                rr = []
                for i, t in enumerate(T):
                    if not ok[i] or "anchor_logits" not in t["sweep"][nv]:
                        continue
                    al = np.asarray(t["sweep"][nv]["anchor_logits"], float)
                    dist = np.linalg.norm(A[:, 3] - gt[i], axis=1)
                    j = int(dist.argmin())
                    rr.append(int((al > al[j]).sum()))       # 0 = the fan picked it
                if rr:
                    ranks[NAV_NAMES[int(nv)]] = {
                        "median_rank_of_nearest_anchor": float(np.median(rr)),
                        "picked_it_rate": round(float(np.mean(np.array(rr) == 0)), 4),
                        "top5_rate": round(float(np.mean(np.array(rr) < 5)), 4),
                        "n": len(rr)}
            res["fan"]["nearest_gt_anchor_rank"] = ranks
    else:
        res["fan"] = {"n_anchors": 0,
                      "note": "⛔ THIS ARM HAS NO TRAJECTORY FAN. `anchor_decoder is "
                              "None`; the 2 s waypoint is a single unimodal linear head "
                              "(`wp_heads`) off one summary token. Mechanism (b) as "
                              "posed — 'the candidate set does not span the curvature' "
                              "— is NOT APPLICABLE: there is no candidate set."}

    # --- speed split (does the nav response depend on how fast we are going?) -------
    res["by_speed"] = {}
    for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 99.0)):
        m = (v >= lo) & (v < hi)
        if m.sum() >= 3:
            res["by_speed"][f"{lo:g}-{hi:g}"] = {
                "n": int(m.sum()),
                "mean_dlat_left_minus_right":
                    round(float((lat[LEFT][m] - lat[RIGHT][m]).mean()), 4),
                "mean_dlon_left_minus_right":
                    round(float((lon[LEFT][m] - lon[RIGHT][m]).mean()), 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True,
                    help="sweep.json=rollout.json, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = []
    for spec in args.sweep:
        s, _, r = spec.partition("=")
        out.append(analyse(Path(s), Path(r)))
        print(json.dumps(out[-1], indent=1)[:4000])
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
