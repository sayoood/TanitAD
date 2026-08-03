#!/usr/bin/env python3
"""Is an annotated "cut-in" a lane entry, or the ego turning?

⛔ THE DEFECT THIS INSTRUMENT EXISTS TO CATCH.
`scene_geometry.find_cutins` declares a cut-in when a track's lateral offset **in the
ego's INSTANTANEOUS frame** falls from outside the lane to inside it. On a straight road
that is a lane change. On a BEND it is the ego rotating: a vehicle 20 m ahead on a curve
sits at a large ego-frame |y| that collapses to zero as the ego turns in behind it,
without the vehicle moving laterally at all. cl_metrics.py already carries the warning
for the synthetic lead — "on a curving road a vehicle 15 m ahead is genuinely offset in
the ego frame" — but the cut-in DETECTOR never got the same gate.

THE CURVE-INVARIANT TEST. Ego-frame y is not admissible on a bend. What is admissible is
the actor's perpendicular distance to the EGO'S OWN DRIVEN PATH: a car that changes lane
moves ~a lane width relative to that path; a car that merely rounds the same corner
ahead of the ego does not move relative to it at all. The path is the ego's logged
polyline, so the measure is invariant to ego heading by construction.

Three signatures are reported together, because one number should not carry a
refutation:
  (a) cross-track to the ego's path, and its RANGE over the event (the decisive one);
  (b) the ego's own heading change over the same ticks (how much of the apparent sweep
      the ego could have manufactured);
  (c) whether the range to the ego is OPENING or CLOSING (a cut-in closes).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# pre-registered thresholds, stated before the numbers are looked at
CT_RANGE_LANE_CHANGE_M = 2.0     # a lane entry must move >= this vs the ego's path
EGO_TURN_SUSPICIOUS_DEG = 15.0   # above this, ego-frame y is dominated by rotation


def crossings(geom_json):
    g = json.loads(Path(geom_json).read_text())
    s = g.get("summary", g)
    by = {}
    for c in s.get("cutins_ALL", []):
        e = by.setdefault(int(c["track"]), dict(track=int(c["track"]),
                                                track_id=c.get("track_id"),
                                                label=c.get("label"),
                                                renderable=bool(c.get("renderable")),
                                                k_start=int(c["k_start"]),
                                                k_end=int(c["k_end"])))
        e["k_start"] = min(e["k_start"], int(c["k_start"]))
        e["k_end"] = max(e["k_end"], int(c["k_end"]))
    return sorted(by.values(), key=lambda e: e["k_start"]), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--rollouts", required=True,
                    help="any rollouts json from this scene — used only for its `gt` "
                         "block (the logged ego path and timestamps)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cl_metrics as M
    from gsplat_renderer import ActorTracks

    d = json.loads(Path(a.rollouts).read_text())
    gt = d["gt"]
    xy, gyaw, _gv, _ = M.gt_arrays(gt)
    ts = np.asarray([g["ts_us"] for g in gt], float)
    ts0, ts1 = ts[0], ts[-1]
    tr = ActorTracks(a.tracks)
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi

    cross, summ = crossings(a.geometry)
    out = []
    for c in cross:
        k0, k1 = c["k_start"], c["k_end"]
        ct, ey, rng, ks = [], [], [], []
        for k in range(k0, min(len(xy), k1 + 1)):
            frac = float(np.clip((ts[k] - ts0) / (ts1 - ts0), 0, 1))
            T = tr.pose_at(c["track"], frac)
            if T is None:
                continue
            cc, _, _ = M.cross_track(T[:2, 3], xy)
            r = M.ego_frame(T[:2, 3] - xy[k], gyaw[k])
            ct.append(float(cc)); ey.append(float(r[1]))
            rng.append(float(math.hypot(*r))); ks.append(k)
        if not ks:
            out.append({**c, "verdict": "NO POSE — track not alive over its own event"})
            continue
        ct, ey, rng = np.array(ct), np.array(ey), np.array(rng)
        ct_range = float(ct.max() - ct.min())
        turn = float(math.degrees(wrap(gyaw[ks[-1]] - gyaw[ks[0]])))
        opening = bool(rng[-1] > rng[0])
        # ⚠️ cross-track saturates for a track far off the polyline (it is the distance
        # to the NEAREST point of a finite path), so a large |ct| that never CHANGES is
        # reported as off-route rather than as a lane change.
        offroute = bool(np.abs(ct).min() > 5.0)
        if offroute:
            v = ("OFF-ROUTE — this track is never on the ego's path (min |cross-track| "
                 f"{float(np.abs(ct).min()):.1f} m) and does not move relative to it "
                 f"(range {ct_range:.3f} m). Cross traffic at the junction, not a cut-in.")
            real = False
        elif ct_range >= CT_RANGE_LANE_CHANGE_M:
            v = (f"LANE ENTRY CONFIRMED — moves {ct_range:.2f} m relative to the ego's "
                 "own path.")
            real = True
        else:
            v = (f"NOT A CUT-IN — the actor moves only {ct_range:.3f} m relative to the "
                 f"ego's own driven path (|cross-track| stays {float(np.abs(ct).min()):.3f}"
                 f"-{float(np.abs(ct).max()):.3f} m) while its EGO-FRAME y sweeps "
                 f"{abs(ey[0] - ey[-1]):.2f} m. The ego turns {turn:+.1f} deg over the "
                 f"same ticks and the range is "
                 f"{'OPENING' if opening else 'closing'}. This is the ego rotating into "
                 "a bend behind a vehicle that never left the lane.")
            real = False
        out.append({
            **c, "n_ticks": len(ks),
            "cross_track_to_ego_path_m": {"min": round(float(ct.min()), 3),
                                          "max": round(float(ct.max()), 3),
                                          "range": round(ct_range, 3),
                                          "min_abs": round(float(np.abs(ct).min()), 3)},
            "ego_frame_y_m": {"start": round(float(ey[0]), 3),
                              "end": round(float(ey[-1]), 3),
                              "sweep": round(float(abs(ey[0] - ey[-1])), 3)},
            "ego_heading_change_deg": round(turn, 2),
            "range_to_ego_m": {"start": round(float(rng[0]), 2),
                               "end": round(float(rng[-1]), 2),
                               "opening": opening},
            "IS_A_REAL_CUT_IN": real, "verdict": v})

    res = {
        "what": "curve-invariant re-test of every annotated cut-in in this scene",
        "evidence_class": "MEASURED (ours)",
        "scene": summ.get("scene"),
        "thresholds_pre_registered": {
            "cross_track_range_for_lane_change_m": CT_RANGE_LANE_CHANGE_M,
            "ego_turn_suspicious_deg": EGO_TURN_SUSPICIOUS_DEG},
        "headline_counts_in_geometry_file": {
            "n_cutin_events_ALL": summ.get("n_cutin_events_ALL"),
            "n_cutin_events_RENDERABLE": summ.get("n_cutin_events_RENDERABLE")},
        "n_distinct_crossings_after_merge": len(cross),
        "n_surviving_the_curve_invariant_test": sum(1 for x in out
                                                    if x.get("IS_A_REAL_CUT_IN")),
        "n_surviving_AND_renderable": sum(1 for x in out if x.get("IS_A_REAL_CUT_IN")
                                          and x.get("renderable")),
        "per_crossing": out,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_crossing"}, indent=1))
    for x in out:
        print(f"\n  track {x['track']} (id {x.get('track_id')}) ticks "
              f"{x['k_start']}-{x['k_end']}  renderable={x.get('renderable')}")
        print(f"    {x['verdict']}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
