#!/usr/bin/env python3
"""Place the NuRec `dynamic_rigids` actors — and FALSIFY the placement before using it.

WHAT THE FILE ACTUALLY STORES (MEASURED, `probe_actors.py`):
  * `dynamic_rigids.positions` are **track-LOCAL**: per-cuboid centroid 0.53 m from the
    origin, mean extent 4.11 x 1.85 x 1.58 m — car-sized. So each cuboid's cloud must be
    rigidly transformed by its track's world pose.
  * `tracks_calib.tracks_delta_{q,t}` are **calibration deltas, not poses**: |Δt|max =
    1.6 cm and row 0 is the identity quaternion. The name was suggestive; the magnitude
    settled it. The base poses therefore live OUTSIDE `volume.nurec`.
  * They live in `sequence_tracks.json` inside the scene USDZ: 78 tracks with
    `tracks_poses` (x,y,z,qx,qy,qz,qw) and `tracks_timestamps_us`.
  * `gaussian_cuboid_ids` indexes the layer's own 35 tracks (`timestamps_us_ranges`
    [35,2]), NOT the 78-track list — so a mapping is required.

THE MAPPING, AND WHY IT IS CHECKABLE: each layer track carries its [t_start, t_end]; each
sequence track carries its own annotation timestamps. Matching on that interval is a
1-D assignment with a natural discriminant — we require the best match to beat the
runner-up by a margin, and we report every residual. A mapping that cannot separate its
candidates is refused rather than used.

THE FALSIFIER: `falsify_actors` renders frame 0 with the actors ON and OFF and scores
both against the scene's own reference video with **gradient-NCC** (FINDINGS: PSNR and
plain NCC are retracted on this night clip — both rank a wrong frame first). Correct
actor placement must IMPROVE grad-NCC; a wrong one adds car-shaped noise and hurts it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def layer_track_ranges(renderer, layer="dynamic_rigids"):
    key = f".gaussians_nodes.{layer}.time_embed.timestamps_us_ranges"
    return np.frombuffer(renderer.scene.sd[key], np.int64).reshape(-1, 2).astype(np.float64)


def build_cuboid_to_track(renderer, tracks, layer="dynamic_rigids", margin_us=2.0e5):
    """cuboid id -> index into `tracks`, matched on the annotation time interval."""
    rr = layer_track_ranges(renderer, layer)
    cand = []
    for i in range(len(tracks)):
        tr = tracks.time_range(i)
        cand.append(tr)
    mapping, report = {}, []
    used = set()
    for c in range(rr.shape[0]):
        t0, t1 = rr[c]
        costs = []
        for i, tr in enumerate(cand):
            if tr is None:
                costs.append((np.inf, i))
                continue
            costs.append((abs(tr[0] - t0) + abs(tr[1] - t1), i))
        costs.sort()
        best, second = costs[0], (costs[1] if len(costs) > 1 else (np.inf, -1))
        ok = np.isfinite(best[0]) and (second[0] - best[0]) > margin_us and best[1] not in used
        if ok:
            mapping[c] = best[1]
            used.add(best[1])
        report.append({"cuboid": c, "range": [float(t0), float(t1)],
                       "best_track": int(best[1]), "best_cost_us": float(best[0]),
                       "runner_up_cost_us": float(second[0]), "accepted": bool(ok),
                       "track_id": tracks.ids[best[1]],
                       "label": (tracks.labels[best[1]] if tracks.labels else None)})
    return mapping, report


def falsify_actors(renderer, scene_dir, frame=0, layer="dynamic_rigids"):
    """Does switching the actors ON make the render MORE like the reference? Measured."""
    from gsplat_renderer import grad_ncc, read_ref_frame
    sd = Path(scene_dir)
    mp4 = sd / "camera_front_wide_120fov.mp4"
    c2n = renderer.gt_cam_to_nre(frame)
    ts = renderer.frame_timestamps_us(frame)[1]
    ref = read_ref_frame(mp4, frame, (renderer.width, renderer.height))
    saved = renderer._actor
    renderer._actor = None
    off, _, _ = renderer.render(c2n)
    renderer._actor = saved
    on, _, _ = renderer.render(c2n, actor_time_us=ts)
    g_off, g_on = grad_ncc(off, ref), grad_ncc(on, ref)
    # negative control: the same actors placed at a WRONG time. If "on" only wins
    # because more gaussians is always better, the wrong-time render wins too.
    span = renderer.t_hi - renderer.t_lo
    wrong, _, _ = renderer.render(c2n, actor_time_us=ts + 0.45 * span)
    g_wrong = grad_ncc(wrong, ref)
    diff = int((np.abs(on.astype(np.int32) - off.astype(np.int32)) > 8).sum())
    return {"frame": frame,
            "grad_ncc_actors_off": round(g_off, 4),
            "grad_ncc_actors_on": round(g_on, 4),
            "grad_ncc_actors_on_WRONG_TIME": round(g_wrong, 4),
            "delta_on_minus_off": round(g_on - g_off, 4),
            "delta_on_minus_wrongtime": round(g_on - g_wrong, 4),
            "pixels_changed_by_actors": diff,
            "pass": bool(g_on > g_off and g_on > g_wrong)}


def attach_actors_verified(renderer, scene_dir, layer="dynamic_rigids", frames=(0, 60, 120)):
    """Build the mapping, attach it, and refuse to proceed if the falsifier says no."""
    from gsplat_renderer import ActorTracks
    sd = Path(scene_dir)
    st = sd / "extracted" / "sequence_tracks.json"
    if not st.exists():
        raise FileNotFoundError(
            f"{st} missing — extract it from the scene USDZ first (it is a plain zip "
            "entry: sequence_tracks.json)")
    tracks = ActorTracks(st)
    mapping, report = build_cuboid_to_track(renderer, tracks, layer)
    renderer.attach_actors(tracks, mapping, layer)
    falsi = [falsify_actors(renderer, sd, f, layer) for f in frames]
    n_pass = sum(1 for f in falsi if f["pass"])
    info = {"n_tracks_json": len(tracks), "n_layer_tracks": len(report),
            "n_mapped": len(mapping), "falsifier": falsi,
            "falsifier_pass_frames": n_pass, "falsifier_n_frames": len(falsi),
            "verdict": ("ACCEPTED" if n_pass >= (len(falsi) + 1) // 2 else "REFUSED"),
            "per_track": report}
    if info["verdict"] != "ACCEPTED":
        renderer._actor = None
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", default="/tmp/actor_map.json")
    a = ap.parse_args()
    from gsplat_renderer import NuRecGsplatRenderer
    r = NuRecGsplatRenderer(Path(a.scene_dir).expanduser())
    info = attach_actors_verified(r, Path(a.scene_dir).expanduser())
    Path(a.out).write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "per_track"}, indent=2))
    print("accepted cuboids:", sum(1 for x in info["per_track"] if x["accepted"]),
          "/", len(info["per_track"]))
