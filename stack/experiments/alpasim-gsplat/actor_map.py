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

⚠️ CORRECTED 2026-08-03 — THE RELATIVE MARGIN ALONE WAS DISCARDING EXACT MATCHES.
MEASURED on scene `7c72937c`: all **92** layer cuboids match their best sequence track at
`best_cost_us == 0` (an EXACT interval match, µs-precise on both sides) and the 92
best-track assignments are a **bijection** — yet the relative rule `second - best >
margin_us` rejected **31** of them, purely because some other track happened to start
within 200 ms. The rule was written to refuse an AMBIGUOUS assignment; a unique, exactly
zero-cost match is not ambiguous, and the rejections were not protecting anything.

The consequence was not cosmetic: the two vehicles the ego actually follows in that scene
(tracks 117/`id 30` and 80/`id 38`, 135 close-following rows, min headway 3.27 m) and all
three cut-in tracks were declared NON-RENDERABLE, so the scene's real close-following
geometry was invisible to the closed-loop panel.

⇒ acceptance is now `unique AND (exact OR margin)`, where `exact` is `best_cost_us <=
exact_us` (default 1 µs). Both verdicts are recorded per cuboid — `accepted` (this rule)
and `accepted_strict_margin` (the old one) — so no number silently changes meaning, and
`attach_actors_verified(ab_compare=True)` runs the PIXEL falsifier on BOTH mappings so the
change is adjudicated by the reference video rather than by this argument.

THE FALSIFIER: `falsify_actors` renders frame 0 with the actors ON and OFF and scores
both against the scene's own reference video with **gradient-NCC** (FINDINGS: PSNR and
plain NCC are retracted on this night clip — both rank a wrong frame first). Correct
actor placement must IMPROVE grad-NCC; a wrong one adds car-shaped noise and hurts it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_TIME_RANGE_KEYS = (
    "time_embed.timestamps_us_ranges",                                    # dynamic_rigids
    "deform_network.feature_volume.time_input_embedding.timestamps_us_ranges",
)


def layer_track_ranges(renderer, layer="dynamic_rigids"):
    """[n_layer_tracks, 2] annotation intervals, in µs.

    ⚠️ The key path DIFFERS between the two dynamic layers — MEASURED 2026-08-03 on
    scene 00040136: `dynamic_rigids` stores them under `time_embed.…` (35 rows) but
    `dynamic_deformables` stores them under
    `deform_network.feature_volume.time_input_embedding.…` (2 rows). Probing both is
    the reason the second layer is renderable at all; looking only at the first path
    reads as "the layer has no timing" (absence at ONE location is not absence).
    """
    sd = renderer.scene.sd
    tried = []
    for suffix in _TIME_RANGE_KEYS:
        key = f".gaussians_nodes.{layer}.{suffix}"
        tried.append(key)
        if key in sd:
            return np.frombuffer(sd[key], np.int64).reshape(-1, 2).astype(np.float64)
    raise KeyError(f"no timestamps_us_ranges for layer {layer!r}; probed: {tried}")


def build_cuboid_to_track(renderer, tracks, layer="dynamic_rigids", margin_us=2.0e5,
                          exact_us=1.0, strict_margin_only=False):
    """cuboid id -> index into `tracks`, matched on the annotation time interval.

    Accept when the assignment is UNIQUE and either
      * `exact`  — the interval residual is <= `exact_us` (an exact match cannot be
        ambiguous no matter how close the runner-up is), or
      * `margin` — the runner-up is beaten by more than `margin_us`.
    `strict_margin_only=True` reproduces the pre-2026-08-03 rule for A/B comparison.
    """
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
        finite = bool(np.isfinite(best[0]))
        unique = best[1] not in used
        by_margin = (second[0] - best[0]) > margin_us
        by_exact = best[0] <= exact_us
        strict_ok = finite and by_margin and unique
        ok = strict_ok if strict_margin_only else (finite and unique and (by_exact or by_margin))
        if ok:
            mapping[c] = best[1]
            used.add(best[1])
        reason = ("rejected_nonfinite" if not finite else
                  "rejected_duplicate_track" if not unique else
                  "exact_zero_cost" if by_exact else
                  "margin" if by_margin else "rejected_ambiguous")
        report.append({"cuboid": c, "range": [float(t0), float(t1)],
                       "best_track": int(best[1]), "best_cost_us": float(best[0]),
                       "runner_up_cost_us": float(second[0]), "accepted": bool(ok),
                       "accepted_strict_margin": bool(strict_ok),
                       "accept_reason": reason,
                       "track_id": tracks.ids[best[1]],
                       "label": (tracks.labels[best[1]] if tracks.labels else None)})
    return mapping, report


def falsify_actors(renderer, scene_dir, frame=0, layer="dynamic_rigids"):
    """Does switching the actors ON make the render MORE like the reference? Measured.

    The on/off toggle goes through `set_actors_enabled` so that ALL attached dynamic
    layers are scored together; the old `_actor = None` idiom only reached the first.
    """
    from gsplat_renderer import grad_ncc, read_ref_frame
    sd = Path(scene_dir)
    mp4 = sd / "camera_front_wide_120fov.mp4"
    c2n = renderer.gt_cam_to_nre(frame)
    ts = renderer.frame_timestamps_us(frame)[1]
    ref = read_ref_frame(mp4, frame, (renderer.width, renderer.height))
    renderer.set_actors_enabled(False)
    off, _, _ = renderer.render(c2n)
    renderer.set_actors_enabled(True)
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
            # ---- TWO DIFFERENT QUESTIONS, kept apart (2026-08-03) ----------------
            # `pass_placement` — is the actor placed at the RIGHT POSE AND TIME? The
            #   discriminating control is the same actors at a WRONG time: a wrong
            #   mapping puts car-shaped gaussians in the wrong place and must score
            #   worse. This is what "falsify the placement" actually means.
            # `pass_strict` — do the actors also IMPROVE the whole-frame metric? They
            #   need not: MEASURED 2026-08-03, the actors change 389-42,631 pixels of a
            #   2.07 Mpx frame (0.02-2 %), so `g_on - g_off` lands in +-0.011 — noise at
            #   this footprint, and its SIGN flips with unrelated render settings (it was
            #   +0.0016 with plain background+road and -0.0001 with cull+sky on the very
            #   same mapping). Gating on it rejects a provably-correct mapping for a
            #   reason that has nothing to do with placement.
            # `pass` follows PLACEMENT. Both are recorded so neither number silently
            # changes meaning, and `delta_on_minus_off` stays visible as an effect size.
            "pass_placement": bool(g_on > g_wrong),
            "pass_strict": bool(g_on > g_off and g_on > g_wrong),
            "pass": bool(g_on > g_wrong)}


def attach_actors_verified(renderer, scene_dir, layer="dynamic_rigids", frames=(0, 60, 120),
                           ab_compare=False):
    """Build the mapping, attach it, and refuse to proceed if the falsifier says no.

    `ab_compare=True` additionally scores the OLD strict-margin mapping on the same
    frames, so the 2026-08-03 acceptance change is adjudicated by the reference video
    instead of by the argument in this module's docstring.
    """
    from gsplat_renderer import ActorTracks
    sd = Path(scene_dir)
    st = sd / "extracted" / "sequence_tracks.json"
    if not st.exists():
        raise FileNotFoundError(
            f"{st} missing — extract it from the scene USDZ first (it is a plain zip "
            "entry: sequence_tracks.json)")
    tracks = ActorTracks(st)
    ab = None
    if ab_compare:
        m_s, r_s = build_cuboid_to_track(renderer, tracks, layer, strict_margin_only=True)
        renderer.attach_actors(tracks, m_s, layer)
        f_s = [falsify_actors(renderer, sd, f, layer) for f in frames]
        ab = {"strict_margin": {"n_mapped": len(m_s), "falsifier": f_s,
                                "mean_grad_ncc_on": round(float(np.mean(
                                    [x["grad_ncc_actors_on"] for x in f_s])), 4),
                                "mean_delta_on_minus_off": round(float(np.mean(
                                    [x["delta_on_minus_off"] for x in f_s])), 4)}}
    mapping, report = build_cuboid_to_track(renderer, tracks, layer)
    renderer.attach_actors(tracks, mapping, layer)
    falsi = [falsify_actors(renderer, sd, f, layer) for f in frames]
    n_pass = sum(1 for f in falsi if f["pass"])
    info = {"n_tracks_json": len(tracks), "n_layer_tracks": len(report),
            "n_mapped": len(mapping),
            "n_mapped_strict_margin": sum(1 for x in report if x["accepted_strict_margin"]),
            "accept_reason_counts": {r: sum(1 for x in report if x["accept_reason"] == r)
                                     for r in sorted({x["accept_reason"] for x in report})},
            "falsifier": falsi,
            "falsifier_pass_frames": n_pass, "falsifier_n_frames": len(falsi),
            "verdict": ("ACCEPTED" if n_pass >= (len(falsi) + 1) // 2 else "REFUSED"),
            "per_track": report}
    if ab is not None:
        ab["relaxed_exact_or_margin"] = {
            "n_mapped": len(mapping), "falsifier": falsi,
            "mean_grad_ncc_on": round(float(np.mean(
                [x["grad_ncc_actors_on"] for x in falsi])), 4),
            "mean_delta_on_minus_off": round(float(np.mean(
                [x["delta_on_minus_off"] for x in falsi])), 4)}
        ab["verdict"] = (
            "RELAXED WINS ON PIXELS"
            if ab["relaxed_exact_or_margin"]["mean_grad_ncc_on"]
            > ab["strict_margin"]["mean_grad_ncc_on"] else
            "STRICT WINS OR TIES — do NOT adopt the relaxed rule")
        info["ab_acceptance_rule"] = ab
    if info["verdict"] != "ACCEPTED":
        renderer.set_actors_enabled(False)
    return info


def attach_all_dynamic_layers(renderer, scene_dir,
                              layers=("dynamic_rigids", "dynamic_deformables"),
                              frames=(0, 60, 120)):
    """Attach EVERY dynamic layer the scene ships, then falsify the whole set together.

    Until 2026-08-03 the renderer drew `background + road` only, so both dynamic layers
    were absent from every frame the programme produced. Rendering them is a coverage
    change, so it is adjudicated the same way everything else is: grad-NCC against the
    scene's own reference video, with a WRONG-TIME negative control.

    Per-layer mappings are built independently (the two layers index different track
    sets and store their time ranges under different keys). A layer whose mapping is
    empty is reported and skipped, never silently drawn at the origin.
    """
    from gsplat_renderer import ActorTracks
    sd = Path(scene_dir)
    st = sd / "extracted" / "sequence_tracks.json"
    if not st.exists():
        raise FileNotFoundError(
            f"{st} missing — extract it from the scene USDZ first (it is a plain zip "
            "entry: sequence_tracks.json)")
    tracks = ActorTracks(st)
    per_layer, attached = {}, []
    for L in layers:
        if f".gaussians_nodes.{L}.positions" not in renderer.scene.sd:
            per_layer[L] = {"present": False}
            continue
        try:
            mapping, report = build_cuboid_to_track(renderer, tracks, L)
        except KeyError as e:  # no time ranges under EITHER probed key
            per_layer[L] = {"present": True, "error": str(e), "n_mapped": 0}
            continue
        # `gaussian_cuboid_ids` may reference only a subset of the layer's tracks
        cid = np.frombuffer(
            renderer.scene.sd[f".gaussians_nodes.{L}.gaussian_cuboid_ids"], np.int32)
        present = set(int(c) for c in np.unique(cid))
        usable = {c: t for c, t in mapping.items() if c in present}
        per_layer[L] = {
            "present": True, "n_layer_tracks": len(report), "n_mapped": len(mapping),
            "n_cuboids_with_gaussians": len(present),
            "n_mapped_and_present": len(usable),
            "n_gaussians": int(cid.size),
            "accept_reason_counts": {r: sum(1 for x in report if x["accept_reason"] == r)
                                     for r in sorted({x["accept_reason"] for x in report})},
            "per_track": report}
        if usable:
            renderer.attach_actors(tracks, usable, L)
            attached.append(L)
    falsi = [falsify_actors(renderer, sd, f) for f in frames]
    n_pass = sum(1 for f in falsi if f["pass_placement"])
    n_strict = sum(1 for f in falsi if f["pass_strict"])
    return {"layers_attached": attached, "per_layer": per_layer,
            "falsifier": falsi, "falsifier_pass_frames": n_pass,
            "falsifier_pass_frames_strict": n_strict,
            "falsifier_n_frames": len(falsi),
            "mean_delta_on_minus_wrongtime": round(float(np.mean(
                [x["delta_on_minus_wrongtime"] for x in falsi])), 4),
            "mean_delta_on_minus_off": round(float(np.mean(
                [x["delta_on_minus_off"] for x in falsi])), 4),
            "verdict": ("ACCEPTED" if n_pass >= (len(falsi) + 1) // 2 else "REFUSED"),
            "verdict_strict_improvement": (
                "ACCEPTED" if n_strict >= (len(falsi) + 1) // 2 else "REFUSED")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", default="/tmp/actor_map.json")
    ap.add_argument("--ab-compare", action="store_true",
                    help="also score the OLD strict-margin mapping on the same frames")
    a = ap.parse_args()
    from gsplat_renderer import NuRecGsplatRenderer
    r = NuRecGsplatRenderer(Path(a.scene_dir).expanduser())
    info = attach_actors_verified(r, Path(a.scene_dir).expanduser(), ab_compare=a.ab_compare)
    Path(a.out).write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "per_track"}, indent=2))
    print("accepted cuboids:", sum(1 for x in info["per_track"] if x["accepted"]),
          "/", len(info["per_track"]))
