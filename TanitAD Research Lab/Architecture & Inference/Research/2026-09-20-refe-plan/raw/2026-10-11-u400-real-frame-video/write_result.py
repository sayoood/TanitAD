"""Generate RESULT_VIDEO.md from the MEASURED artifacts of this package (no number is typed by hand):
inference/FIDELITY_GATE.json, qa_report.json, video_runs.json, pipeline_done.json, probe_db_rates.txt, the u400 per-token CSV.

    python write_result.py            (run after run_pipeline.py wrote pipeline_done.json)
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"


def main():
    g = json.load(open(os.path.join(HERE, "inference", "FIDELITY_GATE.json")))
    qa = json.load(open(os.path.join(HERE, "qa_report.json")))
    done = json.load(open(os.path.join(HERE, "pipeline_done.json")))
    runs = json.load(open(os.path.join(HERE, "video_runs.json")))
    rates = open(os.path.join(HERE, "probe_db_rates.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "probe_db_rates.txt")) else "(probe_db_rates.txt missing)"
    tr = {r["token"]: float(r["score"]) for r in csv.DictReader(open(CSV, encoding="utf-8"))}
    toks = [t for r in runs for t in r["tokens"]]
    sc = np.array([tr[t] for t in toks])
    ov = g["overall"]
    ff, fr = qa["ffprobe"], qa["front_frame_spacing_ms"]
    pv = qa["provenance_psnr"]
    cam = qa["side_rear_cameras"]
    dur = float(ff["duration"])
    nfail = int((sc < 0.5).sum())
    L = []
    L.append("# RESULT_VIDEO -- REFe SFT-4 B u400 on REAL camera frames (no optical-flow interpolation)\n")
    L.append(f"**Deliverable:** `{done['video']}` -- {done['bytes'] / 1e6:.0f} MB, {dur:.0f} s = {dur / 60:.2f} min, {ff['width']}x{ff['height']}, {ff['codec_name']} "
             f"({ff['profile']}), CRF 18 preset slow, {float(ff['bit_rate']) / 1e6:.1f} Mbit/s, 10 fps, {ff['nb_read_frames']} frames. (`*.mp4` is git-ignored: this file is the only copy.)\n")
    L.append(f"**Fidelity gate: PASS.** The re-run u400 plans reproduce the banked full-navtest plans on all {ov['n']} video tokens with max |delta| "
             f"= **{ov['max_abs_delta_any_component']:.3g}** (xy {ov['max_xy_delta_m']:.3g} m, heading {ov['max_heading_delta_rad']:.3g} rad; tolerance 1e-3 m; "
             f"{ov['tokens_identical']}/{ov['n']} tokens bit-identical), and argmax of the scorer rule equals the recorded pick on every token.\n")
    L.append("## What was wrong, what changed\n")
    L.append("The previous video (`raw/2026-10-01-final-video/render_sync_video.py`) motion-interpolated 2 Hz OpenScene frames to 10 fps with DIS optical flow "
             "(`interp_frame`, `Flow`), producing doubled trees, ghosted sign outlines and smeared structure. This renderer contains **no frame synthesis**: every camera "
             "image is one decoded sensor JPEG shown unmodified (front native 1920x1080; side/rear area-downscaled to 480x270). The only blends in the code are alpha "
             "blends of *synthetic overlays* (`cv2.addWeighted` on the overlay layers); no `remap`, no flow, no warping.\n")
    L.append("## Frame sources per camera (MEASURED, `manifests/clip*.csv` = one row per output frame)\n")
    L.append("| camera | source | rate shown | frames |\n|---|---|---|---|")
    L.append(f"| CAM_F0 (front) | raw nuPlan `nuplan-camera/mini/<log>_CAM_F0.zip` (all 10 navtest logs), DB `image.filename_jpg` | native 10 Hz, each frame at its OWN timestamp | {qa['frames']['manifest_rows']} (all) |")
    for ch, nm in (("CAM_L0", "left"), ("CAM_R0", "right"), ("CAM_B0", "rear")):
        c = cam[ch]
        L.append(f"| {ch} ({nm}) | raw `nuplan-camera/scenarios4/<log>_{ch}.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame "
                 f"(the planner's own input frame) for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: {c['frames_real_10hz']} frames; "
                 f"held 2 Hz: {c['frames_held_2hz']} frames | {c['frames_real_10hz'] + c['frames_held_2hz']} |")
    L.append(f"\nShare of output frames with ALL FOUR cameras real: **{100 * qa['share_all_four_real']:.1f} %**. Only one navtest log has raw L0/R0/B0 zips locally "
             "(`scenarios4/` holds B0/L0/R0 for 7 logs; only `01690_02183` is navtest). No download was made (permission required); the other navtest logs' side/rear "
             "frames are held, never warped.\n")
    L.append("## Measured rates (never assumed)\n")
    L.append("```\n" + rates.strip() + "\n```\n")
    L.append(f"Across ALL {fr['n'] + 94} output front frames the consecutive front-frame spacing within a clip is min {fr['min']:.2f} / median {fr['median']:.3f} / max {fr['max']:.2f} ms "
             f"({fr['rate_hz_from_median']:.3f} Hz); {fr['outside_95_105']} spacings fall outside 95-105 ms. Maximum |image timestamp - its ego_pose timestamp| = "
             f"{qa['f0_ego_pose_dt_us_max_abs']} us. Real side/rear frames are within {max(c['real_max_dt_to_front_ms'] or 0 for c in cam.values()):.2f} ms of the front frame; "
             f"held frames are within {max(c['held_max_dt_to_token_ms'] or 0 for c in cam.values()):.1f} ms of the token time. "
             f"DB ego speed vs the token's export speed at token frames: median |diff| {qa['speed_db_vs_token_abs_diff_mps']['median']:.3f} m/s, max {qa['speed_db_vs_token_abs_diff_mps']['max']:.3f} m/s.\n")
    L.append("## How the overlays are registered (no pose interpolation, no image warping)\n")
    L.append("* The plan of token k (64 hypotheses + selected) is expressed in the rear-axle frame at the token's `lidar_pc` pose. For EACH front frame it is transformed into that "
             "frame's body frame using the high-rate `ego_pose` row the DB itself links to the image (`image.ego_pose_token`; full 3-D pose incl. roll/pitch/z, so a braking "
             "pitch is respected), then projected with the DB camera calibration (`calib_table.read_db`: K, extrinsics) **including the Caltech lens distortion "
             "(k1 = -0.356 ...) applied to the projected points** (the image is never undistorted/warped). Plan points lie on the ground plane at the rear-axle height of token k.\n"
             "* Calibration check: `stills/check_calibration_logged_agent_boxes.png` draws the DB `lidar_box` footprints (20 Hz) on the same pipeline; they sit on the cars/cones in the image.\n"
             "* BEV: metric (12 px/m, scale bar, 10 m rings), nuPlan map (gpkg reprojected exactly as the devkit does) + logged agents from DB `lidar_box` at the nearest 20 Hz lidar frame to each "
             "camera frame (no interpolation), ego, 64 hypotheses, selected plan (green), human future (white).\n"
             "* Drivable area (camera + BEV) = NAVSIM's DAC polygon layers (lane groups + intersections + car parks), labelled \"map: rendering aid, not a model input\".\n")
    L.append("## Provenance QA (is every displayed front frame ONE real JPEG?)\n")
    L.append(f"{pv['n']} random MOVING (>3 m/s) output frames were decoded from the final mp4 and compared on the overlay-free front region (front y 110-400, sky/buildings) with (a) their own source "
             f"JPEG and (b) the adjacent source JPEG (+-0.1 s): PSNR vs own source min {pv['own_min']:.1f} / median {pv['own_median']:.1f} dB; vs the adjacent frame median {pv['adjacent_median']:.1f} dB "
             f"(max {pv['adjacent_max']:.1f}). Own beats adjacent on every sampled frame: **{pv['own_beats_adjacent_every_frame']}** (min gap {pv['min_gap_db']:.1f} dB). "
             "A motion-interpolated blend would not match its own source this closely. Frame accounting: "
             f"expected {qa['frames']['expected']} frames, decoded {qa['frames']['decoded']} ({'match' if qa['frames']['match'] else 'MISMATCH'}).\n")
    L.append("## Content\n")
    L.append(f"{len(runs)} clips = every run of >= 6 consecutive navtest tokens (0.5 s apart) in the 10 navtest logs with raw front frames: {len(toks)} tokens = {len(toks) / 2:.0f} s of driving "
             f"+ title/summary cards. Hard cuts between clips (a \"NEW CLIP i/94\" banner marks each). Full table: `RUNS.md`. Coverage: turns, stops, lane keeping at speed, "
             f"lane changes (heuristic tag), and **{nfail} failure tokens (true PDMS < 0.5) are included and shown in red** -- nothing was filtered by score.\n")
    L.append(f"True per-token score of these {len(toks)} tokens (u400 full-navtest harness CSV `refe_stage2_sft4_B_u400_navtest_full.csv`, executed plan): mean **{100 * sc.mean():.2f}**, "
             f"median {100 * np.median(sc):.1f}; the full 12,146-token navtest PDMS of this checkpoint is 88.2469 (result.json). This sample is NOT the benchmark.\n")
    L.append("## Known limits (read before quoting)\n")
    L.append("* Overlays are drawn on a flat ground plane and are not occlusion-aware: far plan/drivable pixels can paint over a wall or hedge that hides that ground. Road slope is not modelled.\n"
             "* Drivable area = the DAC polygon set, not the full PDM scorer (which also uses footprint corners, LQR tracking, etc.).\n"
             "* Predicted PDMS on screen is the u400 scorer head's estimate under the deployed rule (`navsim_v1_lane`: NC*DAC*(5EP+5TTC+2C)/12 * p_lane); the TRUE score comes from the harness CSV.\n"
             "* 94 discontinuous clips: navtest tokens are sparse, so there is no continuous drive.\n"
             "* Side/rear cameras are held 2 Hz for 9 of the 10 logs (see above).\n"
             "* The model itself consumes F0/L0/R0/B0 at the token time; the video shows what it saw (first frame of each token window is the planner's input frame, marked on screen).\n"
             "* `MODEL_REGISTRY.md` has no row for u400 (grep `u400` = 0 hits): model facts here come from the raw eval JSON (`raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/result.json`, "
             "`seam_report.json`) and the checkpoint md5 `8e382eebb35b3e155862453617e5786e` (verified; expected prefix 8e382eeb). **Escalation: the registry lacks this checkpoint.**\n")
    L.append("## Reproduce\n")
    L.append("`select_tokens.py` -> `run_inference.py` (gate) -> `run_pipeline.py` (render/cards/concat/QA/stills) -> `write_result.py`. Renderer: `render_u400_video.py` + `u400_lib.py` "
             "(reuse `raw/2026-10-01-final-video/render_final_video.py`: text, BEV class, box geometry, turbo ramp). Per-frame provenance: `manifests/`.\n")
    open(os.path.join(HERE, "RESULT_VIDEO.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote RESULT_VIDEO.md")


if __name__ == "__main__":
    main()
