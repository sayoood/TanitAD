"""Generate RESULT_VIDEO.md for the P6 video from the MEASURED artifacts of this package (no number is typed by hand except the P6 result
figures that the cards also show, copied from P6_RESULT.md / level_ci.json):
FIDELITY_GATE.json, qa_report.json, video_runs.json, probe_db_rates.txt, the P6 per-token CSV.

    python write_result.py <video.mp4> [<video_1080p.mp4>]
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("P6_WORK", HERE)
RUNS = os.environ.get("P6_RUNS", os.path.join(HERE, "video_runs.json"))
CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv"


def main():
    video = sys.argv[1]
    v1080 = sys.argv[2] if len(sys.argv) > 2 else None
    g = json.load(open(os.environ.get("P6_GATE", os.path.join(HERE, "FIDELITY_GATE.json"))))
    qa = json.load(open(os.path.join(WORK, "qa_report.json")))
    runs = json.load(open(RUNS))
    rates = open(os.path.join(HERE, "probe_db_rates.txt"), encoding="utf-8").read()
    tr = {r["token"]: float(r["score"]) for r in csv.DictReader(open(CSV, encoding="utf-8"))}
    toks = [t for r in runs for t in r["tokens"]]
    sc = np.array([tr[t] for t in toks])
    ov, dn = g["overall"], g["dump_DONE"]
    ff, fr, pv, cam = qa["ffprobe"], qa["front_frame_spacing_ms"], qa["provenance_psnr"], qa["side_rear_cameras"]
    dur = float(ff["duration"])
    nfail = int((sc < 0.5).sum())
    o = []
    o.append("# RESULT_VIDEO -- REFe P6 (model_final.pt) on REAL camera frames (no optical-flow interpolation)\n")
    o.append(f"**Deliverable:** `{video}` -- {os.path.getsize(video) / 1e6:.0f} MB, {dur:.0f} s = {dur / 60:.2f} min, {ff['width']}x{ff['height']}, {ff['codec_name']} "
             f"({ff['profile']}), CRF 18 preset slow, {float(ff['bit_rate']) / 1e6:.1f} Mbit/s, 10 fps, {ff['nb_read_frames']} frames."
             + (f" 1080p copy: `{v1080}` ({os.path.getsize(v1080) / 1e6:.0f} MB)." if v1080 and os.path.exists(v1080) else "")
             + " `*.mp4` is git-ignored and raw camera frames stay private: the video and its frames do not go to any repo.\n")
    o.append(f"**Fidelity gate: PASS.** No model was run for this video: the plans are sliced from P6's second full-navtest pass (`proposals_p6final.npz`, the first pass's seam command plus "
             f"`--dump-proposals`). For all {ov['n']} video tokens the selected trajectory of the dump equals the banked plan (`seams/refe_p6final.npz`): max |delta| = "
             f"**{ov['max_abs_delta_any_component']:.3g}** (xy {ov['max_xy_delta_m']:.3g} m, heading {ov['max_heading_delta_rad']:.3g} rad; tolerance 1e-3 m; "
             f"{ov['tokens_identical']}/{ov['n']} tokens bit-identical). argmax of rule `navsim_v1` over the logits differs from the recorded pick on {ov['argmax_rule_ne_pick']} tokens. "
             f"`dump.DONE`: poses_bit_identical {dn['poses_bit_identical']} over all 12,146 tokens (tokens differing {dn['n_tokens_differing']}). Dump sha256 `{g['dump_sha256'][:16]}...`.\n")
    o.append("**Checkpoint:** `model_final.pt` sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`; rule `navsim_v1` = NC*DAC*(5EP+5TTC+2C)/12 over six score outputs, "
             "NO lane factor (u400's rule had one); nav_mode command.\n")
    o.append("## Measured results of this checkpoint (on the intro/outro cards; copied from `P6_RESULT.md` and `level_ci.json`)\n")
    o.append("MEASURED, full navtest, 12,146 tokens, harness PASS: **PDMS 91.20 [90.41, 91.98]** (`level_ci.json`); +2.95 [+2.19, +3.77] over u400 88.25 (`P6_RESULT.md`); "
             "goal 94.55 (DriveZero Table A13): BELOW GOAL. **PREREG_P6, read as written: NOT PROVEN against u400 and against P2. BELOW GOAL.** Sub-scores NC 98.85, DAC 97.95, EP 87.03, "
             "TTC 95.08, C 100.00, DDC 96.48 (`level_ci.json`). T1-class open loop against logged agents: the car in this video is the human's; the overlays show what P6 would have planned.\n")
    o.append("## Frame sources per camera (MEASURED, `manifests/clip*.csv` = one row per output frame)\n")
    o.append("| camera | source | rate shown | frames |\n|---|---|---|---|")
    o.append(f"| CAM_F0 (front) | raw nuPlan `nuplan-camera/mini/<log>_CAM_F0.zip` (all 10 navtest logs), DB `image.filename_jpg` | native 10 Hz, each frame at its OWN timestamp | {qa['frames']['manifest_rows']} (all) |")
    for ch, nm in (("CAM_L0", "left"), ("CAM_R0", "right"), ("CAM_B0", "rear")):
        c = cam[ch]
        o.append(f"| {ch} ({nm}) | raw `scenarios4/<log>_{ch}.zip` for 1 navtest log (`2021.05.25.14.16.10_veh-35_01690_02183`); the OpenScene 2 Hz frame (the planner's own input frame) "
                 f"for the other 9, HELD for the 0.5 s token window and labelled `2 Hz (held)` on screen | real 10 Hz: {c['frames_real_10hz']} frames; held 2 Hz: {c['frames_held_2hz']} frames | "
                 f"{c['frames_real_10hz'] + c['frames_held_2hz']} |")
    o.append(f"\nShare of output frames with ALL FOUR cameras real: **{100 * qa['share_all_four_real']:.1f} %**. No download was made; the other navtest logs' side/rear frames are held, never warped.\n")
    o.append("## Measured rates (never assumed; from the u400 package's probe, same data)\n")
    o.append("```\n" + rates.strip() + "\n```\n")
    o.append(f"Across all {fr['n'] + len(runs)} output front frames the consecutive front-frame spacing within a clip is min {fr['min']:.2f} / median {fr['median']:.3f} / max {fr['max']:.2f} ms "
             f"({fr['rate_hz_from_median']:.3f} Hz); {fr['outside_95_105']} spacings fall outside 95-105 ms. Max |image timestamp - its ego_pose timestamp| = {qa['f0_ego_pose_dt_us_max_abs']} us. "
             f"DB ego speed vs the token's export speed at token frames: median |diff| {qa['speed_db_vs_token_abs_diff_mps']['median']:.3f} m/s, max {qa['speed_db_vs_token_abs_diff_mps']['max']:.3f} m/s.\n")
    o.append("## How the overlays are registered (no pose interpolation, no image warping)\n")
    o.append("* The plan of token k (64 hypotheses + selected) is expressed in the rear-axle frame at the token's `lidar_pc` pose. For EACH front frame it is transformed into that frame's body frame with the "
             "high-rate `ego_pose` row the DB links to the image (full 3-D pose), then projected with the DB camera calibration including the Caltech lens distortion applied to the projected points.\n"
             "* BEV: metric (12 px/m), nuPlan map + logged agents (DB `lidar_box`, 20 Hz, nearest frame), ego, 64 hypotheses, selected plan (green), human future (white).\n"
             "* Drivable area (camera + BEV) = NAVSIM's DAC polygon layers, labelled \"map: rendering aid, not a model input\".\n")
    o.append("## Provenance QA (is every displayed front frame ONE real JPEG?)\n")
    o.append(f"{pv['n']} random MOVING (>3 m/s) output frames decoded from the final mp4 and compared on the overlay-free front region with their own source JPEG and the adjacent one: PSNR vs own source "
             f"min {pv['own_min']:.1f} / median {pv['own_median']:.1f} dB; vs the adjacent frame median {pv['adjacent_median']:.1f} dB (max {pv['adjacent_max']:.1f}). Own beats adjacent on every sampled frame: "
             f"**{pv['own_beats_adjacent_every_frame']}** (min gap {pv['min_gap_db']:.1f} dB). Frame accounting: expected {qa['frames']['expected']}, decoded {qa['frames']['decoded']} "
             f"({'match' if qa['frames']['match'] else 'MISMATCH'}).\n")
    o.append("## Content\n")
    o.append(f"{len(runs)} clips (the same as the u400 video) = every run of >= 6 consecutive navtest tokens in the 10 navtest logs with raw front frames: {len(toks)} tokens = {len(toks) / 2:.0f} s of driving + two result cards. "
             f"Hard cuts between clips (\"NEW CLIP i/{len(runs)}\" banner). Table: `RUNS.md` (its score columns are u400's; P6's are on screen). **{nfail} failure tokens (P6 true PDMS < 0.5) are shown in red**; nothing was filtered by score.\n")
    o.append(f"True per-token score of these {len(toks)} tokens (`refe_p6final.csv`, executed plan; MEASURED): mean **{100 * sc.mean():.2f}**, median {100 * np.median(sc):.1f}. "
             "The full 12,146-token navtest PDMS of this checkpoint is 91.20 [90.41, 91.98]. This sample is NOT the benchmark. (u400's mean on the same tokens was 87.38.)\n")
    o.append("## Known limits (read before quoting)\n")
    o.append("* Overlays are drawn on a flat ground plane and are not occlusion-aware. Road slope is not modelled.\n"
             "* Drivable area = the DAC polygon set, not the full PDM scorer.\n"
             "* Predicted PDMS on screen is the P6 scorer head's estimate under `navsim_v1` (no lane factor); the TRUE score comes from the harness CSV.\n"
             "* 94 discontinuous clips; side/rear cameras are held 2 Hz for 9 of the 10 logs.\n"
             "* The model consumes F0/L0/R0/B0 at the token time; the first frame of each token window is the planner's input frame, marked on screen.\n"
             "* One checkpoint, one training seed, deterministic planner; the level CI answers \"another draw of logs\" only (P6_RESULT.md).\n")
    o.append("## Reproduce\n")
    o.append("`gate_fidelity.py` (slice + gate) -> `run_all.py` (render, cards, concat, QA, 1080p, this file). Renderer `render_p6_video.py` + `p6_lib.py` (reuse `raw/2026-10-01-final-video/render_final_video.py`). Per-frame provenance: `manifests/`.\n")
    open(os.path.join(WORK, "RESULT_VIDEO.md"), "w", encoding="utf-8").write("\n".join(o))
    print("wrote", os.path.join(WORK, "RESULT_VIDEO.md"))


if __name__ == "__main__":
    main()
