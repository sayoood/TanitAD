"""Add clip_sha12 / window_index to the census tables and write the delivered, named truth tables."""
import json, sys
from pathlib import Path
import numpy as np

COLUMNS = {
 "window_index": "row number == index into the trainer's V3Dataset (ds.index order)",
 "clip_sha12": "sha256(clip_id)[:12] of the window's clip (the only clip identity this package carries)",
 "clip_ix": "index into the clips JSON (provider order of the cache)",
 "t": "window start row t (the trainer's index); the window's NOW is row t + 7 (window 8)",
 "t_now_s": "the trainer's own ds._now_s(ep, t): grid_start_s + (t + w - 1 + n_stack - 1) * dt_s (raw recording seconds)",
 "has_record": "clip joined to a v8 label record AND not G3-excluded (3 eval clips have no measured clock)",
 "in_band": "v7_labels.window_in_band(): |t_now - t0| <= 2.0 s  (t0 = 8.0 s for every record)",
 "lat_v7": "tactical lateral class id (index into vocab.lat_classes); -100 = IGNORE (no loss)",
 "lon_v7": "tactical longitudinal class id (vocab.lon_classes); -100 = IGNORE",
 "n_goal_scored": "number of the 22 goal cells with weight > 0 AS THE DATALOADER WORKERS SAW THEM (eval-blob negative policy); NOT class-masked",
 "n_goal_scored_censusstate": "same, under the TRAIN-blob negative policy the config.json census was computed with",
 "n_goal_pos": "goal cells that are positive (y=1, w=1)",
 "goal_y_bits": "bit i = goal token i (vocab.goal_tokens) is positive",
 "goal_w_bits": "bit i = goal token i has weight>0 (workers' state); AND with the 17 trainable tokens for the supervised-cell count",
 "goal_w_bits_censusstate": "bit i = weight>0 under the train-blob census state",
 "nav_cmd": "fed nav token: 0 follow, 1 left, 2 right (vocab.nav_commands)  -- ONE value per clip",
 "nav_valid": "nav token present (record joined)",
 "vmax_ms": "raw fed ceiling = SPEED_BAND.v_hi_ms (m/s), ONE value per clip",
 "vmax_valid": "ceiling fed",
 "vmax_bin": "index into the {30,50,100,120} km/h ladder (containing-window rule; clamps at 120)",
 "agent_labelled": "reader.lookup(clip, NOW frame) is not None (agent boxes labelled; includes labelled-clear)",
 "n_agents": "number of agent boxes at the NOW frame (-1 = not labelled)",
 "box3d_labelled": "at least one agent of the frame has a 3-D cuboid (zh_for_frame mask any)",
 "n_box3d": "number of agents with a cuboid (-1 = not labelled)",
 "map_label": "SAM3 10 cm map target exists for the NOW frame (trainer census state 'ok')",
 "n_future_valid": "valid future poses after NOW, capped at 60 (the trainer's MAX_H_EXT)",
 "full6": "NOW + round(6/dt) rows exist in the clip (the +6 s geometry columns are defined)",
 "v0": "pose speed at NOW (m/s)", "v2": "speed at +2 s", "v4": "speed at +4 s", "v6": "speed at +6 s",
 "dyaw6_deg": "wrap(yaw[+6 s] - yaw[NOW]) in degrees; LEFT (counter-clockwise, +y) is POSITIVE; NaN unless full6",
 "lat6_m": "lateral offset at +6 s in the NOW ego frame (+ = left), metres",
 "fwd6_m": "forward offset at +6 s in the NOW ego frame, metres",
 "vmax26": "max speed over [+2,+6] s (m/s)", "vmin26": "min speed over [+2,+6] s", "vmin06": "min speed over [0,+6] s",
 "stop_0_6": "min speed over [0,+6] s < 0.5 m/s (defined only when full6)",
 "theta_slot_deg": "route-following package definition: heading of the slot-50 -> slot-60 (5 s -> 6 s) segment in the NOW frame, degrees, + = left (NaN unless slot 60 exists)",
 "path_len_slot_m": "path length through the 8 V3 slots to slot 60 (m)",
 "ttn_s": "seconds from NOW to the next labelled turn START (nav_30s.entries NAV_TURN_L/R only, start >= NOW); NaN = none",
 "ttn_dyaw_deg": "label dyaw_deg of that turn", "ttn_side": "+1 left, -1 right, 0 none",
 "in_turn_now": "a labelled turn is in progress at NOW (start <= NOW <= end)", "turn_now_dyaw_deg": "its label dyaw_deg",
 "n_turns_in_labels": "number of NAV_TURN entries in the clip's nav_30s",
}

def finalize(src: Path, tag: str, out_tag: str, outdir: Path):
    z = np.load(src / f"windows_{tag}.npz")
    clips = json.load(open(src / f"clips_{tag}.json", encoding="utf-8"))
    cols = {k: z[k] for k in z.files}
    sha = np.array([c["sha12"] for c in clips], dtype="S12")
    cols["clip_sha12"] = sha[cols["clip_ix"]]
    cols["window_index"] = np.arange(len(cols["t"]), dtype=np.int32)
    missing = [k for k in cols if k not in COLUMNS]
    assert not missing, missing
    np.savez_compressed(outdir / f"label_truth_{out_tag}.npz", **cols)
    json.dump(clips, open(outdir / f"label_truth_{out_tag}_clips.json", "w", encoding="utf-8"), indent=0, default=str)
    return len(cols["t"]), len(cols)

if __name__ == "__main__":
    base = Path(sys.argv[1])
    out = base / "tables"
    out.mkdir(exist_ok=True)
    print(finalize(base / "raw/thor_train", "train", "train", out))
    print(finalize(base / "raw/thor_eval139", "eval139", "eval139", out))
    json.dump(COLUMNS, open(out / "label_truth_columns.json", "w", encoding="utf-8"), indent=1)
