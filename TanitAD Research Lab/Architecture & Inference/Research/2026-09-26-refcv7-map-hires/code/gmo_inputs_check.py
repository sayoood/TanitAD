"""Thor, read-only, CPU: the G-MAP-OVERFIT INPUTS against the prereg, before any training.

Loads the 16 frames through the candidate harness's OWN `load_frames` (the 1 ms time guard
on, camera timestamps from --cam-ts-dir), then reports:
* the frames (sha12, raw frame) vs the prereg table, and the frameset md5 vs the spec's;
* the axis guard per clip (time_1ms expected);
* the label census in 0-20 m, three ways: (a) inside the prereg's old 60 x +-16 window,
  codes != 255 (no cone); (b) the harness's scored mask (codes != 255 AND lift-valid) over the
  full A7 width; (c) the prereg's published census (an APPROXIMATE camera cone).
Clip ids never printed.
"""
import hashlib
import json
import sys
from pathlib import Path

CODE = Path("/home/nvidia/gmo_early_0327/code")
sys.path[:0] = [str(CODE / "stack"), str(CODE / "stack" / "scripts")]
import torch  # noqa: E402
import map_hires_overfit as M  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7  # noqa: E402

assert str(CODE) in M.__file__
AUD = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
spec = M.load_spec(AUD / "gmo_spec.json", class_weights="uniform",
                   decision_rule="prior_corrected", band_keys=EXTENT_REFCV7.band_keys)
fs_md5 = hashlib.md5((AUD / "gmo_frameset.json").read_bytes()).hexdigest()
fs = json.loads((AUD / "gmo_frameset.json").read_text())
prereg_frames = sorted((e["clip_sha12"], int(f)) for e in fs["frames"] for f in e["raw_v2ep_frames"])
hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
frame = frame_for_width(1024, 416)
data = M.load_frames(spec, Path("/home/nvidia/data/refcv6-b1-416x1024-train"),
                     Path("/home/nvidia/data/sam3_gt_v3"),
                     Path("/home/nvidia/data/refcv6_train_eval139_extrinsics.json"),
                     frame, hcfg, 43,
                     cam_ts_dir=Path("/home/nvidia/data/_b1stage416/r0/camera_front_wide"))
codes = data["codes"]                                   # [16, 1000, 600]
lv = H.lift_valid_to_fine(data["valid"].any(dim=1), tuple(hcfg.out_hw))  # as MapHiresBranch: any height
off = EXTENT_REFCV7.anchor_offset(0.1) if hasattr(EXTENT_REFCV7, "anchor_offset") else 140
band = slice(0, 200)
names = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text",
         "non-drivable edge", "hatched", "sidewalk/verge"]
old = codes[:, band, off:off + 320]
a = {names[k]: int((old == k).sum()) for k in range(8)}
sc = (codes[:, band] != 255) & lv[:, band]
b = {names[k]: int(((codes[:, band] == k) & sc).sum()) for k in range(8)}
out = {"frameset_md5": fs_md5, "spec_frameset_md5": spec.get("frameset_md5"),
       "frames_equal_prereg": sorted(zip(data["sha12"], data["raw_frame"])) == prereg_frames,
       "n_frames": len(data["sha12"]), "x_shape": list(data["x"].shape),
       "codes_shape": list(codes.shape), "anchor_offset_cols": int(off),
       "axis_guard": data["axis_guard"],
       "census_0_20m_old_window_no_cone": a,
       "census_0_20m_scored_mask_full_width": b,
       "census_0_20m_prereg_approx_cone": fs["cells_by_class_0_20m"],
       "min_cells_floor": spec["thresholds"]["min_cells"],
       "scored_mask_classes_below_floor": [k for k, v in b.items()
                                           if v < int(spec["thresholds"]["min_cells"])]}
print(json.dumps(out, indent=1, default=str))
