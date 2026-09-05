# -*- coding: utf-8 -*-
"""Bank the reel: a COMPACT per-window record beside the stills.

The full frame index is 1.3 MB of 2,073 rows, most of them slow-motion
duplicates. What is worth keeping is one row per SCORED WINDOW (40) plus the
ground-truth control's own per-window table (40) -- small, and enough to trace
any still back to the window it shows. ASCII only in print().
"""
import io, json, os, shutil, sys

FINAL = "C:/Users/Admin/tanitad-caches/refav1_reel_out/final"
STILLS = "C:/Users/Admin/tanitad-caches/refav1_reel_out/stills"
DEST = sys.argv[1]
os.makedirs(DEST, exist_ok=True)

ix = json.load(io.open(os.path.join(FINAL, "refav1_arms_step21109_frame_index.json"),
                       encoding="utf-8"))
ctl = ix.get("gt_control") or {}
scored = [r for r in ix["frames"]
          if r.get("kind") == "clip" and r.get("plan_age_s") == 0.0]
seen, uniq = set(), []
for r in scored:
    key = (r["clip_id"], r["t_cache"])
    if key in seen:
        continue
    seen.add(key)
    uniq.append({k: v for k, v in r.items() if k not in ("frame",)})

out = {
    "tool": "render_refav1_arms_video.py",
    "banked_by": "bank_reel.py",
    "step": ix["step"], "fps": ix["fps"], "n_frames": ix["n_frames"],
    "duration_s": round(ix["n_frames"] / ix["fps"], 2),
    "ckpt": ix["ckpt"], "arms": ix["arms"], "arm_geometry": ix["arm_geometry"],
    "arms_dir": ix["arms_dir"],
    "shared_verification": ix["shared_verification"],
    "stats": ix["stats"],
    "gt_control": {k: v for k, v in ctl.items() if k != "per_window"},
    "gt_control_per_window": ctl.get("per_window"),
    "scored_windows": uniq,
    "note": ("one row per SCORED WINDOW (the plan-age-0 frame). The full "
             "2,073-row frame index stays beside the mp4; most of its rows are "
             "slow-motion duplicates of these."),
}
io.open(os.path.join(DEST, "WINDOWS.json"), "w", encoding="utf-8").write(
    json.dumps(out, indent=1))

n = 0
for f in sorted(os.listdir(STILLS)):
    if f.startswith("v_") or not (f.endswith(".png") or f.endswith(".json")):
        continue
    shutil.copyfile(os.path.join(STILLS, f), os.path.join(DEST, f))
    n += 1
shutil.copyfile(os.path.join(FINAL, "refav1_arms_step21109_GT_CONTROL.png"),
                os.path.join(DEST, "GT_CONTROL.png"))
print("scored windows banked:", len(uniq))
print("stills copied:", n, "+ GT_CONTROL.png")
print("dest:", DEST)
for f in sorted(os.listdir(DEST)):
    print("   %-42s %9d B" % (f, os.path.getsize(os.path.join(DEST, f))))
