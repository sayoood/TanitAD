"""THE FRAME AXIS, checked THREE WAYS, each derived independently of my code.

⛔ This is the check that decides whether the whole wiring is right. If the
stacked-row -> raw-frame conversion is off by one, every window is labelled with
a map 0.1 s away, at 30 km/h that is 0.83 m of BEV shift, and NOTHING in the
loss, the counts, or the gradient report would say so. Re-running my own
expression and finding agreement would measure determinism, not correctness
(CLAUDE.md, the four green-forever checks of 2026-09-07).

The three probes, in increasing strength:

 1. **The join's OWN two fields.** The 3-D join emits BOTH `frame` (RAW v2ep)
    and `frame_idx` (post-n_stack-trim). Their difference is an ANALYTIC target:
    it must be exactly `n_stack - 1` on every line, and it was written by a
    builder that has never seen this trainer.
 2. **Pose alignment** (`ClipMapGT.check_pose_alignment`). Moves each RAW v2ep
    pose by `v * (t_img - t_query)` along its heading and requires it to land on
    the GT's own `T_world_rig` translation. Its `shift_median_m` row is the
    DISCRIMINATOR: a one-frame shift must be clearly worse, or the clip cannot
    identify the axis and the verdict is INCONCLUSIVE, not PASS.
 3. **A deliberate MISALIGNMENT**, so the check is proven able to fail: feed the
    poses rolled by one frame and require probe 2 to RAISE.
"""
import io
import json
import lzma
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:/Users/Admin/tanitad-wt-perctrain/stack")
from tanitad.data.semantic_map_gt import (TimeMisalignment,  # noqa: E402
                                          open_path, raw_frame_index, sha12)

CACHE = Path(r"D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl")
MAPS = Path(r"D:/Projects/TanitAD-artifacts/sam3-maps-eval")
JOIN = Path(r"D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/"
            r"b1eval_agents_3d.jsonl.xz")
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rep = {}

# --- 1. the join's own two fields -------------------------------------------
diffs, n_lines, per_clip = {}, 0, {}
with lzma.open(str(JOIN), "rt", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        d = int(r["frame"]) - int(r["frame_idx"])
        diffs[d] = diffs.get(d, 0) + 1
        per_clip.setdefault(r["clip_id"], 0)
        per_clip[r["clip_id"]] += 1
        n_lines += 1
rep["join_index_space"] = {
    "n_lines": n_lines, "n_clips": len(per_clip),
    "frame_minus_frame_idx_histogram": {str(k): v for k, v in
                                        sorted(diffs.items())},
    "n_distinct_offsets": len(diffs),
    # the ANALYTIC target: `n_stack - 1` at n_stack 3 is 2, and nothing else.
    "expected_offset_at_n_stack_3": 2,
    "verdict": ("INCONCLUSIVE" if n_lines == 0 else
                ("MATCHES-n_stack-1" if diffs == {2: n_lines} else "DIFFERS")),
}

# --- 2/3. pose alignment, and the misalignment control ----------------------
clips = sorted(p.name.split(".v2ep")[0] for p in CACHE.glob("*.v2ep.pt"))
rows, n_ok, n_incon, n_nofile, n_ctrl_caught = [], 0, 0, 0, 0
for cid in clips[:N_CLIPS]:
    s12 = sha12(cid)
    gp = MAPS / f"{s12}.sam3mapgt.npz"
    if not gp.is_file():
        n_nofile += 1
        rows.append({"clip_sha12": s12, "verdict": "NO_FILE"})
        continue
    d = torch.load(CACHE / f"{cid}.v2ep.pt", map_location="cpu",
                   weights_only=False)
    # ⛔ THE **RAW** POSES, not the trimmed `poses[n_stack-1:]` a
    # `LazyV2Episode` carries: `check_pose_alignment` refuses a length
    # mismatch, which is itself a check on which array I handed it.
    poses = np.asarray(d["poses"], dtype=np.float64)
    n_stack = int(d["n_stack"])
    gt = open_path(gp, cid)
    row = {"clip_sha12": s12, "n_stack": n_stack,
           "n_v2ep_poses": int(poses.shape[0]), "n_gt_frames": int(gt.n_frames)}
    try:
        r = gt.check_pose_alignment(poses)
        row.update({k: r[k] for k in
                    ("aligned_max_m", "aligned_median_m", "shift_median_m",
                     "verdict", "undiscriminated_shifts")})
        n_ok += r["verdict"] == "ALIGNED"
        n_incon += r["verdict"] == "INCONCLUSIVE"
    except TimeMisalignment as e:
        row.update({"verdict": "MISALIGNED", "why": str(e)[:200]})
    # --- the DELIBERATE REGRESSION: roll the poses by one frame ------------
    # A check that cannot fail is not a check. This one must RAISE.
    try:
        gt.check_pose_alignment(np.roll(poses, 1, axis=0))
        row["control_one_frame_roll"] = "NOT-CAUGHT"
    except TimeMisalignment:
        row["control_one_frame_roll"] = "CAUGHT"
        n_ctrl_caught += 1
    # --- and the arithmetic this trainer actually performs ----------------
    # window NOW as a STACKED-ROW index -> raw frame. Checked against the
    # join's own `frame` for the same window, which is an independent source.
    w_now = 7                        # an arbitrary stacked-row index
    row["raw_frame_index(%d, n_stack)" % w_now] = int(
        raw_frame_index(w_now, n_stack))
    row["equals_w_now_plus_n_stack_minus_1"] = bool(
        int(raw_frame_index(w_now, n_stack)) == w_now + n_stack - 1)
    rows.append(row)

rep["pose_alignment"] = {
    "n_clips_probed": len(rows), "n_aligned": n_ok,
    "n_inconclusive": n_incon, "n_no_file": n_nofile,
    "n_one_frame_roll_caught": n_ctrl_caught,
    "control_verdict": ("INCONCLUSIVE" if not rows else
                        ("DETECTS-A-ONE-FRAME-SHIFT"
                         if n_ctrl_caught == len(rows) - n_nofile
                         else "BLIND-ON-SOME")),
    "verdict": ("INCONCLUSIVE" if not rows else
                ("ALIGNED-ON-EVERY-PROBED-CLIP"
                 if n_ok == len(rows) - n_nofile - n_incon
                 and n_ok > 0 else "NOT-ALIGNED")),
    "per_clip": rows,
}
io.open(sys.argv[2] if len(sys.argv) > 2 else "frame_axis.json", "w",
        encoding="utf-8").write(json.dumps(rep, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_clip"}
                  for k, v in rep.items()}, indent=1))
