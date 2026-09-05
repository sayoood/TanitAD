# -*- coding: utf-8 -*-
"""Pick the reel's committed stills by a MEASURED property, never by eye.

Per clip: the camera-on frame with the LARGEST GROUND-TRUTH lateral extent --
the moment the road most demands a turn. That is a property of the ground truth
alone, so the choice cannot be made by which arm happens to look worst.

One extra still is chosen by the arms and SAID to be: the frame carrying the
largest peak lateral g of any arm, i.e. the worst friction-circle excursion in
the reel.

ASCII only in print().
"""
import io, json, os, shutil, sys

OUT_DIR = sys.argv[1]
IDX = sys.argv[2]
FRAMES = sys.argv[3]
os.makedirs(OUT_DIR, exist_ok=True)

ix = json.load(io.open(IDX, encoding="utf-8"))
rows = ix["frames"]
stills, seen = [], set()


def take(r, name, kind, why):
    src = os.path.join(FRAMES, "f_%06d.png" % int(r["frame"]))
    if not os.path.exists(src):
        print("MISSING frame file:", src)
        return
    dst = os.path.join(OUT_DIR, name)
    shutil.copyfile(src, dst)
    rec = {k: v for k, v in r.items()}
    rec.update(still=name, still_kind=kind, rule=why)
    stills.append(rec)
    print("  %-46s frame %5d  %s" % (name, r["frame"], why))


for card in ("intro", "outro"):
    c = [r for r in rows if r.get("card") == card]
    if c:
        take(c[0], "00_card_%s.png" % card, "card",
             "the card, kept for the n and the estimator it states")

clips = []
for r in rows:
    if r.get("kind") == "clip" and r.get("clip_id") not in seen:
        seen.add(r["clip_id"])
        clips.append(r["clip_id"])

for i, cid in enumerate(clips, 1):
    cand = [r for r in rows if r.get("kind") == "clip" and r["clip_id"] == cid
            and r.get("camera")]
    if not cand:
        print("  clip %s has no camera-on frame" % cid[:8])
        continue
    b = max(cand, key=lambda r: r.get("gt_lat_extent_m", 0.0))
    take(b, "%02d_%s_gtlat%.2fm.png" % (i, cid[:8], b.get("gt_lat_extent_m", 0.0)),
         "clip", "largest GROUND-TRUTH lateral extent among this clip's "
                 "camera-on frames (a GT-only rule)")

worst, worst_g = None, -1.0
for r in rows:
    if r.get("kind") != "clip" or not r.get("arms"):
        continue
    if any(v.get("kamm_over") for v in r["arms"].values()):
        # the ADE is per arm; rank by the arm count outside the circle, then ADE
        n_out = sum(1 for v in r["arms"].values() if v.get("kamm_over"))
        score = n_out + max(v["ade_m"] for v in r["arms"].values()) / 100.0
        if score > worst_g:
            worst, worst_g = r, score
if worst is not None:
    take(worst, "99_worst_friction_circle.png", "clip",
         "CHOSEN BY THE ARMS, and said so: the frame where the most arms leave "
         "the friction circle (mu 0.7). Not a GT-only rule -- it is the "
         "illustration of the finding, not evidence for it")

io.open(os.path.join(OUT_DIR, "STILLS.json"), "w", encoding="utf-8").write(
    json.dumps({"tool": "pick_stills_arms.py", "step": ix.get("step"),
                "arms": ix.get("arms"), "arm_geometry": ix.get("arm_geometry"),
                "fps": ix.get("fps"), "n_frames_in_reel": ix.get("n_frames"),
                "arms_dir": ix.get("arms_dir"),
                "shared_verification": ix.get("shared_verification"),
                "gt_control": {k: v for k, v in (ix.get("gt_control") or {}).items()
                               if k != "per_window"},
                "stats": ix.get("stats"), "stills": stills}, indent=1))
print("wrote %d stills + STILLS.json to %s" % (len(stills), OUT_DIR))
