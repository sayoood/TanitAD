"""Lead-agent census over the clips that ALREADY have cached DINOv3 features.

Decides whether n is sufficient for the perception probe, or whether more
clips must be DINOv3-encoded first.

LEAD DEFINITION (declared here, before any result):
  an agent is the LEAD at a frame iff
    cx > 0                      (ahead of the ego; +x fwd)
    |cy| <= HALF_LANE           (in the ego's corridor; +y LEFT)
    cls in VEHICLE_CLASSES      (a vehicle, not a pedestrian)
  and among those, the one with the SMALLEST cx.
`occ` is NOT used to select (it is reported, so the visible/occluded split
needs no re-derivation).
"""
import glob
import json
import lzma
import os
from collections import defaultdict

HALF_LANE = 1.8          # m; ~3.6 m lane
JOIN = "/home/nvidia/percprobe/raw/train2400_agents.jsonl.xz"
DINO = "/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl"

VEHICLE = None  # filled after we see the class census


def main():
    have = set(os.path.basename(p)[:-3] for p in glob.glob(DINO + "/*.pt"))

    cls_count = defaultdict(int)
    per_clip_frames = defaultdict(int)
    per_clip_lead = defaultdict(int)
    per_clip_lead_vis = defaultdict(int)
    n_lines = 0

    with lzma.open(JOIN, "rt") as f:
        for line in f:
            r = json.loads(line)
            c = r["clip_id"]
            if c not in have:
                continue
            n_lines += 1
            per_clip_frames[c] += 1
            best = None
            for a in r["agents"]:
                cls_count[a["cls"]] += 1
                if a["cx"] > 0 and abs(a["cy"]) <= HALF_LANE:
                    if best is None or a["cx"] < best["cx"]:
                        best = a
            if best is not None:
                per_clip_lead[c] += 1
                if best.get("occ", 1) == 0:
                    per_clip_lead_vis[c] += 1

    print("clips with dinov3 feats AND join lines: %d" % len(per_clip_frames))
    print("total labelled frames on those clips: %d" % n_lines)
    print()
    print("class census (all agents, all frames):")
    for k, v in sorted(cls_count.items(), key=lambda x: -x[1]):
        print("   %-24s %d" % (k, v))
    print()
    tot_lead = sum(per_clip_lead.values())
    tot_vis = sum(per_clip_lead_vis.values())
    print("frames WITH a lead (any):     %d  (%.1f%% of labelled frames)"
          % (tot_lead, 100.0 * tot_lead / max(n_lines, 1)))
    print("frames WITH a VISIBLE lead:   %d  (%.1f%%)"
          % (tot_vis, 100.0 * tot_vis / max(n_lines, 1)))
    print()
    for thr in (10, 30, 60, 120):
        k = sum(1 for c, v in per_clip_lead.items() if v >= thr)
        kv = sum(1 for c, v in per_clip_lead_vis.items() if v >= thr)
        print("clips with >= %3d lead frames: %3d   (visible-lead: %3d)"
              % (thr, k, kv))

    out = {
        "half_lane_m": HALF_LANE,
        "n_clips": len(per_clip_frames),
        "n_labelled_frames": n_lines,
        "n_lead_frames": tot_lead,
        "n_lead_frames_visible": tot_vis,
        "class_census": dict(cls_count),
        "per_clip_frames": dict(per_clip_frames),
        "per_clip_lead": dict(per_clip_lead),
        "per_clip_lead_visible": dict(per_clip_lead_vis),
    }
    os.makedirs("/home/nvidia/percprobe/raw", exist_ok=True)
    with open("/home/nvidia/percprobe/raw/lead_census.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote /home/nvidia/percprobe/raw/lead_census.json")


if __name__ == "__main__":
    main()
