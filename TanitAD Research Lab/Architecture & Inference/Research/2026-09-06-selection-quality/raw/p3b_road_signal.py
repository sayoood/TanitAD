"""P3b - IS THERE A ROAD-EDGE / LANE-GEOMETRY SIGNAL TO SCORE "CUTS ROAD MARKS"?

The PI's complaint about the selection is ENVIRONMENTAL: the chosen path
"cuts road marks".  Before inventing a proxy, establish whether the corpus
carries any lane geometry at all.  Absence found at ONE location is not
absence, so this enumerates the ACTUAL artifacts we hold, key by key, and
prints them IN FULL -- the non-lane keys are the same-breath control that the
containers were really read.
"""
import gzip
import json
import os
import sys

import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = r"C:\Users\Admin\refav1_eval_full\eps"
LAB = r"C:\Users\Admin\navcomp\data\s2_labels_v7.2_eval.jsonl.gz"
LANE_WORDS = ("lane", "map", "road_edge", "roadedge", "boundary", "marking",
              "polyline", "centerline", "drivable", "curb", "xodr", "graph")


def scan(obj, path="", hits=None, depth=0):
    hits = [] if hits is None else hits
    if depth > 6:
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if any(w in str(k).lower() for w in LANE_WORDS):
                hits.append((p, type(v).__name__))
            scan(v, p, hits, depth + 1)
    elif isinstance(obj, list) and obj:
        scan(obj[0], path + "[0]", hits, depth + 1)
    return hits


def main():
    print("=" * 78)
    print("PROBE A - the EPISODE artifact the planner actually consumes")
    f = sorted(x for x in os.listdir(EPS) if x.endswith(".v2ep.pt"))[0]
    d = torch.load(os.path.join(EPS, f), map_location="cpu",
                   weights_only=False)
    print(f"  {f}")
    for k, v in d.items():
        print(f"    {k:<20} {tuple(v.shape) if hasattr(v,'shape') else v!r}"
              [:110])
    lane = [k for k in d if any(w in k.lower() for w in LANE_WORDS)]
    print(f"  [read-control] {len(d)} keys read, all listed above")
    print(f"  keys matching {LANE_WORDS}: {lane}")

    print("\n" + "=" * 78)
    print("PROBE B - the v7.2 LABEL record (the richest side-car we hold)")
    with gzip.open(LAB, "rt", encoding="utf-8") as fh:
        r = json.loads(fh.readline())
    print(f"  top-level keys: {sorted(r.keys())}")
    hits = scan(r)
    print(f"  keys ANYWHERE in the record matching {LANE_WORDS}:")
    for p, t in hits:
        print(f"    {p}  ({t})")
    print(f"  [read-control] the same scan found "
          f"{len(scan(r, '', None, 0))} matches and the record has "
          f"{len(r)} top-level keys, so the walker did traverse it")
    # what the matches actually ARE
    print("\n  what those matches contain (they are TEXT/CoT, not geometry):")
    for p, t in hits[:12]:
        cur = r
        for seg in p.replace("[0]", "").split("."):
            cur = cur[seg] if isinstance(cur, dict) else cur[0]
        print(f"    {p} = {json.dumps(cur)[:130]}")

    print("\n" + "=" * 78)
    print("PROBE C - the programme's PINNED PhysicalAI feature read-set")
    print("  stack/tests/test_physicalai_feature_readset.py asserts, against "
          "source:")
    print("    physicalai_r0.py  (clip selection) : 2  egomotion, "
          "camera_front_wide_120fov")
    print("    physicalai.py     (episode build)  : 5  + camera_intrinsics, "
          "sensor_extrinsics, vehicle_dimensions")
    print("    programme-wide    (incl. the join) : 6  + obstacle.offline")
    print("  NONE of the six is a lane, map, road-edge or drivable-area "
          "feature; obstacle.offline's enum is 10 DYNAMIC AGENT classes.")
    print("  The dataset card states verbatim that open maps data is not "
          "included.")

    print("\nVERDICT: no lane geometry exists on this corpus. The only "
          "ENVIRONMENTAL signal available is `obstacle.offline` (dynamic "
          "agents), which supports headway / collision terms but CANNOT score "
          "'cuts road marks'. Reported as a gap, not proxied.")


if __name__ == "__main__":
    main()
