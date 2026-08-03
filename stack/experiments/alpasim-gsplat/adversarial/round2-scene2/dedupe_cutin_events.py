#!/usr/bin/env python3
"""Are the "44 cut-in events" 44 EVENTS, or one crossing counted once per start tick?

scene_geometry.find_cutins emits one record per *starting row* `a` that is outside the
lane and within `win` ticks of a crossing. The inner loop breaks at the first crossing,
but the OUTER loop does not — so a single lane entry produces one record for every tick
of the approach. This script merges overlapping [k_start, k_end] intervals per track and
reports the DISTINCT crossing count, which is the number a power statement needs.

Also checks the emitted evidence list against the headline count: `cutins_ALL` is written
as `cut_all[:40]` with no truncation note, so a summary that says 44 ships 40 rows.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def merge_per_track(events):
    by = {}
    for c in events:
        by.setdefault(c["track"], []).append((int(c["k_start"]), int(c["k_end"])))
    out = {}
    for t, iv in by.items():
        iv.sort()
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        out[t] = merged
    return out


def main(path):
    g = json.loads(Path(path).read_text())
    s = g["summary"] if "summary" in g else g
    ev = s.get("cutins_ALL", [])
    rend = [c for c in ev if c.get("renderable")]
    print(f"scene                      {s.get('scene')}")
    print(f"headline n_cutin_events_ALL        {s.get('n_cutin_events_ALL')}")
    print(f"rows actually emitted in cutins_ALL {len(ev)}   <- cut_all[:40], no note")
    print(f"headline n_cutin_events_RENDERABLE {s.get('n_cutin_events_RENDERABLE')}")
    print(f"renderable rows in the emitted list {len(rend)}")
    print(f"per-track record counts     {dict(Counter(c['track'] for c in ev))}")
    ma, mr = merge_per_track(ev), merge_per_track(rend)
    print(f"DISTINCT crossings, ALL         {sum(len(v) for v in ma.values())}  {ma}")
    print(f"DISTINCT crossings, RENDERABLE  {sum(len(v) for v in mr.values())}  {mr}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         Path(__file__).resolve().parents[2] / "alpasim-gsplat" / "results" /
         "scene2-realclose" / "SCENE2_GEOMETRY_renderable.json")
