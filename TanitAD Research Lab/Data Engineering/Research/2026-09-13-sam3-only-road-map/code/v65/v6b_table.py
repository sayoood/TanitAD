"""Section 11 table numbers: clip-vote MAP_A2 / MAP_B and the fused +-1 s bar for v6, v6z, v6r, v6n on both clips."""
import json
from pathlib import Path
SM = Path("/home/nvidia/sam3map")
for tag in ("v6", "v6z", "v6r", "v6n"):
    row = [tag]
    for c8 in ("73495082f98b", "4fbd97b6a4b7"):
        p = SM / f"sam3map_score_{c8}_{tag}.json"
        if not p.exists():
            row.append(f"{c8}: missing"); continue
        d = json.loads(p.read_text())
        cv, fu = d["SAM3_clipvote"], d["SAM3_fused_1s"]
        row.append(f"{c8}: A2 {cv['MAP_A2']['value']} (bar {round(fu['MAP_A2']['value'] + 0.05, 4)}) B {cv['MAP_B']['value']} E {cv['MAP_E']['value']} n_frames_A2 {cv['MAP_A2']['n']}")
    print(" | ".join(row))
for c8 in ("73495082f98b", "4fbd97b6a4b7"):
    for tag in ("v61_gt", "v61r_gt", "v61n_gt", "v61z_gt"):
        p = SM / f"sam3map_score_{c8}_{tag}.json"
        if p.exists():
            d = json.loads(p.read_text()); a = d["SAM3_GT_worldmap"]
            print(c8, tag, "GT worldmap A2", a["MAP_A2"]["value"], "B", a["MAP_B"]["value"], "C", a["MAP_C"]["value"], "E", a["MAP_E"]["value"])
