"""Print the LiDAR checks of GT world-map variants: full window, camera-coverage only, mirrored control.
Usage: score_table.py <c8> <label=tag> [...]   (reads /home/nvidia/sam3map/sam3map_score_<c8>_<tag>.json)"""
import json, sys
from pathlib import Path

SM = Path("/home/nvidia/sam3map")
K = ("MAP_A2", "MAP_B", "MAP_C", "MAP_E")


def fmt(r):
    return " ".join(f"{k[4:]} {r[k]['value'] if k in r else '-'}" for k in K)


c8 = sys.argv[1]
for arg in sys.argv[2:]:
    label, tag = arg.split("=", 1)
    p = SM / f"sam3map_score_{c8}_{tag}.json"
    if not p.exists():
        print(f"{label:24s} missing {p.name}")
        continue
    d = json.loads(p.read_text())
    a = d["SAM3_GT_worldmap"]; o = d["observed_only"]["SAM3_GT_worldmap"]; m = d["SAM3_GT_worldmap_MIRRORED"]
    print(f"{label:24s} {fmt(a)} | obs {fmt(o)} | mirror {fmt(m)} | veh-on-walk {d['GT_worldmap_vehicle_centres_on_sidewalk_verge']['value']}")
