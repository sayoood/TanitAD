"""Print the P-GRAD tables (markdown) from raw/pgrad.json: per term x group the share of the summed per-term gradient
norms and the projection share on the total gradient, the trunk-whole cosines, and the controls."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "raw" / "pgrad.json")
r = json.loads(p.read_text(encoding="utf-8"))
T = ["traj", "box3d", "agent", "map_hires", "tac_v6", "rest"]
S = r["summary"]
order = ["stem", "stage_layer1", "stage_layer2", "stage_layer3", "stage_layer4", "fuse", "x_shared_bev_025m",
         "x_planner_bev_pool"]
print("controls:", r["controls"])
print("\n| group (params) | " + " | ".join(T) + " | aux share (in-run def.) |")
print("|---|" + "---|" * (len(T) + 1))
for g in order:
    sh = S["share_of_term_norm_sum"][g]
    print(f"| {g} ({r['group_sizes'][g]:,}) | " + " | ".join(f"{sh[k]:.4f}" for k in T) +
          f" | {S['aux_share_conflict_def'][g]:.4f} |")
print("\nprojection share on the total gradient (sums to 1):")
print("| group | " + " | ".join(T) + " |")
print("|---|" + "---|" * len(T))
for g in order:
    ps = S["proj_share_on_total"][g]
    print(f"| {g} | " + " | ".join(f"{ps[k]:+.4f}" for k in T) + " |")
tw = S["trunk_whole"]
print("\ntrunk whole: norms", {k: round(v, 3) for k, v in tw["norm"].items()})
print("cos to total", {k: round(v, 3) for k, v in tw["cos_to_total"].items()})
print("proj share", {k: round(v, 4) for k, v in tw["proj_share_on_total"].items()})
print("cos pairs", {k: round(v, 3) for k, v in tw["cos_pairs"].items()})
