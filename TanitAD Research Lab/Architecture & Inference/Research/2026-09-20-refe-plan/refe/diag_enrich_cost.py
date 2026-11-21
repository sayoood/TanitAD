"""What does the lane-graph enrichment COST, now that it is measured to change nothing?

A no-op that is free can stay for future consumers. A no-op that costs a second per frame is a tax
on every bank build, and the honest thing is to price it before deciding."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
anchor = AR._anchor_from_ego(smps[step].ego_state)
ts = []
for _ in range(5):
    t = time.time()
    SP.enrich_lane_graph(sd, sc.map_api, anchor, init.route_roadblock_ids)
    ts.append(time.time() - t)
best = min(ts)
print(f"enrich_lane_graph: min {best*1000:.0f} ms  median {sorted(ts)[2]*1000:.0f} ms over 5")
print(f"  over a 654-frame rank that is {best*654/60:.1f} min of pure tax")
print(f"  as a share of the measured ~12 s/frame bank cost: {100*best/12.0:.1f} %")
