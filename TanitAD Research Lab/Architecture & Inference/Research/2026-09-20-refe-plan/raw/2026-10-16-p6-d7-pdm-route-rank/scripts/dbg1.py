import sys, os, json
sys.path.insert(0,"/workspace/ops/d7fix/pkg/refe"); sys.path.insert(0,"/workspace/ops/d7fix/pkg/code")
os.chdir("/workspace/ops/d7fix/pkg/refe")
import build_metric_cache as B, navsim_pdm_targets as PT, pdm_route_rank as R, p6_prep
import diag_d7_twin_route as D
from nuplan.planning.script import driverl_runtime_map_features as M
lg,tk=sys.argv[1],sys.argv[2]; K=int(sys.argv[3])
sc = B._scenarios("/workspace/data/navtrain_dbs", os.environ["NUPLAN_MAPS_ROOT"], lg, {tk})[tk]
S = PT._setup()
print(R.ensure(K, S["proc"]._pdm_closed))
parts = PT.metric_cache_parts(sc)
pc = S["proc"]._pdm_closed
print("D7_LAST", json.dumps(pc._d7_last, default=str))
init_in, init = S["proc"]._get_planner_inputs(sc)
ego,_ = init_in.history.current_state
anchor = R.anchor_from_ego(ego, M)
print("ego rear", ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading)
lane, lid = D._first_lane(parts, init.map_api)
print("first_lane_of_twin_parts", lid)
blk = pc._route_roadblock_dict[lane.get_roadblock_id()]
for e in blk.interior_edges:
    print(" edge", e.id, round(M._route_edge_anchor_score(e, anchor),3), "ncoords", len(M._edge_centerline_coords(e)))
print("corrected route ids (pdm)", list(pc._route_roadblock_dict.keys())[:6])
print("scenario route ids", [str(i) for i in init.route_roadblock_ids][:6])
print("---- centerline head", [(round(s.x,2),round(s.y,2)) for s in parts[3].discrete_path[:3]], "len", len(parts[3].discrete_path))
for e in blk.interior_edges:
    dp = e.baseline_path.discrete_path
    print(" lane", e.id, "n", len(dp), "head", [(round(s.x,2),round(s.y,2)) for s in dp[:2]], "tail", (round(dp[-1].x,2),round(dp[-1].y,2)))
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
roadblocks = list(pc._route_roadblock_dict.values()); ids=list(pc._route_roadblock_dict.keys())
for e in blk.interior_edges:
    start_idx = ids.index(e.get_roadblock_id())
    g = Dijkstra(e, list(pc._route_lane_dict.keys()))
    plan, found = g.search(roadblocks[start_idx:start_idx+30][-1])
    print(" dijkstra from", e.id, "found", found, "plan", [str(x.id) for x in plan][:8], "n", len(plan))
