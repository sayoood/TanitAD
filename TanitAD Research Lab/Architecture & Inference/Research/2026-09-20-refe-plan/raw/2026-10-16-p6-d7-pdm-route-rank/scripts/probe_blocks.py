"""Where is DriveRL's start roadblock relative to PDM's, on the twins where they disagree?  Read-only."""
import sys, os, json
sys.path.insert(0, "/workspace/ops/d7fix/pkg/refe"); sys.path.insert(0, "/workspace/ops/d7fix/pkg/code")
os.chdir("/workspace/ops/d7fix/pkg/refe")
import build_metric_cache as B, navsim_pdm_targets as PT, pdm_route_rank as R
sys.path.insert(0, "/workspace/dz/DriveZero/DriveRL/src")
from driverl.nuplan.route_correction import correct_route_roadblock_ids
from nuplan.planning.script import driverl_runtime_map_features as M
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
S = PT._setup(); pdm = S["proc"]._pdm_closed
out = []
for fn in sys.argv[1:]:
    for t in json.load(open(fn))["twins"]:
        if "error" in t or t["start_block_agrees"]:
            continue
        sc = B._scenarios("/workspace/data/navtrain_dbs", os.environ["NUPLAN_MAPS_ROOT"], t["log"], {t["token"]})[t["token"]]
        pin, init = S["proc"]._get_planner_inputs(sc)
        ego, _ = pin.history.current_state
        anchor = R.anchor_from_ego(ego, M)
        pdm.initialize(init); pdm._iteration = 0; pdm._route_roadblock_correction(ego)
        pdm._drivable_area_map = PDMDrivableMap.from_simulation(pdm._map_api, ego, pdm._map_radius)
        ids_pdm = list(pdm._route_roadblock_dict.keys())
        rids = correct_route_roadblock_ids(ego, init.map_api, [str(i) for i in init.route_roadblock_ids])
        lane0 = pdm._get_starting_lane(ego)
        def best(bid):
            b = M._route_roadblock_for_id(init.map_api, bid)
            sc_ = [(M._route_edge_anchor_score(e, anchor), e) for e in b.interior_edges if M._edge_centerline_coords(e)]
            return round(min(s for s, _ in sc_), 2) if sc_ else None
        rec = {"k": t["k"], "token": t["token"][:8], "pdm_block": t["pdm_block"], "driverl_block": t["driverl_start_block"],
               "idx_in_pdm_route": {"pdm_block": ids_pdm.index(t["pdm_block"]) if t["pdm_block"] in ids_pdm else None, "driverl_block": ids_pdm.index(t["driverl_start_block"]) if t["driverl_start_block"] in ids_pdm else None},
               "idx_in_driverl_route": {"pdm_block": rids.index(t["pdm_block"]) if t["pdm_block"] in rids else None, "driverl_block": rids.index(t["driverl_start_block"]) if t["driverl_start_block"] in rids else None},
               "driverl_best_score": {"pdm_block": best(t["pdm_block"]), "driverl_block": best(t["driverl_start_block"])},
               "ego_in_pdm_block": bool(lane0.get_roadblock_id() == t["pdm_block"]), "routes_equal": ids_pdm == rids, "n_route_pdm": len(ids_pdm), "n_route_driverl": len(rids),
               "pdm_block_type": type(pdm._route_roadblock_dict[t["pdm_block"]]).__name__}
        out.append(rec); print(json.dumps(rec), flush=True)
