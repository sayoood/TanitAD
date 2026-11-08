import sys, os, math
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import gym_env as g, numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
sts = g.select_states(20,10)
cap = {}
orig = g.RewardEvaluator.cycle
def wrap(self, ctx, egos, objs, tl, plan):
    o = orig(self, ctx, egos, objs, tl, plan)
    if o["hard"]: cap.update(hard=o["hard"], step=o["hard_step"], egos=egos, objs=objs)
    return o
g.RewardEvaluator.cycle = wrap
for idx in [int(a) for a in sys.argv[1:]]:
    S = g.GymState(sts[idx]); ev = S.evaluator; e0 = S.ego0
    print("STATE", idx, sts[idx]["log"], "v0=%.2f" % e0.dynamic_car_state.rear_axle_velocity_2d.x, flush=True)
    a0 = ev.ego_areas([e0]*6)[0]
    print(" t0 areas:", {n: bool(a0[getattr(EgoAreaIndex,n)]) for n in ("MULTIPLE_LANES","NON_DRIVABLE_AREA","ONCOMING_TRAFFIC")})
    dm = ev.drivable
    idc = dm.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
    polys = [dm._geometries[i] if hasattr(dm,'_geometries') else None for i in idc]
    print(" n drivable polys", len(idc), "attrs", [a for a in dir(dm) if a.startswith('_') and not a.startswith('__')][:12])
    corners = list(e0.car_footprint.geometry.exterior.coords)[:4]
    U = unary_union([p for p in polys if p is not None]) if all(p is not None for p in polys) else None
    for c in corners:
        print("  corner", np.round(c,2), "in any drivable poly:", None if U is None else bool(U.contains(Point(c))), "dist to U:", None if U is None else round(U.distance(Point(c)),3))
    print(" center in U:", None if U is None else bool(U.contains(Point(e0.center.x,e0.center.y))))
    print(" pdm scores", np.round(S.pdm_scores,2).tolist())
    print(" planner0 own drivable==evaluator? radius evaluator=150, planner map_radius=", S.planner0._map_radius, flush=True)
    cap.clear()
    logs = S.rollout(int(np.argmax(S.pdm_scores)), "log")
    r = logs[-1]; print(" best-arm log rollout: cycles", len(logs), "hard", r["hard"], r["hard_step"])
    if cap.get("hard") == "collision" or cap.get("hard") == "ttc":
        e = cap["egos"][min(cap["step"],4)]; o = cap["objs"][min(cap["step"],4)]
        for ob in o:
            if e.car_footprint.geometry.intersects(ob.box.geometry):
                lx, ly = g._local(e, (ob.center.x, ob.center.y)); ov = getattr(ob,"velocity",None)
                print("  COLLIDED obj", ob.tracked_object_type.name, "local(x,y)=(%.1f,%.1f)" % (lx,ly), "ego v=%.2f" % e.dynamic_car_state.rear_axle_velocity_2d.x, "obj speed=%.2f" % (math.hypot(ov.x,ov.y) if ov else -1), "ego hd-obj hd=%.2f" % (e.rear_axle.heading-ob.center.heading))
    if cap.get("hard") == "off_drivable":
        e = cap["egos"][min(cap["step"],4)]
        print("  off_drivable ego at", round(e.rear_axle.x,1), round(e.rear_axle.y,1), "v=%.2f" % e.dynamic_car_state.rear_axle_velocity_2d.x,
              "corners outside U:", None if U is None else [bool(not U.contains(Point(c))) for c in list(e.car_footprint.geometry.exterior.coords)[:4]])
