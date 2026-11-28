"""DERIVATION-STATE exploration (USED states only: stall 7, 10, 11, 15 of the M-GYM-2 stall set). Not evidence.
usage: explore_corridor.py IDX OUT.json
Runs the 24 arms in log mode with the ttc predicate OFF (pilot_v3_diag/gym_env_cf.py, TTC_BRAKE=-1) and, per tick, evaluates the
legacy ttc predicate and candidate corridor predicates as SHADOW flags (never terminating). Records, per arm, the first tick of each
flag and the tick/type of the first real hard event, so each candidate can be scored against 'does this arm really collide'."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v3_diag")
import numpy as np
import gym_env_cf as g
from shapely import affinity
from shapely.geometry import Point
from shapely.ops import unary_union
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

idx, out = int(sys.argv[1]), sys.argv[2]
g.TTC_BRAKE = -1.0
S = g.GymState(g.select_states_stall(20, 10)[idx])
ev = S.evaluator
route_polys = []
for lid in ev.route_lane_ids:
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        try:
            o = ev.map_api.get_map_object(str(lid), layer)
        except Exception:
            o = None
        if o is not None:
            route_polys.append(o.polygon); break
ROUTE = unary_union(route_polys) if route_polys else None
print("built", idx, "route polys", len(route_polys), flush=True)
orig = ev.cycle
TICK = [0]
FL = {}


def shadow(ego, objs, tick):
    v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    if v <= 0.05:
        return
    hd = ego.rear_axle.heading; ux, uy = math.cos(hd), math.sin(hd)
    epoly = ego.car_footprint.geometry; ec = (ego.center.x, ego.center.y)
    s0 = float(ev.cl.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
    L = 8.0 + v
    seg = ev.cl.substring(max(0.0, s0 - 2.0), min(s0 + L, ev.cl.length))
    corr = {"c115": seg.buffer(1.15, cap_style=2), "c150": seg.buffer(1.5, cap_style=2), "c200": seg.buffer(2.0, cap_style=2)}
    for o in objs:
        if (o.center.x - ec[0]) ** 2 + (o.center.y - ec[1]) ** 2 > 3600 or epoly.intersects(o.box.geometry):
            continue
        lx, ly = g._local(ego, (o.center.x, o.center.y))
        ov = getattr(o, "velocity", None); ovx, ovy = (ov.x, ov.y) if ov is not None else (0.0, 0.0)
        if lx < 0 or (v - (ovx * ux + ovy * uy)) < 0.3:
            continue
        for dt in (0.2, 0.4, 0.6, 0.8, 1.0):
            op = affinity.translate(o.box.geometry, ovx * dt, ovy * dt)
            if affinity.translate(epoly, ux * v * dt, uy * v * dt).intersects(op):
                FL.setdefault("legacy", tick)
                for n, c in corr.items():
                    if op.intersects(c):
                        FL.setdefault(n, tick)
                    if o.box.geometry.intersects(c):
                        FL.setdefault(n + "_now", tick)
                if ROUTE is not None:
                    if op.intersects(ROUTE):
                        FL.setdefault("route", tick)
                    if o.box.geometry.intersects(ROUTE):
                        FL.setdefault("route_now", tick)
                    if ROUTE.contains(Point(o.center.x, o.center.y)):
                        FL.setdefault("route_ctr", tick)
                break


def wrap(ctx, egos, objs_steps, tl, new_plan):
    for e, ob in zip(egos, objs_steps):
        shadow(e, ob, TICK[0]); TICK[0] += 1
    return orig(ctx, egos, objs_steps, tl, new_plan)


ev.cycle = wrap
res = dict(state=idx, rows=[])
for k in range(24):
    TICK[0] = 0; FL.clear()
    logs = S.rollout(k, "log")
    h = [(c["cycle"], c["hard"], c["hard_step"]) for c in logs if c["hard"]]
    hard_tick = (h[0][0] * 5 + h[0][2]) if h else None
    res["rows"].append(dict(k=k, ret=g.episode_return(logs), hard=h[0] if h else None, hard_tick=hard_tick, flags=dict(FL)))
    print(k, res["rows"][-1], flush=True)
    json.dump(res, open(out, "w"))
print("EXPLORE_DONE", flush=True)
