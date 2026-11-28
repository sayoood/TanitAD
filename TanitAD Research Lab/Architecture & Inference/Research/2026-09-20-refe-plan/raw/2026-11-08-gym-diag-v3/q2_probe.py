"""Q2 probe: precision of the q_stall excuses. usage: q2_probe.py SAMPLER IDX OUT.jsonl
Replays, per standing state, {stop policy, pdm_best arm, sweep0.5 arm} x {log, idm}; after every cycle recomputes the
stall precondition EXACTLY as gym_env.RewardEvaluator.cycle does (all last-20-tick v<0.5), then records the excuse
conditions with details and an independent ground-truth reading (strict ego-width swept corridor, object ahead, type, speed)."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np
import gym_env as g
from shapely import affinity
from shapely.geometry import Point
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
sampler, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sel = g.select_states if sampler == "pilot" else g.select_states_stall
st = sel(20, 10)[idx]
S = g.GymState(st); ev = S.evaluator
EW = 2.297 / 2   # Pacifica width/2 (m)
orig = ev.cycle
ROWS = []
CUR = {}
def vel(o):
    v = getattr(o, "velocity", None); return (v.x, v.y) if v is not None else (0.0, 0.0)
def wrap(ctx, egos, objs_steps, tl, new_plan):
    r = orig(ctx, egos, objs_steps, tl, new_plan)
    if r["hard"] is not None: return r
    ego, objs = egos[-1], objs_steps[-1]
    stalled = all(x < 0.5 for x in list(ctx["v_hist"])[-20:])
    v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    row = dict(arm=CUR["arm"], mode=CUR["mode"], cycle=len(ROWS_ARM), v=v, stalled=stalled, q_stall=r["q"]["q_stall"], q_free=r["q"]["q_free"],
               lead_gap=r["lead_gap"], conf_ttc=r["conf_ttc"], lim=S.limit)
    ROWS_ARM.append(1)
    if stalled:
        s_end = float(ev.cl.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
        L = max(2.5 * v, 20.0)
        sub = ev.cl.substring(s_end, min(s_end + L, ev.cl.length))
        ahead = sub.buffer(1.5, cap_style=2); strict = sub.buffer(EW, cap_style=2)
        near = [o for o in objs if (o.center.x-ego.center.x)**2+(o.center.y-ego.center.y)**2 < 3600]
        # lead
        lead = None
        for o in near:
            lx, ly = g._local(ego, (o.center.x, o.center.y))
            if lx > 0 and ahead.intersects(o.box.geometry):
                gp = ego.car_footprint.geometry.distance(o.box.geometry)
                if lead is None or gp < lead["gap"]:
                    ux, uy = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading); vx, vy = vel(o)
                    lead = dict(typ=str(o.tracked_object_type).split(".")[-1], gap=gp, lx=lx, ly=ly, vlon=vx*ux+vy*uy, speed=math.hypot(vx, vy),
                                strict=bool(strict.intersects(o.box.geometry)), lat_edge=float(sub.distance(o.box.geometry)))
        row["lead"] = lead
        # conflicts (every qualifying object)
        confs = []
        for o in near:
            vx, vy = vel(o)
            if math.hypot(vx, vy) < 0.3 or ahead.intersects(o.box.geometry): continue
            for t in (0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0):
                if ahead.intersects(affinity.translate(o.box.geometry, vx*t, vy*t)):
                    lx, ly = g._local(ego, (o.center.x, o.center.y))
                    ux, uy = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading)
                    ts = None
                    for t2 in (0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0):
                        if strict.intersects(affinity.translate(o.box.geometry, vx*t2, vy*t2)): ts = t2; break
                    confs.append(dict(typ=str(o.tracked_object_type).split(".")[-1], t=t, t_strict=ts, lx=lx, ly=ly, speed=math.hypot(vx,vy),
                                      vlon=vx*ux+vy*uy, dist=float(ego.car_footprint.geometry.distance(o.box.geometry)))); break
        row["confs"] = confs
        # red
        red = ev.red_lane_ids(tl)
        reds = []
        if red:
            for lc in ev.map_api.get_proximal_map_objects(g.Point2D_from(ahead.centroid), 30.0, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]:
                if lc.id in red and ahead.intersects(lc.polygon):
                    reds.append(dict(id=str(lc.id), in_route=str(lc.id) in set(map(str, ev.route_lane_ids)), dist=float(ego.car_footprint.geometry.distance(lc.polygon)),
                                     strict=bool(strict.intersects(lc.polygon))))
        row["reds"] = reds
        row["n_red_total"] = len(red)
    ROWS.append(row)
    return r
ev.cycle = wrap
f = open(out, "w")
arms = [("stop", None), ("pdm", int(np.argmax(S.pdm_scores))), ("sw0.5", 15)]
for mode in ("log", "idm"):
    for name, k in arms:
        CUR.update(arm=name, mode=mode); ROWS_ARM = []; ROWS.clear()
        if name == "stop":
            S.rollout(-1, mode, policy="stop", first_traj=S.first_traj_for("stop"), timer=g.Timer())
        else:
            S.rollout(k, mode)
        for r in ROWS: f.write(json.dumps(dict(r, state=idx, sampler=sampler), default=str) + "\n")
        f.flush(); print(mode, name, len(ROWS), "stalled", sum(1 for r in ROWS if r["stalled"]), flush=True)
print("Q2_DONE", flush=True)
