"""Q1 tracer. usage: trace_event.py SAMPLER IDX MODE ARMS(comma) OUT.jsonl [ttc_off]
Imports pilot_v2 gym_env READ-ONLY; wraps RewardEvaluator.cycle to log per-tick ego + nearby objects and the exact
object/predicate that triggers each hard event. ttc_off=1 -> counterfactual: skip the ttc predicate (own copy of the predicate result is ignored by re-running cycle with ttc objects filtered)."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np
import gym_env as g
from shapely import affinity
sampler, idx, mode, arms, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4].split(",")], sys.argv[5]
sel = g.select_states if sampler == "pilot" else g.select_states_stall
st = sel(20, 10)[idx]
S = g.GymState(st); ev = S.evaluator
print("built", S.limit, "slack", ev.map_slack_m, "t0over", ev.t0_overshoot_m, flush=True)
orig = ev.cycle
REC = []
def ttc_probe(ego, objs):
    """re-implementation of the ttc predicate returning (obj, dt, lx, closing) for EVERY object that would trigger"""
    v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x); hd = ego.rear_axle.heading
    ux, uy = math.cos(hd), math.sin(hd); epoly = ego.car_footprint.geometry; ec = (ego.center.x, ego.center.y)
    res = []
    for o in objs:
        if (o.center.x-ec[0])**2+(o.center.y-ec[1])**2 > 3600: continue
        if epoly.intersects(o.box.geometry): continue
        lx, ly = g._local(ego, (o.center.x, o.center.y))
        ov = getattr(o, "velocity", None); ovx, ovy = (ov.x, ov.y) if ov is not None else (0.0, 0.0)
        cl = v - (ovx*ux+ovy*uy)
        if lx < 0 or cl < 0.3: continue
        for dt in (0.2,0.4,0.6,0.8,1.0):
            if affinity.translate(epoly, ux*v*dt, uy*v*dt).intersects(affinity.translate(o.box.geometry, ovx*dt, ovy*dt)):
                res.append(dict(tok=str(o.track_token), typ=str(o.tracked_object_type), lx=lx, ly=ly, ov=math.hypot(ovx,ovy), closing=cl, dt=dt,
                                gap=float(epoly.distance(o.box.geometry)))); break
    return res
def wrap(ctx, egos, objs_steps, tl, new_plan):
    cyc = len(ARM["rows"])
    ticks = []
    for j, (e, ob) in enumerate(zip(egos, objs_steps)):
        v = float(e.dynamic_car_state.rear_axle_velocity_2d.x)
        a = float(e.dynamic_car_state.rear_axle_acceleration_2d.x)
        near = []
        for o in ob:
            lx, ly = g._local(e, (o.center.x, o.center.y))
            if -6 < lx < 45 and abs(ly) < 12:
                ov = getattr(o, "velocity", None)
                near.append(dict(tok=str(o.track_token)[-6:], typ=str(o.tracked_object_type).split(".")[-1], lx=round(lx,2), ly=round(ly,2),
                                 v=round(math.hypot(ov.x, ov.y),2) if ov is not None else None,
                                 gap=round(float(e.car_footprint.geometry.distance(o.box.geometry)),2), hit=bool(e.car_footprint.geometry.intersects(o.box.geometry))))
        near.sort(key=lambda d: d["gap"])
        ticks.append(dict(j=j, v=round(v,2), a=round(a,2), xy=(round(e.rear_axle.x,2), round(e.rear_axle.y,2)), over=round(ev.footprint_overshoot(e),3),
                          ttc=[{k:(round(x,2) if isinstance(x,float) else x) for k,x in d.items()} for d in ttc_probe(e, ob)], near=near[:4]))
    r = orig(ctx, egos, objs_steps, tl, new_plan)
    ARM["rows"].append(dict(cycle=cyc, hard=r["hard"], hard_step=r["hard_step"], reward=r["reward"], lead_gap=r["lead_gap"], ticks=ticks))
    return r
ev.cycle = wrap
f = open(out, "w")
for k in arms:
    ARM = dict(k=k, mode=mode, rows=[])
    logs = S.rollout(k, mode)
    ARM["ret"] = g.episode_return(logs); ARM["hard"] = [(c["cycle"], c["hard"], c["hard_step"]) for c in logs if c["hard"]]
    f.write(json.dumps(ARM, default=str) + "\n"); f.flush()
    print("arm", k, ARM["ret"], ARM["hard"], flush=True)
print("TRACE_DONE", flush=True)
