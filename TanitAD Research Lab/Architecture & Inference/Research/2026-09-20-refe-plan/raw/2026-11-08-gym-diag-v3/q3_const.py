"""Q3: constant-target-speed policies on free-road states. usage: q3_const.py SAMPLER IDX OUT.json
Policy per 0.5 s cycle: centreline trajectory (planner0 centreline, same one the reward measures progress on) with target
speed vt held (accel<=2, decel<=3 m/s^2), vt in {1.5 m/s, 0.3,0.5,0.7,0.9,1.0 x vref}, vref = min(limit, max(v0,...)) -> we use
vref = limit and ALSO report the term decomposition. Log mode (free road: agents irrelevant). Reward evaluator unchanged."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"]="-1"; sys.dont_write_bytecode=True
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np, gym_env as g
from shapely.geometry import Point
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array, state_array_to_ego_states
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
sampler, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sel = g.select_states if sampler=="pilot" else g.select_states_stall
S = g.GymState(sel(20,10)[idx]); cl = S.planner0._centerline
def const_traj(ego, vt):
    v0=float(ego.dynamic_car_state.rear_axle_velocity_2d.x); s0=float(cl.project(Point(ego.rear_axle.x,ego.rear_axle.y)))
    tps=[TimePoint(ego.time_point.time_us+int(i*g.DT*1e6)) for i in range(41)]
    v=np.empty(41); v[0]=v0
    for i in range(1,41): v[i]=v[i-1]+np.clip(vt-v[i-1],-3.0*g.DT,2.0*g.DT)
    s=s0+np.concatenate([[0.0],np.cumsum(v[1:]*g.DT)])
    arr=np.zeros((41,StateIndex.size())); arr[0]=ego_state_to_state_array(ego)
    arr[1:,StateIndex.STATE_SE2]=cl.interpolate(s[1:],as_array=True); arr[1:,StateIndex.VELOCITY_X]=v[1:]
    arr[1:,StateIndex.ACCELERATION_X]=np.diff(v)/g.DT
    return InterpolatedTrajectory(state_array_to_ego_states(arr,tps,g.VEH))
VT=[None]
def pick(planner, pin, ego, policy):
    cur,_=pin.history.current_state
    return const_traj(cur, VT[0])
S._policy_pick = pick
res=dict(state=idx,sampler=sampler,limit=S.limit,v0=float(S.ego0.dynamic_car_state.rear_axle_velocity_2d.x),pdm_best_ret=None,runs={})
res["pdm_best"]=g.episode_return(S.rollout(S.pdm_best if hasattr(S,'pdm_best') else int(np.argmax(S.pdm_scores)),"log"))
targets=[("crawl1.5",1.5)]+[(f"f{f}",f*S.limit) for f in (0.3,0.5,0.7,0.9,1.0)]
for name,vt in targets:
    VT[0]=vt
    logs=S.rollout(-1,"log",policy="const",first_traj=const_traj(S.ego0,vt))
    res["runs"][name]=dict(vt=vt,ret=g.episode_return(logs),logs=[dict(c=r["cycle"],hard=r["hard"],reward=r["reward"],prog=r["progress_norm"],prog_legacy=r["progress_norm_legacy"],v=r["v"],q=r["q"],act=r["active"],qprod=r["qprod"],lead_gap=r["lead_gap"],conf=r["conf_ttc"]) for r in logs])
    print(name,vt,res["runs"][name]["ret"],flush=True)
json.dump(res,open(out,"w"))
print("Q3_DONE",flush=True)
