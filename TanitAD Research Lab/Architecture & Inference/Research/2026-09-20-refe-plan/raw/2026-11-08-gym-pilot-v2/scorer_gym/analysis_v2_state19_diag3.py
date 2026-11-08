import sys, numpy as np
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import gym_env as g
sts=g.select_states(20,10)
S=g.GymState(sts[19]); ev=S.evaluator
k=int(np.argmax(S.pdm_scores))
print("k",k,"slack",ev.map_slack_m,"t0",ev.t0_overshoot_m,"MAX",g.MAX_MAP_SLACK_M)
orig=ev.cycle
rec=[]
def wrap(ctx,egos,objs,tl,np_):
    rec.append([round(ev.footprint_overshoot(e),3) for e in egos]); return orig(ctx,egos,objs,tl,np_)
ev.cycle=wrap
r=S.rollout(k,"log",horizon_s=1.0)
print("sim overshoot per tick, per cycle:",rec)
print([(x["cycle"],x["hard"],x.get("hard_step")) for x in r])
n=min(S.sc.get_number_of_iterations(),100)
print("human 0..1.5s per tick",[round(ev.footprint_overshoot(S.sc.get_ego_state_at_iteration(i)),3) for i in range(0,16)])
print("human max over every-2nd",max(ev.footprint_overshoot(S.sc.get_ego_state_at_iteration(i)) for i in range(0,n,2)),"all ticks",max(ev.footprint_overshoot(S.sc.get_ego_state_at_iteration(i)) for i in range(0,n)))
