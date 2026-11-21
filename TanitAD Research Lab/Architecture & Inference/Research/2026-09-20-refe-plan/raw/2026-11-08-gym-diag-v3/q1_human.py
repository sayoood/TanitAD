"""Q1: compare the sim ego with the logged human ego around an event, and classify the colliding contact.
usage: q1_human.py SAMPLER IDX ARM TOKSUFFIX[,..]
prints: human ego speed/pose and relative pose of the named object(s) every 0.5 s of the log; collision classifier output (ctype, areas) for ARM in log mode."""
import os, sys, math, json
os.environ["CUDA_VISIBLE_DEVICES"]="-1"; sys.dont_write_bytecode=True
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np, gym_env as g
sampler,idx,arm,toks=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4].split(",")
S=g.GymState((g.select_states if sampler=="pilot" else g.select_states_stall)(20,10)[idx]); ev=S.evaluator
print("limit",S.limit,"v0",S.ego0.dynamic_car_state.rear_axle_velocity_2d.x)
n=min(S.sc.get_number_of_iterations(),101)
for i in range(0,n,5):
    e=S.sc.get_ego_state_at_iteration(i); obs=S.sc.get_tracked_objects_at_iteration(i).tracked_objects.tracked_objects
    line="t=%.1f human v=%.2f a=%.2f xy=(%.1f,%.1f)"%(i*0.1,e.dynamic_car_state.rear_axle_velocity_2d.x,e.dynamic_car_state.rear_axle_acceleration_2d.x,e.rear_axle.x,e.rear_axle.y)
    for o in obs:
        if any(str(o.track_token).endswith(t) for t in toks):
            lx,ly=g._local(e,(o.center.x,o.center.y)); v=o.velocity
            line+=" | obj %s lx=%.1f ly=%.1f v=%.1f gap=%.2f"%(str(o.track_token)[-6:],lx,ly,math.hypot(v.x,v.y),e.car_footprint.geometry.distance(o.box.geometry))
    print(line)
# collision classifier trace for the sim arm
orig=g.get_collision_type; calls=[]
def wrap(ego_arr, epoly, o, opoly):
    r=orig(ego_arr,epoly,o,opoly); calls.append((str(o.track_token)[-6:],str(r),float(ego_arr[0]),float(ego_arr[1]))); return r
g.get_collision_type=wrap
origc=ev.cycle
def wc(ctx,egos,objs,tl,np_):
    areas=ev.ego_areas([ctx["prev_ego"]]+egos)[1:]
    r=origc(ctx,egos,objs,tl,np_)
    if r["hard"]=="collision":
        j=r["hard_step"]; print("COLLISION at step",j,"areas flags",{k.name:bool(areas[j,k]) for k in g.EgoAreaIndex}); print("classifier calls",calls[-3:])
    return r
ev.cycle=wc
logs=S.rollout(arm,"log")
print("arm",arm,"hard",[(c["cycle"],c["hard"],c["hard_step"]) for c in logs if c["hard"]])
