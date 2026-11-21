import os,sys,json,pickle,lzma
os.environ.update({"REFE_SIM_HZ":"10","NUPLAN_MAPS_ROOT":"E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0","OMP_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":"-1"})
sys.path[:0]=["C:/Users/Admin/n1_run/w/pk/refe"]
import numpy as np
import navsim_pdm_targets as PT, navtrain_scenarios as NS
PT._setup()
tok="2b482f615de15efb"; log=[json.loads(l)["log"] for l in open("n1t_only.jsonl") if tok in l][0]
dbs=NS.index_dbs(db_root="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
mc=pickle.loads(lzma.open(f"F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/{log}/unknown/{tok}/metric_cache.pkl","rb").read())
sc=next(NS.build_scenarios_for_log(dbs[log],[tok],history_rows=1,future_rows=80))
tr0=sc.get_tracked_objects_at_iteration(0).tracked_objects.tracked_objects
print("ours raw t0 objs",len(tr0))
def f32(x): return float(np.float32(x))==float(x)
print("ours center f32-representable:",[f32(o.center.x) and f32(o.center.y) for o in tr0[:8]])
print("ours center", [(o.center.x,o.center.y) for o in tr0[:3]])
ob=mc.observation
# cache: unique objects
uo=ob._unique_objects
k=list(uo)[:3]; 
print("cache unique obj type",type(uo[k[0]]))
print("cache centers",[(uo[i].center.x if hasattr(uo[i],'center') else None) for i in k][:3])
print("cache f32-representable:",[ (f32(uo[i].center.x) and f32(uo[i].center.y)) for i in list(uo)[:8]])
# match by token
d=[]
for o in tr0:
    t=o.track_token
    if t in uo: d.append((o.center.x-uo[t].center.x,o.center.y-uo[t].center.y, o.box.length-uo[t].box.length, o.box.width-uo[t].box.width))
d=np.array(d); print("matched",len(d),"abs max dx,dy,dl,dw",np.abs(d).max(0))
print("db ego rear x",repr(sc.initial_ego_state.rear_axle.x), repr(mc.ego_state.rear_axle.x))
print("--- per object: type, speed, d(ours-cache), implied dt along velocity")
for o in tr0:
    t=o.track_token
    if t in uo:
        c=uo[t]; dx,dy=o.center.x-c.center.x,o.center.y-c.center.y
        v=getattr(o,'velocity',None); sp=0; dt=None
        if v is not None:
            vx,vy=v.x,v.y; sp=float(np.hypot(vx,vy))
            if sp>0.5: dt=float((dx*vx+dy*vy)/sp**2)
        print(o.tracked_object_type.name, round(sp,2), round(dx,4), round(dy,4), None if dt is None else round(dt,4))
# timestamps
print("ours t0 timestamp_us", sc.get_ego_state_at_iteration(0).time_point.time_us, "cache ego", mc.ego_state.time_point.time_us)
print("ours iter1 ts", sc.get_ego_state_at_iteration(1).time_point.time_us)
