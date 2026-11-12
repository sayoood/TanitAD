import os,sys,json,pickle,lzma
os.environ.update({"REFE_SIM_HZ":"10","NUPLAN_MAPS_ROOT":"E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0","OMP_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":"-1"})
sys.path[:0]=["C:/Users/Admin/n1_run/w/pk/refe"]
import numpy as np
import navsim_pdm_targets as PT
import navtrain_scenarios as NS
S=PT._setup()
from navsim.evaluate.pdm_score import get_trajectory_as_array
sel=json.load(open("diag_tokens.json"))
tl={}
for l in open("n1t_only.jsonl"):
    d=json.loads(l); tl[d["token"]]=d["log"]
dbs=NS.index_dbs(db_root="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
def es(e): return np.array([e.rear_axle.x,e.rear_axle.y,e.rear_axle.heading,e.dynamic_car_state.rear_axle_velocity_2d.x,e.dynamic_car_state.rear_axle_velocity_2d.y,e.dynamic_car_state.rear_axle_acceleration_2d.x,e.time_point.time_us],float)
def occ_summary(obs):
    # per-timestep dict token->bounds of occupancy geometry
    out=[]
    for om in obs._occupancy_maps:
        out.append({k:om[k].bounds for k in om.tokens})
    return out
res=[]
for grp,toks in sel.items():
    for tok in toks:
        log=tl[tok]
        CP=f"F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/{log}/unknown/{tok}/metric_cache.pkl"
        mc=pickle.loads(lzma.open(CP,"rb").read())
        sc=next(NS.build_scenarios_for_log(dbs[log],[tok],history_rows=1,future_rows=80))
        p=PT.metric_cache_parts(sc)
        r={"group":grp,"token":tok,"db_interval":float(sc.database_interval)}
        r["ego_absdiff_max"]=float(np.abs(es(p[1])-es(mc.ego_state)).max())
        ta=get_trajectory_as_array(p[0],S["samp"],p[1].time_point); tb=get_trajectory_as_array(mc.trajectory,S["samp"],mc.ego_state.time_point)
        r["pdm_traj_absdiff_max"]=float(np.abs(ta-tb).max())
        # observation
        oa,ob=p[2],mc.observation
        r["n_occ"]=[len(oa._occupancy_maps),len(ob._occupancy_maps)]
        da,db=occ_summary(oa),occ_summary(ob)
        mx=0.0; ntok_mis=0
        for x,y in zip(da,db):
            if set(x)!=set(y): ntok_mis+=1
            for k in set(x)&set(y): mx=max(mx,float(np.abs(np.array(x[k])-np.array(y[k])).max()))
        r["obs_token_set_mismatch_steps"]=ntok_mis; r["obs_bounds_absdiff_max"]=mx
        r["centerline_len"]=[float(p[3].length),float(mc.centerline.length)]
        res.append(r); print(json.dumps(r),flush=True)
json.dump(res,open("diag_cache_compare.json","w"),indent=1)
