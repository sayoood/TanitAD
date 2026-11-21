"""TIMING ONLY: one metric-cache token (v1.1 devkit), 64 synthetic proposals. Scores are not printed or stored."""
import sys, time, lzma, pickle, numpy as np
V1="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
sys.path[:0]=[V1,"F:/Archive/devbox-C/navsim/nuplan-devkit","F:/tanitad_scratch/p6ship/refe"]
P=sys.argv[1]
t=time.time(); raw=lzma.decompress(open(P,"rb").read()); t_dec=time.time()-t
t=time.time(); mc=pickle.loads(raw); t_unp=time.time()-t
print("decompress s",round(t_dec,3),"unpickle s (incl. first imports)",round(t_unp,2),"raw bytes",len(raw))
print("fields",[k for k in vars(mc)])
ob=mc.observation
print("obs attrs",list(vars(ob)))
print("n occ maps",len(ob._occupancy_maps),"unique objs",len(ob._unique_objects),"sample interval",ob._sample_interval,"samples",ob._observation_samples,"map_radius",ob._map_radius)
print("types",sorted(set(type(o).__name__+":"+o.tracked_object_type.name for o in ob._unique_objects.values())))
print("occ map t0 n geoms",len(ob[0].tokens),"t last",len(ob[ob._observation_samples-1].tokens))
import navsim_pdm_targets as PT
parts=(mc.trajectory, mc.ego_state, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
k=np.arange(1,9,dtype=np.float32)[:,None]; hp=np.concatenate([4.0*k,0*k,0*k],1).astype(np.float32)
rng=np.random.default_rng(0)
props=np.stack([hp+rng.normal(0,0.3,hp.shape).astype(np.float32)*(np.arange(8)[:,None]/7.0) for _ in range(64)])
t=time.time(); PT._setup(); print("setup s",round(time.time()-t,2))
ts=[]
for i in range(3):
    t=time.time(); out=PT.score_plans(parts, props); ts.append(time.time()-t)
print("score_plans 64 proposals s per call",[round(x,3) for x in ts], "keys",list(out))
import resource
