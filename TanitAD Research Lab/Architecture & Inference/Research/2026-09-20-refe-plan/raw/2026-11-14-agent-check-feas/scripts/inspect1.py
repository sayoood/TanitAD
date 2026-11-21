import sys, time, lzma, pickle
sys.path[:0]=["F:/Archive/devbox-C/navsim/devkit","F:/Archive/devbox-C/navsim/nuplan-devkit"]
P=sys.argv[1]
t=time.time()
raw=lzma.decompress(open(P,"rb").read()); print("decompressed bytes",len(raw),"s",round(time.time()-t,2))
t=time.time(); mc=pickle.loads(raw); print("unpickle s",round(time.time()-t,2))
print(type(mc).__name__)
for k,v in vars(mc).items():
    print(k, type(v).__name__, (len(v) if hasattr(v,"__len__") else ""))
print("ts",mc.timepoint.time_s, "scene",mc.scene_type, "log",mc.log_name)
print("ego", mc.ego_state.rear_axle.x, mc.ego_state.rear_axle.y, mc.ego_state.rear_axle.heading)
cur=mc.current_tracked_objects[0].tracked_objects
print("n current objs",len(cur.tracked_objects))
for o in cur.tracked_objects[:3]:
    print(o.tracked_object_type.name, o.track_token, o.box.center.x,o.box.center.y,o.box.length,o.box.width,o.box.center.heading, o.velocity.x,o.velocity.y)
fut=mc.future_tracked_objects
print("future steps",len(fut), "t0 n objs", len(fut[0].tracked_objects.tracked_objects), "last n",len(fut[-1].tracked_objects.tracked_objects))
print("fut t0 time", fut[0].frame_index if hasattr(fut[0],'frame_index') else None, "dt?", [ (f.tracked_objects.tracked_objects[0].metadata.timestamp_us if f.tracked_objects.tracked_objects else None) for f in fut[:3]])
ob=mc.observation
print("obs attrs", [a for a in vars(ob)])
print("obs occ maps", len(ob._occupancy_maps) if ob._occupancy_maps else None, "unique objs", len(ob._unique_objects))
print("human traj poses", len(mc.human_trajectory.poses))
