import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"]="-1"; sys.dont_write_bytecode=True
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np, gym_env as g
from shapely.geometry import Point
st=g.select_states(20,10)[10]
S=g.GymState(st); ev=S.evaluator; cl=ev.cl
print("centerline length",cl.length,"limit",S.limit)
tr=json.loads(open("F:/tanitad_scratch/mgym3/diag/tr_main10.jsonl").readline())
pts=[(t['xy'],t['v'],t['over']) for r in tr['rows'] for t in r['ticks']]
for xy,v,o in pts[::5]+[pts[-1]]:
    p=Point(*xy); s=cl.project(p); q=cl.interpolate(s) if hasattr(cl,'interpolate') else None
    cp=cl.linestring.interpolate(s); lat=p.distance(cp)
    print("xy",xy,"v",v,"over",o,"s=%.1f remaining=%.1f lat_to_cl=%.2f"%(s,cl.length-s,lat))
# human trajectory: overshoot + proximity to ego end pose
end=Point(*pts[-1][0]); best=None
n=min(S.sc.get_number_of_iterations(),100)
for i in range(n):
    e=S.sc.get_ego_state_at_iteration(i); d=Point(e.rear_axle.x,e.rear_axle.y).distance(end)
    if best is None or d<best[0]: best=(d,i,ev.footprint_overshoot(e),float(e.dynamic_car_state.rear_axle_velocity_2d.x))
print("human: nearest pose to ego end pose: dist=%.2f iter=%d overshoot=%.3f v=%.2f"%best)
print("human overshoot max over horizon", max(ev.footprint_overshoot(S.sc.get_ego_state_at_iteration(i)) for i in range(0,n,2)))
hs=[S.sc.get_ego_state_at_iteration(i) for i in range(0,n,10)]
print("human path:", [(round(e.rear_axle.x,1),round(e.rear_axle.y,1),round(float(e.dynamic_car_state.rear_axle_velocity_2d.x),1)) for e in hs])
# drivable union geometry near the ego end pose
u=ev._drivable_union(); print("union type",u.geom_type,"n parts",len(getattr(u,'geoms',[u])))
for c in list(S.sc.get_ego_state_at_iteration(0).car_footprint.geometry.exterior.coords)[:0]: pass
idc=ev.drivable.get_indices_of_map_type([g.SemanticMapLayer.ROADBLOCK,g.SemanticMapLayer.INTERSECTION,g.SemanticMapLayer.DRIVABLE_AREA,g.SemanticMapLayer.CARPARK_AREA])
nearby=[(i,float(ev.drivable._geometries[i].distance(end))) for i in idc]
nearby=sorted([x for x in nearby if x[1]<15],key=lambda x:x[1])
print("polygons within 15 m of ego end pose:",len(nearby))
for i,d in nearby[:8]:
    print(" poly idx",i,"token",ev.drivable._tokens[i] if hasattr(ev.drivable,'_tokens') else '?',"dist %.2f"%d,"area %.0f"%ev.drivable._geometries[i].area, "bounds",[round(x,1) for x in ev.drivable._geometries[i].bounds])
# the footprint at end: corner coordinates and which are outside
import shapely
print("done")
