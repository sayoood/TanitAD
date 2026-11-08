"""Does a Pacifica footprint placed EXACTLY on the PDM route centreline (heading = tangent) exit the strict drivable union?
usage: q1_cl_footprint.py SAMPLER IDX S_FROM S_TO"""
import os, sys, math
os.environ["CUDA_VISIBLE_DEVICES"]="-1"; sys.dont_write_bytecode=True
sys.path.insert(0,"E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/scorer_gym")
import numpy as np, gym_env as g
from shapely.geometry import Point
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.car_footprint import CarFootprint
sampler,idx,s0,s1=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
S=g.GymState((g.select_states if sampler=="pilot" else g.select_states_stall)(20,10)[idx]); ev=S.evaluator; cl=ev.cl
print("map slack",ev.map_slack_m)
u=ev._drivable_union()
ss=np.arange(s0,s1,0.5); P=cl.interpolate(ss,as_array=True)
worst=0
for s,(x,y,h) in zip(ss,P):
    fp=CarFootprint.build_from_rear_axle(StateSE2(float(x),float(y),float(h)),g.VEH)
    o=max(float(u.distance(Point(c))) for c in list(fp.geometry.exterior.coords)[:4])
    worst=max(worst,o)
    print("s=%.1f xy=(%.1f,%.1f) footprint-on-centreline overshoot=%.3f"%(s,x,y,o))
print("WORST on-centreline overshoot",worst)
