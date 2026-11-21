"""PROBE C -- the artifact build_targets.py ACTUALLY consumes (build_targets.py:153-156).
Reads ego_state.car_footprint.vehicle_parameters straight out of the banked SimulationLogs."""
import glob,os,collections
from pathlib import Path
from nuplan.planning.simulation.simulation_log import SimulationLog
root=r"C:\Users\Admin\dz\out\mini\notts_nr\mini-nr-l8-notts\closed_loop_nonreactive_agents_driverl_val14_nr\simulation_log"
files=sorted(glob.glob(os.path.join(root,"**","*.msgpack.xz"),recursive=True))
print("simulation logs found:",len(files))
vals=collections.Counter(); poses=set(); nstates=0
for p in files:
    log=SimulationLog.load_data(file_path=Path(p))
    smps=log.simulation_history.data
    # sample EVERY state in the log, not just the first -- within-log constancy too
    for s in smps:
        vp=s.ego_state.car_footprint.vehicle_parameters
        vals[(vp.length,vp.width,vp.wheel_base,vp.rear_axle_to_center,vp.half_width,
              vp.vehicle_name,vp.vehicle_type)]+=1
        nstates+=1
    poses.add((round(smps[0].ego_state.rear_axle.x,3),round(smps[0].ego_state.rear_axle.y,3)))
    print("  %-46s log=%-40s states=%d"%(os.path.basename(p)[:46],
          getattr(log.scenario,'log_name','?')[:40],len(smps)))
print()
print("EGO STATES READ (control, must be >>0):",nstates)
print("CONTROL: distinct start poses across logs =",len(poses),"of",len(files))
print("DISTINCT (length,width,wheel_base,rear_axle_to_center,half_width,name,type):",len(vals))
for k,c in vals.items(): print("   ",k,"-> n_states =",c)
