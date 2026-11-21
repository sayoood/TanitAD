"""Exercise build_targets.build_one's HAPPY path with a stubbed nuPlan + stubbed camera_index.
No code is modified; only the module's own dependencies are stubbed."""
import sys, types, math, os
import numpy as np
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")

# --- stub the nuplan import that build_one does at call time ---
m = types.ModuleType("nuplan"); sys.modules["nuplan"]=m
for p in ["nuplan.planning","nuplan.planning.simulation","nuplan.planning.simulation.simulation_log"]:
    sys.modules[p]=types.ModuleType(p)
class _TP:
    def __init__(s,us): s.time_us=us
class _V:
    x=6.0; y=0.0
class _DCS:
    rear_axle_velocity_2d=_V(); rear_axle_acceleration_2d=_V(); angular_velocity=0.0
class _RA:
    def __init__(s,i): s.x=float(i)*0.6; s.y=0.0; s.heading=0.0
class _ES:
    def __init__(s,i): s.rear_axle=_RA(i); s.time_point=_TP(1_000_000+i*100_000); s.dynamic_car_state=_DCS(); s.tire_steering_angle=0.0
class _Traj:  goal_points=[10.0,0.0,20.0,0.0]
class _S:
    def __init__(s,i): s.ego_state=_ES(i); s.trajectory=_Traj()
class _Hist:
    data=[_S(i) for i in range(149)]
class _Sc:
    log_name="LOG"; scenario_type="t"; scenario_name="tok"
class _Log:
    scenario=_Sc(); simulation_history=_Hist()
    @staticmethod
    def load_data(file_path=None): return _Log()
sys.modules["nuplan.planning.simulation.simulation_log"].SimulationLog=_Log

import build_targets as bt
# stub camera_index so no DB is needed: all four channels present, 10 Hz, aligned
def fake_camera_index(db_path, channels=bt.CAMERAS):
    return {ch: [(1_000_000+i*100_000, f"{ch}_{i}.jpg") for i in range(149)] for ch in channels}
bt.camera_index = fake_camera_index

print("CAMERAS =", bt.CAMERAS)
try:
    rows, stats = bt.build_one("x.msgpack.xz", "dbdir", None, 0, 120.0)
    print("OK rows:", len(rows), "stats:", stats)
    if rows: print(" row0 image:", rows[0]["image"], " cameras:", rows[0]["cameras"], " dt_ms:", rows[0]["dt_ms"])
except Exception as e:
    print(f"RAISED {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

print()
print("--- now the REFUSAL path (one channel missing) ---")
def fake_missing(db_path, channels=bt.CAMERAS):
    d = {ch: [(1_000_000+i*100_000, f"{ch}_{i}.jpg") for i in range(149)] for ch in channels}
    d["CAM_B0"]=[]
    return d
bt.camera_index = fake_missing
try:
    rows, stats = bt.build_one("x.msgpack.xz","dbdir",None,0,120.0)
    print("OK rows:", len(rows), "stats:", stats)
except Exception as e:
    print(f"RAISED {type(e).__name__}: {e}")
