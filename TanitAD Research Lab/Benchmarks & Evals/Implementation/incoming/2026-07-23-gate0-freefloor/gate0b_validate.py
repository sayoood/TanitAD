#!/usr/bin/env python3
"""GATE-0b gradient-nudge validation (same discipline that caught the world_to_nre frame bug):
  (1) REPLICATION: _decode_nudged(eta=0) == model.forward(steps=2) byte-for-byte (proves the
      driver's decode replication is exact, so the nudge is a clean addition).
  (2) GRADIENT DIRECTION/SIGN: on a KNOWN off-road trajectory, the nudge drives off-road cost
      monotonically toward ~0 (points TOWARD the drivable area).
  (3) COLLISION SIGN: the collision term steers selection AWAY from an agent.
Banks gate0b_gradient_validation.json. If the gradient can't be made trustworthy -> STOP."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import importlib.util
spec = importlib.util.spec_from_file_location("rfd", "/workspace/refc_floor_driver.py")
rfd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rfd)
from alpasim_runtime.scene_loader import ArtifactSceneProvider

val = {"purpose": "Gate-0b gradient-nudge validation (frame/sign) before the suite run",
       "evidence_class": "MEASURED (tanitad-eval, GPU)"}

# ---- load REF-C-base ----
pol = rfd.RefCPolicy(ckpt="/root/models/refc-base-30k/ckpt.pt", preset="base", device="cuda")
dev = pol.device

# ===== (1) REPLICATION: _decode_nudged(eta=0) == model.forward(steps=2) =====
torch.manual_seed(0)
fw = torch.rand(1, pol.window, 9, 256, 256, device=dev)
v0, nav = 8.0, 0
navt = torch.tensor([nav], device=dev); v0t = torch.tensor([v0], device=dev)
out = pol.model(fw, nav_cmd=navt, v0=v0t, steps=2)
ref_traj = out["anchor_traj"][0].cpu().numpy(); ref_conf = out["anchor_logits"][0].cpu().numpy()
rep_traj, rep_conf = pol._decode_nudged(fw, v0, nav, 0.0, 0.0, 1.0, 0.0, None, 0.0, 0)
dtraj = float(np.abs(ref_traj - rep_traj).max()); dconf = float(np.abs(ref_conf - rep_conf).max())
rep_ok = dtraj < 1e-4 and dconf < 1e-4
val["replication"] = {"max_abs_traj_diff": dtraj, "max_abs_conf_diff": dconf,
                      "verdict": "EXACT" if rep_ok else "MISMATCH",
                      "note": "eta=0 nudged decode reproduces model.forward -> nudge is a clean superset"}
print("(1) REPLICATION max|dtraj|=%.2e max|dconf|=%.2e -> %s" % (dtraj, dconf, val["replication"]["verdict"]))

# ===== (2) GRADIENT DIRECTION on a KNOWN off-road trajectory =====
SS = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/986fec83193b1baf3d5121f09462e248"
prov = ArtifactSceneProvider.from_path(SS, smooth_trajectories=True)
sid = [s for s in prov.scene_ids if s.replace("clipgt-", "")[:8] == "00169207"][0]
ds = prov.get_data_source(sid); cm = rfd.SceneCostMap(ds)
gt = np.asarray(ds.rig.trajectory.positions); i0 = 5
cx, cy = float(gt[i0, 0]), float(gt[i0, 1])
d = gt[i0 + 1, :2] - gt[i0, :2]; yaw = float(np.arctan2(d[1], d[0]))
c, sn = float(np.cos(yaw)), float(np.sin(yaw))
# a deliberately OFF-road trajectory (veer hard left in rig frame)
offtraj = np.array([[[3., 6.], [6., 13.], [9., 21.], [12., 30.]]])  # [1,4,2] (N=1)
x = torch.from_numpy(offtraj[None]).float().to(dev)                  # [1,1,4,2]
def offcost(xt):
    a = xt[0, 0].cpu().numpy()
    wx = cx + c * a[:, 0] - sn * a[:, 1]; wy = cy + sn * a[:, 0] + c * a[:, 1]
    return cm.offroad_cost(wx, wy)
traj_costs = [float(offcost(x).mean())]
for _ in range(8):
    x = pol._nudge(x, cx, cy, c, sn, cm, 0.5)
    traj_costs.append(float(offcost(x).mean()))
monotone = all(traj_costs[i + 1] <= traj_costs[i] + 1e-6 for i in range(len(traj_costs) - 1))
reached = traj_costs[-1] < 0.5
grad_ok = monotone and reached and traj_costs[0] > 2.0
val["gradient_direction"] = {"scene": sid, "start_offroad_cost_m": round(traj_costs[0], 3),
                             "cost_trajectory_m": [round(v, 3) for v in traj_costs],
                             "monotone_decreasing": monotone, "reached_onroad_lt0p5m": reached,
                             "verdict": "TOWARD-DRIVABLE" if grad_ok else "REVIEW"}
print("(2) GRADIENT off-road cost %.2f -> %.2f over 8 nudges monotone=%s -> %s" % (
    traj_costs[0], traj_costs[-1], monotone, val["gradient_direction"]["verdict"]))

# ===== (3) COLLISION SIGN: term steers selection AWAY from an agent =====
# two on-road-ish anchors; place a synthetic agent on anchor A's 9 m waypoint.
aA = np.array([[3, 0], [6, 0], [9, 0], [12, 0]], float)
aB = np.array([[3, 0], [6, 0], [9, 0], [12, 0]], float)  # identical geometry...
aB_world_shift = 4.0
anchors2 = np.stack([aA, aB])
conf2 = np.array([0.0, 0.0])
# agent at A's 9 m waypoint (world)
wxA9 = cx + c * 9 - sn * 0; wyA9 = cy + sn * 9 + c * 0
cm2 = rfd.SceneCostMap(ds)
cm2.agents = [(np.array([[wxA9, wyA9]]), np.array([0.0]), False)]  # static agent, always there
# shift B 4 m laterally in rig so it is >coll_r from the agent
anchors2[1, :, 1] += aB_world_shift
drv = rfd.RefCFloorDriver(policy=None, mode="on", lam=0.0, mu=2.0, clamp_m=0.75, coll_r=2.5,
                          eta=0.0, iters=0, sceneset_dir=SS, floor_log=None)
s2 = {"costmap": cm2, "scene_id": sid}
sel_c, diag_c = drv._select(s2, anchors2, conf2, cx, cy, yaw, 0.0)
# and with mu=0 (no collision term) it should be indifferent (picks A=argmax conf tie -> idx0)
drv0 = rfd.RefCFloorDriver(policy=None, mode="on", lam=0.0, mu=0.0, clamp_m=0.75, coll_r=2.5,
                           eta=0.0, iters=0, sceneset_dir=SS, floor_log=None)
sel_c0, _ = drv0._select(s2, anchors2, conf2, cx, cy, yaw, 0.0)
coll_ok = (sel_c == 1)   # with collision term ON, avoid A (the agent)
val["collision_sign"] = {"agent_on_anchorA": True, "sel_with_mu2": int(sel_c), "sel_with_mu0": int(sel_c0),
                         "expected_with_mu2": 1, "verdict": "AVOIDS-AGENT" if coll_ok else "REVIEW"}
print("(3) COLLISION mu=2 sel=%d (expect 1=avoid agent), mu=0 sel=%d -> %s" % (
    sel_c, sel_c0, val["collision_sign"]["verdict"]))

overall = rep_ok and grad_ok and coll_ok
val["overall_verdict"] = ("GRADIENT TRUSTWORTHY - PROCEED TO SUITE" if overall
                          else "NOT TRUSTWORTHY - DO NOT RUN SUITE")
json.dump(val, open("/workspace/gate0b_gradient_validation.json", "w"), indent=2)
print("\nOVERALL:", val["overall_verdict"])
