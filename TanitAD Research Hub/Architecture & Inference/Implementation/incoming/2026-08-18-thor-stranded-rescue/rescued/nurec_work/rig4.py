import json, numpy as np, msgpack

CLIP = "clipgt-00040136-e651-4abd-991d-0655ccda9430"
CAM = f"camera_front_wide_120fov@{CLIP}"
rt = json.load(open("/home/nvidia/nurec_work/x/rig_trajectories.json"))
e = rt["rig_trajectories"][0]

print("cameras_frame_T_rig_worlds type:", type(e["cameras_frame_T_rig_worlds"]))
if isinstance(e["cameras_frame_T_rig_worlds"], dict):
    print("  keys:", [k.split("@")[0] for k in e["cameras_frame_T_rig_worlds"]])
    P = np.array(e["cameras_frame_T_rig_worlds"][CAM])
else:
    P = np.array(e["cameras_frame_T_rig_worlds"])
print("  front_wide poses array shape:", P.shape)

TS = np.array(e["cameras_frame_timestamps_us"][CAM])
print("  timestamps shape:", TS.shape, "first:", TS[0].tolist(), "last:", TS[-1].tolist())
print("  shutter duration us:", (TS[0][1] - TS[0][0]), " frame period us:", TS[1][0] - TS[0][0])

TRW = np.array(e["T_rig_worlds"])
TRWt = np.array(e["T_rig_world_timestamps_us"])
print("\nT_rig_worlds shape:", TRW.shape, " timestamps shape:", TRWt.shape,
      " t0:", TRWt.flat[0], " tN:", TRWt.flat[-1])

W2N = np.array(rt["world_to_nre"]["matrix"])
TSR = np.array(rt["camera_calibrations"][CAM]["T_sensor_rig"])
print("\nT_sensor_rig:\n", np.array2string(TSR, precision=5))
print("world_to_nre t:", W2N[:3, 3])

pose0 = P[0][0]  # frame 0, shutter start
print("\npose0 (frame0, shutter-start) =\n", np.array2string(pose0, precision=5))

print("\n### CONVENTION TEST: where does the rig sit in NRE coords? ###")
for name, T_rig2world in [("T_rig_worlds AS rig->world", pose0),
                          ("T_rig_worlds AS world->rig (inverted)", np.linalg.inv(pose0))]:
    rig_nre = (W2N @ T_rig2world)[:3, 3]
    cam2nre = W2N @ T_rig2world @ TSR
    print(f"  {name}:  rig pos in NRE = {np.array2string(rig_nre, precision=3)}   "
          f"cam pos in NRE = {np.array2string(cam2nre[:3,3], precision=3)}")
    print(f"     cam forward axis (col2) in NRE = {np.array2string(cam2nre[:3,2], precision=4)}  "
          f"cam down axis (col1) = {np.array2string(cam2nre[:3,1], precision=4)}")

# ---- scene extents for the sanity check ----
with open("/home/nvidia/nurec_work/x/volume.msgpack", "rb") as f:
    d = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
sd = d["nre_data"]["state_dict"]
Nr = len(sd[".gaussians_nodes.road.positions"]) // 6
RP = np.frombuffer(sd[".gaussians_nodes.road.positions"], dtype=np.float16).astype(np.float32).reshape(Nr, 3)
print("\nroad NRE extent: x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]" %
      (RP[:, 0].min(), RP[:, 0].max(), RP[:, 1].min(), RP[:, 1].max(), RP[:, 2].min(), RP[:, 2].max()))
for name, T in [("rig->world", pose0), ("world->rig", np.linalg.inv(pose0))]:
    p = (W2N @ T)[:3, 3]
    inside = (RP[:, 0].min() <= p[0] <= RP[:, 0].max()) and (RP[:, 1].min() <= p[1] <= RP[:, 1].max())
    # local ground height: road gaussians within 4 m in xy
    m = (np.abs(RP[:, 0] - p[0]) < 4) & (np.abs(RP[:, 1] - p[1]) < 4)
    gz = np.median(RP[m, 2]) if m.sum() > 10 else float("nan")
    print(f"  {name}: rig NRE={np.array2string(p, precision=2)} inside_road_bbox={inside} "
          f"n_road_within_4m={int(m.sum())} local_ground_z={gz:.3f} -> rig_z-ground={p[2]-gz:.3f}")

# ---- albedo / specular layout ----
N = len(sd[".gaussians_nodes.background.positions"]) // 6
A = np.frombuffer(sd[".gaussians_nodes.background.features_albedo"], dtype=np.float16).astype(np.float32)
print(f"\n### background albedo layout (N={N}, 15 comps, F=5) ###")
print("  [N,5,3] absmean per F:", np.array2string(np.abs(A.reshape(N, 5, 3)).mean(axis=(0, 2)), precision=5))
print("  [N,3,5] absmean per F:", np.array2string(np.abs(A.reshape(N, 3, 5)).mean(axis=(0, 1)), precision=5))
Nrg = len(sd[".gaussians_nodes.dynamic_rigids.positions"]) // 6
Ar = np.frombuffer(sd[".gaussians_nodes.dynamic_rigids.features_albedo"], dtype=np.float16).astype(np.float32)
print(f"### dynamic_rigids albedo (N={Nrg}, 60 comps, F=20) ###")
print("  [N,20,3] absmean per F:", np.array2string(np.abs(Ar.reshape(Nrg, 20, 3)).mean(axis=(0, 2)), precision=4))
print("  [N,3,20] absmean per F:", np.array2string(np.abs(Ar.reshape(Nrg, 3, 20)).mean(axis=(0, 1)), precision=4))
SP = np.frombuffer(sd[".gaussians_nodes.background.features_specular"], dtype=np.float16).astype(np.float32)
print("### background specular layout (45) ###")
print("  [N,15,3] absmean per coeff:", np.array2string(np.abs(SP.reshape(N, 15, 3)).mean(axis=(0, 2)), precision=5))
print("  [N,3,15] absmean per coeff:", np.array2string(np.abs(SP.reshape(N, 3, 15)).mean(axis=(0, 1)), precision=5))
