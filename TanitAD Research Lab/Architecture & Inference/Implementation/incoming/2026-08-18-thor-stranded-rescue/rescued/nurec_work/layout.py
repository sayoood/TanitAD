import json, numpy as np, msgpack

CLIP = "clipgt-00040136-e651-4abd-991d-0655ccda9430"
CAM = f"camera_front_wide_120fov@{CLIP}"
rt = json.load(open("/home/nvidia/nurec_work/x/rig_trajectories.json"))
e = rt["rig_trajectories"][0]
P = np.array(e["cameras_frame_T_rig_worlds"][CAM])
W2N = np.array(rt["world_to_nre"]["matrix"])
TSR = np.array(rt["camera_calibrations"][CAM]["T_sensor_rig"])

with open("/home/nvidia/nurec_work/x/volume.msgpack", "rb") as f:
    d = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
sd = d["nre_data"]["state_dict"]
Nr = len(sd[".gaussians_nodes.road.positions"]) // 6
RP = np.frombuffer(sd[".gaussians_nodes.road.positions"], dtype=np.float16).astype(np.float32).reshape(Nr, 3)

print("### POSE CONVENTION TEST AT LATE FRAMES (pose0 is ~identity, so useless) ###")
print("road NRE extent: x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]" %
      (RP[:, 0].min(), RP[:, 0].max(), RP[:, 1].min(), RP[:, 1].max(), RP[:, 2].min(), RP[:, 2].max()))
for fi in [0, 150, 300, 450, 598]:
    pose = P[fi][0]
    for name, T in [("rig->world", pose), ("world->rig", np.linalg.inv(pose))]:
        p = (W2N @ T)[:3, 3]
        m = (np.abs(RP[:, 0] - p[0]) < 4) & (np.abs(RP[:, 1] - p[1]) < 4)
        gz = np.median(RP[m, 2]) if m.sum() > 10 else float("nan")
        print(f"  frame {fi:3d} {name:11s}: rig NRE={np.array2string(p, precision=2):32s} "
              f"n_road<4m={int(m.sum()):5d} ground_z={gz:7.3f} height_above_ground={p[2]-gz:+7.3f}")

print("\n### AXIS-ORDER TEST via RGB correlation ###")
def axis_test(flat, N, K, label):
    """flat: [N*K*3]; test whether layout is [N,K,3] (coeff-major) or [N,3,K] (channel-major)."""
    a = flat.reshape(N, -1)
    sub = a[:200000] if N > 200000 else a
    # hypothesis A: [K,3] -> same-coeff channels are (3i, 3i+1, 3i+2)
    cA = []
    for i in range(K):
        x, y, z = sub[:, 3 * i], sub[:, 3 * i + 1], sub[:, 3 * i + 2]
        cA += [np.corrcoef(x, y)[0, 1], np.corrcoef(y, z)[0, 1]]
    # hypothesis B: [3,K] -> same-coeff channels are (j, K+j, 2K+j)
    cB = []
    for j in range(K):
        x, y, z = sub[:, j], sub[:, K + j], sub[:, 2 * K + j]
        cB += [np.corrcoef(x, y)[0, 1], np.corrcoef(y, z)[0, 1]]
    cA, cB = np.array(cA), np.array(cB)
    print(f"  {label}: N={N} K={K}")
    print(f"    hypA [N,{K},3] coeff-major : mean|corr| of same-coeff RGB pairs = {np.nanmean(np.abs(cA)):.4f}  (median {np.nanmedian(np.abs(cA)):.4f})")
    print(f"    hypB [N,3,{K}] channel-major: mean|corr| of same-coeff RGB pairs = {np.nanmean(np.abs(cB)):.4f}  (median {np.nanmedian(np.abs(cB)):.4f})")
    print(f"    -> WINNER: {'hypA [N,K,3]' if np.nanmean(np.abs(cA)) > np.nanmean(np.abs(cB)) else 'hypB [N,3,K]'}")

N = len(sd[".gaussians_nodes.background.positions"]) // 6
A = np.frombuffer(sd[".gaussians_nodes.background.features_albedo"], dtype=np.float16).astype(np.float32)
axis_test(A, N, 5, "background features_albedo (15)")
SP = np.frombuffer(sd[".gaussians_nodes.background.features_specular"], dtype=np.float16).astype(np.float32)
axis_test(SP, N, 15, "background features_specular (45)")
Nrg = len(sd[".gaussians_nodes.dynamic_rigids.positions"]) // 6
Ar = np.frombuffer(sd[".gaussians_nodes.dynamic_rigids.features_albedo"], dtype=np.float16).astype(np.float32)
axis_test(Ar, Nrg, 20, "dynamic_rigids features_albedo (60)")
SPr = np.frombuffer(sd[".gaussians_nodes.road.features_specular"], dtype=np.float16).astype(np.float32)
axis_test(SPr, Nr, 15, "road features_specular (45)")

print("\n### background albedo per-F mean/std under [N,5,3] ###")
v = A.reshape(N, 5, 3)
for i in range(5):
    print(f"   F{i}: mean={v[:,i,:].mean():+.5f} std={v[:,i,:].std():.5f} "
          f"per-ch mean={np.array2string(v[:,i,:].mean(axis=0), precision=4)}")
