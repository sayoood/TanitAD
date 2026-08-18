import json, numpy as np, msgpack

rt = json.load(open("/home/nvidia/nurec_work/x/rig_trajectories.json"))
tr = rt["rig_trajectories"]
print("rig_trajectories: list n =", len(tr))
e0 = tr[0]
print("elem0 type:", type(e0))
if isinstance(e0, dict):
    print("elem0 keys:", list(e0.keys()))
    for k, v in e0.items():
        if isinstance(v, list):
            a = np.array(v)
            print(f"  {k}: list shape={a.shape} dtype={a.dtype}")
            if a.size <= 20:
                print("     ", np.array2string(a, precision=6))
        else:
            print(f"  {k}: {type(v).__name__} {v}")
print("\nelem0 full:", json.dumps(e0, indent=1)[:2000])
print("\nelem1 timestamp-ish:", json.dumps({k: v for k, v in tr[1].items() if not isinstance(v, list)}, indent=1)[:600])
ts = [e.get("timestamp_us", e.get("timestamp")) for e in tr[:5]]
print("first 5 timestamps:", ts)
print("last timestamp:", tr[-1].get("timestamp_us", tr[-1].get("timestamp")))

# ---------- albedo layout test ----------
with open("/home/nvidia/nurec_work/x/volume.msgpack", "rb") as f:
    d = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
sd = d["nre_data"]["state_dict"]
N = len(sd[".gaussians_nodes.background.positions"]) // 6
A = np.frombuffer(sd[".gaussians_nodes.background.features_albedo"], dtype=np.float16).astype(np.float32)
print(f"\n\n### background albedo layout test (N={N}, 15 comps, fourier_features_dim=5) ###")
v_F3 = A.reshape(N, 5, 3)  # [N, F, 3]
v_3F = A.reshape(N, 3, 5)  # [N, 3, F]
print("view [N,5,3]: per-F std over gaussians+channels:")
for i in range(5):
    print(f"   F{i}: std={v_F3[:,i,:].std():.5f} mean={v_F3[:,i,:].mean():+.5f} absmean={np.abs(v_F3[:,i,:]).mean():.5f}")
print("view [N,3,5]: per-F std:")
for i in range(5):
    print(f"   F{i}: std={v_3F[:,:,i].std():.5f} mean={v_3F[:,:,i].mean():+.5f} absmean={np.abs(v_3F[:,:,i]).mean():.5f}")

# same test for dynamic_rigids (F=20)
Nr = len(sd[".gaussians_nodes.dynamic_rigids.positions"]) // 6
Ar = np.frombuffer(sd[".gaussians_nodes.dynamic_rigids.features_albedo"], dtype=np.float16).astype(np.float32)
r_F3 = Ar.reshape(Nr, 20, 3)
r_3F = Ar.reshape(Nr, 3, 20)
print(f"\n### dynamic_rigids albedo (N={Nr}, 60 comps, F=20) ###")
print("view [N,20,3] absmean per F:", np.array2string(np.abs(r_F3).mean(axis=(0, 2)), precision=4))
print("view [N,3,20] absmean per F:", np.array2string(np.abs(r_3F).mean(axis=(0, 1)), precision=4))

# ---------- specular layout ----------
SP = np.frombuffer(sd[".gaussians_nodes.background.features_specular"], dtype=np.float16).astype(np.float32)
s_C3 = SP.reshape(N, 15, 3)  # [N, coeff, 3]
s_3C = SP.reshape(N, 3, 15)  # [N, 3, coeff]
print("\n### background specular layout (45 = 15 SH coeffs x 3 ch) ###")
print("view [N,15,3] absmean per coeff:", np.array2string(np.abs(s_C3).mean(axis=(0, 2)), precision=5))
print("view [N,3,15] absmean per coeff:", np.array2string(np.abs(s_3C).mean(axis=(0, 1)), precision=5))
