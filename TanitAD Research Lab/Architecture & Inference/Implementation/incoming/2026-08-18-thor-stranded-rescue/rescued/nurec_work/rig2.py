import json, numpy as np, msgpack

rt = json.load(open("/home/nvidia/nurec_work/x/rig_trajectories.json"))
cc = rt["camera_calibrations"]
CLIP = "clipgt-00040136-e651-4abd-991d-0655ccda9430"
K = f"camera_front_wide_120fov@{CLIP}"
print("--- camera_front_wide_120fov calibration (from rig_trajectories.json) ---")
print(json.dumps(cc[K], indent=1)[:5000])

print("\n\n--- ALL camera keys sorted (to match sensor_models index order) ---")
for i, k in enumerate(sorted(cc.keys())):
    c = cc[k]
    # pull whatever intrinsic-ish fields exist
    print(f"[{i}] {k.split('@')[0]}")

tr = rt["rig_trajectories"]
print("\n--- rig_trajectories ---")
print("type:", type(tr))
if isinstance(tr, dict):
    print("keys:", list(tr.keys()))
    for kk in list(tr.keys())[:3]:
        v = tr[kk]
        print(f"\n  key {kk!r}: {type(v)}")
        if isinstance(v, list):
            print(f"    n={len(v)}")
            print("    first:", json.dumps(v[0], indent=1)[:1200])
        elif isinstance(v, dict):
            print("    subkeys:", list(v.keys())[:20])
            s0 = list(v.keys())[0]
            print(f"    [{s0}]:", json.dumps(v[s0], indent=1)[:1200])

# ---- state_dict sensor models, decoded as fp16 ----
print("\n\n--- state_dict .calib.camera_view_geometry.sensor_models.{i} decoded ---")
with open("/home/nvidia/nurec_work/x/volume.msgpack", "rb") as f:
    d = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
sd = d["nre_data"]["state_dict"]
for i in range(6):
    p = f".calib.camera_view_geometry.sensor_models.{i}"
    res = np.frombuffer(sd[p + ".resolution"], dtype=np.int32)
    pp = np.frombuffer(sd[p + ".principal_point"], dtype=np.float16).astype(np.float64)
    fw = np.frombuffer(sd[p + ".fw_poly"], dtype=np.float16).astype(np.float64)
    bw = np.frombuffer(sd[p + ".bw_poly"], dtype=np.float16).astype(np.float64)
    A = np.frombuffer(sd[p + ".A"], dtype=np.float16).astype(np.float64)
    Ai = np.frombuffer(sd[p + ".Ainv"], dtype=np.float16).astype(np.float64)
    m2 = np.frombuffer(sd[p + ".min_2d_norm"], dtype=np.float16).astype(np.float64)
    # HFOV implied: solve fw(theta) = width/2
    th = np.linspace(0, 2.0, 200001)
    r = np.polyval(fw[::-1], th)
    half = res[0] / 2.0
    j = int(np.argmin(np.abs(r - half)))
    print(
        f"[{i}] res={res.tolist()} pp={pp.tolist()} A={A.tolist()} Ainv={Ai.tolist()} min_2d_norm={m2.tolist()}\n"
        f"     fw_poly={fw.tolist()}\n     bw_poly={bw.tolist()}\n"
        f"     fw(theta)=W/2 at theta={th[j]:.4f} rad -> HFOV={2*np.degrees(th[j]):.1f} deg   (1/fw1={1/fw[1]:.8f} vs bw1={bw[1]:.8f})"
    )

emb = np.frombuffer(sd[".calib.camera_view_geometry.embeds.weight"], dtype=np.float16).astype(np.float32)
print(f"\n--- calib embeds.weight: {emb.shape} nonzero={int((emb!=0).sum())} "
      f"min={emb.min():+.6g} max={emb.max():+.6g} absmax={np.abs(emb).max():.6g}")
print("    calib.enabled =", d["nre_data"]["config"]["calib"]["enabled"])
