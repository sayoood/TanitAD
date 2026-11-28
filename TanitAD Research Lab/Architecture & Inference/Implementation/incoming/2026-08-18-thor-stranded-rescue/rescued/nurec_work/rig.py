import json, numpy as np

rt = json.load(open("/home/nvidia/nurec_work/x/rig_trajectories.json"))
print("top keys:", list(rt.keys()))
print("\n=== T_world_base ===")
print(json.dumps(rt["T_world_base"], indent=1)[:1500])
print("\n=== world_to_nre ===")
print(json.dumps(rt["world_to_nre"], indent=1)[:1500])

cc = rt["camera_calibrations"]
print("\n=== camera_calibrations type", type(cc))
if isinstance(cc, dict):
    print("cam keys:", list(cc.keys()))
    k = "camera_front_wide_120fov"
    print(f"\n--- {k} ---")
    print(json.dumps(cc[k], indent=1)[:4000])
elif isinstance(cc, list):
    print("n:", len(cc))
    for i, c in enumerate(cc):
        nm = c.get("name", c.get("sensor_id", "?")) if isinstance(c, dict) else "?"
        print(f"  [{i}] {nm}  keys={list(c.keys()) if isinstance(c,dict) else type(c)}")
    print("\n--- full [0] ---")
    print(json.dumps(cc[0], indent=1)[:4000])

tr = rt["rig_trajectories"]
print("\n=== rig_trajectories type", type(tr))
if isinstance(tr, dict):
    print("keys:", list(tr.keys())[:20])
    kk = list(tr.keys())[0]
    v = tr[kk]
    print(f"first key {kk!r} -> {type(v)}")
    print(json.dumps(v, indent=1)[:2500] if not isinstance(v, list) else f"list n={len(v)}\nfirst: {json.dumps(v[0], indent=1)[:2000]}")
elif isinstance(tr, list):
    print("n:", len(tr))
    print(json.dumps(tr[0], indent=1)[:2500])
