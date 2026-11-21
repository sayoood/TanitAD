import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], "rb"))
print(type(d), len(d))
fr = d[0]
for k in sorted(fr.keys()):
    v = fr[k]
    if isinstance(v, dict):
        print(k, "{", ", ".join(f"{kk}:{type(vv).__name__}{getattr(vv,'shape','')}" for kk, vv in v.items()), "}")
    else:
        print(k, type(v).__name__, getattr(v, "shape", ""), (v if not hasattr(v, "__len__") or len(str(v)) < 200 else str(v)[:200]))
print("CAM_F0", {k: (v if not hasattr(v,'shape') else np.asarray(v).round(4).tolist()) for k, v in fr["cams"]["CAM_F0"].items()})
print("ts", [d[i]["timestamp"] for i in range(4)])
print("dc", [list(map(int, d[i]["driving_command"])) for i in range(0, len(d), max(1,len(d)//10))])
print("eds", fr.get("ego_dynamic_state"))
print("e2g", np.asarray(fr["ego2global"]).round(3))
print("lidar2ego", np.asarray(fr.get("lidar2ego")).round(3) if "lidar2ego" in fr else None)
print("anns names", sorted(set(fr["anns"]["gt_names"])))
print("box0", np.asarray(fr["anns"]["gt_boxes"])[:3])
