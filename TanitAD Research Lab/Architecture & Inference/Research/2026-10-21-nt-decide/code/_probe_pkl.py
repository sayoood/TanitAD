import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb"))
print(type(d), len(d))
fr = d[0]
print(sorted(fr.keys()))
a = fr.get("anns", {})
print("anns keys", sorted(a.keys()) if isinstance(a, dict) else type(a))
for k in ("gt_boxes", "gt_names", "gt_velocity_3d", "track_tokens", "instance_tokens"):
    if k in a:
        v = a[k]; print(k, getattr(v, "shape", len(v)))
c = collections.Counter()
nb = []
for f in d:
    c.update(list(f["anns"]["gt_names"])); nb.append(len(f["anns"]["gt_names"]))
print(dict(c)); print("boxes/frame min/mean/max", min(nb), sum(nb)/len(nb), max(nb))
print("timestamps dt us", [d[i+1]["timestamp"]-d[i]["timestamp"] for i in range(3)])
print("driving_command", fr.get("driving_command"))
print("cams", sorted(fr["cams"].keys()))
