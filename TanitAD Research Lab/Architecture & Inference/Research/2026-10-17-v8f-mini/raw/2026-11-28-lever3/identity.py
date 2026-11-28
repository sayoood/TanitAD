import sys, io, json, tarfile, hashlib, statistics
import torch
tar_path, dev_root, out = sys.argv[1:4]
dev = dev_root + "/clips/"
res = {}; meta = {}
t = tarfile.open(tar_path, "r:")
for m in t:
    n = m.name
    if n.endswith("meta.json") and "eval" in n and n.count("/") <= 3:
        meta[n] = json.load(t.extractfile(m))
    if not (n.endswith(".pt") and "/clips/" in n and "/eval/" in n): continue
    clip = n.rsplit("/", 1)[1][:-3]
    try:
        a = torch.load(io.BytesIO(t.extractfile(m).read()), map_location="cpu", weights_only=True)
        b = torch.load(dev + clip + ".pt", map_location="cpu", weights_only=True)
    except Exception as e:
        res[clip] = {"error": repr(e)}; continue
    r = {}
    for k in ("pool4x10", "pool8x20"):
        if k in a and k in b:
            x = a[k].float(); y = b[k].float()
            if x.shape != y.shape: r[k] = {"shape_pod": list(x.shape), "shape_dev": list(y.shape)}; continue
            d = (x - y).norm() / x.norm().clamp_min(1e-12)
            r[k] = {"bitwise": bool(torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8))) if a[k].dtype == torch.float8_e4m3fn else bool(torch.equal(a[k], b[k])), "rel_l2": float(d)}
    r["pose_equal"] = bool(torch.equal(a["pose_grid"], b["pose_grid"])) if "pose_grid" in a and "pose_grid" in b else None
    res[clip] = r
def stat(k):
    v = [r[k]["rel_l2"] for r in res.values() if k in r and "rel_l2" in r[k]]
    bw = [r[k]["bitwise"] for r in res.values() if k in r and "bitwise" in r[k]]
    if not v: return None
    return {"n": len(v), "n_bitwise": sum(bw), "mean": statistics.mean(v), "median": statistics.median(v), "max": max(v), "min": min(v)}
summ = {"pool4x10": stat("pool4x10"), "pool8x20": stat("pool8x20"), "n_clips_compared": len(res), "n_errors": sum("error" in r for r in res.values()),
        "pod_meta_files": {k: {x: v.get(x) for x in ("encoder_module_sha256", "encoder_module", "torch", "transformers", "pipeline")} for k, v in meta.items()}}
json.dump({"summary": summ, "per_clip": res}, open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
