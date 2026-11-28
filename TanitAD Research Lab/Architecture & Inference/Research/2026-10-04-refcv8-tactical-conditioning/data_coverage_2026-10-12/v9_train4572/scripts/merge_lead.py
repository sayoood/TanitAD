import json, numpy as np, hashlib
old = np.load("old/lead_train.npz", allow_pickle=True)
add = np.load("out/lead_add203.npz", allow_pickle=True)
keys = ["sha12","k","has","lead","gap","tg","s","v"]
out = {}
for k in keys:
    assert old[k].dtype == add[k].dtype, (k, old[k].dtype, add[k].dtype)
    out[k] = np.concatenate([old[k], add[k]])
so, sa = set(old["sha12"].astype(str)), set(add["sha12"].astype(str))
assert not (so & sa)
mo, ma = json.loads(str(old["meta"])), json.loads(str(add["meta"]))
assert mo["join_md5"] == ma["join_md5"] == "401c9c3119a50632dafc35c595d9e051"
meta = {"join_md5": mo["join_md5"], "n_clips": len(so)+len(sa),
        "builder_md5": ma["builder_md5"],
        "merged_from": {"old_lead_train.npz": {"md5": "67451737f67292b5381db9fc21fa06f1", "n_clips": len(so), "builder_md5": mo["builder_md5"]},
                        "lead_add203.npz": {"n_clips": len(sa), "builder_md5": ma["builder_md5"]}}}
np.savez_compressed("out/lead_train4572.npz", **out, meta=np.asarray(json.dumps(meta)))
print("merged", len(so), "+", len(sa), "clips;", len(out["k"]), "frames")
