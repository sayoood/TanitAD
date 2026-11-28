"""GEO10K step 3d: the lane-change crossing diagnostics are emitted by build_v9_labels only when an lc input exists (clip_labels(lc=None)
never creates them), but the trainer reads two of them as constraint cells (v8f_mini_data.LAT_CONSTRAINTS: lc_t_cross_s, lc_d_cross_m).
Found by g7_check (KeyError on the first reachability run). Fill the three columns with the builder's own no-crossing values, which are
what 99.07 % of the existing release rows hold (MEASURED on the 4,369 release): lc_t_cross_s = NaN, lc_d_cross_m = NaN, lc_side = 0 (int8)."""
import json, sys
import numpy as np
sys.path.insert(0, "/root/geo10k/tree/stack/scripts")
sys.path.insert(0, "/root/geo10k/tree/stack")
import build_v9_labels as B
OUT = "/root/geo10k/out"
z = np.load(f"{OUT}/v9_labels_new.npz", allow_pickle=False)
d = {k: z[k] for k in z.files}
n = len(d["row__k"])
add = {"row__lc_t_cross_s": np.full(n, np.nan, np.float64), "row__lc_d_cross_m": np.full(n, np.nan, np.float64), "row__lc_side": np.zeros(n, np.int8)}
for k, v in add.items():
    assert k not in d, k
    d[k] = v
np.savez_compressed(f"{OUT}/v9_labels_new.npz", **d)
m = json.load(open(f"{OUT}/v9_labels_new.manifest.json"))
m["fields"] = sorted(d)
m["npz_md5"] = B.file_md5(f"{OUT}/v9_labels_new.npz")
m["geo10k"]["fill_after_merge"] = {"fields": sorted(add), "value": "no-crossing value of the builder (NaN / NaN / 0), because no lane-change input exists for the new clips"}
json.dump(m, open(f"{OUT}/v9_labels_new.manifest.json", "w"))
print("[g3d] added", sorted(add), "n", n, "md5", m["npz_md5"])
print("ZZ GEO10K-G3D DONE ZZ")
