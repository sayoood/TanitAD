#!/usr/bin/env python3
"""HiCAP E25/E26: (1) contamination screen of the v9 clips against NVIDIA's public Alpamayo lists,
(2) census of the v9 tactical labels (the [NOW+2, NOW+8] s band) to re-base the v7 counts.
Inputs (MEASURED reads): v9_labels_{train,eval139}.npz + manifests (HF Sayood/tanitad-refcv8-v9-labels, md5 checked),
NVlabs/alpamayo-recipes data/*.parquet. clip__sha12 = sha256(clip_id)[:12] (V9_SCHEMA.md), so NVIDIA's UUIDs are hashed.
Run: /usr/bin/python3 hicap_v9_screen_census.py <hfdata_dir> <alpamayo_recipes_data_dir>"""
import sys, json, hashlib, collections
import numpy as np, pandas as pd
hf, nv = sys.argv[1], sys.argv[2]
def sha12(s): return hashlib.sha256(s.encode()).hexdigest()[:12]
lists = {}
import glob, os
for p in sorted(glob.glob(os.path.join(nv, "*.parquet"))):
    df = pd.read_parquet(p); lists[os.path.basename(p)] = set(sha12(k) for k in df["key"].astype(str))
out = {"evidence": "MEASURED (ours)", "nvidia_lists": {k: len(v) for k, v in lists.items()}, "screen": {}, "census": {}}
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return [round((c-h)/d, 4), round((c+h)/d, 4)]
for split, fn in (("train", "v9_labels_train"), ("eval139", "v9_labels_eval139")):
    z = np.load(os.path.join(hf, fn+".npz"), allow_pickle=True); man = json.load(open(os.path.join(hf, fn+".manifest.json")))
    sh = [str(x) for x in z["clip__sha12"]]; n = len(sh)
    out["screen"][split] = {"n_clips": n}
    out["screen"][split]["hit_sha12_in_0417_calibration_sample"] = sorted(s for s in sh if s in lists.get("0417_5k_train_set_for_calibration_25.10.parquet", set()))[:200] if split == "eval139" else "omitted (train)"
    for ln, S in lists.items():
        k = sum(1 for s in sh if s in S); out["screen"][split][ln] = {"in_list": k, "share": round(k/n, 4), "wilson95": wilson(k, n)}
    lat = z["row__lat_cls_a"]; lon = z["row__lon_cls"]; nrow = len(lat)
    lab = (lat != -100) & (lon != -100)
    c = {"rows": int(nrow), "rows_lat_labelled": float((lat != -100).mean()), "rows_lon_labelled": float((lon != -100).mean()), "rows_both": float(lab.mean())}
    c["lat_cls_a_counts"] = {int(a): int(b) for a, b in zip(*np.unique(lat, return_counts=True))}
    c["lon_cls_counts"] = {int(a): int(b) for a, b in zip(*np.unique(lon, return_counts=True))}
    joint = collections.Counter(zip(lat[lab].tolist(), lon[lab].tolist())); c["joint_lat_lon_nonzero_cells"] = len(joint)
    c["joint_cells_ge30"] = sum(1 for v in joint.values() if v >= 30)
    c["joint_top"] = [[list(k), v] for k, v in joint.most_common(12)]
    c["vocab_LAT9"] = man["vocab"].get("LAT9") if isinstance(man.get("vocab"), dict) else None
    c["vocab_LON9"] = man["vocab"].get("LON9") if isinstance(man.get("vocab"), dict) else None
    gy = z["row__goal_y"]; gw = z["row__goal_w"]
    c["goal_positive_rows"] = {i: int((((gy >> i) & 1) & ((gw >> i) & 1)).sum()) for i in range(22)}
    out["census"][split] = c
json.dump(out, open("hicap_v9_screen_census_result.json", "w"), indent=1, default=str)
print(json.dumps({"screen": out["screen"], "nvidia": out["nvidia_lists"]}, indent=1))
for s in out["census"]: 
    c = out["census"][s]; print(s, "rows", c["rows"], "lat", round(c["rows_lat_labelled"],4), "lon", round(c["rows_lon_labelled"],4), "joint cells", c["joint_lat_lon_nonzero_cells"], ">=30:", c["joint_cells_ge30"]); print(" lat", c["lat_cls_a_counts"]); print(" lon", c["lon_cls_counts"]); print(" vocab", c["vocab_LAT9"], c["vocab_LON9"])
