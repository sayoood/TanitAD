"""D3 check 5a -- the two camera rigs (cy ~543 vs ~755) and vehicle-calibration sharing, train vs eval139.
rig = cy < 650 -> 'A' else 'B' (cy_split 650 from the committed black-row census; the two clusters are ~210 px apart, MEASURED below).
Vehicle fingerprint = rounded extrinsics (x,y,z to 1 mm, quaternion to 1e-4): clips with an identical fingerprint share one physical
calibration (same car + calibration epoch).  Ids hashed (sha12)."""
import csv, json, hashlib, sys, collections
import numpy as np, torch
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
SCR = sys.argv[1]
mt = torch.load(SCR + "/train_v2manifest.pt", map_location="cpu", weights_only=False)
me = torch.load("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", map_location="cpu", weights_only=False)
tr, ev = list(mt["clip_id"]), list(me["clip_id"])
cy = {}; fx = {}
for r in csv.DictReader(open(SCR + "/physicalai_front_wide_intrinsics.csv", encoding="utf-8")):
    cy[r["clip_id"]] = float(r["cy"]); fx[r["clip_id"]] = float(r["fw_poly_1"])
ext = json.load(open("D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
out = {"n_intrinsics_rows": len(cy), "n_extrinsics_rows": len(ext)}
def rig(ids):
    v = np.array([cy.get(c, np.nan) for c in ids])
    miss = int(np.isnan(v).sum()); v = v[~np.isnan(v)]
    a = v < 650
    return {"n": len(ids), "n_without_cy": miss, "rigA_n": int(a.sum()), "rigB_n": int((~a).sum()), "rigB_pct": round(100 * (~a).mean(), 2),
            "cy_A_median": float(np.median(v[a])), "cy_B_median": float(np.median(v[~a])), "cy_A_range": [float(v[a].min()), float(v[a].max())], "cy_B_range": [float(v[~a].min()), float(v[~a].max())],
            "gap_between_clusters_px": float(v[~a].min() - v[a].max())}
out["rig_train"] = rig(tr); out["rig_eval"] = rig(ev)
# 2x2 test rig x split
a = np.array([[out["rig_train"]["rigA_n"], out["rig_train"]["rigB_n"]], [out["rig_eval"]["rigA_n"], out["rig_eval"]["rigB_n"]]], float)
tot = a.sum(); e = a.sum(1, keepdims=True) * a.sum(0, keepdims=True) / tot
chi = float(((a - e) ** 2 / e).sum())
out["rig_x_split_chi2_1dof"] = chi
# fingerprints
def fp(c):
    e_ = ext.get(c)
    if e_ is None: return None
    return (round(e_["x"], 3), round(e_["y"], 3), round(e_["z"], 3), round(e_["qx"], 4), round(e_["qy"], 4), round(e_["qz"], 4), round(e_["qw"], 4))
ftr = collections.Counter(fp(c) for c in tr if fp(c) is not None); fev = [fp(c) for c in ev if fp(c) is not None]
out["extrinsics"] = {"train_with_extrinsics": sum(ftr.values()), "eval_with_extrinsics": len(fev), "distinct_fingerprints_train": len(ftr),
                     "eval_clips_whose_fingerprint_is_in_train": sum(1 for f in fev if f in ftr), "distinct_fingerprints_eval": len(set(fev)),
                     "eval_fingerprints_not_in_train": sum(1 for f in set(fev) if f not in ftr), "train_clips_per_fingerprint_top5": [c for _, c in ftr.most_common(5)]}
# camera height / pitch distributions
h = np.array([ext[c]["_read"]["camera_height_m"] for c in tr if c in ext]); p = np.array([ext[c]["_read"]["pitch_down_deg"] for c in tr if c in ext])
he = np.array([ext[c]["_read"]["camera_height_m"] for c in ev if c in ext]); pe = np.array([ext[c]["_read"]["pitch_down_deg"] for c in ev if c in ext])
out["camera_height_m"] = {"train_q": [round(float(x), 3) for x in np.quantile(h, [0, .01, .5, .99, 1])], "eval_q": [round(float(x), 3) for x in np.quantile(he, [0, .01, .5, .99, 1])]}
out["pitch_down_deg"] = {"train_q": [round(float(x), 3) for x in np.quantile(p, [0, .01, .5, .99, 1])], "eval_q": [round(float(x), 3) for x in np.quantile(pe, [0, .01, .5, .99, 1])]}
out["eval_height_outside_train_range"] = int(((he < h.min()) | (he > h.max())).sum())
# rig B vs camera height (rig is a different physical setup?)
cyv = np.array([cy[c] for c in tr if c in ext and c in cy]); hv = np.array([ext[c]["_read"]["camera_height_m"] for c in tr if c in ext and c in cy])
out["height_by_rig_train"] = {"A_median": float(np.median(hv[cyv < 650])), "B_median": float(np.median(hv[cyv >= 650]))}
json.dump(out, open("raw/c5a_rigs_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
