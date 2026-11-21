import json, numpy as np
D = "/root/dscale10k/merged/"
z = np.load(D + "v9_labels_union.npz", allow_pickle=True)
cl = {k[6:]: z[k] for k in z.files if k.startswith("clip__")}
der = json.load(open(D + "v9_labels_R_s0.derangement.json"))
donor = der["donor"] if "donor" in der else der
print({k: (v if not isinstance(v, (dict, list)) else "...") for k, v in der.items()})
sid2i = {int(s): i for i, s in enumerate(cl["sid"])}
recv = np.array([sid2i[int(s)] for s in donor]); don = np.array([sid2i[int(d)] for d in donor.values()])
out = {"n_deranged": int(len(recv)), "n_clips": int(len(cl["sid"]))}
for k in cl:
    a, b = cl[k][recv], cl[k][don]
    out[k] = {"n_differ": int((a != b).sum()), "frac_differ": round(float((a != b).mean()), 4), "dtype": str(cl[k].dtype)}
# the R release's clip table equals the union's (columns kept)?
r = np.load(D + "v9_labels_R_s0.npz", allow_pickle=True)
out["R_clip_cols_equal_union"] = {k: bool(np.array_equal(r["clip__" + k], cl[k])) for k in cl}
# group structure
nr, k0 = cl["n_rows"][recv], cl["k0"][recv]
out["donor_same_n_rows_k0"] = bool(((cl["n_rows"][don] == nr) & (cl["k0"][don] == k0)).all())
out["has_lc_pos_10k"] = int(cl["has_lc"][recv].sum()); out["has_lead_pos_10k"] = int(cl["has_lead"][recv].sum()); out["has_record_pos_10k"] = int(cl["has_record"][recv].sum())
json.dump(out, open("/root/dscale10k/gate/R_columns_probe.json", "w"), indent=1)
print(json.dumps(out))
