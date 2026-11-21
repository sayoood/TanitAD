import json, numpy as np
z = np.load("release_train4572/v9_labels_train.npz")
add = json.load(open("add_ids.json"))
import sys; sys.path.insert(0, "patched"); import build_v9_labels as B
sid_add = np.array([B.stable_episode_id(c) for c in add["add"]])
sn, r0, nn = z["clip__sid"], z["clip__row0"], z["clip__n_rows"]
isadd = np.isin(sn, sid_add)
rows_add = np.concatenate([np.arange(r0[i], r0[i]+nn[i]) for i in np.nonzero(isadd)[0]])
mask_add = np.zeros(len(z["row__k"]), bool); mask_add[rows_add] = True
lon = z["row__lon_cls"]; al = z["row__lon_allowed"]; rev = z["row__lon_truncated"] if "row__lon_truncated" in z.files else None
for tag, m in (("ADD", mask_add), ("old", ~mask_add)):
    ig = m & (lon == -100)
    print(tag, "rows", int(m.sum()), "lon IGNORE", round(ig.sum()/m.sum(), 4),
          "of which allowed==0 (fully masked)", round((ig & (al == 0)).sum()/m.sum(), 4),
          "partial (allowed>0)", round((ig & (al > 0)).sum()/m.sum(), 4))
    hh = z["row__h_obs_s"][m]
    print("   h_obs_s mean", round(float(hh.mean()),3), "share h_obs<8:", round(float((hh < 7.999).mean()),4),
          "| share rows with h_obs<3 (short):", round(float((hh < 3).mean()),4))
# per-clip: how many ADD clips have >=90% lon IGNORE
per = []
for i in np.nonzero(isadd)[0]:
    sl = slice(r0[i], r0[i]+nn[i]); per.append(float((lon[sl] == -100).mean()))
per = np.array(per); print("ADD clips with >=90% lon IGNORE:", int((per >= 0.9).sum()), "of", len(per), "| >=50%:", int((per >= .5).sum()))
print("fields with 'trunc'/'rev':", [f for f in z.files if "trunc" in f or "rev" in f])
