"""oracle_attr.py -- DESCRIPTIVE (no bar): split the oracle arm's missed-decel / false-decel change vs BASE by what the force
changed per window: group flip, within-group class change, same class (sharpen only), unforced. Seed 0, S6 compliant, stage on."""
import json, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_common as C  # noqa: E402

E = C.load_eval_pf(0)
z = np.load("C:/Users/Admin/refcv8_pf/refcv9_l2/bank/trace_oracle_s0.npz", allow_pickle=True)
W = int(json.loads(str(z["meta_json"])).get("window_W", 0)) if "window_W" in str(z["meta_json"]) else None
key = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(z["e_i"], z["ws"]))}
j = np.array([key.get((int(c), int(w)), -1) for c, w in zip(E["clip"], E["ws"])])
assert (j >= 0).all(), f"unjoined {(j < 0).sum()}"
raw, app, plan_o = z["raw_lon_logits"][j], z["applied_lon_logits"][j], z["plan8"][j]
tab = C.deploy_table()
zb = np.load("C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz", allow_pickle=False)
Wb = int(json.loads(str(zb["meta_json"]))["window_W"])
kb = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(zb["e_i"], zb["t"].astype(np.int64) + Wb - 1))}
jb = np.array([kb.get((int(c), int(w)), -1) for c, w in zip(E["clip"], E["ws"])]); assert (jb >= 0).all()
plan_b = zb["traj_own"][jb]
st_b = C.apply_stage(plan_b, E["prior8"], tab)  # BASE-os (L1 bank, os-only child), NOT the full-arm-set battery draw
st_o = C.apply_stage(plan_o, E["prior8"], tab)
# identity: unforced windows must reproduce the banked stage exactly
forced = np.abs(app - raw).max(1) > 1e-6
idd = float(np.abs(plan_o[~forced] - plan_b[~forced]).max())
L6 = list(C.LON6_V7_IDS)
ch, co = raw[:, L6].argmax(1), app[:, L6].argmax(1)
grp = np.array([0 if i in C.G_DECEL else (1 if i in C.G_CRUISE else 2) for i in range(6)])
typ = np.where(~forced, "U", np.where(grp[ch] != grp[co], "A_groupflip", np.where(ch != co, "B_withingroup", "C_sharpen")))
m = E["compliant"]
fd_b, md_b, gt, _ = C.lon_indicators(st_b, E, m)
fd_o, md_o, _, _ = C.lon_indicators(st_o, E, m)
out = {"identity_unforced_max_abs_m": idd, "n_forced": int(forced.sum()), "types": {}}
for t in ("A_groupflip", "B_withingroup", "C_sharpen", "U"):
    s = typ == t
    dec = s & m & (gt == "decel"); cru = s & m & (gt == "cruise")
    out["types"][t] = {"n_all": int(s.sum()), "n_gt_decel": int(dec.sum()), "n_gt_cruise": int(cru.sum()),
                       "missed_base": float(np.nansum(md_b[dec])), "missed_oracle": float(np.nansum(md_o[dec])),
                       "falsedec_base": float(np.nansum(fd_b[cru])), "falsedec_oracle": float(np.nansum(fd_o[cru]))}
n_dec = int((m & (gt == "decel")).sum()); n_cru = int((m & (gt == "cruise")).sum())
out["MD_base"], out["MD_oracle"] = float(np.nansum(md_b) / n_dec), float(np.nansum(md_o) / n_dec)
out["FD_base"], out["FD_oracle"] = float(np.nansum(fd_b) / n_cru), float(np.nansum(fd_o) / n_cru)
# by forced class (oracle's class) among GT-decel windows whose miss flipped to caught
names = C.LON6
dec = m & (gt == "decel")
fixed = dec & (md_b == 1) & (md_o == 0); broke = dec & (md_b == 0) & (md_o == 1)
out["fixed_by_oracle_class"] = {names[k]: int((fixed & (co == k)).sum()) for k in range(6)}
out["fixed_by_head_class"] = {names[k]: int((fixed & (ch == k)).sum()) for k in range(6)}
out["broke_by_oracle_class"] = {names[k]: int((broke & (co == k)).sum()) for k in range(6)}
out["fixed_by_type"] = {t: int((fixed & (typ == t)).sum()) for t in ("A_groupflip", "B_withingroup", "C_sharpen", "U")}
out["broke_by_type"] = {t: int((broke & (typ == t)).sum()) for t in ("A_groupflip", "B_withingroup", "C_sharpen", "U")}
json.dump(out, open("C:/Users/Admin/refcv8_pf/refcv9_l2/oracle_attr/ORACLE_ATTR.json", "w"), indent=1)
print(json.dumps(out, indent=1))
