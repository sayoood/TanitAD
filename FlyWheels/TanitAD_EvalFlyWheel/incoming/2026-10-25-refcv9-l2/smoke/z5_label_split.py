"""Z5 (EXPLORATORY, post-seal, post-hoc; not in the prereg): is the remaining false-decel concentrated where the GT tactical LABEL (NOW+2..8 s) says brake while the 0-6 s trajectory cruises?"""
import sys, json
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C, l2_fit as F
res = {}
for seed in (0, 1):
    E = C.load_eval_pf(seed)
    stg = C.apply_stage(E["plan8"], E["prior8"]); comp = E["compliant"]
    gt = C.plan_class(E["g"], E["v0"], E["dt"]); pc = C.plan_class(stg, E["v0"], E["dt"]); p7 = C.plan_class(E["r7"], E["v0"], E["dt"])
    lab = E["lon_v7"].astype(int)
    grp = np.full(len(lab), "unlabelled", object)
    grp[np.isin(lab, [2, 3, 4, 5, 6])] = "label-decel(YIELD,BRAKE_TO,CREEP,HOLD,ADAPT)"; grp[np.isin(lab, [0, 1])] = "label-cruise/follow"; grp[lab == 7] = "label-accel"
    cm = comp & (gt == "cruise")
    hg = F.GROUP_OF6[np.argmax(C.lon6(E["logits"]), 1)]
    tot_fd = (pc[cm] == "decel").sum()
    r = {}
    for g in ("label-decel(YIELD,BRAKE_TO,CREEP,HOLD,ADAPT)", "label-cruise/follow", "label-accel", "unlabelled"):
        m = cm & (grp == g)
        if not m.any(): continue
        r[g] = {"n_gt_cruise": int(m.sum()), "fd_stage": float((pc[m] == "decel").mean()), "fd_r7": float((p7[m] == "decel").mean()),
                "head_decel_rate": float((hg[m] == 0).mean()), "share_of_false_decel": float((pc[m] == "decel").sum() / tot_fd)}
    res[str(seed)] = r
    print("seed", seed, "(GT-cruise compliant windows by the GT tactical label)")
    for k, v in r.items(): print("  ", k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
json.dump(res, open("C:/Users/Admin/refcv8_pf/refcv9_l2/result/Z5_label_split_EXPLORATORY.json", "w"), indent=1)
