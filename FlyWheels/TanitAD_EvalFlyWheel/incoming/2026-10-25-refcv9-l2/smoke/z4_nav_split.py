"""Z4 (EXPLORATORY, post-seal, post-hoc; not in the prereg): where does the REMAINING false-decel live at pf 48,852 -- by the per-frame nav token and the head's call."""
import sys, glob, json
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C, l2_fit as F
out = {}
for seed in (0, 1):
    E = C.load_eval_pf(seed)
    D = C.ARCH / C.EVAL_TREE / f"dump_s{seed}" / "decisions"
    nav, navclip, ws_all = [], [], []
    for f in sorted(glob.glob(str(D / "ep*.npz"))):
        d = np.load(f); nav.append(d["nav_cmd"]); navclip.append(d["nav_cmd_clip"]); ws_all.append(d["ws"])
    nav = np.concatenate(nav); navclip = np.concatenate(navclip); ws_all = np.concatenate(ws_all)
    assert np.array_equal(ws_all, E["ex_ws"]), "decisions order != extras order"
    nv = nav[E["rows"]]
    stg = C.apply_stage(E["plan8"], E["prior8"])
    comp = E["compliant"]
    gt = C.plan_class(E["g"], E["v0"], E["dt"])
    L6 = C.lon6(E["logits"]); hg = F.GROUP_OF6[np.argmax(L6, 1)]
    r = {}
    for nm, m in (("FOLLOW(0)", nv == 0), ("TURN_L(1)", nv == 1), ("TURN_R(2)", nv == 2), ("other", ~np.isin(nv, [0, 1, 2]))):
        cm = comp & (gt == "cruise") & m
        if cm.sum() == 0: continue
        pc = C.plan_class(stg, E["v0"], E["dt"]); p7 = C.plan_class(E["r7"], E["v0"], E["dt"])
        r[nm] = {"n_gt_cruise": int(cm.sum()), "fd_stage": float((pc[cm] == "decel").mean()), "fd_r7": float((p7[cm] == "decel").mean()),
                 "head_decel_rate": float((hg[cm] == 0).mean()),
                 "fd_stage_given_head_cruise": float((pc[cm & (hg != 0)] == "decel").mean()) if (cm & (hg != 0)).any() else None,
                 "fd_stage_given_head_decel": float((pc[cm & (hg == 0)] == "decel").mean()) if (cm & (hg == 0)).any() else None,
                 "share_of_all_false_decel": float((pc[cm] == "decel").sum() / max(1, (pc[comp & (gt == 'cruise')] == "decel").sum()))}
    out[str(seed)] = r
    print("seed", seed)
    for k, v in r.items(): print("  ", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
json.dump(out, open("C:/Users/Admin/refcv8_pf/refcv9_l2/result/Z4_nav_split_EXPLORATORY.json", "w"), indent=1)
