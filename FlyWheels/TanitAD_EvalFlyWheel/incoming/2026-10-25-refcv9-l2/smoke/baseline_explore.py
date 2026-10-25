import sys, time, json
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C
t0=time.time()
st = C.stack(); PR=st["PR"]
for seed in (0,1):
    E = C.load_eval_pf(seed)
    print("seed", seed, E["checks"], "compliant", E["compliant"].sum(), "eps", len(set(E["eid"][E["compliant"]].tolist())))
    table = C.deploy_table()
    stg = C.apply_stage(E["plan8"], E["prior8"], table)
    print(" stage reproduces banked S6_pos: max abs", float(np.abs(stg - E["stage_banked"]).max()))
    for surf, m in (("compliant", E["compliant"]), ("all", np.ones(len(E["eid"]), bool))):
        for arm, P in (("os", E["os"]), ("stage", stg), ("r7", E["r7"])):
            fd, md, gt, pc = C.lon_indicators(P, E, m)
            print(f"  {surf:9s} {arm:6s} FD {np.nanmean(fd):.4f} (n {int(np.isfinite(fd).sum())})  MD {np.nanmean(md):.4f} (n {int(np.isfinite(md).sum())})")
print("wall", time.time()-t0)
