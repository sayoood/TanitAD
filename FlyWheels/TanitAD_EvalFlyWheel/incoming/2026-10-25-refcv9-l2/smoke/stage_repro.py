import sys
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C
RS = C.stack()["RS"]
tab = C.deploy_table()
for seed in (0,1):
    E = C.load_eval_pf(seed)
    full = RS.retime_plan(E["plan8"], E["prior8"], C.ALL_T, tab, cast=True)
    print(seed, "8-slot call -> S6 slots vs banked S6_pos: max abs", float(np.abs(full[:, C.S6_SLOTS]-E["stage_banked"]).max()),
          "bit-equal", bool(np.array_equal(full[:, C.S6_SLOTS], E["stage_banked"])))
    s2 = np.load(C.DEPLOY_DIR / f"DEPLOY8_POS_step48852ema_pf_s{seed}.npz")["S2_pos8"]
    print("   S2_pos8 vs 8-slot call on S2 rows", float(np.abs(RS.retime_plan(np.load(C.DEPLOY_DIR / f"DEPLOY8_POS_step48852ema_pf_s{seed}.npz")["S2_plan8"], np.load(C.DEPLOY_DIR / f"DEPLOY8_POS_step48852ema_pf_s{seed}.npz")["S2_prior8"], C.ALL_T, tab, cast=True) - s2).max()))
