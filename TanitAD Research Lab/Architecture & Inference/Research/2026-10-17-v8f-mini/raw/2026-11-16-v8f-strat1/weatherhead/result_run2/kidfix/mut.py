import sys, numpy as np, shutil
sys.path.insert(0, "F:/TanitAD-scratch/strat1/whead/build")
import wh_read as R
for arm in ("H_s0","H_s1"):
    ev=f"F:/TanitAD-scratch/strat1/whead/run2/t0/eval_WH-{arm}_real.npz"; kid=f"F:/TanitAD-scratch/strat1/whead/run2/kid/kid_{arm}.npz"
    print(arm,"fixed", R.k_id(ev,kid))
    z=dict(np.load(kid,allow_pickle=True)); lg=z["ctx_pool_logits"].copy(); lg[:,[0,1]]=lg[:,[1,0]]; z["ctx_pool_logits"]=lg
    m=f"F:/TanitAD-scratch/strat1/whead/kidfix/tmp/mut_{arm}.npz"; np.savez(m,**z); print(arm,"mutation col0<->col1", R.k_id(ev,m))
