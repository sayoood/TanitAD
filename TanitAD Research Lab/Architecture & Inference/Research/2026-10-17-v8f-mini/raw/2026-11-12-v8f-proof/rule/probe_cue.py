import numpy as np,sys
sys.path.insert(0,'.')
import stop_shield as SH
for s in (0,1):
    t=np.load(f'F:/TanitAD-scratch/stopbal_read/t1/ticks/ticks_BAL-NULL_s{s}_m10_causal_heads.npz'); m=t['tick']==0
    z=np.load(f'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-NULL_s{s}.npz')
    cp=t['t_cons_pred'][m]; gp=t['t_goal_prob'][m][:,6]; sh=z['stop_hit']; cg=z['cons_gt']; plan=t['plan'][m]
    print(s,'flag>=.5',(gp>=.5).sum(),'flag&stop',((gp>=.5)&sh).sum(),'fin',np.isfinite(cp[:,16:18]).all())
    idx=np.where(sh)[0]
    e=np.hypot(cp[idx,16]-cg[idx,16],cp[idx,17]-cg[idx,17]); print(' stop xy err med/mean',np.median(e),e.mean(),' gt x med',np.median(cg[idx,16]), 'pred x neg',(cp[idx,16]<0).sum())
    dd=[SH.rest_arc(plan[i],cg[i,16:18])[0] for i in idx[:40]]; off=[SH.rest_arc(plan[i],cg[i,16:18])[1] for i in idx]
    print(' arc d0 med',np.median(dd),'lateral off med/max',np.median(off),np.max(off))
    print(' v0 on stop med',np.median(z['v0'][idx]), 'sg0 med',np.median([SH.seg_speeds(plan[i],.2)[0] for i in idx]))
