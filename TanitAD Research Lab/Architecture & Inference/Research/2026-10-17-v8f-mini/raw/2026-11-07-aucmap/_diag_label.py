import sys, os, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aucmap as A
A.init(); rig=A.Rig(levers=["LF100"])
post,gop,v0p=A.load_post(rig); sc=np.isfinite(post)
y=A.labels_from(rig.v0,rig.g)["GO"]
mm=sc&(y!=gop)
print('mismatch',mm.sum(),'v0 range',rig.v0[mm].min(),rig.v0[mm].max(),'y true among mm',int(y[mm].sum()),'gop true',int(gop[mm].sum()))
lt6=sc&(rig.v0<6)
print('v0<6: n',lt6.sum(),'mismatch',int((lt6&(y!=gop)).sum()),' 6-8: n',int((sc&(rig.v0>=6)).sum()),'mismatch',int((sc&(rig.v0>=6)&(y!=gop)).sum()),'gop true in 6-8',int(gop[sc&(rig.v0>=6)].sum()))
for nm,m in (('all',sc),('v0<6',lt6)):
    print(nm,'AUC post vs probe-go',A.auc_mw(post[m],gop[m]),'vs my GO',A.auc_mw(post[m],y[m]),'GO rate',y[m].mean(),'gop rate',gop[m].mean())
