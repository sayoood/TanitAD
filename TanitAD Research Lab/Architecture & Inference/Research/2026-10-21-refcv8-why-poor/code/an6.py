import sys, numpy as np, json
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
from an2 import load, boot_ci
X=load("step48852ema_pf","0"); eid=X['eid']
# lat: head vs token-lookup echo (fit on same windows = optimistic for the control)
m=X['lat_label']!=-100; l=X['lat_label']; t=X['nav_cmd']; p=X['lat_pred_nav_true']
look=np.zeros_like(l)
for v in np.unique(t[m]):
    mm=m&(t==v); look[mm]=np.bincount(l[mm]).argmax()
a=np.where(m,(p==l).astype(float),np.nan); b=np.where(m,(look==l).astype(float),np.nan)
print("LAT head",round(float(np.nanmean(a)),4),"lookup-of-fed-token",round(float(np.nanmean(b)),4),"head-lookup paired",[round(x,4) for x in boot_ci(a-b,eid)])
print("lookup table (token->label): ",{int(v):int(np.bincount(l[m&(t==v)]).argmax()) for v in np.unique(t[m])})
# accuracy per token cell
for v in (0,1,2):
    mm=m&(t==v); print(" nav",v,"n",int(mm.sum()),"head acc",round(float((p[mm]==l[mm]).mean()),4),"label dist",np.bincount(l[mm],minlength=8)[[0,6,7]].tolist(),"pred dist",np.bincount(p[mm],minlength=8)[[0,6,7]].tolist())
# vision-only (nav-zero) TURN recall
pz=X['lat_pred_nav_zero']
for c,nm in ((6,"TURN_L"),(7,"TURN_R")):
    mc=m&(l==c); print(nm,"recall nav_true",round(float((p[mc]==c).mean()),4),"nav_zero",round(float((pz[mc]==c).mean()),4),"n",int(mc.sum()))
print("lat false-turn rate on LANE_KEEP windows: nav_true",round(float(np.isin(p[m&(l==0)],(6,7)).mean()),4),"nav_zero",round(float(np.isin(pz[m&(l==0)],(6,7)).mean()),4))
# lon confusion, row-normalised
ml=X['lon_label']!=-100; L=X['lon_label']; P=X['lon_pred_nav_true']
names=["FOLLOW","CRUISE","YIELD","BRAKE_TO","CREEP","HOLD","ADAPT","ACCEL"]
print("LON confusion rows=label cols=pred (counts)  classes",names)
for c in (0,1,3,4,5,7):
    row=[int(((L==c)&ml&(P==q)).sum()) for q in range(8)]
    print(names[c],row)
Dset=(2,3,4,5,6)
print("LON head false-decel on label CRUISE:",round(float(np.isin(P[ml&(L==1)],Dset).mean()),4),"n",int((ml&(L==1)).sum()),"; head ACCEL on label CRUISE",round(float((P[ml&(L==1)]==7).mean()),4))
print("LON head missed-decel on label BRAKE_TO:",round(float((~np.isin(P[ml&(L==3)],Dset)).mean()),4),"n",int((ml&(L==3)).sum()))
