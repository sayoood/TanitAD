import sys, numpy as np
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
from an2 import load, boot_ci
X=load("step48852ema_pf","0"); eid=X['eid']
P=X['plan_full_nav_true'].astype(np.float64); v0=X['ego_t0'][:,0].astype(np.float64)
lat3=TC.lat3_class(P); lonc=TC.lon_class(P,v0)
ll=X['lat_label']; m=ll!=-100
tgt=np.where(m,np.asarray(TC.V7_LAT_TO_LAT3)[np.clip(ll,0,7)],-100)
acc=np.where(m,(lat3==tgt).astype(float),np.nan); const=np.where(m,(tgt==0).astype(float),np.nan)
print("PLAN-derived lat3 vs v9 lat label: acc",[round(x,4) for x in boot_ci(acc,eid)],"constant LANE_KEEP",round(float(np.nanmean(const)),4))
for c,nm in ((1,"TURN_L"),(2,"TURN_R")):
    mc=m&(tgt==c); print("  plan recall",nm,round(float((lat3[mc]==c).mean()),4),"n",int(mc.sum()),"| head recall (nav_true)",round(float((np.asarray(TC.V7_LAT_TO_LAT3)[X['lat_pred_nav_true']][mc]==c).mean()),4))
# lon: plan class -> v7 id mapping, strict and partial
Lb=X['lon_label']; ml=Lb!=-100
A=TC.plan_lon_to_v7_allowed(lonc)
strict=np.where(ml,(np.asarray(TC.PLAN_LON_TO_V7)[np.clip(lonc,0,6)]==Lb).astype(float),np.nan)
part=np.where(ml,A[np.arange(len(Lb)),np.clip(Lb,0,7)].astype(float),np.nan)   # label inside allowed set of plan class? (inverse direction check below)
# correct reading: is the LABEL's v7 id allowed given plan class? i.e. plan said class c -> allowed v7 ids contain label
pl_allowed=np.array([ (Lb[i] in [i2 for i2 in range(8) if A[i,i2]]) if ml[i] else False for i in range(len(Lb))])
part=np.where(ml,pl_allowed.astype(float),np.nan)
cmaj=np.bincount(Lb[ml]).argmax()
print("PLAN-derived lon vs v9 lon label: strict acc",[round(x,4) for x in boot_ci(strict,eid)],"partial(FOLLOW-allowed) acc",[round(x,4) for x in boot_ci(part,eid)],"constant (class %d) acc"%cmaj,round(float((Lb[ml]==cmaj).mean()),4))
# plan vs head: which is more right on the label (strict)?
hs=np.where(ml,(X['lon_pred_nav_true']==Lb).astype(float),np.nan)
print("head strict lon acc",round(float(np.nanmean(hs)),4),"plan strict lon acc",round(float(np.nanmean(strict)),4))
