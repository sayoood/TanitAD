import sys, numpy as np, json
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
from an2 import boot_ci, load
tag=sys.argv[1]; seed=sys.argv[2]
X=load(tag,seed); eid=X['eid']; n=len(eid)
P=X['plan_full_nav_true'].astype(np.float64); v0=X['ego_t0'][:,0].astype(np.float64)
lat3=TC.lat3_class(P); lonc=TC.lon_class(P,v0)
headlat=np.asarray(TC.V7_LAT_TO_LAT3)[X['lat_pred_nav_true']]
headlon=X['lon_pred_nav_true']
rng=np.random.default_rng(0)
print(tag,"seed",seed,"n",n,"plan lat3 counts",np.bincount(lat3[lat3>=0],minlength=3).tolist(),"plan lon counts (HOLD,CREEP,STOP,FOLLOW,DECEL,ACCEL,KEEP)",np.bincount(lonc[lonc>=0],minlength=7).tolist())
# LAT consistency: head lat3 == plan lat3
ok=(lat3>=0)
agree=np.where(ok,(headlat==lat3).astype(float),np.nan)
null=[]
for _ in range(200):
    p=rng.permutation(n); null.append(np.nanmean(np.where(ok,(headlat[p]==lat3).astype(float),np.nan)))
print("LAT head-vs-plan lat3 agreement",[round(x,4) for x in boot_ci(agree,eid)],"NULL(permuted head) mean",round(float(np.mean(null)),4),"range",round(float(np.min(null)),4),round(float(np.max(null)),4))
# agreement when head says TURN or plan says TURN
for nm,m in (("head TURN",headlat>0),("plan TURN",lat3>0),("head TURN or plan TURN",(headlat>0)|(lat3>0))):
    print("  ",nm,"n",int(m.sum()),"agree",round(float(np.nanmean(np.where(m&ok,agree,np.nan))),4))
cm=np.zeros((3,3),int)
for a in range(3):
    for b in range(3): cm[a,b]=int(((headlat==a)&(lat3==b)).sum())
print("  confusion rows=head lat3, cols=plan lat3:",cm.tolist())
# LON consistency: plan class allowed mapping
A=TC.plan_lon_to_v7_allowed(lonc)
okl=lonc>=0
a_strict=np.where(okl,(headlon==np.asarray(TC.PLAN_LON_TO_V7)[np.clip(lonc,0,6)]).astype(float),np.nan)
a_part=np.where(okl,A[np.arange(n),np.clip(headlon,0,7)].astype(float),np.nan)
nl_s=[];nl_p=[]
for _ in range(200):
    p=rng.permutation(n); h=headlon[p]
    nl_s.append(np.nanmean(np.where(okl,(h==np.asarray(TC.PLAN_LON_TO_V7)[np.clip(lonc,0,6)]).astype(float),np.nan))); nl_p.append(np.nanmean(np.where(okl,A[np.arange(n),np.clip(h,0,7)].astype(float),np.nan)))
print("LON head-vs-plan class (strict)",[round(x,4) for x in boot_ci(a_strict,eid)],"NULL",round(float(np.mean(nl_s)),4),"| partial(FOLLOW allowed)",[round(x,4) for x in boot_ci(a_part,eid)],"NULL",round(float(np.mean(nl_p)),4))
# decel/accel consistency on the coarse 3-way (decel: head in {YIELD2,BRAKE3,CREEP4,HOLD5,ADAPT6}; plan in {STOP2,DECEL4,CREEP1,HOLD0}); accel head 7 vs plan 5
hd=np.where(np.isin(headlon,(2,3,4,5,6)),"D",np.where(headlon==7,"A","C"))   # C = follow/cruise
pdn=np.where(np.isin(lonc,(0,1,2,4)),"D",np.where(lonc==5,"A","C"))
ct={}
for h in "DCA":
    ct[h]=[int(((hd==h)&(pdn==q)&okl).sum()) for q in "DCA"]
print("LON coarse crosstab rows=head D/C/A cols=plan D/C/A:",ct)
agree3=np.where(okl,(hd==pdn).astype(float),np.nan)
nl=[np.nanmean(np.where(okl,(hd[rng.permutation(n)]==pdn).astype(float),np.nan)) for _ in range(200)]
print("LON coarse 3-way agreement",[round(x,4) for x in boot_ci(agree3,eid)],"NULL",round(float(np.mean(nl)),4))
# head-vs-GT label vs plan-vs-GT label (does the plan realise the head's correct decisions?)
ll=X['lon_label']; 
print("done")
