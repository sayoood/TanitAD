import sys, glob, numpy as np, json, math
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
from an2 import boot_ci
seed=sys.argv[1]
Z=dict(np.load(S+f"Z_s{seed}.npz",allow_pickle=True)); eid=Z['eid']
n=len(eid)
# ---------- route following (S6 grid, 6 slots 1..6 s), instrument = taniteval route_following (re-typed)
valid=np.ones((n,6),bool)
def rf(pick,gt,label):
    r=TC.route_following(pick.astype(np.float64),gt.astype(np.float64),valid)
    cls=r['gt_cls']; dk=r['dir_ok']
    out={"n_turn":r['summary']['turn']['n'],"dir_correct_turn":[round(x,4) for x in boot_ci(np.where(np.isin(cls,['turnL','turnR']),dk,np.nan),eid)],
         "n_straight":r['summary']['straight']['n'],"dir_correct_straight":[round(x,4) for x in boot_ci(np.where(cls=='straight',dk,np.nan),eid)],
         "head15_turn":round(float(r['summary']['turn']['head15']),4)}
    print(f"ROUTE {label}:",json.dumps(out)); return r,cls
r8,cls=rf(Z['os'],Z['g'],"refcv8-48852-pf os")
r7,_=rf(Z['r7'],Z['g'],"refcv7-50400 (same windows)")
if seed=="0":
    rf(Z['osz'],Z['g'],"refcv8 nav-ZERO (intervention)")
rf(Z['g'],Z['g'],"CONTROL GT-vs-GT (must be 1.0)")
straight=np.stack([np.stack([np.linspace(1,6,6)*Z['v0'][i],np.zeros(6)],-1) for i in range(n)])
rf(straight,Z['g'],"CONTROL hold-straight at v0 (must be ~0 on turns)")
# direction split by whether the fed per-frame nav token agrees with the GT turn side (token: 1=L? 2=R?)
nav=Z['navcmd']; turn=np.isin(cls,['turnL','turnR'])
print("turn windows",int(turn.sum()),"; nav token vs GT side crosstab (rows nav 0/1/2, cols turnL/turnR)")
for v in (0,1,2):
    print(v,[int(((nav==v)&(cls==c)).sum()) for c in ('turnL','turnR')])
for v in (0,1,2):
    m=turn&(nav==v)
    if m.sum(): print("dir_ok given nav",v,"n",int(m.sum()),round(float(np.nanmean(r8['dir_ok'][m])),4),"refcv7:",round(float(np.nanmean(r7['dir_ok'][m])),4))
# false-turn: straight GT windows where pick turns
th=r8['pick_theta']; print("pick |terminal heading|>=30deg on GT-straight:",round(float((np.degrees(np.abs(th[cls=='straight']))>=30).mean()),4),"refcv7",round(float((np.degrees(np.abs(r7['pick_theta'][cls=='straight']))>=30).mean()),4))
