"""PART 6 -- the EGO-DROPOUT regime. `refc.py:1341-1344`: rows whose speed is
withheld roll the bank at `ref_speed_ms = 10.0` instead of the measured v0, and
the live run carries `ego_dropout 0.5`, so HALF of training rows see that bank.
Supply and the oracle ceiling are re-measured there. STRICTLY MODEL-FREE."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\taniteval")
import tanitad.refs.refc_select as sl
from taniteval.ci import paired_episode_cluster_bootstrap as PB
OUT = os.path.join(HERE, "turncov_out"); DUMP = os.path.join(HERE, "refcv3_40284_dump")
DT, H = 0.1, 60
HORIZONS = (5,10,15,20,30,40,50,60); SLOTS=[k-1 for k in HORIZONS]
V_FLOOR, KAPPA_CAP, SEL_A, REF_SPEED = 4.0, 0.12, 2.0, 10.0
deg = np.degrees
d = np.load(os.path.join(OUT,"per_window.npz"))
v0=d["v0"]; eid=d["eid"]; ws=d["ws"]; dem_thx=d["dem_thx"]; ade_meas=d["ade_oiv"]; N=len(v0)
CTRL = torch.load(os.path.join(HERE,"pull","anchors_live.pt"),map_location="cpu",
                  weights_only=False)["controls"].double().numpy()
man = torch.load(os.path.join(HERE,"pull","_v2manifest.pt"),map_location="cpu",weights_only=False)
pbc={c:man["poses"][i].double().numpy() for i,c in enumerate(man["clip_id"])}
dman=json.load(open(os.path.join(DUMP,"manifest.json")))
clips=[]
for e in dman["episodes"]: clips += [e["clip_id"]]*e["n_windows"]
gt=np.zeros((N,8,2)); sv=np.zeros((N,8))
for i in range(N):
    P=pbc[clips[i]]; T=P.shape[0]; t0=int(ws[i]); a=np.array([t0+k for k in HORIZONS])
    sv[i]=(a<=T-1).astype(float); q=P[np.clip(a,None,T-1),:2]-P[t0,:2]
    c,s=np.cos(-P[t0,2]),np.sin(-P[t0,2])
    gt[i,:,0]=q[:,0]*c-q[:,1]*s; gt[i,:,1]=q[:,0]*s+q[:,1]*c
svn=np.maximum(sv.sum(1),1.0)
def roll(ctrl, v0v):
    B,M=len(v0v),len(ctrl)
    kap=np.clip(ctrl[None,:,1]/np.maximum(v0v,V_FLOOR)[:,None]**2,-KAPPA_CAP,KAPPA_CAP)
    a=np.repeat(ctrl[None,:,0],B,axis=0)
    x=np.zeros((B,M)); y=np.zeros((B,M)); yaw=np.zeros((B,M))
    v=np.repeat(v0v[:,None],M,axis=1).copy(); wp=np.zeros((B,M,8,2))
    for k in range(H):
        x=x+v*np.cos(yaw)*DT; y=y+v*np.sin(yaw)*DT
        yaw=yaw+v*kap*DT; v=np.maximum(v+a*DT,0.0)
        if k in SLOTS: wp[:,:,SLOTS.index(k),0]=x; wp[:,:,SLOTS.index(k),1]=y
    return wp, yaw
# the withheld-row bank is IDENTICAL for every window: one roll at ref_speed
wp_ref, yaw_ref = roll(CTRL, np.array([REF_SPEED]))
sup_ref = float(deg(np.abs(yaw_ref[0])).max())
eb_ref  = float(deg(np.abs(np.arctan2(wp_ref[0,:,-1,1], wp_ref[0,:,-1,0]))).max())
n30 = int((deg(np.abs(yaw_ref[0]))>30).sum()); n45=int((deg(np.abs(yaw_ref[0]))>45).sum())
n60 = int((deg(np.abs(yaw_ref[0]))>60).sum())
print(f"withheld-row bank (rolled at ref_speed {REF_SPEED} m/s, identical for every window):")
print(f"  max terminal heading {sup_ref:.1f} deg, max end-bearing {eb_ref:.1f} deg; "
      f"candidates >30/45/60 deg = {n30}/{n45}/{n60} of 117")
# STOP THE OBVIOUS MISTAKE: on a WITHHELD row the reach band is NOT applied at
# the true v0. `refc.py:1538` and `:1681` both do `keep = keep | (~ego_keep)`,
# precisely so the withheld channel cannot decide which candidates exist. So a
# withheld row keeps ALL 117. Applying the band at the true v0 here would have
# manufactured a defect (it empties 130 windows) that the code already forbids.
CH=512; ade_ref=np.zeros(N); sup_ref_w=np.zeros(N); nsurv=np.zeros(N,dtype=np.int64)
for s in range(0,N,CH):
    e=min(s+CH,N); b=e-s
    wp = np.repeat(wp_ref, b, axis=0); yw = np.repeat(yaw_ref, b, axis=0)
    keep = np.ones((b, wp.shape[1]), dtype=bool)      # ego_keep=False -> keep all
    band = sl.anchor_reachability_mask(torch.from_numpy(wp), torch.from_numpy(v0[s:e]),
             accel_max=SEL_A, horizon_s=6.0).numpy()
    if s == 0:
        print("  (for scale only: the band AT THE TRUE v0 would keep "
              "%.1f/117 and empty %d of these %d rows -- NOT what the code does)"
              % (band.sum(1).mean(), int((~band.any(1)).sum()), b))
    nsurv[s:e]=keep.sum(1)
    sup_ref_w[s:e]=np.where(keep,np.abs(yw),-1).max(1)
    dd=wp-gt[s:e,None]
    ade=(np.sqrt((dd**2).sum(-1))*sv[s:e,None]).sum(-1)/svn[s:e,None]
    ade_ref[s:e]=np.where(keep,ade,np.inf).min(1)
empty=int((nsurv==0).sum())
assert empty == 0, empty
r={"ref_speed_ms":REF_SPEED,
   "withheld_bank_max_terminal_heading_deg":sup_ref,
   "withheld_bank_max_end_bearing_deg":eb_ref,
   "withheld_bank_n_gt30_45_60":[n30,n45,n60],
   "survivors_per_window":float(nsurv.mean()),"empty_windows":empty,
   "pct_no_gt30deg_terminal_heading":float(100.0*(deg(sup_ref_w)<=30).mean()),
   "pct_no_gt45deg_terminal_heading":float(100.0*(deg(sup_ref_w)<=45).mean()),
   "pct_no_gt60deg_terminal_heading":float(100.0*(deg(sup_ref_w)<=60).mean()),
   "pct_demand_exceeds_supply":float(100.0*(dem_thx>sup_ref_w).mean()),
   "oiv_ade_withheld_m":float(ade_ref.mean()),
   "oiv_ade_measured_v0_m":float(ade_meas.mean())}
bs = PB(ade_ref, ade_meas, eid, n_boot=2000, seed=0)
r["paired_delta_withheld_minus_measured"]={"delta":bs["delta"],"lo":bs["lo"],
    "hi":bs["hi"],"separated":bs["separated"],"n_windows":bs["n_windows"],
    "n_episodes":bs["n_episodes"],"estimator":bs["estimator"]}
print(f"  reach clamp at the window's OWN v0: survivors/window {r['survivors_per_window']:.1f}, "
      f"empty {empty}")
print(f"  windows with NO >30/45/60 deg candidate: "
      f"{r['pct_no_gt30deg_terminal_heading']:.2f} / "
      f"{r['pct_no_gt45deg_terminal_heading']:.2f} / "
      f"{r['pct_no_gt60deg_terminal_heading']:.2f} %   "
      f"demand>supply {r['pct_demand_exceeds_supply']:.2f} %")
print(f"  OIV ADE withheld {r['oiv_ade_withheld_m']:.4f} m vs measured-v0 "
      f"{r['oiv_ade_measured_v0_m']:.4f} m  delta {bs['delta']:+.4f} "
      f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}] "
      f"{'SEPARATED' if bs['separated'] else 'not separated'}")
# by-speed split: the withheld ceiling is a mechanical consequence of a FIXED
# 10 m/s bank meeting a speed-varying corpus, so it must be read per speed band.
rows=[]
print("  by v0 band (withheld bank vs measured-v0 bank):")
for lo,hi in ((0,5),(5,10),(10,15),(15,20),(20,25),(25,99)):
    m=(v0>=lo)&(v0<hi)
    if not m.any(): continue
    rows.append({"v0_band_ms":[lo,hi],"n":int(m.sum()),
                 "oiv_withheld_m":float(ade_ref[m].mean()),
                 "oiv_measured_v0_m":float(ade_meas[m].mean())})
    print("    [%2d,%2d) n %5d   withheld %8.4f   measured-v0 %7.4f"
          % (lo,hi,int(m.sum()),ade_ref[m].mean(),ade_meas[m].mean()))
r["by_v0_band"]=rows
json.dump(r, open(os.path.join(OUT,"TURNCOV6S_PART6.json"),"w"), indent=1)
print("wrote TURNCOV6S_PART6.json")
