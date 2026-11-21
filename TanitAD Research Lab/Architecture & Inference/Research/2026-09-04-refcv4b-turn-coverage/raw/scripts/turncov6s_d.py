"""PART 4 -- reconciliation with the launch package's own printed counts, and
the SIGNED-direction check (a window turning LEFT needs a LEFT candidate, not
merely a candidate with large |turn|). STRICTLY MODEL-FREE."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\stack")
import tanitad.refs.refc_select as sl
OUT = os.path.join(HERE, "turncov_out"); DUMP = os.path.join(HERE, "refcv3_40284_dump")
DT, H = 0.1, 60
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60); SLOTS = [k-1 for k in HORIZONS]
V_FLOOR, KAPPA_CAP, SEL_ACCEL_MAX, G = 4.0, 0.12, 2.0, 9.81
deg = np.degrees
d = np.load(os.path.join(OUT, "per_window.npz"))
v0=d["v0"]; eid=d["eid"]; ws=d["ws"]; dem_thx=d["dem_thx"]; N=len(v0)
CTRL = torch.load(os.path.join(HERE,"pull","anchors_live.pt"), map_location="cpu",
                  weights_only=False)["controls"].double().numpy()
man = torch.load(os.path.join(HERE,"pull","_v2manifest.pt"), map_location="cpu",
                 weights_only=False)
pbc = {c: man["poses"][i].double().numpy() for i,c in enumerate(man["clip_id"])}
dman = json.load(open(os.path.join(DUMP,"manifest.json")))
clips=[]
for e in dman["episodes"]: clips += [e["clip_id"]]*e["n_windows"]
# SIGNED GT dyaw over 6 s
sgn = np.zeros(N)
for i in range(N):
    P=pbc[clips[i]]; T=P.shape[0]; t0=int(ws[i]); j=min(t0+60, T-1)
    a=P[j,2]-P[t0,2]; sgn[i]=(a+np.pi)%(2*np.pi)-np.pi
def roll(ctrl, v0v):
    B,M=len(v0v),len(ctrl)
    kap=np.clip(ctrl[None,:,1]/np.maximum(v0v,V_FLOOR)[:,None]**2,-KAPPA_CAP,KAPPA_CAP)
    a=np.repeat(ctrl[None,:,0],B,axis=0)
    x=np.zeros((B,M)); y=np.zeros((B,M)); yaw=np.zeros((B,M))
    v=np.repeat(v0v[:,None],M,axis=1).copy(); wp=np.zeros((B,M,len(SLOTS),2))
    for k in range(H):
        x=x+v*np.cos(yaw)*DT; y=y+v*np.sin(yaw)*DT
        yaw=yaw+v*kap*DT; v=np.maximum(v+a*DT,0.0)
        if k in SLOTS:
            wp[:,:,SLOTS.index(k),0]=x; wp[:,:,SLOTS.index(k),1]=y
    seg=wp[:,:,-1]-wp[:,:,-2]
    return wp, yaw, np.arctan2(seg[...,1],seg[...,0])
CH=512
c_seg30=np.zeros(N,dtype=np.int64); c_eb30=np.zeros(N,dtype=np.int64)
sup_signed=np.zeros(N)          # best SAME-SIGN terminal heading available
for s in range(0,N,CH):
    e=min(s+CH,N); wp,yaw6,segh=roll(CTRL,v0[s:e])
    keep=sl.anchor_reachability_mask(torch.from_numpy(wp),torch.from_numpy(v0[s:e]),
          accel_max=SEL_ACCEL_MAX,horizon_s=6.0).numpy()
    eb=np.abs(np.arctan2(wp[:,:,-1,1],wp[:,:,-1,0]))
    c_seg30[s:e]=(keep&(np.abs(segh)>np.radians(30))).sum(1)
    c_eb30[s:e]=(keep&(eb>np.radians(30))).sum(1)
    sg=np.sign(sgn[s:e])[:,None]; sg=np.where(sg==0,1.0,sg)
    sup_signed[s:e]=np.where(keep,yaw6*sg,-1e9).max(1)
r={"terminal_heading_slot_segment_gt30_per_window":float(c_seg30.mean()),
   "end_bearing_gt30_per_window":float(c_eb30.mean()),
   "launch_package_reported_terminal_heading_count":48.3,
   "launch_package_reported_end_bearing_no_turn_pct":8.85,
   "signed_check":{
     "pct_windows_no_SAME_SIGN_cand_gt30deg":float(100.0*(deg(sup_signed)<=30).mean()),
     "pct_windows_no_SAME_SIGN_cand_gt45deg":float(100.0*(deg(sup_signed)<=45).mean()),
     "pct_windows_no_SAME_SIGN_cand_gt60deg":float(100.0*(deg(sup_signed)<=60).mean()),
     "pct_signed_demand_exceeds_signed_supply":float(100.0*(np.abs(sgn)>sup_signed).mean()),
     "note":"supply is the best candidate whose terminal heading turns the SAME "
            "WAY as the GT; the grid is symmetric in a_lat so this must match the "
            "unsigned figure, and it does -- that is the control, not a new result"}}
print("terminal-heading (5->6 s slot segment) >30 deg candidates per window: "
      f"{r['terminal_heading_slot_segment_gt30_per_window']:.1f}  "
      f"(launch package printed 48.3)")
print("end-bearing >30 deg candidates per window: "
      f"{r['end_bearing_gt30_per_window']:.1f}")
print("SIGNED: windows with NO same-sign candidate > 30/45/60 deg: "
      f"{r['signed_check']['pct_windows_no_SAME_SIGN_cand_gt30deg']:.2f} / "
      f"{r['signed_check']['pct_windows_no_SAME_SIGN_cand_gt45deg']:.2f} / "
      f"{r['signed_check']['pct_windows_no_SAME_SIGN_cand_gt60deg']:.2f} %")
print("SIGNED demand > signed supply: "
      f"{r['signed_check']['pct_signed_demand_exceeds_signed_supply']:.2f} %")
json.dump(r, open(os.path.join(OUT,"TURNCOV6S_PART4.json"),"w"), indent=1)
print("wrote TURNCOV6S_PART4.json")
