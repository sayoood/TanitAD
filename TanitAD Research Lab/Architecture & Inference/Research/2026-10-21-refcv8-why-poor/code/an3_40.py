import sys, glob, numpy as np, json
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
from an2 import boot_ci, pcb
R="C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf"
seed=sys.argv[1] if len(sys.argv)>1 else "0"
fs=sorted(glob.glob(f"{R}/s6_s{seed}/ep*.npz")); fd=sorted(glob.glob(f"{R}/dump_s{seed}/decisions/ep*.npz"))
assert len(fs)==len(fd)
def cls_thr(P,v0,dt=1.0,thr=1.5):
    v_end=np.linalg.norm(P[:,-1]-P[:,-2],axis=-1)/dt; dv=v_end-v0
    return np.where(dv<=-thr,"decel",np.where(dv>=thr,"accel","cruise")).astype(object)
rows=[]
A={k:[] for k in ("eid","v0","g","os","r7","osz","lonhead","lathead","lonhead_z","navcmd","lonlab","latlab","ws","plan8","gt8")}
for i,(a,b) in enumerate(zip(fs,fd)):
    s=np.load(a,allow_pickle=True); d=np.load(b,allow_pickle=True)
    # align by ws
    ws_s=s['ws']; ws_d=d['ws']
    common=np.intersect1d(ws_s,ws_d)
    is_=np.searchsorted(ws_s,common) if (np.diff(ws_s)>0).all() else None
    ids=np.searchsorted(ws_d,common) if (np.diff(ws_d)>0).all() else None
    assert is_ is not None and ids is not None
    assert (ws_s[is_]==common).all() and (ws_d[ids]==common).all()
    A['eid'].append(np.full(len(common),i)); A['v0'].append(s['v0'][is_]); A['g'].append(s['g'][is_]); A['os'].append(s['os'][is_]); A['r7'].append(s['b_refcv7_50400_s'+seed][is_]); A["osz"].append(s["os_navzero"][is_] if "os_navzero" in s.files else s["os"][is_])
    A['lonhead'].append(d['lon_pred_nav_true'][ids]); A['lathead'].append(d['lat_pred_nav_true'][ids]); A['lonhead_z'].append(d['lon_pred_nav_zero'][ids] if 'lon_pred_nav_zero' in d.files else np.full(len(ids),-1))
    A['navcmd'].append(d['nav_cmd'][ids]); A['lonlab'].append(d['lon_label'][ids]); A['latlab'].append(d['lat_label'][ids]); A['ws'].append(common)
    A['plan8'].append(d['plan_full_nav_true'][ids]); A['gt8'].append(d['gt_future_ext'][ids][:,:, :2])
Z={k:np.concatenate(v) for k,v in A.items()}
n=len(Z['ws']); print("windows joined",n,"seed",seed)
print("control: plan8 slot idx [1,3,4,5,6,7] equals s6 os:",np.abs(Z['plan8'][:,[1,3,4,5,6,7]]-Z['os']).max())
# GT cross-check: s6 g vs gt_future_ext ticks
gi=[9,19,29,39,49,59]
print("control: g vs gt_future_ext ticks max abs diff:",np.abs(Z['gt8'][:,gi]-Z['g']).max())
v0=Z['v0'].astype(np.float64)
gtc=cls_thr(Z['g'].astype(np.float64),v0); c8=cls_thr(Z['os'].astype(np.float64),v0); c7=cls_thr(Z['r7'].astype(np.float64),v0); cz=cls_thr(Z['osz'].astype(np.float64),v0)
eid=Z['eid']
cru=gtc=="cruise"
print("GT-cruise windows (all S6, no compliance mask):",int(cru.sum()),"GT decel",int((gtc=="decel").sum()),"GT accel",int((gtc=="accel").sum()))
def ind(c,kind):
    if kind=="fd": return np.where(cru,(c=="decel").astype(float),np.nan)
    if kind=="md": return np.where(gtc=="decel",(c!="decel").astype(float),np.nan)
for nm,c in (("refcv8_os",c8),("refcv7_50400",c7),("refcv8_os_navzero",cz)):
    print(nm,"false-decel on GT cruise",[round(x,4) for x in boot_ci(ind(c,'fd'),eid)],"missed-decel on GT decel",[round(x,4) for x in boot_ci(ind(c,'md'),eid)])
print("paired r8-r7 false decel",[round(x,4) for x in pcb(ind(c8,'fd'),ind(c7,'fd'),eid)])
# head-level
D_SET=(2,3,4,5,6)
lh=Z['lonhead']; inD=np.isin(lh,D_SET)
print("head-in-D on GT cruise (head label cruise per plan class):",round(float(inD[cru].mean()),4))
fb=cru&(c8=="decel")
print("share of false-brake windows with head in D",round(float(inD[fb].mean()),4),"n_fb",int(fb.sum()))
print("P(plan decel | head in D, GT cruise)",round(float((c8[cru&inD]=='decel').mean()),4),"n",int((cru&inD).sum()),"; P(plan decel | head not in D, GT cruise)",round(float((c8[cru&~inD]=='decel').mean()),4),"n",int((cru&~inD).sum()))
# nav-token conditioned
nav=Z['navcmd']
for v,nmv in ((0,"nav FOLLOW"),(1,"nav L"),(2,"nav R")):
    m=cru&(nav==v)
    if m.sum(): print(nmv,"GT-cruise n",int(m.sum()),"false-decel r8",round(float((c8[m]=='decel').mean()),4),"r7",round(float((c7[m]=='decel').mean()),4),"head-in-D",round(float(inD[m].mean()),4))
np.savez_compressed(S+f"Z40_s{seed}.npz",**{k:v for k,v in Z.items()})
