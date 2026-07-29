"""E-DPSI — is our target-speed head keyed to HEADING rather than to the scene?

PlanT 2.0 (DS 92.4, above every sensor model listed) carries a POSITIONAL SHORTCUT: its
predicted speed jumps abruptly at ego rotations of 10-15 deg. We can test a STRICTLY CLEANER
version than its authors could: pseudosim dyaw is an EXACT camera rotation (H = K R K^-1,
max|dH| = 0.000e+00 over 30 conditions) applied to BIT-IDENTICAL REAL FOOTAGE, so only the
observed heading changes. PlanT rotates the ego in simulation, moving world pose AND object
input together.

LOOK FOR A STEP, NOT A SLOPE.

PRE-REGISTERED (PREREG_deep_research_2026-07-29.md):
 - STEP inside the envelope => part of our 88.7% longitudinal oracle gap is a SHORTCUT, and
   the fix is heading AUGMENTATION rather than a new channel or more capacity.
 - FLAT/monotone-smooth => strengthens the estimation-problem reading (consistent with the
   IDM monocular scale limit) and closes the shortcut hypothesis cheaply.
 - THE SHARP LIMIT: our measurement-grade envelope is |dpsi| <= 12 deg; PlanT onset is
   10-15 deg. WE COVER THE LOWER EDGE ONLY. A null inside 12 deg means "NO SHORTCUT BELOW
   12 DEG" and NEVER "we are clean". A null also would NOT contradict PlanT, whose CARLA
   scripted-expert root cause is absent from human logs by construction.
"""
from __future__ import annotations
import sys, glob, json, time
sys.path.insert(0,"/workspace/TanitAD/stack"); sys.path.insert(0,"/workspace/TanitAD/stack/scripts")
sys.path.insert(0,"/workspace/TanitAD/taniteval")
import torch, numpy as np
import eval_flagship_v4 as E
from tanitad.data.mixing import load_episode
from tanitad.models.flagship_v15 import SPEED_SCALE
from tanitad.models.strategic_goal import GoalScalarHead, GoalScalarConfig
from taniteval.clhorizon import sampling_homography, warp_batch
CK=sys.argv[1]; OUT=sys.argv[2]; EPS=int(sys.argv[3]) if len(sys.argv)>3 else 40
VAL="/workspace/val40cache"; STRIDE,BATCH=8,8
DPSI=(-12.0,-8.0,-4.0,0.0,4.0,8.0,12.0)          # GridSpec default = the validated envelope
dev=torch.device("cuda")
ck=torch.load(CK,map_location="cpu",weights_only=False)
world,grounding,step=E.load_v1_from_ck(ck,dev)[:3]
gsd=ck.get("goal_head")
assert isinstance(gsd,dict) and gsd, "checkpoint carries no goal_head — E-DPSI needs one"
in_dim=gsd["net.0.weight"].shape[1]
gh=GoalScalarHead(GoalScalarConfig(in_dim=in_dim,hidden=gsd["net.0.weight"].shape[0],
                                   n_out=gsd["net.2.weight"].shape[0]))
gh.load_state_dict(gsd); gh=gh.to(dev).eval()
eps=[load_episode(f,mmap=True) for f in sorted(glob.glob(VAL+"/ep_*.pt"))]
cfg=E._eval_cfg(None); plan=E._plan(cfg); ds=E.build_val_dataset_base(eps,cfg,plan)
sel=[i for i,(e,t) in enumerate(ds.index) if e<EPS and t%STRIDE==0]
print(f"[dpsi] windows={len(sel)} dpsi={DPSI}",flush=True)
rows={}; t0=time.time()
for dp in DPSI:
    H=sampling_homography(0.0,float(dp)).to(dev).float()
    ts=[]; epi=[]
    for b0 in range(0,len(sel),BATCH):
        items=[ds[i] for i in sel[b0:b0+BATCH]]; ets=[ds.index[i] for i in sel[b0:b0+BATCH]]
        fr=torch.stack([x["frames"] for x in items]).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=True):
            fw = fr if dp==0.0 else warp_batch(fr, H.unsqueeze(0).expand(fr.shape[0],-1,-1))
            st=world.encode_window(fw)
            sc=gh(st[:,-1])                       # [B,4] = ttm, curv3s, curv5s, tspeed_5s
        ts.append(sc[:,3].detach().float().cpu()); epi.append(torch.tensor([e for (e,t) in ets]))
    T=torch.cat(ts); EPi=torch.cat(epi)
    rows[f"{dp:+.1f}"]={"tspeed_5s_mean":round(float(T.mean()),5),
                        "tspeed_5s_median":round(float(T.median()),5),
                        "tspeed_5s_std":round(float(T.std()),5),"n":int(T.numel())}
    np.savez(f"/workspace/edpsi_dp{dp:+.0f}.npz",tspeed=T.numpy(),episode=EPi.numpy())
    print(f"[dpsi] dpsi={dp:+6.1f} tspeed_5s mean={float(T.mean()):.5f} "
          f"median={float(T.median()):.5f} n={T.numel()} ({time.time()-t0:.0f}s)",flush=True)
base=rows["+0.0"]["tspeed_5s_mean"]
out={"ckpt_step":int(step),"val":VAL,"dpsi_deg":list(DPSI),"rows":rows,
     "delta_vs_zero":{k:round(v["tspeed_5s_mean"]-base,5) for k,v in rows.items()},
     "envelope_note":"|dpsi| <= 12 deg is the MEASUREMENT-GRADE envelope (0% out-of-envelope). "
                     "PlanT 2.0 onset is 10-15 deg: WE COVER THE LOWER EDGE ONLY. A null here "
                     "means NO SHORTCUT BELOW 12 DEG, never we-are-clean.",
     "read":"LOOK FOR A STEP, NOT A SLOPE."}
json.dump(out,open(OUT,"w"),indent=1); print(json.dumps(out["delta_vs_zero"],indent=1),flush=True)
