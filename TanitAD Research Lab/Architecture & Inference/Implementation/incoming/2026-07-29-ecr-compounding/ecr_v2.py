"""E-CR v2 — CR on LATENT error, no decoder in the path (C63 redesign).

e_k = 1 - cos(z_hat_k, z_true_k). BOTH arms are the predictor own outputs compared against
the encoder, so nothing crosses a distribution boundary and step_readout is never invoked.
  rollout : window advanced with its OWN predictions  -> compounding allowed
  teacher : window advanced with TRUE latents         -> 1-step-from-truth at every k
CR_k = e_rollout / e_teacher. Pre-registered outcomes unchanged (PREREG 2026-07-29):
CR~1 => H-TASK, CR>1 => H-COMPOUND, CI covering => UNDERPOWERED.
WARNING: this answers the WORLD-MODEL question and does NOT speak in metres. The link from
latent error to ADE runs through the readout C63 showed is domain-sensitive. Do not convert.
"""
from __future__ import annotations
import sys, glob, json, time
sys.path.insert(0,"/workspace/TanitAD/stack"); sys.path.insert(0,"/workspace/TanitAD/stack/scripts")
import torch, numpy as np, torch.nn.functional as F
import eval_flagship_v4 as E
from tanitad.data.mixing import load_episode
from tanitad.models.flagship_v15 import SPEED_SCALE
CK=sys.argv[1]; OUT=sys.argv[2]; EPS=int(sys.argv[3]) if len(sys.argv)>3 else 40
VAL="/workspace/val40cache"; K=20; RK=(4,8,16,20); STRIDE,BATCH=8,8
dev=torch.device("cuda")
ck=torch.load(CK,map_location="cpu",weights_only=False)
world,grounding,step=E.load_v1_from_ck(ck,dev)[:3]
eps=[load_episode(f,mmap=True) for f in sorted(glob.glob(VAL+"/ep_*.pt"))]
cfg=E._eval_cfg(None); plan=E._plan(cfg); ds=E.build_val_dataset_base(eps,cfg,plan)
pos={(e,t):i for i,(e,t) in enumerate(ds.index)}
sel=[i for i,(e,t) in enumerate(ds.index) if e<EPS and t%STRIDE==0]
print(f"[ecr2] step={step} windows={len(sel)}",flush=True)
R,T,EPi=[],[],[]; t0=time.time()
for b0 in range(0,len(sel),BATCH):
    idx=sel[b0:b0+BATCH]; items=[ds[i] for i in idx]; ets=[ds.index[i] for i in idx]
    fr=torch.stack([x["frames"] for x in items]).to(dev)
    aw2=torch.stack([x["actions"] for x in items]).to(dev).float()
    fa2=torch.stack([x["future_actions"] for x in items]).to(dev).float()
    pl=torch.stack([x["pose_last"] for x in items]).to(dev).float()
    vch=(pl[:,3]/SPEED_SCALE)[:,None,None]
    aw=torch.cat([aw2,vch.expand(-1,aw2.shape[1],-1)],-1); fa=torch.cat([fa2,vch.expand(-1,fa2.shape[1],-1)],-1)
    # C63-followup FIX: drop offending windows INDIVIDUALLY. The previous version hit
    # `continue` and discarded the ENTIRE BATCH of 8 whenever any one window lacked a
    # full K-step future, which threw away 393/881 windows and skewed the retained set
    # systematically toward episode STARTS. Keep-mask instead.
    futf=[];keep=[]
    for bi,(e,t) in enumerate(ets):
        sq=[];okw=True
        for j in range(K):
            kk=pos.get((e,t+j+1))
            if kk is None: okw=False;break
            sq.append(ds[kk]["frames"][-1])
        if okw:
            futf.append(torch.stack(sq)); keep.append(bi)
    if not futf: continue
    if len(keep)!=len(ets):
        km=torch.tensor(keep,device=dev)
        fr=fr[km]; aw=aw[km]; fa=fa[km]; pl=pl[km]
        ets=[ets[i] for i in keep]
    fut=torch.stack(futf).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=True):
        st=world.encode_window(fr); Bn=fut.shape[0]
        zt=world.encode(fut.reshape(Bn*K,*fut.shape[2:])).reshape(Bn,K,-1)
        ws,wa,zr=st,aw,[]
        for j in range(K):
            zh=world.predictor(ws,wa)[1]; zr.append(zh)
            if j<K-1: ws=torch.cat([ws[:,1:],zh.unsqueeze(1)],1); wa=torch.cat([wa[:,1:],fa[:,j].unsqueeze(1)],1)
        ws,wa,zf=st,aw,[]
        for j in range(K):
            zh=world.predictor(ws,wa)[1]; zf.append(zh)
            if j<K-1: ws=torch.cat([ws[:,1:],zt[:,j].unsqueeze(1)],1); wa=torch.cat([wa[:,1:],fa[:,j].unsqueeze(1)],1)
        zr=torch.stack(zr,1); zf=torch.stack(zf,1)
    R.append((1-F.cosine_similarity(zr.float(),zt.float(),dim=-1)).cpu())
    T.append((1-F.cosine_similarity(zf.float(),zt.float(),dim=-1)).cpu())
    EPi.append(torch.tensor([e for (e,t) in ets]))
    if b0%(BATCH*40)==0: print(f"[ecr2] {b0}/{len(sel)} ({time.time()-t0:.0f}s)",flush=True)
R=torch.cat(R);T=torch.cat(T);EPi=torch.cat(EPi)
out={"ckpt_step":int(step),"metric":"1 - cos(z_hat_k, z_true_k)","n_windows":int(R.shape[0]),
     "n_episodes":int(EPi.unique().numel()),"CR":{}}
for k in RK:
    er,et=float(R[:,k-1].mean()),float(T[:,k-1].mean())
    out["CR"][f"k{k}"]={"e_rollout":round(er,6),"e_teacher_forced":round(et,6),
                        "CR":round(er/max(et,1e-12),4),
                        "ER":round(float((R[:,k-1]-R[:,k-2]).mean()),6)}
np.savez("/workspace/ecr_v2_arrays.npz",rollout=R.numpy(),tf=T.numpy(),episode=EPi.numpy())
json.dump(out,open(OUT,"w"),indent=1); print(json.dumps(out["CR"],indent=1),flush=True)
print(f"n_windows={out[chr(34)+chr(110)+chr(95)+chr(119)+chr(105)+chr(110)+chr(100)+chr(111)+chr(119)+chr(115)+chr(34)]} n_eps={out[chr(34)+chr(110)+chr(95)+chr(101)+chr(112)+chr(105)+chr(115)+chr(111)+chr(100)+chr(101)+chr(115)+chr(34)]}",flush=True)
