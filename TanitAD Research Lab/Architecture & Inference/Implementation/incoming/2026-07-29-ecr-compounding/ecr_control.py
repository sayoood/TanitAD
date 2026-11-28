"""E-CR DISCRIMINATING CONTROL — is CR<1 a readout input-distribution artifact?

Three arms decoded by the SAME grounding.step["op"], on the SAME windows:
  A ROLLOUT   pairs (z_hat_{j-1}, z_hat_j)   -- self-consistent PREDICTED (the canary)
  B TEACHER   pairs (z_true_{j-1}, z_hat_j)  -- MIXED (what CR currently uses)
  C ORACLE    pairs (z_true_{j-1}, z_true_j) -- self-consistent TRUE, no prediction at all

C is the diagnostic. If C BEATS A, the readout handles true pairs fine and B fails only
because the pair is MIXED => CR as defined is mis-specified. If C LOSES to A, the readout
itself is tuned to predicted-latent statistics => decoded displacement cannot carry CR at
all and CR must move to LATENT error.
"""
from __future__ import annotations
import sys, glob, json, time
sys.path.insert(0,"/workspace/TanitAD/stack"); sys.path.insert(0,"/workspace/TanitAD/stack/scripts")
import torch, numpy as np
import eval_flagship_v4 as E
from tanitad.data.mixing import load_episode
from tanitad.models.flagship_v15 import SPEED_SCALE
from tanitad.models.metric_dynamics import gt_ego_waypoints, accumulate_se2
CK=sys.argv[1]; OUT=sys.argv[2]; EPS=int(sys.argv[3]) if len(sys.argv)>3 else 4
VAL="/workspace/val40cache"; K=20; RK=(4,8,16,20); STRIDE,BATCH=8,8
dev=torch.device("cuda")
ck=torch.load(CK,map_location="cpu",weights_only=False)
world,grounding,step=E.load_v1_from_ck(ck,dev)[:3]; sr=grounding.step["op"]
eps=[load_episode(f,mmap=True) for f in sorted(glob.glob(VAL+"/ep_*.pt"))]
cfg=E._eval_cfg(None); plan=E._plan(cfg); ds=E.build_val_dataset_base(eps,cfg,plan)
pos={(e,t):i for i,(e,t) in enumerate(ds.index)}
sel=[i for i,(e,t) in enumerate(ds.index) if e<EPS and t%STRIDE==0]
print(f"[ctl] step={step} windows={len(sel)}",flush=True)
def dec(trans,k):
    return accumulate_se2(torch.stack([sr(trans[j][0],trans[j][1]) for j in range(k)],1))
A,B,C,EPi=[],[],[],[]
for b0 in range(0,len(sel),BATCH):
    idx=sel[b0:b0+BATCH]; items=[ds[i] for i in idx]; ets=[ds.index[i] for i in idx]
    fr=torch.stack([x["frames"] for x in items]).to(dev)
    aw2=torch.stack([x["actions"] for x in items]).to(dev).float()
    fa2=torch.stack([x["future_actions"] for x in items]).to(dev).float()
    fp=torch.stack([x["future_poses"] for x in items]).to(dev).float()
    pl=torch.stack([x["pose_last"] for x in items]).to(dev).float()
    vch=(pl[:,3]/SPEED_SCALE)[:,None,None]
    aw=torch.cat([aw2,vch.expand(-1,aw2.shape[1],-1)],-1); fa=torch.cat([fa2,vch.expand(-1,fa2.shape[1],-1)],-1)
    gt=gt_ego_waypoints(pl,fp,list(range(1,K+1)))
    futf=[];ok=True
    for (e,t) in ets:
        sq=[]
        for j in range(K):
            kk=pos.get((e,t+j+1))
            if kk is None: ok=False;break
            sq.append(ds[kk]["frames"][-1])
        if not ok: break
        futf.append(torch.stack(sq))
    if not ok: continue
    fut=torch.stack(futf).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=True):
        st=world.encode_window(fr)
        Bn,Kk=fut.shape[0],fut.shape[1]
        zt=world.encode(fut.reshape(Bn*Kk,*fut.shape[2:])).reshape(Bn,Kk,-1)
        ws,wa,tA=st,aw,[]
        for j in range(K):
            zh=world.predictor(ws,wa)[1]; tA.append((ws[:,-1],zh))
            if j<K-1:
                ws=torch.cat([ws[:,1:],zh.unsqueeze(1)],1); wa=torch.cat([wa[:,1:],fa[:,j].unsqueeze(1)],1)
        ws,wa,tB=st,aw,[]
        for j in range(K):
            zh=world.predictor(ws,wa)[1]; tB.append((ws[:,-1],zh))
            if j<K-1:
                ws=torch.cat([ws[:,1:],zt[:,j].unsqueeze(1)],1); wa=torch.cat([wa[:,1:],fa[:,j].unsqueeze(1)],1)
        chain=torch.cat([st[:,-1:],zt[:,:K-1]],1)
        tC=[(chain[:,j],zt[:,j]) for j in range(K)]
        wA,wB,wC=dec(tA,K),dec(tB,K),dec(tC,K)
    A.append((wA.float()-gt).norm(dim=-1).cpu()); B.append((wB.float()-gt).norm(dim=-1).cpu())
    C.append((wC.float()-gt).norm(dim=-1).cpu()); EPi.append(torch.tensor([e for (e,t) in ets]))
A=torch.cat(A);B=torch.cat(B);C=torch.cat(C);EPi=torch.cat(EPi)
out={"ckpt_step":int(step),"n_windows":int(A.shape[0]),"n_episodes":int(EPi.unique().numel()),"arms":{}}
for k in RK:
    a,b,c=float(A[:,k-1].mean()),float(B[:,k-1].mean()),float(C[:,k-1].mean())
    out["arms"][f"k{k}"]={"A_rollout":round(a,5),"B_teacher_mixed":round(b,5),"C_oracle_true":round(c,5),
                          "CR_B_over_A":round(b and a/b or 0,4),"C_beats_A":bool(c<a)}
np.savez("/workspace/ecr_control_arrays.npz",A=A.numpy(),B=B.numpy(),C=C.numpy(),episode=EPi.numpy())
json.dump(out,open(OUT,"w"),indent=1); print(json.dumps(out["arms"],indent=1),flush=True)
