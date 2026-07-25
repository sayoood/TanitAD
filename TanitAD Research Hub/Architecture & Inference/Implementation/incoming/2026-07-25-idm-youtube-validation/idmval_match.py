"""Are the two PhysicalAI-val builds the same clips? Fingerprint-match
0c5f7dac3b11 (eval) episodes to the pod3 va_* latents (f1b378f295ae) by pose
profile, and for ep_00009 compare per-clip head metrics: eval-encode z vs the
pod3-latent z. Localizes the from-video gap to cache-build vs encoder-env."""
import sys, glob, torch, numpy as np
sys.path.insert(0,"/root/v4eval/stack"); sys.path.insert(0,"/root/v4eval/stack/scripts")
import idm_head as ih, run_idm_proof as R
dev="cuda"
VAL="/root/valdata/physicalai-val-0c5f7dac3b11"

def sig(poses):
    v=poses[:,3].numpy()
    pl=float(np.linalg.norm(np.diff(poses[:,:2].numpy(),axis=0),axis=1).sum())
    return np.array([poses.shape[0], float(v.mean()), float(v.max()), float(v.min()), pl])

# signatures
va=sorted(glob.glob("/root/idmval/va40/*.pt"))
va_sig={}
for f in va:
    d=torch.load(f,weights_only=False); va_sig[f]=sig(d["poses"].float())
ev_sig={}
for i in range(40):
    d=torch.load(f"{VAL}/ep_{i:05d}.pt",weights_only=False); ev_sig[i]=(sig(d["poses"].float()), int(d.get("episode_id",-1)))

# greedy nearest match by normalized signature
alls=np.stack([s for s,_ in ev_sig.values()]+[va_sig[f] for f in va])
mu,sd=alls.mean(0),alls.std(0)+1e-9
def nz(x): return (x-mu)/sd
matches=0; near=[]
for i in range(40):
    es=nz(ev_sig[i][0])
    best=min(va,key=lambda f:np.linalg.norm(nz(va_sig[f])-es))
    dst=float(np.linalg.norm(nz(va_sig[best])-es))
    near.append((i,ev_sig[i][1],best.split("/")[-1],dst))
    if dst<0.05: matches+=1
print("SIG COLS = [T, mean_v, max_v, min_v, path_len]")
print("exact-ish matches (dist<0.05):",matches,"/40")
print("closest va per eval ep (idx, epid, va, dist):")
for r in sorted(near,key=lambda x:x[3])[:6]: print("  MATCH",r)
for r in sorted(near,key=lambda x:-x[3])[:4]: print("  FAR  ",r)

# ep_00009 direct: eval-encode vs pod3-latent (its nearest va)
i9=9; e9=ev_sig[i9]
best9=min(va,key=lambda f:np.linalg.norm(nz(va_sig[f])-nz(e9[0])))
print("\nep_00009 epid",e9[1],"sig",np.round(e9[0],2))
print(" nearest va",best9.split("/")[-1],"sig",np.round(va_sig[best9],2),
      "dist",round(float(np.linalg.norm(nz(va_sig[best9])-nz(e9[0]))),4))
hd=torch.load("/root/idmval/idm_head_v1.pt",map_location="cpu",weights_only=False)
head=ih.IDMHead(**hd["config"]["head_kwargs"]).to(dev); head.load_state_dict(hd["state_dict"]); head.eval()
# eval-encode path for ep_00009
enc,ro,_=R.load_encoder("/root/models/flagship-30k/ckpt.pt",dev)
d9=torch.load(f"{VAL}/ep_{i9:05d}.pt",weights_only=False)
z_eval=R.encode_frames(enc,ro,d9["frames_u8"],dev,32).float()
def clipmetrics(z,poses,actions):
    Z,S,T=ih.build_windows(z.float(),poses.float(),actions.float(),k=4,stride=2)
    return ih.evaluate(head,Z,S,T,device=dev)
m_eval=clipmetrics(z_eval,d9["poses"],d9["actions"])
# pod3-latent path for its nearest va twin
dva=torch.load(best9,weights_only=False)
m_va=clipmetrics(dva["z"],dva["poses"],dva["actions"])
print(" eval-encode(0c5f) ep_00009 : speed_r2 %.3f speed_mae %.3f ade_2s %.3f"%(m_eval["r2"]["speed"],m_eval["mae"]["speed"],m_eval["ade_2s"]))
print(" pod3-latent(%s): speed_r2 %.3f speed_mae %.3f ade_2s %.3f"%(best9.split("/")[-1],m_va["r2"]["speed"],m_va["mae"]["speed"],m_va["ade_2s"]))
