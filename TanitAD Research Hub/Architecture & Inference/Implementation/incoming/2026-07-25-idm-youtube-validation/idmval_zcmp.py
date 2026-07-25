"""On the clips SHARED by both val builds (identical poses), compare
eval-encode z vs pod3-latent z and per-clip head ADE. Isolates encoder-env
drift from the clip-set effect."""
import sys, glob, torch, numpy as np
sys.path.insert(0,"/root/v4eval/stack"); sys.path.insert(0,"/root/v4eval/stack/scripts")
import idm_head as ih, run_idm_proof as R
dev="cuda"; VAL="/root/valdata/physicalai-val-0c5f7dac3b11"

def sig(p):
    v=p[:,3].numpy()
    pl=float(np.linalg.norm(np.diff(p[:,:2].numpy(),axis=0),axis=1).sum())
    return np.array([p.shape[0],float(v.mean()),float(v.max()),float(v.min()),pl])
va=sorted(glob.glob("/root/idmval/va40/*.pt"))
va_sig={f:sig(torch.load(f,weights_only=False)["poses"].float()) for f in va}
ev={i:torch.load(f"{VAL}/ep_{i:05d}.pt",weights_only=False) for i in range(40)}
ev_sig={i:sig(ev[i]["poses"].float()) for i in range(40)}
alls=np.stack(list(ev_sig.values())+list(va_sig.values())); mu,sd=alls.mean(0),alls.std(0)+1e-9
nz=lambda x:(x-mu)/sd
pairs=[]
for i in range(40):
    best=min(va,key=lambda f:np.linalg.norm(nz(va_sig[f])-nz(ev_sig[i])))
    if np.linalg.norm(nz(va_sig[best])-nz(ev_sig[i]))<1e-3: pairs.append((i,best))
print("exact shared clips:",len(pairs))
hd=torch.load("/root/idmval/idm_head_v1.pt",map_location="cpu",weights_only=False)
head=ih.IDMHead(**hd["config"]["head_kwargs"]).to(dev); head.load_state_dict(hd["state_dict"]); head.eval()
enc,ro,_=R.load_encoder("/root/models/flagship-30k/ckpt.pt",dev)
def cm(z,p,a):
    Z,S,T=ih.build_windows(z.float(),p.float(),a.float(),k=4,stride=2); return ih.evaluate(head,Z,S,T,device=dev)
EA=[];EG=[];TP=[];TG=[];PA=[];PG=[];TPp=[];TGp=[]
cos=[]
for i,vf in pairs:
    d=ev[i]; z_eval=R.encode_frames(enc,ro,d["frames_u8"],dev,32).float()
    dva=torch.load(vf,weights_only=False); z_pod=dva["z"].float()
    n=min(z_eval.shape[0],z_pod.shape[0])
    c=torch.nn.functional.cosine_similarity(z_eval[:n],z_pod[:n],dim=1).mean().item(); cos.append(c)
    me=cm(z_eval,d["poses"],d["actions"]); mv=cm(z_pod,dva["poses"],dva["actions"])
    print(f"ep_{i:05d}<->{vf.split('/')[-1]:14s} zcos {c:.4f}  ade_eval {me['ade_2s']:.3f}  ade_pod3 {mv['ade_2s']:.3f}  spmae_eval {me['mae']['speed']:.2f} spmae_pod3 {mv['mae']['speed']:.2f}")
print("mean z cosine(eval,pod3) over shared clips: %.4f"%np.mean(cos))
