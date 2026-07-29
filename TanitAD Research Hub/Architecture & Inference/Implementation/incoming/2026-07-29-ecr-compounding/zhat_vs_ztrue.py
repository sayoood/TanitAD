Pseudo-terminal will not be allocated because stdin is not a terminal.
Welcome to Ubuntu 24.04.3 LTS (GNU/Linux 6.8.0-65-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro


"""Is z_hat even TRYING to be z_true? If not, teacher forcing is meaningless here."""
import sys, glob, torch
sys.path.insert(0,"/workspace/TanitAD/stack"); sys.path.insert(0,"/workspace/TanitAD/stack/scripts")
import eval_flagship_v4 as E
from tanitad.data.mixing import load_episode
from tanitad.models.flagship_v15 import SPEED_SCALE
import torch.nn.functional as F
CK=sys.argv[1]; dev=torch.device("cuda")
ck=torch.load(CK,map_location="cpu",weights_only=False)
world,grounding,step=E.load_v1_from_ck(ck,dev)[:3]
eps=[load_episode(f,mmap=True) for f in sorted(glob.glob("/workspace/val40cache/ep_*.pt"))[:2]]
cfg=E._eval_cfg(None); plan=E._plan(cfg); ds=E.build_val_dataset_base(eps,cfg,plan)
pos={(e,t):i for i,(e,t) in enumerate(ds.index)}
sel=[i for i,(e,t) in enumerate(ds.index) if t%8==0][:8]
items=[ds[i] for i in sel]; ets=[ds.index[i] for i in sel]
fr=torch.stack([x["frames"] for x in items]).to(dev)
aw2=torch.stack([x["actions"] for x in items]).to(dev).float()
pl=torch.stack([x["pose_last"] for x in items]).to(dev).float()
vch=(pl[:,3]/SPEED_SCALE)[:,None,None]
aw=torch.cat([aw2,vch.expand(-1,aw2.shape[1],-1)],-1)
nxt=[]
for (e,t) in ets:
    k=pos.get((e,t+1)); nxt.append(ds[k]["frames"][-1])
nxt=torch.stack(nxt).to(dev)
with torch.autocast("cuda",dtype=torch.bfloat16,enabled=True):
    st=world.encode_window(fr)
    zh=world.predictor(st,aw)[1]          # 1-step prediction
    zt=world.encode(nxt)                  # TRUE next latent
    zl=st[:,-1]                           # last CONTEXT latent (true)
a,b,c=zh.float(),zt.float(),zl.float()
print("cos(z_hat, z_true_next) =", round(float(F.cosine_similarity(a,b,dim=-1).mean()),5))
print("cos(z_hat, z_last_ctx)  =", round(float(F.cosine_similarity(a,c,dim=-1).mean()),5))
print("cos(z_true, z_last_ctx) =", round(float(F.cosine_similarity(b,c,dim=-1).mean()),5))
print("||z_hat||=0.000 ||z_true||=0.000 ||z_ctx||=0.000"%(a.norm(dim=-1).mean(),b.norm(dim=-1).mean(),c.norm(dim=-1).mean()))
print("rel L2 (z_hat vs z_true) =", round(float((a-b).norm()/b.norm()),5))
