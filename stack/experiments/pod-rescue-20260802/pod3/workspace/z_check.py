import sys, glob, torch
sys.path.insert(0,"/workspace/TanitAD/stack"); sys.path.insert(0,"/workspace/TanitAD/stack/scripts")
import eval_flagship_v4 as E
from tanitad.data.mixing import load_episode
CK=sys.argv[1]; dev=torch.device("cuda")
ck=torch.load(CK,map_location="cpu",weights_only=False)
world,grounding,step=E.load_v1_from_ck(ck,dev)[:3]
eps=[load_episode(f,mmap=True) for f in sorted(glob.glob("/workspace/val40cache/ep_*.pt"))[:2]]
cfg=E._eval_cfg(None); plan=E._plan(cfg); ds=E.build_val_dataset_base(eps,cfg,plan)
it=[ds[i] for i in range(4)]
fr=torch.stack([x["frames"] for x in it]).to(dev)
with torch.autocast("cuda",dtype=torch.bfloat16,enabled=True):
    sw = world.encode_window(fr)                      # [B,W,D]
    se = world.encode(fr[:,-1])                       # [B,D]  last frame alone
print("encode_window ->",tuple(sw.shape),"encode(last) ->",tuple(se.shape))
a=sw[:,-1].float(); b=se.float()
print("max|abs diff| =", float((a-b).abs().max()))
print("cosine        =", float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean()))
print("rel L2        =", float((a-b).norm()/a.norm()))
