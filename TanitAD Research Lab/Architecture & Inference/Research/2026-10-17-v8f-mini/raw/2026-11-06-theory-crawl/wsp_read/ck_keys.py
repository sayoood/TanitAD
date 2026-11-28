import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch, json
p="E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
ck=torch.load(p,map_location="cpu",weights_only=False)
print(type(ck), list(ck.keys())[:20] if isinstance(ck,dict) else '')
for k,v in ck.items():
    if isinstance(v,dict):
        print(k, "dict", len(v), list(v.keys())[:6])
    elif hasattr(v,'shape'): print(k, v.shape)
    else: print(k, str(v)[:300])
sd=ck.get("model") or ck.get("state_dict") or ck.get("model_state") 
print(type(sd))
if sd:
    tot=0
    mods={}
    for k,v in sd.items():
        tot+=v.numel(); top=".".join(k.split(".")[:2]); mods[top]=mods.get(top,0)+v.numel()
    print(tot); 
    for m,n in sorted(mods.items(), key=lambda x:-x[1])[:40]: print(m,n)
