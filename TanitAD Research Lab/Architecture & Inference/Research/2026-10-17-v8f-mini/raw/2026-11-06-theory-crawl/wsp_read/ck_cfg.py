import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
ck=torch.load("E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt",map_location="cpu",weights_only=False)
print([k for k in ck.keys() if isinstance(ck[k],dict) and any(hasattr(v,'shape') for v in ck[k].values())])
c=ck["cfg"]; print({k:c[k] for k in ("d","heads","ff","d_feat","n_tok","hist","nb","res_scale_m","depth_o","depth_s","depth_t","n_q","navonly","o_interface","code_mix","wm") if k in c})
sd=ck.get("model") or ck.get("state_dict") or ck.get("model_state") or ck.get("sd")
print([ (k,tuple(v.shape)) for k,v in sd.items() if k.startswith("prior.") or k in ("e_ego.weight","e_nav.weight","e_code.weight","adapter.1.weight","head_o.0.weight","head_o.2.weight","basis","cons_mu")])
