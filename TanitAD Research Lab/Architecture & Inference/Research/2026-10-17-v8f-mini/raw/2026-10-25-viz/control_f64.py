"""control_f64.py -- is the ~1e-5 float32 gap between our explicit weights and torch's need_weights weights round-off?
Capture the real (query, key, padding) of every cross-attention call, recompute the weights in FLOAT64 (reference), and compare each float32 path to it."""
import sys; sys.path.insert(0, "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/out")
from attn_extract import *
import json
res={}
for seed in ("M2b_s0","M2b_s1"):
    model,cfg=load_model(seed)
    ew=EvalWorld(cfg, cache=f"{WORK}/cache_viz")
    idx=np.concatenate([ew.clip_windows(c) for c in ("79e30ac2-c619-4d61-98a3-1007037b4d05","ed87040c-cf3c-4fe4-9a95-50c1114f0d38")])
    b=ew.batch(idx)
    cap=[]
    mhas=[(lv,li,l.multihead_attn) for lv in LEVELS for li,l in enumerate(getattr(model,READER[lv]).dec.layers)]
    origs=[]
    for lv,li,mha in mhas:
        o=mha.forward
        def w(*a,_o=o,_m=mha,_k=(lv,li),**k):
            out=_o(*a,**k); cap.append((_k,_m,a[0].detach().clone(),a[1].detach().clone(),k.get("key_padding_mask"))); return out
        mha.forward=w; origs.append((mha,o))
    with torch.no_grad(): model.forward_infer(b)
    for mha,o in origs: mha.forward=o
    d=0.0; e32=0.0; etn=0.0
    for key,mha,q,kk,pm in cap:
        ref=explicit_attn_weights_f64=None
        m64=mha.double() if False else None
        # f64 reference
        Wt,bt=mha.in_proj_weight.double(),mha.in_proj_bias.double(); dd=mha.embed_dim; H=mha.num_heads; dh=dd//H
        qq=torch.nn.functional.linear(q.double(),Wt[:dd],bt[:dd]); kx=torch.nn.functional.linear(kk.double(),Wt[dd:2*dd],bt[dd:2*dd])
        B,nq,Sx=qq.shape[0],qq.shape[1],kx.shape[1]
        qq=qq.reshape(B,nq,H,dh).transpose(1,2); kx=kx.reshape(B,Sx,H,dh).transpose(1,2)
        sc=(qq@kx.transpose(-1,-2))/(dh**0.5)
        if pm is not None: sc=sc.masked_fill(pm[:,None,None,:],float("-inf"))
        ref=torch.softmax(sc,-1)
        ex32=explicit_attn_weights(mha,q,kk,pm).double()
        with torch.no_grad(): _,tw=mha(q,kk,kk,key_padding_mask=pm,need_weights=True,average_attn_weights=False)
        e32=max(e32,float((ex32-ref).abs().max())); etn=max(etn,float((tw.double()-ref).abs().max()))
        d=max(d,float(sc[...,:].masked_fill(torch.isinf(sc),0).abs().max()))
    res[seed]={"n_windows":len(idx),"explicit_f32_vs_f64ref_maxabs":e32,"torch_need_weights_f32_vs_f64ref_maxabs":etn,"max_abs_logit":d}
print(json.dumps(res,indent=1)); json.dump(res,open("C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/out/control_f64.json","w"),indent=1)
