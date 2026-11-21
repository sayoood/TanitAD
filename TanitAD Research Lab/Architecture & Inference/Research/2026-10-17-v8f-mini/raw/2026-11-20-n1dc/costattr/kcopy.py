import torch, json, sys
B="F:/TanitAD-scratch/n1dc"
out={}
for s in (0,1):
    u=torch.load(f"{B}/costattr/ckpts/N1DC_s{s}_zuse/ckpt_last.pt",map_location="cpu",weights_only=False)
    c=torch.load(f"{B}/read/eval141/ckpts/N1DC_s{s}/ckpt_last.pt",map_location="cpu",weights_only=False)
    def flat(d,p=""):
        r={}
        if isinstance(d,dict):
            for k,v in d.items(): r.update(flat(v,p+str(k)+"."))
        elif torch.is_tensor(d): r[p]=d
        return r
    fu,fc=flat(u),flat(c)
    zk=[k for k in fu if k.endswith("o_code.weight.")]
    eq=[k for k in fu if k in fc and torch.equal(fu[k],fc[k])]
    neq=[k for k in fu if k not in fc or not torch.equal(fu[k],fc[k])]
    nm=sum(1 for k in fu if k.startswith("model.")); out[s]={"n_model_tensors":nm,"n_u":len(fu),"n_c":len(fc),"keys_same":sorted(fu)==sorted(fc),"zero_keys":zk,
      "zero_all":all(bool((fu[k]==0).all()) for k in zk),"n_equal":len(eq),"neq":neq,
      "neq_nonzero_in_c":[bool((fc[k]!=0).any()) for k in neq if k in fc]}
    print(s,out[s])
ok=all(o["keys_same"] and o["zero_all"] and len(o["zero_keys"])==1 and o["neq"]==o["zero_keys"] for o in out.values())
out["pass"]=ok; print("PASS",ok)
json.dump(out,open(B+"/costattr/K-COPY.json","w"),indent=1)
