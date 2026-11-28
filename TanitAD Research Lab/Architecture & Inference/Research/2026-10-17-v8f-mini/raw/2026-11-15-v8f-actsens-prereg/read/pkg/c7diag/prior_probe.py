"""C7 diagnosis, stage probe: run run_hwm.prepare()'s prior fit for R35_s<k> under the caller's environment and dump every
intermediate the HWM path reads but the flat path does not need: enc.act[1], enc.act[2] (st.encode of the TRAIN latents)
and the level 1/2/3 action priors as set on the stack (mean, std, lo, hi). Also the level-1 actions A_sc.
usage: prior_probe.py <chassis> <out_npz>     (env sets threads / code root, as one_unit.sh)"""
import json, os, sys
sys.dont_write_bytecode = True
H = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"
sys.path.insert(0, H)
import numpy as np
import torch
import hwm_data as D
import plantest_lib as P
import tanitad

k, out = int(sys.argv[1]), sys.argv[2]
tag = f"R35_s{k}"
m, cfg, ck = D.load_model(f"C:/Users/Admin/t2v8f_pulls/ladder_{tag}/ckpt_last.pt")
wm, st = m.wm, m.wm.stack
Z1, A = D.trainset_from_latents_pt(f"F:/TanitAD-scratch/hwm_plan/latents_train_{tag}.pt", tag)
A_sc = wm.scale_actions(A)[:, :-1]
with torch.no_grad():
    enc = st.encode(Z1, A_sc)
d = dict(A_sc=A_sc.detach().numpy(), act1=enc.act[1].detach().numpy(), act2=enc.act[2].detach().numpy())
D.fit_priors(st, wm, Z1, A)
for lv in (1, 2, 3):
    for nm in ("mean", "std", "lo", "hi"):
        found = None
        for cand in (f"prior_{nm}_{lv}", f"_prior_{nm}_{lv}", f"act_{nm}_{lv}"):
            if hasattr(st, cand):
                found = getattr(st, cand)
        if found is None:
            pr = getattr(st, "_action_prior", None) or getattr(st, "action_prior", None)
            if isinstance(pr, dict) and lv in pr:
                found = pr[lv].get(nm) if isinstance(pr[lv], dict) else getattr(pr[lv], nm, None)
        if found is not None:
            d[f"L{lv}_{nm}"] = torch.as_tensor(found).detach().numpy()
for n_, b_ in st.named_buffers():
    if "prior" in n_.lower():
        d["buf_" + n_] = b_.detach().numpy()
np.savez(out, **d)
print(json.dumps(dict(torch_threads=torch.get_num_threads(), tanitad=os.path.abspath(tanitad.__file__), keys=sorted(d), shapes={kk: list(v.shape) for kk, v in d.items()})), flush=True)
