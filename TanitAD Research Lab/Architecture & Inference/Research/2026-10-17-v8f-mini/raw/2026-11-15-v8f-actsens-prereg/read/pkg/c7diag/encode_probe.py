"""C7 diagnosis, op-level probe: forward hooks on every submodule of R35_s<k>'s HJEPA stack while st.encode(TRAIN Z1, A_sc) runs
(the call behind the level-2/3 priors, hwm_data.fit_priors). Records, in call order, each module's first input and output tensor.
Run it under two thread counts and compare: the first module whose INPUT is bit-equal but whose OUTPUT is not is the thread-sensitive op.
usage: encode_probe.py <chassis> <out_npz>"""
import json, os, sys
sys.dont_write_bytecode = True
H = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"
sys.path.insert(0, H)
import numpy as np
import torch
import hwm_data as D

k, out = int(sys.argv[1]), sys.argv[2]
tag = f"R35_s{k}"
m, cfg, ck = D.load_model(f"C:/Users/Admin/t2v8f_pulls/ladder_{tag}/ckpt_last.pt")
wm, st = m.wm, m.wm.stack
Z1, A = D.trainset_from_latents_pt(f"F:/TanitAD-scratch/hwm_plan/latents_train_{tag}.pt", tag)
A_sc = wm.scale_actions(A)[:, :-1]
rec, names = {}, []


def first_t(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            t = first_t(y)
            if t is not None:
                return t
    if isinstance(x, dict):
        for y in x.values():
            t = first_t(y)
            if t is not None:
                return t
    if hasattr(x, "_fields"):
        for y in x:
            t = first_t(y)
            if t is not None:
                return t
    return None


def hook(name):
    def f(mod, inp, outp):
        i = len(names)
        ti, to = first_t(inp), first_t(outp)
        names.append(f"{i:04d}|{name}|{type(mod).__name__}")
        if ti is not None:
            rec[f"in{i:04d}"] = ti.detach().float().numpy().copy()
        if to is not None:
            rec[f"out{i:04d}"] = to.detach().float().numpy().copy()
    return f


for n, mod in st.named_modules():
    if n:
        mod.register_forward_hook(hook(n))
with torch.no_grad():
    enc = st.encode(Z1, A_sc)
rec["act1"], rec["act2"] = enc.act[1].numpy(), enc.act[2].numpy()
np.savez(out, names=np.array(names), **rec)
print(json.dumps(dict(threads=torch.get_num_threads(), n_calls=len(names))))
