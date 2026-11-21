"""Run ONE tree (old d014ff2 or the fix) through a fixed sequence on the rig and dump every output / grad / generator
state: TRAIN fwd+bwd #1 -> EVAL fwd x2 (forked, seed 1234) -> EVAL fwd (seed 777) -> TRAIN fwd+bwd #2. CPU only."""
import os, sys, torch
import repro_lib as L
from tanitad.refs import refcv8_conditioning as C
out_path = sys.argv[1]
emit_live = sys.argv[2] == "1"
print("tanitad from", __import__("tanitad").__file__, flush=True)
cfg, m = L.build("r8")
dec = m.core.decoder
dec.r8_emit_live = emit_live
bt = L.R.batch(cfg, b=3)

def flat_tensors(o, p="out"):
    r = {}
    if torch.is_tensor(o):
        r[p] = o.detach().clone()
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            r.update(flat_tensors(o[k], f"{p}.{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r.update(flat_tensors(v, f"{p}[{i}]"))
    return r

def train_step(tag):
    m.zero_grad(set_to_none=True)
    out = L.fwd_forked(cfg, m, bt, seed=2, train=True)
    ft = flat_tensors(out)
    loss = sum(v.float().mean() for k, v in _live(out).items())
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return {"out": ft, "loss": loss.detach(), "grads": grads, "gen_state": dec.r8_gen.get("cpu").get_state().clone()}

def _live(out, p="out"):
    r = {}
    if torch.is_tensor(out) and out.requires_grad and out.is_floating_point():
        r[p] = out
    elif isinstance(out, dict):
        for k in out:
            r.update(_live(out[k], f"{p}.{k}"))
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            r.update(_live(v, f"{p}[{i}]"))
    return r

res = {"emit_live": emit_live, "n_base": int(dec.anchors.shape[0])}
res["train1"] = train_step("t1")
res["eval_a"] = {"out": flat_tensors(L.fwd_forked(cfg, m, bt, seed=1234)), "gen_state": dec.r8_gen.get("cpu").get_state().clone()}
res["eval_b"] = {"out": flat_tensors(L.fwd_forked(cfg, m, bt, seed=1234)), "gen_state": dec.r8_gen.get("cpu").get_state().clone()}
res["eval_c_seed777"] = {"out": flat_tensors(L.fwd_forked(cfg, m, bt, seed=1234 + 0 if False else 777))}
res["train2"] = train_step("t2")
res["global_state_end"] = torch.get_rng_state().clone()
torch.save(res, out_path)
print("saved", out_path, "n_grads", len(res["train1"]["grads"]), "n_train_outputs", len(res["train1"]["out"]), flush=True)
