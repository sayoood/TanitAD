"""P2a - CALIBRATE the at-init detector on real artifacts before writing it.

Two-sided, on real files, not synthetic assumptions:

  NEGATIVE side (must read AT_INIT): the 9 banked v7-tiny step_readout_op.
  POSITIVE side (must read TRAINED): a flagship v4 `grounding['step.op.*']`
      -- a structurally IDENTICAL StepDisplacementReadout that really trained.
      The smoke ckpts carry step=1, i.e. ONE optimizer step, which makes them
      the strictest available positive control.

Then a sensitivity sweep: how big a relative perturbation of a banked readout
does the detector need to flip AT_INIT -> TRAINED?
"""
import glob
import json
import math
import os

import torch

CKROOT = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p2_calibrate.json"

MIN_N = 64          # below this a uniform draw's order statistics are useless


def sig(w):
    """(max_ratio, std_ratio, n) against nn.Linear's own U(-b, b) init."""
    w = w.float()
    fan_in = w.shape[-1] if w.dim() >= 2 else None
    return w, fan_in


def linear_stats(weight, tensor, fan_in):
    b = 1.0 / math.sqrt(fan_in)
    t = tensor.float()
    n = t.numel()
    return {"n": int(n), "bound": b,
            "max_ratio": float(t.abs().max()) / b,
            "std_ratio": float(t.std()) / (b / math.sqrt(3.0)),
            "max_margin": 1.0 - 20.0 / n if n else None,
            "std_margin": 8.0 / math.sqrt(2.0 * n) if n else None}


def readout_sig(sd, prefix):
    """sd: state dict; prefix e.g. '' or 'step.op.'  -> per-tensor stats."""
    out = {}
    w1 = sd.get(prefix + "net.1.weight")
    w3 = sd.get(prefix + "net.3.weight")
    if w1 is None:
        return out
    fan1 = w1.shape[1]
    fan3 = w3.shape[1] if w3 is not None else None
    for k, fan in ((prefix + "net.1.weight", fan1), (prefix + "net.1.bias", fan1),
                   (prefix + "net.3.weight", fan3), (prefix + "net.3.bias", fan3)):
        if k in sd and fan:
            out[k] = linear_stats(None, sd[k], fan)
    for k in (prefix + "net.0.weight", prefix + "net.0.bias"):
        if k in sd:
            t = sd[k].float()
            want = torch.ones_like(t) if k.endswith("weight") else torch.zeros_like(t)
            out[k] = {"n": int(t.numel()), "layernorm_exact_init":
                      bool(torch.equal(t, want))}
    return out


RES = {"MIN_N": MIN_N, "negative_side": {}, "positive_side": {}}

# ---- negative side: the 9 banked v7-tiny readouts ------------------------- #
for d in sorted(os.listdir(CKROOT)):
    p = os.path.join(CKROOT, d, "ckpt.pt")
    if not (d.startswith("v7tiny_") and os.path.isfile(p)):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck["stack"] if "stack" in ck else ck
    ro = {k[len("step_readout_op."):]: v for k, v in sd.items()
          if k.startswith("step_readout_op.")}
    RES["negative_side"][d] = readout_sig(ro, "")
    del ck, sd

# ---- positive side: real flagship grounding step readouts ----------------- #
cands = sorted(glob.glob(r"C:\Users\Admin\AppData\Local\Temp\v4smoke_*\ckpt.pt"))
kept = 0
for p in cands:
    if kept >= 6:
        break
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        RES["positive_side"][p] = {"status": f"UNREADABLE {e}"}
        continue
    g = ck.get("grounding")
    if not isinstance(g, dict) or "step.op.net.1.weight" not in g:
        del ck
        continue
    RES["positive_side"][os.path.basename(os.path.dirname(p))] = {
        "trained_steps": ck.get("step"),
        "stats": readout_sig(g, "step.op.")}
    kept += 1
    del ck

# ---- sensitivity: how small a move does the detector still see? ---------- #
p = os.path.join(CKROOT, "v7tiny_emao14_30k", "ckpt.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck["stack"]
ro = {k[len("step_readout_op."):]: v.clone()
      for k, v in sd.items() if k.startswith("step_readout_op.")}
del ck, sd

sweep = {}
g = torch.Generator().manual_seed(0)
for rel in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
    mut = {k: v.clone() for k, v in ro.items()}
    for k in ("net.1.weight", "net.3.weight"):
        t = mut[k].float()
        mut[k] = t + torch.randn(t.shape, generator=g) * t.std() * rel
    s = readout_sig(mut, "")
    sweep[f"rel_{rel:g}"] = {
        k: {kk: round(vv, 8) if isinstance(vv, float) else vv
            for kk, vv in v.items()}
        for k, v in s.items() if k.endswith(("net.1.weight", "net.3.weight"))}
RES["perturbation_sweep"] = sweep

# ---- what one real AdamW step at the arm's own hyper-params does ---------- #
from torch import nn                                       # noqa: E402
lin = nn.Linear(4096, 512)
lin.weight.data.copy_(ro["net.1.weight"].float())
lin.bias.data.copy_(ro["net.1.bias"].float())
opt = torch.optim.AdamW(lin.parameters(), lr=1e-4, weight_decay=0.05)
xs = torch.randn(64, 4096, generator=torch.Generator().manual_seed(3))
adam = {}
for step in range(1, 6):
    opt.zero_grad()
    lin(xs).pow(2).mean().backward()
    opt.step()
    adam[f"after_{step}_adamw_steps"] = linear_stats(None, lin.weight.data, 4096)
RES["one_real_adamw_step"] = adam

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1)

print("=== NEGATIVE SIDE (banked v7-tiny; must read AT_INIT) ===")
for a, s in RES["negative_side"].items():
    w = s.get("net.1.weight", {})
    w3 = s.get("net.3.weight", {})
    print(f"  {a:28s} net.1.w max_ratio={w.get('max_ratio'):.7f} "
          f"std_ratio={w.get('std_ratio'):.5f} | net.3.w max_ratio="
          f"{w3.get('max_ratio'):.7f} std_ratio={w3.get('std_ratio'):.5f} | "
          f"LN exact={s.get('net.0.weight',{}).get('layernorm_exact_init')}"
          f"/{s.get('net.0.bias',{}).get('layernorm_exact_init')}")
print()
print("=== POSITIVE SIDE (real flagship step.op; must read TRAINED) ===")
for a, s in RES["positive_side"].items():
    st = s.get("stats", {})
    w = st.get("step.op.net.1.weight", {})
    if not w:
        print(f"  {a}: {s.get('status')}")
        continue
    print(f"  {a:28s} steps={s['trained_steps']} net.1.w max_ratio="
          f"{w['max_ratio']:.7f} (>1.0 == moved past the init bound) "
          f"std_ratio={w['std_ratio']:.5f}")
print()
print("=== PERTURBATION SWEEP (banked emao14_30k net.1.weight) ===")
for k, v in sweep.items():
    w = v["net.1.weight"]
    print(f"  {k:12s} max_ratio={w['max_ratio']:.8f} std_ratio={w['std_ratio']:.6f}"
          f"  (margins: max>={w['max_margin']:.8f}, std +-{w['std_margin']:.6f})")
print()
print("=== ONE REAL AdamW STEP (lr 1e-4, wd 0.05 -- the arm's own) ===")
for k, v in RES["one_real_adamw_step"].items():
    print(f"  {k:26s} max_ratio={v['max_ratio']:.8f} std_ratio={v['std_ratio']:.6f}")
print("[wrote]", OUT)
