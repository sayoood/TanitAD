"""Audit part 2: v7-label gradient census, route mask mutation, echo test.

ASCII only. CPU only.
"""
import importlib.util as iu
import sys

import torch

STACK = sys.argv[1]
sys.path.insert(0, STACK)
spec = iu.spec_from_file_location("t", f"{STACK}/scripts/refc_v3_train.py")
T = iu.module_from_spec(spec)
_a, sys.argv = sys.argv, ["t"]
try:
    spec.loader.exec_module(T)
except SystemExit:
    pass
sys.argv = _a

v3 = T.v3
v7l = T.v7l


def build(vocab):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = vocab
    m = v3.RefCV3Model(cfg)
    m.train()
    return cfg, m


def batch_for(cfg, n=2):
    eps = T._synth_episodes(n, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    return torch.utils.data.default_collate([ds[i] for i in range(n)])


def census(m):
    out = {}
    for name, p in m.named_parameters():
        mod = name.rsplit(".", 1)[0]
        r = out.setdefault(mod, [0, 0])
        r[0] += 1
        if p.grad is None:
            r[1] += 1
    return out


def dead(m):
    return sorted(k for k, (n, z) in census(m).items() if z == n)


print("=" * 78)
print("A) v7.0 ARM WITH v7-SHAPED LABELS (the trainer's own preflight recipe)")
print("=" * 78)
cfg, m = build("v7.0")
b = batch_for(cfg)
n_lat = len(v7l.HEADS["tac_lat"])
n_lon = len(v7l.HEADS["tac_lon"])
print("  v7 HEADS declared:", {k: len(v) for k, v in v7l.HEADS.items()})
b["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
b["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID], dtype=torch.long)
res = T.compute_losses_v3(m, b, "cpu")
loss = res["loss"]
m.zero_grad(set_to_none=True)
loss.backward()
print("  loss = %.5f" % float(loss))
print("  NO-GRADIENT modules:", dead(m))
print("  CONTROL lat_head_tac grad is None:", m.lat_head_tac.weight.grad is None)
print("  tac_label_v7 flag in extra:",
      float(res.get("tac_label_v7", float("nan"))))
print("  tac_label_rows (in-band rows of 2):",
      float(res.get("tac_label_rows", float("nan"))))

print()
print("=" * 78)
print("B) ROUTE HEAD - is the no-gradient a MASK artifact or a dead head?")
print("=" * 78)
cfg, m = build("kin3")
b = batch_for(cfg)
print("  route_valid in synth batch:", b["route_valid"].tolist())
print("  route_target in synth batch:", b["route_target"].tolist())
for label, mut in (("as-built", None), ("route_valid FORCED True", True)):
    cfg2, m2 = build("kin3")
    b2 = batch_for(cfg2)
    if mut:
        b2["route_valid"] = torch.ones_like(b2["route_valid"], dtype=torch.bool)
        b2["route_target"] = torch.zeros_like(b2["route_target"])
    r = T.compute_losses_v3(m2, b2, "cpu")
    m2.zero_grad(set_to_none=True)
    r["loss"].backward()
    g = m2.core.route_head.weight.grad
    print("  %-26s route loss=%.5f  route_head grad is None: %s"
          % (label, float(r.get("route", float("nan"))), g is None))

print()
print("=" * 78)
print("C) ECHO TEST - is route_target a function of nav_cmd?")
print("   Both are minted from the SAME ego future by refb_labels.")
print("=" * 78)
import numpy as np
from tanitad.refs import refb
try:
    print("  NAV_COMMANDS  :", list(refb.NAV_COMMANDS))
except Exception as e:
    print("  NAV_COMMANDS unavailable:", e)
try:
    print("  ROUTE_CLASSES :", list(refb.ROUTE_CLASSES))
except Exception as e:
    print("  ROUTE_CLASSES unavailable:", e)

cfg, _ = build("kin3")
eps = T._synth_episodes(24, cfg.core, seed=3)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                 channels=cfg.core.encoder.in_channels)
navs, routes, both = [], [], 0
for i in range(len(ds)):
    it = ds[i]
    if not bool(it["route_valid"]) or not bool(it["nav_valid"]):
        continue
    navs.append(int(it["nav_cmd"]))
    routes.append(int(it["route_target"]))
    both += 1
print("  windows with BOTH nav_valid and route_valid: %d of %d"
      % (both, len(ds)))
if both:
    a = np.array(navs)
    r = np.array(routes)
    print("  nav distribution  :", {int(k): int(v) for k, v in
                                    zip(*np.unique(a, return_counts=True))})
    print("  route distribution:", {int(k): int(v) for k, v in
                                    zip(*np.unique(r, return_counts=True))})
    print("  EXACT AGREEMENT nav==route: %d/%d = %.4f"
          % (int((a == r).sum()), both, float((a == r).mean())))
