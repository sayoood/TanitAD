import sys, pathlib
sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack/scripts")
import torch
import refc_v3_train as t
from tanitad.refs import refc_v3 as v3, refc_agents as ra

def build(kind, **kw):
    c = v3.refc_v3_sized_config("tiny", hier=True)
    c.core.anchors.v0_conditioned = True
    a = ["--arm","hier","--out","/tmp/x","--agents",kind]
    for k,v in kw.items(): a += ["--"+k.replace("_","-"), str(v)]
    args = t.build_parser().parse_args(a)
    t._pin_refcv5_seams(c, args)
    return v3.RefCV3Model(c), c, args

print("=== WP-6 ORACLE: does the seam BUILD and RUN end to end? ===")
torch.manual_seed(0)
m, c, args = build("oracle", agent_queries=8, agent_sigma_range=0.0)
print("  agent_head :", type(m.core.agent_head).__name__)
print("  agent_embed:", type(m.core.agent_embed).__name__)
print("  layers with cross_agent:", sum(1 for l in m.core.decoder.layers if l.cross_agent is not None), "/", len(m.core.decoder.layers))
print("  agent_gate init (must be 0.0):", [float(l.agent_gate) for l in m.core.decoder.layers])

B,N = 2,8
enc = c.core.encoder
frames = torch.randn(B, c.core.window, enc.in_channels, *enc.image_hw())
gt = {"box": torch.randn(B,N,4).abs()*5, "yaw": torch.randn(B,N),
      "cls": torch.randint(0,3,(B,N)), "valid": torch.ones(B,N,dtype=torch.bool)}
gt["valid"][1,4:] = False                      # a partially-empty row
m.eval()
with torch.no_grad():
    out = m(frames, v0=torch.tensor([12.0, 3.0]), steps=2, agent_gt=gt)
print("  forward OK. agent_slots keys:", sorted(out["agent_slots"].keys())[:6], "...")
print("  traj", tuple(out["traj"].shape), "finite:", bool(torch.isfinite(out["traj"]).all()))

print()
print("=== THE EMPTY-SCENE CASE: every slot invalid (an empty road) ===")
gt2 = {k: v.clone() for k,v in gt.items()}
gt2["valid"][:] = False
with torch.no_grad():
    o2 = m(frames, v0=torch.tensor([12.0,3.0]), steps=2, agent_gt=gt2)
print("  traj finite:", bool(torch.isfinite(o2["traj"]).all()),
      "| any NaN anywhere:", bool(torch.isnan(o2["traj"]).any()))

print()
print("=== ZERO-INIT PARITY: at agent_gate=0 the planner output must be UNCHANGED ===")
torch.manual_seed(0); m0, c0, _ = build("oracle", agent_queries=8)
torch.manual_seed(0)
c1 = v3.refc_v3_sized_config("tiny", hier=True); c1.core.anchors.v0_conditioned=True
t._pin_refcv5_seams(c1, t.build_parser().parse_args(["--arm","hier","--out","/tmp/x"]))
torch.manual_seed(1234); base = v3.RefCV3Model(c1).eval()
torch.manual_seed(1234); withag = v3.RefCV3Model(c0).eval()
sb, sw = base.state_dict(), withag.state_dict()
n_shared = sum(1 for k in sb if k in sw and sb[k].shape==sw[k].shape)
for k in sb:
    if k in sw and sb[k].shape==sw[k].shape: sw[k]=sb[k].clone()
withag.load_state_dict(sw)
with torch.no_grad():
    a = base(frames, v0=torch.tensor([12.0,3.0]), steps=2)
    b = withag(frames, v0=torch.tensor([12.0,3.0]), steps=2, agent_gt=gt)
print(f"  shared tensors copied: {n_shared}")
print("  traj IDENTICAL:", bool(torch.equal(a["traj"], b["traj"])),
      "| max|d| =", float((a["traj"]-b["traj"]).abs().max()))
print("  anchor_logits IDENTICAL:", bool(torch.equal(a["anchor_logits"], b["anchor_logits"])))

print()
print("=== the gate is GATED, not DEAD: gradient must reach agent_gate ===")
m.train()
o = m(frames, v0=torch.tensor([12.0,3.0]), steps=2, agent_gt=gt)
o["traj"].square().mean().backward()
g = [float(l.agent_gate.grad.abs().max()) for l in m.core.decoder.layers]
print("  |d loss / d agent_gate| per layer:", [f"{x:.3e}" for x in g])
print("  all non-zero:", all(x>0 for x in g))
