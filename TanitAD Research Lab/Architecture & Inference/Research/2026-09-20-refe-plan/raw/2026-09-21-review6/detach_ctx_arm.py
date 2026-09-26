"""validate_model.py ASSERTS `backbone_clean` (REFe's extra visual_ctx detach) but never MUTATES it.
Is that arm live? Flip cfg.detach_scorer_context and require the assertion to go RED.
Expectation is a LITERAL: clean run -> grad EXACTLY 0.0; mutated run -> grad > 0."""
import sys, torch
sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
from model import REFe, REFeConfig
def lora_grad(flag):
    torch.manual_seed(0)
    cfg = REFeConfig.for_backbone("vits16"); cfg.detach_scorer_context = flag
    m = REFe(cfg)
    img = torch.randn(1, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)
    ego = torch.randn(1, cfg.ego_dim); goal = torch.randn(1, 2*cfg.n_goal_points)
    _t, s = m(img, ego, goal); s.sum().backward()
    g = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
            if n.startswith("backbone") and p.requires_grad and p.grad is not None)
    sp = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
             if n.startswith("scene_proj") and p.grad is not None)
    return g, sp
g1, s1 = lora_grad(True)
g2, s2 = lora_grad(False)
print(f"  detach_scorer_context=True  (shipped): backbone LoRA grad {g1:.6f}  scene_proj grad {s1:.6f}")
print(f"  detach_scorer_context=False (mutated): backbone LoRA grad {g2:.6f}  scene_proj grad {s2:.6f}")
print(f"  validate_model's assertion `backbone_clean` would read: clean={g1==0.0}  mutated={g2==0.0}")
print("  -> " + ("ARM IS LIVE: it goes RED under its own declared departure being removed"
                 if (g1 == 0.0 and g2 > 0.0) else "** ARM IS INERT **"))
print("  -> but validate_model.py never performs this flip; it is the one asserting arm in the")
print("     file with NO deliberate-regression control, and the control is these four lines.")
