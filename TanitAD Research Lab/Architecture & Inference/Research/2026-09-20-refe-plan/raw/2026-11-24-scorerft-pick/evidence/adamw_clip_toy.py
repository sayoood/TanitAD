#!/usr/bin/env python3
"""Toy check, CPU, torch as installed: does a GLOBAL gradient rescale change an AdamW trajectory?

Card PREREG_SCORERFT_PICK.md section 3.2 (CARD_QUESTIONS.md Q-P17). Same optimiser call as refe/train.py:1541 (AdamW(params, lr, weight_decay), eps default),
lr 5e-5 and wd 0.01 as C's argv. Four arms on ONE fixed gradient stream (the toy loss's gradients, recomputed at each arm's own weights):
  A  raw gradients
  B  every gradient x 0.3 (a CONSTANT clip coefficient: the case "the clip binds on every step by the same factor")
  V  gradient x c_t, c_t alternating 1.0 / 0.3 (a VARYING coefficient: "binds on some steps only")
  K  torch.nn.utils.clip_grad_norm_(..., 1.0) on gradients pre-scaled x 50 so it binds on every step (the real function, varying N_t)
Prints the max |theta_X - theta_A| after 200 steps, relative to the max |theta_A - theta_0| (how far A moved).
Expected (analytic): B ~ 0 up to eps (1e-8) and rounding; V clearly non-zero; K small but non-zero (its coefficient 1/N_t varies with N_t).
"""
import torch

torch.manual_seed(20261124)
D_IN, D_H, N, STEPS = 16, 32, 256, 200
X = torch.randn(N, D_IN, dtype=torch.float64)
Y = torch.randn(N, 1, dtype=torch.float64)


def make():
    torch.manual_seed(7)
    return torch.nn.Sequential(torch.nn.Linear(D_IN, D_H), torch.nn.Tanh(), torch.nn.Linear(D_H, 1)).double()


def run(mode):
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01)
    th0 = torch.cat([p.detach().flatten().clone() for p in m.parameters()])
    coefs = []
    for t in range(STEPS):
        g = torch.Generator().manual_seed(1000 + t)
        idx = torch.randperm(N, generator=g)[:32]
        loss = ((m(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mode == "B":
            for p in m.parameters():
                p.grad.mul_(0.3)
        elif mode == "V":
            c = 1.0 if t % 2 == 0 else 0.3
            for p in m.parameters():
                p.grad.mul_(c)
        elif mode == "K":
            for p in m.parameters():
                p.grad.mul_(50.0)
            n = float(torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0))
            coefs.append(min(1.0, 1.0 / (n + 1e-6)))
        opt.step()
    th = torch.cat([p.detach().flatten() for p in m.parameters()])
    return th0, th, coefs


th0, thA, _ = run("A")
moved = float((thA - th0).abs().max())
print(f"torch {torch.__version__}  float64  steps {STEPS}  AdamW lr 5e-5 wd 0.01 eps default")
print(f"A moved max |dtheta| = {moved:.6e}")
for mode in ("B", "V", "K"):
    _, th, coefs = run(mode)
    d = float((th - thA).abs().max())
    extra = ""
    if coefs:
        extra = f"   clip coefficient over steps: min {min(coefs):.4g} max {max(coefs):.4g} (bound on {sum(c < 1.0 for c in coefs)}/{len(coefs)} steps)"
    print(f"{mode}: max |theta_{mode} - theta_A| = {d:.6e}   relative to A's movement = {d / moved:.3e}{extra}")
