"""The A/B that confirmed P-RC21's collapse mechanism (2026-08-29).

Toy: mean parameter m, reward = -(sample-3)^2, multiplicative scale |m|*sigma.
DEGENERATE logp = -0.5*eps^2 - log(scale) with eps the raw draw: the
(sample-mean) dependence cancels symbolically, so d(logp)/d(mean) flows only
through log|m| — the update can shrink/inflate |m| by advantage sign but can
never move toward good samples.  MEASURED: m 0.50 -> +0.500 after 400 steps.
CORRECT  logp from eps_eff = (sample.detach()-m)/scale: m 0.50 -> +2.750.

On the real pilot the degenerate form inflated offsets until the fan left the
road: R2 11.113% -> 0.000% BECAUSE R3 exploded 1.969 m -> 347.227 m.
The differentiability test passed throughout — a nonzero gradient is not a
correct gradient. Pinned by the DIRECTIONAL test in
tests/test_rl_refcv3_integration.py::test_score_function_gradient_points_toward_good_samples.
"""
import torch


def run(fixed, steps=400, lr=0.05, m0=0.5, sigma=0.3, G=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.tensor([m0], requires_grad=True)
    opt = torch.optim.SGD([m], lr=lr)
    for _ in range(steps):
        scale = (m.abs() * sigma).clamp_min(1e-3)
        eps = torch.randn(G, generator=g)
        sample = m + scale * eps
        if fixed:
            eps_eff = (sample.detach() - m) / scale
            logp = -0.5 * eps_eff.pow(2) - torch.log(scale)
        else:
            logp = -0.5 * eps.pow(2) - torch.log(scale)
        reward = -(sample.detach() - 3.0).pow(2)
        adv = reward - reward.mean()
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(m.detach())


if __name__ == "__main__":
    for name, fixed in (("DEGENERATE", False), ("FIXED", True)):
        print(f"{name:10s} m: 0.50 -> {run(fixed):+.3f}  (target +3.0)")
