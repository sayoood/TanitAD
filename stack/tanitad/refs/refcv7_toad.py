"""refcv7 — TOAD test-time trajectory search (arXiv 2606.07170, valeo.ai, lib-banked).

TOAD treats the trained scorer as a trajectory-level REWARD and runs a Cross-Entropy Method
at inference, warm-started from the planner's pick and confined to a trust region around it.
MEASURED by its authors: DrivoR 54.6 -> 56.3 EPDMS on NAVSIM-v2 navhard, +1.9 ms at
K = 5, M = 64 when the scorer shares the planner's backbone. It works ONLY with a scorer that
generalises off its training proposals (a vocabulary-fit scorer as the reward dropped iPad
34.7 -> 23.9), which is why refcv7 trains its scorer on perturbed candidates too.

refcv7's choices, each stated:
* **Search space = REF-C's own slot controls** ``u [S, 2] = (a_lon, kappa)`` integrated by
  ``refc_sampler.roll_controls`` — the exact integrator the 117-anchor vocabulary uses. TOAD
  uses (accel, yaw-rate) under a bicycle model; here yaw-rate = v * kappa, so the space is
  the same up to reparameterisation, and there is ONE physics in the programme, not two.
* **Warm start** = the selected candidate's controls, recovered by fitting ``roll_controls``
  to its waypoints (:func:`fit_slot_controls`, deterministic LBFGS; the fit residual is
  returned so a poor inversion is visible, never silent).
* **Objective** J(u) = S(roll(u)) - lambda_a ||u - u_base||^2 - lambda_c C_comf(roll(u)),
  with TOAD's defaults lambda_a 0.5, lambda_c 0.05, M 64, E = M / 8, beta 0.5.
* **Return rule** = the better of {rolled CEM mean, base} under S - lambda_c C_comf, so the
  search can never return a plan its own reward rates below the base (asserted in tests).
* **Determinism**: every draw comes from an explicit ``torch.Generator`` seed, so the
  INFERENCE variance source (the refav1 lesson: iCEM's seed floor ~0.30 m) is controlled and
  reportable by varying the seed on purpose.

Inputs at inference: the candidate waypoints and v0 only (v0 is admissible, PI 2026-09-02).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .refc_sampler import roll_controls
from .refcv7_oracle import comfort_violation, path_kinematics


@dataclass
class ToadConfig:
    iters: int = 5              # K (TOAD: 5 for on-the-fly generators)
    samples: int = 64           # M
    elites: int = 8             # E = M / 8
    beta: float = 0.5           # initial std = max(beta * std(proposals), eps)
    eps_a: float = 0.1          # m/s^2   (TOAD floor)
    eps_kappa: float = 0.0025   # 1/m     (TOAD's 0.025 rad/s at 10 m/s)
    lam_anchor: float = 0.5
    lam_comf: float = 0.05
    control_units: str = "kappa"
    tick: float = 0.1


RewardFn = Callable[[Tensor], Tensor]   # waypoints [B, M, S, 2] -> reward [B, M]


def fit_slot_controls(xy: Tensor, v0: Tensor, horizons: tuple[int, ...],
                      steps: int = 400, tick: float = 0.1) -> tuple[Tensor, Tensor]:
    """Invert ``roll_controls`` for waypoints ``xy`` [B, S, 2] from speed ``v0`` [B].

    Returns (controls [B, S, 2] = (a_lon, kappa), rms waypoint residual [B] in metres).
    Initialised from finite differences, refined by LBFGS through the exact integrator.
    """
    B, S, _ = xy.shape
    slot_t = torch.tensor(horizons, dtype=torch.float32, device=xy.device) * tick
    with torch.no_grad():
        kin = path_kinematics(xy.float(), slot_t, v0.float())
        v = kin["v"].clamp_min(0.5)
        init = torch.stack([kin["a_lon"], (kin["yaw_rate"] / v).clamp(-0.12, 0.12)], -1)
    # float64: 16 unknowns against 16 coordinates admit an exact inverse, and float32
    # LBFGS stalls centimetres short of it over a 60 m path
    u = init.detach().double().clone().requires_grad_(True)
    target = xy.detach().double()
    v0d = v0.detach().double()
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=steps, tolerance_grad=1e-12,
                            tolerance_change=1e-14, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = roll_controls(u[:, None], v0d, horizons, "kappa", tick)[:, 0]
        loss = ((p - target) ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        p = roll_controls(u[:, None], v0d, horizons, "kappa", tick)[:, 0]
        rms = ((p - target) ** 2).sum(-1).mean(-1).sqrt()
    return u.detach().to(xy.dtype), rms.to(xy.dtype)


def toad_search(reward: RewardFn, u_base: Tensor, v0: Tensor, horizons: tuple[int, ...],
                u_proposals: Tensor | None = None, cfg: ToadConfig | None = None,
                seed: int = 0) -> dict[str, Tensor]:
    """CEM over slot controls around ``u_base`` [B, S, 2].

    ``u_proposals`` [B, N, S, 2] (the planner's other candidates, as controls) set the
    initial spread, TOAD's "exploration depends on the planner's own uncertainty".
    Returns {"xy": [B, S, 2], "u": [B, S, 2], "reward": [B], "base_reward": [B],
    "took_search": bool [B]}.
    """
    cfg = cfg or ToadConfig()
    B, S, _ = u_base.shape
    dev = u_base.device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    slot_t = torch.tensor(horizons, dtype=torch.float32, device=dev) * cfg.tick
    floor = torch.tensor([cfg.eps_a, cfg.eps_kappa], device=dev)
    if u_proposals is not None and u_proposals.shape[1] > 1:
        std = (cfg.beta * u_proposals.float().std(dim=1)).maximum(floor)
    else:
        std = floor.expand(B, S, 2).clone()
    mu = u_base.float().clone()

    def roll(u):                                  # [B, M, S, 2] -> [B, M, S, 2]
        return roll_controls(u, v0.float(), horizons, cfg.control_units, cfg.tick)

    def comf(xy):
        return comfort_violation(path_kinematics(xy, slot_t, v0.float()[:, None].expand(
            xy.shape[0], xy.shape[1])))

    with torch.no_grad():
        for _ in range(cfg.iters):
            noise = torch.randn(B, cfg.samples, S, 2, generator=g).to(dev)
            u = mu[:, None] + std[:, None] * noise
            xy = roll(u)
            j = (reward(xy) - cfg.lam_anchor * ((u - u_base[:, None]) ** 2).sum((-1, -2))
                 - cfg.lam_comf * comf(xy))
            top = j.topk(cfg.elites, dim=1).indices
            el = u.gather(1, top[:, :, None, None].expand(-1, -1, S, 2))
            mu = el.mean(dim=1)
            std = el.std(dim=1).maximum(floor)
        cand_u = torch.stack([mu, u_base.float()], dim=1)             # [B, 2, S, 2]
        cand_xy = roll(cand_u)
        final = reward(cand_xy) - cfg.lam_comf * comf(cand_xy)        # anchor term excluded
        pick = final.argmax(dim=1)                                    # 0 = search, 1 = base
        idx = pick[:, None, None, None].expand(-1, 1, S, 2)
        return {"xy": cand_xy.gather(1, idx)[:, 0], "u": cand_u.gather(1, idx)[:, 0],
                "reward": final.gather(1, pick[:, None])[:, 0], "base_reward": final[:, 1],
                "took_search": pick == 0}
