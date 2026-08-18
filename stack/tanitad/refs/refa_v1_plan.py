"""REF-A v1 — the REPAIRED action search: iCEM over unicycle controls, with a
PROVABLE floor against the trivial baselines.

⛔ WHY THIS FILE EXISTS. C101 measured our CEM planner **35.8 % WORSE than
constant velocity at T1** — "the loss is in the ACTION SEARCH, not the WM". The
frozen-encoder literature (`…/Research/2026-08-18-frozen-encoder-literature/`)
puts *test-time optimisation* at the centre of every successful frozen-encoder
system (DINO-WM, V-JEPA 2-AC, GPC), which means adopting that recipe **moves our
known-worst component onto the critical path**. This module is the precondition
for doing that safely.

⭐ FOUR MECHANISMS, each answering a named failure:

1. **Kinematic action space** — search over ``(a, kappa)`` at 10 Hz through the
   unicycle (Alpamayo-2 form, v6 §4b) instead of free waypoints. Every sample is
   feasible and C2-smooth by construction, so the optimiser cannot spend its
   budget discovering that cars have a turning circle. Jerk is a
   *parameterisation*, not a penalty.

2. **iCEM, not vanilla CEM** (Pinneri et al., `martius-lab/iCEM`). Vanilla CEM
   samples i.i.d. Gaussian noise *per timestep*: the resulting control sequences
   are white noise, which a vehicle cannot execute and whose rollouts cluster in
   a useless region of trajectory space. iCEM samples **temporally correlated
   (coloured) noise**, ``S(f) ∝ f^-beta``, and carries elites across MPC ticks.
   ⇒ This is the single most likely mechanical cause of C101.

3. ⭐ **BASELINE INJECTION — the floor.** Constant-velocity, hold-``v0`` and the
   imitation proposal are injected into **every** iteration's candidate set and
   into the final argmin. ⇒ the returned plan's cost is **≤ min(baseline cost)
   BY CONSTRUCTION**, so "planner loses to CV" becomes structurally impossible
   *in modelled cost*. Pinned by :mod:`tests.test_refa_v1` with an adversarial
   cost function that makes CEM's own optimum arbitrarily bad.

4. ⛔ **AND THE HONEST LIMIT OF (3), STATED HERE SO IT TRAVELS.** The floor is a
   floor in **modelled** cost. If the cost model is miscalibrated, a planner that
   provably wins on modelled cost can still lose on realised metrics — that is
   exactly how C101 happened, and mechanism 3 alone would NOT have caught it.
   ⇒ :func:`cost_fidelity` measures rank correlation between modelled cost and
   realised outcome on banked windows, and the pre-registered admission gate
   (design doc §7) refuses to quote any planner number until it passes. **A floor
   without a fidelity check is a false comfort.**

Defaults are DINO-WM's published planning configuration (N=300 samples, 30 CEM
iterations, 30 elites, initial variance 1.0, receding horizon = planning
horizon), so the arm reproduces the recipe it is testing before it varies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor

# The programme's single unicycle integrator — never re-derive it here (a second
# integrator is a second convention, and two conventions is a retraction).
from tanitad.models.kinematic import rollout_unicycle

__all__ = ["PlanConfig", "PlanResult", "colored_noise", "icem_plan",
           "cost_fidelity", "DINO_WM_DEFAULTS"]

#: DINO-WM's published planning configuration, kept as a named constant so an
#: arm that claims "we used their setup" can be checked against one place.
DINO_WM_DEFAULTS = {"n_samples": 300, "n_iters": 30, "n_elites": 30,
                    "init_var": 1.0}


@dataclass
class PlanConfig:
    """Planner configuration. Defaults = DINO-WM + iCEM extensions OFF-by-value.

    ``beta = 0.0`` reduces the coloured noise to white noise, i.e. **vanilla
    CEM** — kept reachable on purpose so the iCEM contribution is measurable as
    a one-flag ablation rather than asserted.
    """

    # --- DINO-WM's published values -------------------------------------- #
    n_samples: int = 300
    n_iters: int = 30
    n_elites: int = 30
    init_var: float = 1.0

    # --- horizon / control ------------------------------------------------ #
    horizon: int = 10                 # optimised steps (2.0 s at dt=0.2)
    dt: float = 0.2
    a_max: float = 4.0                # m/s^2, matches v6 UnicycleEmission
    kappa_max: float = 0.2            # 1/m,   matches v6 UnicycleEmission

    # --- iCEM ------------------------------------------------------------- #
    beta: float = 2.5                 # coloured-noise exponent (0 => vanilla)
    decay: float = 1.25               # population decay per iteration
    min_samples: int = 32
    elite_memory: float = 0.3         # fraction of elites reused next MPC tick
    shift_init: bool = True           # warm-start from the shifted last plan
    alpha: float = 0.1                # mean/var smoothing across iterations

    # --- the floor -------------------------------------------------------- #
    inject_baselines: bool = True

    seed: int | None = 0

    def sanity(self) -> None:
        if self.n_elites > self.n_samples:
            raise ValueError("n_elites must be <= n_samples")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0.0 <= self.elite_memory < 1.0):
            raise ValueError("elite_memory must be in [0, 1)")


@dataclass
class PlanResult:
    controls: Tensor                  # [H, 2] chosen (a, kappa) sequence
    cost: float                       # its modelled cost
    source: str                       # "cem" | "baseline:<name>"
    baseline_costs: dict = field(default_factory=dict)
    n_evaluated: int = 0
    elites: Tensor | None = None      # [n_elites, H, 2], for elite memory
    # --- coarse-to-fine verification (filled by RefAV1.plan when the search
    # ran on the tactical field). ``coarse_fine_agree`` False means the coarse
    # level picked a plan the full field ranks below a baseline — a REPORTABLE
    # event, not a silent correction.
    fine_costs: dict = field(default_factory=dict)
    fine_best: str | None = None
    coarse_fine_agree: bool | None = None


def colored_noise(shape: tuple[int, ...], beta: float, *,
                  device=None, generator=None) -> Tensor:
    """Temporally correlated noise with power spectrum ``S(f) ∝ f^-beta``.

    ``shape = (n, H, A)``; correlation runs along **H** (time). ``beta = 0``
    returns white noise, which is exactly what vanilla CEM samples — the reason
    its control sequences are physically unrealisable at any horizon.

    Implemented by shaping a real FFT and inverting, then standardising to unit
    variance per sample so ``init_var`` keeps its meaning across ``beta``.
    """
    n, H, A = shape
    if beta == 0.0:
        return torch.randn(n, H, A, device=device, generator=generator)
    freqs = torch.fft.rfftfreq(H, device=device)
    freqs[0] = freqs[1] if freqs.numel() > 1 else 1.0     # avoid div-by-zero
    scale = freqs.pow(-beta / 2.0)
    white = torch.randn(n, A, H, device=device, generator=generator)
    spec = torch.fft.rfft(white, dim=-1) * scale
    out = torch.fft.irfft(spec, n=H, dim=-1)
    out = out - out.mean(dim=-1, keepdim=True)
    std = out.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (out / std).permute(0, 2, 1).contiguous()      # [n, H, A]


def _clip(controls: Tensor, cfg: PlanConfig) -> Tensor:
    a = controls[..., 0].clamp(-cfg.a_max, cfg.a_max)
    k = controls[..., 1].clamp(-cfg.kappa_max, cfg.kappa_max)
    return torch.stack([a, k], dim=-1)


def _baseline_controls(cfg: PlanConfig, v0: float, device,
                       proposal: Tensor | None) -> dict[str, Tensor]:
    """The candidates the planner must never lose to.

    ``cv`` / ``hold_v0`` are the same control sequence in this parameterisation
    (zero accel, zero curvature) but are kept as SEPARATE named entries because
    the eval reports them as separate floors and a silent merge would make one
    of them look absent. ``proposal`` is the imitation head's plan (GPC's
    "generative control proposes, MPC disposes").
    """
    z = torch.zeros(cfg.horizon, 2, device=device)
    out = {"cv": z, "hold_v0": z.clone()}
    if proposal is not None:
        out["proposal"] = _clip(proposal.to(device), cfg)
    # A gentle-decel candidate: the single most common correct action in dense
    # traffic, and the one a white-noise CEM population reliably misses.
    dec = z.clone()
    dec[:, 0] = -min(1.5, cfg.a_max)
    out["decel_1.5"] = dec
    return out


@torch.no_grad()
def icem_plan(cost_fn: Callable[[Tensor], Tensor], *, v0: float,
              cfg: PlanConfig | None = None,
              proposal: Tensor | None = None,
              prev_elites: Tensor | None = None,
              device=None) -> PlanResult:
    """Plan one MPC tick. ``cost_fn`` maps ``[n, H, 2]`` controls -> ``[n]`` cost.

    The returned plan is the **argmin over the union** of the CEM optimum and the
    injected baselines, which is what makes the floor structural rather than
    hoped-for.
    """
    cfg = cfg or PlanConfig()
    cfg.sanity()
    device = device or (proposal.device if proposal is not None else "cpu")
    gen = None
    if cfg.seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(cfg.seed))

    H = cfg.horizon
    mean = torch.zeros(H, 2, device=device)
    if cfg.shift_init and prev_elites is not None and prev_elites.numel():
        shifted = torch.roll(prev_elites.mean(0), shifts=-1, dims=0)
        shifted[-1] = shifted[-2] if H > 1 else 0.0
        mean = shifted
    var = torch.full((H, 2), float(cfg.init_var), device=device)

    baselines = ({} if not cfg.inject_baselines
                 else _baseline_controls(cfg, v0, device, proposal))
    base_stack = (torch.stack(list(baselines.values()))
                  if baselines else torch.empty(0, H, 2, device=device))

    n_eval = 0
    elites = None
    best_cem, best_cem_cost = None, float("inf")

    for it in range(cfg.n_iters):
        n = max(cfg.min_samples, int(cfg.n_samples / (cfg.decay ** it)))
        noise = colored_noise((n, H, 2), cfg.beta, device=device, generator=gen)
        samples = _clip(mean + noise * var.sqrt(), cfg)

        # Elite memory: carry a fraction of the previous tick's elites in, so a
        # good plan found under one observation is not thrown away at the next.
        if it == 0 and prev_elites is not None and prev_elites.numel():
            keep = max(1, int(cfg.elite_memory * cfg.n_elites))
            samples = torch.cat([samples, prev_elites[:keep].to(device)], 0)
        # The baselines compete INSIDE the loop too, so they can seed the mean.
        if base_stack.numel():
            samples = torch.cat([samples, base_stack], 0)

        costs = cost_fn(samples)
        n_eval += samples.shape[0]
        k = min(cfg.n_elites, samples.shape[0])
        idx = torch.topk(-costs, k=k).indices
        elites = samples[idx]

        if costs[idx[0]].item() < best_cem_cost:
            best_cem_cost = float(costs[idx[0]].item())
            best_cem = samples[idx[0]].clone()

        new_mean = elites.mean(0)
        new_var = elites.var(0, unbiased=False).clamp_min(1e-6)
        mean = (1 - cfg.alpha) * new_mean + cfg.alpha * mean
        var = (1 - cfg.alpha) * new_var + cfg.alpha * var

    # ---- the floor: argmin over CEM optimum UNION baselines ---------------- #
    base_costs: dict[str, float] = {}
    if base_stack.numel():
        bc = cost_fn(base_stack)
        n_eval += base_stack.shape[0]
        base_costs = {k: float(v) for k, v in zip(baselines.keys(), bc)}

    # ⛔ ``<=``, NOT ``<`` — AND THE REASON IS A REAL BUG THIS CAUGHT.
    # Because the baselines also compete INSIDE the CEM loop (they can usefully
    # seed the mean), the CEM's own best sample is frequently the baseline
    # itself. With a strict ``<`` the tie left the result labelled ``source =
    # "cem"`` while the returned controls were byte-identical to constant
    # velocity: the floor HELD but the provenance LIED, and a planner report
    # would have credited the search for a plan it did not find. MEASURED by
    # ``test_THE_FLOOR_...`` on first run. Ties now attribute to the baseline
    # and return the baseline's own (deterministic) controls.
    best_name, best_ctrl, best_cost = "cem", best_cem, best_cem_cost
    for name, c in base_costs.items():
        if c <= best_cost:
            best_name, best_cost = f"baseline:{name}", c
            best_ctrl = baselines[name]

    if best_ctrl is None:                       # pathological: n_iters == 0
        best_ctrl = torch.zeros(H, 2, device=device)
        best_name, best_cost = "baseline:cv", float(cost_fn(best_ctrl[None])[0])

    return PlanResult(controls=best_ctrl, cost=best_cost, source=best_name,
                      baseline_costs=base_costs, n_evaluated=n_eval,
                      elites=elites)


def unicycle_paths(controls: Tensor, v0: Tensor, dt: float) -> Tensor:
    """``[n, H, 2]`` controls -> ``[n, H, 2]`` (x, y) paths in the ego frame.

    Thin wrapper over the programme's integrator so every planner cost is
    computed in the SAME metric convention as the eval (x forward, y left).
    """
    n, H, _ = controls.shape
    v = v0.expand(n) if v0.ndim else v0.repeat(n)
    state0 = torch.zeros(n, 4, device=controls.device)
    state0[:, 3] = v
    return rollout_unicycle(state0, controls, dt=dt)[..., :2]


def cost_fidelity(modelled: Sequence[float],
                  realised: Sequence[float]) -> dict:
    """⛔ THE GATE MECHANISM 3 CANNOT PROVIDE — is the cost model even right?

    Spearman rank correlation between the planner's modelled cost and the
    realised outcome over banked windows. The floor guarantees we win on
    ``modelled``; only this says whether ``modelled`` means anything.

    Returns ``rho``, ``n``, and ``admissible`` against the pre-registered
    threshold (design doc §7: rho >= 0.5 on >= 200 windows). Never silently
    passes an under-powered sample: ``n < 200`` returns ``admissible=False``
    with the reason, rather than a bare correlation.
    """
    m = torch.as_tensor(list(modelled), dtype=torch.float64)
    r = torch.as_tensor(list(realised), dtype=torch.float64)
    if m.numel() != r.numel():
        raise ValueError("modelled and realised must be the same length")
    n = int(m.numel())
    if n < 2:
        return {"rho": float("nan"), "n": n, "admissible": False,
                "reason": "fewer than 2 paired windows"}

    def _rank(x: Tensor) -> Tensor:
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(x.numel(), dtype=x.dtype)
        return ranks

    rm, rr = _rank(m), _rank(r)
    rm = rm - rm.mean()
    rr = rr - rr.mean()
    denom = (rm.norm() * rr.norm()).clamp_min(1e-12)
    rho = float((rm @ rr) / denom)
    ok = bool(rho >= 0.5 and n >= 200)
    reason = ("" if ok else
              ("n < 200 (under-powered)" if n < 200 else "rho < 0.5"))
    return {"rho": rho, "n": n, "admissible": ok, "reason": reason}
