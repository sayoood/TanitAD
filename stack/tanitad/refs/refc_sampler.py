"""refcv5 WP-4 / ``E-DDA-3`` — the truncated-diffusion sampler, in CONTROL space.

⭐ **This is the programme's advance over DiffusionDrive, and it is one decision:
the noise lives in ``(a_lon, a_lat)``, not in metres.** Every sample therefore
re-rolls through the programme's own integrator and is **flyable by
construction**.

## Why control space — the argument, with the arithmetic

DD noises the WAYPOINTS. Under its ``scaled_linear`` schedule at the truncation
point ``t = 8``, ``sqrt(1 - alpha_bar_8) = 0.0316``; in DD's metre normalisation
that is **sigma_x ~ 0.90 m / sigma_y ~ 0.73 m PER WAYPOINT, INDEPENDENTLY**. Over
a 0.5 s slot a 0.73 m lateral excursion implies a lateral acceleration of
**~5.8 m/s^2 (0.6 g)**, and a 0.91 m along-track one a **1.8 m/s speed jump**.
The sampled fan would be full of trajectories no car can drive — precisely what
``D-REFCV4B-EGODROP2`` says we must not add (selected-trajectory jerk already
reads **2.42 vs the human's 0.86 m/s^3**).

The SAME schedule in control units gives ``sigma(t=8) = 0.126 m/s^2 along /
0.095 m/s^2 lateral``, i.e. **~2.3 m along-track and ~1.7 m lateral at the 6 s
endpoint** — comparable spread to DD's, but every sample is a smooth, integrable
path. *(The lateral figure is speed-independent because ``kappa = a_lat / v^2``
cancels against path length.)*

⚠️ **The resolution objection, answered by the SEQUENCE and not by metres.** A
constant ``(a_lon, a_lat)`` held for 6 s cannot trace a real 6 s trajectory. The
**8-slot control sequence** has the same 16 degrees of freedom as the waypoint
offset it replaces, so nothing expressible before becomes inexpressible.

## What is adopted from DD verbatim (PUBLISHED, `hustvl/DiffusionDrive@9b52ed0`)

| item | value |
|---|---|
| schedule | ``DDIMScheduler(1000, beta_schedule="scaled_linear", prediction_type="sample")``, V2's ``steps_offset=1`` |
| train timestep | ``t ~ U[0, 50)`` |
| inference | fresh eps at ``t = 8``, two DDIM steps ``[10, 0]``, ``eta = 0`` |
| prediction target | ``x0`` ("sample"), not eps |
| timestep embedding | ``SinusoidalPosEmb -> Linear -> Mish -> Linear``, injected **per layer** |

⛔ **The scheduler is RE-IMPLEMENTED here, ~40 lines, rather than imported from
``diffusers``** — and that is a deliberate decision, not NIH. Pods have no
reliable way to add a dependency (``uv pip install <anything>`` has TWICE
silently replaced torch with a wheel the driver cannot run, `CLAUDE.md`), and an
analysis-time import that fails **after** the rollout destroys a run whose
compute is already paid for. :func:`assert_matches_diffusers` pins this
implementation against the real ``diffusers`` object wherever it IS installed, so
the re-implementation is checked rather than trusted.

## The controls this module is bound by

* ``sigma == 0`` at ``t == 0`` must reproduce the deterministic decoder
  **bit-for-bit** — the control that must read a known value.
* A **constant** control sequence rolled by :func:`roll_controls` must be
  **bit-identical** to ``AnchoredDiffusionDecoder.roll_bank``, or the sampler and
  the vocabulary are integrating two different physics.
* The **deliberate regression** is the DD-literal metre-space arm
  (``sampler_space="metre"``): it is pre-registered to FAIL the flyability gate.
  If it does NOT fail, the flyability instrument cannot see what it is cited for
  and the whole arm is VOID.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from tanitad.models.kinematic import rollout_unicycle

__all__ = [
    "DDIMSchedule", "SinusoidalPosEmb", "build_time_mlp", "roll_controls",
    "controls_to_ticks", "alat_to_curvature", "assert_matches_diffusers",
    "BETA_START", "BETA_END", "N_TRAIN_TIMESTEPS",
]

#: DD's schedule constants, verbatim. Named so a reader can check them against
#: the published config rather than finding them inlined in an expression.
BETA_START: float = 0.0001
BETA_END: float = 0.02
N_TRAIN_TIMESTEPS: int = 1000
#: Rounding budget for :func:`assert_matches_diffusers`, in units of the
#: reference dtype's epsilon. MEASURED 2026-09-06 against diffusers 0.40.0:
#: 1.689e-07 = 1.42 x float32 eps over the 1000-step cumprod.
ACC_EPS_MULT: float = 16.0


class DDIMSchedule:
    """``scaled_linear`` DDIM with ``prediction_type="sample"`` (x0) and
    ``eta = 0``. Stateless w.r.t. the model; holds only the alpha table.

    ⛔ ``scaled_linear`` means ``betas = linspace(sqrt(b0), sqrt(bT), T)**2`` —
    the square is on the LINSPACE, not on the betas. Getting this backwards
    produces a schedule that looks plausible and noises by the wrong amount at
    every t, which would silently change what ``t = 8`` means.
    """

    def __init__(self, num_train_timesteps: int = N_TRAIN_TIMESTEPS,
                 beta_start: float = BETA_START, beta_end: float = BETA_END,
                 steps_offset: int = 1, device=None,
                 dtype: torch.dtype = torch.float64):
        self.num_train_timesteps = int(num_train_timesteps)
        self.steps_offset = int(steps_offset)
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                               self.num_train_timesteps, dtype=dtype,
                               device=device) ** 2
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def to(self, device) -> "DDIMSchedule":
        """Move the alpha table onto ``device``. Returns ``self``.

        ⛔ **Why this is not optional.** ``add_noise`` unsqueezes the gathered
        alpha to ``x0``'s rank before multiplying, so a CPU table meeting a CUDA
        ``x0`` does NOT get the 0-dim scalar exemption -- it raises
        ``Expected all tensors to be on the same device``. The schedule is
        constructed at build time, when the module may still be on CPU, and the
        model is moved afterwards; ``nn.Module.to`` cannot follow it because the
        table is deliberately NOT a buffer (it is a constant of the published
        schedule, and putting it in ``state_dict`` would change checkpoint
        compatibility for a quantity no run can alter). So the move happens at
        use time, here.

        ``Tensor.to`` returns ``self`` when the tensor is already on ``device``,
        so calling this every forward costs nothing.
        """
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    # -- the two quantities everything else is built from ------------------ #
    def sqrt_abar(self, t: Tensor | int) -> Tensor:
        return self._gather(t).sqrt()

    def sqrt_one_minus_abar(self, t: Tensor | int) -> Tensor:
        return (1.0 - self._gather(t)).sqrt()

    def _gather(self, t) -> Tensor:
        idx = torch.as_tensor(t, device=self.alphas_cumprod.device)
        idx = idx.clamp(0, self.num_train_timesteps - 1).long()
        return self.alphas_cumprod[idx]

    # -- forward noising ---------------------------------------------------- #
    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """``x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps``.

        ``t`` broadcasts against ``x0``'s LEADING dims; the trailing dims are
        unsqueezed here so a per-sample timestep works without the caller
        reshaping (the classic silent-broadcast bug in this family).
        """
        a = self.sqrt_abar(t).to(x0.dtype)
        s = self.sqrt_one_minus_abar(t).to(x0.dtype)
        while a.ndim < x0.ndim:
            a = a.unsqueeze(-1)
            s = s.unsqueeze(-1)
        return a * x0 + s * noise

    # -- the reverse step, eta = 0, x0-parameterised ------------------------ #
    def step(self, x0_hat: Tensor, x_t: Tensor, t: Tensor,
             t_prev: Tensor) -> Tensor:
        """One deterministic DDIM step from ``t`` to ``t_prev``.

        With ``eta = 0``:
            ``eps_hat = (x_t - sqrt(abar_t) x0_hat) / sqrt(1 - abar_t)``
            ``x_prev  = sqrt(abar_prev) x0_hat + sqrt(1 - abar_prev) eps_hat``

        ⭐ At ``t_prev == 0``, ``abar_prev`` is the FIRST table entry (not 1.0),
        which is what diffusers does with ``steps_offset = 1`` and is why the
        final sample is not exactly ``x0_hat``. Pinned in the tests.
        """
        a_t = self.sqrt_abar(t).to(x_t.dtype)
        s_t = self.sqrt_one_minus_abar(t).to(x_t.dtype)
        a_p = self.sqrt_abar(t_prev).to(x_t.dtype)
        s_p = self.sqrt_one_minus_abar(t_prev).to(x_t.dtype)
        while a_t.ndim < x_t.ndim:
            a_t, s_t = a_t.unsqueeze(-1), s_t.unsqueeze(-1)
            a_p, s_p = a_p.unsqueeze(-1), s_p.unsqueeze(-1)
        eps = (x_t - a_t * x0_hat) / s_t.clamp_min(1e-12)
        return a_p * x0_hat + s_p * eps

    def infer_timesteps(self, start_t: int, steps: int) -> list[int]:
        """DD's truncated inference ladder, DESCENDING and ending at 0.

        ``start_t=8, steps=2`` -> ``[10, 0]`` — the value the audit recorded
        from `hustvl/DiffusionDrive@9b52ed0`. The ``8 -> 10`` shift is
        ``steps_offset``: the ladder's top is ``start_t + steps_offset * steps``,
        and the ``steps`` entries descend linearly from there to 0.

        ⚠️ ``steps == 1`` gives ``[start_t + steps_offset]`` — there is no
        descent to interpolate, so the single pass denoises from the truncation
        point straight to 0. Stated because "one step" is the degenerate case a
        reader will assume behaves like the first entry of the two-step ladder,
        and it does not.
        """
        if steps < 1:
            return []
        hi = int(start_t) + self.steps_offset * steps
        if steps == 1:
            return [int(start_t) + self.steps_offset]
        return [int(round(hi * (1.0 - i / (steps - 1)))) for i in range(steps)]


def assert_matches_diffusers(sched: DDIMSchedule,
                             atol: float | None = None) -> bool:
    """Pin this re-implementation against the real ``diffusers`` object.

    ⭐ **Why this exists.** The scheduler is re-implemented so pods need no new
    dependency, and a re-implementation that is never checked is just an
    unverified copy. Where ``diffusers`` IS importable the alpha table must
    agree to ``atol``; where it is not, the caller SKIPS rather than passing —
    returns ``False`` so a test can mark itself skipped instead of green.

    ⛔⛔ **THE `atol = 1e-10` THIS REPLACES COULD NEVER PASS, AND NOBODY KNEW
    BECAUSE IT HAD NEVER RUN.** MEASURED 2026-09-06, the first time
    ``diffusers`` was actually installed beside this module (0.40.0): the check
    raised at **1.689e-07**. The cause is a DTYPE comparison, not an error --
    ``DDIMScheduler`` builds its table in **float32** while
    :class:`DDIMSchedule` builds it in **float64**, and 1.689e-07 is float32
    epsilon (1.192e-07) territory. Against an exact float64 reference,
    **ours reads 6.939e-18 and diffusers reads 1.689e-07**: the residual is the
    REFERENCE's rounding, and this implementation is the more accurate of the
    two. A tolerance below the reference's own epsilon is not a strict test, it
    is an UNRUNNABLE one -- the mirror image of a guard that cannot fail.

    ⭐ **The replacement is STRICTER, not looser.** At the reference's own dtype
    the two tables must be **BIT-EQUAL** (MEASURED: max |d| = **0.000e+00** over
    all 1000 entries), which admits no tolerance at all; the float64 residual is
    then bounded by the reference dtype's epsilon, which is exactly what it
    costs to hold the table in float32 -- ``ACC_EPS_MULT`` epsilons, because a
    1000-step ``cumprod`` accumulates. ``atol`` overrides that bound.
    """
    try:
        from diffusers import DDIMScheduler          # noqa: PLC0415
    except Exception:
        return False
    ref = DDIMScheduler(num_train_timesteps=sched.num_train_timesteps,
                        beta_start=BETA_START, beta_end=BETA_END,
                        beta_schedule="scaled_linear",
                        prediction_type="sample",
                        steps_offset=sched.steps_offset)
    a = sched.alphas_cumprod.cpu()
    b = ref.alphas_cumprod.cpu()
    if a.shape != b.shape:
        raise AssertionError(f"alpha table shape {a.shape} != {b.shape}")
    # PRIMARY: THE FORMULA, proven exactly. Rebuild OUR schedule at the
    # reference's own dtype and require BIT-EQUALITY -- same operations, same
    # order, same precision, so any difference at all is a formula difference.
    # ⚠ Casting the float64 table DOWN instead would NOT be this test and
    # fails at 1.788e-07: rounding once at the end is not the same number as
    # rounding at each of 1000 cumprod steps, which is what diffusers does.
    same = DDIMSchedule(sched.num_train_timesteps, BETA_START, BETA_END,
                        steps_offset=sched.steps_offset, dtype=b.dtype)
    if not torch.equal(same.alphas_cumprod.cpu(), b):
        d = float((same.alphas_cumprod.cpu().to(torch.float64)
                   - b.to(torch.float64)).abs().max())
        raise AssertionError(
            f"alphas_cumprod built at the reference dtype {b.dtype} is NOT "
            f"bit-equal to diffusers: max |diff| = {d:.3e} -- the SCHEDULE "
            f"FORMULA disagrees, which no dtype choice can explain")
    # SECONDARY: bound the PRECISION gap between the caller's table and the
    # reference. The formula is already proven bit-exact above, so the only
    # thing left to bound is accumulated rounding.
    # ⚠ The bound is `ACC_EPS_MULT * eps`, NOT one epsilon: a 1000-step
    # `cumprod` accumulates rounding, and the MEASURED gap between a float64
    # table and diffusers' float32 one is 1.689e-07 = 1.42 x float32 eps. One
    # eps would fail on a correct implementation -- the same unrunnable-pin
    # error as the 1e-10 it replaced, one order less obvious. The margin is
    # still ~4 orders below any REAL disagreement (mistaking `scaled_linear`
    # for `linear` moves beta[499] by ~1e-2).
    tol = (ACC_EPS_MULT * float(torch.finfo(b.dtype).eps)
           if atol is None else float(atol))
    err = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
    if err > tol:
        raise AssertionError(
            f"alphas_cumprod differs from diffusers by {err:.3e} > {tol:.3e} "
            f"({ACC_EPS_MULT} x the {b.dtype} epsilon) -- far larger than "
            f"accumulated rounding, so this is a real disagreement")
    return True


# ---------------------------------------------------------------------------
# The per-layer timestep embedding (DD's, verbatim in shape)
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """``t -> [B, d]`` sinusoidal features. DD's, and the standard form."""

    def __init__(self, d: int):
        super().__init__()
        if d % 2:
            raise ValueError(f"SinusoidalPosEmb needs an even d, got {d}")
        self.d = int(d)

    def forward(self, t: Tensor) -> Tensor:
        half = self.d // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.reshape(-1, 1).to(torch.float32) * freqs.reshape(1, -1)
        return torch.cat([a.sin(), a.cos()], dim=-1)


def build_time_mlp(d: int) -> nn.Module:
    """``SinusoidalPosEmb(d) -> Linear -> Mish -> Linear`` — DD's block."""
    return nn.Sequential(SinusoidalPosEmb(d), nn.Linear(d, 4 * d), nn.Mish(),
                         nn.Linear(4 * d, d))


# ---------------------------------------------------------------------------
# Control -> path, integrated the way the VOCABULARY is integrated
# ---------------------------------------------------------------------------
def alat_to_curvature(a_lat: Tensor, v: Tensor, v_floor: float,
                      kappa_cap: float) -> Tensor:
    """``kappa = a_lat / max(v, v_floor)^2``, clamped — the EXACT expression
    ``AnchoredDiffusionDecoder.roll_bank`` uses.

    ⛔ It is duplicated NOWHERE ELSE: this is the one spelling, and
    ``roll_bank`` calls it too. Two spellings of this formula is how the
    sampler and the vocabulary would silently integrate different physics.

    ``v`` broadcasts against ``a_lat``.
    """
    vv = v.clamp_min(float(v_floor)) ** 2
    return (a_lat / vv).clamp(-float(kappa_cap), float(kappa_cap))


def controls_to_ticks(u: Tensor, horizons: tuple[int, ...]) -> Tensor:
    """Expand a PER-SLOT control ``[..., S, 2]`` to the native tick grid
    ``[..., max(horizons), 2]`` by holding each slot's control over its span.

    ⭐ **This is the step that makes the sampler and the vocabulary agree.**
    ``roll_bank`` integrates at the native 0.1 s tick and subsamples at the slot
    indices (``rollout_unicycle_grid``'s "recommended emission"). A sampler that
    instead took ONE variable-length step per slot would be a *different
    integrator*, and a constant control would then NOT reproduce the bank — so
    every anchored-Gaussian claim would be measured against a fan the vocabulary
    never emitted.
    """
    h = [int(x) for x in horizons]
    if any(b <= a for a, b in zip(h, h[1:])) or h[0] <= 0:
        raise ValueError(f"horizons must be positive and increasing, got {h}")
    if u.shape[-2] != len(h):
        raise ValueError(f"u has {u.shape[-2]} slots but horizons has {len(h)}")
    reps = [b - a for a, b in zip([0] + h[:-1], h)]
    return torch.repeat_interleave(
        u, torch.tensor(reps, device=u.device), dim=-2)


def roll_controls(u: Tensor, v0: Tensor, horizons: tuple[int, ...],
                  control_units: str = "kappa", tick: float = 0.1,
                  alat_v_floor: float = 4.0, kappa_cap: float = 0.12
                  ) -> Tensor:
    """``u [B, N, S, 2] -> waypoints [B, N, S, 2]``, through the programme's own
    integrator, from each window's measured ``v0 [B]``.

    ⚠️ Integrated in **float32 regardless of the AMP dtype** — the same rule
    ``roll_bank`` states, and for the same reason: ~60 sequential integration
    steps in fp16 accumulate visible drift, and this geometry is what every
    anchor target is measured against.

    ⛔ **The bit-identity contract:** for a control that is CONSTANT across the
    slot axis, this must return exactly ``roll_bank``'s bank. That is a test,
    not an aspiration (`test_refc_sampler.py`).
    """
    if control_units not in ("kappa", "alat"):
        raise ValueError(f"control_units {control_units!r} not in "
                         f"('kappa', 'alat')")
    b, n, s = u.shape[0], u.shape[1], u.shape[2]
    out_dtype = u.dtype
    u32 = u.to(torch.float32)
    v = torch.as_tensor(v0, dtype=torch.float32,
                        device=u.device).reshape(-1)
    if v.shape[0] != b:
        raise ValueError(f"v0 has {v.shape[0]} rows, expected {b}")
    a_lon = u32[..., 0]
    if control_units == "alat":
        kap = alat_to_curvature(u32[..., 1], v[:, None, None],
                                alat_v_floor, kappa_cap)
    else:
        kap = u32[..., 1]
    ctrl = torch.stack([a_lon, kap], dim=-1)                 # [B, N, S, 2]
    ticks = controls_to_ticks(ctrl, horizons)                # [B, N, T, 2]
    t_max = ticks.shape[-2]
    ticks = ticks.reshape(b * n, t_max, 2)
    state0 = torch.zeros(b * n, 4, device=u.device, dtype=torch.float32)
    state0[:, 3] = v[:, None].expand(b, n).reshape(-1)
    path = rollout_unicycle(state0, ticks, dt=float(tick))[..., :2]
    idx = torch.tensor([int(k) - 1 for k in horizons], device=u.device,
                       dtype=torch.long)
    return path.index_select(1, idx).reshape(b, n, s, 2).to(out_dtype)
