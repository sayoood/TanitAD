"""Spectral sizing of the action-conditioned latent transition operator (backlog #0, L2).

WHY THIS EXISTS
---------------
The JEPA generalization theory (arXiv 2606.27014, analyzed in
``Architecture & Inference/Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md``)
makes the latent dimension a **measurable** design decision, not a guess:

  * JEPA pretraining learns the top-k singular structure of the action-conditioned
    transition operator M̄(a)  (Thm 3.1).
  * Approximation error is the **spectral tail** Σ_{i>k} σᵢ² beyond latent dim k
    (Thm 4.3), which DECREASES in k.
  * Sample error grows ~O(k²) in k (Thm 4.4).
  * Finite-sample planning regret ≈ √(spectral tail (↓k) + complexity (↑k)) — a
    trade-off with an interior optimum at the **knee** of the spectrum (Thms 4.5/4.6).

Leverage action **L2 (`p0-spectral-sizing`)**: empirically estimate that spectrum —
fit a linear map ``(z_t, a_t) -> z_{t+1}`` on driving latents, look at σᵢ decay, and
place the readout/state dim at the knee. This validates (or corrects) the 2048-dim
readout of D-008 (grid 4×4 × d_readout 128). It is also the offline counterpart of
the live ``erank`` collapse-health rows already logged every training step.

WHAT IS AND IS NOT VALIDATED HERE
---------------------------------
The estimator is pure linear algebra and is unit-tested on synthetic data with a
KNOWN rank (it must recover the knee) and on the real ``WorldModel`` latent path
(smoke checkpoint) to prove the extraction hook. The *decision-grade* comma2k19
spectrum needs a **trained** checkpoint (an untrained encoder's latents are
near-isotropic and carry no dynamics structure — a degenerate spectrum, honestly
useless for sizing). That run is queued behind the A40 Stage-0 checkpoint
(``stack/RUNPOD_RUNBOOK.md``); this module is the tool it will call. (P8: no sizing
recommendation is claimed on untrained latents.)

Reuses the tested closed-form ``RidgeProbe`` (``stack/tanitad/models/readout.py``)
for the operator fit — no reimplementation. Proposed target on integration:
``stack/tanitad/eval/spectral.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from tanitad.models.readout import RidgeProbe

CURRENT_READOUT_DIM = 2048   # D-008: grid 4×4 × d_readout 128 (base250)


# --------------------------------------------------------------------------- #
# Spectral primitives                                                          #
# --------------------------------------------------------------------------- #
def effective_rank(svals: Tensor) -> float:
    """Entropy-based effective rank exp(-Σ pᵢ log pᵢ), pᵢ = σᵢ/Σσ.

    Same quantity as the training-time ``FallbackMonitor._effective_rank`` collapse
    row, here computed from a supplied singular spectrum. Uses σ (not σ²) to match
    that monitor exactly."""
    s = svals.double().clamp_min(0)
    p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def energy_knee(svals: Tensor, energy_threshold: float = 0.99) -> int:
    """Smallest k whose top-k σ² retains ``energy_threshold`` of the total energy."""
    e = svals.double() ** 2
    cum = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
    k = int(torch.searchsorted(cum, torch.tensor(energy_threshold, dtype=cum.dtype)).item()) + 1
    return min(k, len(svals))


def spectral_tail(svals: Tensor, k: int) -> float:
    """Normalized approximation error Σ_{i>k} σᵢ² / Σ σᵢ²  (Thm 4.3)."""
    e = svals.double() ** 2
    tot = e.sum().clamp_min(1e-12)
    return float(e[k:].sum() / tot) if k < len(e) else 0.0


def optimal_k(svals: Tensor, n_samples: int, complexity_weight: float = 1.0) -> int:
    """argmin_k [ spectral_tail(k) + complexity_weight · k² / n_samples ].

    Operationalizes the finite-sample trade-off (Thms 4.5/4.6): approximation
    (spectral tail, ↓k) vs sample error (~O(k²), ↑k). ``complexity_weight`` is an
    exposed KNOB, not a derived constant (the theory's constants do not transfer —
    L5). What is robust and validated is the *shape*: the optimum sits near the
    spectral knee and moves toward it as ``n_samples`` grows.
    """
    ks = range(1, len(svals) + 1)
    cost = [spectral_tail(svals, k) + complexity_weight * (k * k) / max(n_samples, 1) for k in ks]
    return 1 + int(torch.tensor(cost).argmin().item())


# --------------------------------------------------------------------------- #
# Operator fit + report                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class SpectrumReport:
    n_samples: int
    state_dim: int
    action_dim: int
    fit_r2: float                                  # operator fit quality (sanity floor, like I1)
    operator_svals: list[float]                    # SVD of the fitted state-transition block A
    repr_svals: list[float]                        # PCA spectrum of z_next (representation's own use)
    operator_effective_rank: float
    repr_effective_rank: float
    energy_knee_k: int
    optimal_k_theory: int
    current_readout_dim: int
    spectral_tail_at: dict[int, float] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def fit_transition_operator(z_t: Tensor, a_t: Tensor, z_next: Tensor,
                            alpha: float = 1e-3) -> tuple[Tensor, float]:
    """Fit centered linear operator on [z_t | a_t] -> z_next via RidgeProbe.

    Returns (A, fit_r2) where A [S, S] is the state-transition block of the fitted
    operator (the map z_t -> z_next with action marginalized into the fit). fit_r2
    is the operator's R² (sanity: a near-zero R² means the linear-operator proxy is
    inappropriate for these latents — do not size from its spectrum).
    """
    feats = torch.cat([z_t, a_t], dim=-1)
    probe = RidgeProbe(alpha=alpha).fit(feats, z_next)
    s = z_t.shape[-1]
    A = probe.W[:s]                                # [S, S] state block (double)
    return A, probe.r2(feats, z_next)


def estimate_transition_spectrum(z_t: Tensor, a_t: Tensor, z_next: Tensor,
                                 current_readout_dim: int = CURRENT_READOUT_DIM,
                                 energy_threshold: float = 0.99,
                                 complexity_weight: float = 1.0,
                                 alpha: float = 1e-3) -> SpectrumReport:
    """Estimate the action-conditioned transition spectrum and recommend a latent dim.

    z_t, z_next: [N, S] consecutive latent states; a_t: [N, A] action at t.
    """
    assert z_t.shape == z_next.shape and z_t.dim() == 2, "z_t/z_next must be [N, S]"
    n, s = z_t.shape
    A, r2 = fit_transition_operator(z_t, a_t, z_next, alpha)
    op_sv = torch.linalg.svdvals(A.double())

    zc = z_next.double() - z_next.double().mean(0, keepdim=True)
    repr_sv = torch.linalg.svdvals(zc)

    knee = energy_knee(op_sv, energy_threshold)
    kstar = optimal_k(op_sv, n, complexity_weight)
    cand = sorted({knee, kstar, current_readout_dim, s, s // 2, s // 4} - {0})
    tail_at = {k: spectral_tail(op_sv, k) for k in cand if k <= s}

    if current_readout_dim > 4 * max(knee, kstar):
        rec = (f"OVER-PROVISIONED: readout dim {current_readout_dim} >> knee {knee} / k* {kstar}. "
               f"Task-relevant dynamics rank is far below the state dim — sample-error term "
               f"(~O(k²)) is paid for nothing. Consider a smaller d_readout or a low-rank probe.")
    elif current_readout_dim < knee:
        rec = (f"UNDER-PROVISIONED: readout dim {current_readout_dim} < knee {knee}. Spectral tail "
               f"{spectral_tail(op_sv, current_readout_dim):.3f} of dynamics energy is truncated — "
               f"approximation error. Consider a larger d_readout.")
    else:
        rec = (f"IN RANGE: readout dim {current_readout_dim} brackets the knee {knee} / k* {kstar}. "
               f"Spectrum supports the D-008 sizing on this data.")

    return SpectrumReport(
        n_samples=n, state_dim=s, action_dim=a_t.shape[-1], fit_r2=r2,
        operator_svals=op_sv.tolist(), repr_svals=repr_sv.tolist(),
        operator_effective_rank=effective_rank(op_sv),
        repr_effective_rank=effective_rank(repr_sv),
        energy_knee_k=knee, optimal_k_theory=kstar,
        current_readout_dim=current_readout_dim, spectral_tail_at=tail_at,
        recommendation=rec)


# --------------------------------------------------------------------------- #
# Latent extraction from a checkpoint                                          #
# --------------------------------------------------------------------------- #
def pairs_from_states(states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """[N, W, S] states + [N, W, A] actions -> consecutive (z_t, a_t, z_next) pairs
    flattened over the window: z_t=states[:, :-1], z_next=states[:, 1:]."""
    assert states.dim() == 3 and actions.dim() == 3
    z_t = states[:, :-1].reshape(-1, states.shape[-1])
    z_next = states[:, 1:].reshape(-1, states.shape[-1])
    a_t = actions[:, :-1].reshape(-1, actions.shape[-1])
    return z_t, a_t, z_next


@torch.no_grad()
def latents_from_world(world, frames_window: Tensor) -> Tensor:
    """[N, W, C, H, W'] frame windows -> [N, W, S] readout states via a WorldModel."""
    return world.encode_window(frames_window)
