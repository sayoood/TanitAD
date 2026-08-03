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




# --------------------------------------------------------------------------- #
# Isotropy / orthogonality primitives                                          #
# --------------------------------------------------------------------------- #
def covariance_eigs(z: Tensor) -> Tensor:
    """Eigenvalues (descending, >=0) of the centered covariance of z [N, S]."""
    assert z.dim() == 2, "z must be [N, S]"
    zc = z.double() - z.double().mean(0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov)              # ascending, real (cov is symmetric PSD)
    return eig.flip(0).clamp_min(0.0)             # descending


def isotropy_ratio(eigs: Tensor) -> float:
    """Normalized whiteness = geometric-mean / arithmetic-mean of eigenvalues, in (0, 1].

    =1 iff perfectly isotropic (all eigenvalues equal). Rotation-invariant (depends only
    on the spectrum), which is the *right* invariance for the orthogonal-identifiability
    condition — a basis-free measure of how close the marginal is to ∝ I. A relative
    floor keeps the geometric mean finite when a dead tail is present.
    """
    e = eigs.double().clamp_min(0)
    if e.numel() == 0:
        return 0.0
    floor = e.max() * 1e-12
    e = e.clamp_min(floor)
    log_gm = e.log().mean()
    am = e.mean().clamp_min(1e-30)
    return float(torch.exp(log_gm) / am)


def participation_ratio(eigs: Tensor) -> float:
    """(Σλ)² / Σλ² — effective number of significant directions (≤ len)."""
    e = eigs.double().clamp_min(0)
    num = e.sum() ** 2
    den = (e ** 2).sum().clamp_min(1e-30)
    return float(num / den)


def condition_number(eigs: Tensor, floor_ratio: float = 1e-8) -> float:
    """λ_max / max(λ_min, floor_ratio·λ_max). Floored so a dead tail can't send it to ∞."""
    e = eigs.double().clamp_min(0)
    if e.numel() == 0:
        return float("inf")
    lmax = e.max()
    lmin = torch.clamp_min(e.min(), lmax * floor_ratio)
    return float(lmax / lmin.clamp_min(1e-30))


def rms_offdiag_correlation(z: Tensor, var_floor_ratio: float = 1e-4) -> tuple[float, int]:
    """RMS of the off-diagonal correlation matrix over the variance-active coordinates.

    A basis-DEPENDENT supplement to ``isotropy_ratio``: are the readout coordinates
    themselves decorrelated (the coordinate-space face of orthogonality)? Restricted to
    dims with variance > ``var_floor_ratio``·max-var so the near-dead tail does not blow
    up the D^-1/2 rescaling. Returns (rms_offdiag_corr, n_active_coords). 0 = coordinates
    fully decorrelated.
    """
    assert z.dim() == 2
    zc = z.double() - z.double().mean(0, keepdim=True)
    var = (zc ** 2).mean(0)
    keep = var > (var.max() * var_floor_ratio)
    zc = zc[:, keep]
    m = int(keep.sum())
    if m < 2:
        return 0.0, m
    std = zc.std(0, unbiased=False).clamp_min(1e-12)
    zn = zc / std
    corr = (zn.T @ zn) / zc.shape[0]
    off = corr - torch.diag(torch.diag(corr))
    rms = float(torch.sqrt((off ** 2).sum() / (m * (m - 1))))
    return rms, m


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class OrthogonalityReport:
    n_samples: int
    state_dim: int
    active_k: int                       # cov energy knee = active-subspace dim
    # global (all S dims) — dead-tail dominated when over-provisioned
    iso_ratio_global: float
    cond_number_global: float
    cov_effective_rank: float
    participation_ratio_global: float
    # active subspace (top active_k eigen-directions) — the theorem-relevant read
    iso_ratio_active: float
    cond_number_active: float
    # coordinate-space decorrelation (supplement)
    rms_offdiag_corr: float
    n_active_coords: int
    # verdict + exposed knobs (theorem constants do NOT transfer — thresholds are knobs)
    iso_threshold: float
    corr_threshold: float
    verdict: str = ""
    top_eigs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def orthogonality_report(z: Tensor,
                         active_energy: float = 0.99,
                         iso_threshold: float = 0.5,
                         corr_threshold: float = 0.1) -> OrthogonalityReport:
    """Admissibility instrument for the D-021 linear-sizing / optimal-planning claim.

    z: [N, S] readout latents (the space `spectral.py` sizes and planning runs in).
    Verdict compares the ACTIVE-subspace isotropy against ``iso_threshold`` and the
    coordinate decorrelation against ``corr_threshold``. Both are exposed knobs, not
    derived constants (2605.26379's constants do not transfer to a finite SIGReg run —
    the robust content is the *shape*: isotropy rises toward 1 as SIGReg converges).
    """
    assert z.dim() == 2, "z must be [N, S]"
    n, s = z.shape
    eigs = covariance_eigs(z)
    active_k = energy_knee(eigs.sqrt(), active_energy)   # sqrt: energy_knee squares its input
    top = eigs[:active_k]

    iso_g = isotropy_ratio(eigs)
    iso_a = isotropy_ratio(top)
    cond_g = condition_number(eigs)
    cond_a = condition_number(top)
    pr_g = participation_ratio(eigs)
    cov_er = effective_rank(eigs.sqrt())                 # matches spectral.py repr_effective_rank
    rms_off, n_act = rms_offdiag_correlation(z)

    admissible = (iso_a >= iso_threshold) and (rms_off <= corr_threshold)
    if admissible:
        verdict = (
            f"ADMISSIBLE: within its active subspace (k={active_k} of {s}) the SIGReg "
            f"readout is near-isotropic (iso_ratio_active={iso_a:.3f} >= {iso_threshold}, "
            f"coord decorrelation rms={rms_off:.3f} <= {corr_threshold}). The orthogonal-"
            f"identifiability precondition (2605.26379) is met on this checkpoint -> the "
            f"linear transition-sizing proxy (D-021) is admissible and latent-space planning "
            f"is theorem-optimal for rotation-invariant trajectory cost. Global isotropy "
            f"(iso_ratio_global={iso_g:.3g}) is low BY DESIGN (over-provisioned dead tail), "
            f"not a failure."
        )
    else:
        why = []
        if iso_a < iso_threshold:
            why.append(f"active-subspace anisotropy (iso_ratio_active={iso_a:.3f} < {iso_threshold}; "
                       f"cond_active={cond_a:.1f})")
        if rms_off > corr_threshold:
            why.append(f"coordinate correlation (rms_offdiag={rms_off:.3f} > {corr_threshold})")
        verdict = (
            f"NOT-YET-ADMISSIBLE: {'; '.join(why)}. SIGReg has not reached its isotropic-"
            f"Gaussian target within the active subspace (k={active_k}) on this checkpoint, so "
            f"the 2605.26379 optimal-planning guarantee is not licensed. The `spectral.py` "
            f"spectrum stays DESCRIPTIVE (rank/knee), but the D-021 'optimal' interpretation "
            f"must wait for a checkpoint whose readout isotropy has converged. (Expect iso to "
            f"rise with SIGReg training — re-measure at the final Stage-0 ckpt.)"
        )

    return OrthogonalityReport(
        n_samples=n, state_dim=s, active_k=active_k,
        iso_ratio_global=iso_g, cond_number_global=cond_g,
        cov_effective_rank=cov_er, participation_ratio_global=pr_g,
        iso_ratio_active=iso_a, cond_number_active=cond_a,
        rms_offdiag_corr=rms_off, n_active_coords=n_act,
        iso_threshold=iso_threshold, corr_threshold=corr_threshold,
        verdict=verdict, top_eigs=top[:64].tolist())
