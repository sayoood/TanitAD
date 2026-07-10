"""Orthogonality / isotropy instrument for the SIGReg-trained readout latent.

WHY THIS EXISTS  (backlog P1 #3b; theory anchor arXiv 2605.26379)
----------------------------------------------------------------
`spectral.py` sizes the latent by fitting a LINEAR action-conditioned transition
operator ``(z_t, a_t) -> z_{t+1}`` and reading its singular spectrum (D-021). That
whole procedure only *licenses an optimal-planning claim* when a precondition holds:

  *When Does LeJEPA Learn a World Model?* (Klindt/LeCun/Balestriero, arXiv 2605.26379,
  Lean-4-verified) proves LeJEPA/SIGReg **linearly and orthogonally identifies** the
  world latent under stationary additive-noise transitions, and that **"linear,
  orthogonal identifiability enables optimal latent-space planning"** — provided the
  planning cost is rotation-invariant. The **Gaussian is the UNIQUE prior** for which
  the guarantee holds; SIGReg = *Sketched Isotropic Gaussian Regularization* targets
  exactly that isotropic-Gaussian marginal.

So the theorem's precondition, made falsifiable on OUR own checkpoint, is:
**has the SIGReg-trained readout marginal actually reached (near-)isotropy?**
  * If YES  -> the identification is (approximately) orthogonal -> the linear
    transition-sizing proxy of `spectral.py` is *admissible* and latent-space planning
    is (theorem) optimal for rotation-invariant costs.
  * If NO   -> SIGReg has not reached its isotropic target -> the identifiability
    guarantee is void; the spectrum is still descriptive, but the "optimal planning"
    interpretation of the D-021 sizing claim is NOT licensed.

This module measures that. It is the admissibility gate on the sizing claim — an
instrument row, NOT an architecture change (no gate, no change — G-AI1).

HONEST CAVEATS (P8)
-------------------
1. SIGReg regularizes the encoder embedding's 1-D projections; the *readout state*
   z (dim 2048 = grid 4x4 x d_readout 128, D-008) is a downstream spatial reshape.
   Isotropy of z is therefore a **proxy** for the theorem's exact regularized quantity,
   chosen because z is precisely the space `spectral.py` sizes and the space planning
   runs in. It is the right space for the D-021 admissibility question, not for a claim
   about SIGReg's own loss.
2. Isotropy is the canonical target ONLY under rotation-invariant downstream cost.
   *Beyond Isotropy in JEPAs* (HamJEPA, arXiv 2605.20107) proves **"no geometry-
   independent fixed marginal target is canonical"** and beats SIGReg on structured
   tasks with a non-isotropic (Hamiltonian) target. That is a Phase-1 architecture
   question; for the Phase-0 rotation-invariant trajectory cost the isotropy target is
   the correct admissibility criterion.
3. Global isotropy over all S dims is EXPECTED to be low when the readout is
   over-provisioned (most dims near-dead — the `spectral.py` finding: repr effective
   rank ~tens out of 2048). The theorem-relevant number is isotropy *within the active
   subspace* (the dims the representation actually uses). This module reports both and
   never conflates them.

Pure ``torch`` — no ``tanitad`` import, so the tests run standalone
(``pytest <pkg>/tests``). On integration the two private primitives below fold into the
identical ``effective_rank`` / ``energy_knee`` already in ``stack/tanitad/eval/spectral.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Primitives (identical to spectral.py; inlined for standalone tests)          #
# --------------------------------------------------------------------------- #
def _effective_rank(svals: Tensor) -> float:
    """Entropy effective rank exp(-Σ pᵢ log pᵢ), pᵢ = σᵢ/Σσ (scale-invariant)."""
    s = svals.double().clamp_min(0)
    p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def _energy_knee(svals: Tensor, energy_threshold: float = 0.99) -> int:
    """Smallest k whose top-k σ² retains ``energy_threshold`` of the total energy."""
    e = svals.double() ** 2
    cum = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
    k = int(torch.searchsorted(cum, torch.tensor(energy_threshold, dtype=cum.dtype)).item()) + 1
    return min(k, len(svals))


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
    active_k = _energy_knee(eigs.sqrt(), active_energy)   # sqrt: energy_knee squares its input
    top = eigs[:active_k]

    iso_g = isotropy_ratio(eigs)
    iso_a = isotropy_ratio(top)
    cond_g = condition_number(eigs)
    cond_a = condition_number(top)
    pr_g = participation_ratio(eigs)
    cov_er = _effective_rank(eigs.sqrt())                 # matches spectral.py repr_effective_rank
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
