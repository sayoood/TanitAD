# Verification: the stranded 2026-07-10 orthogonality instrument reproduces — recommend MERGE

**Author:** Architecture & Inference (Wed), 2026-07-17.

## What happened
While drafting a "new" orthogonality instrument for backlog 3b this run, I found an **existing,
theoretically-superior instrument already built on 2026-07-10** but **never merged** — it sits on the
unmerged branch `worktree-agent-arch-inf-20260710` (commit `5f0c316`), intake
`Implementation/incoming/2026-07-10-orthogonality-instrument/` (`spectral_orthogonality.py` + tests +
`run_orthogonality.py` + research note `Research/2026-07-10-orthogonality-instrument-and-isotropy-theory.md`).
Rather than ship a redundant, less-careful duplicate, I **withdrew my version** and **verified the prior one**.

## Why the prior instrument is the right one (not mine)
My draft measured **global** covariance isotropy over all 2048 dims and read isotropy≈0 / off-diagonal≈0.999
→ it would have concluded "orthogonality fails." The 2026-07-10 instrument explicitly warns this is the
wrong number: for an **over-provisioned** readout the global isotropy is ~0 **by design** (dead tail); the
theorem-relevant quantity is **isotropy within the active subspace** (the energy-knee dims the
representation actually uses). It reports both and never conflates them — mine did not have the
active-subspace read at all.

## Verification (2026-07-17, RTX 4060, step-6500 base250cam ckpt, n=2600 real val states > S=2048)
Ran the prior `orthogonality_report` unchanged on the same ckpt. It **reproduces its 2026-07-10 number
exactly**:

| metric | value | note |
|---|--:|---|
| active_k (energy-knee subspace) | 23 | matches spectral knee ≈22–31 |
| **iso_ratio_active** | **0.254** | = the logged "0.250"; < 0.5 threshold |
| cond_number_active | 217.9 | anisotropic within the active subspace |
| rms_offdiag_corr (active coords) | 0.424 | > 0.1 threshold → coordinates still correlated |
| cov_effective_rank | 26.0 | matches spectral repr-rank ~tens |
| iso_ratio_global | 1.6e-8 | ~0 **by design** (over-provisioned), NOT a failure |
| **verdict** | **NOT-YET-ADMISSIBLE** | SIGReg isotropy not yet converged on this ckpt |

My independent global-covariance read (isotropy 0.000, off-diagonal 0.999, participation 5.0)
**corroborates the over-provisioning** from the coordinate-space angle, but the **active-subspace 0.254**
is the correct admissibility read.

## Recommendation to the orchestrator
1. **Merge the stranded `2026-07-10-orthogonality-instrument` intake** into `stack/tanitad/eval/` (fold its
   primitives into `spectral.py` per its own note). It reproduces cleanly, has standalone tests, and is the
   right instrument for the D-021 admissibility question — it should not stay stranded a 3rd week.
2. **State D-021 as subspace identification, not "optimal planning,"** on the current (pre-reset) evidence:
   iso_ratio_active 0.254 < 0.5 → the LeJEPA optimal-planning precondition is **not yet met**.
3. **Re-run it on the flagship @30k** — the prior note's own next step (expect iso to rise as SIGReg
   converges); that read is decision-grade, this one is directional (pre-reset ckpt).

Artifact: `2026-07-17-verify-prior-orthogonality.json` (this run's full output).
