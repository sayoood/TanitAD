# Architecture & Inference — 2026-07-10

**Orthogonality/isotropy admissibility instrument (backlog P1 #3b) + isotropy-theory watch
(HamJEPA "Beyond Isotropy", PGSA "Identifiability Without Gaussianity")**

> Calendar note: wall-clock run date 2026-07-10 (Wednesday agent). Dating by wall clock per the
> Data-Eng precedent; the autonomous loop forward-dates some hub notes to mid/late-July.

LAST WEEK I measured the K-step rollout arm and read *When Does LeJEPA Learn a World Model?*
(2605.26379), whose theorem is: LeJEPA/SIGReg **linearly + orthogonally identifies** the world latent
under a **unique Gaussian prior**, and *"linear, orthogonal identifiability enables optimal latent-space
planning"* (for rotation-invariant cost). I flagged the obvious next move: that theorem's precondition
is checkable **on our own checkpoint** — is the SIGReg readout actually isotropic? This week I built
that instrument, ran it on the step-6500 trained checkpoint, and swept the isotropy-theory frontier for
the caveats. Headline: the instrument works (it exactly reproduces the independent spectral-sizing
numbers), and it returns **NOT-YET-ADMISSIBLE** at step-6500 — an honest tempering of the D-021 sizing
story and a concrete convergence tripwire for the final checkpoint.

---

## 1. The instrument (G-E increment) — what & why

`spectral.py` (D-021) fits a **linear** transition operator `(z_t, a_t) → z_{t+1}` and reports its
spectrum → "the 2048 readout ≫ the ~tens-dim task-relevant rank ⇒ OVER-PROVISIONED." The
*optimal-planning* reading of that finding rides entirely on the 2605.26379 precondition. So I added an
**admissibility gate** measured on the readout latent `z` (the space `spectral.py` sizes and planning
runs in):

| Metric | What it tests | Isotropic target |
|---|---|---|
| `iso_ratio` = geo-mean/arith-mean of cov eigenvalues ∈(0,1] | basis-free whiteness (rotation-invariant) | → 1 |
| `iso_ratio_active` (top active_k eigen-dirs) | **the theorem-relevant read** — isotropy *within the used subspace* | → 1 |
| `cond_number_active` = λ_max/λ_min on active dims | anisotropy magnitude | → 1 |
| `participation_ratio` = (Σλ)²/Σλ² | effective # significant directions | → S |
| `rms_offdiag_corr` (variance-active coords) | coordinate-space decorrelation | → 0 |

Verdict = ADMISSIBLE iff `iso_ratio_active ≥ 0.5` **and** `rms_offdiag_corr ≤ 0.1` (both **exposed
knobs**, not derived constants — 2605.26379's constants don't transfer to a finite SIGReg run; the
robust content is the *shape*: isotropy rises toward 1 as SIGReg converges). Global isotropy over all
2048 dims is expected LOW when the readout is over-provisioned (dead tail) — the instrument separates
that from active-subspace isotropy and never conflates them.

Standalone package (pure `torch`, 0 new deps): `spectral_orthogonality.py` + `run_orthogonality.py`
+ `tests/` (**8 tests, ground-truth**: isotropic→ADMISSIBLE; steep→NOT-YET; correlated-coords→flagged;
**over-provisioned r-isotropic-dims+dead-tail → recovers active_k≈r + ADMISSIBLE**; primitives on
closed forms). Intake pkg `Implementation/incoming/2026-07-10-orthogonality-instrument/`; proposed
target = extend `stack/tanitad/eval/spectral.py`. Stack suite unaffected: **189✓/1s**.

## 2. Measured result (G-H) — step-6500 trained checkpoint

`run_orthogonality.py --ckpt ckpt_full.pt --cache-dir …/comma2k19-val-61c46fca8f7f --episodes 24`
→ RTX 4060, **72 s, $0**, 24 comma2k19 val eps, **7 200 readout latents, dim 2048**:

| quantity | value | reading |
|---|---|---|
| `active_k` (cov energy knee) | **21** | = spectral run's `optimal_k` (21) ✔ |
| `cov_effective_rank` | **24.93** | = spectral run's `repr_effective_rank` (24.93) ✔ **cross-instrument check** |
| `participation_ratio_global` | **4.92** | ~5 directions carry most variance (steeper than eff-rank) |
| `iso_ratio_global` | 2.0e-8 | dead-tail dominated — **expected** (over-provisioned) |
| **`iso_ratio_active`** | **0.250** | within top-21: **anisotropic** (target 1) |
| `cond_number_active` | 246.3 | steep active spectrum (top eigs 616→296→261→158→92→69) |
| **`rms_offdiag_corr`** | **0.428** | readout coords are **strongly correlated**, not orthogonal |
| **VERDICT** | **NOT-YET-ADMISSIBLE** | SIGReg's isotropic target not reached at step-6500 |

**Two things happened, both valuable:**

1. **Cross-instrument consistency (I-row sanity).** `cov_effective_rank` 24.93 and `active_k` 21
   land *exactly* on the independent spectral-sizing numbers (repr eff-rank 24.93, optimal_k 21,
   note 2026-07-08). The two instruments read the same latent geometry — the isotropy tool is not
   inventing structure. (This is the instrument's own I1-analogue: agreement with an independent
   estimator on the shared quantity.)

2. **New content beyond sizing.** Spectral-sizing said *"the operator is low-rank ⇒ readout
   over-provisioned."* The isotropy instrument adds the **orthogonality axis**: even *within* that
   low-rank active subspace the SIGReg isotropic-Gaussian target has **not converged** at step-6500
   (`iso_active` 0.25, `cond_active` 246, coord-corr 0.43). Picture: 2 032 coordinates carry variance
   but the eigen-rank is ~5–25 → the coordinates are **redundant/correlated**, i.e. a non-orthogonal
   over-complete basis, not the whitened isotropic one the theorem needs.

## 3. Interpretation — honest consequence for D-021 (P8)

The over-provisioning finding (readout 2048 ≫ dynamics rank ~tens) **stands as descriptive**. What
this instrument *falsifies on this checkpoint* is the stronger reading I floated last week — that the
low-rank spectrum + LeJEPA theorem licenses an **"optimal latent-space planning"** claim. At step-6500
it does **not**: the orthogonality precondition is unmet. So:

- **Admissible now:** "the action-conditioned dynamics live in a ~20-dim subspace; 2048 is
  over-provisioned" (descriptive, both instruments agree).
- **NOT yet licensed:** "…therefore linear latent planning is optimal / we can safely resize to the
  knee." That waits on isotropy convergence (D-021 stays *keep 2048, keep measuring* — unchanged, and
  now with a sharper admissibility bar, D-004/G-AI1). **No architecture change is motivated** by this
  (it is a not-yet-passed admissibility row — exactly the doctrine).

This is the instrument earning its keep: it stopped a plausible-but-premature "optimal" claim.

## 4. Prediction / falsifier for the final checkpoint (decision-grade tripwire)

SIGReg's entire job is to push the marginal toward isotropic Gaussian; step-6500 is early (target
30k). **Prediction:** at 15k/30k, `iso_ratio_active` climbs toward 1, `cond_active` falls, and
`rms_offdiag_corr` → 0. **Falsifier:** if isotropy *stalls* low at the final ckpt, then either (a) the
SIGReg weight is too low (couples to the pending "raise SigReg if erank stalls" intervention — a
D-018 Tactic, escalate), or (b) the readout reshape breaks the isotropy SIGReg enforces upstream (an
architecture question). Either way the D-021 *optimal* claim is withheld and we learn something real.

## 5. Theory watch (D-013) — the isotropy frontier

The instrument's "isotropy = admissible" verdict has two important recent counter/extension results;
both are anchor-citation-graph children of 2605.26379 and both are Lean-verified:

- **HamJEPA — "Beyond Isotropy in JEPAs: Hamiltonian Geometry and Symplectic Prediction"
  (arXiv 2605.20107).** Proves **"no geometry-independent fixed marginal target is canonical: every
  fixed covariance shape can be maximally misaligned for some structured geometry."** Isotropy is
  optimal *only* under rotation-invariant downstream cost (= exactly 2605.26379's condition). For
  structured geometry they put the bias in the **cross-view coupling** (phase-space (q,p) + learned
  Hamiltonian leapfrog map, non-isotropic scale + spectral floors for anti-collapse) and beat SIGReg:
  **+3.5 / +7.5 / +10.6 linear-probe pts** on CIFAR-100/ImageNet-100 at matched epochs; ablations
  confirm the *symplectic structure* (not the MLP) drives it. **Impact on us:** validates the
  instrument's *scoping* (isotropy is the right admissibility target for our Phase-0 rotation-invariant
  trajectory cost) and seeds a **Phase-1 lever** (a symplectic/Hamiltonian predictor coupling as an
  alternative to isotropic SIGReg for structured driving geometry). Not Phase-0 (changes the trained
  objective; D-018 Tactic). — H1/H3/H5.

- **PGSA — "Identifiability Without Gaussianity: Symbolic World Models and Near-Infinite Temporal
  Consistency" (arXiv 2606.12471).** Overturns the *uniqueness* half of 2605.26379: a Physics-Grounded
  Symbolic Architecture achieves **exact linear identifiability for all physical regimes regardless of
  latent distribution**, replacing the Gaussianity requirement with **"symbolic grounding in the causal
  generator."** Key measured claim: statistical world models suffer representation error **"growing
  monotonically with time"** under non-Gaussian dynamics, while PGSA holds **"near-infinite temporal
  consistency."** **Impact on us:** this is the *theoretical mechanism* behind the horizon-degradation I
  measured last week (K-step: `imag_rel` fine at 1-step, worse at 4-step) and behind the D3 2-s
  challenge — if our SIGReg latent hasn't reached the Gaussian target (which the isotropy instrument
  now shows it hasn't, at step-6500), identifiability is only approximate → rollout error compounds.
  It reinforces **H1 (hierarchy as the compounding-error answer)** over flat long rollout (cf. FF-JEPA
  2606.09311). PGSA itself = a Phase-1+ comparison thesis (symbolic grounding is a heavy architectural
  commitment; our latent-statistical bet is deliberate), not an adoption. — H1/H3.

Also seen (watch, no action): Var-JEPA (2603.20111, variational JEPA bridging predictive/generative),
Equilibrium World Models (2606.23463). Searches: 3 web + 2 fetches (< caps).

## 6. Recommendations (G-B, each tied to a gate — G-AI1)

1. **Re-run this instrument at the final Stage-0 ckpt** (15k preview, 30k decision-grade), paired with
   the spectral re-run (backlog P0 #1). It is the **admissibility gate on any D-021 resize**: no
   resize proposal is admissible until `iso_ratio_active` has converged (falsifier: stalls low →
   withhold). Turnkey: `run_orthogonality.py --ckpt <final> --cache-dir <val cache dir>`.
2. **Keep 2048, keep measuring** (D-021 unchanged). The *descriptive* over-provisioning holds; the
   *optimal* interpretation is now gated behind a measured isotropy bar.
3. **Log active-subspace isotropy as a live training row** (companion proposal): the trainer already
   logs `erank`; add `iso_ratio_active` so isotropy convergence is watchable in-flight, not only at
   eval. Cheap (reuses this module on the readout batch) — feeds the "raise SigReg if it stalls"
   decision with a *direct* signal instead of erank alone.
4. **Phase-1 backlog seed:** HamJEPA symplectic-predictor arm as the non-isotropic alternative to
   SIGReg for structured geometry (bake-off lever, escalate before trained-config).

## 7. Ledger / gates

- HYPOTHESIS_LEDGER: H3 evidence row (no status change, P8 — measured mid-training, feeds D-021
  admissibility, not a passed gate).
- No trained-config change executed (D-004/D-018). Instrument doctrine satisfied: this is a
  not-yet-passed admissibility row, and it *blocks* rather than motivates a change.
- Resource: RTX 4060 local, 72 s, $0 (no cloud spend; Master Plan §4 respected).
