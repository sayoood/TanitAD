# Blind K-step belief rollout dissipates uncertainty + collapses to an attractor; the 2048 readout is not orthogonal

**Agent:** Architecture & Inference (Wednesday) — run 2026-07-17
**Calendar:** wall-clock 2026-07-17 (dating by wall clock; LAST_RUN was 2026-07-15).
**Compute:** RTX 4060 (local), tanitad venv (torch 2.11 cu128); $0. Trained substrate:
`ckpt_full.pt` = **step-6500 `base250cam`** checkpoint (the same trained ckpt as the 2026-07-08
spectral run) + 90-episode real comma2k19 val cache. **Honesty (P8):** this is the **pre-reset**
9-ch ckpt (before the 2026-07-14 speed/scale reset). It has a fully-trained ImaginationField and a
real 2048 readout, so it is a valid *directional* substrate for both diagnostics; it is **not** the
operative flagship. Numbers are directional, not decision-grade — the falsifiers below are what carry.

Two measured experiments this run, both backlog items, both instrument-only (no trained-config change;
D-004 / D-018 respected):

- **E1** (backlog P0 #0b): does the H15 imagination field survive a **blind multi-step rollout**, and
  does its **epistemic σ** stay calibrated? — motivated by UWM-JEPA (2605.25313) + **"Biased Dreams"
  (2604.25416)**.
- **E2** (backlog P1 #3b): is the trained **2048 readout covariance orthogonal/isotropic** — the
  LeJEPA (2605.26379) condition the D-021 "optimal-planning" corollary rests on?

---

## E1 — Blind belief rollout: uncertainty dissipation + attractor collapse (CONFIRMED)

### Setup
The H15 `ImaginationField` is trained **one step only** (`h15_loss`: from a sector-masked frame,
predict the *next* frame's hidden-cell tokens). I roll it **fully blind** on real val frames: mask a
half-sector at `t`, encode belief₀, then for k=1…8 run `imagination(belief, vis)` and **feed the
imagined tokens back** as the next belief (no re-observation; sector stays hidden). Stride 6 frames
(~0.3 s/step → ~2.4 s at k=8). At each horizon I compare imagined hidden-cell tokens to the **true**
encoded tokens of the real future frame. Metrics on **centered** tokens (raw ViT tokens share a huge
DC component that saturates cosine ~1 for everything incl. the chance floor — centering exposes the
spatial structure). Two seeds; the trends are identical.

### Arms (hidden-cell centered cosine vs the true future tokens, seed 0, n=150 windows)

| horizon | **rollout** (fed back) | **freeze-1** (hold the k=1 imagination) | **persistence** (hold masked-frame tokens) | chance floor |
|--:|--:|--:|--:|--:|
| 1 | **0.357** | 0.357 | 0.235 | 0.030 |
| 2 | 0.219 | 0.251 | 0.238 | 0.006 |
| 3 | 0.109 | 0.257 | 0.239 | 0.016 |
| 4 | 0.011 | 0.254 | 0.230 | 0.014 |
| 5 | −0.029 | 0.244 | 0.230 | 0.007 |
| 6 | −0.035 | 0.234 | 0.229 | 0.016 |
| 8 | −0.047 | 0.223 | 0.226 | 0.010 |

### σ (epistemic log-variance) and attractor signals (seed 0)

| horizon | σ hidden (logvar) | σ visible | inter-sample cos (attractor) | belief energy | true energy |
|--:|--:|--:|--:|--:|--:|
| 1 | −7.79 | −7.99 | 0.206 | 0.101 | ~0.33 |
| 4 | −8.24 | −8.32 | 0.516 | 0.008 | ~0.33 |
| 8 | −8.55 | −8.66 | 0.570 | 0.010 | ~0.33 |

### Findings (seed-robust; seed 1 reproduces every trend)
1. **The 1-step imagination helps only at horizon 1.** At k=1 rollout beats persistence (0.357 vs
   0.235) *and* fixes the magnitude (rel-L2 5.1 vs ~100). From **k≥2 the recursively-rolled belief
   falls to/below persistence**, and by **k=4 it is at the chance floor (0.011)** and goes negative
   after. Autoregressing the field on its own outputs destroys the signal in ~3 steps.
2. **σ *dissipates* — the exact H11/D8 failure mode flagged 2026-07-15.** Hidden-cell log-variance
   **falls monotonically −7.79 → −8.55** as the prediction decays to garbage. The field becomes
   *more* confident precisely as it becomes *worthless* → **false confidence**. A σ-threshold
   self-monitor trigger (H11/D8) reading this signal at horizon > 1 would fire *less* exactly when it
   should fire *more*. (calib_gap hidden−visible stays ≈ +0.10 — the *spatial* ordering is right; the
   *temporal* magnitude moves the wrong way.)
3. **Attractor collapse — the "Biased Dreams" (2604.25416) prediction, measured.** Belief energy
   collapses **~11× by k=4** (0.101 → 0.008) toward a low-energy point, while inter-sample cosine
   **rises 0.21 → 0.57** — different samples' beliefs converge to a **common attractor**. True-token
   energy is flat (~0.33), so this is the *model* drifting, not the scene. This is literally the
   "latent transitions have attractor behaviour → rollouts drift to well-represented regions" claim.
4. **The fix is not more capacity at k=1 — it is the recursion.** **Freezing** the (good) k=1
   imagination and holding it retains **~0.25 cosine flat across all 8 horizons** and beats
   persistence at every step. So the 1-step prediction is *fine*; **feeding it back is what kills it.**

### Verdict & falsifier
The backlog 0b falsifier ("σ collapses with horizon OR R² drops as fast as no-imagination") is
**met — σ collapses (dissipates) AND rolled fidelity drops below the no-imagination baseline by k≥2**.
Read substantively (not "close the item" but "diagnose the cause"): **the 1-step-trained field cannot
be autoregressed for the operative K-step (K=4) horizon that H11/D8 self-monitoring assumes.** Two
mutually-exclusive design responses, both **D-018 tactics → escalate before touching the trained
config**:
- **(A) Train multi-step belief rollout** (the 0b build): supervise the *recursive* path with NLL at
  each k∈{1,2,4} so σ is forced to grow and the attractor is penalised. Highest-value, but a
  training-recipe change.
- **(B) Operate imagination in parallel-horizon (non-autoregressive) mode:** predict each horizon
  directly from the last *real* observation instead of feeding beliefs back — matches how the
  operative predictor already emits multiple horizons via parallel heads, and **freeze-1 shows this
  recovers ~0.25 flat fidelity for free.** Cheaper, no retrain, and it removes the σ-dissipation
  channel entirely at operative use. **Recommended default** pending the 0b training read.

Either way, **the H11/D8 self-monitor trigger must be capped at a 1-step (or parallel-horizon)
lookahead until a multi-step-trained σ is validated** — using the autoregressive σ at horizon > 1 is
currently anti-calibrated. This is a measured *constraint on H15's operative use*, not an H15 status
change (P8; pre-reset directional ckpt).

Artifacts: `Implementation/belief_rollout_diagnostic/blind_rollout.py` +
`results/2026-07-17-blind_rollout-seed{0,1}.json`.

---

## E2 — Readout orthogonality: a prior instrument already existed (unmerged) — I verified it instead of duplicating

### What happened (P8 / D-026 honesty)
LeJEPA (2605.26379, KB 2026-07-09) proves SIGReg **linearly *and orthogonally*** identifies world
latents, and that *"linear, orthogonal identifiability enables optimal latent-space planning."* The
D-021 spectral tool uses the *linear* half (fit R²≈0.99). Backlog 3b asks for the *orthogonal* half. I
started building the instrument — then found a **theoretically-superior one already built on 2026-07-10**
but **never merged** (branch `worktree-agent-arch-inf-20260710`, intake
`Implementation/incoming/2026-07-10-orthogonality-instrument/`). **I withdrew my draft and verified the
prior one** rather than ship a redundant, less-careful duplicate.

**Why the prior instrument is the correct one:** my draft measured **global** covariance isotropy over
all 2048 dims (isotropy≈0, off-diagonal≈0.999) and would have wrongly concluded "orthogonality fails."
The 2026-07-10 instrument explicitly warns this is the wrong number — for an **over-provisioned** readout
global isotropy is ~0 **by design** (dead tail); the theorem-relevant quantity is **isotropy within the
active subspace** (the energy-knee dims actually used). My draft lacked that distinction entirely.

### Verification (ran the prior `orthogonality_report` unchanged; step-6500 ckpt, n=2600 real states > S)
It **reproduces its 2026-07-10 number exactly:**

| metric | value | meaning |
|---|--:|---|
| active_k (energy-knee subspace) | 23 | matches spectral knee ≈22–31 |
| **iso_ratio_active** | **0.254** | = the logged "0.250"; < 0.5 → not yet isotropic |
| cond_number_active | 217.9 | anisotropic even within the active subspace |
| rms_offdiag_corr (active coords) | 0.424 | > 0.1 → coordinates still correlated |
| cov_effective_rank | 26.0 | matches spectral repr-rank ~tens |
| iso_ratio_global | 1.6e-8 | ~0 **by design** (over-provisioned), NOT a failure |
| **verdict** | **NOT-YET-ADMISSIBLE** | SIGReg isotropy not converged on this ckpt |

My independent **global** read (isotropy 0.000, off-diagonal 0.999, participation 5.0) **corroborates the
over-provisioning** from the coordinate-space angle — but the **active-subspace 0.254** is the correct
admissibility read.

### Findings
- **The LeJEPA "optimal-planning" corollary does NOT hold on the pre-reset ckpt** (iso_ratio_active
  0.254 < 0.5). State the **D-021 claim as "identifies a low-dim subspace,"** not "optimal latent-space
  planning" — an **admissibility constraint on claim language**, not an architecture change (G-AI1).
- Two independent instruments now agree the readout is **over-provisioned** (transition-operator rank ≈43;
  representation-covariance active rank ≈23–26 ≪ 2048) **and not yet orthogonal within its active
  subspace** — so latent *capacity* is not a D1 bottleneck (it is over-provisioned *and* under-isotropic).
- **Theory-to-practice gap:** SIGReg enforces isotropy in its **sliced 1-D projections**, yet the
  active-subspace covariance is anisotropic (cond 218, rms-offdiag 0.424) — slice-wise Gaussianity does
  **not** imply active-subspace isotropy here. A readout-whitening / orthogonality-penalty lever would
  restore the condition (D-018 escalate; one-lever bake-off, not executed).
- **Process finding:** the 2026-07-10 orthogonality instrument has been **stranded unmerged for a 3rd
  week** — flagged to the orchestrator for merge (it reproduces cleanly, has standalone tests, is the
  right instrument). See `Implementation/orthogonality_verification/`.

Artifacts: `Implementation/orthogonality_verification/` (README merge-recommendation +
`2026-07-17-verify-prior-orthogonality.json`). No competing intake shipped (duplicate withdrawn).

---

## Literature consumed (bounded sweep, 3 searches + 2 fetches, well under caps)
- **"Biased Dreams: Limitations to Epistemic UQ in Latent Space Models" (2604.25416)** — latent
  transitions show **attractor behaviour** biasing rollouts to well-represented regions ("discrepancy
  masking", reward overestimation) → uncertainty estimates untrustworthy for exploration. **Directly
  predicts E1's attractor collapse + σ dissipation; now measured on our stack.** New citation anchor.
- **UWM-JEPA (2605.25313)** — the belief-space prior whose spectrum-preservation *prevents* this
  dissipation; the design target if we build 0b.
- **VJEPA (2601.14354) / Var-JEPA (2603.20111)** — variational σ grounding; the principled route to a
  σ head that grows correctly under rollout (parked, H15 design watch).
- **JEPA generalization theory (2606.27014)** — JEPA pretraining as conditional spectral graph learning,
  bounds planning regret by pretraining error; ties the E2 orthogonality gap to a regret story (watch).
- Recency scan (cs.RO recent): **BadWAM (2607.15207) "dream right but act wrong"** — echoes 2512.24497
  (faithful unroll ≠ planning success); Phase-1 comparison watch. Benchmark/edge items → Bench-Eval /
  Prod-Opt per D-028. Ressources inbox: no such folder present (grep-verified).

## Actionable recommendations (each tied to a hypothesis + falsifying gate — G-A/G-B/G-AI1)
1. **Cap the operative H15 self-monitor lookahead at 1-step / parallel-horizon** until a multi-step σ is
   validated (H11/D8; falsifier: D8 AUROC). **D-018 escalate** — do not change the trained config here.
2. **Prototype 0b multi-step belief-rollout training** (NLL at k∈{1,2,4} on the *recursive* path;
   penalise attractor collapse). Target: σ grows with horizon, rolled fidelity ≥ freeze-1. Gate D9/D8.
3. **Re-run E1 + E2 on the operative flagship @30k** the moment it lands — the pre-reset caveat drops and
   the σ-dissipation / orthogonality reads become decision-grade (couples to the flagship verdict).
4. **Whitening / orthogonality bake-off lever** for the readout (E2): one-lever smoke first; falsifier =
   Δ within noise on D2/D1. **D-018 escalate.**
