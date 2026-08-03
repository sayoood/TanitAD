# PRE-REGISTRATION — D-LATENT: is the LATENT the bottleneck for `long_accel`, or is the VIDEO?

**Registered** 2026-08-03, BEFORE any arm was fitted. **Owner** latent/world-model-representation
stream. **Run directory**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck/`.
**Substrate** dev box RTX 4060, **0 pod GPU-h**. Nothing here touches a training pod.

*(Both outcomes and their thresholds are fixed in this file. The result file must cite this
registration by name and must state which outcome fired, including the one that kills the thesis.)*

---

## 1. THE CLAIM UNDER TEST

The programme's current reading is a **latent-bottleneck thesis**: `long_accel` is missing from the
model's decisions because the *representation* does not carry it. Four results are cited for it:

| # | result | source | class |
|---|---|---|---|
| 1 | `long_accel` not separated from a shuffled-latent control across 17 head architectures | `…/Benchmarks & Eval/…/2026-08-03-idm-accel-recoverability/results_accel_recoverability.json` | MEASURED |
| 2 | sitclf capacity curve peaks at 129 parameters; a 2 049-param ridge beats a 2.17 M head | sitclf stream | INHERITED |
| 3 | REF-C's manoeuvre head is READOUT-limited (`auc_lon_active` 0.7294) | REF-C stream | INHERITED |
| 4 | REF-C keeps only the LAST frame's feature map for the anchor cross-attention | `stack/tanitad/refs/refc.py:1412-1422` | MEASURED (read from source) |

⚠️ **Result 1 already refutes the naive version of the fix.** Its `window` basis is all **nine**
frames' latents (18 432 features) and it lands at the null. So "keep W frames instead of the last"
is **already measured to be insufficient at the POOLED-latent level**. The thesis therefore has to
be the sharper one: *the frozen v1 latent has no target-correlated temporal content to keep.*

**That sharper thesis has never been falsified, because no arm in this programme has ever probed a
representation that is temporal by construction.** Every arm to date probes the same frozen v1
latent. A null from one representation cannot distinguish:

* **L (LATENT-LIMITED)** — the information is in the video and the v1 encoder destroys it
  ⇒ the lever is the representation (approaches A/C);
* **V (VIDEO-LIMITED)** — the information is not in monocular 10 Hz 256 px video at this n at all
  ⇒ **no encoder can fix it**, the lever is a measurement channel or a different sensor, and the
  latent-bottleneck thesis is FALSE for this channel.

L and V point at opposite work and cost different orders of magnitude. This is the cheapest
discriminating experiment between them.

## 2. DESIGN

**Substrate — held identical to the panel the null came from, so the contrast lands on the same
windows.** 50 content-clean comma2k19 episodes `ep_00040…ep_00089`; `k=4` (9-frame, NON-CAUSAL)
windows at stride 2; episode-disjoint split `i % 3` → 33 train / 17 held-out; inner split of TRAIN
only (22 fit / 11 select) for every hyperparameter; the held-out episodes are scored once.
Labels re-derived from `poses`/`actions` with the same heading repair.

**The new rung: a PIXEL substrate.** The same windows, but the "representation" is the raw video
itself, average-pooled to a small grid. It is by construction (a) purely visual, (b) legal at
inference under the vision-only rule, (c) temporal — nothing has been trained to discard motion.

| substrate | per-frame dim | what it is |
|---|---:|---|
| `v1` | 2 048 | frozen flagship-v1 encoder + spatial-grid readout, step 29999 (the banked cache) |
| `pix32` | 1 024 | 32×32 grayscale, exact 8×8 average pool of the 256×256 frame |
| `pix8` | 64 | 8×8 grayscale — a coarse motion-energy grid |
| `pix1` | 1 | whole-frame mean intensity — the parsimony rung |

**Arms.** Every arm is the SAME closed-form protocol as the panel that produced the null: exact
kernel ridge in the dual over the full regularisation path (α ∈ 10⁻⁴…10¹⁰, 29 values, **plus an
exact-mean sentinel**), per-output selection on the inner split, **skill gate 0.01** → train mean,
one-SE shrinkage tie-break, refit on full train, scored once on held-out. No learning rate, no epoch
budget, no initialisation can be blamed for any outcome.

Feature bases per substrate: `centre` (single-instant), `window` (all 9), `diff` (symmetric),
`tdiff` (adjacent first differences), `abstdiff` (motion energy).
⚠️ `tdiff`/`abstdiff` are **added to the instrument for this run**
(`stack/tanitad/eval/accel_probe.py`) because the existing `diff` basis spans lags 2/4/8 and can
never form the **1-step** difference — the wrong stencil for a per-step derivative.

**Controls — every one of them binding:**

1. **A matched SHUFFLED control for EVERY arm** (row permutation of the substrate; marginals, targets,
   split and recipe untouched). All verdicts are the **paired** ΔR² vs that arm's own control.
2. **Positive control** — `speed` must separate positive on the same arm in the same draw, or that
   arm's `long_accel` null is inadmissible.
3. **The EMPIRICAL NULL as a first-class arm** (constant = train mean). R² = 0 is not the null:
   held-out `long_accel` mean −0.1305 vs train +0.0163 makes the no-information score ≈ **−0.0626**.
4. **The ORACLE-INPUT arm** (ridge on the TRUE 9-position speed window) must reach ≈ +0.93, or the
   protocol is broken and no null from it is readable.
5. **⭐ The SINGLE-INSTANT control, `pix32/centre`.** This is REF-C's mechanism reproduced in pixel
   space: one frame, no motion. If the temporal pixel bases separate and this one does not, the
   effect is TEMPORAL. If this one ALSO separates, there is a static-appearance shortcut and the
   temporal claim is not established (see outcome M).
6. **A planted-signal sensitivity floor on the pixel substrate**, so a pixel-side null is quotable
   with a magnitude instead of as an absence.

**Estimator.** Paired episode-cluster bootstrap over the held-out episodes,
`taniteval.ci.paired_episode_cluster_bootstrap` / `tanitad.eval.ap_ci.stat_episode_cluster_bootstrap`,
B = 2000. ⛔ `overlapping_holdout_se` is not used anywhere.

**Reporting.** All four metric families (LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC) per arm,
per-family, never pooled; ADE is one row and never the result. A family with no instrument on this
corpus is reported **UNAVAILABLE with its reason and its n**, as a work item.

## 3. THE PRE-REGISTERED READING — fixed now, in advance

Let Δ = paired ΔR²(`long_accel`) of an arm against **its own** shuffled control, and R²ₕ = its
held-out `long_accel` R².

### OUTCOME L — LATENT-LIMITED (thesis SUPPORTED)
**Fires iff:** at least one **temporal** pixel arm (`tdiff` / `abstdiff` / `window` / `diff` on any
pixel substrate) has **Δ separated positive** (paired CI excludes zero, lower bound > 0) **AND**
R²ₕ ≥ **+0.05**; **AND** `pix32/centre` (the single-instant control) does **not** separate positive;
**AND** every `v1` arm stays unseparated; **AND** `speed` separates positive on the same arms.

⇒ **The information is in the video and the frozen v1 latent destroys it.** The lever is the
representation. Approach A (make the cross-attended tokens temporal) and approach C (pretrain the
encoder for dynamics) are both live; approach B (augment without retraining the trunk) is worth its
cost only if it can inject the same quantity. **The magnitude of R²ₕ becomes the headline budget:
that is how much of the channel a *better latent* could in principle recover.**

### OUTCOME V — VIDEO-LIMITED (thesis FALSIFIED for this channel)
**Fires iff:** **no** pixel arm reaches Δ separated positive with R²ₕ ≥ +0.05, while the ORACLE-INPUT
arm still reaches ≈ +0.93 and `speed` still separates positive on the pixel arms.

⇒ **`long_accel` is not recoverable from monocular 10 Hz 256 px video at this n and above the
measured sensitivity floor.** A better latent cannot be argued to fix it, and any proposal that
promises to must be refused. The programme must then either (i) accept `long_accel` as a
label-derived quantity from a speed track (already measured recoverable at R² 0.9262 from the TRUE
speed window by a **9-feature** ridge), or (ii) attack the frame rate / resolution / sensor, not the
encoder. **⚠️ This outcome retires items 1–4's shared "the latent is the bottleneck" framing for
`long_accel` specifically. It does NOT retire it for `speed`, TTC, headway or the manoeuvre
decision, which must then be argued on their own evidence.**

### OUTCOME M — MIXED / SHORTCUT (no verdict)
**Fires iff:** `pix32/centre` — a **single frame** — also separates positive on `long_accel`.

⇒ A static-appearance shortcut exists (visible hood/dashboard, scene-specific memorisation, or a
label artefact correlated with appearance). **Neither L nor V may be claimed.** The required next
step is named now: re-run with the shortcut audited (per-episode leave-one-out, and an arm on a
frame with the bottom 25 % of rows masked), before any architecture conclusion.

### Ties and edge cases, decided in advance
* An arm separated positive with **R²ₕ < +0.05** counts as **evidence too weak to license a
  retrain** — it is reported, but it does NOT fire outcome L. It fires a "weak-L" note only.
* If the ORACLE-INPUT arm does **not** reach ≥ +0.80, the run is VOID and no outcome may be quoted.
* If `speed` fails to separate on the pixel arms, the pixel substrate is inadmissible as a probe and
  the run is VOID for the pixel family.
* Any arm whose two predictions are numerically identical to its control is forced
  `separated=False` by the instrument's degeneracy guard, and that is respected.

## 4. WHAT WOULD MAKE ME WRONG, STATED NOW

* If outcome **V** fires I will write it as the headline of the deliverable, above every ranked
  approach, and mark the affected rungs of the ranking as **not supported by measurement**.
* The ranked approach list in the deliverable must be readable under BOTH outcomes: each approach
  carries the outcome under which it is worth its cost.
* No number in the deliverable may be quoted without its evidence class, and no approach may be
  ranked on an INHERITED number alone.

## 5. WHAT THIS RUN DOES NOT ESTABLISH (limits, fixed in advance)

* **One corpus.** comma2k19 highway, 50 episodes, 17 held out. Nothing transfers to PhysicalAI-AV or
  to v5f's 429-token 176×624 geometry without re-running, and it must not be quoted as if it did.
* **One channel.** `long_accel`. The lateral and strategic families are measured but are not what
  this design is powered for.
* **A pixel probe is a lower bound on what a trained temporal encoder could carry, not an upper
  bound.** A positive is decisive; a null is bounded by the reported sensitivity floor.
* **n = 17 held-out episodes** is the independent unit and sets the interval width.
