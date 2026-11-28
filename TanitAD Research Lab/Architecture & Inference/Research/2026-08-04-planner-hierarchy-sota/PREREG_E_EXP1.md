# PRE-REGISTRATION — E-EXP-1: is the LONGITUDINAL axis the one our fan cannot reach?

**Written and content-pinned BEFORE any statistic below was computed.** Stream: arch-inf.
Date: 2026-08-04 (Europe/Berlin). GPU cost: **0** — dev-box CPU, banked artifacts only.

---

## 1. Why this experiment, and why it is not another re-ranking study

Every selection lever this programme has killed — rank-on-refined-confidence, the
candidate-conditioned consequence port, the route-readout graft, prior-corrected decoding,
factorisation alone, per-situation horizons, the `cond_imagination` port, and now
`S1_IS_DEAD_AFTER_ALL` (2026-08-04) — is **a re-ranking of a FIXED fan**. The measured bound is
K1/K2: a learned re-scorer recovers **≤ 8.4 %** of the 0.3075 m oracle gap, and the gap is
**~92 % irreducible** because it is a minimum over candidates scored against ONE realised future.

⭐ **That bound is a bound on RANKING. It says nothing about whether the answer is in the set.**
Three independent 2026 papers refine *off* the candidate set instead of ranking within it, and all
three factorise the refinement into an **along-path** and a **cross-path** axis:

| source | operator | class |
|---|---|---|
| **HAD** (arXiv 2604.03581) Structure-Preserved Trajectory Expansion | polar: radial scale **λ ∈ {0.92, 0.96, 1.00, 1.04, 1.08}** × angular offset **δ ∈ {−6°, −3°, 0°, +3°, +6°}** around each selected anchor | PUBLISHED |
| **AlignDrive** (arXiv 2601.01762) | predict the drive path, then predict **1-D displacement along it** | PUBLISHED |
| **TOAD** (arXiv 2606.07170) | CEM search in **control space (acceleration, yaw-rate)** | PUBLISHED |

**This experiment asks the necessary question before we spend anything on that family:**
when the fan is expanded by HAD's published operator, **does the LONGITUDINAL axis open more
reachability than the LATERAL axis?** Our own measurements say 88.7 % of the oracle gap and
87.6–89.9 % of the selection gap are longitudinal. If the axis-factorised expansion does NOT
preferentially open the longitudinal axis, the whole family is demoted for 0 GPU.

⚠️ **This is a NECESSARY, NOT SUFFICIENT condition, and it is stated that way in advance.** An
oracle over an expanded set measures **reachability**, never **findability**. LLM-Assist
(arXiv 2401.00125, Table 1) measured that handing PDM-Closed 8,505 proposals instead of 15 drops
its score 92.51 → 77.78 with TTC collapsing 93.11 → 62.89 — **a larger set is an adversarial
search against the scorer's error.** A PASS here therefore licenses a **regressed low-dimensional
residual** (AlignDrive's M=5 along-path anchors), never a search. HAD's own ablation is the same
warning: K=2 selected-then-locally-expanded scores 88.5–88.6 EPDMS, K=20 flat scores **79.8**.

---

## 2. Data, fixed in advance

- `…/Implementation/incoming/2026-08-03-s1-climbout/raw/fan_emitted_refc-base-30k.pt` — 128 anchors
- `…/Implementation/incoming/2026-08-03-s1-climbout/raw/fan_emitted_refc-xl-30k.pt` — 256 anchors
- Both: **881 canonical windows / 40 val episodes**, 4 waypoints, step 29999, `nav_mode=follow_constant`.
- Keys used: `fan`, `gt`, `sel`, `eid`, `v0`. Nothing else is read.
- ⛔ No episode re-selection of any kind. The parity invariant is untouched — this reads a banked
  dump and never touches corpus selection.

**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, unit = **episode**, `n_boot=2000`,
`seed=0`, `reduce="mean"`. ⛔ `overlapping_holdout_se` is never called.

**Frame convention, to be VERIFIED not assumed** (reported in the output): along-track = ego-forward
axis at t0, cross-track = lateral axis at t0. The runner asserts that GT forward displacement is
predominantly positive and correlates with `v0`; if that assertion fails the run reports
INSTRUMENT-FAIL and no verdict is issued.

---

## 3. Arms — degrees of freedom matched by construction

Let `f[i,j]` be candidate *j* of window *i*. HAD's operator, applied about the t0 origin:

- **L (longitudinal)**: `f · λ`, λ ∈ the published 5-point grid, δ = 0. → N × **5** candidates
- **A (lateral)**: `rot(f, δ)`, δ ∈ the published 5-point grid, λ = 1. → N × **5** candidates
- **LA (both)**: the full 5 × 5 grid → N × **25** candidates

⭐ **L and A have exactly 5 grid points each.** The comparison is DoF-matched by using HAD's own
symmetric grid rather than grids I chose, which is the reason the published operator is used
verbatim instead of a tuned one.

Reported per arm: `ade_oracle_*` = min over the expanded set of mean-over-4-waypoints L2 error.
Baselines: `ade_sel` (shipped selection) and `ade_oracle_fan` (min over the unexpanded fan).

---

## 4. PRIMARY read + DIRECTION predicate

**Δ_primary = mean(`ade_oracle_A`) − mean(`ade_oracle_L`)**, paired episode-cluster bootstrap.

Positive Δ ⇒ the longitudinal expansion reaches lower error than the lateral one at equal DoF.

**Three-sided verdict table, committed now:**

| outcome | verdict | what we do |
|---|---|---|
| **Δ > 0, CI excludes 0** | **PASS-L** | The fan's binding shortfall is on the along-path axis and a *local* correction reaches it. Fund an **AlignDrive-style 1-D along-path displacement head** (M=5 longitudinal anchors, regressed not searched). Proceed to E-EXP-2 (learnability + v0-echo control) BEFORE any training. |
| **Δ < 0, CI excludes 0** | **PASS-A** | ⛔ **My ranking is wrong.** The lateral axis is the reachable-set shortfall despite 88.7 % of the *error* being longitudinal. Report the contradiction prominently, re-rank, and open a retraction row — a longitudinal-first lever would have been funded on a false premise. |
| **CI includes 0** | **NOT SEPARATED** | The axis-factorised local expansion does not preferentially open the longitudinal axis. **Demote the entire HAD/AlignDrive/TOAD refinement family for us**, and strengthen the K7 reading that the along-track quantity is absent from the representation rather than absent from the parameterisation. |

**Secondary (reported always, decides nothing on its own):** `ade_oracle_fan − ade_oracle_LA`, the
total reachability opened; and the along-/cross-track decomposition of every reduction, so the
LONGITUDINAL family is read directly rather than inferred from a scalar ADE.

---

## 5. ⭐ The control that separates a 1-PARAMETER fix from a HEAD

A per-window oracle λ could be explained trivially: our arms have a **systematic** speed bias
(MEASURED: +0.66 m/s speed over-prediction on flagship at 30k). If so the fix costs **one scalar**,
not a head.

**Arm L-global:** fit a SINGLE λ over episodes, apply it to the **shipped selected trajectory** —
no oracle anywhere — and compare `ade(λ_global · sel)` against `ade(sel)`.

⛔ **Fitted leave-one-episode-out (LOEO), never in-sample.** λ is fitted on 39 episodes and applied
to the held-out 40th, so the reported number is genuinely out-of-episode.

| outcome | reading |
|---|---|
| L-global separably improves `ade_sel` | a **1-parameter deployable calibration** exists. Ship the calibration first; it is strictly cheaper than any head and it must be the baseline any head is then measured against. |
| L-global does not separate, but the per-window oracle λ does | the correction is **genuinely per-window** ⇒ a head is required, and its target is well posed. |
| neither | the λ axis is not the lever. |

---

## 6. What this experiment explicitly CANNOT say

1. **Nothing about findability.** Oracle arms use the ground-truth future. No number here may be
   quoted as an achievable improvement, and none may size a training run on its own.
2. **Nothing about closed-loop.** These are banked open-loop fans on 881 windows; open-loop does
   not predict closed-loop (MEASURED, ours: 0.45 m open-loop → 1.69 m closed-loop).
3. **Nothing about the other three metric families** beyond the along/cross decomposition reported.
   TACTICAL and STRATEGIC are untouched by this instrument and are reported as NOT-APPLICABLE with
   the reason, per the binding rule's clause 5.
4. **It is two REF-C arms only** — it is not a statement about the flagship, which has no fan at all
   (`anchor_decoder is None`, four unimodal `Linear(d,2)` heads).

---

## 7. INSTRUMENT-FAIL branch (C63's lesson)

The run reports INSTRUMENT-FAIL and issues **no verdict** if any of:
- the frame-convention assertion in §2 fails;
- `ade_oracle_fan` does not reproduce the banked oracle-in-fan value for the arm;
- λ = 1.00, δ = 0° does not reproduce the unexpanded fan **bit-identically**;
- window/episode counts differ from 881 / 40.
