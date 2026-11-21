# RUNG 1 — THE PLANNER-ACTION SWEEP: what the model's own actions do to the rollout, and how much is recoverable without retraining

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC). **Stream:** blind-imagination driving, T_blind ladder.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any number here existed**
(its mtime precedes every file in `artifacts/`). It is **not edited**; every deviation is an
amendment in §8.
**Host:** pod2 only (A40, MEASURED idle before launch: `0 MiB, 0 %`). ⛔ pod1 (training), pod3
(situation-classifier build) and the eval pod were **never connected to**; the val cache was read
only. Disk verified with a **real 2 GB `dd` write** (1.9 GB/s, full 2,097,152,000 bytes, no short
write; `/root` holding 270 MB) — never `df`.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Estimator, everywhere:** **paired episode-cluster bootstrap** — `taniteval/ci.py`, **B = 2000,
seed 0**, unit = **episode cluster**, identical windows for every arm. `overlapping_holdout_se`
appears nowhere in this folder.
**Arm:** v1 = `flagship4b-speedjerk-30k` @ step **29999** (`MODEL_REGISTRY.md` §1.2) ·
**599 windows / 596 episode clusters** · K = 185 · calibrated `str` (k = 20) readout throughout.

---

# 0. VERDICT

> ## ⭐⭐ **CONFIRM. Damping the model's own action toward the last observed one — no retraining, no weight change, one filter on the action tensor — takes deployable `T_blind` from 2.5 s to 11.6 s [11.6, 17.2]. A 12.5 % damping, with the model still supplying 87.5 % of the command, already MORE THAN DOUBLES it (25 → 55 steps). The registered +9.0 s ceiling is not merely approached, it is reached: `frac_of_ceiling_recovered = 1.011`.**
>
> ## ⭐⭐ **AND THE CAPABILITY CAP BREAKS — the first time in this program. Rung 0 published its 2.5 s at DECISION-GRADE but capped capability at PROVISIONAL because the deployable arm NEVER beat constant velocity (0/185) and `T_useful@1m` was stuck at 1.4 s. At α = 0.25 — the model still supplying three quarters of the command — the arm is separated-BETTER than constant velocity over 1.2 – 5.4 s (43/185) and `T_useful@1m` rises to 1.9 s. At α = 0.75: 81/185 over 0.7 – 8.7 s and 2.3 s. Both pre-registered capability bars fire.**
>
> ## 🔴 **THE MECHANISM IS THE LONGITUDINAL COMMAND'S AMPLITUDE, AND NOTHING ELSE. The model's own acceleration command averages 2.058 m/s² over the first 0.5 s and sits AT the ±3 m/s² clamp 46.4 % of the time; the SAME inverse map fed TRUE motion gives 0.539 m/s² and 0.53 % — 3.8× the magnitude, 87× the saturation. Its mean is −0.018 m/s² against a mean magnitude of 1.09, so it is a near-zero-mean CLAMP-SATURATING OSCILLATION, not a drift. The steering channel is innocent: `steer_clip` moves nothing at any setting.**
>
> ## ⛔ **AND THE MOST USEFUL RESULT IS A REFUTATION OF MY OWN PRE-REGISTERED SIGNATURE. Reducing the action-update frequency — the classic de-jitter — is CATASTROPHIC: every 2 / 5 / 20 steps all give 9 steps of `T_blind`, far BELOW the 25-step baseline, with `de@2s` degrading 1.82 → 2.81 / 3.99 / 4.63 m. Zero-order-holding a sample of a zero-mean ±3 m/s² oscillation removes its cancellation. What matters is the oscillation's AMPLITUDE, not its presence — and my pre-registration had bundled `ema` and `every` into one signature as though they were the same operation. They are not.**

| what the brief asked | the answer, `MEASURED` |
|---|---|
| *What about the model's own action sequence destroys the rollout?* | **The amplitude of its ACCELERATION command.** It is produced by a one-tick finite difference of the model's own decoded speed (`(v−v_prev)/0.1 s`), which turns decode wobble into a command that saturates the ±3 m/s² clamp 46.4 % of the time and whose **mean magnitude over the first 0.5 s (2.058) exceeds the maximum \|accel\| present in the corpus (1.9, `INHERITED` from `closedloop.py:155`)**. |
| *Compounding drift?* | ✅ **CONFIRMED.** Penalty exponent **2.098** (R² 0.995, n 19, window 2–20) and **1.346** (R² 0.997, n 166, window 20–185). A level shift would be ≈ 0. Damage accrues **in proportion to time-in-loop**, not in a burst. |
| *Action-magnitude blow-up?* | ✅ **CONFIRMED, and binding.** Amplitude-reducing filters recover monotonically; `accel_clip` 3.0→1.0→0.3→0.0 gives 25→47→62→78 steps, all separated. `steer_clip` gives 24/25/27, **none separated**. |
| *Feedback instability?* | ⚠️ **The oscillation is real (sign-flip 0.289/tick, jitter 3.15 m/s²) but my pre-registered signature is REFUTED** — the de-jitter intervention makes it 2.8× worse. |
| *Horizon dependence — onset or linear?* | ⛔ **No onset knee.** Recovery is monotone in switch time in both directions. |
| *How much is recoverable without retraining?* | **All of the registered ceiling and marginally more (101.1 %),** and the honest deployable recommendation is **α ∈ [0.25, 0.5]**, where the model still supplies 50–75 % of the command and the arm beats constant velocity. |

**Tier.** The **`T_blind` number is DECISION-GRADE**: pre-registered with buckets fixed in advance,
primary statistic and estimator named in advance, comparator matched in **both** the readout and the
action filter, window-set identity gated, validated in both directions, and with a falsifier
**demonstrated to fire on real arms in this very sweep** (§2.4). The **capability claim is CONFIRMED
but not DECISION-GRADE** — the bar was pre-registered but the specific α is selected in-sample from
18 eligible arms (§6.2 shows the conclusion does not depend on that selection: **7 of 18** eligible
arms clear the capability bar and the dose-response is monotone).

---

# 1. Pre-registration, and what it fixed before any number existed

`PRE_REGISTRATION.md` fixed, before any computation: the primary statistic; the two MEASURED
endpoints and the 9.0 s budget between them; the window-set identity gate; the plumbing self-test in
both directions; the seven intervention families and the **eligibility rule**; the three outcome
buckets and the multiplicity handling; the capability cap; the failing values; and a
**confirm/refute signature for each of the four candidate mechanisms**.

Two things it declared in advance rather than discovering late:

* ⚠️ **`own_kinematic` is one action policy, not "the" deployable policy, and this rung does NOT run
  v1's tactical planner** (`closedloop.wp_to_control`). The ladder's R1 row names the planner
  variant; this sweeps the **filter** axis, which is strictly cheaper and interpolates the two
  *measured* endpoints. §7.2 states what the filter result predicts for the planner, on the record.
* ⚠️ **The blow-up hypothesis testable here is SATURATION against an existing clamp, not unbounded
  actions** — `kinematic_action_from_dpose` already clamps `steer ∈ [−0.05, 0.05]` and
  `accel ∈ [−3, 3]`. The v5 stream's 108.7 m / 181 km/h fan (`INHERITED`,
  `V5_IMAGINATION_SELECTION.md` §2.2, `raw/v5_posthoc.json:per_window_span_mean_m = 108.736`) is a
  *plan-space* observation; this is an *action-space* one and is worded that way.

---

# 2. The gates — nothing below was read until all of them passed

`artifacts/rung1_gates.json`, rendered from raw JSON in `artifacts/_tables.md`.

## 2.1 ⛔ Window-set identity

| check | result |
|---|---|
| windows | **599 new vs 599 committed** ✅ |
| episode clusters | **596** |
| `eid` ordering identical vs **both** committed dumps | ✅ |
| `t0` ordering identical | ✅ |
| anchor `a_imagination__own__roSTR` (in `bi_perwindow_compact.pt`) | max \|Δ\| = **0.0 m** ✅ |
| anchor `a_imagination__hold__roSTR` (same) | max \|Δ\| = **0.0 m** ✅ |
| anchor `b_frozenlast__own__roSTR` (in `perwindow_matched_K185.pt`) | max \|Δ\| = **3.05e-05 m** ✅ (tol 1e-4) |
| anchor `b_frozenlast__hold__roSTR` (same) | max \|Δ\| = **3.05e-05 m** ✅ |

⭐ The two `a`-arm anchors reproduce at **exactly 0.0** — bit-identical, not merely within tolerance.
The two `b`-arm anchors carry the same **3.05e-05 m** float-kernel noise Rung 0b measured between two
encode passes, which is the expected signature and is what the tolerance exists for.

## 2.2 ⭐ The plumbing self-test — in BOTH directions, at the full K = 185

A filter knob that is silently a no-op produces a flat, confident, wrong curve. The blend axis'
endpoints reduce **algebraically** to arms that already exist, so this is checkable exactly:

| direction | test | required | result |
|---|---|---|---|
| fidelity (lower endpoint) | `own_kinematic\|blend=0.0` vs the unfiltered own arm | max \|Δ\| = 0.0 | ✅ **0.0 — BIT-IDENTICAL** |
| fidelity (null knob) | `own_kinematic\|every=1` vs the unfiltered own arm | max \|Δ\| = 0.0 | ✅ **0.0 — BIT-IDENTICAL** |
| fidelity (upper endpoint) | `own_kinematic\|blend=1.0` vs `hold_last` | max \|Δ\| = 0.0 | ✅ **0.0 — BIT-IDENTICAL** |
| **anti-no-op** | every filter arm must differ from `own` | non-empty change | ✅ arms identical to `own`: **`[]`**; smallest max \|Δ\| across all **32** filter arms: **226.67 m** |

> ### ⭐ **This is the load-bearing gate of the whole rung. Because α = 0 and α = 1 are BIT-IDENTICAL to the two independently-measured endpoint arms, every interior point of the blend curve is interpolating between REAL ARMS, not between two reimplementations. The curve is trustworthy for that reason and no other.**

The same pair of tests is pinned on a CPU fixture in `taniteval/tests/test_blindimag.py` (26 new
tests; full `taniteval` suite **449 passed**), including the anti-no-op assertion for all nine knobs.

## 2.3 Fidelity against the ladder's committed numbers

Every headline the ladder published for these two arms reproduces from this run's fresh encode pass:

| quantity | committed | recomputed here |
|---|---:|---:|
| `T_blind` own \| `str` | 25 steps [2.5, 3.9] | **25 steps [2.5, 3.9]** |
| `T_blind` hold \| `str` | 115 steps [11.5, 17.4] | **115 steps [11.5, 17.4]** |
| `de@2s` own / hold | 1.8165 / 0.6718 | **1.8165 / 0.6718** |
| `ade_0_2s` own / hold | 0.8710 / 0.3351 | **0.8710 / 0.3351** |
| `T_useful@1m` own / hold | 1.4 s / 2.3 s | **1.4 s / 2.3 s** |

`LEVEL_FIDELITY_PASS = True` · `T_BLIND_EXACT_REPRODUCTION = True` · `T_useful_reproduces = True`.
⚠️ The level agreement was pre-declared **blocking**; the `T_blind` integer reproduction was declared
**reported, non-blocking** (a step count is a threshold crossing and a 3e-05 m shift could in
principle move it). It did not need the latitude.

## 2.4 ⚠️ THE C13 CHECK — the failing values, and proof they fired

Two predecessors shipped diagnostics that could not fire. This rung was told not to add a third.

**(a) The primary statistic's failing value is 1 step (0.1 s), not 0**, returned when the first
evaluable horizon already fails. Probe on real arms: **identical arms → 1 step; swapped arms → 1
step.** ✅

**(b) ⭐ The VERDICT rule's failing bucket fired on real ELIGIBLE arms in this very sweep.** This is
not a synthetic demonstration:

| eligible arm | `T_blind` | vs the 25-step baseline | bucket it would have produced |
|---|---:|---|---|
| `every2` | **9 steps** | **−21, not separated** | ⛔ REFUTE (≤ 30) |
| `every5` | **9 steps** | **−22, not separated** | ⛔ REFUTE |
| `every20` | **9 steps** | **−22, not separated** | ⛔ REFUTE |
| `own_vupd` | **9 steps** | **−21, not separated** | ⛔ REFUTE |
| `steerclip0.02` | **24 steps** | **−1, not separated** | ⛔ REFUTE |
| `steerclip0.005` | 25 steps | +2, not separated | ⛔ REFUTE |

⇒ **Had the sweep contained only the `every`, `steer_clip` and speed-channel families, the
pre-registered verdict would have been REFUTE.** Six of eighteen eligible arms land in the failing
bucket. The rule discriminates.

**(c) Diagnostic vacuity audit** (`artifacts/rung1_gates.json → diagnostic_vacuity_audit`), declared
in advance and emitted as an artifact:

| diagnostic | admissible? | why |
|---|---|---|
| `frac_draws_T_blind_is_zero` | ⛔ **NO** | structurally 0 under A4 — **not emitted anywhere in this folder** |
| `blend0.0 == own` **alone** | ⛔ **NO** | satisfied by a no-op; admissible only paired with the anti-no-op check, which is also run |
| `frac_draws_T_blind_at_floor_1step` | ✅ yes | both 0.000 and 1.000 attainable — Rung 0 measured 1.000 on its unmatched contrast. **It is 0.000 for every arm here; that is a result, not a tautology.** |
| `T_useful` | ✅ yes | returns 0.0 when step 1 is already above the bar |
| anti-no-op | ✅ yes | a zero-difference arm is attainable and would fail |

---

# 3. ⭐ THE BLEND CURVE — the priority deliverable

`a_fed = (1−α)·a_own + α·a_hold0` on the (steer, accel) channels, where `a_hold0` is the **last
OBSERVED action** — causally available at rollout start, not future information. Every row has a
comparator matched in the readout **and** in the action filter.
`artifacts/rung1_blend_curve.json`.

| α | model's share of the command | `T_blind` | CI95 | paired gain vs own | `de@2s` | `ade_0_2s` | **beats CV** | `T_useful@1m` |
|---:|---:|---:|---|---|---:|---:|---|---:|
| **0** ⛔ endpoint | 100 % | **25** (2.5 s) | [2.5, 3.9] | — | 1.8165 | 0.8710 | ⛔ **0/185** | 1.4 s |
| **0.125** ✅ | 87.5 % | **55** (5.5 s) | [5.5, 8.4] | **+36** ✅ | 1.3755 | 0.6793 | ⛔ 0/185 | 1.6 s |
| ⭐ **0.25** ✅ | 75 % | **85** (8.5 s) | [8.5, 13.1] | **+74** ✅ | 1.0736 | 0.5440 | ✅ **43/185 (1.2–5.4 s)** | **1.9 s** |
| **0.375** ✅ | 62.5 % | **101** (10.1 s) | [10.1, 15.9] | **+94** ✅ | 0.9103 | 0.4611 | ✅ 62/185 (0.8–6.9 s) | 2.0 s |
| **0.5** ✅ | 50 % | **111** (11.1 s) | [11.1, 16.9] | **+105** ✅ | 0.7924 | 0.4010 | ✅ 72/185 (0.8–7.9 s) | 2.2 s |
| **0.625** ✅ | 37.5 % | **115** (11.5 s) | [11.5, 17.2] | **+109** ✅ | 0.7184 | 0.3633 | ✅ 78/185 (0.7–8.4 s) | 2.3 s |
| ⭐ **0.75** ✅ **best** | 25 % | **116** (11.6 s) | **[11.6, 17.2]** | **+109** ✅ | 0.6842 | 0.3437 | ✅ **81/185 (0.7–8.7 s)** | 2.3 s |
| **0.875** ✅ | 12.5 % | **116** (11.6 s) | [11.6, 17.3] | +109 ✅ | 0.6724 | 0.3362 | ✅ 82/185 (0.7–8.8 s) | 2.3 s |
| **1** ⛔ endpoint (no policy) | 0 % | **115** (11.5 s) | [11.5, 17.4] | +109 ✅ | 0.6718 | 0.3351 | ✅ 83/185 (0.6–8.8 s) | 2.3 s |

**Shape.** Strictly increasing over α ∈ [0, 0.75] (25 → 55 → 85 → 101 → 111 → 115 → 116), then flat
within one step (116 / 116 / 115 — far inside CI widths of ±5.6 s). **Spearman(α, `T_blind`) = 0.95.**
⚠️ `monotone_nondecreasing` is reported as **False** because of that terminal 116 → 115 step; the
honest description is **monotone to α = 0.75, then indistinguishable**, and it is stated that way
rather than rounded into a monotonicity claim.

> ### ⭐ **Why the shape matters more than the maximum.** A best-of-N maximum over 18 arms is biased upward. A **dose–response between two endpoints that are BIT-IDENTICAL to independently measured arms** is not: a selection effect cannot manufacture a monotone rise from 25 to 116 across seven interior points, each with its own matched comparator and each separated from the baseline (`frac_draws_gain > 0 = 1.000` at every α ≥ 0.125). This was the pre-registered primary evidence and it is what the verdict rests on.

**The two rows that matter operationally are the low-α ones**, where the model is still doing the
driving: **α = 0.125 already more than doubles `T_blind`** with the model supplying 87.5 % of the
command, and **α = 0.25 is where the constant-velocity floor first breaks** with the model still
supplying 75 %.

---

# 4. Mechanism separation — all four, with the outcome that would have refuted each

`artifacts/rung1_mechanism.json`, `artifacts/rung1_action_signature.json`. Action statistics are
reconstructed over all 599 windows from `psi`/`pred_speed`/`v_last` — a reconstruction **proved, not
assumed**: pinned on a CPU fixture in `test_blindimag.py` and re-verified on pod2 against the real
model's `fed_actions` (`max abs diff 3.76e-07` steer / `2.38e-07` accel), with both failing
directions firing (`gtkin`, which derives its action from true motion, correctly does **not**
reproduce — off by 0.0824 rad; a deliberately wrong speed input is off by 6.0 m/s²).

## 4.1 ✅ Compounding drift — CONFIRMED

The comparator-free penalty `de_own − de_hold`:

| step | 2 | 5 | 10 | 20 | 40 | 80 | 120 | 185 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| penalty (m) | 0.008 | 0.066 | 0.274 | **1.145** | 3.272 | 8.796 | 15.031 | 23.979 |
| own / hold | 1.09 | 1.62 | 2.44 | **2.70** | 2.01 | 1.59 | 1.44 | 1.31 |

| fit window | exponent | R² | n | admissible (R² ≥ 0.80) |
|---|---:|---:|---:|---|
| steps 2–20 | **2.098** | 0.995 | 19 | ✅ |
| steps 20–185 | **1.346** | 0.997 | 166 | ✅ |
| steps 2–185 | 1.559 | 0.984 | 184 | ✅ |

A pure level shift gives exponent ≈ 0; independent per-step noise accumulating as a random walk in
velocity gives ≈ 1.5. **2.098 over the first 2 s is the double-integration signature.** The
*relative* penalty peaks at **2.70× at 2 s** and decays to 1.31× by 18.5 s — worst exactly where the
program quotes its canary.

## 4.2 ✅ Action-magnitude blow-up — CONFIRMED, and it is the binding mechanism — in the ACCELERATION channel only

Reconstructed over 599 windows (cumulative to the step shown):

| arm | step | mean \|steer\| | mean \|accel\| | frac steer at clamp | **frac accel at clamp** | jitter(accel) |
|---|---:|---:|---:|---:|---:|---:|
| **own** | 5 | 0.00978 | **2.0582** | 0.0928 | ⛔ **0.4641** | 3.1524 |
| **own** | 20 | 0.01110 | **2.0057** | 0.1011 | ⛔ **0.4580** | 2.8091 |
| **own** | 185 | 0.01337 | 1.0892 | 0.1331 | 0.2074 | 1.4125 |
| `gtkin` — **the same inverse fed TRUE motion** | 5 | 0.00984 | ✅ **0.5387** | 0.0805 | ✅ **0.0053** | 0.2976 |
| `gtkin` | 185 | 0.01075 | 0.5028 | 0.0830 | 0.0040 | 0.1031 |
| hold | 20 | 0.01155 | 0.9341 | 0.1235 | 0.1171 | 0.9891 |

> ### 🔴 **The model's own acceleration command is 3.8× the magnitude and 87× the clamp-saturation rate of the SAME inverse map fed true motion. Its MEAN over the first 0.5 s (2.058 m/s²) exceeds the MAXIMUM \|accel\| present in the corpus (1.9 m/s², `INHERITED` from `closedloop.py:155`'s own annotation). The steering channel is inside its corpus envelope (mean \|steer\| 0.0098–0.0134 rad against a corpus max of 0.016, `closedloop.py:154`).**

The interventions agree, and they separate the channels cleanly:

| family | setting | `T_blind` | gain vs own | separated? |
|---|---|---:|---|---|
| **`steer_clip`** (rad) | 0.02 | 24 | −1 | ⛔ **no** |
| | 0.005 | 25 | +2 | ⛔ no |
| | 0.0 *(diagnostic)* | 27 | +8 | ⛔ no |
| **`accel_clip`** (m/s²) | 1.0 | **47** | **+27** | ✅ |
| | 0.3 | **62** | **+46** | ✅ |
| | 0.0 *(diagnostic)* | **78** | **+67** | ✅ |
| **channel** *(diagnostic)* | own steer, accel **held** | **90** | **+77** | ✅ |
| | own accel, steer **held** | 49 | +31 | ✅ |

⇒ **Clamping the steering to a tenth of its band changes nothing. Removing the acceleration command
entirely buys 25 → 78 steps.** Keeping the model's steering and holding its acceleration is worth
**90 steps**; the reverse is worth **49**.

## 4.3 ⚠️ Feedback instability — the oscillation is REAL, the pre-registered signature is REFUTED

The loop **is** oscillating, and hard:

| statistic, `own` arm, all 184 fed steps | value |
|---|---:|
| sign-flip rate of the accel command, per tick | **0.2886** |
| step-to-step accel jitter | **3.15 m/s²** (first 0.5 s) |
| fraction at **+3** / at **−3** | 0.1143 / 0.0937 |
| **mean SIGNED accel** | **−0.0176 m/s²** |
| mean \|accel\| | 1.0917 m/s² |
| ⇒ **bias / amplitude** | ⭐ **0.0162** |

**It is a near-zero-mean, clamp-saturating oscillation, not a drifting bias.** My pre-registered
CONFIRM signature for this mechanism was *"`ema` / `every` — which break the per-step feedback
without changing the mean action — recover a large share"*. **`every` fired in the opposite
direction:**

| family | setting | `T_blind` | `de@2s` | `ade_0_2s` |
|---|---|---:|---:|---:|
| baseline (own) | — | 25 | 1.8165 | 0.8710 |
| **`every`** | m = 2 | ⛔ **9** | ⛔ **2.8106** | 1.4625 |
| | m = 5 | ⛔ **9** | ⛔ **3.9867** | 1.9242 |
| | m = 20 | ⛔ **9** | ⛔ **4.6320** | 2.1458 |
| **`ema`** | β = 0.5 | 38 | 1.3862 | 0.6550 |
| | β = 0.8 | **64** | 0.9864 | 0.4667 |
| | β = 0.95 | **111** | 0.6966 | 0.3464 |

> ### ⛔ **Zero-order-holding a SAMPLE of a zero-mean ±3 m/s² oscillation removes its cancellation and converts it into a sustained off-distribution command. Reducing the action-update frequency — the textbook de-jitter — is 2.8× WORSE at 2 s than leaving the loop alone.**

`ema` helps for a different reason than "smoothing": it **shrinks amplitude**. MEASURED on the dense
audit subset — `ema0.8`'s fed \|accel\| is **0.344 m/s² with 0 % saturation**, against `own`'s
**1.244 with 23 %**. `accelclip0.3` likewise: **0.196, 0 %**. Every intervention that helps reduces
amplitude; the one that does not (`every` preserves amplitude and removes cancellation) hurts.

**Adjudication, per the pre-registered rule:** mechanism 3's CONFIRM signature required `every` to
recover a large share. It did not. **REFUTED as stated.** The corrected mechanism is 4.2's: what
binds is the oscillation's **amplitude**, not its presence, and `ema`/`every` are not the
interchangeable pair my pre-registration assumed. Recorded as amendment **C1** (§8.1).

## 4.4 ⛔ Horizon onset — REFUTED, no knee

Recovery of the own→hold gap, `(de_own − de_arm)/(de_own − de_hold)`; 0 = no better than letting the
model act throughout, 1 = as good as never letting it act. *Diagnostic arms — they read the horizon,
so they are not deployable.*

| arm | @5 | @20 | @40 | @185 |
|---|---:|---:|---:|---:|
| own first **5** then hold | 0.000 | 0.798 | **0.997** | **1.031** |
| own first **10** then hold | 0.000 | 0.388 | 0.825 | 1.074 |
| own first **20** then hold | 0.000 | 0.000 | 0.263 | 0.979 |
| own first **40** then hold | 0.000 | 0.000 | 0.000 | **0.654** |
| hold first **5** then own | 1.000 | 0.627 | 0.301 | **0.276** |
| hold first **10** then own | 1.000 | 0.902 | 0.564 | 0.458 |
| hold first **20** then own | 1.000 | 1.000 | 0.886 | 0.622 |
| hold first **40** then own | 1.000 | 1.000 | 1.000 | **0.705** |

**Monotone in switch time in both directions, with no threshold.** Five steps of own action cost
essentially nothing permanent (103 % recovered by 18.5 s); forty steps cost 35 % of the gap forever.
And handing control to the model *late* still loses: after 40 steps of hold, switching to own decays
from 1.000 to **0.705**. ⇒ **the damage is proportional to time-in-loop, not to an early transient.**

## 4.5 The convention control, re-measured at the calibrated readout

| arm | `T_blind` | `de@2s` | `ade_0_2s` | beats CV |
|---|---:|---:|---:|---|
| own | 25 | 1.8165 | 0.8710 | 0/185 |
| **`gtkin` — the same inverse, fed TRUE motion** | **185 ⚠️ saturated** | **0.4361** | **0.2552** | **179/185 (0.7–18.5 s)** |

🔴 **The inverse map is not the fault. Fed true motion it saturates the sweep and beats constant
velocity over essentially the whole horizon** — better even than hold-last. The ladder carries
`INHERITED` *"89.8 % of the own-action penalty is the model, not the measurement convention"*,
measured at the **`op`** readout; **at the calibrated `str` readout the convention control is at the
ceiling, so the figure is ~100 %.** Escalation **E-3**.

---

# 5. The other intervention families, in full

`artifacts/rung1_interventions.json`. Baseline own = 25 steps, `de@2s` 1.8165, `ade_0_2s` 0.8710.

| family | config | eligible | `T_blind` | CI95 | gain | `de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` |
|---|---|---|---:|---|---|---:|---:|---|---:|
| clip steer | 0.02 rad | ✅ | 24 | [2.4, 3.8] | −1 ⛔ | 1.8497 | 0.8833 | 0/185 | 1.4 s |
| | 0.005 | ✅ | 25 | [2.5, 4.4] | +2 ⛔ | 1.8289 | 0.8757 | 0/185 | 1.4 s |
| | 0.0 | ⛔ diag | 27 | [2.7, 5.5] | +8 ⛔ | 1.8143 | 0.8718 | 0/185 | 1.4 s |
| clip accel | 1.0 m/s² | ✅ | 47 | [4.7, 7.3] | +27 ✅ | 1.3513 | 0.6477 | 0/185 | 1.7 s |
| | 0.3 | ✅ | **62** | [6.2, 10.1] | +46 ✅ | 1.2053 | 0.5774 | ✅ 14/185 | 1.8 s |
| | 0.0 | ⛔ diag | 78 | [7.8, 12.7] | +67 ✅ | 1.1684 | 0.5608 | ✅ 23/185 | 1.8 s |
| smooth EMA | β 0.5 | ✅ | 38 | [3.8, 5.5] | +15 ✅ | 1.3862 | 0.6550 | 0/185 | 1.7 s |
| | β 0.8 | ✅ | **64** | [6.4, 8.9] | +44 ✅ | 0.9864 | 0.4667 | ✅ 34/185 | 2.0 s |
| | β 0.95 | ✅ | **111** | [11.1, 16.1] | +101 ✅ | 0.6966 | 0.3464 | ✅ 76/185 | 2.3 s |
| update every | m 2 | ✅ | ⛔ **9** | [0.9, 1.1] | −21 ⛔ | 2.8106 | 1.4625 | 0/185 | 1.0 s |
| | m 5 | ✅ | ⛔ **9** | [0.9, 1.0] | −22 ⛔ | 3.9867 | 1.9242 | 0/185 | 0.9 s |
| | m 20 | ✅ | ⛔ **9** | [0.9, 1.0] | −22 ⛔ | 4.6320 | 2.1458 | 0/185 | 0.9 s |
| channel | own steer, accel held | ⛔ diag | **90** | [9.0, 13.8] | +77 ✅ | 0.8797 | 0.4306 | ✅ 60/185 | 2.1 s |
| | own accel, steer held | ⛔ diag | 49 | [4.9, 7.8] | +31 ✅ | 1.5493 | 0.7580 | 0/185 | 1.5 s |
| convention | `gt_kinematic` | ⛔ privileged | 185 ⚠️ | [18.5, 18.5] | +154 ✅ | 0.4361 | 0.2552 | ✅ 179/185 | 3.0 s |
| speed channel | own predicted `v` fed back | ✅ | ⛔ **9** | [0.9, 1.0] | −21 ⛔ | 🔴 **23.9351** | 9.6020 | 0/185 | 0.6 s |

🔴 **`own_vupd` — feeding the model's own predicted speed into the `v0` action channel instead of
holding the observed constant — is catastrophic: `de@2s` 1.82 → 23.94 m, a 13.2× degradation.** This
is a plausible-sounding "fix" that the E-IMAG-3 sensitivity list carries; it must be struck.
Escalation **E-2**.

---

# 6. The verdict, applied mechanically

`artifacts/rung1_verdict.json`.

## 6.1 The pre-registered buckets

| | |
|---|---|
| rule (fixed before any number) | CONFIRM: best ELIGIBLE `T_blind` **≥ 50 steps** AND paired gain separated · PARTIAL: 31–49 separated, or ≥ 50 not separated · REFUTE: **≤ 30 steps** |
| baseline, this run | **25 steps** (pre-registered 25) |
| ceiling, this run | **115 steps** (pre-registered 115) |
| eligible arms | **18**, fixed in advance |
| **best eligible** | **`blend0.75`** |
| **`T_blind`** | ⭐ **116 steps = 11.6 s [11.6, 17.2]** |
| paired gain vs own | **+109 steps [85.0, 140.0]**, ✅ separated, `frac_draws_gain > 0 = 1.000` |
| Bonferroni requirement (1 − 0.05/18 = 0.9972) | ✅ **met** |
| fraction of the registered ceiling recovered | **1.011** |
| **VERDICT** | ⭐⭐ **CONFIRM** |

## 6.2 ⚠️ The capability cap — and this time it breaks

Pre-registration §5.1 fixed: a capability **CONFIRM** requires beats-CV > 0 with a separated interval
**or** `T_useful@1m` > 1.4 s. Both are comparator-free — no readout or filter mismatch can enter
them, because the constant-velocity floor is pure kinematics.

| | own (baseline) | `blend0.25` | `blend0.75` (best) |
|---|---|---|---|
| beats constant velocity | ⛔ **never, 0/185** | ✅ **43/185, 1.2 – 5.4 s** | ✅ **81/185, 0.7 – 8.7 s** |
| `T_useful@1m` | 1.4 s | **1.9 s** | **2.3 s** |
| `T_useful@2m` | 2.1 s | 2.7 s | 3.2 s |
| **capability verdict** | — | ✅ **CONFIRM** | ✅ **CONFIRM** |

> ## ⭐⭐ **Rung 0's honest headline was "the world model now demonstrably adds something blind; it still does not add enough to beat assuming nothing changes." THAT SENTENCE NO LONGER HOLDS. With a 25 % damping of its own action — no retraining — the deployable arm is separated-better than constant velocity over 1.2 – 5.4 s while still supplying three quarters of the command. The floor that stood through the whole blind-imagination stream is broken.**

**And the distinction the ladder insists on is kept.** The `T_blind` number is an extension **against
a frozen percept**; beats-CV and `T_useful` are the comparator-free capability statistics. Here they
move **together**, which is precisely why the capability claim is admissible — Rung 0's was not,
because its two comparator-free statistics did not move at all.

**Robustness of the capability claim to the in-sample α.** On the **stricter** half of the
pre-registered bar (beats-CV > 0), **9 of 18 eligible arms clear it** — `blend0.25 … blend0.875` (6),
`accelclip0.3`, `ema0.8`, `ema0.95` — across **three independent families**. On the bar as written
(beats-CV > 0 **or** `T_useful@1m` > 1.4 s), **12 of 18** clear it. The conclusion does not rest on a
tuned α; the specific α = 0.75 does, and is not quoted as a recommendation (§7.3).

---

# 7. 🔴 ESCALATIONS — in the headline, not written into a README

**E-1. R3's pre-registered bar is now stale by 8.4 s, and R3 is a ~59-hour run.** The ladder's R3
(action-channel scheduled sampling) carries the falsifier *"REFUTE if deployable `T_blind` does not
exceed **3.2 s** — the held-last no-policy value; a learned fix that cannot beat 'hold the last
action' is not a fix."* That 3.2 s is the `op`-readout hold value. At the calibrated readout the
matched hold value is **11.5 s**, and **a zero-training action filter already reaches 11.6 s**.
⇒ **R3's bar must be re-derived to ≥ 11.6 s before the run is approved**, or the program will spend
59 GPU-hours to clear a bar a one-line filter already cleared. **This is a decision the PI is owed
before R3 is launched.**

**E-2. `own_vupd` must be struck from the candidate-fix list.** `de@2s` 1.8165 → **23.9351** (13.2×
worse). It appears in the blind-imagination stream's E-IMAG-3 lever list as an untested sensitivity;
it is now tested at the calibrated readout and it is destructive.

**E-3. The ladder's "89.8 % of the own-action penalty is the model" is decoder-conditional.**
Measured at `op`. At `str` the convention control (`gtkin`) reaches the sweep terminus (185 steps,
`de@2s` 0.4361, beats CV 179/185), so the figure is **~100 %** and the inverse map is exonerated
entirely. Same class as the ladder's own E-2.

## 7.1 What this unblocks, per stream

| stream | what it gets |
|---|---|
| 🔴 **R3 / scheduled sampling** | a **re-aimed target and a much higher bar** (E-1), plus a named mechanism: the fix must attack the **longitudinal command's amplitude**, not the steering (`steer_clip` is null at every setting) and not the sampling rate (`every` is destructive). |
| ⭐ **the deployable blind configuration** | a concrete, zero-cost recommendation — **α ∈ [0.25, 0.5]** — that beats constant velocity, with its interval. |
| **the ladder's R1 (planner arm)** | a **pre-registered prediction** (§7.2) it can be scored against. |
| **the three-planner / hierarchy direction** | the action loop is now not merely "co-primary" but **fully recoverable without retraining** — which moves the hierarchy's value proposition away from stabilising the loop and toward what the loop should be *aiming at*. |
| **v5 imagination selection** | the fan's implausible-plan problem has an action-space analogue with a measured fix: clipping the **acceleration** channel to a corpus-plausible band is worth +37 steps on its own. |

## 7.2 ⭐ A pre-registered prediction for the planner arm (the ladder's R1), on the record now

`closedloop.wp_to_control` differs from the kinematic inverse in exactly the two gains this rung
swept: it steers from a **0.5 s pure-pursuit lookahead** (`LOOKAHEAD_STEP = 5`) instead of a one-tick
yaw increment, and it accelerates with **`(v_target − v)/SPEED_TC`, `SPEED_TC = 0.5 s`** instead of
`(v − v_prev)/0.1 s` — a **5× lower longitudinal gain**. An EMA at β = 0.8 at 10 Hz has time constant
`DT/(1−β) = 0.5 s`, **numerically the planner's own `SPEED_TC`**, and it delivers **64 steps
(6.4 s)**.

> **PREDICTION, recorded before R1 runs: v1's tactical planner will land between 6 and 12 s of deployable `T_blind` — comfortably above the 2.5 s kinematic-inverse baseline and at or below the 11.6 s filter result — because its advantage over the inverse is a lower longitudinal gain, which is the same lever measured here. If it lands BELOW 6.4 s, the planner is adding a failure mode the gain argument does not explain and that is the finding.**

## 7.3 What is deliberately NOT claimed

* ⛔ **α = 0.75 is not a tuning recommendation.** It is an in-sample maximum, and α ∈ [0.625, 1.0] is
  one step wide. The deployable recommendation is **α ∈ [0.25, 0.5]** — where the model still
  supplies 50–75 % of the command, beats-CV is 43–72/185, and `T_blind` is 85–111 steps.
* ⛔ **α = 1 is not a result.** It removes the policy entirely; it is the measured ceiling and was
  excluded from eligibility before any number existed.
* ⛔ **This is not a safety result.** PhysicalAI-AV ships no map, lane graph or agent boxes. Drift
  only.
* ⛔ **The planner arm was NOT run.** §7.2 is a prediction, not a measurement.

---

# 8. Amendments and owed retraction rows

## 8.1 Amendments to my own pre-registration — recorded here, not by editing it

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **C1** | **Mechanism 3's CONFIRM signature bundled `ema` and `every` as one test** (*"which break the per-step feedback without changing the mean action"*). The data separated them decisively: `ema` helps (+101 steps at β = 0.95), `every` destroys (−22). | Applied in the **stricter** direction: the signature is reported as **REFUTED as written**, not quietly re-read as confirmed by `ema` alone. **BINDING LESSON: a mechanism signature must name ONE operation. `ema` shrinks amplitude; `every` preserves amplitude and removes cancellation. Bundling two operations under one mechanism label means whichever fires can be quoted as the confirmation.** No bucket, bar or eligibility set was touched. |
| **C2** | The blend-curve monotonicity statistic returned `False` on a terminal 116 → 115 step. | Reported as measured (`monotone_nondecreasing = False`, Spearman 0.95) and described as *"monotone to α = 0.75, then indistinguishable"* rather than rounded up into a monotonicity claim. |
| **C3** | The `T_blind` **integer** fidelity check against the ladder was declared **non-blocking** (level agreement blocking) before the numbers were read, because a step count is a threshold crossing and the arms come from a second encode pass. | It reproduced exactly and the latitude was not used. Declared in advance so it could not be a post-hoc escape. |

## 8.2 🔴 Retraction-log rows this stream owes

⛔ **Escalated, not filed** — `Project Steering/RETRACTION_LOG.md` is a shared append-only steering
document and other agents are writing this session.

| date | what is withdrawn | root-cause class | the correction |
|---|---|---|---|
| 07-26 | The ladder's **R3 falsifier bar of 3.2 s** ("a learned fix that cannot beat hold-the-last-action is not a fix") | **C-STALE-BAR — a bar derived under a configuration the program has since replaced** | 3.2 s is the **`op`-readout** hold value. Rung 0 adopted the calibrated `str` readout, where matched hold-last is **11.5 s** and a zero-training filter reaches **11.6 s**. **BINDING LESSON: when a decoder/configuration change is adopted, every pre-registered BAR derived under the old one must be re-derived — a bar is a function of the configuration, not a constant.** |
| 07-26 | *"89.8 % of the own-action penalty is the model, not the measurement convention"* (`INHERITED`, blind-imagination stream, carried into the ladder) | **C-DECODER-CONDITIONAL — a decomposition quoted without the decoder it was measured under** (a recurrence of the ladder's own E-2 class) | Measured at the `op` readout. At the calibrated `str` readout the convention control reaches the sweep terminus (185 steps, `de@2s` 0.4361), so the share is **~100 %**. The direction was right; the number is decoder-conditional and is not quotable without naming the readout. |

---

# 9. Limitations, stated plainly

1. **One arm, one action policy, one readout.** v1 `flagship4b-speedjerk-30k` @ 29999; the
   `own_kinematic` inverse; the `str` (k = 20) readout. Nothing on v4, REF-B or REF-C.
2. ⚠️ **The planner arm was not run.** The ladder's R1 remains open. Declared in the pre-registration
   (§1), not narrowed after the fact.
3. ⚠️ **α is selected in-sample** from 18 pre-registered eligible arms on the same 599 windows. The
   dose–response shape and the fact that 7 of 18 arms across 3 families clear the capability bar are
   what carry the conclusion; the specific α does not (§7.3).
4. ⚠️ **`a_hold0` is a CONSTANT.** It is causally available (the action the ego just executed, not
   future information), but blending toward a constant means the arm degenerates toward the no-policy
   ceiling as α → 1. Part of the advantage reflects that this corpus is mostly near-constant-action
   driving. The low-α rows are the informative ones.
5. ⚠️ **The window set is EPISODE-INITIAL** (596 of 599 windows at `t0 = 0`) and runs ~6–12 % low in
   absolute level (`INHERITED-MEASURED`). All contrasts are paired on identical windows.
6. ⚠️ **Everything past 0.4 s is extrapolation** — the operative readout was trained at k = 4, the
   `str` readout at k = 20. The 11.6 s numbers sit deep in the region where `INHERITED-MEASURED`
   20 % of steps (at 6 s) and 52 % (at 12 s) are outside the measured envelope.
7. ⚠️ **`gtkin` saturates at the sweep terminus** — C14: a LOWER BOUND on our configuration, not a
   horizon.
8. **No safety metric.** Drift only; PhysicalAI-AV ships no map, lane graph or agent boxes.
9. **The four new comparator arms come from a second encode pass**, bounded at **3.05e-05 m** by the
   identity gate; the two `a`-arm anchors reproduced at exactly 0.0.
10. **The `own_before` / `own_after` switch arms read the horizon** and are diagnostic only — they
    were never eligible and set no verdict.

---

# 10. Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-tblind-rung1/`

| artifact | what it is | where it lives |
|---|---|---|
| `PRE_REGISTRATION.md` | written **before** any number existed: endpoints, identity gate, plumbing self-test, the 7 families + eligibility rule, buckets, multiplicity handling, capability cap, failing values, the 4 mechanism signatures | **repo** |
| `TBLIND_RUNG1.md` | this document | **repo** |
| `scripts/tb_rung1_sweep.py` | the 61-arm pod driver — `bi_run.stage_sweep` reused **verbatim**, arm list replaced | **repo** · `pod2:/root/tbr1/` |
| `scripts/tb_rung1_action_audit.py` | the dense `fed_actions` audit + the **reconstruction gate** in both directions | **repo** · `pod2:/root/tbr1/` |
| `scripts/tb_rung1_compact.py` | pod-side compaction + the **plumbing self-test at full K** | **repo** · `pod2:/root/tbr1/` |
| `scripts/tb_rung1_analyze.py` | ⭐ the adjudication — identity gate, fidelity gate, vacuity audit, blend curve, mechanism, families, verdict | **repo** |
| `scripts/tb_rung1_tables.py` | renders every table above **from the raw JSON**, so report and artifacts cannot drift | **repo** |
| `artifacts/rung1_gates.json` | ⛔ identity gate · plumbing self-test · fidelity vs the ladder · diagnostic-vacuity audit · failing-value probe | **repo** |
| `artifacts/rung1_blend_curve.json` | ⭐ **the headline** — 9 α points, each with matched comparator, gain, level, beats-CV, `T_useful` | **repo** |
| `artifacts/rung1_mechanism.json` | penalty curve + log-log fits · action statistics over 599 windows · onset sweep · pod reconstruction gate · dense fed-action stats | **repo** |
| `artifacts/rung1_action_signature.json` | the bias-vs-amplitude decomposition that identified the mechanism | **repo** |
| `artifacts/rung1_interventions.json` | all 6 non-blend families | **repo** |
| `artifacts/rung1_verdict.json` | the mechanical verdict + capability cap + full ranking | **repo** |
| `artifacts/_tables.md` | every table in this report, generated from the JSON | **repo** |
| `artifacts/rung1_analyze.log` · `rung1_sweep_run.log` · `rung1_audit_run.log` | run logs incl. all gates | **repo** |
| `artifacts/sweep_meta_K185.json` · `action_audit_meta_K185.json` | run manifests: 61 arms, parity block, timings, reconstruction gate | **repo** |
| `perwindow/rung1_perwindow_compact.pt` (32.0 MB) | ⭐ **dense per-window `de` [599 × 185] for all 58 arms + both floors**, plus `psi`/`pred_speed` for the 6 action-analysis arms — **any bar, horizon or stratification recomputes with no GPU** | **repo** · `pod2:/root/tbr1/perwindow/` |
| `perwindow/action_audit_K185.pt` (2.4 MB) | dense `fed_actions`/`psi`/`pred_speed`/`step_dpose` for 10 arms on the 41-window audit subset | **repo** · `pod2:/root/tbr1/perwindow/` |
| `taniteval/taniteval/blindimag.py` | **+167 lines, 0 deletions** — the action-filter suffix, `parse_action_source`, `apply_action_filter`, `reconstruct_kinematic_actions`. Every pre-existing path bit-identical | **repo (modified, staged)** · `pod2:/root/taniteval/` |
| `taniteval/tests/test_blindimag.py` | **+147 lines, 0 deletions** — 26 tests incl. both endpoint identities, the anti-no-op parametrisation over all 9 knobs, and the reconstruction with its failing direction | **repo (modified, staged)** |

**Living in only ONE place (declared, per rule 2):** the full `perwindow_sweep_K185.pt` (**111.3 MB**,
md5 `116c787e84ae2d83ecd555d54d14dec4`) exists only at `pod2:/root/tbr1/perwindow/`. **The 32.0 MB
compaction in the repo carries everything every number in this report needs**; the full dump rebuilds
in ~30 min on an idle A40 (deterministic, `torch.manual_seed(0)`, no sampling). Both pulled files were
**md5-verified identical to the pod** (`4053342c0f06da4733af25d896a5b932`,
`1b851fde4d260ce893ced63c31e35438`).

**Suites.** The only files touched outside this folder are `taniteval/taniteval/blindimag.py` and its
test — both **purely additive** (0 deletions). `taniteval`: **449 passed**. Nothing under `stack/`
was modified.

**Cost.** pod2, one job: encode **793.9 s** + rollout of **61 arms × 599 windows × 185 steps** in
**1034.9 s** = **30.5 min**, plus a **~2 min** audit. `MEASURED`,
`artifacts/sweep_meta_K185.json`.

---

# 11. Reproduction

```
# pod2 (A40, must be idle) — ~33 GPU-min total
PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts OMP_NUM_THREADS=8 \
python3 tb_rung1_sweep.py       --out /root/tbr1/perwindow --episodes 600 --kmax 185
python3 tb_rung1_action_audit.py --out /root/tbr1/perwindow --episodes 40  --kmax 185
python3 tb_rung1_compact.py --dump /root/tbr1/perwindow/perwindow_sweep_K185.pt \
                            --out  /root/tbr1/perwindow/rung1_perwindow_compact.pt

# dev box — ZERO GPU, ~2 min
python scripts/tb_rung1_analyze.py \
  --new      perwindow/rung1_perwindow_compact.pt \
  --bi       ../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt \
  --matched  ../2026-07-26-tblind-ladder/perwindow/perwindow_matched_K185.pt \
  --audit    perwindow/action_audit_K185.pt --out artifacts
PYTHONIOENCODING=utf-8 python scripts/tb_rung1_tables.py --art artifacts > artifacts/_tables.md
```
