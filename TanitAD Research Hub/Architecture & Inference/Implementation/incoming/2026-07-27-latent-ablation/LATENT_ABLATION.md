# THE LATENT ABLATION — is blind driving imagined semantics, or action integration?

**Date:** 2026-07-27 (Europe/Berlin; pods log UTC). **Stream:** Architecture & Inference —
blind-imagination driving.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before any number here existed**.
It is **not edited**; every deviation is an amendment in §8.
**Host:** pod2 only (A40, MEASURED idle before launch: `0 MiB, 0 %`; disk verified with a real
**2 GB `dd` write** to `/workspace`, 2,097,152,000 bytes at 103 MB/s, no short write — never `df`).
⛔ pod1 (training), pod3 and the eval pod were **never connected to**; the val cache was read only.
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

> ## ⛔ **FIRST, A STRUCTURAL CORRECTION THAT REFRAMES THE WHOLE QUESTION. The program's `T_blind` is ALREADY a latent ablation.** `tb_rung0.t_blind(de_a, de_b)` is *"the largest horizon at which (a) is separated-better than (b), contiguously from N = 2"*, and **every published `T_blind` in this program uses `b = frozen_last`** — the last real percept held constant. So the PI's ablation 1 is not a new arm: **it is the denominator of the headline number.** Consequences: `T_blind(FROZEN vs FROZEN)` is **1 step by construction** and is emitted as VACUOUS, never adjudicated; and the frozen-latent arm at **every** α was already measured by Rung 1 and sits in the committed dump. The priority-1 deliverable therefore cost **zero GPU**.

> ## ⭐⭐ **AND THE ANSWER TO THE PI, ON THE CLEANEST CONTRAST AVAILABLE, IS THAT THE LATENT IS DOING REAL WORK — NOT NOTHING.** At **α = 1.00 the fed steer/accel are the last OBSERVED action for every arm and are bit-identical across state sources** (test-pinned), and the `v0` channel is the constant true speed for every arm. **The only thing that differs is whether the latent evolves.** Letting it evolve is worth **`de@2s` 0.6718 vs 2.0503 — a 3.05× degradation when it is frozen** (paired Δ **1.3785 [1.2503, 1.5122]**, separated), and it is the entire difference between **beating constant velocity 83/185 and never beating it (0/185)**, and between `T_useful@1m` **2.3 s and 1.1 s**.

> ## ⭐⭐ **THE COMPARATOR-FREE FORM IS SHARPER STILL, AND IT SPLITS THE PI's DICHOTOMY IN TWO. Against `hold_v0` — go straight at the current speed, ONE SCALAR, no latent at all — the blind model with its OWN undamped action loop is 1.46× WORSE (ratio 0.685 at 2 s, beats it 0/185 steps). With that action loop damped out, the SAME weights and the SAME latent beat it 1.852× at 2 s, separated over 0.7 – 8.6 s (80/185).**
>
> ## ⇒ **Both halves of the PI's question are true of different parts of the system. The ACTION loop is not merely un-helpful, it is NET DESTRUCTIVE — it turns a model that beats a one-scalar integrator into one that loses to it. The LATENT is what beats the integrator. "The world model drives blind" is not withdrawn; what must be withdrawn is any claim that it drives blind *through its own action loop*.**

> ## ⚠️ **AND THE HORIZON DEPENDENCE IS THE FINDING NOBODY ASKED FOR.** The frozen-latent penalty peaks at **1 s (+3.61)**, is **+2.05 at 2 s**, decays to **+0.14 at 9 s** and **+0.04 at 12 s**, and is **−0.06 at 18.5 s — the frozen latent is separated-BETTER there.** **The imagined latent carries driving content for roughly the first 3–6 s and essentially none past 9 s.** The headline `T_blind` of **11.5 s** sits deep inside the region where it contributes nothing — and at that horizon **both arms are ~74–79 m from the truth.** `T_blind` at 11.5 s is a *relative* statistic about which of two catastrophically wrong paths is marginally less wrong; the capability statistic is `T_useful@1m = 2.3 s`.

| what the brief asked | the answer, `MEASURED` |
|---|---|
| *Frozen latent, action channel intact* | **Already measured, at every α.** ⭐ It is `T_blind`'s own comparator. At α = 0.25 it costs **+100.1 %** of `de@2s`; at α = 1.00 **+205.2 %**. |
| *Shuffled latent* | §4 |
| *Zeroed / mean latent* | §4 |
| *Does the ablation barely move `T_blind` (INTEGRATOR) or collapse it (SEMANTIC)?* | §5 — the pre-registered buckets, applied mechanically. |
| *Do the imagined latents drift to a fixed point?* | §6 |
| *Report `beats-CV` and `T_useful@1m` explicitly* | Every row of every table. M10's tier split is kept throughout. |

---

# 1. What the architecture makes separable — read from the code, not from prose

`taniteval/blindimag.py::blind_rollout` (lines 485–583) and `build_windows` (799–807):

1. ⭐ **The `v0` action channel carries the TRUE observed speed and is CONSTANT for the whole
   rollout.** `build_windows` broadcasts `pose_last.v / SPEED_SCALE` across the window *and the entire
   future*; `_pack_action` carries channels ≥ 2 through unchanged; every arm here runs
   `update_speed_channel = False`. **It survives every latent ablation below** — which is exactly what
   makes these ablations a test of the PI's dichotomy rather than of something else. Test-pinned:
   `test_latent_ablation_leaves_the_constant_v0_channel_untouched`.
2. ⚠️ **At α < 1 the fed steer/accel are a FUNCTION of the latent** — they are the kinematic inverse
   of `dpose = step_readout(win_s[:,-1], z_hat)`. So a latent ablation there *also* perturbs the
   action. That is not a defect of the design, it is the architecture: outside the constant `v0` and
   the hold-last blend there is **no action channel independent of the latent**. Test-pinned in both
   directions: `test_at_alpha_one_the_fed_action_is_identical_across_state_sources` (the fed tensor is
   bit-identical at α = 1) and `test_at_alpha_zero_the_fed_action_DOES_depend_on_the_latent` (it is
   not at α = 0). **α = 1.00 is therefore the attributable row and it is adjudicated as such.**
3. ⛔ **`T_blind`'s comparator IS `frozen_last`** — see §0.

**⚠️ Because of (2), a latent ablation at α = 0.25 is a JOINT ablation of the latent and of the
action derived from it.** It is reported because the PI asked for the deployable point; it is not the
attributable one, and the report never quotes it as though it were.

---

# 2. ⛔ THE GATES — nothing below them was read until they passed

## 2.1 Window-set identity — `artifacts/la_gates.json`, and `la_stage_a_frozen.json → gate_windows`

| check | result |
|---|---|
| windows | **599** ✅ | 
| episode clusters | **596** ✅ |
| windows at `t0 = 0` | 596 (the episode-initial window set; §9) |
| `eid` / `t0` ordering vs the committed Rung-1 dump | §2.4 |
| the eight anchors' dense `de` | §2.4, tolerance 1e-4 m |

## 2.2 Fidelity to the committed headline — **PASS, exactly**

`la_stage_a_frozen.json → gate_fidelity`. `LEVEL_FIDELITY_PASS = True`, and the `T_blind` **integers**
reproduced too although they were declared non-blocking:

| α | `T_blind` | `de@2s` | `ade_0_2s` | `T_useful@1m` | beats-CV |
|---:|---|---|---|---|---|
| 0 | 25 = **25** | 1.8165 = **1.8165** | 0.8710 = **0.8710** | 1.4 = **1.4** | 0 = **0** |
| 0.25 | 85 = **85** | 1.0736 = **1.0736** | 0.5440 = **0.5440** | 1.9 = **1.9** | 43 = **43** |
| 0.75 | 116 = **116** | 0.6842 = **0.6842** | 0.3437 = **0.3437** | 2.3 = **2.3** | 81 = **81** |
| 1 | 115 = **115** | 0.6718 = **0.6718** | 0.3351 = **0.3351** | 2.3 = **2.3** | 83 = **83** |

⇒ ⭐ **The brief's correction to the headline is confirmed exactly.** At **α = 1.00 the model
contributes 0 % of the steer/accel command and `T_blind` is 11.5 s with `de@2s` 0.6718**; at
**α = 0.75 it is 11.6 s with `de@2s` 0.6842 — worse accuracy.** The 11.6 s headline is the hold-last
ceiling. The genuinely model-driven point is **α = 0.25: 8.5 s, `de@2s` 1.0736, beats-CV 43/185.**

## 2.3 The plumbing self-test and the vacuity audit

See §2.4 (they are resolved on the sweep dump, at the full K = 185).

## 2.4 *(populated by `la_analyze.py` — the sweep stage)*

---

# 3. ⭐ STAGE A — THE FROZEN-LATENT ABLATION. Zero GPU, and it answers half the question

`artifacts/la_stage_a_frozen.json`, rendered in `artifacts/_tables.md` §A.
`MEASURED` · tier **DECISION-GRADE for the `T_blind`/`de` numbers** (they reproduce the committed
headline exactly on a matched window set) · **CONFIRMED-but-not-decision-grade for the capability
claim** (beats-CV and `T_useful` are comparator-free, but α is not selected here — it is enumerated).

| α | model share | INTACT `de@2s` | FROZEN `de@2s` | `R_FROZEN` | paired Δ [CI95] | `T_blind` | beats-CV int/frz | `T_useful@1m` int/frz |
|---:|---:|---:|---:|---:|---|---:|---|---|
| 0 | 100 % | 1.8165 | 2.2295 | **+0.2273** | 0.4130 [0.2651, 0.5673] ✅ | 25 (2.5 s) | 0 / 0 | 1.4 / 1.0 s |
| **0.25** | **75 %** | 1.0736 | 2.1487 | **+1.0014** | 1.0751 [0.9459, 1.2099] ✅ | 85 (8.5 s) | **43** / 0 | **1.9** / 1.0 s |
| 0.75 | 25 % | 0.6842 | 2.0704 | **+2.0259** | 1.3862 [1.2612, 1.5206] ✅ | 116 (11.6 s) | 81 / 0 | 2.3 / 1.1 s |
| **1** ⭐ | **0 %** | **0.6718** | **2.0503** | **+2.0518** | **1.3785 [1.2503, 1.5122]** ✅ | 115 (11.5 s) | **83** / **0** | **2.3** / **1.1** s |

> ### ⭐⭐ **The α = 1 row is the decisive one and it is unconfounded. The fed action tensor is bit-identical between the two arms; the `v0` channel is the same constant true speed in both. Freezing the latent multiplies `de@2s` by 3.05, destroys beats-CV entirely (83/185 → 0/185) and halves `T_useful@1m`. Whatever else is true, the imagined latent's EVOLUTION is carrying real driving content at short horizon.**

## 3.1 ⚠️ The horizon dependence — where the latent stops contributing

`R_FROZEN` across the grid (positive = the frozen latent is worse):

| α | 0.5 s | 1 s | 2 s | 3 s | 4.5 s | 6 s | 9 s | 12 s | 18.5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +1.681 | +1.092 | +0.227 | +0.015 | −0.107 | −0.175 | −0.236 | −0.260 | −0.265 |
| 0.25 | +2.042 | +1.909 | +1.001 | +0.625 | +0.336 | +0.184 | +0.042 | −0.029 | −0.098 |
| 0.75 | +2.716 | +3.429 | +2.026 | +1.146 | +0.609 | +0.363 | +0.139 | +0.042 | −0.058 |
| **1** | **+2.939** | **+3.606** | **+2.052** | **+1.183** | **+0.624** | **+0.370** | **+0.143** | **+0.043** | **−0.057** |

And the **absolute** deviations behind those ratios (m), INTACT / FROZEN at α = 1:

| 0.5 s | 1 s | 2 s | 3 s | 4.5 s | 6 s | 9 s | 12 s | 18.5 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.11 / 0.42 | 0.19 / 0.88 | 0.67 / 2.05 | 1.67 / 3.64 | 4.25 / 6.89 | 8.09 / 11.09 | 19.19 / 21.94 | 34.32 / 35.80 | 78.53 / 74.02 |

> ### ⚠️ **Two things follow, and the second is uncomfortable.** (1) **The latent's contribution is a SHORT-HORIZON phenomenon** — it peaks at 1 s, is halved by 3 s, and is inside noise by 9–12 s; at 18.5 s the frozen latent is separated-*better*. (2) **The 11.5 s `T_blind` lives entirely inside the region where the latent contributes nothing, and where BOTH arms are 74–79 m from the truth.** `T_blind` is a *relative* contiguity statistic, and M10's tier split exists precisely for this: the capability number is `T_useful@1m = 2.3 s`, and it is 2.3 s, not 11.5 s.

## 3.2 ⭐ The comparator-free form: is it better than a PURE kinematic integrator at all?

`la_stage_a_frozen.json → vs_pure_kinematic_integrators`. `hold_v0` is *"go straight at the current
speed"* — **one observed scalar, no latent whatsoever** (`blindimag.hold_v0_dense_path`). This needs
no ablation and no matched comparator, so no readout or filter mismatch can enter it.

| α | model share of command | ratio `hold_v0` / arm at 2 s | separated-better window vs `hold_v0` |
|---:|---:|---:|---|
| **0** | 100 % | ⛔ **0.685** — the one-scalar integrator is **1.46× BETTER than the model** | ⛔ **none, 0/185** |
| 0.25 | 75 % | ✅ 1.159 | ✅ 1.3 – 5.2 s (40/185) |
| 0.75 | 25 % | ✅ 1.818 | ✅ 0.7 – 8.5 s (79/185) |
| **1** ⭐ | 0 % | ✅ **1.852** | ✅ **0.7 – 8.6 s (80/185)** |

> ### 🔴 **THE SINGLE MOST DIRECT ANSWER TO THE PI, AND IT SPLITS THE QUESTION. Blind, driving its own action loop, v1 is 1.46× WORSE than a one-scalar straight-line integrator and never beats it. Freeze that action loop — same weights, same latent, same readout — and it beats the same integrator by 1.852× at 2 s, separated over 0.7 – 8.6 s. The action loop is net DESTRUCTIVE; the latent is what beats kinematics.**

*(This is the action-space counterpart of X1's finding that a true-`v0` integrator beats every
latent PROBE. Both are true and they are not in conflict: X1 probed what a linear read of the latent
can recover; this measures what the trained readout plus predictor produce when the latent is or is
not allowed to evolve.)*

---

# 4. *(the destructive ablations — populated by the sweep)*

# 5. *(the verdict, applied mechanically)*

# 6. *(the fixed-point probe)*

# 7. ESCALATIONS

# 8. Amendments

# 9. Limitations
