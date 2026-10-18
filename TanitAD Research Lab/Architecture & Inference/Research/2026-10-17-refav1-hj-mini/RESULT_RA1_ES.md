# RA-1 budget sweep (PREREG_RA1_ES): the readout never improves past 250 steps, and at 250 the A1 family beats P10, but A1x does too

**2026-10-18, MEASURED on box 2 (SAFE MODE). Tier: T1-class open-loop trajectory. Estimator: paired episode-cluster
bootstrap, B = 2000** (episode variance only). The training-seed floor is |s0 − s1|; there is no inference variance
(deterministic readout).
* Representation: measured yaw rate (`D-PI-YAWRATE-1/-2`); prior P10-yaw-540.
* Eval-141 = 2,399 windows over 141 episodes ⚠️ 11 of those clips are in the canonical refcv train corpus, so compare
  with refcv models only on the 130-clip subset. See `RESULT_RA1_YAW.md`.
* Holdout = the 60-clip Amendment-1 set (4,076 windows; P10 3.0245).
* Pre-registrations: `PREREG_RA1_ES.md` (ES-1/ES-2, then Addendum A for ES-3) and `PREREG_RA1_RIDGE.md`, each written
  before its data.
* Raw: `raw/ra1_es/`. Runs on box 2: `D:\tanitad\runs\ra1_yaw_es\st<S>\`.

## ES-1 — R0 at five budgets (selection on the holdout; eval-141 reported, never selecting)

| steps | holdout R0 − P10, seed 0 | seed 1 | seed-mean | eval-141 ADE@6s s0 / s1 | eval-141 R0 − P10, s0 / s1 |
|---|---|---|---|---|---|
| **250** | −0.0608 [−0.1448, +0.0246] | −0.0210 [−0.0927, +0.0551] | **−0.0409** | 2.5771 / 2.5648 | +0.0144 / +0.0020 (n.s.) |
| 500 | +0.0414 [−0.0617, +0.1605] | +0.0841 [−0.0321, +0.2140] | +0.0628 | 2.6454 / 2.6990 | +0.0827 / +0.1362 (sep) |
| 1,000 | +0.2227 [+0.0698, +0.3877] | +0.3787 [+0.1710, +0.6141] | +0.3007 | 2.8826 / 2.9414 | +0.3199 / +0.3787 (sep) |
| 2,000 | +0.2737 [+0.0753, +0.5099] | +0.4872 [+0.2730, +0.7494] | +0.3805 | 2.9564 / 3.0166 | +0.3936 / +0.4539 (sep) |
| 6,000 | +0.3971 [+0.1806, +0.6519] | +0.5253 [+0.3218, +0.7642] | +0.4612 | 3.0931 / 3.0765 | +0.5303 / +0.5137 (sep) |

⇒ **S\* = 250.** It is the only budget below 0, so the stop rule did not fire and ES-2 ran.
⚠️ **The readout gets worse with every step past 250, and has no early phase in which it improves.** At 250 steps it is
still ≈ P10, which is where the zero-initialised residual starts. Since 200 of those steps are warmup, R0 has barely
trained there. This is not classical overfitting with a sweet spot: **the MLP readout takes nothing transferable from
the pooled features at 540 clips.**

## ES-2 — A1 b0.08g50 ×2 seeds and A1x at S\* = 250

* **Reading 2 (the instrument), binding: STILL BROKEN.** A1x − A1 = **−0.0011 [−0.0046, +0.0026]**, not separated.
  ⇒ **K is not read** (reading 3 is not made).
* Reported, not read (eval-141 ADE@6s):

| arm | seed | ADE@6s | − P10 [95 % CI] | holdout ADE@6s |
|---|---|---|---|---|
| R0 | 0 / 1 | 2.5771 / 2.5648 | +0.0144 [−0.0361, +0.0717] / +0.0020 [−0.0493, +0.0543] | 2.9637 / 3.0035 |
| **A1** | 0 | **2.5110** | **−0.0517 [−0.0792, −0.0234]** | 2.9416 |
| **A1** | 1 | **2.5148** | **−0.0480 [−0.0764, −0.0201]** | 2.9393 |
| A1x | 0 | ≈ A1 − 0.0011 | (A1x − A1 above) | 2.9492 |

* The training-seed floor |A1_s0 − A1_s1| is **0.0038 m**, so |Δ| ≈ 0.05 is about 13× it.

⇒ At 250 steps the world-model arms sit about 5 cm below P10 at 6 s, at both seeds and on both the holdout and eval-141.
**But the time-shuffled control is just as good**, so the gain cannot be credited to temporal prediction.
Source (`ra1_hjepa.py` `forward_batch`): A1x permutes `z1` in time before `stack.loss`, so its IDM cannot fit
(`idm_term/L1` 14.42 vs 0.5862 at 6,000 steps). What survives the shuffle is **SIGReg** (a marginal over frames) and
**a world-model gradient reaching the adapter at all**. R0 has neither. That is H-RA1-ES-2, tested in ES-3.

## ES-3 — attribution (Addendum A) and the ridge probe (PREREG_RA1_RIDGE)

### ES-3: A0 (SIGReg + prediction, NO IDM) at 250 steps

Source: `raw/ra1_es/es3_attribution.json`, `code/es3_attribution.py`. Paired deltas on eval-141 ADE@6s, all on the same
2,399 windows.

| delta | seed 0 | seed 1 |
|---|---|---|
| A0 − P10 | −0.0249 [−0.0557, +0.0060] | −0.0247 [−0.0573, +0.0079] |
| **A1 − A0** | **−0.0268 [−0.0508, −0.0044] SEP** | **−0.0233 [−0.0480, −0.0004] SEP** |
| A1 − R0 | −0.0661 [−0.1186, −0.0197] SEP | −0.0500 [−0.0997, −0.0029] SEP |
| A0 − R0 | −0.0393 [−0.0838, +0.0023] | −0.0267 [−0.0685, +0.0116] |
| A1x − A0 | −0.0279 [−0.0504, −0.0063] SEP | — |

* Training-seed floors: R0 0.0124, A0 0.0003, A1 0.0038 m. Every separated delta above is more than 6× its floor.
* ⇒ **Addendum A, READING 2: the IDM term carries the gain.** A0 is not separated from P10 at either seed. Adding the IDM
  (A1 − A0) is separated at both seeds, although seed 1's upper bound is only −0.0004.
* **A1x keeps the gain even though its IDM cannot fit**: its IDM is asked to predict an action from time-shuffled latents
  of the same 3.6 s window.
  * **HYPOTHESIS:** that pressure still makes `z1` encode the window's kinematic level (speed, yaw-rate magnitude). That
    is ego-motion supervision of the visual latent, not world-model content. Ego labels at training time are admissible
    (*"labels may use ego; inference is vision-only"*).
  * Consequence: A1x as built is NOT a regression for this mechanism. That is why it cannot gate K here.
* SENS (reported, not read; it needs the gate): A1 − A0 yaw-rate sensitivity is +0.1285 [+0.0945, +0.1665] at seed 0
  and +0.0049 [−0.0024, +0.0137] at seed 1.

### Ridge probe (PREREG_RA1_RIDGE)

Source: `raw/ra1_es/ridge_probe.json`, `stack/scripts/ra1_ridge_probe.py`.
* All controls pass: P10 reproduces 2.5627 / 3.0245 exactly, the holdout and train sha256s match RA-1's, and train,
  holdout and eval are pairwise disjoint.
* 36,684 train windows (540 clips), 4,076 holdout windows, 2,399 eval windows.

| features | λ (holdout-chosen) | holdout − P10 | eval-141 − P10 |
|---|---|---|---|
| bias only | ∞ | −0.0024 [−0.0047, +0.0001] | +0.0005 [−0.0011, +0.0020] |
| f8 (no vision) | 1e-4 (⚠ grid edge) | −0.0072 [−0.0145, −0.0005] | −0.0012 [−0.0059, +0.0040] |
| mean (vision) | 10 | −0.0261 [−0.0469, −0.0057] | −0.0113 [−0.0279, +0.0042] |
| **mean+f8** | 10 | −0.0271 [−0.0480, −0.0066] | **−0.0115 [−0.0282, +0.0043]** |

* Vision − f8 on eval-141 = −0.0103 [−0.0272, +0.0058].
* ⇒ **Pre-registered READING 3: NO LINEAR SIGNAL.** The holdout-chosen ridge readout does not beat P10 on eval-141.
  It does beat it on the holdout, by a small separated margin of −0.027.
  ⚠️ A linear negative is not a learnability negative.
* The f8 set chose λ at the grid's lower edge. That is disclosed, and the edge is not extended after the fact. The f8
  set is ≈ P10 at every λ (holdout 3.0173–3.0221), so the edge does not change its reading.

## What the three tests say together

1. **A direct readout of the frozen pooled features cannot beat P10 at 540 clips.** Neither the MLP at any budget (R0)
   nor ridge at any λ manages it.
2. **The IDM objective on the adapter latent can.** A1 at 250 steps is −0.05 against P10 at both seeds, −0.025 against
   A0, and −0.05 to −0.066 against R0, all separated.
3. **The world model's temporal content is NOT shown to contribute.** A1x, whose temporal order is destroyed, is as good
   as A1. The pre-registered A1x gate therefore still fails, and **K stays unread.**
4. ⇒ **Next levers, ranked by measured effect:**
   * **(i) A1's own budget.** ES-4, Addendum B, is below.
   * **(ii) An instrument that removes the TRUE action information.** For example, A1 with its actions permuted across
     windows. This tests the mechanism found here. It would be a NEW pre-registration, and whether K may be read under
     it is the PI's call, because it replaces a pre-registered control.
   * **(iii) The IDM-only arm** (no prediction loss; needs code).
   * **(iv) Data scale.** Blocked: the 256-px frames of 3,972 clips are not on box 2, and C: on the dev box is nearly
     full.

## ES-4: A1's own budget (Addendum B)

Source: `raw/ra1_es/es4_budget.json`, `code/es4_budget.py`. Selection uses the seed-mean A1 HOLDOUT ADE@6s. eval-141 is
reported and is NOT K.

| steps | holdout (seed-mean) | eval-141 A1 − P10, seed 0 / seed 1 | A1x − A1 (seed 0) | gate | seed floor |
|---|---|---|---|---|---|
| **250** | **2.9405** | −0.0517 [−0.0792, −0.0234] SEP / −0.0480 [−0.0764, −0.0201] SEP | −0.0011 [−0.0046, +0.0026] | BROKEN | 0.0038 |
| 500 | 2.9626 | −0.0856 [−0.1290, −0.0397] SEP / −0.0760 [−0.1174, −0.0342] SEP | +0.0014 [−0.0103, +0.0132] | UNRESOLVED | 0.0095 |
| 1,000 | 3.0195 | +0.0089 [−0.0692, +0.0875] / +0.0172 [−0.0493, +0.0856] | **+0.0461 [+0.0033, +0.0969] SEP** | **OK** | 0.0083 |

⇒ **S\*(A1) = 250.** The pre-registered branch for this case reads: *"if it is best at 250, the IDM gain is also a
short-training effect, and that is reported as such."* It is a short-training effect.
⚠️ eval-141 alone would have picked 500, with a larger gain. That is the "tune on what you score" trap, and 500 is not
used.

⭐ **The informative row is 1,000 steps.** It is the first budget at which the time-shuffled control LOSES to A1,
separated, so the temporal content of the world model starts to matter. By then, though, the readout has memorised
enough that A1 is back at P10. The two effects cross between 500 and 1,000 steps.

⇒ **The next lever is the one that decouples them:** keep training long enough for the gate to pass while stopping the
readout from memorising. Two candidates:
* regularisation at about 1,000 steps (`--wd`, which is already a trainer flag, or readout dropout);
* data scale (more clips slow memorisation).

Each needs its own pre-registration with the A1x gate read at the same budget.

## ES-5: weight decay at 1,000 steps (Addendum C, pre-registered before its data)

Source: `raw/ra1_es/es5_wd.json`, `code/es5_wd.py`; runs `raw/ra1_es/st1000_wd0.3/`, `st1000_wd3.0/`.
* The launch gate's new G-DVB row `weight_decay` read back 0.3 and 3.0 from every built optimizer group (G-DVB 22/22).
* Its deliberate-regression test, an optimizer that ignores `--wd`, makes the gate refuse (`stack/tests/test_ra1_hjepa.py`).

| wd | holdout (seed-mean) | A1x − A1 (seed 0) | gate | eval-141 A1 − P10, seed 0 / seed 1 | seed floor |
|---|---|---|---|---|---|
| 1e-3 (ES-4) | 3.0195 | +0.0461 [+0.0033, +0.0969] | OK | +0.0089 [−0.0692, +0.0875] / +0.0172 [−0.0493, +0.0856] | 0.0083 |
| 0.3 | 3.0139 | +0.0325 [−0.0053, +0.0759] | UNRESOLVED | −0.0007 [−0.0743, +0.0732] / −0.0031 [−0.0644, +0.0601] | 0.0023 |
| **3.0** | **2.9834** | +0.0109 [−0.0086, +0.0335] | UNRESOLVED | −0.0671 [−0.1188, −0.0149] SEP / −0.0524 [−0.1019, +0.0002] | 0.0147 |

⇒ **The holdout selects wd 3.0. Pre-registered READING 1 FAILS there:** the A1x gate does not pass. ⇒ *"Regularisation
trades the instrument away."* **K is not read.**

**Four families, wd 3.0, seed 0 (T1-class; point values, CIs in the run's `eval.json` → `families`):**

| family | metric | A1 wd 3.0 | P10 |
|---|---|---|---|
| LONGITUDINAL | speed MAE m/s | 0.9518 | 0.9753 |
| | along-track MAE m | 1.8103 | 1.8681 |
| | accel MAE m/s² | 0.5274 | 0.5676 |
| LATERAL | heading MAE ° | 5.798 | 5.7652 |
| | yaw-rate MAE °/s | 3.6342 | 4.1255 |
| | curvature MAE 1/m | 0.0167 | 0.0185 |
| | cross-track MAE m | 1.2137 | 1.2426 |
| TACTICAL | from the trajectory | see `eval.json` | see `eval.json` |
| STRATEGIC | — | **PAUSED** (PI 2026-09-19) | PAUSED |

* Distance-keeping / lead-braking: **NOT COMPUTABLE** from the pooled cache, which has no lead-agent track (n = 2,399
  windows would have been scored).

## What five regimes say together

Across 250 steps, 500, 1,000, 1,000 + wd 0.3 and 1,000 + wd 3.0, **every regime in which A1 beats P10 is one in which
the time-shuffled control A1x is just as good.** The one regime where A1x clearly loses (1,000 steps, no decay) is one
where A1 does not beat P10. ES-3 found A1 − A0 separated. ⇒ On frozen pooled DINOv3 at 540 clips:
* the measurable gain over P10 (about −0.05 to −0.07 m at 6 s) comes from **IDM supervision of the adapter latent**;
* it does NOT come from temporal world-model content.

The pre-registered A1x gate therefore keeps H-HJ-K-1 unread, as designed. Both remaining levers need a decision:

* **Data scale (all 4,572 clips)**, the plan PREREG_HJ_STAGE0 already states. With 8× the clips, memorisation slows,
  and it is the only lever not yet tried that could let the gate and the gain co-exist.
  **Blocked:** the 256-px frames of 3,972 clips are not on box 2. Building them is a multi-hour CPU job (about 11 h at
  the measured 360 eps/h) plus about 140 GB, on the PI's own machine. The dev box C: has 6 GB free.
* **A control for the mechanism actually found:** A1 with its actions permuted across windows, which removes the TRUE
  action information that A1x keeps. It is cheap (minutes) but needs a new arm in code. It would test the IDM-latent
  claim, NOT reopen K. Reading K under a replacement control is the PI's call.
