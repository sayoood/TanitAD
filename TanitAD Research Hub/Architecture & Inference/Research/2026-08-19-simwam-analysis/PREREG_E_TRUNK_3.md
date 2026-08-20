# PRE-REGISTRATION — E-TRUNK-3: was the environment information LOST, or never ACQUIRED?

`PRE-REGISTERED 2026-08-20, BEFORE ANY LADDER NUMBER WAS COMPUTED.` Both
outcomes and both consequences are committed below. **T0-DIAGNOSTIC.**
**0 GPU, 0 retraining** — the caches already exist.

---

## 1. Why this exists

E-TRUNK-2 measured that the v6 operative latent decodes **no** environment
property (and no ego-motion property) above its controls, while DINOv3 through
the **same 40× pool** decodes `lead_gap_m` at R² +0.379, `ego_speed` at +0.696
and lane occupancy at AUC .83. The spectrum at admissible *n* shows
**participation ratio 4.90 vs 40.77** (8.3×) and **top-8 share 0.806 vs 0.348**.

⛔ **That says WHAT, not WHY, and the two candidate WHYs demand opposite fixes.**

| world | mechanism | the fix it implies |
|---|---|---|
| **A — LOST** | the action-conditioned prediction objective *actively removes* environment content: other agents are stochastic, so encoding them raises prediction error and dropping them lowers it | **anchor or regularise** the encoder (aux perception head, EMA target, variance regularisation) — the architecture is fine, the objective needs a counterweight |
| **B — NEVER ACQUIRED** | the encoder never learned to see: too little data / capacity / the objective simply never induces perception at all | **freeze a strong pretrained encoder** (DINO-WM's actual recipe, REF-D's bet); no counterweight will conjure what was never there |

## 2. The experiment

**Nine banked cell caches on the IDENTICAL 5,617 frames, stride 4, 130
episodes — only the checkpoint moves:**

`step 2000, 9000, 9250, 10000, 11250, 12000, 14000, 16000, 18000` (+ the
`20000` cache already measured).

For each step, run the **unchanged** E-TRUNK-2 battery on the `cells` arm:
same targets, same **episode-disjoint** folds, same dual-ridge with
**scale-normalised Gram**, same **episode-cluster bootstrap**. Plus
`spectrum_report` at admissible *n* (ceiling 2048 ≫ the 1024 bar) for
participation ratio, effective rank and top-8 share at every step.

⚠️ **Controls are unchanged and re-reported at every step** — `C-EGO` and
`C-PIXEL` do not depend on the checkpoint, so they are **constant lines** across
the ladder and any drift in them would indicate a harness fault, not a finding.
That is the built-in falsifier for this run.

## 3. ⛔ Committed decision rule — written before the numbers

Primary readout: **`lead_gap_m` R²** and **`left/right_occupied` AUC** on the
`cells` arm, versus step. Secondary: **participation ratio** versus step.

| observation | verdict | consequence, committed now |
|---|---|---|
| decodability **starts higher and falls** (any target separated at an early step, at/below its control by 20 k) | ⭐ **WORLD A — LOST** | The objective is destroying content. **Route 2/3** — aux perception anchor and/or anti-collapse — becomes the primary fix, and a frozen encoder is no longer *required*. Also makes the S-W loss weights the first thing to inspect (`o6_sigreg` weight included). |
| decodability is **flat at ~0 from step 2000** | ⭐ **WORLD B — NEVER ACQUIRED** | No counterweight will help. **Route 1** (frozen strong encoder + trained predictor) becomes the primary recommendation, and REF-D's frozen-prior bet is directly supported. |
| decodability **rises** across the ladder | **NEITHER — the trunk is still learning** | The 20 k reading is a snapshot of an unfinished process. Do **not** act on E-TRUNK-2 until 30 k; re-run this ladder at 30 k first. |
| participation ratio **falls** while decodability falls | collapse is the mechanism, not merely correlated | strengthens the case for an explicit anti-collapse term over an aux head |
| decodability falls while participation ratio **rises** | ⚠️ the deficit is **not** dimensional collapse | drop the collapse framing entirely; the content is being *replaced*, not compressed |

⚠️ **The mixed case is real and is NOT a get-out.** If some targets fall and
others are flat, the verdict is reported **per target with its n**, and the
strongest committed claim is restricted to the targets that moved.

## 4. What this experiment CANNOT settle

* ⛔ **It cannot exonerate the objective on its own.** World B is consistent with
  *"this objective never induces perception"* AND with *"too little data"* —
  those separate only by training a different objective on the same corpus.
* ⛔ **It cannot speak to closed-loop driving.** T0-DIAGNOSTIC throughout;
  decodability is a representation property, never driving performance.
* ⛔ **It is one seed, one corpus, one architecture.** 130 episodes, and the
  probe is **linear** — non-linearly encoded content reads as absent.
* ⚠️ **Step 2000 is already 6.7 % through S-W.** If content were present at
  initialisation and destroyed within 2 k steps, this ladder cannot see it. The
  earliest available cache bounds the claim, and that bound is stated, not
  hidden.

## 5. Falsifiers built in

1. `C-EGO` and `C-PIXEL` **must** be flat across steps (they never see the
   checkpoint). Any drift ⇒ harness fault, discard the run.
2. `C-EGO`→`ego_speed` **must** read ≈ 1.0 at every step (identity map). It read
   **−1.81** under an absolute λ grid and **+0.9845** normalised — this is the
   check that caught that defect and it stays in.
3. `C-MEAN` is 0 / 0.5 by construction.

## 6. Manifest

| artifact | where |
|---|---|
| this pre-registration | `…/simwam-analysis/PREREG_E_TRUNK_3.md` |
| result (to be written) | `…/simwam-analysis/E_TRUNK_3_LADDER.md` + `raw/e_trunk3_ladder.json` |
| harness (unchanged from E-TRUNK-2) | `…/simwam-analysis/code/e_trunk2_probe.py` |
