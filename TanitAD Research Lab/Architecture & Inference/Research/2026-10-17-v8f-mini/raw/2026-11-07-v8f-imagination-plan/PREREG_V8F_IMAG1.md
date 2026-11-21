# PREREG_V8F_IMAG1: the Imagination Audit, first pass (Master Mind, 2026-11-07, written before any IMAG1 code or statistic exists)

## Why
`research_imagination/GAPS.md` lists nine gaps in what the v8F world model imagines. Our yardsticks so far (SKILL against "nothing changes", a nearest-real-frame decoder, single-target probes) cannot say which level holds what, whether a prediction beats copying the present, or whether a level answers the action with the right size. `research_imagination/PLAN_IMAGINATION.md` §2 sets out a battery taken from how JEPA predictors are validated elsewhere (REPORT_A F3, F4, F5, F7, F9, F12; REPORT_B top 1 and 2). This card runs modules IA-0, IA-1, IA-2, IA-3 and IA-6 (timescale index, cross-level consistency). Tier **T0**: a world-model diagnostic, never a driving claim.

## Data and models
* **Deciding set:** eval-141 (`theory/wmact/latents.pt`: 1,976 windows, 141 clips, the LATSEM frame bank and windows). Targets come from the label files LATSEM2 / LATSEM2B already built for these windows, plus speed, acceleration and yaw rate from the v8F loader. Where a target has no label for a window, that window is dropped from that target only, with the n reported per row.
* ⚠️ Speed and lead-gap targets on these windows were read before by LATSEM / LATSEM2 (other statistics). Rows for those targets are labelled **REPEAT TARGET** in the result; none of them alone may be a new finding.
* **Beside (optimistic, not deciding):** WMACT's TRAIN windows (`theory/wmact/latents_train.pt`, 1,833 windows, 141 clips).
* Models: M2b s0 and s1 (`E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s{0,1}/ckpt_last.pt`). Zero training of the model. Only probes are fitted.
* Levels L1 / L2 / L3 (0.2 / 0.4 / 0.8 s steps). Horizons k = 1, 2, 3 steps of each level, as far as the window holds the true future.

## Targets (the four families plus scene)
* LON: speed, acceleration, lead gap, lead closing speed, TTC (lead present only).
* LAT: yaw rate, curvature.
* TAC: the 7-class manoeuvre, lead present.
* SCENE: agents within 30 m.
* STRATEGIC: PAUSED (PI item 25); not read.

## Estimators (fixed here)
* Probes: ridge (λ 10) and a 2-layer MLP (hidden 64, 3 probe seeds, early stop on an inner split of the fit fold). Both are reported. Classification: logistic (C 0.1) and the MLP; the score is balanced accuracy minus the majority rate. Continuous scores are **NMSE** = MSE / Var(y) on the scoring fold (1 = the marginal mean).
* 2-fold episode split, as in LATSEM2; probes are fitted on **ENCODED latents of the fit fold only** and never on predicted latents.
* Intervals: episode-cluster bootstrap, B 2,000, seed 0; paired (the same episodes) for every difference.
* Two-part separation (LATSEM2B): the bootstrap 95 % interval excludes the null on the claimed side AND the point lies beyond the 97.5th percentile of a permutation reference with R = 199. For differences between two latents of the same window, the reference flips the sign of whole episodes' differences (a paired sign-flip permutation). For single-latent probe rows, it is LATSEM2B's derangement of labels across episodes inside the fit fold.

## Modules and committed outcomes

**IA-0 health (a gate).** Per level and seed:
* effective rank (RankMe) of encoded and of predicted latents;
* retrieval MRR of the encoded true future at k = 1 among all windows of the same clip, with the predicted latent as the query, against exact chance (the mean of 1/rank under a uniform rank).

A level whose MRR is not two-part separated above chance at both seeds is **UNHEALTHY**. Its IA-2 rows are reported but not read.

**IA-1 capture.** For each target, level and seed: the NMSE (or balanced accuracy minus majority) of the probe on the encoded latent at time t, with a random-init row (the same architecture, untrained weights) beside it.
* **ENCODED** = separated below NMSE 1 (above 0 for classes) at both seeds.

**IA-2 probe transfer.** For each target, level, horizon and seed, three NMSEs with the same probe (fitted on encoded latents at t + k):
* on the PREDICTED latent;
* on Copy-Last (the encoded latent at t used as the prediction);
* on the Oracle (the encoded true future).

Each row gets exactly one label:
* **UNREADABLE**: the Oracle is not separated below 1 at both seeds. The lever is capture, not prediction.
* **IMAGINES**: the predicted NMSE is separated below Copy-Last (paired) at both seeds.
* **WORSE-THAN-COPY**: the predicted NMSE is separated above Copy-Last at both seeds.
* **COPIES**: everything else that is readable.

A kinematic null (speed from constant acceleration over the last 0.4 s of logged speed; yaw rate held) is reported beside. It carries no bar.

**IA-3 action audit.** From each window's logged action, contrast the longitudinal component (±1.0 m/s², the LATSEM accelerate/brake contrast) and the lateral component (yaw rate ±0.25 rad/s), with everything else held. Then:
* **Gain** = (imagined effect, decoded by the IA-2 probe on predicted latents) / (the physical effect: Δv = Δa·h for speed, Δψ̇ for yaw rate), per level and horizon.
* **Direction**: the cosine between the imagined and physical 2 × 2 Jacobians (rows speed and yaw rate, columns the longitudinal and lateral actions).
* A row **ANSWERS** when, at both seeds, the gain is in [0.5, 1.5], its sign is right, it is two-part separated from 0, and the cosine is above 0.7.
* **OVERSHOOTS** when the gain is separated above 1.5. **WEAK** when it is in (0, 0.5) and separated from 0. **WRONG-SIGN** when it is separated below 0. **INERT** when it is not separated from 0.

Beside these, with no bar:
* the ATM inverse-probe 2 × 2: a ridge from (z_t, z_{t+k}) to the logged action, fitted on real encoded pairs and on predicted pairs, each scored on both kinds; R² per cell;
* the IWM action bank: 16 candidate actions per window (the logged one plus 15 drawn from other windows of the deciding set, seed 0); the MRR of the logged action's prediction when ranked by distance to the encoded true future, against exact chance;
* the turning vs straight share of M2b's world-model training windows (|mean yaw rate| above 0.1 rad/s over the window). This is a count only; E7 needs it.

**IA-6 hierarchy.**
* **Timescale index** TSI_l = (mean IA-1 NMSE over the fast targets: acceleration, yaw rate, closing speed) − (mean over the slow ones: speed, lead present, the manoeuvre, agents within 30 m; classes as 1 − balanced accuracy). The outcome is **SPLIT** when TSI(L3) − TSI(L1) is separated above 0 at both seeds, else **NO-SPLIT**.
* **Cross-level consistency:** the decoded Δspeed from L2 one step against L1 two steps (both 0.4 s), and L3 one step against L2 two steps (0.8 s). The slope comes from a paired regression per episode cluster. The outcome is **CONSISTENT** when the slope's interval contains 1 and excludes 0 at both seeds; otherwise the slope is reported as is.

## What each outcome triggers (committed now)
* lead gap UNREADABLE (IA-2 Oracle) at every level → **E3 CAPLADDER** (the capture ladder) is the next card.
* lateral IA-3 WEAK, INERT or WRONG-SIGN at every level → **E7 DIVRS** (action-diverse resampling), with this run's turning share as its design input.
* longitudinal L1 OVERSHOOTS → an L1 residual parameterisation arm (Δ on Copy-Last) is carded after E7's design.
* **NO-SPLIT** → MRATE (`50667a79`) is read for TSI as well, and a level-factor sweep is carded.
* a level UNHEALTHY → its rows are not read. The next lever is a collapse diagnosis of that level (SIGReg weight, predicted-latent rank over training) before any arm.
* L2 / L3 IMAGINES on readable LON rows while L1 COPIES or is WORSE-THAN-COPY → this agrees with WMACT's "L1 DOES NOT FIT" and points at MRATE's L1 spacing. No new arm.
* any outcome → **E9 REL** (the reliance arm) is carded from the levels and targets this run finds IMAGINES or ANSWERS.

## Controls that must read known values (the run is INVALID if any fails)
1. The Oracle NMSE of every probe on its own fit-fold encoded latents is below the scoring-fold Oracle (no leakage the wrong way). Reported, not a bar.
2. **Real-vs-real:** IA-2 with the "predicted" latent replaced by the encoded latent at t + k from another window of the same clip, offset by 0.2 s. Its gap to the Oracle must be separated above 0 (an offset window is not the future), and it must not be labelled IMAGINES against the Oracle.
3. Calibration of the two-part rule: 20 extra permutation draws (seeds 1001–1020) scored as if real. Per family, at most 5 % of rows may come out separated.
4. A synthetic test with a planted effect (a latent that is the future plus noise; one that is the present; one that answers the action with a gain of 1.0 and another with 2.0) must give IMAGINES, COPIES, ANSWERS and OVERSHOOTS respectively before the real run.

## Implementation choices
The implementer writes `theory/imag1/IMPL_CHOICES_IMAG1.md` (data paths, label sources per target, window n per target, the exact action-contrast mechanism, the inverse-probe and bank details) and records its sha256 **before any statistic is computed**. Anything not fixed in this card is fixed there or reported as a deviation.

## Log
* 2026-11-07 14:50:03 frozen: prefix 9212 bytes, sha256 14846eef59472b64a27063dcc72a4d52013d0ae839fcbd209e38db5805c46eb8 (everything above this line). No IMAG1 code or statistic existed at this time.
