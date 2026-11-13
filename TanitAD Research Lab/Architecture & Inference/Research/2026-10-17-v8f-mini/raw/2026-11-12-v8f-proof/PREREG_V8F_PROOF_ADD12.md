# PREREG_V8F_PROOF_ADD12: SPATIAL, a world-model state that keeps location (2026-11-13)

Frozen before any data of this card is read. Parent: `PREREG_V8F_PROOF_ADD10.md` (LATVIS), whose committed next lever this is.
Theorems served: T6 (the state must carry what the plan depends on), T7 (imagination is admissible only where the state carries the thing to be imagined).

## 1. What was measured and what follows from it

MEASURED (LATVIS, T0, `latvis/VERDICT_LATVIS.md`): the level-1 state of the v8F world model is the MEAN of the 40 adapter tokens, projected to 128 numbers
(`stack/tanitad/models/v8f_mini.py`, `WMAux.z1`: `self.proj(adapt.mean(2))`). It decodes to the scene gist at skill 0.36 and shows no object; the token grid
decodes at 0.78. MEASURED (FULLRES, T0): the frozen tokens carry the lead vehicle (gap R2 0.40 to 0.46, presence AUROC 0.77 to 0.84); MEASURED (LATSEM of the
programme): the lead gap is at no level of the world-model state. The mean over cells is where location is lost.

## 2. The lever (one)

The level-1 state keeps the 4x10 cell layout: each cell is projected to `c` numbers and the cells are concatenated, state width `40 c` (c = 16: 640 numbers).
Nothing else moves: same tokens, same readers S / T / O, same levels 2 and 3 (width 128), same losses and weights, same data, same steps.

## 3. Question

Q1 (retention). Does a state that keeps location retain the objects that the mean state loses?
Q2 (imagination). Is the predicted state at 2.0 s closer to the true future than a copy of the present, in pixels and for the lead vehicle, beyond what ego motion alone explains?
Q3 (cost). Does training with this state cost any of the four metric families at T1-kin (kinematic rollout on the model's own actions)?

## 4. Two stages

### Stage 0 (dev box, T0, no model trained; runs first)

Purpose: prove the instrument for Q1 and Q2 on a state built WITHOUT training, and fix the floor and the ceiling of the lead read before any arm exists.
* State S0g: per cell, the first c = 16 principal directions of the frozen pool4x10 tokens (fit on the training folds only), concatenated: 640 numbers per frame.
  State S0m: the mean token over the 40 cells, first 128 principal directions (the stand-in of the present design).
* Predictor: cross-fitted ridge from (state at t, ego actions t .. t+2.0 s) to the state at t+2.0 s, 5 folds by episode; the ridge strength is chosen inside the training folds only.
* Windows: the eval-141 windows of LATVIS (139 episodes). Logged future actions are an input, so every number of this stage is T0, a world-model diagnostic.
* Reads, for S0g and S0m:
  * r-a retention: lead gap R2 and lead presence AUROC from the present state (probe of FULLRES: ridge and kernel ridge, cross-fitted by episode).
  * r-b imagined lead: lead gap at t+2.0 s (windows with a lead at both times) read from (i) the predicted state, (ii) the present state (copy), (iii) the true future state (ceiling),
    (iv) ego only: speed, acceleration, yaw rate at t and the logged actions to t+2.0 s (floor: what ego motion alone explains). The probe is fitted on true future states of the training folds and applied to (i) to (iii).
  * r-c imagined pixels: the LATVIS decoder read at 2.0 s (decoded predicted state against decoded present state, against the true future frame), if the 64x160 frame targets are rebuilt within budget; else named as not run.
* Outcome names (stage 0, per state):
  * `S0-LEAD-KEPT` if r-a gap R2 of the state minus the ego-only R2 is above 0 with the paired episode-cluster bootstrap interval excluding 0; else `S0-LEAD-LOST`.
  * `S0-IMAGINE-LEAD` if r-b (i) minus (ii) is above 0 AND (i) minus (iv) is above 0, both intervals excluding 0; `S0-IMAGINE-EGO` if (i) beats (ii) but not (iv); else `S0-IMAGINE-FLAT`.
  * The stage-0 headline is the PAIR (S0g against S0m): `GRID-ADDS` if S0g minus S0m is above 0 with the interval excluding 0 on r-a gap R2, else `GRID-EQUAL`.
* Controls (each must read its known value, or the stage is not quoted):
  * s1: rows permuted within fold: every R2 at or below 0.02.
  * s2: a planted object (the f5 construction of FULLRES: a synthetic cell-local signal with a known position) is read at AUROC above 0.95 from S0g and at or below 0.60 from S0m.
  * s3: the ceiling (iii) is at or above (i) and (ii) on r-b; the ego-only floor reproduces the FULLRES value E (gap R2 0.133) within 0.03 on the same rows.
  * s4: predicting with horizon 0 returns the present state (difference exactly 0).
* A negative of stage 0 is a negative about a LINEAR predictor on an untrained state. It does not close stage 1.

### Stage 1 (pod A40, after the HIERREP chain; each arm under its own selfcheck gate token)

* Arms, from scratch, 19,164 steps, argv of the HIERREP HIER arm plus the lever:
  * G: `--wm-state grid --wm-cell-dim 16` (state 640; level-1 predictor width 256).
  * W (capacity control): `--wm-state mean --wm-dim 640` (a mean state of the same width and the same predictor; location lost as today).
  * Baseline: the HIERREP HIER arms of the same seeds (mean, 128), trained by the pinned tree on the same cache. Seeds 0 and 1 for G and W: 4 new arms.
* Noise floor: the six HIER seeds of HIERREP (15 pairs) give the run-to-run spread of every T1-kin family cell; a G minus HIER difference inside that spread is not an effect.
* Reads: the LATVIS read (retention; imagination in pixels at 2.0 s) and the r-a / r-b reads of stage 0 on the trained states (level-1 state of G, W, HIER), T0;
  the four families at T1-kin strict (`--limit causal`), eval-141, read tool of HIERREP, STRATEGIC reported PAUSED (PI item 25).
* Outcome names (stage 1), each needs BOTH seeds:
  * Q1: `Z-NEAR-G` if the decode skill of G is at or above 0.90 of the token grid skill; `Z-ABOVE-MEAN` if below that bar but G minus HIER is above 0 with the interval excluding 0; else `Z-FLAT`.
  * Q2: `IMAGINE-LEAD` if on r-b the learned rollout of G beats the copy AND the ego-only floor (intervals excluding 0); `IMAGINE-EGO` if it beats the copy only; else `IMAGINE-FLAT`. Pixels: the LATVIS names.
  * Q3: `NO-COST` if no family cell of G minus HIER is worse with the two-level interval (seeds, then episodes) excluding 0 and outside the six-seed spread; else `COST-<family>`. A family cell that is better under the same rule is reported as `GAIN-<family>`.
  * Attribution: an outcome is assigned to LOCATION only if G differs from W in the same direction; if W equals G it is assigned to WIDTH.
* Controls:
  * c1 identity: the patched tree with the default `--wm-state mean` reproduces the pinned tree bit-for-bit (same state-dict keys; same loss trace over 200 steps, same seed, same device).
  * c2 reachability: with `grid` the new projection and the adapter both get a non-zero gradient from the world-model term; with `--w-wm 0` that gradient is exactly 0.0;
    the declared-against-built check refuses a declared `grid` on a built `mean` (deliberate regression: the gate must FAIL).
  * c3 location: permuting the cells of one frame leaves the mean state unchanged to 1e-5 (tokens before the position term) and changes the grid state.
  * c4 collapse: the participation ratio of the level-1 state of G on eval-141 is reported; below 32 the arm is flagged collapsed and Q1 / Q2 are not quoted for it.
  * c5: the LATVIS controls v1 to v3 on the new states.
* Fallback fixed now: if G seed 0 diverges or is flagged by c4, ONE retry with `--wm-cell-dim 8` (state 320), named G8.
* Budget: 4 arms (5 with the fallback). ESTIMATED from the earlier pod (59 min per arm on an RTX 4000 Ada): 4 to 6 A40-h; replaced by the measured HIERREP time per arm.

## 5. Both outcomes, committed

* `Z-NEAR-G` and `IMAGINE-LEAD` with `NO-COST`: the grid state becomes the proposed default of the next flagship arm; next lever = imagination rollouts for planning on this state (T7), and decoders for levels 2 and 3.
* `Z-NEAR-G` or `Z-ABOVE-MEAN` with `IMAGINE-FLAT` / `IMAGINE-EGO`: the state keeps objects and the predictor does not move them. Next lever = a predictor that attends across cells and time on the same state, own card.
* `Z-FLAT`: the regulariser removes cell content. Next lever = predict the frozen token grid itself as the target (no learned state to collapse), own card.
* Any `COST-<family>`: the lever is not promoted; the cost is reported as written and the next lever above still runs.

## 6. Limits

Stage 0 states are linear and untrained. Stage 1 has two seeds per new arm. The world model is an auxiliary: the readers do not read its state, so any T1-kin effect passes through the adapter gradient only.
The pixel read is small in absolute terms (LATVIS: about 2 % of the pixel error). T0 numbers are world-model diagnostics, never driving performance.
