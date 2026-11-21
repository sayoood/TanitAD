# PREREG_V8F_PT3N: does a decoder fitted on 1,053 episodes change PLANTEST3's NO-GAIN? (Master Mind, 2026-11-09, written before any PT3N code or run)

## Why
`VERDICT_PLANTEST3.md` read NO-GAIN on every decoded arm, with DECODER-WEAK raised: its ridge decoder was fitted on 53 to 70 TUNE episodes and read speed at held-out R2 0.271 / 0.314. DECN (`decinfo/decn/RESULT_DECN.md`, T0) read the same ridge, same held-out 17 episodes, at 1,053 fit episodes: speed 0.680 / 0.647, yaw rate -0.043 / +0.105 (no interval). DECN's first run is INVALID as run on its shuffled-target control, on 10 of 216 reader rows; all 10 are MLP rows and all 108 ridge rows pass, and the ridge is the reader this card uses. DECN's card commits this rerun under its N-LIMITED outcome. This card makes the rerun a stand-alone pre-registered experiment, so its validity does not rest on how DECN's control is resolved. Tier: **T0, an oracle-goal diagnostic, never a driving result.**

## Objects (one change: the decoder's fit set)
The PLANTEST3 harness and windows, unchanged: M2b s0 and s1 world models, zero training, 1,976 windows, TUNE and SCORE halves (salt `plantest2-v1`), 71 SCORE episodes / 994 windows, G-TRUE, G-RAND, HOLD, the TRAIN action prior, the SEED + REG(lambda) solver, the metric code.

**The decoder.** `D_1`, `D_2`: the same closed-form ridge (1e-3 times the sample count), the same targets (speed, yaw rate, each divided by its standard deviation over the fit samples), fitted per training seed on encoded latents of **N1053** = the 53 TUNE inner-fit episodes plus the 1,000 streamed train clips of DECN (`decinfo/decn/clips_extra.json`, `wmact/train_stream/clips141.json`). The 17 inner-held-out TUNE episodes stay held out, so the gate read is the same held-out set as DECN and PLANTEST3. One decoder per seed serves both the gate and planning (PLANTEST3 refitted on all 70 TUNE episodes for planning; here the fit set is N1053, as the DECN card states).

## Arms
FLAT1 DEC-END, plain and SHRINK(c), c as PLANTEST3 (0.496 / 0.526). KIN-CEILING beside, under the new decoder. TD3 and DEC-PATH are not run.

## Tuning and order
Lambda chosen on TUNE windows under G-TRUE per (form, seed, family) from {0, 0.1, 1}, as PLANTEST3. Then SCORE: G-TRUE and G-RAND at the chosen lambdas, planner seed 0; G-TRUE planner seed 1 as replicate. CPU only, at most 2 workers, RAM guard 6 GB.

## Read (paired episode-cluster bootstrap over the SCORE episodes, B 2,000, seed 0, both training seeds required)
Deciding, per family: action MAE as PLANTEST3 (LONGITUDINAL acceleration, LATERAL yaw rate, 12 logged actions). Beside, no bar: speed, heading and ADE at 2.4 s; distance of the plan from HOLD; chosen lambda; the planner-seed replicate; and PT3N minus PLANTEST3 on the same windows, paired, per arm, seed and family. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

## Committed outcomes (per form and family, both seeds; PLANTEST3's table unchanged)
* **PLANNABLE:** G-TRUE minus HOLD is separated-better and G-TRUE minus G-RAND is separated-better.
* **GOAL-BLIND:** G-TRUE minus G-RAND is not separated.
* **NO-GAIN:** G-TRUE minus G-RAND is separated-better, and G-TRUE minus HOLD is not separated-better.
* **HOLD-COLLAPSE (flag):** the plan's mean absolute distance from HOLD is below 10 % of the prior std; the row does not count as planning.
* **DECODER-WEAK (flag on the whole read):** held-out R2 on encoded latents below 0.70 for speed or below 0.50 for yaw rate at either seed.

## What each outcome triggers (committed now, Rule Zero)
* **PLANNABLE in either family:** this cost, with the N1053 decoder, becomes the HWM-PLAN level-1 cost and is rerun on the ladder world models.
* **NO-GAIN on every row, KIN-CEILING beats HOLD in a family:** the world model's action response is the limit; next is the action-sensitivity lever on the ladder arms.
* **NO-GAIN on every row, and KIN-CEILING does not beat HOLD or DECODER-WEAK:** the decoded quantities are still the limit; next is a nonlinear multi-frame readout in the cost, fitted at N1053 (DECN's MLP, four frames, reads yaw rate 0.570 / 0.441).
* **GOAL-BLIND:** action sensitivity first.
* **Every outcome:** PLANTEST3's NO-GAIN is stated together with its 53-to-70-episode decoder, and the harness is read on the ladder world models when they finish.

## Controls (the read is INVALID if one fails)
1. **Decoder tie to DECN:** on the 17 held-out episodes, the N1053 `D_1` reads within 0.01 of DECN's ridge m = 1 row at N1053, speed and yaw rate, both seeds.
2. **Reference:** the N53 fit through this code reads within 0.001 of PLANTEST3's gate (speed and yaw rate, both seeds).
3. **Harness reproduction:** with PLANTEST3's stored planning decoder loaded, this runner reproduces PLANTEST3's stored SCORE plans bit for bit on one full unit (M2b s0, FLAT1 DEC-END plain, G-TRUE, lambda 1, planner seed 0).
4. **Split:** no train clip is an eval clip, and the decoder fit reads no SCORE or inner-held-out window (asserted on ids; a mutation goes red).
5. **Clips:** the 1,000 shards are DECN's, md5-verified at pull; their count and order are asserted.
6. **Isolation:** PLANTEST3's `tune_choice.json` and `runs/` are unchanged by this run (sha256 of `tune_choice.json` before and after; every write goes under `plantest3/n1053/`).

## Log
* 2026-11-09 09:34:12 frozen: prefix 5326 bytes, sha256 de94f72c51d0f4a2da743285e369a9f333e128f673d7a3ff67aaa2618944a0fb (everything above this section). No PT3N code or run existed at this time.
