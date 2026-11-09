# CONTROLS_EGOWM (measured; JSON: CONTROLS_EGOWM.json; script: controls_egowm.py)
CPU, 2 threads. Batch: 4 REAL val windows of 91 frames (zmean, act) from `ladder_read/cache/frames_s0.npz`. That file carries no speed, so `v` is PLANTED (12 m/s at the window start plus the floored integration of the real accelerations). Control 3 also uses planted decelerations that reach the floor. Model: the full `V8FMini` built from R35H's argv (synthetic P10 prior); trainer functions are called as they are.

| control | result | deliberate-regression arm |
|---|---|---|
| 1 Off identity | PASS. Baseline tree vs patched tree, same seed, same batch, one WM-only AdamW step: 579 of 579 arrays bit-equal (init, 25 loss terms, gradients, parameters after the step); no extra key. | 1b: with non-zero speed weights only 9 of 25 terms stay equal: the comparison goes red. |
| 1b Zero weights | PASS (kin and now). With zero speed weights all 25 loss terms and every control tensor at init are bit-equal; the only extra keys are `wm.stack.spd.{0,1,2}.weight`. | as above |
| 2 Reachability | PASS (kin and now). The trainer's `wm_only_live_check` passes; speed-channel gradient abs-sum per level 0.0014 / 0.0029 / 0.0035 (kin), all non-zero; non-WM parameters read exactly 0.0. | Speed input deleted (embedding times 0): the live check raises, RED. |
| 3 Speed values | PASS. Speeds the channel was handed, captured from the real loss call, equal a python hand integration at the three levels (cum stride 1, 3, 15; k 10, 10, 6): kin max abs error 4.8e-7 m/s, now 0.0; teacher-forced inputs equal the measured speed; the floor was engaged in the batch; primitive gives [1.0, 0.0, 2.0, 4.0]. | Integration sign flipped: RED (error 27 m/s). E-NOW taking a future speed: RED (error 12 m/s). |
| 4 Echo guard | PASS (kin and now). With non-zero speed weights, target latents of all 3 levels and the goal latents (a window lifted to levels 2 and 3) are bit-equal to the control and invariant to a different future speed; the action embeddings do differ, so the test can see an effect. | Speed leaked into the level-1 target latent: RED. |
| 5 Launch | PASS. All 5 argvs name `--wm-speed`; the BUILT stack rule reads back the declared mode; `bad_wm_speed` returns []; the R35H argv reads back off. Differences to R35H: `--wm-speed`, `--seed` (replicate), `--out` only. | The line that applies `--wm-speed` deleted (config built off): refused for all 5 arms. |
| 6 IMAG1B controls 2, 3, 4 | NOT RUN here: they belong to the read, after training. | |

The per-run token route belongs to the chain on box 2; it was not exercised.
