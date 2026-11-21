# PREREG_V8F_D1: lateral-event-stratified training batches (written 2026-10-23, before its data)

**Status: PRE-REGISTERED, NOT RUN.** Its box-2 run follows `PREREG_V8F_HIST`'s chain and needs the PI's go (PI, 2026-10-23: *"our goal is to accumulate achievements and not fails … Dont stop at fails"*). Instruments are built under this file before any run.

## Why this lever

* **T's lateral constraints are the measured bottleneck, and three arms have not moved them.**
  * The arms: WC (loss weight), PGT (O's exposure), GRID (scene resolution). latcons RMSE moved −1 % to +1.2 % (`RESULT_V8F_WC.md`, `RESULT_V8F_PGT.md`, `RESULT_V8F_GRID.md`).
  * With the GT code ADE6 falls from 2.175 to 0.971 m at M2b_s0 (T0, GT-code oracle).
* **The events latcons describes are rare in what T trains on.**
  * The turn and lane-change geometry slots carry information only on turn and lane-change windows.
  * On the eval grid the lateral manoeuvre support is 1,910 LANE_KEEP against 47 TURN_L, 150 TURN_R and 0 lane changes (`eval_M2b_s0.json` `raw.tactical.lat.support`, 2,107 labelled windows; turns 9.3 %).
  * WC measured that a heavier loss weight fits the training set but does not generalise (train constraint loss −18–25 %, eval present-slot error −2.4 %). More exposure to the rare event, not more weight on it, is the untested data-side lever.
* **The plan names it.** `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13.6 row 1, after more history frames:
  * "D1 rebalancing"; the plan's data list, item D1 (line 381): *"event-stratified sampling … so that rare tactical classes are 15–25 % of every batch. Eval untouched."*
* **It is free:** the same clips, windows, steps and eval; only the order and multiplicity of training windows change.

## The arm (ONE lever: the share of lateral-event windows in a training batch)

* **The base B is fixed now, before `PREREG_V8F_HIST` has any data:**
  * if HIST's FINAL read is **SUCCESS**: **B = M2b-h10** (`--hist 10 --window-start 3`), compared against its two seeds, F from them;
  * otherwise (any FAIL, including a seed-0 STOP): **B = M2b**, the banked runs, F = 0.1207 m (2F 0.2415).
* **E = B + `--event-share 0.2`.**
  * A training window is an **event window** iff its GT lateral manoeuvre class at NOW (the label T's `h_lat` head trains on) is TURN_L, TURN_R, LANE_CHANGE_L or LANE_CHANGE_R. These labels are offline GT and are used for SAMPLING only (labels may use ego; inference is unchanged and vision-only).
  * Within each sampler chunk, the chunk's windows are redrawn to the SAME count, with replacement, with weights set so the expected event share is 20 %. The middle of the plan's 15–25 % band, committed now.
  * A chunk with no event window is left as it is.
  * The draw is seeded by (seed, epoch), so it is resumable.
  * Everything else is B's: the clips (4,512, RA-1 holdout withheld), the window list, the step budget (19,164), batch 64, the held-age draws' rule, the losses and weights.
* **Default `--event-share 0` is today's sampler, bit for bit** (a test pins it).
* ⚠️ **Declared with the lever, each MEASURED in the build:**
  * the training set's natural event share;
  * the realised share per batch;
  * how many distinct event and non-event windows an epoch touches (oversampling repeats event windows, and undersampling skips some lane-keep windows each epoch).
* ⚠️ **The rebalance feeds all three layers' training, not T alone.** A PASS is "the stack trained on lateral-event-stratified batches". The mechanism is read on T.

## The reading (the instrument and bars of `PREREG_V8F_GRID` / `PREREG_V8F_HIST`, with E in place of G / H)

* **Tier:** T1-class open loop (eval-141, 2,399 windows, 141 episodes); the mechanism read uses the absmask and GT-oracle evals (T0).
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes; training-seed variance enters through F.
* **PRIMARY**, per seed s: δ_s = ADE6(E_s) − ADE6(B_s); PASS iff δ_s < 0, separated, |δ_s| > 2F.
* **MECHANISM (M-LAT):** T's slot-pooled present-slot latcons RMSE falls ≥ 12.5 % relative to B_s, separated. It decides the reading of a FAIL, never the verdict.
* **Stage 1 (seed 0):** CONTINUE iff seed 0 PASSES; otherwise STOP with the seed-0 verdict. **Stage 2:** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | the T1-kin read of E, then stack the next lever on E: LoRA on DINOv3 (needs the PI's AR-2 go), under its own PREREG |
| **FAIL-MECHANISM-OK** | latcons fell ≥ 12.5 % at every seed read; driving did not move beyond 2F | T is more accurate on events but O does not convert it: an O-side arm (O's longitudinal head read against T's own code; `RESULT_V8F_PGT.md` section 4), under its own PREREG |
| **FAIL-NO-MECHANISM** | latcons did not fall ≥ 12.5 % | neither the frozen tokens' space, their time nor the data balance limits T: the encoder itself, LoRA on DINOv3 (AR-2), with D3 (more data) behind it |

* **Reported beside it, deciding nothing:**
  * the four families;
  * T's present-slot RMSE per group and per latcons slot;
  * ADE6 and the lateral family on the turn windows (GT TURN_*) and on the LANE_KEEP windows SEPARATELY. The trade-off: oversampling turns may cost the majority;
  * lateral macro-F1 and per-class F1;
  * Λ and the GT-code oracle ADE6;
  * the realised event share and the epoch coverage counts.
* **Refusals (rc 2):**
  * B's base rule contradicted by HIST's files;
  * E's argv not B's plus `--event-share 0.2`;
  * built `event_share` not 0.2, or recorded realised share outside [0.15, 0.25];
  * a parameter count different from B's (the lever adds none);
  * runs not on one clip list or step count;
  * evals not on one grid;
  * GT codes that differ;
  * any loss weight different from B's (`w_diff`).

## Proofs, before any run

* **Reachability:**
  * `--event-share 0.2` → config → `ChunkedSampler` → the batches the trainer consumes. MEASURED on the real train labels: the realised event share is 0.2 ± tolerance, while the default sampler reproduces today's batches bit for bit.
  * A mutation whose flag never reaches the sampler must turn the share test red.
* **G-DVB:** the new flag has its declared-vs-built entry; `built_levers` / the run's info records `event_share` and the realised share.
* **The eval is untouched:** a static check that no eval path reads `event_share`.
* **Chain and reader:**
  * as HIST, plus the base decision from HIST's files, tested on both branches;
  * an identity control (+0 [0, 0]); a synthetic regression fails, a synthetic success passes.
