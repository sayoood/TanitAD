# PREREG_V8F_WMACT: does v8F-mini's world model predict, level by level, and does its prediction react to the action? (Master Mind, 2026-11-07, written before any number of it exists; asked for by the PI in chat)

## Why this card exists
* The PI asked: *"Did you check the prediction quality of each abstraction layer? Is the prediction reacting on potential actions?"*
* v8F-mini M2b carries an H-JEPA world model (`WMAux`, `stack/tanitad/models/v8f_mini.py:648`). It has three levels z1 / z2 / z3 on the 0.2 / 0.4 / 0.8 s cycles, level 1 is conditioned on (acceleration, yaw rate), and it is trained with an inverse-dynamics term (IDM, gamma 100). It is a training auxiliary: S, T and O do not read its predictions.
* What is measured about it (register): `H-MINI-WM-1` FAIL (no trajectory gain from the auxiliary) and `H-V8F-BETA-1` FAIL-GOAL (as a scorer of a 117-plan fan it is far worse than M2's plan; with the true goal latent its ranking recovers 0.838 / 0.841 of the signal). Neither measures, per level, whether the WM beats "nothing changes", or whether its prediction depends on the action it is given. The tool for the second exists (`HJEPAStack.action_sensitivity`, the H-JEPA stage-0 collapse test) and has never been run on v8F.

## Data and arms
* eval-141: 2,399 windows, 141 episodes, the dev-box cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache`. CPU only.
* Arms: M2b s0 / s1 (the WM was trained with the model; the weights are the HACC-baked files' WM tensors, which `MODEL_REGISTRY.md` §17.1 records as bit-for-bit the source's). C0 s0 / s1 beside (`v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt`): if C0's WM tensors are bit-equal to M2b's, that is recorded and C0 is not read again.
* Latents and actions follow training exactly: z1 = `WMAux.z1` of the adapter tokens, actions = the cache's measured (a, yaw rate) channels through `WMAux.scale_actions`, transition j to j+1 paired with the action at frame j. True future latents at levels 2 and 3 come from `HJEPAStack.encode` / `lift` of the real future frames.

## The statistics (per level l, horizon k, seed; window sums within an episode, episodes resampled)
* Copy baseline error C = sum of ||z_start - z_k||^2, where z_k is the TRUE future latent.
* **SKILL(l, k)** = 1 - sum ||ẑ_k(a_true) - z_k||^2 / C. Above 0 means the WM beats "nothing changes".
* **ACTION-USE(l, k)** = [sum ||ẑ_k(a_donor) - z_k||^2 - sum ||ẑ_k(a_true) - z_k||^2] / C. a_donor is another episode's action sequence (`donor_indices`, seed 0). Above 0 means the true action predicts the real future better than another episode's.
* **ACTION-USE-MEAN(l, k):** the same, with every action replaced by the TRAIN-mean action (the strict real-vs-mean test of `wm_reliance.py`).
* At level 1 the action channels are also read one at a time: acceleration only, then yaw rate only.
* Beside, deciding nothing: `sens_rom/all/k` and `sens/all/k` from `action_sensitivity`, per level.
* Horizons: level 1 k = 1, 5, 10 (0.2, 1.0, 2.0 s); levels 2 and 3 k = 1, 3 where the window holds the future frames. Each row reports the n of windows it used.
* Estimator: episode-cluster bootstrap over the 141 episodes, B 2,000, seed 0. "Separated" means the 95 % interval excludes 0. This answers another draw of episodes; the two seeds answer the training run; inference is deterministic.

## Controls (binding: if one fails its known value, the read is INVALID and nothing below is reported as a result)
1. **Identity:** donor = the window itself. ACTION-USE and sens read exactly 0.0.
2. **Deliberate regression:** a copy of the model with level 1's action embedding zeroed (weights and bias). At level 1 sens and ACTION-USE read exactly 0.0; SKILL is reported.
3. **Scorer check:** the rollout started from another window's z_start (same donor rule). SKILL must be below 0 and separated at both seeds.
4. **Reproduction:** the model's own WM loss terms on 64 eval windows, recomputed through `HJEPAStack.loss`, equal a direct call to `WMAux.forward` to float tolerance 1e-6.

## Outcomes per level, committed now
* **PREDICTS(l):** SKILL(l, 1) > 0, separated at both seeds.
* **ACTION-REACTIVE(l):** ACTION-USE(l, 1) > 0 AND ACTION-USE-MEAN(l, 1) > 0, each separated at both seeds.
* **ACTION-INERT(l):** ACTION-USE(l, 1) not separated at either seed, or sens_rom(l, 1) < 0.01.
* **MIXED(l):** anything else, reported as written.

## Committed next, by outcome at level 1
* **PREDICTS and ACTION-REACTIVE:** the WM is a working action-conditioned predictor that the planner does not use. The next lever is reliance (Sayed, 2026-08-06, `wm_reliance.py`: *"the decoder must rely on the representation and prediction of the wm"*). That is an arm in which O reads the WM's level-1 rollout under its own draft plan, under its own card.
* **ACTION-INERT:** the IDM term did not make the predictor use its action. The next read, zero-training and run in the same session, is the attenuation probe: the action embedding's magnitude against the residual stream at each predictor block, as the refav1 diagnosis did.
* **PREDICTS fails:** the WM does not beat "nothing changes" at its own cycle. The next read is the same statistics on TRAIN windows, to separate fit from generalisation.

## Scope
* Tier T0 (teacher-forced: logged actions, true future frames). This is a WM diagnostic, never a driving claim, and never pooled with T1 or T2.
* The planner layers S, T and O are not read here.

## Log (dated, append-only after the freeze)
* 2026-11-07 08:52:03 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 5478 bytes above this log, sha256 `d21b14ccc3ea41692f2b615720b2814e81119019e78fc1ad85ba5b58474d9bb8`, before any statistic of it exists.
* 2026-11-07 09:05 (dev-box clock, from `date`), Master Mind. **Implementation choices, logged before any statistic exists** (`theory/wmact/HANDOFF_WMACT.md`).
  * Code is a read-only `git archive` export of the mirror tip: the E: `stack/` checkout is from 2026-09-19 and lacks `v8f_beta.py` and `WMAux.z1`. `tanitad.__file__` is asserted under the export.
  * The rollout starts at frame 0 of the 16-frame training window, and only windows with all 16 frames real are read (about 82 % of 2,399; the exact n is reported per row). A NOW-start variant is not run.
  * The mean action of ACTION-USE-MEAN is the mean over eval-141's real cached frames, not over TRAIN. The TRAIN cache is on box 2, which is busy. It is one constant for every window and never the window's own action; reported under that name.
  * Rollouts use `HJEPAStack.rollout` with context 1, the planning convention.
  * MEASURED (`wmact/weights_check.json`): M2b's HACC-baked `wm.*` equal the source checkpoints' bit for bit, 145 of 145 at both seeds. C0's `wm.*` differ from M2b's (12 of 145 equal; adapter 0 of 5), so C0 is read beside M2b, as the card requires.
* 2026-11-07 09:12 (dev-box clock, from `date`), Master Mind. **Result, M2b both seeds, C0 beside** (`theory/wmact/RESULT_WMACT.md`, `RESULT_WMACT.json`, `RESULT_WMACT_C0.json`). All four controls read their known values. 1,976 windows, 141 episodes.
  * **L1: ACTION-REACTIVE; PREDICTS not met.** SKILL k1 is +0.0010 [−0.0114, +0.0118] / −0.0010 [−0.0141, +0.0103]; at k10 it is −0.1500 / −0.1724, separated. ACTION-USE k1 is +0.0054 / +0.0049 and ACTION-USE-MEAN +0.0029 / +0.0030, all separated.
  * **L2 and L3: PREDICTS and ACTION-REACTIVE.** SKILL k1 is +0.0295 / +0.0264 at L2 and +0.1316 / +0.1351 at L3. ACTION-USE k1 is +0.0424 / +0.0462 at L2 and +0.1850 / +0.2000 at L3.
  * **Committed next, by the card at level 1 (PREDICTS fails): the same statistics on TRAIN windows.** The sample is fixed now, before any TRAIN data is read.
    * Clips: the 141 TRAIN clips with the smallest sha256(clip id) among the 4,512 in box 2's `cache_full/train/manifest.jsonl` used by M2b (the `--exclude 540:600` holdout removed).
    * Windows: 17 window starts per clip, evenly spaced over the trainer's valid starts (np.linspace, rounded, unique). hist 4, hmax 16, grid pool4x10, kin_mode yaw.
    * Code: the same `wmact_lib` statistics and the same four controls, with the store pointed at the pulled TRAIN clips.
    * The mean action stays the eval-141 constant, so the two reads share it.
  * The upper levels' outcome commits the reliance arm, under its own card.
* 2026-11-07 09:16 (dev-box clock, from `date`), Master Mind. **TRAIN read, implementation, before any TRAIN data:** the dev box has 1.6 / 3.9 / 2.1 GB free on C: / E: / F:, and the 141 clips are about 2.9 GB. So the clips are pulled one at a time from box 2 into one reused scratch file, and only each clip's latents and actions are kept. The source files on box 2 are untouched, so no data is lost. Window starts, hist, grid and statistics are as in the 10:xx entry.
* 2026-11-07 09:17 (dev-box clock, from `date`), Master Mind. The freeze line above carries one stray CR byte (offset 5664, right after the sha): Windows Python printed the sha with CRLF and the shell kept the CR. It lies outside the frozen prefix, whose hash holds. The `.sha256` file had the same byte and is rewritten without it.
