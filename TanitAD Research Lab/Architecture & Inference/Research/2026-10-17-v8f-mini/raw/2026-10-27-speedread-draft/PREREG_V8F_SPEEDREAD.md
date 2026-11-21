# PREREG_V8F_SPEEDREAD (DRAFT): is the per-window speed information O needs present in T's own INPUT?

**Status: DRAFT, not promoted.** Written 2026-10-27, before any SPEEDREAD number. No real train or eval feature has been opened by this
work; the instrument has run on synthetic data only. Claim row to file on promotion: `H-V8F-SPEEDREAD-1`. Commissioned by
`RESULT_V8F_FLATDAG.md` §4 (a): *"A box-2-feasible form of the T-side speed lever, inside LORA's own venue rule (≤ 36 h per seed), drafted
under its own PREREG."* Base commit read: `95063711`.

## 0. Why this experiment, and why first

| step | finding | class |
|---|---|---|
| KINDX | Replacing only T's 5 SPEED cells by the GT cuts O's own-code ADE6 by 0.716 / 0.668 m (T0). Speed is the carrier at both seeds. | INHERITED (`RESULT_V8F_KINDX.md`, not re-read from raw) |
| SPEEDAVG | The two seeds' speed-cell errors correlate 0.71–0.85. The error is systematic, not variance. | INHERITED (register row `H-V8F-SPEEDAVG-1`) |
| SPEEDCAL | An affine map per cell moves `v_a_ms` eval RMSE from 1.92 to 1.33 m/s (s0), and ADE6 not at all, even when it is fitted on eval (CEILING +0.0296 / −0.0075). The error O responds to is per window. | INHERITED (`RESULT_V8F_SPEEDCAL.md` at `95063711`; MM re-read in `raw/2026-10-25-speedcal/mm_verify/`) |
| WC | An 8× constraint weight cuts T's TRAIN loss by 18–25 % and its eval present-slot RMSE by 2.4 %. T's eval error is limited by generalisation. | INHERITED (`RESULT_V8F_WC.md`) |
| H-R9-L1-PROGRESS-1 | A head fitted on a trunk's own training clips learned a relation that did not transfer. Cross-fitted, the information was there. | INHERITED (register row) |
| LORA | Perception (encoder adaptation) projects 494 h per seed on box 2, so its venue rule names the A40. The A40 is not allocated. | INHERITED (`PREREG_V8F_LORA.md` Amendment 3) |

Two readings are left, and they call for different compute:

* **The speed is in T's input, but T does not extract it in a way that generalises.** The next lever is then T-side and fits on box 2.
* **The speed is not in T's input.** Then perception is the lever, and it waits for the A40.

A read-out fitted on frozen inputs with **no** training of the driving model separates the two readings. It needs about 2 h of dev-box CPU
(§8). It is the cheapest experiment that tells them apart, so it runs before any box-2 job on this line.

## 1. Hypothesis

**H-V8F-SPEEDREAD-1.** A read-out of the 5 GT speed cells, computed from T's input only and fitted on TRAINING clips only, replaces T's 5
speed cells in the code O reads. On eval-141 it then cuts X_s's own-code ADE6 by at least F = 0.1207 m, separated, at both seeds.

* X_s = `M2b-mixo_s<s>`, the SPEEDCAL / KINDX base.
  * s0 sha256 `c6a628762790b4fc…`, at `C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt`.
  * s1 sha256 `68a0f3356c37b25c…`, at `…/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt`.
  * MEASURED: both files are present, 86,969,237 B each. The sha256 values are SPEEDCAL's dry run, INHERITED.
* F = |M2b_s0 − M2b_s1| = 0.1207 m, INHERITED from `RESULT_V8F_GRID.md`.

## 2. What is where (MEASURED 2026-10-27 unless marked)

| input | location | evidence |
|---|---|---|
| **eval-141 cache** | `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/` | 141 clips and 141 manifest lines, 2.8 GB. Both grids, `pool4x10` [101, 40, 1024] and `pool8x20` [101, 160, 1024], fp8. Has a `cache_md5_verify.json`. FLATDAG's T1 chain reads it. |
| eval labels / P10 / tracks | `…/sfresh_inputs/inputs/{v9/v9_labels_eval139.npz, v9/lead_eval139.npz, v9/refcv6_speed_max_v8_eval.jsonl, p10/p10_prior.pt, native_tracks.pt}` | present; these are the eval argv inputs (`sfresh_inputs/eval/argv_*.txt`) |
| **train label join** | `C:/Users/Admin/ar3_work/dc/join_train.npz` | 4,572 clips, `cons` [461,518, 26]. sha256 `978f41bf…`, equal to the trainer's `--join-train` as recorded by HGATE and by SPEEDCAL's fit json. |
| train native tracks | `…/sfresh_inputs/inputs/native_tracks.pt` (also `C:/Users/Admin/ar3_work/dc/inputs/`) | 4,713 keys, which cover all 4,572 train clips. sha256 `1db431f1…`. |
| train manifest | `C:/Users/Admin/box2_ship/logs/v8f_lora_ar2/cache/train/manifest.jsonl` | 4,572 lines, each with `bytes` and `md5`. Both grids. |
| **train clip features** | **box 2 only**: `D:\tanitad\cache_full\train\clips\<clip>.pt` | The path is X_s's own `argv.json` (`--cache-dir D:\tanitad\cache_full`). The files themselves were NOT probed, because box 2 is training. |

**Absence of the train clips on the dev box, probed at three places plus the owning script:**

* `find -maxdepth 7` over `E:/Projects/TanitAD-artifacts`, `C:/Users/Admin` and `F:/` found:
  * 4 real train clips in `C:/Users/Admin/ar3_work/dc/train/clips`;
  * 38 in `C:/Users/Admin/refav1_hj_cache/test_runscript/train/clips`;
  * 0 in the manifest-only pulls under `box2_ship/logs/v8f_lora_ar2*/cache/train`.
* The trainer's default `--cache-dir C:/Users/Admin/refav1_hj_cache` holds only test subsets.
* The HBLEND pull of 141 train clips that SPEEDCAL fitted on (`scratchpad/hblend/cache/train/clips`) has been deleted.
* D: is not mounted on the dev box.

**Parity, reproduced on the dev box before any pull (MEASURED).** The trainer's own rule gives exactly X_s0's `summary.json` / `selfcheck.json`:

* the rule: `read_manifest`, then drop sorted-by-sha12 `[540:600]`, then `select_train_clips(.., 1.0)`, then `train_windows(hist 4)`;
* the result: **4,512 clips, 306,566 windows, excluded sha256 `f6a1dee0…`**;
* eval-141 ∩ train is **0 clips**, for the 4,572 and for the 4,512.

⇒ The instrument's `extract` refuses any other list (`TRAIN_PARITY`).

**The pull (the Master Mind schedules it; this draft does not contact box 2).**

* **What:** the 4,512 `<clip>.pt` files only, **94.27 GB** (MEASURED, summed from the manifest's `bytes`). Join, tracks and manifest are already local.
* **Where to:** `F:/TanitAD-scratch/speedread/cache/train/clips/`, with the manifest copied next to it. F: has **339.5 GB free** (MEASURED). E: (3.7 GB free) and C: (3.8 GB free) cannot take it.
* **When:** only while box 2 is **idle between chains**, with no BOX2_BUSY marker. The trainer reads the same disk, so a 94 GB read during a run adds I/O load to a training box (invariant).
* **How:** resumable, one file at a time, over the `tanitad-box2` alias (`launch.ps1` route; SSH drops).
  * Verify each file against the manifest `md5` and re-fetch on mismatch.
  * A clip that cannot be verified **refuses the extract**: no partial fit.
  * ESTIMATED 20–60 min at 30–80 MB/s.
* **Alternative with zero box-2 I/O:** none. Re-encoding from frames would need the GPU and the frames, which is the LORA path.

## 3. The probe (Stage P): zero training of the driving model, dev-box CPU

**Input = exactly what `V8FMini.run_t` reads**, minus S's checkpoints (S's output, computed from the same input class):

* the `hist` = 4 scene frames ending at T's tick, `last_t = F − 1 − a_t`, with their `frame_ok`;
* `p10f` and `nav` at the tick, `limit`, `limit_ok`.

Keys read: `feats, frame_ok, p10f, nav, limit, limit_ok, a_t`, a subset of `INFERENCE_INPUT_KEYS` (checked at run time).

* No label, no future pose, no steering angle, no situation-classifier output.
* The ego channel is the measured yaw-rate kinematics T already gets (`--kin-mode yaw`, PI ruling D-PI-YAWRATE-1).
* T's actual grid is **`pool4x10`** (MEASURED, X_s's `argv.json`). The brief's `pool8x20` is the richer cached grid; GRID already tested it
  as T's input (FAIL-NO-MECHANISM, ADE6 −0.076 ns). It is out of scope here: one input, T's own.

**The vector (fixed now, never tuned).**

* Scene: the per-frame token mean of the 4 frames (4 × 1024), plus T's tick frame pooled 4×10 → 2×5 (10 × 1024). That is 14,336 dims.
  * They are reduced to **PCA 512**, a randomised PCA with seed 20261027.
  * The PCA is fitted on every 16th TRAIN window: unsupervised, no label read.
* Ego: 23 dims (p10f 8, nav 5, limit 5, limit_ok 1, frame_ok 4).
* The whole vector is standardised with TRAIN statistics.
* Targets: the 5 GT speed cells at T's tick frame (`lon_v_target_ms, v_a_ms, v_end_ms, v_lo_ms, v_hi_ms`, physical m/s).
  * Masked like the trainer's `cons_absent=mask`: a v9 row and a finite value.
* Ages follow the eval's own cycle (a_t = i mod 3, a_s = a_t + (i div 3) mod 5), clamped by the dataset.

**Read-outs, all fitted on the 306,566 TRAIN windows only.** Folds = sha256(clip) mod 5. Fold 0 is the held-out TRAIN fold.

* **LIN.** Ridge per cell with an unpenalised intercept.
  * λ ∈ {0.1, 1, 10, 10², 10³, 10⁴, 10⁵}, chosen by grouped CV over TRAIN folds 1–4 (ties go to the larger λ).
  * Its held-out score is computed on fold 0, then it is refit on all 5 folds.
* **MLP.** 535 → 256 → 256 → 5, GELU, dropout 0.1, AdamW (lr 10⁻³, wd 10⁻²), batch 1,024, ≤ 40 epochs.
  * Masked MSE on standardised targets.
  * Early stopping (patience 4) on TRAIN fold 0, fitted on folds 1–4, torch seed 0.
  * A linear negative is not a negative about learnability (TRAPS). The MLP is the learnability arm.
* **PRIMARY** = the one of LIN / MLP with the lower fold-0 RMSE (train-SD units, mean over cells). A tie goes to LIN.
  * It is selected and written to `fit.json` **before any eval batch is opened**.
  * The other one is **ALT**.
* **Floors and controls, fitted the same way.**
  * CONST: the train mean per cell.
  * EGO-only LIN / MLP (beside: does vision add anything over ego?).
  * PERM: the PRIMARY's class on window-permuted targets, seed 20261028, fitted on folds 1–4.

**The substitution read (the decision instrument).**

* Per seed, `v8f_mini_eval.py run` is imported unmodified. It rides SPEEDCAL's carrier (`--tac oracle --tac-groups speed --tac-alpha 1`, `--families`).
* A hook (SPEEDAVG's base class) replaces **only the 5 speed cells of the code O reads** by the read-out's prediction, standardised by the
  checkpoint's `cons_mu / cons_sd`. Every other cell, S, T's decisions and O are X_s's own.
* The read-out is computed **inside the hook from the same batch O sees**. Window identity is therefore exact, and the substitution is a
  deployable function of inference inputs.
* Modes: `own, identity, gt, read, alt, const, perm`.
* Tier: **T1-class open loop**, the tier of SPEEDCAL's and KINDX's ADE6 rows, so that CAPTURE is comparable to them. `gt` is **T0** (ceiling).
* Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired.
  * It answers *"another draw of EPISODES"* only.
  * The read-out is seed-independent: one fit serves both seeds. The two seeds are therefore O's two training runs, which is the replicate
    `H-ESTIM-SEED-1` asks for at this stage.
  * The planner is deterministic, so there is no inference-seed term.

## 4. Controls: each must read its known value before any row is read; any failure ⇒ REFUSED

| id | must read | catches |
|---|---|---|
| C1 | identity − own ADE6 max \|Δ\| = 0.0 and plans equal | the hook path changes nothing by itself |
| C2 | non-speed cells O read = X_s's own, max \|Δ\| = 0.0, every hooked mode | a wider substitution than declared |
| C3 | the hook's tick slice = the model's own `_scene` pad mask for T (first batch) | the read-out reading other frames than T does |
| C4 | episode ids equal across modes | window misalignment |
| C5 | own ADE6 = SPEEDCAL's banked `own` per window, max \|Δ\| = 0.0 | environment drift since SPEEDCAL |
| C6 | gt ADE6 = SPEEDCAL's banked `gt` per window, ≤ 10⁻⁴ m | the oracle path |
| C7 | what O read in `read` = the identity run's read-out prediction, ≤ 10⁻⁵ (std units) | the read-out seeing different batches than scored |
| C8 | CONST eval skill = 0.0 exactly | a broken skill metric (the trap's constant control) |
| C9 | PERM held-out-TRAIN skill ≤ 0.02, and PERM not separated-better than own | leakage or structure that survives a shuffle |
| C10 | train parity (4,512 / 306,566 / `f6a1dee0…`) | re-selection |
| C11 | eval-141 ∩ fit clips = ∅ | fitting on what is scored |

`n` and `d` are printed in `fit.json`: n = 306,566 windows and d = 535, with 14,336 raw scene dims.

## 5. The decision (both outcomes committed now; `decide()` in the instrument is this table, first match wins)

PRIMARY_s = ADE6(read) − ADE6(own) at seed s. ALT_s is the same for ALT. "Separated-better" = the CI's upper end is below 0.

| outcome | condition | committed next |
|---|---|---|
| **INFO IN INPUT** | PRIMARY_s ≤ −F **and** separated-better, at both seeds | §6: Stage Z, then the box-2 arm SPEEDHEAD |
| **PARTIAL** | PRIMARY separated-better at both seeds, not INFO IN INPUT | SPEEDHEAD still runs. It is the only measured lever on this line that fits box 2, and its bar is unchanged. CAPTURE is reported. |
| **NOT IN INPUT** | neither PRIMARY nor ALT is separated-better at either seed | §7: perception stays the lever, blocked on the A40 |
| **MIXED** | anything else | Stage Z (T1-kin, both seeds) decides: ≤ −F separated at both seeds ⇒ SPEEDHEAD; otherwise §7 |

**Beside (decide nothing).**

* CAPTURE = PRIMARY / (gt − own).
* ALT and CONST rows.
* The per-cell eval RMSE and skill of T's own cells against every read-out, including EGO-only (from the identity run's dump).
* The four families from `v8f_paired_families.py` on read vs own (SPEEDCAL's tool), per seed.
  * LONGITUDINAL: speed MAE, accel MAE, distance keeping, LEAD-VIOL.
  * LATERAL.
  * TACTICAL.
  * STRATEGIC: reported for v8F, as in FLATDAG.

⚠️ **Scope of a negative.** NOT IN INPUT means *"not recoverable by a ridge or a 2×256 MLP from the pooled 4-frame summary, on unseen
clips"*. It does not cover the full 40-token attention view T itself has. T fails on that view too (SPEEDCAL), and the two negatives
together are what route to perception.

## 6. INFO IN INPUT (or PARTIAL / MIXED via Stage Z): the T-side arm, box-2 feasible

**Stage Z (zero training, dev-box CPU).**

* The PRIMARY read-out becomes a `forward_infer` **speed sub-head**: buffers that are absent by default, so every earlier checkpoint stays
  bit-identical, like HACC's `hacc_w`.
* It is re-read at **T1-kin**, using FLATDAG's rollout: self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene.
* Two rollouts plus X_s's own if not banked. Each takes about 260 s: FLATDAG measured 258–261 s.
* Bar: Z − X_s ADE6 ≤ −F, separated, at both seeds ⇒ **ZERO-TRAINING WIN**, banked as a deployable post-map. Families are reported per seed.

**The arm SPEEDHEAD.** Chosen by expected effect: it is the only candidate with a measured expected effect, namely Stage P's PRIMARY.

* Regularisation or feature dropout on T have no measured size.
* GRID (more input capacity) already failed.
* WC (more loss weight) moved eval accuracy by 2.4 %.

**Lever (one): the source of the 5 speed cells O reads.**

* **Training.** Each window's T-tick speed cells, as O reads them, are the **out-of-fold** read-out prediction: 5 folds by clip; each fold's
  read-out is fitted on the other four, with the PRIMARY's class and hyper-parameters.
  * The table is precomputed on the dev-box CPU (`--speed-oof <npz> --speed-oof-sha256 <h>`).
  * Out-of-fold, because an in-sample read-out is more accurate on training windows than on eval. O would learn to over-trust it, which is
    the H-R9-L1-PROGRESS-1 failure.
* **Inference.** The all-fold read-out runs inside `forward_infer` (`--speed-head <readout.pt> --speed-head-sha256 <h>`).
* **Unchanged.** T's own speed head still trains on its loss. Teacher forcing (`--code-mix kind`, p_gt 1 → 0.2) still draws GT per unit.
* **The pair.** SH-0 = X_s fine-tuned 3,000 steps with DAG's recipe at `--dagger-p 0`, which is FLAT-DAG-0's recipe with `--o-interface full`.
  SH = SH-0's argv + the four speed-head flags. Pair rule: the argv is identical but those four flags and `--out`.

**Launch blockers (implementation, under the launch gate; nothing here is built yet).**

1. **G-DVB.** The built speed-head buffers' sha equals the argv, and the OOF table's sha equals the argv. The OOF table covers every
   training window's tick: COVERAGE 1.0, refused otherwise.
2. **G-LIVE, on a real batch.** The code O reads equals the OOF value in the 5 cells (≤ 10⁻⁶) and T's own elsewhere (exactly 0.0).
   * O's `e_code` gets a non-zero gradient through those cells.
   * The head has no trainable parameter: a control that must read exactly 0.0.
   * **Deliberate regression:** deleting the substitution line must make the gate FAIL.
3. **G-EVAL.** On eval-141, the model's `forward_infer` with the head reproduces Stage P's hook substitution per window (ADE6 ≤ 10⁻⁵ m).
   This ties the arm to the measured effect.
4. **Admissibility.** The head reads `READ_KEYS` only. `assert_no_situation_classifier_input` is unchanged. There is no goal path: the
   head writes speed cells, not a goal.

**Runs (box 2, one job at a time, safe mode, the temperature guard).**

* SH_s0, SH-0_s0, SH_s1, SH-0_s1, then the replicate **SH_s0′**: the same argv re-run, for the training-run floor (`H-ESTIM-SEED-1`).
* About 38 min each: FLATDAG measured 2,251–2,263 s per 3,000 steps. ⇒ **≈ 3.2 h in all, ESTIMATED**, far inside ≤ 36 h per seed.
* Needs the PI's go for box 2.

**Decisive rows, T1-kin, per seed.** Paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows.

* **PRIMARY:** ADE6 SH − SH-0 ≤ −F and separated, at both seeds, **and** |PRIMARY_s0| > |ADE6 SH_s0 − SH_s0′| (the replicate floor).
* **GUARD (LONGITUDINAL):** speed MAE, accel MAE, DK-VIOL and DK-TG are not separated-worse at either seed.
* **Families reported per seed, never pooled:**
  * LONGITUDINAL: speed MAE, along-track, accel, DK-VIOL / HW / TG.
  * LATERAL: heading, yaw rate, curvature, cross-track, L1 lateral accel, L2 curvature.
  * TACTICAL: TAC-LAT, TAC-LON, lat / lon head F1.
  * STRATEGIC: S-POS@30 / 50 / 80 vs CTRV, per arm. The paired contrast is "not computed" until its instrument exists, and never counts as passing.

**Outcomes.**

| outcome | condition | committed next |
|---|---|---|
| **PASS** | PRIMARY + GUARD + floor | SH becomes the v8F-mini base. The sub-head joins `forward_infer`. DAgger composition follows under its own PREREG. |
| **FAIL-GUARD** | PRIMARY met, GUARD broken | Compose with OACC's accel term (`--w-acc`) under its own PREREG. |
| **FAIL** | PRIMARY not met at either seed | If Stage Z won, the zero-training sub-head stands, and the next arm freezes O and fine-tunes T's speed path toward the OOF targets. If Stage Z did not win, §7. |
| **MIXED** | anything else | Replicate per seed: seeds before windows. |

## 7. NOT IN INPUT: perception stays the lever, and here is exactly what blocks it

* **The arm:** `PREREG_V8F_LORA.md`, DINOv3-L/16 + LoRA on the fly. Its AR-2 smoke PASSED in run 3. It projects 494 h per seed on box 2
  (INHERITED, Amendment 3), so its own venue rule (≤ 36 h per seed) names the A40.
  * On the A40 it is ESTIMATED at about 100+ GPU-h per seed (LORA Amendment 3, scaled by an assumed 4–5× throughput).
* **What unblocks it, all PI decisions:**
  1. the A40 provisioned (`D-V8F-SPEED-PERCEPTION`, `PI_DECISION_QUEUE.md`);
  2. the frames on HF private or a disk provision (LORA §7, AR-4);
  3. the go for ~200 GPU-h (2 seeds).
* **Nothing on box 2 on this line.** The box-2 streams already queued (DAG-OACC, DAG-ORC) are unaffected.

## 8. Run (dev-box CPU; every stage longer than 4 min runs DETACHED and is handed off by file)

```bash
export CUDA_VISIBLE_DEVICES=-1 SPEEDREAD_THREADS=8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; T=<tip tree>/stack/scripts/v8f_mini_eval_speedread.py; W=F:/TanitAD-scratch/speedread; SF=E:/Projects/TanitAD-artifacts/sfresh_inputs
$PY -I $T extract --cache $W/cache --join C:/Users/Admin/ar3_work/dc/join_train.npz --tracks $SF/inputs/native_tracks.pt --out $W/train        # ~20-30 min
$PY -I $T fit --train $W/train --eval-manifest $SF/cache/eval/manifest.jsonl --out $W/fit                                                      # ~10-20 min
for S in 0 1; do $PY -I $T evals --seed $S --x <X_s ckpt> --expect-sha <X_s sha256> --readout $W/fit/readout.pt --fit-json $W/fit/fit.json \
   --eval-dir $W/eval_s$S --eval-cache $SF/cache --tracks $SF/inputs/native_tracks.pt --v9-eval $SF/inputs/v9/v9_labels_eval139.npz \
   --lead-eval $SF/inputs/v9/lead_eval139.npz --speed-eval $SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl --p10 $SF/inputs/p10/p10_prior.pt; done   # 7 modes x ~4-6 min per seed
$PY -I $T read --fit-json $W/fit/fit.json --eval-dir-s0 $W/eval_s0 --eval-dir-s1 $W/eval_s1 \
   --speedcal-dir-s0 "<PK>/raw/2026-10-25-speedcal/run/eval" --speedcal-dir-s1 "<PK>/raw/2026-10-25-speedcal/run/eval" --out $W/read                # < 1 min
```

* **CPU time, ESTIMATED, after the pull:** about 1.5–2.5 h. That is extract 20–30 min (94 GB of fp8 read once), fit 10–20 min, evals
  60–85 min and read < 1 min. Peak RAM is about 3 GB: memmaps on F:, PCA on a 1/16 subsample.
* **Disk:** about 8.8 GB on F: (the 306,566 × 14,336 fp16 scene memmap) plus 94.27 GB for the pulled cache.
* The four families come from `stack/scripts/v8f_paired_families.py` on each `read` / `own` pair, as in SPEEDCAL's run.
* The SPEEDCAL banked dumps for C5 / C6 must be the `eval_M2b-mixo_s<s>-speedcal-{own,gt}.npz` files at `95063711`.

## 9. Instrument and self-test (synthetic only; this is not a result)

* `stack/scripts/v8f_mini_eval_speedread.py`: extract / fit / evals / read; `decide()` is §5 verbatim.
* `stack/tests/test_v8f_speedread.py`: 25 tests in about 8 s on CPU.
  * The tick slice is checked against an independent loop.
  * Causality: frames outside T's window are not read.
  * Padded frames read 0, and no key outside `READ_KEYS` is read.
  * A planted signal is recovered on unseen clips. EGO-only reads none.
  * CONST reads skill 0.0 exactly, and PERM reads none.
  * A **purely non-linear target is found by the MLP and missed by the ridge**, and PRIMARY selects the MLP.
  * The read-out survives a save / load round trip.
  * The substitution touches only the speed cells, and identity is bit-exact.
  * The hook hands O the read-out's value, and the mirror check catches a model whose T slice differs.
  * Every outcome row of §5 is reached, including both −F boundaries.
  * Each control goes RED on its own break.
* `mutate_speedread.py`: 19 mutants, each deleting or breaking one value-emitting line. The result is in `mutation_results.json`.

## 10. What this does not license

* ⛔ No bar moves after data. An outcome is reported as the table says, then Rule Zero applies.
* ⛔ The read-out is never fitted, tuned or selected on eval-141. Nor is anything else.
* ⛔ A separated PRIMARY is a result about two O training runs (the seeds) and one read-out fit. It is not a lever claim for the SH arm,
  which carries its own replicate.
* ⚠️ **Disclosed design limits.**
  * The PCA-512 summary is lossy.
  * The 2×5 pooling assumes the `pool4x10` tokens are row-major (4 rows × 10 columns). If they are not, the pooling is a different fixed
    linear summary, not a leak.
  * S's checkpoints are not read-out inputs.
  * The fit uses the eval's age cycle, not the trainer's random ages.
