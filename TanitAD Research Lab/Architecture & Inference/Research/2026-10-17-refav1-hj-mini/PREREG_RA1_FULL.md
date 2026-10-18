# PREREG_RA1_FULL — does 8.4× the train data let the pooled readout and the IDM world model beat P10?

**Written 2026-10-18 ~11:30, AFTER ES-1…ES-5 and the ridge probe, BEFORE any full-corpus clip was built or any run launched.**
This is the data-scale lever that `PREREG_RA1_ES.md` reading 1 names, and that `Project Steering/PREREG_HJ_STAGE0.md` already
plans (*"600 TRAIN clips first, then all 4,572"*). It amends neither file and moves none of their bars.
⚠️ It does **not** reopen H-HJ-K-1 under a different control: the A1x gate is read first, as written, and K is read only behind it.

PI go, 2026-10-18: *"Yes, run the data-scale build on box 2 … Use safe mode and stop if it crashes."*

Addendum D of PREREG_RA1_ES (A1a, the action-permutation control) is a separate experiment owned by another session. It runs on
the 600-clip cache root. Nothing here reads or changes it, and the full-corpus stages read their own root (§Data).

## What the 540-clip series said (MEASURED, T1-class open loop, eval-141 = 2,399 windows / 141 episodes)

Sources: `raw/ra1_es/es1.json`, `es3_attribution.json`, `es4_budget.json`, `es5_wd.json`, `RESULT_RA1_ES.md`. Estimator: paired
episode-cluster bootstrap, B = 2000 (EPISODE variance only).

| regime | A1 − P10-yaw-540 (s0 / s1) | A1x − A1 (the gate) | what it says |
|---|---|---|---|
| 250 steps | −0.0517 [−0.0792, −0.0234] / −0.0480 [−0.0764, −0.0201] | −0.0011 [−0.0046, +0.0026] | a gain, and the control keeps it |
| 1,000 steps | +0.0089 / +0.0172 | +0.0461 [+0.0033, +0.0969] SEP | the control loses, and the gain is gone |
| 1,000 steps + wd 3.0 | −0.0671 [−0.1188, −0.0149] / −0.0524 [−0.1019, +0.0002] | +0.0109 [−0.0086, +0.0335] | a gain, and the control keeps it |

* **Memorisation:** R0's seed-mean holdout delta vs P10 grows with the budget: 250: −0.0409; 500: +0.0628; 1,000: +0.3007;
  2,000: +0.3805; 6,000: +0.4612. A1's holdout ADE@6s does the same: 2.9405 → 2.9626 → 3.0195 (250 → 500 → 1,000).
* **Attribution:** A1 − A0 at 250 steps is −0.0268 [−0.0508, −0.0044] / −0.0233 [−0.0480, −0.0004]. So the IDM term carries the gain,
  but the gain survives a time shuffle of the frames.

⇒ At 540 clips, **every regime where A1 beats P10 is one where A1x matches it, and the one regime where A1x loses is one where the
readout has already memorised**. Both failures share one candidate cause: 540 clips is too few for the readout to train long enough
for any temporal content to matter before it overfits. More data is the lever that addresses both at once.

## The hypothesis (H-RA1-FULL-1)

With 4,512 train clips (8.4×), memorisation sets in later. If so, there is a budget where A1 has trained long enough for A1x to lose
to it, AND short enough that the readout still generalises. At that budget the gate passes and K can be read.

## Data

* **Corpus:** the 4,572 clips of the v7.2 TRAIN label release (B1 listing minus eval-141). **Train = 4,512**, i.e. all of them minus
  the SAME 60-clip Amendment-1 holdout (the last 60 of the first 600 in sha12 order, clip-list sha256 `f6a1dee0…`). The queue passes
  no `--n-clips`, so the trainer takes every clip in its cache.
* **⛔ A separate cache root, `D:\tanitad\cache_full`.** It is seeded with a copy of the 600-clip root's train and eval splits and
  `native_tracks.pt`. Because the trainer takes every clip in its cache, encoding the new clips into `D:\tanitad\cache` would make
  every later 540-clip stage, Addendum D's A1a included, silently train on 4,512. Only the `full` / `full2` queue stages read the new
  root. That is pinned by `test_queue_full_stages_read_their_own_cache_root_and_prior_and_nothing_else_does`.
* **The 3,972 new clips are built the way the 600 were:**
  * same builder: `stack/scripts/pod_build_b1_epcache.py`, unchanged; box 2's copy is byte-identical to the tip (sha256 `055dba9f…`);
  * same geometry: native 256×640, cylindrical, 120° HFOV, png;
  * same encoder: `code/pull_encode_pooled.py`, unchanged except `--only-file`, which only passes the clip list (cmd's 8,191-char
    line limit);
  * same flags: those of the 600-clip encode on box 2 (`encode_A2`).
* **Inputs:** staged on the dev box from HF (`stage_train_all_native256.py`). `stage_report.json` records:
  * mp4 sha256 matches HF's ledger for 4,572 of 4,572 clips;
  * timestamps and egomotion match HF for 4,572 of 4,572;
  * the intrinsics match HF to within 1.14e-13.

  After the transfer to box 2, every file's size is compared, and the egomotion zip and the intrinsics csv are compared by sha256.
  Every mp4's sha256 is re-checked against HF's ledger on box 2 before its chunk builds.
* **C0 — identity control (known value; it runs first and is a STOP rule).** Box 2 rebuilds 2 clips that the dev box already built
  (`8f387977…`, `bf5ec76e…`).
  * MEASURED 2026-10-18: the dev box rebuilt both from the full stage **byte-identical** to its own S4b build (sha256 `4d229504…`,
    `28ca420f…`). So a difference on box 2 is box 2's, not the stage's.
  * Box 2 runs the same library versions: torch 2.11.0, torchvision 0.26.0, av 18.0.0, numpy 2.5.1.
  * **PASS** = the sha256 matches. **PASS-CONTENT** = the bytes differ but the decoded frames, actions and poses are bit-equal. That
    is disclosed, because it is what reaches the encoder.
  * **Anything else ⇒ STOP:** nothing built on box 2 is used. Why it binds: a within-train preprocessing split would be invisible
    downstream. S4b measured that a different swscale build moves pooled features by ~2.5 % rel-L2.
* **Verification:**
  * per chunk of 400: `verify_cache.py` C1 (md5 of every file), C2 (content on 40 sampled), C3 (disjointness) and C4 (partial
    coverage);
  * at the end: C1–C4 over all 4,572, with C2 on every file.
  * ⚠️ C5 (bit-exact re-encode) is not run per chunk. Its candidates are always the first clips in sha12 order, i.e. the 600 already
    redo-checked on box 2. This is disclosed.
* **Native tracks:** `native_tracks.pt` covers 4,572 of 4,572 train clips (sha256 `1db431f1…`, identical on box 2).

## The prior: P10-yaw-4512

The recipe is unchanged (`stack/scripts/v8f_mini_fit_p10.py --kin-mode yaw`: MLP 8→256→60, inner CV, stride 8, seed 0). The fit set
is all 4,572 minus the holdout (`--exclude 540:600`, which is exactly the holdout): **4,512 clips, 40,599 rows**. The holdout stays
out of sample. Source: `raw/p10_yaw4512/p10_fit.json`, MEASURED.

| prior | eval-141 ADE@6s | holdout ADE@6s (60 clips, 540 windows) |
|---|---|---|
| P10-yaw-540 (the ES comparator) | 2.5627 | 3.0529 |
| **P10-yaw-4512 (this comparator)** | **2.4870** | **3.0682** |

**K is read against P10-yaw-4512.** It is the prior the arms are a residual on, and the harder bar: −0.0757 m on eval-141. Beating
the weaker P10-yaw-540 with 8.4× its data would not be a result. P10-yaw-540 deltas are reported beside it, descriptively.

## Arms

All arms run with `--kin-mode yaw --tracks <cache_full>\native_tracks.pt --holdout-n 60`, grid 4x10. Every training constant is the
queue's `:run` block, unchanged; that includes `--wd 1e-3`.

* **Phase 1:** `ra1_box2_queue.cmd full "500 1000 3000 6000" 0.08 50`. At each budget, R0 at seeds 0 and 1, and A1 b0.08g50 at
  seeds 0 and 1: 16 runs, in `runs\ra1_yaw_full\st<S>\`.
* **Phase 2:** `ra1_box2_queue.cmd full2 <S*_A1> <S*_R0> 0.08 50`. A1x b0.08g50 at seed 0, at S*_A1, followed by the queue's verdict.
* **Why this grid.** At 540 clips both R0 and A1 were best at 250 steps on the holdout. If the onset scales with the data (×8.4), it
  moves to about 2,100 steps, which the grid brackets. 250 is the pre-registered downward extension, not part of the first grid.
* **Why b0.08g50:** it is stage 1's holdout pick, kept. Re-selecting the cell would move a second lever.
* **Why wd 1e-3, not 3.0:** one lever at a time. wd 3.0 is the named next lever if K fails (reading 4).
* **RAM plan:** everything resident. MEASURED: 4.14 MB of fp8 4x10 tokens per clip (separate storages), so 4,512 clips ≈ 18.7 GB
  in box 2's 32 GB. If the first run cannot hold it (a MemoryError, or more than 28 GB committed), then EVERY arm switches to
  `--chunk-clips 2256 --chunk-steps 500`. That is decided on the first run, before any eval number exists, and disclosed.

## Selection — on the holdout only (`code/ra1_full_verdict.py select`)

* **S\*_arm** (separately for R0 and A1) is the budget with the lowest seed-mean holdout ADE@6s. Ties within 0.01 m go to the
  shorter budget.
* **Edge rule:**
  * S* on the grid's lower edge ⇒ run the extension S/2 (that is, 250). S* on the upper edge ⇒ run 2S (that is, 12,000).
  * The extension is ONE run set (R0 ×2, A1 ×2) per direction. Selection is then re-made over the extended grid with `--extended`.
  * An S* still on the edge after that is read there and disclosed, never extended again.
* **The K arm** is the lower seed-mean holdout ADE of R0 at S*_R0 and A1 at S*_A1. A tie within 0.01 m goes to **R0**: the world
  model has to earn its place.
* eval-141 numbers are reported for every budget and **select nothing**. The verdict refuses a run that does not meet all of these:
  * its cache holds 4,572 train clips;
  * it was scored on the ES holdout;
  * it shares ONE train list and ONE prior with the other runs.

## Readings — committed before any data

Read in this order. A later reading is not made if an earlier one stops.

0. **C0 FAILS ⇒ STOP.** No box-2 build is used. **Next lever:** build the 3,972 clips on the dev box CPU (437 episodes/h at 8
   workers, MEASURED for the 600; about 9 h). ⚠️ That is blocked by disk: the dev box's C: has ~29 GB free and the v2ep needs
   ~137 GB. So it would have to run in chunks that ship to box 2 and are deleted, which is the transfer route this design avoids.
1. **Memorisation (descriptive, not a gate):** R0's seed-mean holdout ADE minus P10-yaw-4512, by budget.
   * Below 0 at S*_R0 ⇒ more data let the readout beat its prior. At 540 clips that held only at 250 steps.
   * Phase 2 runs whatever this reads, because the gate and K are A1's. (ES-1's stop rule was about R0 alone, at a scale where A1's
     regime was already known.)
2. **The gate, at S*_A1:** A1x − A1 on eval-141 ADE@6s > 0, separated (the queue verdict's `A1x_control` = OK).
   * **FAIL ⇒ K is NOT read.** 8.4× the data did not make the time shuffle cost A1 anything. **Next lever:** Addendum D's A1a
     control at the full corpus, under Addendum D's own readings. It asks whether the IDM gain needs the true action pairing, which
     A1x cannot test.
3. **K, only if 2 passes.** This is PREREG_HJ_STAGE0's bar with two changes stated here: P10 = P10-yaw-4512, and "best" = the
   holdout-chosen K arm. The bar: `K arm − P10-yaw-4512` ADE@6s < 0, separated at both seeds, and |Δ| > 2× the training-seed floor
   |A1_s0 − A1_s1|.
   * **PASS ⇒ data scale is the lever.** Next: A2, and SENS with A0, at S* on the full corpus, under PREREG_HJ_STAGE0's other bars.
   * **FAIL ⇒** the gate holds, so the instrument now reads, and the frozen pooled readout still does not beat a P10 refit on the
     same data. **Next lever, run automatically:** A1 ×2 + A1x at S*_A1 with `--wd 3.0`. That is ES-5's one regulariser that
     separated A1 − P10 at 540 clips, and it is read under readings 2 and 3 unchanged. If that also fails, the unfrozen encoder (LoRA
     r 32, PREREG_HJ_STAGE0's T-B) is next. It needs GPU memory that box 2 does not have (8 GB), so it is a named PI compute decision.

**Reported, NOT read:**
* The four metric families for the K arm and for A1 vs P10-yaw-4512, from each run's `eval.json`:
  * longitudinal: speed MAE, along-track MAE, acceleration MAE, target-speed bands, ego progress;
  * lateral: heading, yaw-rate, curvature and cross-track MAE;
  * tactical: the strata;
  * STRATEGIC = **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).
  * Distance-keeping is **NOT COMPUTABLE** here: the pooled cache carries no lead-agent track (n = 0).
* eval-141 deltas at every budget (non-deciding).
* P10-yaw-540 deltas.
* The 540-clip result at each matched budget (the data-scale effect, descriptive).
* The training-seed floors.

## Variance named

* **Episodes:** the paired episode-cluster bootstrap.
* **Training seed:** two seeds for R0 and A1, with the floor |s0 − s1| reported per budget. A1x has ONE seed, as in ES-2, so the
  gate's separated CI answers *"would another draw of EPISODES say this?"* and not *"would another training run?"* (`H-ESTIM-SEED-1`).
* **Inference seed:** none. The readout is deterministic.

## Cost, safety, provenance

* **Where it runs:** box 2, SAFE MODE (`%RUNSAFE%`, env.cmd: AVX2, below-normal priority, affinity 0xFF), one job at a time, watched
  by the boot-time crash guard. A reboot stops the chain.
* **The build:** `code/ra1_full_box2.py`. It runs in chunks of 400 and resumes from a state file. The CPU build (4 workers, 1 thread
  each) and the GPU encode never overlap. A chunk's v2ep and mp4s are deleted only after its verify passes.
* **Its estimated time (ESTIMATED):** ~60 s per clip per worker, so ~16.5 h, plus the encode at 5.0 s per clip (MEASURED under SAFE
  MODE), so ~5.5 h.
* **The RA-1 runs (ESTIMATED):** about 15 min of clip loading per run (~95 GB read from the USB HDD) plus 2–13 min of training, so
  ~6 h for 17 runs.
* **Exit codes:** 10 control mismatch · 11 disk guard (< 35 GB free) · 12 mp4 sha mismatch · 13 build · 14 encode · 15 verify.
* **Instruments and their guards:**
  * the driver, `code/ra1_full_box2.py`;
  * the verdict, `code/ra1_full_verdict.py`;
  * the queue's `full`, `full2` and `:fullenv`;
  * 4 tests in `stack/tests/test_ra1_hjepa.py`.
  * 11 source mutations each turn a test RED (`raw/ra1_full/mutate_full.json`).

## Implementation notes (2026-10-18 ~12:00, still before any full-corpus data)

* **Phase 3 is the K-FAIL lever of reading 3, as queue stage `full3 SA SR 0.08 50 3.0`.** It runs A1 ×2 + A1x at S\*_A1 with
  `--wd 3.0`, in `st<SA>_wd3.0`, and its verdict pairs them with the unregularised R0 at S\*_R0, as ES-5 did. "Readings 2 and 3
  unchanged" means the K arm is re-picked **on the holdout** between the regularised A1 and that R0 (a tie goes to R0):
  `ra1_full_verdict.py read --phase3-root`. A phase-3 K FAIL names the LoRA compute decision, not another sweep.
* **Every run is attributable to a landed commit.** Before phase 1, `code/box2_ops/chain_ra1_full_runs.sh` refreshes box 2's
  `stack/`, `taniteval/` and the two packages from the REMOTE tip. It then sha-checks six key files against the tip's blobs (the
  trainer, the evaluator, the model, the loader, the queue and the prior), and records the tip in `TIP_FULL.txt`. The whole
  sequence runs as one script with no model turn waiting on it: phase 1, selection, the edge extension, phase 2, and phase 3 if
  K fails.
