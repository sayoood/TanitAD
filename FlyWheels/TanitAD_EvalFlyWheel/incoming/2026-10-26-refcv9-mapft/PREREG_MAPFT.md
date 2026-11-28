# PREREG_MAPFT -- refcv8's map-only fine-tune for PRECISION, read by the planner's own map-path instrument (Master Mind, 2026-10-26 ~05:57 CET)

Sealed BEFORE any timing smoke, step, checkpoint, raster, pick or bar of the fine-tune exists. Authority: PI D-R9-MAPHEAD-PRECISION (a),
2026-10-26 ~03:30, cap **6 Thor-hours**, Thor after NT A1, 50-step timing smoke first, stop if the measured cost exceeds the cap; planner
bit-identical, asserted. Infrastructure landed 70211d6 (`--r8-train-only-map`, `assert_map_only_finetune.py`, gate profile `refcv8-mapft`).

## 0. What is known (disclosed, MEASURED)
* MAPPATH (de0bdc1, PREREG_MAPPATH 34a030ee + A1/A2): on the progress-matched fan the GT map cuts sidewalk contact 0.1792 -> 0.0219 (GT-T1);
  refcv8's OWN map buys nothing (MAPSOFT-G-XF fitted (r0, lam_s, lam_e) = (0, 0, 0) in 5/5 folds; M1/M2 +0.0000), although its candidate-level
  AUC at x >= 4 m is 0.905 (deranged 0.712). The own rasters already use the declared `prior_corrected` rule (softmax(z - log w)), so this is
  not a calibration artefact (checked 2026-10-26 in `refcv9_mapsel/code/mapsel_roll.py:144-154`).
* Own sidewalk flags in the |y| < 3 m corridor (`refcv9_mapedge/diag/nearfield_prec.out`): exact precision 0.251 at 0-5 m, 0.716-0.744 at 5-40 m.
  Cells with x < ~4 m are outside the loss's `seen` mask under `--map-lift-valid-mask` (DEFAULT): the fine-tune does NOT supervise them and is
  not expected to fix them; MAPSOFT-G's r0 already lets the cost ignore them.

## 1. The arm MAPFT (one arm; recipe fixed here)
* Warm start `ckpt_ema_48852.pt` (md5 b4364029b68f29a33eeb4aebbbbd002b, `ema_model_only`; on Thor at `/home/nvidia/r8ns/ckpt/`), refcv8's
  argv from its `config.json` with the gate-documented changes only: `+ --r8-train-only-map --conflict-detector off`, `--init-from` = that
  file, minus `--b1-guard-*`, `--grad-share-*`, `--conflict-every`. Same data, corpus, parity key, `--batch 16`, `--w-map-hires 1.0`, the
  same frozen class-weight file, `--map-lift-valid-mask` (default). Trainable: `_map_hires.{refine, near, near_refine}` = 105,032 parameters.
* **Budget (MAPFT-BUDGET, fixed by rule here):** the 50-step timing smoke measures `s_step` (the trainer's own per-step figure; check which
  divisor the log uses). `N = min(10000, floor(0.85 * (21600 - smoke_wall_s - 1800) / s_step))` (1800 s reserved for in-run evals and the
  save); **if N < 1000 the run does not launch** (STOP, report s_step to the PI). `--steps N --warmup 200 --lr 1e-4` (refcv8's peak, the
  trainer's schedule), `--eval-every max(500, N // 4)`, `--ema-decay round(1 - 10 / N, 5)` (horizon N/10). Scored weights: the EMA at step N;
  the raw weights read beside, never scored.
* Launch only on a launch-gate PASS of profile `refcv8-mapft` (incl. its two RED arms) at the landed commit. One training seed (0).

## 2. Controls (a failed control voids the bars)
* **K-FROZEN:** `assert_map_only_finetune.py --src ckpt_ema_48852.pt --dst <final>` exits 0 on the EMA view AND the raw view; it must name
  changed tensors ONLY under the three map modules, and at least one in each.
* **K-PLAN-ID (the PI's "planner bit-identical", asserted on outputs):** the MAPSEL roll of MAPFT on eval139 (os-only, per-frame feed,
  inference seed 0, the registered window list) emits, on EVERY window, a plan, fan and `sel` bit-equal to the base trace
  `F:/refcv9_mapsel/trace_s0.npz` (and its MAPPATH derivative). Failure = the planner reads these modules: STOP, report, no bar is read.
* **K-RAST-MOVED:** the MAPFT own rasters differ from the base rasters on some window (else the run trained nothing).
* PREREG_MAPPATH's controls re-run unchanged on the swapped rasters (K-ID, K-BASE, K-PROG per A2, S-K1..S-K3, the deranged arm).

## 3. Bars (PREREG_MAPPATH's instrument VERBATIM, own rasters swapped BASE-map -> MAPFT-map; S6c windows; paired episode-cluster bootstrap
B 2000 seed 0 over EPISODES; one training seed; inference seed 0 -- the MAPSEL trace is seed 0 only; a seed-1 roll is the replicate below)
* **PASS = PREREG_MAPPATH's M1-M5 on `MAPSOFT-G-XF(MAPFT)`** exactly as registered (M1 sidewalk separated below BASE; M2 edge separated
  below BASE; M3 no four-family row worse; M4 deranged not below BASE; M5 STOP guard).
* Beside, never a bar: the share of GT-T1's sidewalk gain recovered; OWN-T1 (hard filter) on MAPFT; HUMAN own false-alarm rate per band and
  the corridor precision per band (the RANGE_DIAG / nearfield_prec instruments) for BASE-map vs MAPFT-map; the T0 map read (perc8 / R8-P2-MAPTOL
  instruments: P / R / F / IoU per class x 20 m band, paired vs EMA-48,852) -- tier T0 perception, never "driving performance".
* **Replicate:** on PASS, one seed-1 MAPSEL roll of MAPFT + BASE; REPLICATED iff M1 and M3 hold at seed 1 too.

## 4. Tier / families
T1 self-action OPEN LOOP on logged eval139 frames (not driving performance) for every planning row; T0 for the perception rows. Families: the
map-compliance rows (sidewalk / edge / lane cross) + `PR.METRICS` LONGITUDINAL, LATERAL, TACTICAL + `ade_m`; distance-keeping (no lead track,
n = 0) and STRATEGIC (refcv8 runs --no-strategic, n = 0) not computable -- stated per family.

## 5. Committed next steps
* **PASS (+ REPLICATED)** -> map-guided path choice on MAPFT's map is a refcv8 deliverable candidate: a SHIP-R8 amendment needs a PI go (it
  changes the emitted plan and the shipped map weights); refcv9 design input: the map head is trained for precision with the planner frozen.
* **FAIL, map improved** (corridor precision at 4-40 m and the HUMAN false-alarm rate both better, T0 rows not worse) -> the cost form is the
  bottleneck: next = a learned map critic on the own-map softmax fitted on TRAIN windows with GT-map labels (option (b), DrivableCritic),
  pre-registered first, 0 Thor-h.
* **FAIL, map not improved** -> the loss / head class is the bottleneck: next = a precision-targeted term (boundary / distance-transform or a
  false-positive-weighted CE) -- a code change, a new gate run and more Thor time = a PI decision, named with the measured numbers.
* **K-PLAN-ID fails** -> the freeze premise is false; report to the PI before anything else. **Smoke cost over the cap (N < 1000)** -> STOP.
