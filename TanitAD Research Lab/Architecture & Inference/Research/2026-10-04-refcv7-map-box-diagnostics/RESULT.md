# RESULT — refcv7-r101-s0 map and box diagnostics (final checkpoint, step 50,400)

Architecture & Inference, 2026-10-04. Pre-registration: `SPEC.md`, sha256 `0add0048…3661`, recorded
2026-10-03T22:27:52Z (`raw/SPEC_SHA256.txt`), before any number below was computed.

**Tier: OPEN-LOOP PERCEPTION DIAGNOSTIC.** One teacher-forced forward per window through the trainer's own
`compute_losses_v3` in eval mode. No planner in the loop; nothing here is a T1 or driving number.
**Estimator:** every interval is an episode-cluster bootstrap (B = 1,000, 95 % percentile) over the split's
episodes. It answers *"would another draw of EPISODES say this?"* only — it is blind to training-run variance
(one seed, no replicate). The forward of the map and box heads was checked for inference determinism (C7).

## Headline

> **ESCALATION — INTEGRATION (Master Mind):** fixes F1 / F2 / F4 below are inference-time code changes to LANDED
> modules (`stack/tanitad/models/map_head_hires.py::decide` + the trainer's in-run map monitor;
> `stack/tanitad/eval/detection_metrics.py` gate / the A10 alarm). They are specified and measured here, NOT
> implemented in the stack (hand-over rule). F3 / F6 need a branch training run — blocked on the launch gate.

**ALARM 1 — map thin classes: a DECISION-RULE ARTEFACT on top of a LOCALISATION-LIMITED head. No collapse, no
generalisation gap.** The declared `prior_corrected` rule elects almost no thin-class cell (EVAL pred/gt: edge
0.0014, hatched 0.000, lane 0.114); only 1.2 × 10⁻⁶ of GT-edge cells reach p̂ ≥ 0.5. In the run's own record the
declared-rule edge IoU never exceeded 0.0008 at any of 101 in-run eval rows — the thin classes were never
predicted, so nothing "collapsed at 5,000". **Inference-time fix F1 (TRAIN-fitted per-class thresholds), EVAL
IoU: lane 0.068 → 0.164, crosswalk 0.048 → 0.105, arrow 0.027 → 0.059, edge 0.000 → 0.042, hatched 0.000 →
0.062** (paired CIs exclude 0), drivable / sidewalk unchanged or better — and it equals the EVAL oracle, so the
decision lever is exhausted. The remaining ceiling is localisation × range: AUROC 0.85–0.95, train ≈ eval (all
gap CIs contain 0), edge IoU 0.042 → 0.113 at 0.2 m tolerance (0.062 → 0.235 in 0–20 m), a 20× near→far drop,
labels registered (94 % of GT-edge cells within 0.2 m of the GT drivable boundary) while the PREDICTED drivable
boundary is within 0.2 m of the GT edge for only 37.5 % (0–20 m) → 9.5 % (80–100 m). Next lever F3 (recipe) is
blocked on the launch gate.

**ALARM 2 — box conf_ratio: CLEARS ITS BAR with an inference-time gate; the focal-prior hypothesis is NOT
supported.** conf_ratio reproduced exactly (2 / 339). Under the focal loss (α 0.25) the 0.5 gate means "match
belief ≥ 0.75" (`slot_presence.gate_match_belief` = 0.7500), and the presence score is FLAT, not low (Σp / n_pos
6.4; bulk over-, top under-confident; the agent head's max p is 0.378 over 318,300 EVAL slots). **F4: the TRAIN
P = R gate (box3d 0.257 / agent 0.231) gives EVAL conf_ratio 0.973 [0.850, 1.090] / 0.995 [0.807, 1.187], F1
0.016 → 0.303 (box3d), AP unchanged (0.248 / 0.131).** No calibration map reaches the band; undoing the 0.01 prior
overshoots to 88.9. The planner consumes the agent head's σ(z) directly — F4 is for the detection readout only.

Compute: Thor GPU, 10 passes, **77.4 min** of pass wall-clock in total (8,296 window forwards; peak `torch.cuda.max_memory_allocated()` 8.1 GB at batch 8, 15.8 GB at batch 16), ~27 min queued behind the video renderer on the shared GPU lock, ~20 min Thor CPU for the analyses; dev box CPU only (tests, tables). Per-pass records: `raw/run_*.json`.

## 0. What was run (compute and controls)

| pass | windows | episodes | ckpt | wall | artifact |
|---|---|---|---|---|---|
| inrun_final (eval_inrun) | 128 | 80 | 50400 | 3.1 min | `raw/run_inrun_final.json` |
| train_fitpass (train_diag) | 1112 | 139 | 50400 | 9.4 min | `raw/run_train_fitpass.json` |
| eval_final (eval_diag) | 1112 | 139 | 50400 | 10.9 min | `raw/run_eval_final.json` |
| train_final (train_diag) | 1112 | 139 | 50400 | 10.6 min | `raw/run_train_final.json` |
| eval_s5000 (eval_diag) | 1112 | 139 | 5000 | 9.2 min | `raw/run_eval_s5000.json` |
| eval_s15000 (eval_diag) | 1112 | 139 | 15000 | 9.1 min | `raw/run_eval_s15000.json` |
| eval_s20000 (eval_diag) | 1112 | 139 | 20000 | 9.1 min | `raw/run_eval_s20000.json` |
| eval_s30000 (eval_diag) | 1112 | 139 | 30000 | 9.1 min | `raw/run_eval_s30000.json` |
| cal256_final (train_calib256) | 256 | 64 | 50400 | 3.8 min | `raw/run_cal256_final.json` |
| inrun_final_rep (eval_inrun) | 128 | 80 | 50400 | 3.1 min | `raw/run_inrun_final_rep.json` |

Model: the run's eval loader `stack/tanitad/eval/refcv7_loader.py` (STRICT load, 0 missing / 0 unexpected
keys, G-DVB 0 mismatches over 221 registry entries, documented departures: `--trunk-compile` dropped, data paths remapped to the kit layout (the same files on Thor), tactical-goal constants read from `config.json`).
GPU: Thor, every pass under `flock /home/nvidia/refcv7_post/thor_gpu.lock` (the video renderer held it
23:02–≈23:29 UTC; passes waited, none ran concurrently). Peak `torch.cuda.max_memory_allocated()`
8.1 GB at batch 8 (15.8 GB at batch 16, in-run reproduction).

| control | must read | read | verdict |
|---|---|---|---|
| C1 my map inter/union == the trainer's own `map_hires_inter*`/`union*` (both rules), every batch of every pass | 0 difference | max abs diff **0.0** over 161,760 values | PASS |
| C1' map keys of the reproduced in-run row == the logged step-50,400 row | rel ≤ 1e-6 | 445 map keys: **444 bit-identical**; the map loss 0.95292573 vs 0.95292567 (5.5e-8 rel) | PASS |
| C2 (B-a) box census on the in-run 128 windows == the logged row | 2 / 2 / 339 (box3d), 0 / 0 / 339 (agent) | box3d n_conf 2, tp 2, n_pos 339, conf_ratio 0.0059; agent 0 / 0 / 339 / 0.0 | PASS (exact) |
| C3 unit: GT-as-pred IoU 1; empty pred IoU 0 / R 0; 1-cell shift IoU_0 = 0, IoU_1 = 1 | literals | `code/test_diag_metrics.py` 11/11 on Thor with the run's stack (`-W error::UserWarning`); 11 pass on the dev box, where the focal-inversion test returns early with a warning (stack not importable there) | PASS |
| C4 unit: histogram AUROC perfect 1.0 / constant 0.5; histogram IoU == direct mask IoU | literals | same tests | PASS |
| C5 AP@0.5/1/2/4 m identical across T0…T3 | ≤ 1e-12 | max spread 0.00e+00 (float64 transforms) | PASS |
| C6 unit: shift / temperature fit recovers planted +2.5 / T 2; ECE(calibrated) < 0.02; focal inversion p 0.5 → π 0.75 | literals | same tests | PASS |
| C7 inference determinism: in-run 128 windows run twice | identical | map counts max abs diff 0; histogram cells changing bin 0; box presence logits max abs diff 0 over 72,600 slots; gate flips 0 | PASS (bit-identical) |

## 1. ALARM 1 — the 10 cm map's thin classes

### 1.1 There was no collapse: under the declared rule the thin classes were never predicted (in-run record)

MEASURED from the run's own `metrics.jsonl` (banked as `raw/inrun_curve.json`, 101 in-run eval rows, and
`raw/inrun_train_rows_curve.json`, 1,008 training rows in 54 buckets):

* **edge** IoU under the declared `prior_corrected` rule: max **0.0008** (all bands) / **0.0042** (0–20 m) over
  all 101 eval rows (steps 500 → 50,400); max **0.0006** / **0.0029** over the training rows. The raw-argmax IoU
  over the same cells rose from 0.000 (step 500) to 0.025–0.031 at every row from 30,000 on.
* **hatched**: 0.000 under both rules at every eval row.
* lane / crosswalk / arrow: declared-rule IoU low throughout (in-run max 0.081 / 0.042 / 0.040, all bands). The
  128-window monitor is noisy for these classes (crosswalk reads 0.021 at 50,400 after 0.042 at 17,500); the
  1,112-window M-e (§1.6) shows no decline for any of them.

⇒ The brief's "collapsed from step 5,000 on" is **not what the record shows**: the declared-rule thin-class
signal was ~0 from the first eval row. The EVAL-DIAG milestone passes (M-e, §1.6) are the decision-grade
version of this curve.

### 1.2 M-c — the declared rule suppresses the thin classes; a TRAIN-fitted rule recovers them to the ceiling

EVAL-DIAG, 1,112 windows / 139 episodes, all 5 bands pooled, supervised cells only. IoU [95 % CI].
`pc` = declared `prior_corrected`, `raw` = plain argmax, `thr_phat` = one-vs-rest threshold τ*_c on
p̂ = softmax(z − ln w) fitted on TRAIN-DIAG, `off` = argmax(z − ln w + δ) with δ fitted on TRAIN-DIAG.
ORACLE = the best threshold tuned on EVAL (a ceiling, NOT a result). MEASURED, `raw/M_c.json`.

| class | n_gt cells | pc IoU | thr_phat IoU | raw IoU | off IoU | pc pred/gt | thr_phat pred/gt | ORACLE |
|---|---|---|---|---|---|---|---|---|
| lane | 5,026,914 | 0.068 [0.056, 0.078] | **0.164** [0.145, 0.181] | 0.163 | 0.161 | **0.114** | 1.061 | 0.164 |
| crosswalk | 1,517,290 | 0.048 [0.033, 0.067] | **0.105** [0.083, 0.132] | 0.102 | 0.084 | **0.096** | 0.654 | 0.105 |
| arrow | 180,291 | 0.027 [0.018, 0.037] | **0.059** [0.045, 0.072] | 0.057 | 0.056 | **0.063** | 0.623 | 0.059 |
| edge | 1,717,086 | 0.000 [0.000, 0.000] | **0.042** [0.037, 0.047] | 0.032 | 0.037 | **0.0014** | 1.667 | 0.042 |
| hatched | 164,449 | 0.000 [0.000, 0.000] | **0.062** [0.005, 0.106] | 0.000 | 0.041 | **0.000** | 1.319 | 0.062 |
| drivable | 103,061,216 | 0.574 [0.551, 0.595] | 0.576 [0.554, 0.597] | 0.558 | 0.532 | 0.976 | 1.015 | 0.576 |
| sidewalk | 156,552,688 | 0.467 [0.429, 0.500] | 0.500 [0.468, 0.530] | 0.473 | 0.485 | 0.925 | 1.325 | 0.500 |
| nocls | 192,388,388 | 0.605 [0.567, 0.639] | 0.608 [0.571, 0.643] | 0.601 | 0.595 | 1.115 | 1.190 | 0.608 |

Paired EVAL difference `thr_phat − pc` (same windows, episode-cluster bootstrap): lane +0.097 [0.086, 0.107],
crosswalk +0.058 [0.044, 0.070], arrow +0.032 [0.022, 0.042] (the pre-registered "best by TRAIN" for arrow is `thr_q`:
+0.030 [0.021, 0.039]), edge +0.042 [0.036, 0.047], hatched +0.062 [0.005, 0.106].

**Why the declared rule does this (mechanism, MEASURED):** only **1.2 × 10⁻⁶** of GT-edge cells and **0** of
GT-hatched cells have p̂ ≥ 0.5 (lane 6.1 %, crosswalk 3.6 %, arrow 2.1 %; `M_c.json`
`frac_gt_cells_score_ge_0p5`). `prior_corrected` is the Bayes (accuracy-optimal) decision on the calibrated
posterior; on a 1–3-cell line whose position the head knows only to a few cells, the posterior mass is spread
over the line and its neighbours, so the line class almost never holds the plurality. It is the right rule for
pixel accuracy and the wrong one for thin-class IoU. The TRAIN-fitted thresholds are low for exactly that
reason: τ* on p̂ = 0.130 lane, 0.106 crosswalk, 0.054 arrow, **0.041 edge**, 0.044 hatched (0.373 drivable,
0.308 sidewalk, 0.399 nocls) — `raw/fit_map.json`.

**The decision lever is exhausted — measured three ways:** (i) `thr_phat` equals the EVAL ORACLE to 3 dp for
every class, so no threshold could do better on these windows; (ii) POST-HOC per-class × per-band thresholds
(fitted on TRAIN per band) are WORSE pooled (edge 0.035, crosswalk 0.052 vs 0.042 / 0.105;
`raw/posthoc_band_thresholds.json`); (iii) the partition rule `off` is within 0.005 of `thr_phat` for lane /
arrow / edge and below it for crosswalk / hatched.

### 1.3 M-a — no generalisation gap; the head does not fit its own TRAINING windows either

TRAIN-DIAG (139 train episodes × 8 windows, same pipeline, eval mode) vs EVAL-DIAG, all bands, gap = train −
eval with an unpaired episode bootstrap. MEASURED, `raw/M_a.json`.

| class | pc train / eval | thr_phat train / eval | thr_phat gap [95 % CI] |
|---|---|---|---|
| lane | 0.070 / 0.068 | 0.162 / 0.164 | −0.003 [−0.031, 0.026] |
| crosswalk | 0.055 / 0.048 | 0.123 / 0.105 | +0.018 [−0.020, 0.054] |
| arrow | 0.037 / 0.027 | 0.071 / 0.059 | +0.012 [−0.008, 0.032] |
| edge | 0.000 / 0.000 | 0.043 / 0.042 | +0.001 [−0.006, 0.008] |
| hatched | 0.000 / 0.000 | 0.054 / 0.062 | −0.007 [−0.073, 0.071] |
| drivable | 0.588 / 0.574 | 0.591 / 0.576 | +0.015 [−0.014, 0.043] |

Every interval contains 0 (D4 FALSE for every class). The TRAIN τ* IoU is **< 0.15 for crosswalk, arrow, edge
and hatched** (lane 0.162). ⇒ Not over-fitting and not a train/eval shift: the head reaches the same low ceiling
on the windows it was trained on. The pre-launch overfit binding (edge ≈ 0.55, INHERITED from the brief) shows the
head can MEMORISE a small set; on 1,112 training windows it does not localise a 1-cell line (0.043).

### 1.4 M-b + M-d — the signal is there; the cell is not (localisation, worsening with range)

AUROC of p̂ on GT-c vs other supervised cells (EVAL, all bands / 0–20 m / 80–100 m; `raw/M_d.json`): lane
0.909 / 0.952 / 0.861, crosswalk 0.922 / 0.970 / 0.856, arrow 0.910 / 0.959 / 0.869, **edge 0.850 / 0.889 /
0.787**, hatched 0.946 / 0.988 / 0.875, drivable 0.921 / 0.960 / 0.875. ⇒ D2 ("no signal": AUROC < 0.80 and no
alternative ≥ 0.05) is FALSE for every class.

Boundary-tolerant IoU_k under `thr_phat`, k cells of 0.1 m (k = 0, 1, 2 pre-registered; **k = 5, 10 POST-HOC**,
§5). MEASURED, `raw/M_b.json`:

| class | band | IoU_0 | IoU_1 (0.1 m) | IoU_2 (0.2 m) | IoU_5 (0.5 m)* | IoU_10 (1 m)* |
|---|---|---|---|---|---|---|
| edge | all | 0.042 | 0.086 | 0.113 | 0.159 | 0.197 |
| edge | 0–20 m | 0.062 | 0.161 | 0.235 | 0.374 | 0.484 |
| edge | 80–100 m | 0.003 | 0.004 | 0.006 | 0.009 | 0.011 |
| lane | all | 0.164 | 0.225 | 0.264 | 0.330 | 0.385 |
| lane | 0–20 m | 0.283 | 0.418 | 0.494 | 0.601 | 0.679 |
| lane | 80–100 m | 0.062 | 0.079 | 0.092 | 0.120 | 0.151 |
| crosswalk | 0–20 m | 0.236 | 0.311 | 0.363 | 0.451 | 0.544 |
| hatched | 0–20 m | 0.101 | 0.131 | 0.146 | 0.164 | 0.178 |
| drivable | all | 0.576 | 0.612 | 0.635 | 0.677 | 0.724 |

Pre-registered D3 ("AUROC ≥ 0.90, IoU_0 < 0.15, IoU_2 ≥ 2 × IoU_0") is FALSE for every class **by its letter**:
edge has IoU_2 = 2.7 × IoU_0 but AUROC 0.850 < 0.90; lane has IoU_0 0.164 > 0.15; crosswalk / arrow / hatched gain
< 2× by 0.2 m. **What the table shows instead (POST-HOC reading, not a pre-registered verdict):** the loss is
dominated by RANGE — edge IoU_0 falls 20× from 0–20 m to 80–100 m, and even 1 m of tolerance recovers only 0.011
at 80–100 m — and, in the near band, by sub-metre localisation (edge 0.062 → 0.235 at 0.2 m → 0.484 at 1 m).

ESTIMATED geometric bound (arithmetic on MEASURED config stamps: cylindrical frame, f_ref 488.92 px, 416 × 1024,
stride-8 tap `layer2` 52 × 128, camera height 1.20–1.69 m in the run's extrinsics census): one stride-8 feature
column spans 8 / 488.92 = 0.0164 rad = **0.16 m lateral at 10 m, 0.33 m at 20 m, 0.65 m at 40 m, 1.6 m at
100 m**; one stride-8 row spans ≈ 8x² / (f·h) of ground depth ≈ **1.1 m at 10 m, 4.5 m at 20 m** (h = 1.45 m).
The 10 cm label grid is 3–16× finer than the feature footprint beyond 20 m.

### 1.5 M-f — the labels are self-consistent; the PREDICTED drivable boundary is not where the GT edge is

MEASURED, `raw/M_f.json` (EVAL; the TRAIN side reads alike — registration buckets within 0.04, median widths within 1 cell):

* GT class fraction of supervised cells: lane 1.09 %, crosswalk 0.33 %, arrow 0.04 %, edge 0.37 %, hatched 0.04 %.
* GT line width (lateral run length per fine row): **edge median 1 cell, p90 2 cells (0.1–0.2 m)** in every band;
  lane median 3 (p90 7); arrow 3 (8–9); crosswalk 5 (10–13); hatched 5–6 (9–17). The label width does not change
  with range while the image footprint grows ~10× (§1.4).
* Registration spot-check, Chebyshev distance from each GT-edge cell (1,717,086 cells) to the nearest drivable
  boundary: to the **GT** drivable boundary **87.6 % at 1 cell, 94.4 % ≤ 2 cells** (94.0–94.9 % in every band) ⇒
  the edge label sits on the GT drivable boundary: the GT is registered and self-consistent. To the **PREDICTED**
  (pc) drivable boundary: ≤ 2 cells for **37.5 % (0–20 m), 18.0 % (20–40), 12.1 % (40–60), 10.2 % (60–80), 9.5 %
  (80–100 m)**, and > 2 m for 12 % → 49 % of edge cells. ⇒ the head's own drivable boundary — on which the edge
  class must sit — is off by > 0.5 m for most edge cells beyond 20 m.

### 1.6 M-e — when (EVAL-DIAG at every banked checkpoint)

EVAL-DIAG (same 1,112 windows) at each checkpoint; ORACLE = best one-vs-rest threshold on p̂ tuned on EVAL (a trend diagnostic of separable signal, never a result). MEASURED, `raw/M_e.json`.

| class | metric | 5000 | 15000 | 20000 | 30000 | 50400 |
|---|---|---|---|---|---|---|
| lane | IoU pc (declared) | 0.042 | 0.070 | 0.053 | 0.056 | 0.068 |
| lane | IoU raw | 0.124 | 0.148 | 0.154 | 0.157 | 0.163 |
| lane | pred/gt pc | 0.066 | 0.131 | 0.086 | 0.089 | 0.114 |
| lane | AUROC p̂ | 0.883 | 0.897 | 0.899 | 0.905 | 0.909 |
| lane | ORACLE thr IoU | 0.138 | 0.152 | 0.155 | 0.161 | 0.164 |
| crosswalk | IoU pc (declared) | 0.043 | 0.036 | 0.048 | 0.044 | 0.048 |
| crosswalk | IoU raw | 0.084 | 0.091 | 0.096 | 0.097 | 0.102 |
| crosswalk | pred/gt pc | 0.117 | 0.074 | 0.104 | 0.089 | 0.096 |
| crosswalk | AUROC p̂ | 0.876 | 0.905 | 0.910 | 0.918 | 0.922 |
| crosswalk | ORACLE thr IoU | 0.084 | 0.105 | 0.098 | 0.101 | 0.105 |
| arrow | IoU pc (declared) | 0.001 | 0.013 | 0.025 | 0.029 | 0.027 |
| arrow | IoU raw | 0.016 | 0.041 | 0.051 | 0.058 | 0.057 |
| arrow | pred/gt pc | 0.002 | 0.024 | 0.060 | 0.075 | 0.063 |
| arrow | AUROC p̂ | 0.883 | 0.896 | 0.900 | 0.907 | 0.910 |
| arrow | ORACLE thr IoU | 0.038 | 0.054 | 0.055 | 0.059 | 0.059 |
| edge | IoU pc (declared) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge | IoU raw | 0.011 | 0.033 | 0.029 | 0.029 | 0.032 |
| edge | pred/gt pc | 0.000 | 0.003 | 0.002 | 0.001 | 0.001 |
| edge | AUROC p̂ | 0.822 | 0.843 | 0.842 | 0.847 | 0.850 |
| edge | ORACLE thr IoU | 0.029 | 0.039 | 0.039 | 0.041 | 0.042 |
| hatched | IoU pc (declared) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hatched | IoU raw | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hatched | pred/gt pc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hatched | AUROC p̂ | 0.879 | 0.928 | 0.932 | 0.943 | 0.946 |
| hatched | ORACLE thr IoU | 0.045 | 0.059 | 0.059 | 0.059 | 0.062 |
| drivable | IoU pc (declared) | 0.544 | 0.564 | 0.568 | 0.568 | 0.574 |
| drivable | IoU raw | 0.527 | 0.547 | 0.556 | 0.552 | 0.558 |
| drivable | pred/gt pc | 1.064 | 0.972 | 0.934 | 1.009 | 0.976 |
| drivable | AUROC p̂ | 0.907 | 0.917 | 0.917 | 0.919 | 0.921 |
| drivable | ORACLE thr IoU | 0.546 | 0.567 | 0.571 | 0.569 | 0.576 |

**M-e reading (SPEC §5 date rule: first milestone whose declared-rule IoU falls below half its 5,000-step value):** none for any thin class (`raw/DECISIONS_map.json` `M_e_first_step_pc_iou_below_half_of_5000` = lane None, crosswalk None, arrow None, edge None, hatched None; for edge and hatched the 5,000-step value is already 0, so the rule is undefined). Every signal-side quantity ends higher at 50,400 than at 5,000 for every thin class — AUROC (edge 0.822 → 0.850, hatched 0.879 → 0.946; one dip of 0.001, edge at 20,000), raw IoU and ORACLE IoU (edge 0.029 → 0.042; dips of at most 0.007, crosswalk ORACLE at 20,000). **Nothing collapsed; the declared rule never let the thin classes through, and the head improved slowly underneath it.**

### 1.7 Alarm-1 verdict (SPEC §5 applied)

| class | D1 rule suppression | D2 no signal | D3 localisation (letter) | D4 gen. gap | TRAIN τ* IoU < 0.15 |
|---|---|---|---|---|---|
| lane | **TRUE** | false | false (IoU_0 0.164 > 0.15) | false | false (0.162) |
| crosswalk | **TRUE** | false | false | false | **TRUE** (0.123) |
| arrow | **TRUE** | false | false | false | **TRUE** (0.071) |
| edge | **TRUE** | false | false (AUROC 0.850 < 0.90) | false | **TRUE** (0.043) |
| hatched | **TRUE** | false | false | false | **TRUE** (0.054) |

(`raw/DECISIONS_map.json`; "best alternative" for D1 = the TRAIN-fitted decision with the highest TRAIN IoU.)

**Most likely root cause, with evidence — two stacked causes.** (1) The alarm itself is a **decision-rule
artefact**: `prior_corrected` cannot elect a 1–3-cell class whose position the head knows only to several cells
(edge p̂ ≥ 0.5 on 1.2 × 10⁻⁶ of edge cells); TRAIN-fitted thresholds recover every thin class to the EVAL ceiling.
(2) The ceiling is low because the head is **localisation- and range-limited**, not under-fitted on a shift:
AUROC 0.85–0.95, no train/eval gap, TRAIN IoU equally low, 0.2 m of tolerance multiplies edge IoU 2.7×, the near
band ≫ the far band, labels registered, and the predicted drivable boundary itself misses the GT edge by > 0.5 m
beyond 20 m — consistent with a stride-8 feature footprint of 0.33 m × 4.5 m at 20 m (ESTIMATED).

⚠️ Scope: the planner reads the map through `--bev-source map_hires_pool` (the 0.25 m encoder FEATURES), not
through any decision rule — so (1) changes the map PRODUCT and its monitor, not the planner's input.

## 2. ALARM 2 — box `conf_ratio`

### 2.1 B-a — definition and reproduction

`conf_ratio = n_conf / n_pos` — `stack/tanitad/eval/detection_metrics.py:243-253` (`_census`: greedy 2 m rows
over slots with `sigmoid(presence_logit) ≥ 0.5`; IGNORE-matched rows are DontCare and excluded; `n_pos` = VIS-1
positives); alarm band `:59` `CONF_RATIO_BAND = (0.5, 1.5)`, applied at `:273-275`; gate
`stack/tanitad/models/slot_presence.py:68` `DETECTION_GATE = 0.5`. Reproduced EXACTLY on the in-run 128 windows
at `ckpt.pt` (C2): box3d n_conf **2**, tp **2**, n_pos **339**, conf_ratio **0.0059**; agent **0 / 0 / 339 /
0.0**. (AP and box loss terms reproduce within 6 % relative — box3d AP@2 m 0.22776 vs logged 0.22867. C7 shows this
forward is bit-deterministic, so the residual is most likely the loader's one documented departure, dropping
`--trunk-compile` (INFERRED, not isolated); the census at the gate and every map inter/union key are exact.) In the in-run record the box3d
conf_ratio never exceeded **0.0059** and the agent's was **0.0** at all 101 eval rows (`raw/inrun_curve.json`):
like the map, this alarm was present from the first eval.

### 2.2 B-b — the scores are compressed, not low on average

EVAL-DIAG, box3d: 1,061 labelled windows, 137 episodes, n_pos 3,390, 316,216 scored slot rows (DontCare
excluded). MEASURED, `raw/B_box.json`.

* Max presence probability over all 318,300 slots: **box3d 0.688** (29 slots ≥ 0.5); **agent 0.378** (0 slots ≥
  0.5 — on these windows no agent slot passes the declared gate). Per-window best slot: median 0.255 (box3d), 0.199
  (agent).
* Reliability (box3d, 15 bins): the BULK is over-confident (p 0.032 → TP rate 0.000 on 191,492 rows; 0.095 →
  0.006 on 82,614), the TOP is under-confident (p 0.357 → 0.55; 0.417 → 0.82). ECE_all 0.059, ECE_{p≥0.05}
  0.091. **Σp / n_pos = 6.42** (agent 7.23): in aggregate the head "predicts" 6× more objects than exist.
  "Under-confident" is wrong as a global statement; the score distribution is too FLAT (the TRAIN BCE temperature
  fit gives T* = 0.46, i.e. sharpen).
* Histograms (20 bins of p): box3d TPs peak at 0.15–0.25, FPs at 0.00–0.05; 27 of 3,390 EVAL positives are found
  above 0.5.

### 2.3 B-c / B-d — recalibration without retraining (fitted on TRAIN-DIAG, scored on EVAL-DIAG)

**box3d** — EVAL-DIAG 1061 labelled windows / 137 episodes, n_pos 3390, 316216 scored slot rows. TRAIN-DIAG fit: shift b* -2.052, T* 0.464, Platt a 3.106 b +1.680, P = R gate 0.2567 (TRAIN rows 320614, TP 3134). MEASURED, `raw/B_box.json`.

| transform | conf_ratio [95 % CI] | prec | rec | F1 [95 % CI] | F1 − T0, paired | AP@2 m | AP@4 m | ECE_all | ECE_{p≥0.05} | Σp / n_pos |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 identity (declared, gate 0.5) | 0.008 [0.004, 0.014] | 0.964 | 0.008 | 0.016 [0.007, 0.026] | [0.000, 0.000] | 0.2483 | 0.3299 | 0.0590 | 0.0905 | 6.42 |
| T1a prior shift +4.595 (no fit) | 88.859 [75.561, 105.873] | 0.011 | 0.985 | 0.022 [0.018, 0.026] | [-0.005, 0.016] | 0.2483 | 0.3299 | 0.7902 | 0.7902 | 74.69 |
| T1b fitted shift | 0.000 [0.000, 0.000] | – | 0.000 | – – | – | 0.2483 | 0.3299 | 0.0007 | 0.3637 | 0.92 |
| T1c focal inversion (no fit) | 0.024 [0.016, 0.035] | 0.877 | 0.021 | 0.041 [0.027, 0.057] | [0.014, 0.040] | 0.2483 | 0.3299 | 0.0015 | 0.0165 | 0.99 |
| T2 temperature | 0.008 [0.004, 0.014] | 0.964 | 0.008 | 0.016 [0.007, 0.026] | [0.000, 0.000] | 0.2483 | 0.3299 | 0.0022 | 0.0777 | 0.79 |
| T2b Platt | 0.064 [0.049, 0.081] | 0.724 | 0.046 | 0.087 [0.067, 0.109] | [0.053, 0.093] | 0.2483 | 0.3299 | 0.0004 | 0.0087 | 0.96 |
| T3 TRAIN P = R gate | 0.973 [0.850, 1.090] | 0.307 | 0.299 | 0.303 [0.283, 0.323] | [0.266, 0.308] | 0.2483 | 0.3299 | 0.1546 | 0.1763 | 15.40 |

**agent** — EVAL-DIAG 1061 labelled windows / 137 episodes, n_pos 3390, 316294 scored slot rows. TRAIN-DIAG fit: shift b* -2.170, T* 0.462, Platt a 3.221 b +1.906, P = R gate 0.2307 (TRAIN rows 320660, TP 3130). MEASURED, `raw/B_box.json`.

| transform | conf_ratio [95 % CI] | prec | rec | F1 [95 % CI] | F1 − T0, paired | AP@2 m | AP@4 m | ECE_all | ECE_{p≥0.05} | Σp / n_pos |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 identity (declared, gate 0.5) | 0.000 [0.000, 0.000] | – | 0.000 | – – | – | 0.1314 | 0.1783 | 0.0672 | 0.0939 | 7.23 |
| T1a prior shift +4.595 (no fit) | 91.081 [77.254, 108.840] | 0.011 | 0.984 | 0.021 [0.018, 0.025] | – | 0.1314 | 0.1783 | 0.8209 | 0.8209 | 77.58 |
| T1b fitted shift | 0.000 [0.000, 0.000] | – | 0.000 | – – | – | 0.1314 | 0.1783 | 0.0006 | 0.3218 | 0.92 |
| T1c focal inversion (no fit) | 0.000 [0.000, 0.000] | – | 0.000 | – – | – | 0.1314 | 0.1783 | 0.0018 | 0.0147 | 1.02 |
| T2 temperature | 0.000 [0.000, 0.000] | – | 0.000 | – – | – | 0.1314 | 0.1783 | 0.0015 | 0.0678 | 0.85 |
| T2b Platt | 0.001 [0.000, 0.002] | 1.000 | 0.001 | 0.001 [0.001, 0.004] | – | 0.1314 | 0.1783 | 0.0007 | 0.0121 | 0.97 |
| T3 TRAIN P = R gate | 0.995 [0.807, 1.187] | 0.207 | 0.206 | 0.207 [0.185, 0.224] | – | 0.1314 | 0.1783 | 0.1951 | 0.2066 | 19.19 |


**Readings.** AP does not move (C5). No calibration map (T1b shift, T2 temperature, T2b Platt) reaches the band:
a calibrated P(TP) seldom exceeds 0.5 because, at the operating point where precision = recall, precision is only
~0.31 (box3d) / ~0.21 (agent); the BCE-fitted shift is **negative** (−2.05 box3d, −2.17 agent). The analytic
focal inversion T1c (zero fit) produces CALIBRATED match beliefs (box3d ECE_{p≥0.05} 0.017, Σπ/n_pos 0.99) — the
head's scores are exactly what the focal objective (α 0.25, γ 2) makes them — but only 81 slots reach π ≥ 0.5.
**Only the threshold transform T3 puts conf_ratio in band**, at the TRAIN P = R gate **0.257 (box3d) / 0.231
(agent)** — matching the run's own A10 informative readout on its 256 banked train windows (0.2589 / 0.2357 at
step 50,400, `raw/inrun_curve.json`).

**Sensitivity — the same fits on the A10 banked TRAIN-CALIB256 set** (64 clips × 4 windows), scored on EVAL-DIAG (`raw/B_box.json` `eval_with_calib256_fit`):

| head | fit set | P = R gate | T3 conf_ratio [CI] | T3 F1 | T1b shift | T2 T* | T2b conf_ratio |
|---|---|---|---|---|---|---|---|
| box3d | TRAIN-DIAG (primary) | 0.2567 | 0.973 [0.850, 1.090] | 0.303 | -2.052 | 0.464 | 0.064 |
| box3d | TRAIN-CALIB256 | 0.2598 | 0.913 [0.794, 1.027] | 0.299 | -2.108 | 0.456 | 0.060 |
| agent | TRAIN-DIAG (primary) | 0.2307 | 0.995 [0.807, 1.187] | 0.207 | -2.170 | 0.462 | 0.001 |
| agent | TRAIN-CALIB256 | 0.2358 | 0.811 [0.649, 0.971] | 0.202 | -2.217 | 0.457 | 0.000 |

Both TRAIN fit sets land the EVAL conf_ratio inside [0.5, 1.5] for both heads (0.81–0.99); the P = R gate moves by ≤ 0.005 between them and the fitted shift / temperature by ≤ 0.06 / 0.008. The F4 gate is not a fit-set artefact.

### 2.4 Alarm-2 verdict (SPEC §5 applied)

* **E1 CALIBRATION, ranking intact: TRUE via T3 only** — EVAL conf_ratio box3d **0.973 [0.850, 1.090]**, agent
  **0.995 [0.807, 1.187]**, AP unchanged; F1 at the gate box3d 0.016 → **0.303** (paired Δ [0.266, 0.308]), agent
  undefined (no detections) → **0.207**.
* **E2 transfer failure: FALSE** (the TRAIN gate lands in band on EVAL for both heads).
* **E3 the focal-prior hypothesis: NOT SUPPORTED.** Undoing the 0.01 prior (T1a, +4.595) overshoots to conf_ratio
  **88.9 / 91.1** — box3d moves from 2.08 to 1.95 decades away from 1, i.e. only 0.13 decades closer, not the ≥ 10× required; the analytic focal inversion
  T1c moves box3d 0.008 → 0.024 (3×) and leaves the agent at 0. The prior is an INITIALISATION and is long gone;
  what remains is the focal loss's α = 0.25, which by the module's own analytic function
  (`slot_presence.gate_match_belief(0.5, "focal")` = **0.7500**, `raw/B_box.json` `definition`) makes the 0.5
  gate mean "match belief ≥ 0.75". Under refcv6's BCE + 0.1 no-object weight the same gate meant π ≥ 1/11 (the
  module docstring) — the one-line mechanism for refcv6 over-firing (≈ 3.4, INHERITED from the brief) and refcv7
  under-firing (0.008): **a fixed 0.5 gate is not invariant to the presence loss**, so the [0.5, 1.5] band at gate
  0.5 tests the loss, not the detector. (Cross-run statement, one seed each: a mechanism, not a measured lever
  effect.)
* **Most likely root cause:** the declared gate/statistic, combined with a flat (high-temperature) presence
  score. The detector's real limit is ranking quality — AP@2 m **0.248** (box3d) / **0.131** (agent), P = R at
  ≈ 0.31 / 0.21 — which no inference-time map changes.

## 3. Fixes — classified, with expected effect and verification

| # | alarm | fix | class | expected effect (MEASURED on EVAL-DIAG) | how to verify |
|---|---|---|---|---|---|
| F1 | map | replace the declared `prior_corrected` decision by the TRAIN-fitted per-class probability thresholds on p̂ (`raw/fit_map.json` `tau_phat_prob`): a third `DECISION_RULES` entry in `map_head_hires.decide` + the in-run monitor | **inference-time, no retraining** | lane 0.068 → 0.164, crosswalk 0.048 → 0.105, arrow 0.027 → 0.059, edge 0.000 → 0.042, hatched 0.000 → 0.062; drivable 0.574 → 0.576, sidewalk 0.467 → 0.500 (= EVAL ORACLE). Multi-label output; the partition alternative `off` (δ in `fit_map.json`) is within 0.005 except crosswalk 0.084 / hatched 0.041 | re-run `code/run_diag.py --split eval_diag --fit` with the rule wired; C1 must still read 0 |
| F2 | map monitor | quote thin-class health as the thresholded IoU AND IoU at 0.2 m tolerance, per band; stop alarming on the declared-rule IoU | inference / logging | removes an alarm that has read ~0 since step 500 | the Watch shows edge 0.042 / 0.113 (k=2) instead of 0.000 |
| F3 | map ceiling | **recipe**: a tolerance-aware thin-class target (±2-cell dilated positives, or a distance-to-line regression for lane / edge) and/or a stride-4 image tap for the 0–20 m near lift; branch from `ckpt.pt` | **recipe change, branch run** | the measured headroom it targets: edge 0.042 → 0.113 at 0.2 m (0.062 → 0.235 in 0–20 m). NOT measured that a loss change converts it | v7-tiny ladder first (TanitAD_ValidateAIDesign: pre-reg + deliberate-regression arm), then a branch behind the launch gate; score with this harness (k = 0 and 2, per band) |
| F4 | box | detection gate = the TRAIN-fitted P = R gate per head (box3d 0.257, agent 0.231), or evaluate the A10 alarm at that gate | **inference-time, no retraining** | conf_ratio 0.008 → 0.973 (box3d), 0.000 → 0.995 (agent); F1 0.016 → 0.303 / – → 0.207; AP unchanged | the in-run row already logs `eval_{h}_calib_pr_gate`; recompute the census at that gate (`code/analyze_box.py`) |
| F5 | agent → planner | do NOT feed a re-gated or recalibrated presence to the planner: it consumes `sigmoid(presence_logit)` as a feature and as a SOFT token scale (`refc_agents.py:345, :347, :360`; `presence_hard` False in `config.json`) | recipe if wanted | any change of that input is a distribution shift for a planner trained on the flat scores | branch run only |
| F6 | box ceiling | the real deficit is AP (0.248 / 0.131) and a flat score; the temperature is learnable (T* 0.46) — e.g. a sharper presence objective or more decoder refinement | recipe | not measured here | branch run behind the launch gate |

## 4. Where this stops (Rule Zero)

* **Alarm 2 clears its committed bar** (E1 via T3: in band on EVAL for both heads, AP unchanged).
* **Alarm 1: the decision lever is cleared and exhausted** (F1 reaches the EVAL ORACLE; per-band thresholds and
  the offset rule do not beat it). The remaining lever (F3) needs a TRAINING run — **blocked on the launch gate**
  (PI-binding: no training without a launch-gate PASS token) and on a design choice between its two forms;
  unblocked by a Master-Mind SPEC + gate PASS for a branch from `ckpt.pt`. The cheapest zero-training experiments
  that could still have moved the map number — per-class thresholds (two scores), a partition offset rule,
  per-band thresholds, tolerance and range attribution, label registration — were all run here.

## 5. Post-hoc additions and deviations (logged, not pre-registered)

* **k = 5 and k = 10 tolerances** (0.5 m / 1 m) added to M-b at 22:40 UTC, after reading the run's training-row
  curve and BEFORE the EVAL-DIAG pass started (22:51). The pre-registered k = 0, 1, 2 are reported unchanged; the
  TRAIN fit pass carries k ≤ 2 only (it is used for fitting, not for M-b).
* **Code provenance per pass:** `inrun_final` and `train_fitpass` imported `code/diag_metrics.py` md5 `419c33a0…`
  (k ≤ 2); every later pass imported `34982975…` = the file in `code/` (adds k = 5, 10 and an analysis-side
  guard in `tolerant_summary`). `run_diag.py` and the trainer code are the same for all passes; the package's
  `code/` is byte-identical to Thor's working copy (15 files, md5 compared).
* **Per-band thresholds** (`raw/posthoc_band_thresholds.json`) — post-hoc, zero GPU; worse than the pre-registered
  single τ per class.
* **D3 implementation note:** IoU_2 = IoU_0 = 0 satisfies "IoU_2 ≥ 2 IoU_0" vacuously; applied as requiring
  IoU_2 > 0 (no class hit the vacuous case in the final run).
* **D1 "best alternative"** read as the TRAIN-fitted decision with the highest TRAIN IoU (selection on TRAIN).
* **C5 (AP identical to 1e-12):** the first analysis cast transformed logits to float32 and AP spread up to 6.6e-7 (`raw/B_box_v1_float32.json`); re-run in float64: max spread 0.00e+00. PASS.
* **Box TRAIN fit set:** the TRAIN-DIAG packs come from the map fit pass (`train_fitpass`, `ckpt.pt`, the same
  1,112 windows).
* The brief's "collapse at 5,000" premise is corrected in §1.1 from the run's own record.
* I printed raw clip file names to my own terminal twice while locating the train cache (never written to any
  banked file); every banked artifact carries sha12 only (packs: `ep` removed by `code/bank_strip.py`). A regex scan for UUIDs over all 44 package files reads 0 (positive control: 1).

## 6. For the register (proposed rows — the Master Mind owns `GOALS_AND_CLAIMS.md` / `RETRACTION_LOG.md`)

* **CORRECTION (retraction candidate, class "a monitor read as a model state"):** "refcv7's 10 cm thin classes
  collapsed from step 5,000" — the record shows the declared-rule thin-class IoU was ~0 from step 500 while the
  raw-argmax IoU rose (§1.1, §1.6). The in-run thin-class alarm measures the DECISION RULE.
* **REFUTED:** "the focal presence loss with prior 0.01 made refcv7 under-confident" — the score is flat, not low
  (Σp / n_pos 6.4); undoing the prior overshoots to conf_ratio 88.9; the 0.5 gate's meaning under focal α 0.25 (π ≥ 0.75) is the
  mechanism (§2.4). Next lever F4 RESULT: in band on EVAL for both heads.
* **SUPPORTED (open-loop perception diagnostic, one seed):** the 10 cm thin-class ceiling is localisation × range,
  with no train/eval gap (§1.3–1.5); decision-rule fix F1 measured (§1.2).

