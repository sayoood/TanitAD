# PREREG — emit_off diagnostic on the refcv8 ladder's L1 arms (POST-HOC, lever-choice only)

Written 2026-10-12 12:30Z by the Master Mind, BEFORE any full-grid emit_off number existed. The only emit_off data at the
time was a 16-window smoke (the change count), not a metric.

## Why

Descriptive pre-I-0 reads (`mm_desc/VR8_vs_V0.json`, `VR8d_vs_V0.json`, `VR8_vs_VR8d.json`, paired episode-cluster
bootstrap, both sampler seeds) show V-R8 AND V-R8d both separated WORSE than V0 on the lateral family:
`cross_abs_6s` +0.62..+0.92 m, `curv_mae_0_2s`, `yaw_mae_0_2s`, plus `ade_all` and `off_drivable`. V-R8 and V-R8d differ
only by `--r8-derange-feed`, so the shared loss comes from the R8 bundle, not the fed condition. The bundle part that
most directly changes the DRIVEN plan is emission (`--r8-n-alloc 32 --r8-alloc-emit`, no `--r8-alloc-emit-start`, so
emission is live from step 0). The other parts change TRAINING (cons / v9-cons / alloc-l1 / listwise losses,
nav-from-v9 + nav-args dropout, defect masks).

## Instrument

`ladder_eval_diag.py` = the live `ladder_eval.py` (md5 87d4dfa2) + ONE row, `emit_off`
(`model.core.decoder.r8_emit_live = False`). Same checkpoint, so no training-seed noise enters. It refuses an arm that
does not emit, because the row would then be the identity (R35's class). Dev box RTX 4060, ladder tree `1df35031db`
(byte-equal to Thor's), inputs md5-verified against Thor.
Controls, MEASURED before the full run:
- Cross-host: dev-box `base` vs Thor `base_s0` on 16 identical windows differs by at most 1.3 cm (median 4 mm).
- Live: `emit_off` changes 13 of 16 plans (median 0.72 m, max 5.8 m).
- Scorer: `diag_describe.py` reproduces the Thor V-R8 vs V0 read on all 36 cells (|dCI| ≤ 8.6e-16, 0 flag flips).
- Instrument amendment (13:35Z, before any full-grid number). The first full run crawled at >125 s/batch: the
  loader parsed the full train+eval agent joins (6.6 GB of JSON) into 19.3 GB private memory on a 32 GB box. Both joins
  were filtered to the 147 eval-label clips (139 eval139 clips kept, 26,394 of 875,657 lines) and passed as RECORDED
  overrides. Control: the loader's join census is identical (139/139 episodes, 22,663/23,772 windows, 776,801 boxes),
  and the 16-window `base` plans are bit-identical to the unfiltered run (max |dtraj| 0.000000 m).
- Instrument amendment 2 (13:45Z, before any full-grid number). Batch 8 hit CUDA OOM at batch 2. Other sessions' jobs
  hold about 4 GB of the 8 GB card, and the eval needs about 7.9 GB at batch 8. On CPU it runs about 100-200 s per batch,
  roughly 20 h per arm. So EVERY arm is re-evaluated on the dev-box GPU at `--batch 4`: V0 (base), V-R8 and V-R8d
  (base + emit_off), both sampler seeds. Batch size sets the sampler-seed schedule (`seed * 100003 + j`), so the Thor
  batch-8 dumps are NOT mixed into any comparison. All three arms share host, batching and seeds, and the readout above
  uses the dev-box batch-4 `base` of V0 as its V0 reference. The Thor-vs-dev-box comparison of `base` (batch 8 vs 4,
  same windows) is reported separately as a read of the inference-seed spread.

## Readout (fixed now)

Primary metric `cross_abs_6s`; also `curv_mae_0_2s`, `yaw_mae_0_2s`, `head15_all`, `dir_correct_turn`, `ade_all`,
`off_drivable`, and every other family as reported. For V-R8 (and V-R8d as the replication of the conclusion), "both
seeds" means sampler seeds 0 and 1:

- **(A) EMISSION makes it.** `emit_off` vs the arm's own `base` is separated BETTER on `cross_abs_6s` on both seeds,
  AND `emit_off` vs V0 `base` is NOT separated worse on `cross_abs_6s` on both seeds.
  Next lever: the emission (allocated candidates trained but not emitted, a later `--r8-alloc-emit-start`, or the
  selector). That is a launch-argv question.
- **(B) TRAINING makes it.** `emit_off` vs V0 is still separated WORSE on `cross_abs_6s` on both seeds, AND `emit_off`
  vs the arm's `base` is NOT separated better. Next lever: the training components. That needs training ablations.
- **(C) MIXED.** Anything else. Report the gap fraction closed per metric:
  `(arm_base - emit_off) / (arm_base - V0_base)`.

## Q2 — selection or candidates? (added 16:45Z, after Q1 read B, before any candidate-level number)

Q1 read **B (training makes it)** for both arms: `emit_off` closes 10-18 % of the cross-track gap, not separated. A
zero-training stratification then put the gap on the 107 GT-TURN windows (~80 % of the cross-track gap, ~100 % of the
ADE gap), in windows where the v7 and v9 nav AGREE (so the nav source is refuted). Training logs show V-R8's selector
is trained by the listwise soft-target loss, which REPLACES the selection CE (`refc_v3_train.py:5658-5661`). The CE,
still logged, ends at 3.96 against V0's 2.20.

Instrument: `ladder_eval_diag.py --save-cands` saves each window's full fan (`anchor_traj`: V0 117, V-R8 149 = 117 base +
32 extra). The 8-window probe matched the emitted plan to a fan row in 8/8 windows on both arms. No refcv7 WTA
candidates exist on these arms. From ONE capture per arm/seed (`base` row, batch 4, both sampler seeds), plans are
re-scored with `ladder_score.window_metrics`:
- SEL: the emitted plan, as already scored.
- ORACLE-BASE: the base fan row (index < 117) with the lowest ADE over valid horizons vs GT. This is label-side and an
  upper bound, never deployable.
- ORACLE-ALL (V-R8 only): the same over all 149 rows.
- RANDOM-BASE: a uniform base row per window (numpy seed 0). This is the control an oracle gap is read against.

Readout on the GT-turn windows (turnL ∪ turnR), `cross_abs_6s` and `ade_all`, paired episode-cluster bootstrap, both
sampler seeds:
- **SELECTION** — V-R8's ORACLE-BASE is NOT separated worse than V0's on both metrics and both seeds, while its SEL is
  worse (the Q1 result). The bank holds candidates as good as V0's; the pick is the defect. Next lever: the selector's
  training (listwise replacing the selection CE).
- **CANDIDATES** — V-R8's ORACLE-BASE IS separated worse than V0's on `cross_abs_6s` on both seeds. The decoded fan got
  worse on turns, so the lever is in the decoder's training terms.
- Otherwise **MIXED**: report both gaps per metric.

## What this is NOT

Not a ladder row, not a ladder verdict, not a model claim (tiny rung, one training seed per arm). It ranks one lever
against the others. Any launch-argv change it suggests needs its own pre-registration.
