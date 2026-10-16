# refcv8: the bars are scored on the EMA weights, and the averaging effect is measured, not assumed

Registered 2026-10-16 ~21:45 Berlin by the Master Mind, on the PI's decision. Run: `refcv8-r101-s0` (launch fb47a583763f),
at step ~15,000 of 48,852 when this was written. The final checkpoint does not exist yet.

## 1. Decision (PI, 2026-10-16)

PI, verbatim: *"option 1, and queue the refcv7 averaged baseline"*. Option 1 as put to the PI:

* For every bar in `SPEC_REFCV8_DRAFT.md` section 5 (BAR-R8-E1, E3, NR and the rest), the scored model is the **EMA weights**
  (decay 0.9998, `--ema-decay 0.9998`). That means `ckpt_ema_<N>.pt` at the milestones (5,000 / 15,000 / 20,000 / 30,000) and
  `model_ema.pt` at the end of the run (`stack/scripts/refc_v3_train.py:10837-10842`; written at `step == args.steps`).
* The **raw weights** are reported beside them at every milestone and at the end, on the same windows, with the same estimator and
  the same inference seeds. At the milestones that is `ckpt_<N>.pt`. At the end it is the rolling `ckpt.pt`, slimmed with
  `slim_rolling_ckpt_r8.py`.
* **What had been seen when this was registered:** the step-5,000 batteries on both weights (T1 open loop, standalone clock,
  PROVISIONAL because G0 is FAIL as coded). Raw: NR FAIL, E3 PASS, E1 FAILED. EMA: NR PASS, E3 PASS, E1 FAILED (ties the
  hold-action echo). The spec named no weight file, so the verdict depended on an unmade choice. The PI made the choice, not the
  Master Mind, and it was made about 34,000 steps before the final checkpoint.
  Evidence: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/step5000/` and `raw/step5000ema/`.

## 2. The refcv7 averaged baseline: cannot be built from what was saved (MEASURED)

* refcv7 trained **without an EMA**. Its `config.json` carries no `ema_decay`, and its in-run `ema_eval_*` values are null.
* refcv7's saved checkpoints are at steps 5,000, 15,000, 20,000 and 30,000, plus the final 50,400 (`ckpt.pt`). This is the Thor
  run dir `/home/nvidia/refcv7_run/runs/refcv7-r101-s0/`, listed 2026-10-16.
* A decay-0.9998 average at 50,400 gives step 30,000 a weight of 0.9998^20,400 = 1.7 %. Averaging 30,000 with 50,400 is
  therefore not an EMA; it would mostly measure a blend with a mid-training model. **Not run.**
* A true refcv7 EMA needs refcv7 re-trained from its 30,000 checkpoint to 50,400 with the EMA on. That is ~20,400 steps, about
  2.2 Thor-days at ~9.4 s/step (ESTIMATED), and only after refcv8 finishes (~2026-10-20). It needs the PI's authorisation.
  **Not queued.**

## 3. The replacement: the same question, measured on refcv8 at no extra compute

**Why it is the same question.** What an EMA can add depends on how far the weights inside its window moved. Measure that as the
EMA-weighted learning-rate mass behind the checkpoint: sum over k of w_k * (LR summed from k to T), with w_k proportional to
0.9998^(T-k). The values below were MEASURED from the logged `lr` of both `metrics.jsonl` (watch copies, read 2026-10-16 21:3x):

| checkpoint | EMA-weighted LR mass behind it |
|---|---|
| refcv7 final (50,400; cosine to ~0) | 1.19e-2 |
| refcv8 @5,000 | 1.03e-1 |
| refcv8 @10,500 | 1.67e-1 |
| refcv8 @14,900 | 1.85e-1 |
| refcv8 final (48,852; cosine projected, ESTIMATED) | 1.16e-2 |

refcv8's final checkpoint sits in the **same regime** as refcv7's final. At step 5,000 the mass was ~9x larger, and there
averaging moved ADE 0-2 s by -0.074 m and ADE 0-6 s by -0.43 m (step-5,000 batteries above).

**Measure.** delta_avg = refcv8 EMA-final minus refcv8 raw-final. Paired episode-cluster bootstrap over the same windows, inference
seeds 0 and 1, all four families at S2 and S6. Both batteries run anyway under section 1, so this costs nothing extra.

**Committed readings (both directions fixed now):**
* **(A)** delta_avg is not separated, or lies inside the inference-seed floor, on every family at both seeds. Then averaging adds
  nothing at the end of this schedule, and refcv8-EMA versus refcv7 (raw) is **attribution-clean**: any refcv8 win is the design.
* **(B)** delta_avg is separated with EMA better on some family. Then on that family the refcv8-versus-refcv7 comparison is reported
  as **"design + averaging"**. The design-only claim for that family is read on **refcv8 raw-final versus refcv7 raw-final**,
  neither averaged, and only that row may be quoted as a design effect. The bar verdict stays on EMA (section 1).
* **(C)** delta_avg is separated with EMA worse: report it. The bar verdict stays on EMA, and the design-only read is as in (B).

**Also reported, not a bar:** at 15,000, 20,000 and 30,000, the raw-minus-EMA gap beside its LR mass, to show the trend.

**Caveats that travel with every number here:**
* single training seed; training-seed floor unmeasured;
* T1 open loop on logged frames, not driving performance;
* the per-frame nav and the road-type speed label are ego-future derived, so optimistic by construction;
* G0 is FAIL as coded at step 5,000. The wrapper clause is settled (W-R8-1, GPU PASS). One detection term is open, localised to a
  single near-threshold match; the GPU arms are queued.
