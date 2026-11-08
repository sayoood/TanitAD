# Master Mind ruling on SCORER1 Stage 0's inputs (2026-11-08, after 20:12Z)

**Written before Stage 0 ran.** At 20:12Z the chain was waiting for the GPU lock; no logit dump, no statistic and no `stage0.json` exists. No bar, estimator or reading changes.

| deviation | ruling | reason |
|---|---|---|
| Sets come from `refe_heldout_v5.tar.zst` instead of `refe_navtrain/onpolicy/sets` | **Not a deviation from the pre-registration.** The error was in `LAND_PLAN.md` step B. | `PREREG_SCORER1` section 1 names `refe_heldout/sets` (24 logs). The on-policy sets are a different fan (ckpt_step 3.7k to 4.3k against 10075). |
| PDM and lane labels come from `Sayood/tanitad-refe-s1aux` `labels/` instead of `pdm_labels_sft4r` | **Accepted, disclosed in the result.** | `LABEL_SOURCE.md`, MEASURED: sets, targets and PDM join 3,137 of 3,137 with 0 misses, and teacher-lane is bit-identical to the banked `heldout_teacher_lane_labels.jsonl.xz` by full key. The PDM tar has no independent banked copy: its evidence is its manifest sha plus agreement with a separate labeller on comfort (100 %), DAC (100 %), NC (97.4 %) and DDC (95.5 %). TTC and EP use different definitions there and say nothing. |
| Train side is 58 sets, not a minimal one | **Accepted.** | Preflight needs at least 24. The train side cannot reach the scores: `run_evaluate("step0")` is at `scorer_finetune.py:2403`, the first `o.step()` is at 2437, and the dump runs at lr 0 from the sha-checked u400. |
| The chain runs the trainer's own `--preflight` (lr 1e-4) first | **Accepted.** This is not a training launch. | The trainer requires its `PREFLIGHT_PASS.json`. Nothing from the preflight enters the dump, and no model is produced or kept, so the launch-gate rule (a token per training run) is not engaged. No token or marker is hand-written. |

Stage 0 is read as committed: Z1, Z2 or inconclusive under `PREREG_SCORER1`, and the result names these inputs.
