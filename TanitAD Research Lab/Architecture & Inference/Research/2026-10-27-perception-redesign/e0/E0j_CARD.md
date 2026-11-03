# E0j card -- Gram-matrix gradient share on the E1 rig, with the banded map loss (REFCV9_DESIGN §2.2, §4.1 E0j; code item C11)

**Question.** Which `w_map` makes the map term's *projected* share of the update reach 0.05 (arm M2a) and 0.15 (arm M2b) at the trunk's layer2 tap, with box3d still holding ≥ 0.20 at layer2 and at the whole trunk and ≥ 0.25 at the lift? No training: one gradient census at the E1 starting point, then exact arithmetic for every weight.

**Status: BUILT, TESTED (CPU, 11 PASS), SMOKED ON THOR (CPU, 1 window, ALL controls ran and passed: L 3.4e-6 / 2.0e-7, Z exactly 0, T 4.1e-6 -- plumbing only, those values are NOT the reading), Thor GPU chain QUEUED behind agent OVF's overfit records. The numbers are in the result block below once `ZZE0J-BANDED-DONEZZ` appears in `/home/nvidia/r8ns/r9gram/logs/chain.log`; until then this card quotes NO weight.**

## What was built (and where)

| file (this folder) | role |
|---|---|
| `gram_core.py` | pure numpy/torch: `s_i(w) = w_i <g_i, G(w)> / ‖G(w)‖²` from the 3×3 Gram, Gram checks (symmetric, PSD, Cauchy-Schwarz), the pooled Gram and linearity report from the block matrix K, the smallest-`w_map` selector (vectorised grid + bisection), the sum-to-one control |
| `pgrad_gram.py` | **C11**: the GPU job. A COPY in the refcv9 package, not an edit of the refcv8 `pgrad.py` (that tip blob lacks the TF32-off, requires_grad-re-enable and parts fixes of the stranded E: draft and targets refcv7's six-term loss). Builds the E1 rig through the tip's own eval loader, 64 TRAIN-DIAG windows, stores K per parameter group |
| `e0j_read.py` | CPU reader: controls first (VOID ⇒ no weight), shares at w = 1 and over a `w_map` ladder, the selection for 0.05 / 0.15, a batch bootstrap, the card block |
| `run_e0j_thor.sh` | Thor launcher: `smoke` (CPU, niced, no lock) and `chain` (flock-queued; banded stage, CPU read, matched-instrument refcv8 `baseline8` stage, CPU read) |
| `test_e0j.py` | 11 CPU tests: closed-form shares, vector-definition equivalence with an *opposing* term (a negative share), PSD catch, selector vs closed form + infeasible + unreachable, end-to-end linearity on a tiny shared-trunk model run through the job's own procedure, **deliberate-regression arms** (a 0.2 % rescaled direct gradient and a direct backward on a different forward must each FAIL L and VOID the reading; a Z, T or value perturbation must each VOID; a dead map row must VOID), `stage_argv` |

**The model.** `stack/ops/runs.d/r9e1_C0.argv.json` + `--r9-map-banded` + `--map-hires-near-lift-m 10` (⚠ the banded head REFUSES near-lift 20 — `MapHiresConfig.__post_init__`; the brief's "C0 + `--r9-map-banded`" alone would not build; the M1 smoke argv carries 10). Built by `refcv7_loader.build_model` (G-DVB against that argv), weights = refcv8 EMA-48,852 (md5 `b4364029…`) through the rig's own predicate; the 18 `_map_hires.banded.*` keys are NEW (a **fresh seed-0 init**), the source's `_map_hires.refine.cls.*` is dropped; anything else missing or unexpected refuses. Stage `baseline8` is the same rig without the flag (refcv8's 8-way CE, near-lift 20).
**The windows.** pgrad's rule: 139 TRAIN clips (TRAIN-DIAG cache `perc8/cache/train_diag139`) sorted by sha12, 8 evenly spaced windows each (grid 1,112), `numpy default_rng(0)` draw of 64 without replacement, 8 batches of 8. The window list is stored as (sha12, ws) — no raw clip id is written.
**Terms.** The rig's loss is exactly `w_agent·agent + w_box3d·box3d + w_map_hires·map` (`compute_losses_perc_rig`; checked per batch by the value control and on batch 0 by the gradient control T). `w_map` in the design **is `--w-map-hires`** (the banded term's weight `model._w_map_hires`); `--w-map` stays 0 and `--w-map-fp` is refused under the flag. All three configured weights are 1.0 in C0, so the Gram at w = 1 is the configured update.
**Groups** (disjoint; larger groups are sums of Grams; sizes MEASURED by the Thor CPU smoke: `stage_layer2` 1,219,584, whole trunk 58,231,872, `lift` **131,200** (= the design's ESTIMATE, now measured), `map_encoder` 406,912, lift + encoder 538,112): trunk `stem`, `stage_layer1..4`, `fuse`; `lift` = `_map_hires.lift.`; `map_encoder` = `_map_hires.encoder.` (the shared 64-ch BEV encoder); `layer2` = `stage_layer2`; **whole trunk** = all trunk groups; `x_shared_bev_025m` = lift + map_encoder (the refcv7 P-GRAD's name, for comparison).

## Controls (all must PASS or the reading is VOID and the weight falls back to the E1 dose arms)

| control | definition | bar |
|---|---|---|
| **L linearity** | one backward of `4·map + 0.5·box3d + 2·agent` (w2, non-unit on every term) vs the Gram, every batch and pooled, in layer2 / trunk / lift: max \|share diff\| and ‖Σ w2 g − G2‖/‖G2‖ | both ≤ 1e-4 |
| **Z** | `0.0 × box3d` backward reads EXACTLY 0 in every group (batch 0) | == 0.0 |
| **T** | the trainer's own `losses["loss"]` gradient equals Σ w_cfg·g (no hidden term; batch 0) | ≤ 1e-4 |
| **value** | \|loss − Σ w_cfg·L\| / max(1,\|loss\|), every batch | ≤ 1e-6 |
| **S** | Σ_i s_i(w) = 1 at several w | ≤ 1e-9 |
| **Gram** | symmetric, PSD, \|cos\| ≤ 1 for the three decision groups | all |
| **live** | map and box3d reach layer2, trunk and lift; agent reaches layer2 and trunk (the agent head does not read the lift: a STRUCTURAL zero there, as refcv7's P-GRAD banked it) | norm > 0 |

Precision: TF32 off for matmul and cuDNN, fp32 trunk, cuDNN deterministic, one forward per batch with `retain_graph` backwards (the stranded draft MEASURED a 6.2e-4 linearity error with TF32 ON on refcv7; G-SMOKE-L4's 4.2e-4 is the same class — HYPOTHESIS that this was the cause, the L control here is what decides it).

## Disclosed limits

1. **A fresh head.** The share is read at the E1 *starting point*: the banded head is a seed-0 init, not a trained head. As the head trains its gradient into the trunk changes (its loss falls); the share at step 0 is the cleanest read of "the dose E1 starts with", NOT a prediction of the share at step 4,000. The in-run `--grad-share-every` reading (C12) is what monitors it. HYPOTHESIS that step 0 is representative.
2. **Shared encoder.** C2 (the map's own 128-ch BEV encoder) is not landed, so the lift and encoder here are shared by map and box3d. Layer2 and whole-trunk shares transfer more directly than the lift share, which C2 will change.
3. **Scope of "share".** Denominator = the three-term update (the rig has no planner, no tactical term). refcv7's P-GRAD denominators included traj / tac_v6 / rest (22 % at layer2), so its numbers (layer2 map 0.00098, box3d 0.327, agent 0.452) are NOT directly comparable; the matched-instrument `baseline8` reading is the like-for-like "refcv8 at w = 1".
4. **Estimator.** 64 windows, one checkpoint, one head init, one forward seed per batch. The bootstrap resamples the 8 batches (window-draw variance only, 8 units); it is blind to training variance (`H-ESTIM-SEED-1`) and inference variance. Per-batch selections (8 windows each) are listed beside it.
5. **A projected share is not a controller.** It can be negative (a term opposing the update) and is not monotone in `w_map` in general; the selector scans a log grid (1e-3 … 1e4) and bisects the first crossing, and reports monotonicity.
6. **Not a driving claim.** Open-loop perception diagnostic; no plan scored; four metric families N/A (gradient census).

## Result

<!--E0J-RESULT-BEGIN-->
**Controls (read first).** L linearity: max |share diff| 5.64e-07, max rel grad err 2.34e-06 (PASS, <= 1e-4); Z zero-weight max |g| 0.0 (PASS); T hidden-term 5.28e-06 (PASS); value 6.0e-08; S 2.2e-16; Gram sym/PSD PASS; all terms live True.

**Verdict: controls PASS. M2a (layer2 map share >= 0.05): w_map* = 0.5107, box3d floors HELD; M2b (layer2 map share >= 0.15): w_map* = 1.018, box3d floors HELD**

Shares at w = 1 (map 1 / box3d 1 / agent 1; the E1 C0 weights), projected share of the three-term update:

| group | map | box3d | agent |
|---|---|---|---|
| layer2 | +0.1459 | +0.3509 | +0.5032 |
| trunk | +0.0659 | +0.3578 | +0.5763 |
| lift | +0.5009 | +0.4991 | +0.0000 |
| x_shared_bev_025m | +0.6397 | +0.3603 | +0.0000 |

Selected weights (smallest `w_map` with the layer2 map share at target, box3d floors layer2 >= 0.20, trunk >= 0.20, lift >= 0.25):

| arm | target | w_map* | layer2 map / box3d | trunk map / box3d | lift map / box3d | floors | batch-bootstrap w* p05 / p50 / p95 |
|---|---|---|---|---|---|---|---|
| M2a | 0.05 | 0.5107 | +0.0500 / +0.3809 | +0.0145 / +0.3686 | +0.2920 / +0.7080 | held (w limit 2.2908676527677754) | 0.5206441920032264 / 0.6913714700705811 / 1.03626307355101 (found 1.00, floors ok 1.00) |
| M2b | 0.15 | 1.018 | +0.1500 / +0.3495 | +0.0684 / +0.3572 | +0.5069 / +0.4931 | held (w limit 2.2908676527677754) | 1.0055955876622713 / 1.342149849483882 / 2.0291224558824807 (found 1.00, floors ok 0.95) |
<!--E0J-RESULT-END-->
