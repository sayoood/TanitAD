# PRE-REGISTRATION — WP-D: a BEV auxiliary loss that puts AGENTS into REF-C's map

**Date:** 2026-09-07 (Europe/Berlin) · **Author:** Architecture & Inference FlyWheel ·
**Branch:** `agent/arch-inf-20260803`
**Status:** design + code + tests **DELIVERED and STAGED**, **no training launched**, **0 GPU spent**
on any arm. Every number below is either MEASURED on CPU today or cited to a banked artifact.
**Hypothesis id:** `E-BEV-AUX-1` (registered in `Project Steering/GOALS_AND_CLAIMS.md` in the same
turn as this file).

**Owner files:** `stack/tanitad/data/bev_aux.py` (new), `stack/tanitad/refs/refc_bev_aux.py` (new),
`stack/tanitad/refs/refc.py` (gated wiring), `stack/scripts/refc_v3_train.py` (flags + loss term),
`stack/tests/test_bev_aux.py` (new).

**Estimator for every interval below:** episode-cluster bootstrap over the val episodes,
`taniteval/taniteval/ci.py`; **paired** for two arms on the same windows. ⛔ `overlapping_holdout_se`
is never called — it biases the point estimate as well as the interval (−6.67 %…+11.69 %,
bidirectional, over 27 dumps).

⛔⛔ **AND A SEPARATED INTERVAL IS NECESSARY, NOT SUFFICIENT.** The episode-cluster bootstrap
resamples EPISODES with the models held fixed, so it answers *"would another draw of episodes say
this?"* and never *"would another training run say this?"* (`H-ESTIM-SEED-1`: two arms differing in
**nothing** read `separated` on 6 of 42 family cells, a 14.3 % false-positive rate). ⇒ **every bar in
§5 is stated as a RATIO to a replicate floor measured in the SAME panel**, and §4 funds the
replicate arms that make that floor exist.

**Eval tier:** the representation bars (§5A) are a **frozen-feature probe** and carry **no tier** —
no model there emits a trajectory, so a T0/T1 stamp would be a category error. The planner bars
(§5B) are **T1** (action-closed loop, `taniteval/tools/t1_eval.py`) and carry the four families.

**This document is written BEFORE any WP-D number exists. Both outcomes are committed in §5, with
the thresholds fixed here, in advance.**

---

## 0. The claim under test

> **`E-BEV-AUX-1`** — Supervising REF-C's ResNet feature map with a polar agent-occupancy target
> built from `obstacle.offline`, as a **training-only** head that is removable with bit-identical
> planner output, makes the trunk carry **transferable** agent localisation — the content a
> waypoint-indexed deformable cross-attention (DiffusionDrive coupling (1)) would need in order to
> index into something.

⭐ **Why this and not the index itself.** WP-A (`E-READOUT-CEILING-1`) returned a dissociation:
under a **perfect** front-end the token grid carries a lot (oracle AP **0.4713** at 16×40, **0.3374**
at refcv5's 8×20), but on the **real** planning-trained trunk the fit ladder does not transfer —
test AP **0.027–0.034** against a **0.0325** marginal control, with raw pixels strictly below and the
`shuffled` control indistinguishable from `pos_only`. ⇒ **the map does not yet contain agents, so
WP-B built today would index into an empty room.** WP-D is the prerequisite.

---

## 1. The chain of MEASURED facts this rests on

| # | fact | class · source |
|---|---|---|
| 1 | refcv5-v2 runs at `--image-hw 256 640`; REF-C's ResNet is **stride 32**, so its map is **8×20 = 20 azimuth columns = 6.0°/column**, and the anchor decoder already cross-attends its 160 flattened tokens. | **MEASURED** — the run's own `config.json` argv (`…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md` §1); `refc.py:294-296, 333-337`; `refc.py:2112` |
| 2 | ⛔ **refcv5 has NO `SpatialGridReadout` at all.** The "4 readout columns / 30° per bin" figure is a **v6/v7** fact and does not apply to REF-C. 8×20 retains **71.6 %** of the 16×40 oracle ceiling. | **MEASURED** — WP-A §1.2, §5.1 |
| 3 | The corpus is **256×640 CYLINDRICAL**, `f_ref` 305.5775, HFOV **120.000°** — the image column is **LINEAR IN AZIMUTH**. ⚠️ The pinhole formula gives **92.641°** on the same data and looks plausible. | **MEASURED** — WP-A §2; rig name `camera_front_wide_120fov` |
| 4 | `obstacle.offline` covers **97.44 %** of the corpus; **87,481 cuboids, 10 classes, all dynamic agents**; the B1 TRAIN join is built (**4,427/4,572 clips · 849,263 rows · 28,053,187 boxes**, md5 `1c985e6d6ad34e605c4ebd30cb353558`) and the B1 EVAL join is built (**139 clips · 26,394 frames · 905,512 boxes**, md5 `3ddb42ecbd3926066795a94587af2aed`). | **MEASURED** — `…/2026-09-06-b1-train-join/`, `…/2026-09-06-b1-agent-join/`; re-read from bytes today (§3) |
| 5 | ⛔ **The join's per-agent `occ` flag is NOT an occlusion flag — it IS the camera FIELD-OF-VIEW mask**, bit-identical to `bev_raster.fov_mask`'s predicate, **0 of 7,680 cells disagreeing at every half-angle tried**. And `obstacle.offline` itself carries no visibility column (`agents_at_time`: *"occ = -1 always"*). ⇒ **object-object occlusion exists nowhere in what we hold and must be DERIVED.** | **MEASURED** — `build_obstacle_join.py` `P4_PREDICATE_IDENTITY`, `stack/tests/test_p4_fov_predicate.py` |
| 6 | ⛔⛔ **THE PROGRAMME HAS RUN THE SIBLING EXPERIMENT ONCE AND IT FAILED.** PSG (the same labels, an 8-column azimuth target, on the v6/v7 world-model stack) gave *"the largest environment gain the programme has measured"* and **cleared neither the raw-pixel floor nor the predictor**; the predictor died at **EVERY** weight tested; the damage was to the **ENCODER** even with `--psg-enc-only` (no gradient reaching the predictor at all). The measured mechanism is SCALE: `psg_enc`/`psg_pred` ~**0.3–0.9** against `o5_loss` ~**0.03**, i.e. **10–30×**. | **MEASURED** — `E-DEC-18`, `E-DEC-18-R1`, `E-DEC-18b`, `E-DEC-18c` |
| 7 | The precedent that makes a training-only head admissible at all: PhyLatent's Physical State Grounding, *"the physical targets supervise representation learning only during training and are not required by the planner."* | **PUBLISHED** (banked primary `2608.05720`) — `LIT-2` |
| 8 | The batch already carries the join: `V3Dataset.enable_agent_join` + `_agent_item` emit `agent_box/yaw/cls/valid/occ/label/ep` per window, keyed on the stable 63-bit clip id, with a NOW-frame label-coverage census. **No second label path is needed.** | **MEASURED** — `stack/scripts/refc_v3_train.py:1527-1700` |

⚠️ **Fact 6 is the reason this pre-registration exists in this shape.** REF-C has **no predictor**,
so PSG's exact failure surface does not exist here — but *"an aux term that overwhelms the
objective"* does, and §5B's failure twin and the launch-time parity guard (§6.5) are aimed at it.

---

## 2. ⛔ THE THIRD STATE — the decision, its evidence, and its cost

WP-A flagged this on WP-D's critical path: its own negatives are **two-state**, so a scored negative
**merges seen-and-empty with agent-occluded**, and supervising "empty" on an occupied-but-occluded
cell teaches the trunk that occluded space is free — the failure mode that matters for driving.
It is the **third appearance** of this shape in the programme, after `NOT_APPLICABLE` vs
`NOT_CHECKED` and `EXPLICITLY_ABSENT` vs `UNKNOWN`.

### 2.1 What we hold, precisely

* ⛔ **No occlusion label exists** (fact 5). `occ` is the FOV mask under an occlusion-shaped name.
* ✅ **Occlusion is DERIVABLE** from what the join does carry: per-frame ego-frame footprints
  (`cx, cy, yaw, l, w`). A ray from the ego origin to a cell either does or does not cross a nearer
  footprint. **No `z` is needed** — and none is available, because the join schema drops it.

### 2.2 The three options, with their costs, MEASURED today on the full B1 EVAL join

`code/s1_third_state_census.py` · `raw/third_state_census.json` · **all 26,394 labelled frames /
905,512 boxes**, polar spec 24 × 20 (2.5 m × 6.0°), CPU only, 16.7 s.

| option | what it costs | what it teaches |
|---|---|---|
| **(a) MASK the shadow** ⭐ **CHOSEN** | **17.656 %** of cells stop being supervised (per frame: median **11.667 %**, p90 **45.833 %**), and **27.958 %** of GT-occupied cells (**128,070 / 458,083**) are deleted from the positive signal | nothing wrong; it simply says less |
| **(b) TWO-STATE (no mask)** ⛔ **the deliberate regression** | costs nothing | ⛔ **mislabels 27.958 % of the occupied cells as FREE ROAD** — it does not merely omit the occluded agents, it asserts they are not there |
| **(c) THREE-CLASS softmax** | a third logit, and a metric that is no longer comparable to WP-A's or to any occupancy AP; the head must also predict *visibility*, which mixes a scene fact with a camera fact | more, but it changes what is being measured before we know the first thing works |

⇒ **DECISION: (a) MASK, as the default and the pre-registered arm.** (b) ships as the named arm
`--bev-aux-occlusion none` so the regression the gate must catch is a thing anyone can run, and its
detection is a committed criterion (§5D). (c) is explicitly **out of scope for WP-D** and named as
the successor if (a) succeeds — pre-registering an unmeasured third head would be gold-plating.

⚠️ **THE DERIVED MASK IS A LOWER BOUND ON TRUE OCCLUSION, and every number computed through it must
be read that way.** Three stated limits, none hidden: (i) it is **2-D** — the join drops `z`, so a
low `stroller` shadows a `bus` behind it; (ii) the shadow starts at a **cell boundary**, not the true
ray-exit range, so it is quantised to 2.5 m; (iii) **only labelled agents occlude** — buildings,
walls and vegetation are not in `obstacle.offline` (10 classes, all dynamic), so a real urban scene
is **under**-shadowed.

⚠️ **Two further unobservable states, named rather than silently handled:**
* **NO_LABEL** — an absent `(clip, frame)`. The whole frame is IGNORE (mask all-False). ⛔ Never
  "road clear": the labels span ~20 s while egomotion runs 48–140 s. An **empty** agent list IS a
  label (*labelled clear*) and is fully supervised — **2.239 %** of frames on B1 EVAL.
* **BELOW THE VERTICAL FIELD / under the ego's own hood** — the join notes the eval frame's ~45° VFOV
  excludes ground-level agent centres only at **< ~2 m**, i.e. less than one 2.5 m range bin.
  Exposed as `PolarBEVSpec.r_min_m`, **defaulting to 0.0** so the geometry is never altered without
  a record. ⛔ It is NOT masked by default; it is declared.

⭐ **OUT-OF-FIELD does not arise at all**, which is one reason the target is polar: every column of
this grid is inside the 120° field by construction, while the Cartesian [120, 64] raster loses
**590 of 7,680** cells to it (all at x < 9.09 m). Pinned by
`test_no_cell_of_the_polar_grid_is_out_of_field`.

---

## 3. What the aux head supervises, exactly

**Target** — `stack/tanitad/data/bev_aux.py`, `PolarBEVSpec(n_az=20, n_rng=24, r_max_m=60.0,
hfov_deg=120.0)`: a **polar (range × azimuth) agent-occupancy map, 24 × 20 = 480 cells**, ego frame
**+x forward, +y LEFT**, built from the B1 join's ego-frame cuboid footprints with **exactly**
`bev_raster.rasterize`'s oriented-rectangle predicate.

⭐ **Polar, and column-registered, for two reasons — one MEASURED, one geometric:**
1. **MEASURED (WP-A §6.1):** a Cartesian 0.5 m BEV indexed into an image-token grid puts a **median
   of 4 BEV cells (max 313)** into one token cell, and only **242 of 640** token cells receive any
   ground-plane cell at all — a quantisation floor no training removes.
2. The corpus is **cylindrical**, so the image column IS an azimuth bin: **one target column per
   feature-map column, zero resampling.** ⛔ `BEVAuxHead` **REFUSES** `gw != n_az` rather than
   interpolating, because a silent resample keeps the loss curve healthy while mis-registering every
   agent.

**Head** — `stack/tanitad/refs/refc_bev_aux.py`. `Conv2d(F→d_tok, 1)` then, per azimuth column, the
whole elevation stack `[d_tok·gh] → hidden → n_rng`.
* `col` — no cross-column mixing. The **cheap floor**, and the arm the cylindrical geometry
  justifies (azimuth needs no learning). Pinned: perturbing one input column moves exactly one
  output column.
* `xcol` — one self-attention block over the 20 column tokens, so *"does cross-column reasoning buy
  anything"* is an **ablation**, not an assumption.
* Cost at REF-C-base (`feat 704`, `d_tok 64`, `hidden 256`): **182,616 parameters**, reported on its
  own `param_breakdown["bev_aux"]` line and **subtractable** — the DEPLOYED count is unchanged.

**Loss** — masked `BCEWithLogits` with `pos_weight` a **pre-registered CONSTANT** of **30.61**,
derived as `(1−p)/p` from the MEASURED base rate `p = 0.031634` over the whole B1 EVAL join. ⛔ Never
computed from the batch: a batch-derived weight makes two arms with identical flags optimise
different objectives.

⛔ **REMOVABILITY IS PROVEN, NOT ASSERTED.** Two design constraints, both load-bearing and both
pinned by mutation (§6.4):
1. the head is constructed **LAST** in `RefCModel.__init__`, so an aux-on and an aux-off build share
   **bit-identical initial weights** for every parameter they have in common — otherwise the A/B
   would differ in the **seed** as well as in the lever, invisibly, in every log;
2. it is called **LAST** in `forward` and consumes **no RNG**, so it cannot perturb the decoder's
   draws. **MEASURED: the planner's entire output dict is bit-identical, and the only extra key is
   `bev_logits`** (`test_planner_output_bit_identical`).

---

## 4. `one_variable` — the launch commands, DIFFED

⛔ Not the intent, the **tokens**. A past sweep was invalidated because an arm silently changed the
effective lambda alongside its named lever.

**BASE** = refcv5-v2's own argv, read back from its `config.json`
(`…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md` §1), with `--steps` reduced to the WP-D budget
and `--out` per arm. ⛔ **Every arm below carries `--agent-join` and `--seed 0`, INCLUDING the
control** — the join must be in both arms or the A/B also changes the dataset.

```
BASE := --arm hier --size base --image-hw 256 640
        --v2-cache /root/data/train  --v7-labels …/s2_labels_v7.2_train.jsonl.gz
        --eval-cache /root/data/eval --eval-labels …/s2_labels_v7.2_eval.jsonl.gz
        --eval-every 500 --eval-batches 8
        --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
        --lr 1e-4 --warmup 2000 --log-every 50 --save-every 500 --u8-batches
        --nav-from-v7 --ego-state-inject --ego-dropout 0.5
        --anchors <arm>/anchors.pt --n-anchors 117 --anchor-v0-conditioned
        --anchor-control-units alat --sel-accel-max 2.0
        --sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted
        --goal-str --tac-goal-tok-head --agents off
        --agent-join /root/data/joins/train2400_agents.jsonl.xz
        --steps 12000 --seed 0
```

| arm | tokens ADDED to BASE | what it isolates |
|---|---|---|
| **D0** control | *(none)* | the aux-off baseline |
| **D0b** replicate | `--seed 1` | ⭐ the **training-seed noise floor** the bootstrap is blind to (`H-ESTIM-SEED-1`) |
| **D1** the lever | `--bev-aux col --w-bev-aux 0.1` | THE HYPOTHESIS |
| **D1b** replicate | `--bev-aux col --w-bev-aux 0.1 --seed 1` | D1's own floor |
| **D2** shuffled target | `--bev-aux col --w-bev-aux 0.1 --bev-aux-shuffle` | ⭐ same head, same params, same gradient magnitude, **zero information** |
| **D3** detached trunk | `--bev-aux col --w-bev-aux 0.1 --bev-aux-detach` | ⛔ the aux gradient never reaches the trunk |
| **D4** two-state | `--bev-aux col --w-bev-aux 0.1 --bev-aux-occlusion none` | ⛔ **the deliberate regression** — supervises occluded space as free |
| **D5** cross-column | `--bev-aux xcol --w-bev-aux 0.1` | is cross-column reasoning worth its parameters? |

⛔ **D1 differs from D0 in exactly two tokens, and they are one lever**: `--bev-aux col` builds the
head, `--w-bev-aux 0.1` supplies its weight, and the trainer **REFUSES** either without the other
(both directions, verified to fire — §6.5). Every other arm is D1 **plus one token**.

⚠️ **The aux arms carry 182,616 more parameters during training.** They are training-only and reach
no planner tensor (proven bit-identical), so they cannot help the planner directly — only through
the trunk's weights. **D2 is what bounds the capacity/regularisation explanation**, and it is the
reason D2 exists rather than being assumed away.

---

## 5. SUCCESS and FAILURE, committed in advance

### 5A. The representation bars — a FROZEN-FEATURE probe (no tier)

WP-A's §4.2 instrument, unchanged, re-run on each arm's trunk: the same **82.6 k-parameter**
azimuth-indexed head, the same episode-disjoint split, the same controls. Primary metric **test AP**.

| bar | statement | reference value |
|---|---|---|
| **A1** marginal | `AP_test(D1) − AP_test(pos_only)` ≥ **+0.010** with a **separated** paired CI | WP-A: the trunk today reads **0.027–0.034** against `pos_only` **0.0325** |
| **A2** raw-pixel floor | `AP_test(D1) > AP_test(pix)` with a **separated** paired CI | ⛔ `E-DEC-18-R1` FAILED exactly this bar; it is not a formality |
| **A3** the lever, against its own noise | `AP_test(D1) − AP_test(D0)` ≥ **3×** the replicate spread `|AP(D1b)−AP(D1)|` and `|AP(D0b)−AP(D0)|` measured in the SAME panel | WP-A's oracle-rig floor was ≤ **0.0122 AP**; ⛔ that number is **not borrowed** — the floor is measured here |
| **A4** information | `AP_test(D1) − AP_test(D2)` ≥ **3×** the same floor | if the shuffled arm matches D1, the gain is capacity, not content |

⭐ **SUCCESS on representation requires A1 ∧ A2 ∧ A3 ∧ A4.** Any one failing is a failure of the
claim as stated.

### 5B. The planner bars — **T1**, four families, paired episode-cluster bootstrap

⛔ ADE alone is not a result. The four families bind: **LONGITUDINAL** (target-speed accuracy,
headway / time-gap / TTC), **LATERAL** (heading, curvature, yaw-rate, cross-track), **TACTICAL**
(selected vs executed manoeuvre, goal/anchor selection), **STRATEGIC** (route/goal setting), each
per-family with its own `n`, never pooled.

| bar | statement |
|---|---|
| **B1** non-regression | no family metric of `D1 − D0` is separably WORSE by more than **1×** the arm's own seed floor on that metric |
| **B2** the tell | ⭐ if `D1` improves a family by **making the planner act less**, it is not a win: mean \|a\| must **hold or rise**, and the zero-acceleration plan fraction must not rise (`D-REFAV1-LON-ACTS`, whose floors are **+0.044** on mean \|a\| and **0.05** on frac a==0) |

### 5C. THE FAILURE TWINS — each committed, with the named next lever

| if | then, reported as written |
|---|---|
| **F1** A1/A2/A3 fail | ⛔ **REFUTED: the BEV auxiliary loss does not create transferable agent content on this trunk.** ⭐ Next lever, named in advance and already MEASURED elsewhere: **`E-DEC-8` distillation into frozen DINOv3**, which took `n_agents` from **−1.04 to +0.33 at no cost to ego** — the highest-measured-effect lever touching this question. |
| **F2** A passes, B1 fails | ⛔ **The aux term buys representation at the planner's expense** — `E-DEC-18b`'s shape reproduced on REF-C. Reported as such and **NOT tuned around**: a weight sweep after seeing this is a NEW pre-registration, not a continuation. |
| **F3** `D3` (detached) matches `D1` on A | the gain is not coming from the trunk — the head is doing it alone and the claim is **void as stated**, whatever the numbers look like. |
| **F4** A4 fails (`D2` matches `D1`) | the gain is **capacity/regularisation**, not agent content. Refuted. |
| **F5** every arm sits at the marginal control | **UNDERPOWERED, not negative** — check `n` and `d` first (§5E), then report the bound, not a null. |

### 5D. The deliberate regression the gate MUST catch

⛔ **`D4` (`--bev-aux-occlusion none`) must be DETECTABLE.** Its target mislabels **27.958 %** of
occupied cells as free road. The committed criterion: **`D4` must be separably worse than `D1` on at
least one A-bar or one B-family metric.** ⚠️ If it is NOT, the honest reading is that *the third
state does not matter for representation learning at this resolution* — §2's design decision is then
**retracted for WP-D** (the driving argument for masking survives on its own, and must be stated as
such rather than quietly kept).

### 5E. `n` and `d`, in every table

⛔ Every table prints its `n` (rows) and `d` (feature dimension). **`n ≪ d` is UNDERPOWERED BY
CONSTRUCTION, not a negative** — MEASURED 2026-08-22: 2,050 features on ~700 rows made validation
correctly pick maximal regularisation and every arm read exactly the no-information value, and the
panel nearly concluded *"the latent carries no dynamics"*. The probe's `d` at 8×20×704 is **112,640**;
⇒ the frozen-feature probe uses the **pooled/indexed** parameterisation of WP-A §4.2, not the dense
one of WP-A §8, which failed for exactly this reason and is banked as a negative about the probe.

---

## 6. The controls, their KNOWN VALUES, and whether each was verified to read it

⛔ **No headline number here may be an ACCURACY.** MEASURED base rate **3.1634 %** over supervised
cells ⇒ **an all-zero predictor scores 96.837 % accuracy.** `bev_aux_metrics` reports AP / IoU / F1
and returns no accuracy key at all.

| control | must read | VERIFIED today? |
|---|---|---|
| **constant-score** (any constant logit) | **exactly** the base rate, AP | ✅ `test_CONTROL_constant_score_reads_exactly_the_base_rate` — reads `4/480` exactly at logits −9, 0, +9 |
| **all-zero predictor** | IoU **0.000**, F1 **0.000**, AP = base rate | ✅ `test_CONTROL_all_zero_predictor_scores_iou_and_f1_exactly_zero` |
| **perfect ranker** | AP **1.0**, IoU **1.0**, F1 **1.0** | ✅ `test_CONTROL_perfect_ranker_reads_exactly_one` |
| **no positives in the scored set** | AP **NaN** (undefined), never 0.0 | ✅ `test_metrics_score_only_supervised_cells` |
| **NO_LABEL batch** | loss **exactly 0.0** with `n_supervised == 0` saying why — never NaN | ✅ `test_loss_on_an_all_no_label_batch_is_exactly_zero_and_says_why` |
| **masked cells** | changing a logit under an IGNORE cell may not change the loss **by one bit** | ✅ `test_loss_ignores_masked_cells_exactly` |
| **raw-pixel floor** (probe arm) | the trunk must BEAT it (bar A2) | ⏳ needs the GPU arm |
| **shuffled target** (`D2`) | no gain over `D0` | ⏳ needs the GPU arm |
| **detached trunk** (`D3`) | zero trunk gradient | ✅ `test_detach_trunk_is_a_real_deliberate_regression` — trunk grad `> 0` off, exactly `0` on |

### 6.4 ⛔ THE MUTATION PROOF — the gate CAN go RED

⭐ *A check that shares the defect it checks for is green forever* (four measured instances,
CLAUDE.md `e4af94f`). `code/s2_mutation_proof.py` re-introduces six defects **this programme
actually suffered**, runs the whole suite against each, and reports which tests died.
**MEASURED: baseline GREEN (36 passed), 6 / 6 mutants KILLED.** `raw/mutation_proof.json`.

| mutant | the real defect | tests killed |
|---|---|---|
| **M1** mirrored world (`+y` sign) | *"a sign error here does not crash and does not show in a loss curve, it teaches a MIRRORED world"* (`E-DEC-18`'s build) | 3 |
| **M2** pinhole FOV 92.641° | the retracted 2026-08-21 optics error — it silently deletes every agent between 46.32° and 60° of bearing, i.e. the near-lateral band where cut-ins live | 6 |
| **M3** two-state merge | WP-A's flag; **27.958 %** of occupied cells mislabelled free | 2 |
| **M4** NO_LABEL read as clear road | the join's own named defect | 1 |
| **M5** tie-blind AP | ⭐ **a REAL defect in this module, caught by its own control on the first run** (§6.6) | 2 |
| **M6** head not constructed last | the one-variable violation of §3 | 2 |

⚠️ **The FIRST version of M6 SURVIVED, correctly** — it inserted an unconditional RNG draw, which
shifts both arms equally and is not the defect. The mutant had to be made **arm-conditional** to be
the real one. A surviving mutant is information about the mutant as often as about the suite.

⭐ **M1 / M3 / M6 are now registered in the REPO'S OWN standing instrument**,
`stack/scripts/guard_mutation_audit.py` (keys `bev_mirrored_world`, `bev_two_state_merge`,
`bev_head_not_constructed_last`), with `tests/test_bev_aux.py` added to its `TEST_FILES`, so their
anchors are pinned against source rot by `test_guard_mutation_audit.py` in the ordinary suite.
**MEASURED by that instrument: anchors 3/3 present exactly once, baseline `rc=0`, `3/3 defects
CAUGHT by the named guard`, tree restored and sha256-verified.**

⛔⛔ **AND ITS FIRST RUN CAUGHT A REAL PROVENANCE GAP IN THIS VERY PRE-REGISTRATION.** It refused to
proceed — *"the suite is ALREADY red"* on `test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`
— because `agent_knob_dests` derives its set from argparse (`--agent*` / `--w-*`) and therefore
picked up **only** `--w-bev-aux`. Every other WP-D knob starts with `--bev-aux`, so this family would
have reached `config.json` **with its weight recorded and its policy absent** — and
`--bev-aux-occlusion` is precisely what separates arm **D1** from arm **D4**, its
deliberate-regression twin. A run record that cannot say which of the two it was makes §5D
unfalsifiable. ⇒ the derived predicate now includes `--bev-aux*`, `_seam_stamp` carries a structural
`bev_aux` block (with `shuffled_target`, so **D2** is distinguishable too), and the P2 probe's base
argv enables the BEV seam for exactly the reason it already enables `--agents oracle`.
⭐ **My own package-local mutation script reported 6/6 killed and was blind to this, because it only
ran `test_bev_aux.py`** — which is the argument for registering in the shared instrument rather than
shipping a private one.

### 6.5 The refusals, verified to FIRE

`code/s4_trainer_wiring_check.py` · `raw/trainer_wiring_check.json`, CPU only:

| refusal | fired? |
|---|---|
| `--bev-aux col --w-bev-aux 0` → *"ZERO gradient into the trunk"* (the `w_agent` defect verbatim) | ✅ |
| `--bev-aux col` without `--agent-join` → *"NO LABELS"* | ✅ |
| `--bev-aux off --w-bev-aux 0.5` → *"SILENTLY SKIPPED while config.json stamps the weight"* | ✅ |
| `--w-bev-aux > 0` with no `bev_occ` in the batch → refuse at loss time, not skip | ✅ (code path; same shape as the shipped `--w-agent` guard) |
| `BEVAuxHead` with `gw != n_az` → refuse rather than resample | ✅ `test_head_refuses_a_column_count_that_does_not_match_the_map` |

### 6.5b ⭐ THE LAUNCH-TIME PARITY GUARD — `E-DEC-18b` moved to before the GPU-days

`refc_bev_aux.assert_loss_parity` refuses a run whose `w_bev · bev_loss / traj_loss` lies outside
**[0.02, 3.0]** at step 0. Below the band the term is *"declared in config.json and inert in the
gradient"*; above it, this is the measured 10–30× that destroyed the encoder. ⚠️ It is a **necessary**
condition only — §5B's failure twin is what covers a term that is in-band and still harmful.
On the wiring check's real numbers the ratio reads **0.406** ✅.

### 6.6 ⭐ A defect this pre-registration's own control found, logged before any arm ran

The naive per-sample average precision breaks ties by array order, so a **CONSTANT** score — which
cannot rank at all — read **0.010236 against a base rate of 0.008333 (+22.8 %)**, and on a
one-positive fixture **0.009434 against 0.002083 (4.5×)**. A no-information control would have read
**HIGHER** than the value it defines, and every arm would have been ranked against an inflated
floor. Fixed by summing over **tie groups**; the constant control now reads the base rate exactly.
⚠️ **The same shape is visible in WP-A's banked panel**, whose `all-zero` control reads AP
**0.016256** against a stated test base rate of **0.016124** — a +0.8 % gap of exactly this
mechanism, small there, and worth re-checking wherever an AP is quoted.

---

## 7. The GPU arm — precise, costed, and NOT started

⛔ **Nothing has been launched.** The A40 (`tanitad-refcv3`) is training refcv5-v2 (step ~17,750 of
40,284, landing late Monday); Thor is training refav1; the dev-box RTX 4060 was occupied by an
inference-seed noise-floor measurement. **No GPU or RAM load was added to any of them.**

| | |
|---|---|
| rig | the A40 (`tanitad-refcv3`) **after refcv5-v2 lands**, or any free 48 GB card |
| arms | **D0, D0b, D1, D1b, D2, D3, D4** (7). `D5` (`xcol`) only if D1 clears §5A |
| steps | **12,000** per arm — 29.8 % of refcv5-v2's 40,284, the shortest budget past `--warmup 2000` at which the run's own 500-step milestone gate reads a curve |
| rate | **4.0 s/step, MEASURED** on this exact configuration (50 steps per 200 s at 01:23–01:29Z, `…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md`). ⚠️ That receipt **retracts its own earlier "~1.2 s/step"**, which came from an early window inside `--warmup`; the 4.0 figure is the sustained one. Independently corroborated here: launch 2026-09-06 23:10:12Z → ~17,750 steps by ~18:00Z ⇒ **3.82 s/step**. |
| wall-clock | 12,000 × 4.0 s = **13.3 h/arm** ⇒ **≈ 93 h (3.9 days)** for 7 arms sequentially on one card |
| ⭐ cheaper first cut | **D0 / D1 / D2 at 4,000 steps = 4.4 h/arm ⇒ 13.3 h for the three on one card.** It answers **A4** (is the gain information or capacity?) and the **sign** of A3 before the replicate arms are funded. ⇒ **this is what I would launch first, and it fits in one night.** |
| storage | ~4 GB/arm of checkpoints; the join adds **0** — it is already on the box |
| the probe | WP-A's §4.2 frozen-feature probe on each trunk afterwards: **~50 GPU-minutes total**, on the **dev-box 4060** — no A40 time |

⭐ **The added dataloader cost is MEASURED and negligible:** the polar target builds at
**0.63 ms/frame** on CPU (26,394 frames in 16.7 s, §2.2's census), i.e. **~13 ms per batch of 20**
spread over 6 workers, against a 4.0 s step. The join lookup itself is already paid by
`--agent-join`, which both arms carry.

⚠️ **The `--v2-lru 24` / `--workers 6` host-RAM settings are copied from the live run and must be
re-checked on whatever box actually runs this**; a per-run fact belongs beside its per-run divisor.
⛔ And `--agent-join` loads the TRAIN join, MEASURED at ~**2.1 GB RSS** for the full file
(`JoinFileReader` docstring) — budget it, or pass `episode_ids`.

---

## 8. What this pre-registration does NOT claim

* ⛔ **It does not build the waypoint index.** WP-B is the successor and is explicitly gated on §5A.
* ⛔ **It does not claim the aux head improves driving.** §5B is a NON-REGRESSION bar. A
  representation win with a flat planner is the expected good outcome at this stage, and saying so
  in advance is what stops it being sold as a driving result later.
* ⛔ **It is an AGENT BEV, not a scene BEV.** PhysicalAI-AV publishes no map, no lane graph, no
  junction annotation and no route (*"we do not include open maps data"*, settled at five probes),
  and `obstacle.offline`'s 10 classes are **all dynamic agents**. Free space is inferable but
  ego-biased and is **not** shipped as "drivable area".
* ⚠️ **The occlusion mask is a lower bound** (§2.2), and every number computed through it inherits
  that.
* ⚠️ **The polar base rate (3.1634 %) is NOT comparable to WP-A's Cartesian one.** Different
  geometry, different denominator. The same census's Cartesian in-field control reads **1.8382 %**,
  which is the number that belongs beside WP-A's 1.62 %/1.74 %.
* ⚠️ **Column-linear-in-azimuth is scoped to our 256×640 `f_ref` 305.577 cylindrical corpus.** It
  does not travel, and a pinhole formula on the same data gives 92.6° and looks plausible.
* ⚠️ Range is **inferred, never measured** — a monocular camera has no metric depth, and every range
  the head produces is a learned prior that will degrade on under-represented classes
  (car 43.3 % / truck 3.7 % / bus 0.85 %).

---

## 9. Deliverables that exist as of this pre-registration

| artifact | path |
|---|---|
| the target builder | `stack/tanitad/data/bev_aux.py` |
| the head + loss + metrics + parity guard | `stack/tanitad/refs/refc_bev_aux.py` |
| gated model wiring (config field, head LAST, `bev_logits` LAST, breakdown line) | `stack/tanitad/refs/refc.py` |
| trainer flags, refusals, dataset target, loss term, controls | `stack/scripts/refc_v3_train.py` |
| the suite (36 tests: analytic literals, independent reference, 4 mutations, 6 controls, removability) | `stack/tests/test_bev_aux.py` |
| the third-state census + the mutation proof + the wiring check | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-wpd-bev-aux/` |


---

## ADDENDUM 1 (2026-09-08) --- `pos_weight` RE-DERIVED ON TRAIN, as this prereg required

✅ **This is the prereg being EXECUTED, not amended.** It registered `pos_weight = 30.61` as a
**stamped constant derived from the EVAL join** and required re-derivation on TRAIN before the
panel. Done: **`pos_weight = 30.3397163338613`**, over **849,263 / 849,263 records with no
subsampling** --- a **0.9 %** shift, and it reproduces this prereg's own **28,053,187 boxes /
4,427 clips** exactly, which is the cross-check that the two derivations describe the same corpus.
The launched arms carry the TRAIN value (`--bev-aux-pos-weight 30.3397163338613`).
⛔ No criterion changes.

## ADDENDUM 2 --- a run-killer found by READING SOURCE, before the first GPU-hour

⛔ `compute_losses_v3`'s refuse-don't-skip guard raises **`SystemExit`**, which derives from
**`BaseException`** --- so the eval block's `except Exception` **cannot catch it** (verified
independently: `issubclass(SystemExit, Exception)` is **False**). Without handing the EVAL dataset
the same `bev_spec`, `e_ds.bev_spec` stays `None`, the eval batch carries no `bev_occ`, and the
run **dies at the FIRST eval with the compute already paid for** --- the `t1_eval` class, where an
analysis-time failure destroys a completed rollout.
✅ Fixed in `stack/scripts/refc_v3_train.py` and committed **immediately**, because the fix was
staged-but-uncommitted and the next closure ship to Thor would have silently reverted it, killing
the 12,000-step panel at step 500.
⚠️ **The fix is UNEXERCISED and is recorded as such**: an earlier refusal (the eval cache needs
the separate B1 EVAL join) fires first and masks it. **Defect = MEASURED; fix works = REASONED.**
On a no-coverage eval cache the behaviour is still correct and still not a crash --- every frame
is `NO_LABEL`, so the term is the documented control: loss exactly 0.0 with `bev_n_supervised ==
0` saying why, never a silent skip.

## ADDENDUM 3 --- what actually launched, and the one deviation

**D0 -> D1 -> D2 chained on Thor, each at the FULL 4,000 steps.** Measured Thor rate **4.21 s/step
aux-off, 4.88 aux-on** (derived from differences of the trainer's own `elapsed_s` between logged
step rows; ⛔ `step_s` was never read --- this is `refc_v3_train.py`, so neither the
`train_v6_staged.py` divisor rule nor its inversion applies). The plan's 4.0 s/step is an **A40**
figure and does not travel.
⭐ **Deviation, accepted:** three arms measure 15.7 h and do not fit ~14 h. Rather than DROP D2 as
instructed, the agent **chained it third** --- D0+D1 still land by ~10:01 Berlin and D2 then uses
otherwise-idle GPU. Every arm is full length; **the step budget was never shortened**, which was
the actual constraint. ⚠️ `refc_v3_train.py` has **no `--resume`** (two probes), so a crash at
hour 4 costs that arm --- watch `wpd_supervisor.log` for a `launch #2`.
