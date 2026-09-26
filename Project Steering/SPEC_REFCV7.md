# SPEC_REFCV7 — refcv6 done right, plus a residual on a kinematic prior

**Status:** REGISTERED 2026-09-26 ~21:00 Berlin by the Master Mind, before any refcv7 code or number exists.

**Decision (PI, verbatim, 2026-09-26):**
> "go with b, stop refcv6, do all fixes, build refcv7, validate the setup again and retrain. push the last valid refcv6 checkpoint to my hf account … Assure that this is not happening again … If we are starting a training session, we are sure about the correctness of the config"

- **refcv6-r101-s0** was STOPPED at step 38,250 (the last logged training row). Its last checkpoint is step 38,000, md5 `5a2e7222…`, being uploaded PRIVATE to the PI's HF account (in progress at registration).
- This SPEC supersedes nothing in `SPEC_REFCV6_V2.md` except what §1 names.

## 1. What refcv7 is

**Base:** everything in `SPEC_REFCV6_V2.md`, unchanged:
- 416×1024 cylindrical front camera; ResNet-101 trunk;
- nav mandatory; strategic layer off;
- SAM3 map GT for the BEV head; agents and 3-D boxes;
- tactical decoder v6; DDIM over anchors;
- the same corpus, splits and parity.

**Plus the fixes.** Each one is a defect MEASURED on refcv6, and each ships with a guard and a test that goes RED on the defect.

| id | defect on refcv6 | refcv7 |
|---|---|---|
| FIX-1 | F3 per-stage cascade loss never reached the loss (D-REFCV6-F3-WHITELIST) | live since 82c2331, with a refusal if the per-stage outputs are missing |
| FIX-2 | tactical labels ~0.37 s early (D-REFCV6-LABEL-CLOCK) | true per-clip clock since 82c2331, plus guard G3: \|t_trainer − t_true\| ≤ 0.05 s, or the run refuses |
| FIX-3 | trunk C26 bottom-strip equalisation dropped by the `--image-hw` rebuild (D-REFCV6-EQUALIZE-DROPPED) | a declared config field, plus a post-build assert that the trunk equalises exactly what argv says |
| FIX-4 | 2 of 3 declared selection mechanisms not built: nav-compliance term, speed-ceiling filter (D-REFCV6-CONFIG-BUILD) | BUILT as declared, or REMOVED from the config with the PI's word. No declared-but-unbuilt lever may exist. |
| FIX-5 | guard blind spots (the A16 audit Q6): the pretrained-weights check reads only the stem; the sidecar md5 check no-ops without `.meta.json` | a per-stage weight fingerprint; the sidecar check refuses when its meta is absent |

**Plus ONE new modelling change, NEW-1: the plan is a residual on a causal kinematic prior.**
- The prior `P` is the constant-velocity / constant-yaw-rate extrapolation from the measured t0 state (v0, ω0). Measured v0 at t0 is admissible (PI ruling 2026-09-02).
- The planner denoises the residual `Δ`, with anchors defined in residual space; the plan is `P + Δ`.
- The exact prior used by the battery's echo (`ha0_ext`) is to be matched and cited `file:line`.
- **Why this lever:** the largest measured one. SPEC A4 L2 (MEASURED, T1, 4,754 windows): the zero-training blend `w·os + (1−w)·ha` beats the echo by −0.0122 at 5k and **−0.0364 at 30k**, separated at both inference seeds. The fitted w rose to ≈ 0.5. The complementarity also holds for refcv4b and refcv5-v2 (exploratory).
- **Pending:** SPEC A6, the train-fitted blend. It is informative; it does not gate the design.

## 2. The launch gate (BINDING, the PI's "sure about the correctness of the config")

No refcv7 training step runs unless `launch_gate_refcv7.py` has written a PASS token for the EXACT (commit, argv sha256, data-manifest sha256s) being launched. The supervisor refuses to start the trainer without a matching token. A token is never hand-edited.

| check | what must hold | its regression arm (must FAIL the gate) |
|---|---|---|
| G-HYG config hygiene | config dataclasses refuse undeclared attributes at assignment (the FIX-3 mechanism is impossible) | set an undeclared attribute |
| G-DVB declared vs built | every argv lever equals the built model's value: trunk equalise, cascade outputs, selection terms and filters, residual prior, heads ↔ loss weights | drop the FIX-3 field; unwire one selection term |
| G-LIVE loss and gradient liveness | a 30-step smoke on the REAL config and REAL data. Every declared loss term appears with a finite value, every declared-trainable group gets a non-zero gradient, `cascade` is present, and the residual prior is non-zero where v0 > 0 | remove `cascade` from the pass-through (the historical F3 defect) |
| G-CLOCK | the label-time guard (FIX-2) on the train and eval caches | the historical `(t + w − 1)·0.1` mapping |
| G-EVAL | the eval loader builds the IDENTICAL model: strict 0/0, and bit-identical outputs on a fixed batch vs the trainer's model | a loader that skips `_pin_trainer_cfg` |
| G-CKPT | save → load round-trip bit-identical, incl. optimizer and data position | — |
| G-SUITE | `pytest -q` on a clean tree of the launch commit shows 0 regressions vs the tip baseline, and every guard's mutation arm is RED | — |

⭐ **Every future arm inherits this gate.** A trainer flag without a G-DVB entry is refused by the gate itself.

## 3. Pre-registered bars (T1 four-family battery + NavSim), committed now

- **BAR-R7-1:** `os − echo` (ADE 0–2 s) < 0, separated, at both inference seeds. This is the bar refcv6 failed at 30k (+0.0225).
- **BAR-R7-2:** `os − refcv6@38k` < 0, separated, on the same windows.
- **BAR-R7-3** (non-regression): at 6 s, `os − echo` < 0, separated.
- **BAR-R7-N1:** NavSim navtest PDMS > STOP, with the paired log-cluster interval excluding 0, on the **FULL** split.

Also required:
- all four metric families; TACTICAL on the true clock; STRATEGIC n/a (layer off);
- the inference-seed floor;
- one training seed only (`H-ESTIM-SEED-1`), so any bar within 2× the replicate floor is reported as NOT PROVEN.
- ⛔ A missed bar is reported as FAILED. No goalpost moves after data.

## 4. Training plan

- Thor, from ImageNet init (a clean start: FIX-3 changes the trunk input and NEW-1 the output space).
- b16 × 50,400 steps (the refcv6 budget); checkpoints every 500; snapshots; battery + NavSim at 5k / 15k / 30k / final.
- Launch ONLY after the gate PASSes on the dev box (G-HYG, G-DVB, G-EVAL, G-CKPT, G-SUITE) AND on Thor (G-LIVE on the real data).

## 5. Sequence and owners

1. Fixes + G-DVB guard (the fixes agent) → landed.
2. NEW-1 residual prior (the refcv7 model agent: new modules first; shared-file edits rebased on the tip after 1 lands) → landed.
3. The launch gate + supervisor enforcement (the gate agent, in parallel on separate files) → landed.
4. The full gate on the dev box and a Thor smoke → the report to the PI → launch.

## 6. Amendment A1 (2026-09-26 ~21:35 Berlin): the name, and NEW-2, a map head at 10 cm

Registered BEFORE any NEW-2 code or number exists.

### 6.1 The name

**PI, verbatim (2026-09-26):** *"you can name drivor-t request with an other name"*.

- **refcv7 = this spec (path (b)).**
- The 2026-09-19 DrivoR-T draft carried this file name, landed at `7e9ccdd`, and was displaced by `d0cdbc8`. It now lives in **`Project Steering/SPEC_DRIVORT.md`**, byte-identical below a rename header.
- DrivoR-T's code still uses `refcv7_*` names: `refs/refcv7_heads.py`, `refcv7_oracle.py`, `refcv7_toad.py`, `scripts/refcv7_derive_nav_tau.py`, and the trainer's `--refcv7`, `--w-r7-wta` and `--w-r7-scorer`.
  - It gets a mechanical rename to `drivort_*` after the FIX landings.
  - ⛔ **Until then a refcv7 launch passes none of those flags, and G-DVB lists them as DrivoR-T levers that must be OFF.**

### 6.2 NEW-2: the map head predicts at 10 cm

**PI, verbatim (2026-09-26):** *"Why did we choose 0.5 m cells? The original sam3 maps were very good and fine. Can we increase the resolution to 10 cm?"*

**Evidence:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/RESULT.md`. MEASURED on 137 SAM3 GT files, 2,867 frames:
- Every GT file already carries the map at **10 cm** (`fine_codes`, 600 × 320). The 0.5 m target is exactly its 5 × 5 average: 0 of all compared cells differ. No SAM3 rebuild is needed.
- The 0.5 m grid keeps only **0.9 %** of the non-drivable-edge area and **62 %** of the lane-line area, under the eval's ≥ 0.5 rule.
- refcv6@35k's argmax IoU is **0.009** for lane lines and **0.001** for crosswalks (513 windows, 3 clips).

**What changes:**
1. **Target:** `fine_codes` @ 0.1 m through a reader with `cart_frac`'s identity and time guards. It is a hard-label CE on seen cells.
2. **Features:** a map-only lift at 0.25 m (240 × 128) sampling the **stride-8** trunk map (`fmap_s8`), a small BEV decoder, and 600 × 320 logits.
   - The existing 0.5 m lift, map head, box3d, BEV cross-attention and 30 × 16 BEV tokens are UNCHANGED.
3. **Loss:** median-frequency class weights, computed once from the TRAIN split's 10 cm GT, then frozen and recorded. Nothing is tuned on eval.
4. **Guards:**
   - `fmap_s8` must reach the branch; that is the F3 whitelist class.
   - G-DVB: the head is built at 600 × 320 with the declared weights.
   - G-LIVE: the 10 cm loss is finite, and the stride-8 path and the new decoder get a non-zero gradient.
   - Each check has a regression arm.
5. **Cost:** measured in the Thor G-LIVE smoke as `torch.cuda.max_memory_allocated()` and s/step. **More than +25 % over refcv6's 6.4 s/step goes to the PI before launch.**

**Bars.** All are scored on the eval kit's 137 SAM3 GT clips, every eval window, at 10 cm. The baseline is refcv6@38k's 0.5 m argmax, nearest-upsampled to 10 cm. The estimator is the paired episode-cluster bootstrap over clips.

- **BAR-M7-1:** lane / road line IoU, 0–20 m band: refcv7 − refcv6@38k > 0, separated.
- **BAR-M7-2:** crosswalk IoU, 0–20 m band: refcv7 − refcv6@38k > 0, separated.
- **BAR-M7-3:** non-drivable edge, precision / recall / F1 at 0.2 m tolerance, 0–20 m band: refcv7 − refcv6@38k > 0 on F1, separated.
- **BAR-M7-4 (non-regression):** drivable IoU is not separated WORSE than refcv6@38k in any band (0–20 / 20–40 / 40–60 m).

**Also required:**
- all 8 classes × 3 bands, with n (windows, clips) printed;
- a **positional-prior control** (each cell's train-set majority class) scored on the same windows, which every class bar must also beat;
- §3's single-seed rule applies: a margin within 2× the replicate floor is NOT PROVEN.
- ⛔ A missed bar is reported FAILED.

## 7. Amendment A2 (2026-09-26 ~22:10 Berlin): FIX-4, all three selection mechanisms are ON

**PI, verbatim (2026-09-26):** *"go with your recommendation for E1, assure that the three selection mechanism are on"*.

Declared-vs-built batch 1 builds the three mechanisms refcv6 declared and never built, and wires them OFF by default. For refcv7 all three are **ON**:

| mechanism | flag | refcv7 |
|---|---|---|
| 8×8 tactical prior | `--graft-tac8-prior` (needs `--tac-decoder-v6`) | **ON**. It replaces the undeclared image-only lat3/lon3 prior that refcv6 ran instead (+1,872 params). |
| nav compliance | `--graft-nav-compliance --nav-compliance-tau-rad <τ>` | **ON**. τ is derived ONCE on the full TRAIN split by `taniteval.nav_compliance.derive_tolerance`, recorded with its input sha256s, and never tuned on eval. |
| max-speed ceiling filter | `--speed-ceiling-filter` (needs `--max-speed-input-v6`) | **ON, at inference only.** See below. |

**The ceiling filter acts at inference only.** A `not self.training` guard at the mask (`refc.py:3337`) keeps the oracle ceiling out of training. The ceiling (`speed_max_derivation_v6`) is derived from the ego's future. Without the guard it would shape training through the LAW input, because the selected plan feeds `law_head` (`refc.py:4461`).

Every refcv7 evaluation reports, on the same windows:
- the filter **ON**, the model as configured and the primary reading;
- the filter **OFF**, the sensitivity reading without the oracle.

**Gate consequences (binding):**
- **G-DVB** asserts that all three mechanisms are BUILT and ON for a refcv7 launch. A refcv7 argv missing any one of them FAILS the gate; that is the regression arm.
- **The τ file** must exist, and its sha256 is recorded in `config.json`. `--graft-nav-compliance` without a recorded τ file is refused.
- **G-LIVE** asserts the ceiling mask is inactive in a training step and active in an eval step. Its regression arm activates the mask in training, which must FAIL.

**Interaction with NEW-1.** The nav-compliance term and the ceiling filter act on the ABSOLUTE candidate plans (prior P + residual Δ), never on Δ alone.

## 8. Amendment A3 (2026-09-26 ~22:20 Berlin): the map is 10 cm, and every class must be shown to learn before launch

**PI, verbatim (2026-09-26):** *"We need the 10 cm map, no way to use 50 cm. Did we review the wiring, the architecture, the training and the training signal flow for the map heads to verfiy its trained to extact all sematic classes?"*

1. **NEW-2 is REQUIRED, not optional.** refcv7 does not launch without the 10 cm map head. The map is predicted, supervised, evaluated and reported at 10 cm; nothing at 0.5 m is ever reported as "the map".
   - Whether refcv6's 0.5 m head survives as a declared internal auxiliary, which shapes the planner's 120×64 BEV features, is under audit (`…/2026-09-26-map-signal-audit/`, task 7), and the PI decides.
   - It is built switchable (`--map-lowres {on,off}`).
2. **The finding behind this amendment**, MEASURED on `metrics.jsonl`, 4,621 rows: refcv6 logged map quality for the DRIVABLE class only (`map_iou_drivable` and its fractions). Per-class IoU was never logged, so the thin-class collapse went unseen for 38,000 steps. At step 35,000 the argmax IoU was 0.009 for lane lines, 0.001 for crosswalks, and 0.000 for arrows and edges.
3. **New launch-gate family G-MAP (BINDING)**, each check with a regression arm that must FAIL:
   - **G-LIVE per-class signal:** every class present in the smoke batches (with ≥ M labelled 10 cm cells) has a finite, non-zero loss contribution and a non-zero gradient on its logit channel.
   - **G-DVB logging:** the eval metric list includes per-class 10 cm IoU for all 8 classes × 3 x-bands (0–20 / 20–40 / 40–60 m) plus per-class loss shares, and the Training Watch shows them. Drivable-only logging FAILS.
   - **G-MAP-OVERFIT:** the 10 cm head, trained on a small fixed set of real TRAIN frames, reaches a literal per-class IoU threshold on every class present. The protocol and thresholds are pre-registered by the audit in `raw/PREREG_G_MAP_OVERFIT.md` BEFORE any run. Its PASS record, bound to the launch commit, is a launch prerequisite.
4. **The per-class signal audit** traces the refcv6 path and the NEW-2 design end to end. Its evidence gates any further change to the map head: the loss, the weights, the lift resolution and the auxiliary head.

## 9. Amendment A4 (2026-09-26 ~23:00 Berlin): NEW-2 corrections from the map-signal audit, made before any NEW-2 number exists

Source: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/` (interim), reviewed and accepted by the Master Mind. Everything below is ANALYTIC or from source; no NEW-2 model output has been read.

1. **Decision rule: the argmax is prior-corrected.** Under median-frequency class weights w, the softmax learns q ∝ w·P(c|x). The raw argmax therefore calls a rare class at a small posterior; for lane lines that is about 5 %.
   - NEW-2 declares `decision_rule` (default `prior_corrected`: argmax(z − log w), with the same frozen weights). It logs both rules and scores the bars (BAR-M7-1..4) and G-MAP-OVERFIT on the corrected rule.
   - ANALYTIC per-class ceilings at 0–20 m, raw → corrected: lane 0.51 → 0.91, crosswalk 0.56 → 0.95, arrow 0.38 → 0.89, edge 0.21 → 0.73, hatched 0.48 → 0.92, drivable 0.91 → 0.99.
2. **G-MAP-OVERFIT's must-fail arm is `s8_zeros`** (the stride-8 features zeroed). `s8_detached` stays informative only, because ImageNet stride-8 features can still memorise 16 frames, so it is not a valid must-fail arm.
   - The protocol is `raw/PREREG_G_MAP_OVERFIT.md` + `raw/gmo_spec.json`: 16 TRAIN frames from 4 clips (frameset md5 4eafa03c), 1,000 steps, batch 4, AdamW 1e-3, seed 0.
   - Pass, on pooled IoU at 0–20 m: big classes ≥ 0.85; lane, crosswalk, arrow, edge and hatched ≥ 0.50; ≥ 1,000 scored cells per class, else FAIL.
   - `lane_w0` must fail on lane; `s8_zeros` must fail on all five thin classes; controls C1–C3 must read their known values.
3. **Log names.** Under `--map-hires on`, the 0.5 m auxiliary's keys are `aux05_*`, so no report or Watch can label them "the map". The legacy `map*` names remain only with `--map-hires off`.
4. **Gradient-reach logging must be live** (D-REFCV6-GRAD-REACH-DEAD). It is fixed trainer-wide in declared-vs-built batch 2. G-LIVE asserts the `ga_*` keys for every trainable group, and its regression arm, the old off-by-one, gives zero keys and must FAIL.

## 10. Amendment A5 (2026-09-26 ~23:30 Berlin): NEW-1's prior is `ha0_ext_pose`

NEW-1 builder, MEASURED from source and on the battery's step-30k surface (4,754 windows, 139 eval clips, paired episode-cluster bootstrap). Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior/`.

**§1 said the prior is CV/yaw-rate "matching the battery's echo `ha0_ext`". Those two cannot both hold.** `ha0_ext` is constant ACCELERATION + constant CURVATURE, and it reads the recorded STEER at t0 (`refcv3_arm.py:2125-2128` → `refav1_arm.py:445-446` → `kinematic.py:220-264`).

| mode | reads | prior − echo, ADE 0–2 s | 0–6 s | status for refcv7 |
|---|---|---|---|---|
| `ha0_ext` | poses + recorded steer at t0 | 0 (bit-equal on 4,754/4,754) | 0 | **excluded**: no PI ruling on the steer channel at inference; NavSim has no steer channel, so BAR-R7-N1 could not be scored |
| **`ha0_ext_pose`** | past poses only | **+0.0055 [+0.0036, +0.0076]** | +0.0223 | **CHOSEN**: past-only ego history is already admissible (refcv6 §10.3; v0 at t0, PI 2026-09-02); runs in every harness |
| `cv_yawrate` | past poses only | +0.2293 [+0.1971, +0.2642] | +0.53 | **excluded**: starts 0.23 m behind the bar it must beat |

- **The chosen argv:** `--residual-prior ha0_ext_pose`. It requires `--ego-history`, and G-DVB refuses any other mode for a refcv7 launch.
- **Not a goalpost move.** BAR-R7-1 still compares against the echo `ha0_ext`, with steer. The prior choice rests on how well each PRIOR FUNCTION matches the echo on real poses; no refcv7 model output exists.
- **The anchors are unchanged in bytes.** The same 117 anchors now mean residuals around the prior, and `config.json` records `vocabulary_space`. The best-anchor distance to the driven path, residual − absolute, MEASURED: 0–2 s eval −0.0232 [−0.0293, −0.0171] (better); 0–6 s eval −0.0205 [−0.0670, +0.0253] (tie).
- **Selection sees absolute plans.** The prior is composed upstream of selection, so nav compliance and the ceiling filter read P + Δ (the A2 requirement). Pinned by `test_A2_…` with a red arm that feeds Δ.
- **With `--residual-prior off` the model is bit-identical to refcv6**, by a sha256 over every loss scalar, gradient and planner output.
- A PI ruling on the steer channel at inference would make `ha0_ext` available in a later arm. It is not needed for refcv7.

## 11. Amendment A6 (2026-09-27 ~00:20 Berlin): one high-resolution lift for everything (PI option c), and the map range is MAXIMAL

**PI, verbatim (2026-09-27):** *"do c and assure that the range of the map is maximal and not only 20 m"*.

### 11.1 Architecture: option (c)
- **ONE lift** samples the stride-8 trunk map into a 0.25 m BEV grid, followed by one BEV encoder. It feeds two things:
  - **(i) the map decoder**, which outputs 10 cm logits, the only map;
  - **(ii) a pooled BEV at the planner's grid** (0.5 m), which replaces the stride-16 lift for EVERY consumer: box3d, the planner's BEV cross-attention and the 30 × 16 BEV tokens.
- **Removed:** the stride-16 0.5 m lift and the 0.5 m map head and loss. There is no `aux05_*`, and nothing at 0.5 m is supervised as a map.
- **This changes the planner's input** relative to refcv6. It is declared here, and G-DVB checks that every consumer reads the pooled high-resolution BEV. The regression arm is a consumer still wired to a stride-16 lift.
- **Cost is measured in the Thor G-LIVE smoke.** More than +25 % s/step over refcv6's 6.4 s goes back to the PI, as in §6.2.

### 11.2 The map range is MAXIMAL, fixed by a rule set BEFORE any measurement
1. **Every range band is evaluated and carries a bar**, not only 0–20 m. The bands are 0–20 / 20–40 / 40–60 m, plus every further 20 m band the grid gains.
   - The per-class IoU bars BAR-M7-1..3 apply in EVERY band: refcv7 − refcv6@38k > 0, separated. The positional-prior control must also be beaten in every band.
   - Far bands are reported with their n. A band where refcv6's baseline and refcv7 both read 0 is reported as such, never dropped.
2. **The grid extent is the largest one the SAM3 ground truth supports.** The ground truth is the per-clip 10 cm world map, stored in the corpus as `semantic_maps/worldmap`.
   - **The rule:** x_max (ahead) and y_half (to each side) are the largest 10 m steps at which **at least 50 % of TRAIN frames have seen ground truth** in that ring.
   - **Measured by a coverage census** on Thor, reading the stored world maps. It uses the TRAIN split only; nothing is tuned on eval.
   - **Constraints:** no smaller than today's 60 m × ±16 m, and within the Thor memory and time budget (the +25 % rule). If the budget binds, the PI chooses between range and cost.
   - The census, its per-ring coverage table and the chosen extent are banked BEFORE any refcv7 map number exists.
3. **The ground truth is re-exported** from the stored world maps at the chosen extent, on Thor, as a new schema version (`tanitad.sam3_map_gt/3`). SAM3 is NOT re-run.
   - The 10 cm codes inside the old 60 × 32 m window must be byte-identical to `/2`. That is the control.
4. **The 10 cm head, reader, metrics and pooling take the extent as a declared parameter.** G-HYG and G-DVB check that the built grid equals the declared extent.
