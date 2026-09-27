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

## 12. Amendment A7 (2026-09-27 ~00:55 Berlin): the map extent the §11.2 rule selected: 100 m ahead × ±30 m

**The census**, MEASURED on Thor. Prereg `…/2026-09-26-map-signal-audit/raw/PREREG_MAP_EXTENT_CENSUS.md` (md5 d4d02505) was written before any world map was read.
- **Coverage:** 4,369/4,369 TRAIN clips, 78,321 frames. A frame counts as "seen" in a ring when ≥ 20 % of its 10 cm samples are not-255.
- **Controls:** re-crops of the old window match the stored `/2` `fine_codes` byte for byte on 24/24 frames.

| ahead (\|y\| < 16 m) | 0–40 m | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 | 100–110 |
|---|---|---|---|---|---|---|---|---|
| share of frames seen | 1.00 | 0.94 | 0.85 | 0.75 | 0.67 | 0.60 | **0.53** | 0.47 ✗ |

| sideways (x < 60 m) | \|y\| 0–10 | 10–20 | 20–30 | 30–40 |
|---|---|---|---|---|
| share of frames seen | 1.00 | 1.00 | **0.9996** | 0.21 ✗ |

**Selected by the rule: x_max = 100 m, y_half = 30 m (100 × 60 m, 3.125× the 60 × 32 m area).**
- The result is robust at 5 %, 20 % and 50 % seen, with narrower rings, and clip-weighted.
- **Why the sides stop at 30 m:** the SAM3 world map only labels cells within R_MAX = 35 m of a camera position (`sam3map_render_v5m.py:33`). Wider ground truth would need a SAM3 re-render, which A6 excludes.

**Consequences (BINDING for the build):**
1. **The 10 cm map grid is 1000 × 600**: x 0–100 m, y ±30 m. The lift runs at 0.25 m over the same extent, 400 × 240. The metrics report bands 0–20 / 20–40 / 40–60 / 60–80 / 80–100 m, and the bars apply in every band. Per the audit's physics caveat, far bands are expected to read near 0 for both models; they are reported, never dropped.
2. **The planner BEV is CROPPED to 60 m × ±16 m** before the 0.5 m pooling, so the planner's input grid and its token geometry (30 × 16 tokens of 2 × 2 m) are unchanged from refcv6. G-DVB checks it.
3. **The `/3` ground truth is re-exported** from the stored world maps on Thor with ANCHORED coordinates: lateral centre `y = −16 + (j_rel + 0.5)·cell`, `j_rel = j − 140` on the 10 cm grid and `j − 28` on the 0.5 m grid. The control is byte identity with `/2` inside the old 60 × 32 window, for both `fine_codes` and `cart_frac`, on every frame. Timed at 7.26 s/clip, about 1.5 h on 6 workers, about 19.6 GB.
4. **Memory:** the 10 cm decoder's saved activations at b16 grow from 6.85 to about 21.4 GB (analytic). **Gradient checkpointing is ON for the 10 cm decoder**, as a declared flag, which takes it to about 1.14 GB. Its s/step cost is MEASURED in the Thor G-LIVE smoke; more than +25 % goes to the PI, which is reserved decision (b).

## 13. Amendment A8 (2026-09-27 ~02:20 Berlin): NEW-2's class weights are sqrt(median-frequency), the per-band bar rule is fixed, and the Watch contract has 40 keys

Registered BEFORE any NEW-2 model number exists. The NEW-2 build record is `…/2026-09-26-refcv7-map-hires/BUILD.md`.

1. **Class weights: sqrt(median-frequency), computed once from TRAIN, frozen, clip 25.**
   - Option (c) (§11) REMOVED the 0.5 m auxiliary head. Under plain median-frequency weights the big classes keep only **2.7 %** of the gradient at convergence (ANALYTIC; the audit's shares reproduced to 1e-17).
   - Drivable is the class the planner's BEV depends on, so that would starve it. BAR-M7-4 (drivable non-regression) would be at risk.
   - sqrt(MF) gives the big classes **10.4 %** (MF with a floor of 0.25 would give 8.6 %) and still up-weights every thin class relative to unweighted CE.
   - The weights script names each option; refcv7 uses `sqrt_mf`, and its sha256 is recorded in config.json.
2. **The per-band bar rule, as §11.2 registered:**
   - BAR-M7-1..3 are evaluated PER BAND (0–20 … 80–100 m). A bar PASSES iff it passes in EVERY band that has ground-truth cells for its class.
   - A band where the class is absent from GT and from every prediction is reported UNDEFINED with n = 0, and never dropped.
   - Far bands that read ~0 for both models are reported as such. A missed band makes the bar FAILED, with no goalpost move.
3. **The Watch contract (G-MAP item 4): 40 keys**, `eval_map_hires_iou_{cls}_{band}`.
   - cls ∈ {nocls, drivable, lane, crosswalk, arrow, edge, hatched, sidewalk}; band ∈ {0_20, 20_40, 40_60, 60_80, 80_100}.
   - Beside them, `eval_map_hires_iouraw_*` (the raw decision rule) and `eval_map_hires_lshare_*` (per-class loss share), 40 each.
   - The refcv7 Watch shows all of them plus a thin-class alarm tile.
4. **Defect found and fixed in the build, recorded:** under `prior_corrected`, a class with weight 0 was decided on EVERY cell (z − log 0 = +∞), in both the torch and the numpy rule. It is now never decided, and red arms pin it.

## 14. Amendment A9 (2026-09-27 ~02:35 Berlin): the box head is REFINED following proven reference heads, and detection is measured

**PI, verbatim (2026-09-27 ~01:15):** *"based on the compariosn with proven refernce heads, refine our bb head and validate it"*. Registered BEFORE any refcv7 box number exists. This amends A1's "box3d unchanged".

**Evidence:**
- the literature comparison, `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-box-head-literature/RESULT.md`: DETR3D, PETR, BEVFormer, StreamPETR and UniAD configs; Efficient DETR; DETR; CityPersons; KITTI; MonoDLE. Primary sources are read online; banking them awaits the PI.
- the box-head audit, `…/2026-09-26-box-head-audit/`.

**MEASURED on refcv6@38k:**
- presence AUROC 0.83;
- at the 0.5 gate, 1,899 confident slots for 509 targets: precision 0.15, recall 0.57, best F1 0.30;
- no visibility information anywhere in the labels; 26 % of targets under 10 % visible on a representative sample, 56 % on the crowded clips;
- 0 detection metrics logged in 38k steps.

The changes apply to BOTH slot heads (box3d and the planner's agent head), which share `AgentSlotDecoder`:

| id | change | evidence (published / measured) |
|---|---|---|
| **R1** | **Sigmoid focal presence loss** (α 0.25, γ 2, weight 2.0, normalised per matched GT), a **focal matching cost**, and a **prior of 0.01**. This REPLACES the BCE with `NO_OBJECT_W` 0.1. The audit's ln 10 tilt correction is NOT also applied. | Every nuScenes camera head read (5/5) uses focal. Under our BCE the 0.5 gate fires at a 9.1 % match belief. |
| **R2** | **Per-layer supervision:** shared heads after each of the 3 decoder layers, re-matched per layer (deep supervision). 0 parameters. | Efficient DETR at 3 layers: removing the per-layer loss costs −11.5 AP (39.8 → 28.3). |
| **R3** | **A camera-visibility supervision set (VIS-1, IGNORE semantics).** One function is used for training targets AND eval scoring, applied after `visible_target_filter`. | 26 % / 56 % of targets under 10 % visible (MEASURED z-buffer). CityPersons and KITTI DontCare never use unlabelable objects as negatives. |
| **R4** | **300 queries** (M17 re-ruled; +51,200 parameters). The observed maximum is 120 targets per window, so the rule of ≥ 2× the maximum holds before VIS-1 is applied. | DETR: 100 queries find every instance only up to ~50. The nuScenes heads use 900. M17's zero-drop rule was already broken at 100. |
| **T1** | **heavy_truck:** its z/height regression term is DROPPED (label defect). No label is invented. | MEASURED: truck bottoms at −1.15 m (LiDAR clip) and −0.63 m pooled; car and person are fine. |

**R3, VIS-1 in detail:**
- **POSITIVE:** vis_frac ≥ 0.30 AND ≥ 100 visible px.
- **IGNORE:** every other real GT object. That covers vis 0.05–0.30, too few pixels, AND vis < 0.05.
  - ⚠️ This deviates from the audit's proposal, which made vis < 0.05 background. An existing object must never be taught as "no object".
  - IGNORE rows are excluded from matching. An unmatched slot whose centre is within 2 m (BEV) of an IGNORE row gets presence weight 0.
  - At eval, a detection greedy-matched to an IGNORE row leaves the PR count (DontCare).
- vis_frac is a per-pixel ray/cuboid z-buffer over the clip's own camera and extrinsics, with every GT cuboid as an occluder. It is pre-computed ONCE for train and eval on Thor, as a sidecar keyed by clip sha12, window and track, with its sha256 recorded in config.json.

**Deferred, and named** so the next arm can pick them up:
- R5, geometry-grounded image-token positions: validate on the ladder first;
- R6, denoising queries;
- R7, a detection warm start: a PI decision;
- R8, class-weight reform (1,299:1 today);
- R9, a planner-gradient detach arm.

**Detection is MEASURED (P0), with pre-registered bars.**
- **In-run logging, every eval:**
  - mAP at 0.5 / 1 / 2 / 4 m BEV centre distance, per class and per band (0–20 / 20–40 / 40–60 m), on VIS-1 positives with IGNORE as DontCare;
  - presence AUROC;
  - confident-slots / VIS-1-positives ratio;
  - P/R at the declared decision rule.
- **BAR-B7-1:** box3d mAP@2 m (VIS-1, pooled classes): refcv7 − refcv6@38k > 0, separated (paired episode-cluster bootstrap). refcv6@38k is re-scored under the same VIS-1 rule.
- **BAR-B7-2:** F1 at the declared decision rule (focal, gate 0.5): refcv7 − refcv6@38k at refcv6's best-F1 gate > 0, separated.
- **G-BOX-OVERFIT (launch prerequisite):** can the head learn to detect on ~16 fixed real TRAIN frames?
  - Literal thresholds, must-fail arms (e.g. image memory zeroed; presence loss off) and controls, pre-registered by the box-head audit BEFORE any run.
  - Its PASS record is bound to the launch commit.
- **G-LIVE presence sanity:** at the end of the Thor smoke the presence logits are not saturated: the fraction of slots with p > 0.5 < 0.5, with the literal set in the gate profile. Every new loss term (focal presence, per-layer aux, VIS-1 masking) is finite and has gradient.

### 14.1 Correction to A9 (2026-09-27 ~02:45 Berlin, before any box number): T1 is WITHDRAWN

A9's **T1** ("heavy_truck: the z/height regression term is dropped, label defect") rested on a finding the box-head audit has now RETRACTED, MEASURED:
- The 639 heavy_truck GT rows on the GT-validation clips sit at a median range of **90–148 m**, far outside the 60 m decode box, so they are not trainer targets. Their absolute bottoms (−0.97 / −0.76 / −0.32 m) are terrain and pitch at that range; a 0.5° pitch alone gives 0.8–1.3 m there.
- Measured as the offset d = bottom − the median bottom of the same frame's cars, persons and riders within 15 m, heavy_truck d = −0.13 m (GT-val), **+0.021 m** (139 eval clips, n 90) and **+0.030 m** (64 train clips). That is the same as automobile (−0.006) and person (+0.020).

⇒ **There is NO truck-specific target change in refcv7.** Re-seating trucks on z = 0 would be wrong, because far trucks sit on a slope their neighbours share. R1–R4 stand unchanged.

## 15. Amendment A10 (2026-09-27 ~03:20 Berlin): G-BOX-OVERFIT registered, reconciled to A9's IGNORE rule; the box checks fixed before any box number

Registered BEFORE any refcv7 box number exists. No G-BOX-OVERFIT harness has run. The box-head builder is still computing the VIS-1 sidecar on Thor. The evidence is landed:
- the literature comparison: 4f02b08;
- the box-head audit, with the prereg: 35e8207.

### 15.1 G-BOX-OVERFIT: the prereg is binding, with ONE reconciliation

**Prereg:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/raw/PREREG_G_BOX_OVERFIT.md`.
- md5 `594c71196cc5bbd527fee40b2cb0e3f1`.
- Frame set `raw/visibility/gbo_frameset.json`, md5 `b291404c36f83c3e397e3b90367e8e7b`.
- Selector `code/gbo_select.py`, md5 `f9f93f92b60a979b581b8f78be2ec6db`.

**Its literals stand as written:**
- 16 TRAIN frames and 113 VIS-1 POSITIVES.
- Batch 4 and N = 2,000 steps.
- AdamW, lr 2e-4 constant, wd 0, seed 0.
- The model is the LAUNCH box path with the LAUNCH loss.

**PASS, at step 2,000, requires ALL of:**
- AP@2 m BEV ≥ 0.90;
- precision ≥ 0.90 AND recall ≥ 0.90 at the declared gate;
- |Σ confident − 113| / 113 ≤ 0.10;
- median centre error ≤ 0.30 m, median |Δl| + |Δw| ≤ 0.30 m, and median |Δz| ≤ 0.15 m;
- greedy-TP class accuracy ≥ 0.90;
- the presence term ≤ 0.25 × its step-0 value.

**Must fail:**
- `memory_zeros`: AP < 0.90;
- `presence_w0`: criteria 2 and 3 fail.

If either of them passes, the result is VOID. Controls C1–C4 must read their stated values.

**Reconciliation with A9 R3 (binding, A9 wins).**
- The prereg labels rows with vis_frac < 0.05 as "DROPPED". Under A9 R3 they are IGNORE, not background. An existing object is never taught as "no object".
- There are 49 such rows on the 16 frames (the DROP column of the prereg's §2 table).
- So control **C3 reads 113 POSITIVE and 77 IGNORE**: 28 in the 0.05–0.30 or < 100 px band, plus 49 with vis_frac < 0.05. The per-frame counts are POS and IGN + DROP of `gbo_frameset.json`.
- The loss mask (presence weight 0 for an unmatched slot within 2 m BEV of an IGNORE row) and the DontCare scoring both use all 77.
- Rows removed by `visible_target_filter` (outside the 120° field or the decode box) stay outside the target set, as in A9 R3.

**Declared gate:** A9's rule, σ(presence logit) ≥ 0.5 on the focal head. The prereg's §4 phrase "the probability the declared objective makes calibrated" is read as exactly this rule. No calibration map is fitted.
- ANALYTIC, from the literature package's `raw/presence_optimum_by_loss.json`: under the focal objective the 0.5 gate corresponds to a match belief of 0.75, so it is a conservative gate.
- On memorised frames the focal optimum at belief 1 is p → 1, so criteria 2 and 3 are reachable.

**Where:** Thor (canonical, where all 16 clips' payloads live), GPU. Its PASS record, the harness JSON with every arm and control, is bound to the launch commit (A9).

### 15.2 Box checks at the launch smoke (G-LIVE), fixed now

- **G-LIVE-PRES** (A9): at the end of the Thor smoke, on BOTH slot heads, the fraction of slots with σ ≥ 0.5 is **< 0.5**.
  - refcv6 red arm: 71–99 of 100 slots per window at its gate.
  - Every new loss term (focal presence, per-layer aux, VIS-1 masking) is finite and has gradient.
- **NOT used: the audit's G-LIVE-COUNT** (Σ calibrated presence / positives in [0.7, 1.4]).
  - It needs a calibrated probability. A9 R1 chose focal WITHOUT a calibration map, per the audit's and the literature's own "never both" rule.
  - It is recorded as NOT APPLICABLE, never as passed.

### 15.3 Eval-time monitors (Watch alarms, not stops) and informative readouts

Keys follow P0 (A9) and `raw/LOGGING_SPEC_BOX.md`.

- **Box confidence ratio, band [0.5, 1.5]** (the audit's G-LIVE-GATE, tightened to the literature's band). Ratio = confident slots (σ ≥ 0.5, detections greedy-matched to IGNORE rows excluded) / VIS-1 positives, per head. Outside the band is a Watch ALARM.
  - refcv6 red arm: 3.40 (box3d) / 3.75 (agent) at 0.5.
  - At its TRAIN P = R gate against VIS-1 positives: 1.77 / 1.55.
- **INFORMATIVE only** (neither replaces the declared rule nor enters a bar):
  - the P = R gate derived on a FIXED TRAIN calibration set (the audit's 256 windows of 64 TRAIN clips), with P/R there;
  - the class-prior-corrected argmax (logit − ln w_c of the stamped class weights) next to the raw argmax. The weights span 1,298:1, and protruding_object is predicted 420 times against 19 GT among matched slots.

### 15.4 The audit's other proposals, and where they go

Each stays DEFERRED with A9's list and carries its MEASURED support for the next arm:
- **Duplicates** (18 % of FPs at 0.5, 25 % at the P = R gate) → R6, denoising queries.
- **Mislocalised 2–5 m** (45 % of FPs; presence carries no localisation quality, Spearman −0.016) → R5 plus a quality-aware presence target.
- **Class tilt** → R8.
- **M17 re-ruled after VIS-1 on TRAIN** → A9's 300 stays. It is ≥ 2× the pre-VIS-1 maximum of 120; VIS-1 only lowers the count.

### 15.5 A10.1 (2026-09-27 ~04:05 Berlin, BEFORE any G-BOX-OVERFIT number): the pre-registered next lever

**Trigger.** The box-head builder found, INFERRED on a toy only, that with surplus queries a small slot decoder's presence stayed at the base-rate optimum for 4k steps, under focal and BCE alike: the slots did not specialise. If that transfers, G-BOX-OVERFIT's criteria 2 (P and R ≥ 0.90 at σ ≥ 0.5) and 3 (count within ±10 %) are the ones at risk.

Per RULE ZERO the next arm is fixed now, before the data:

| if the MAIN arm… | the next arm (same prereg literals, same frames, same 2,000 steps) |
|---|---|
| FAILS criterion 2 or 3 while presence sits near its base rate, i.e. the slots did not specialise | **+R6, denoising queries (training only).** DN-DETR-style for 3-D boxes: noised GT boxes (centre, size, yaw), plus a no-object negative group (contrastive DN), enter as extra queries. They are attention-masked from the matching queries and reconstructed with the box, presence and class losses. Inference is unchanged. PUBLISHED: DN parity at 50 % of the epochs and +1.9 AP (S3); CDN +0.5 AP and fewer duplicates (S4); used by StreamPETR and Sparse4D v3. |
| FAILS criterion 4 (localisation) while 2 and 3 pass | the box-regression path: its weights and the matching-cost balance (the literature package's `raw/matching_cost_balance.json`). Diagnose first; no pre-picked arm. |
| passes | nothing; R6 stays deferred as in A9. |

**Fixed with it:**
- The G-BOX-OVERFIT bars and must-fail arms do NOT change for the next arm. A lever that "passes" only with a relaxed bar has not passed.
- The +R6 arm needs its own must-fail pair: `memory_zeros` and `presence_w0`, as in the prereg.
- If +R6 also fails criterion 2 or 3, the next lever is an architecture addition: query selection from a BEV proposal heatmap (DINO-style mixed query selection). That goes to the PI as a named blocker, not a silent build.
- Every early G-BOX-OVERFIT run is NON-BINDING (stamped `binding: false`). Only the run on the launch commit binds.

## 16. Amendment A11 (2026-09-27 ~04:30 Berlin, before any BINDING overfit run): the overfit PASS records bind to a code closure, not a commit sha

**What changes.** A9 (§14) and the prereg bind the G-MAP-OVERFIT and G-BOX-OVERFIT PASS records to the launch COMMIT. They now bind to a **code-closure digest** that must be EQUAL at the launch commit. The literals, bars, must-fail arms and controls are unchanged.

**Why.** The two harnesses' binding arms need roughly 3–4 h of Thor GPU. A commit binding forces all of it to run AFTER the last landing. That landing also carries the launch gate and the Training Watch, neither of which the model imports. A closure binding gives the same guarantee, namely that the code which passed is the code that launches, and it can start as soon as the MODEL code has landed.

**The closure** is recorded by a wrapper (`stack/scripts/closure_run.py`) around the unmodified harness. It holds:
- (path, git blob) for EVERY module in `sys.modules` whose file lies inside the run's tree, recorded from the process at exit, never from a hand list;
- the sha256 of every data-contract file the harness opened: frame set, prereg, spec JSON, class weights, VIS-1 sidecar;
- the harness argv sha256;
- the harness PASS JSON's path and sha256;
- the exit status.

**What the gate enforces on the launch commit.** It recomputes each path's blob and requires:
- the closure digest to be equal;
- the argv to be the same;
- `binding: true`;
- exit status 0.

**Red arms, which must be REFUSED:**
- one blob differs;
- a non-binding (early) record is offered;
- a module imported but not recorded;
- a crashed harness, i.e. a non-zero exit status in the closure.

Early non-binding records stay unusable by construction.
