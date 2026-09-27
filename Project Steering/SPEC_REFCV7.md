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

## 17. Amendment A12 (2026-09-27 ~09:50 Berlin, BEFORE the lever arm's first number): the map's LIFT lever is a 0.1 m near-range lift

**PI, 2026-09-27 ~09:25:** "Build the lift lever". The pre-registered 1,000-step bar is KEPT.

**Evidence** (the early G-MAP-OVERFIT record, NON-BINDING, banked with the NEW-2 package under `raw/gmo_early/`):
- **The harness is valid.** R1 `lane_w0` failed lane and R2 `s8_zeros` failed all five thin classes, both as required. C1–C3 reproduce: drivable 414,196 / 907,276 = 0.456527 exactly, 1.0 on all classes, and bit-identical.
- **MAIN FAILS** at step 1,000: lane 0.417 and edge 0.076 (declared rule); under the raw rule lane 0.490 and edge 0.233, so §9 lever 1 (the decision rule) is excluded. A same-seed replicate reads lane 0.480 and edge 0.096, putting the run-to-run spread on lane at ~0.06.
- **Informative MAIN_long** (3,000 steps): lane crosses 0.50 at step 1,200; **edge never does** (0.430 at 3,000, decelerating).
  - Edge is FOUND but misplaced by ~1 cell: at 3,000, F1 at the 0.2 m tolerance is 0.906 while IoU is 0.430.
- **GT-only grid oracle** on the 16 frames, scored exactly like the harness: a decoder that sees only **0.25 m** class fractions caps **edge at 0.328** (lane 0.770); at 0.5 m, edge reaches 0.022. The 10 cm identity control reads 1.000. **The 0.25 m lift cell is the bottleneck for edges.**
- **Range does not matter:** edge IoU is flat over 0–25 m (0.40–0.52), while the stride-8 lateral footprint grows 5.5×. So stride 4 attacks a range-dependent limit the record does not show. Nothing implicates height mixing, so there is no measured support for a Z = 0 channel.

**The lever: (b), a near-range 0.1 m lift, map-only.**
- **Flag and config:** `--map-hires-near-lift-m` (default 0 = off; the arm and the launch use **20**) → `MapHiresConfig.near_lift_x_m`, a declared field (G-HYG).
- **G-DVB:** the new kind `map_hires_near_lift_m` takes the registry 213 → 214. The flag is refused when `--map-hires` is off, when it is not a multiple of 0.5 m, and when it exceeds x_max.
- **`NearLiftSkip`:**
  - it reads the same stride-8 map and applies a 1×1 projection per height (512 → 32);
  - it samples at the 0.1 m cell centres of x 0–20 m × y ±30 m (200 × 600) at the main lift's 4 heights, summed;
  - it is ADDED to the 10 cm decoder's upsampled input on those rows, inside the decoder's gradient checkpoint;
  - it is **zero-initialised**, so the arm starts as MAIN's function; it is built last, so every other parameter initialises exactly as MAIN's (fingerprint pinned);
  - its geometry is derived from the batch's own 0.25 m geometry and pinned against the exact 0.1 m geometry by a test;
  - reach keys `ga_mh_near` and `ga_mh_near_n` are declared automatically (D3).
- **Cost (ANALYTIC):** fwd +0.9 % FLOPs per sample; saved activations +0 GiB at b16 (1.391 GiB, grad checkpointing on); a transient 0.92 GiB at b16; +65,600 params. Time: ≲ +0.1 s/step at b16 (ESTIMATED). NEW-2's own step time is still UNMEASURED; the G-LIVE smoke reads it against the 8.0 s line.
- **The planner's pooled BEV is untouched.** The shared encoder's input does not change (the A6 seam).

**The lever arm (G-MAP-OVERFIT, same literals).**
- The prereg literals are unchanged: 16 frames (md5 `4eafa03c`), 1,000 steps, batch 4, AdamW 1e-3, seed 0, the bars, the presence floor, §6.4, the 1 ms guard, C1–C3, both rules, the declared rule, the TRAIN sqrt_mf weights.
- **MAIN** is the lever arm, with `near_lift_m 20`.
- **Must fail:**
  - `s8_zeros`, zeros into BOTH lifts: all five thin classes fail (must_fail_all);
  - **`near_zeros`**, the near lift's input zeroed with the 0.25 m path intact, i.e. MAIN plus a constant skip: **edge must fail**;
  - `lane_w0` is kept: lane must fail.
- **Informative:** `s8_detached`.
- **If the arm FAILS while its must-fails hold:** §9's next lever is (3), the decoder.
- **Binding:** A11 applies. Only a run on the launch closure binds; this arm's early run is non-binding.

## 18. Amendment A13 (2026-09-27 ~10:05 Berlin, BEFORE the corrected arm's first number and BEFORE the one-frame ladder reports): G-BOX-OVERFIT's optimiser is the LAUNCH optimiser

**The defect.** The prereg (§3) claims to test "the refcv7 LAUNCH box path, as built and configured for the launch", but its optimiser line contradicts that. The box builder read it STATIC at tip ab1fb45:

| | the harness (prereg §3) | the LAUNCH (refcv6 argv = refcv7 canonical argv) |
|---|---|---|
| optimiser | AdamW, lr **2e-4 constant for every trainable tensor**, trunk included | `--opt dd` → `timm_trunk.param_groups_dd` (`timm_trunk.py:1316`) → AdamW, two groups by name: `core.encoder.*` (trunk) and the rest |
| peak lr | 2e-4 | head **1e-4** (`--lr`); trunk **5e-5** (`--encoder-lr-mult` default 0.5, `refc_v3_train.py:10644`) |
| schedule | constant | 2,000-step linear warm-up, then cosine (`:8644`) |
| weight decay | 0 | 1e-4 (`:10641` default) |
| grad clip | none | `clip_grad_norm_(…, 10.0)` every step (`:9203`) |
| config | MAIN ran on the stride-16 lift | refcv7 canonical: `--map-hires on --bev-source map_hires_pool`, so the box head reads NEW-2's pooled BEV |

The harness's step is 2× the launch's peak head lr and 4× its trunk lr, and up to 40× during the launch's warm-up. The early diagnosis MEASURED the image memory losing its frame-specific variation (37 % → 6 %) within 20 steps. A step size the launch never takes is a candidate cause.

**The correction (a), registered now.** Every G-BOX-OVERFIT arm uses:
- **the launch's optimiser AS BUILT**: the trainer's own `build_optimizer` with the canonical argv's `--opt dd --lr 1e-4`, the default `--encoder-lr-mult 0.5`, the default `--weight-decay 1e-4`, and `clip_grad_norm_` 10.0;
- **held at the launch's PEAK lrs, constant, with no warm-up**: a 2,000-step capacity test inside the launch's warm-up would never leave it (option (b), rejected: it would fail from under-training, which is not what the test measures);
- **on the refcv7 canonical config plus the A9 box flags**: the launch-gate package's `stack/ops/runs.d/refcv7-r101-s0.argv.json` plus R1–R4, with the box head reading NEW-2's pooled BEV. The harness takes the canonical object file (`["argv"]`).

**Unchanged:** every other literal (16 frames, 113 POS / 77 IGNORE, batch 4, 2,000 steps, seed 0, the PASS bars, must-fail `memory_zeros` and `presence_w0`, C1–C4, the declared gate σ ≥ 0.5). The early MAIN / +R6 results under the old optimiser stay on the record as NON-BINDING evidence of the defect.

**The order that follows:**
1. The one-frame ladder (diagnostic, running) reports. Its `lr_2e-5` and `frozen_trunk` rungs attribute the collapse.
2. The corrected MAIN arm runs (non-binding). If it FAILS criterion 2/3 at base rate, the pre-registered A10.1 chain resumes: +R6, then the PI-authorised (2026-09-27) BEV-heatmap query selection.
3. Only a run on the launch closure (A11) binds.

## 19. Amendment A14 (2026-09-27 ~10:55 Berlin, BEFORE its first number): heatmap query selection (HQS) for the slot heads

**PI, 2026-09-27 ~09:25:** "Diagnose, then build if clean". The diagnosis finds no defect in the A9 refinement. The defect is in the shared slot-decoder design, so the pre-registered A10.1 architecture lever is built.

**The diagnosis** (MEASURED, NON-BINDING; `bx_diag_0929`, `bx_ladder2_0954` on Thor). One frame (`384cb23868d0` t = 149, 13 POSITIVE), 500 steps, one variable per rung.

**Every rung FAILS the one-frame overfit:**
- the rungs: MAIN, frozen trunk, lr 2e-5, BCE presence, no deep supervision, 100 queries, and **the refcv6 head bundle** (BCE-0.1, prior 0.05, 100 queries, no deep supervision, refcv6 targets);
- matched-slot presence stays at the base rate: median 0.15–0.28 under focal, 0.63 under BCE-0.1, where every slot fires and every slot scores ~ the same;
- **the Hungarian assignment never stabilises on a fixed frame**: only 0.4–6.5 % of targets keep their slot over 25 steps;
- the box memory's frame-specific share falls in every rung (0.41 → 0.04–0.22), frozen trunk included.

**The signal audit** shows the loss wiring is correct:
- matched slots receive positive presence gradient, 100 % pushed up;
- the IGNORE mask never zeroes a matched slot (33/33 readings);
- per-layer targets equal `match_slots` on each layer (33/33).

**Reading (INFERRED, with published support):** this is the known bipartite-matching instability of learned, un-anchored queries with absolute box regression, which DN-DETR, DINO and anchored queries exist to fix. It predates A9: the refcv6 head fails the same test. It also explains the PI's "messy boxes": BCE-0.1's base rate sits ABOVE the 0.5 gate (every slot fires), focal's sits below it (none fires).

**The lever: HQS, box heads only, flag `--slot-query-select {learned,heatmap}`.**
- The default `learned` is bit-identical to today. `heatmap` is a declared field, with a G-HYG entry and a G-DVB entry.
- **Heatmap:** 2 conv layers → 1 channel on the box memory's BEV features (under the canonical config, NEW-2's pooled BEV), prior 0.01.
  - Target: CenterNet Gaussian splats of the VIS-1 POSITIVE centres, with IGNORE cells masked.
  - Loss: penalty-reduced focal (α 2, β 4), normalised by the number of positives, weight 1.0, inside the box3d term. So `memory_zeros` and `presence_w0` keep their meaning.
- **Selection:** per frame, 3×3 max-pool NMS, then top-K with K = n_queries (300). Each selected cell is that query's anchor.
  - The query CONTENT stays the learned table (DINO mixed query selection).
  - The query POSITION is a sine embedding of the anchor through an MLP, added at every decoder layer.
  - Anchors are detached; the selection is non-differentiable.
- **Box centre** = anchor + tanh(raw) × 4 m. Every other field decodes as today.

**The one-frame test, PASS literals.** Same frame, same 500 steps, the A13 launch optimiser, canonical config. At step 500 BOTH must hold:
- (i) the median presence of Hungarian-matched slots is **≥ 0.50**;
- (ii) the mean share of targets that keep their slot over the final 100 steps (25-step windows, the ladder's metric) is **≥ 0.80**.

Every rung above reads ≤ 0.065 on (ii).

**Red arm `anchors_removed`:** the heatmap anchors are replaced by the learned reference points, everything else the same. It must FAIL (i) or (ii). If it passes, the test is not measuring the anchors.

**Then the gate.** G-BOX-OVERFIT with HQS: every prereg literal, the A13 optimiser, the A10 reconciliation (113 / 77), the must-fail `memory_zeros` and `presence_w0`. Only a run on the launch closure (A11) binds.

**The pre-registered chain is unchanged and runs on otherwise-idle GPU while HQS is built:**
- the corrected MAIN (A13, 2,000 steps);
- then a one-frame +R6 rung (whether denoising alone stabilises the assignment; informative).

If the corrected MAIN or +R6 PASSES G-BOX-OVERFIT, HQS is not needed and stays default-off.

### 19.1 A14.1 (2026-09-27 ~11:15 Berlin, BEFORE any Thor number): the one-frame test's metric (ii) and red arm are corrected, and learned reference points become a candidate

**Disclosure: what revealed it.** An INFORMATIVE bench on the dev-box GPU. The box memory of the ladder frame was captured on Thor under the A13 canonical config; the trunk and BEV were frozen; only the box head trained, at the launch head lr 1e-4, weight decay 1e-4, clip 10, for 500 steps, on the landed `box3d_loss_row`. NON-BINDING.

| mode | (i) matched presence median | confident / 13 | registered (ii): slot INDEX kept | matched ANCHOR kept | unmatched max |
|---|---|---|---|---|---|
| learned (MAIN decoder) | 0.172 | 0 | 0.000 | — | 0.263 |
| HQS (A14) | 0.764 | 12 | 0.115 | 0.942 | 0.382 |
| learned reference points (the registered red arm) | 0.879 | 13 | 1.000 | (the query) | 0.053 |

**The two defects, both in the TEST, not in any arm:**
1. **Registered (ii) measured top-K RANK churn, not the assignment.** Under HQS the slot index is the rank in a score-sorted top-K, and ranks reorder as scores move. The quantity A14 meant is whether each target keeps its matched ANCHOR.
2. **The registered red arm is itself an anchoring mechanism.** Learned per-query reference points are DAB-DETR-style anchors, and they PASSED both literals. So the registered test would read VOID by construction, whatever HQS did.

**The corrections.** The bars are unchanged (i ≥ 0.50, ii ≥ 0.80), and so are the frame, the 500 steps, the A13 optimiser and the canonical config.
- **(ii), for anchored arms:** the mean share of targets that keep their matched ANCHOR over the final 100 steps, in 25-step windows. For HQS the anchor is the selected heatmap cell. For learned reference points the anchor is the query.
- **The red arm becomes `unanchored`:** zero query position and absolute box regression, i.e. the MAIN decoder. It must FAIL (i) or (ii).
- **A new candidate arm, LRP:** `--slot-query-select learned_ref`, default off.
  - DAB-DETR-style learned per-query 2-D reference points; the query position is a sine embedding of the point, added at every layer through the layers' own modules; the box centre is the point + tanh(raw) × 4 m.
  - It adds +600 parameters (300 × 2) and needs no heatmap. It is a smaller member of the same anchored-query family the PI authorised.

**The order on Thor**, the full model, trunk training, A13 optimiser:
1. The one-frame test for HQS, LRP and `unanchored`. An arm that fails its one-frame test is dropped.
2. G-BOX-OVERFIT, with every prereg literal plus A13 and the 113/77 reconciliation, on **LRP first**, the simpler lever. If it PASSES it is the launch configuration and HQS stays default-off. If it FAILS, G-BOX-OVERFIT runs on HQS.
3. The corrected MAIN (A13) and the one-frame +R6 rung are DEPRIORITISED. They run only if both anchored arms fail. The ladder and this bench both show the learned decoder failing the one-frame test.

## 20. Amendment A15 (2026-09-27 ~11:40 Berlin, BEFORE its first number): the map's next lever is the DECODER, a dilated near-range refine block stacked on the A12 lift

**Evidence** (the A12 early arm, NON-BINDING; the harness is valid). At step 1,000, declared rule:
- **MAIN** (R2, near lift on) FAILS: lane 0.496, edge 0.194.
- **Every must-fail failed as required:** `s8_zeros` reads 0 on all five thin classes; `near_zeros` reads edge 0.072; `lane_w0` reads lane 0.
- **The lever's isolated effect** (MAIN − near_zeros): edge +0.122, lane +0.030.
- **Edge is RECALL-limited:** P 0.902, R 0.429, F1 0.582 at the 0.2 m tolerance.

**Why the decoder (§9 lever 3) ranks before the weights (lever 4), MEASURED:**

| class | shape | TRAIN sqrt_mf weight | CE@1000 / CE@0 | crosses its bar (MAIN_long) |
|---|---|---|---|---|
| crosswalk | area | 1.07 | 0.067 | step 200 |
| hatched (0.05 % of cells) | area | 2.15 | 0.026 | step 200 |
| arrow (0.04 %) | area | 2.96 | 0.079 | step 600 |
| lane | LINE | 0.94 | 0.280 | step 1,200 |
| edge | LINE | 1.57 | 0.290 | never by 3,000 |

- The lag follows SHAPE, not weight or rarity: the rarest area classes are the fastest. Lane and crosswalk carry nearly the same weight, yet lane's CE ratio is 4.2× higher and it crosses 6× later.
- The path that must shape 1–2-cell LINES at 0.1 m is the decoder: two 3×3 convs, a receptive field of ≈ 0.5 m.
- Lever (4) is structurally weak for lane: lane's frequency sits at the median, so every median-frequency variant leaves its weight near 1.

**The lever: `--map-hires-near-refine-blocks`** (default 0 = off; the arm and a PASSING launch use **1**) → `MapHiresConfig.near_refine_blocks`, a declared field.
- It requires `near_lift_x_m > 0`.
- A new G-DVB kind, `map_hires_near_refine_blocks`, takes the registry 214 → 215.
- **`NearRefineBlock`:** one residual block on the NEAR rows (x 0–20 m, full width) at 0.1 m.
  - Its input is the near skip plus the upsampled features, before conv1.
  - Layers: 3×3 conv (32 → 32, dilation 2) → GroupNorm → GELU → 3×3 conv (32 → 32, dilation 4), added residually.
  - The last conv is ZERO-initialised, so the arm starts as R2's function.
  - It adds ≈ 1.3 m of receptive field, so line evidence can connect along lines.
- **Map-only:** `map_hires_bev`, the pooled BEV the planner reads, is untouched.
- **Cost (ANALYTIC):** fwd +4.3 % (103.4 → 107.8 GFLOP per sample); saved activations +0 GiB (inside the decoder checkpoint); a transient +0.25 GiB at b16; +18.5 k params; ≲ +0.1 s/step at b16 (ESTIMATED).
- **Rejected as costlier and untargeted:** a deeper full-map 0.1 m conv (+10.7 % FLOPs); d_up 64 (+66 % FLOPs, 2.29 GiB per 0.1 m activation at b16).

**The arm.**
- Setup: the A12 spec plus `near_refine_blocks: 1`, with EVERY prereg literal unchanged. MAIN = R2 + the block.
- **Must fail:**
  - `s8_zeros`, zeros into both lifts: all five thin classes fail (must_fail_all);
  - **`near_block_zeros`**, zeros into the block, which can then add only learned constants and is therefore the A12 function: edge must fail.
- **Kept:** `lane_w0`, which must fail lane.
- **Informative:** `near_zeros` and `s8_detached`.
- **If it FAILS with its must-fails holding:** §9 lever (4), the weights, is next.
- **Binding:** A11. Only a run on the launch closure binds.
- **GPU:** after the box G-BOX-OVERFIT arm (the PI's priority, 2026-09-27 11:30) and after A12's paused `s8_detached` completes.

### 20.1 Correction to A15 (2026-09-27 ~12:55 Berlin, before A15's first number): one ranking argument was wrong

A15's sentence *"Lever (4) is structurally weak for lane: lane's frequency sits at the median, so every median-frequency variant leaves its weight near 1"* is **RETRACTED** (RETR-2026-09-27-A15-WEIGHT-ARGUMENT). It was raised by the NEW-2 builder, who authored it.

- It is true of lane's ABSOLUTE weight, but that is the wrong quantity.
- Under weighted CE, what moves a class is its weight RELATIVE to the classes that fill most cells:

| ratio | sqrt_mf (the launch weights) | mf |
|---|---|---|
| w_lane / w_drivable | 0.939 / 0.206 = **4.6** | 0.88 / 0.042 = **20.8** (×4.6) |
| w_edge / w_drivable | **7.6** | **58** (×7.6) |

- So lever (4), the weights, IS a real lever for lane and edge. Its cost is the big classes' gradient share: about 10.4 % → 2.7 % each at convergence. They pass their 0.85 bars today at 0.89–0.94.

**The A15 choice stands** on the other, MEASURED argument. At nearly the same weight (lane 0.94 vs crosswalk 1.07), the LINE classes lag the AREA classes by 4.2× in CE ratio, and the rarest area classes are the fastest. The order (3) → (4) is unchanged. (4) is now framed as a real, mechanistic lever, not a weak one.

**Readiness for (4)**, if A15 fails with its must-fails holding: the TRAIN mf weights are being computed on Thor (PID 3729978, nice 19 + ionice idle, the landed script blob 8012922f, `--definition mf`, 100 × 30). The file is stamped `pre_registered: false` until (4) is registered.

## 21. Amendment A16 (2026-09-27 ~15:20 Berlin, BEFORE its first number): the map's lever (4), median-frequency weights stacked on R3

**Evidence** (the A15 early arm, NON-BINDING; the harness is valid). At step 1,000, declared rule:
- **MAIN** (near lift + near refine block) **FAILS on edge alone: 0.259.**
- **Lane PASSES at 0.522**, although that is inside the ~0.06 replicate spread. The other six classes pass.
- **The must-fails hold:** s8_zeros reads 0 on all thin classes; near_block_zeros reads edge 0.134; lane_w0 reads lane 0.
- C1–C3 and the 1 ms guard hold.
- **The lever effects stack:** edge 0.08–0.10 (NEW-2) → 0.194 (A12) → 0.259 (A15). Edge's CE ratio went 0.290 → 0.238 → 0.157. Edge is still RECALL-limited: P 0.937, R 0.581 at the 0.2 m tolerance.
- The block's step time is 0.711 s/step at b4, against A12's 0.712, so its cost is not measurable.

**The lever: the TRAIN median-frequency (mf) weights**, `map_hires_class_weights_train_100x30_MF.json`, sha256 `8ff4fd6d8798031a4af59991833ff724db251618f32c794beb2e5b8575702c98`.
- It is built from the same inputs (sha256) as the sqrt_mf file, by the landed script, over 4,369 clips, with n_clipped 0.
- **Weights** (nocls, drivable, lane, crosswalk, arrow, edge, hatched, sidewalk): 0.0199, 0.0424, 0.8825, 1.154, 8.745, 2.469, 4.635, 0.0272.
- **The quantity that moves** (§20.1): w_edge / w_drivable is 58, against 7.6 under sqrt_mf; w_lane / w_drivable is 20.8, against 4.6.
- **Cost:** the big classes' gradient share falls (~10.4 % → 2.7 % each). They must still clear their 0.85 bars, and those are in the arm's criteria.

**The arm.**
- Setup: R3 plus `--class-weights <the mf file>`. Every prereg literal is unchanged. The spec adds `class_weights_definition: "mf"`, and the harness refuses a weights file of any other definition before a trunk is built.
- **Must fail:**
  - `s8_zeros`: all five thin classes fail;
  - **`edge_w0`**: edge's LOSS weight is 0 and its decision weight unchanged. Edge must fail, which shows an edge pass comes from edge's own weighted loss.
- **Kept:** `lane_w0`.
- **Cross-run lever-off reference:** A15's MAIN, with the same seed, init, frames and harness.

**If it PASSES:**
- A8's launch weights change from sqrt_mf to **mf**: the canonical argv's class-weights file.
- The launch gate's profile must then accept `definition_id: mf` with this amendment as its registration.
- The file's own `pre_registered: false` stamp is the script's convention for anything but sqrt_mf. A16 is the registration.

**If it FAILS with its must-fails holding:** §9's pre-registered levers are EXHAUSTED, and the next step goes to the PI.

## 22. Amendment A17 (2026-09-27 ~15:20 Berlin, BEFORE the binding run): G-BOX-OVERFIT's lr decays over the final 10 %, per the PI

**PI, 2026-09-27 ~13:40 in chat:** "Add an lr decay to the harness". The PI chose this over a robust multi-point read-out and over keeping the literal snapshot.

**Why it was needed.** The early LRP run (non-binding, the A13 constant-peak optimiser) memorised the 16 frames:
- **10 of 12** readings from step 900 on pass all six criteria, several at AP 1.000, P = R = 1.000 and 113 of 113 confident.
- It FAILED the registered step-2,000 snapshot on a late transient: R 0.832, 95 confident, while the loss rose 2.16 → 3.05 over the last 100 steps.

A constant peak lr leaves the final snapshot noisy. The real launch decays its lr (cosine to step 50,400). **The early LRP FAIL stays on the record.**

**The change.** In every G-BOX-OVERFIT arm, both optimiser groups keep A13's peak lrs (trunk 5e-5, head 1e-4) for steps 0–1,799. Over **steps 1,800–2,000** (the final 10 %), both follow a **cosine decay from their peak to 0**, which keeps the groups' ratio. Weight decay 1e-4 and clip 10 are unchanged.

**Unchanged:** every other literal, i.e. the bars, the step-2,000 read-out, the must-fail arms, C1–C4, the 113/77 reconciliation, the declared gate σ ≥ 0.5, the canonical config and the LRP launch configuration (A14.1).

**The binding run** (A11 closure; its three arms are MAIN, memory_zeros and presence_w0) runs with A17 once the harness change has landed.

### 22.1 A17.1 (2026-09-27 ~15:45 Berlin, PI in chat): the same lr decay for G-MAP-OVERFIT; and A16's result

**PI, 2026-09-27 ~15:44, answering whether to apply A17's rule to the map test:** "Yes, same rule for the map". NEW-2 proposed it at 13:32Z (15:32 Berlin), before A16's first number. The PI answered as that number arrived.

**The change.** In every G-MAP-OVERFIT arm of the binding run and of any re-run, the lr holds at 1e-3 for steps 0–899, then follows a **cosine decay to 0 over steps 900–1,000**.
- The spec key is `lr_decay: {kind: cosine_to_zero, start_step: 900}`. Without the key, the constant-lr path is bit-identical.
- Literal pins: the multiplier reads 1.0 at 899 and 900, 0.5 at 950, and 0.0 at 1,000.
- **Unchanged:** every bar, arm and must-fail; C1–C3; the step-1,000 read-out; the 16 frames; the declared rule; the weights.
- **Why:** the map's constant-lr snapshot jitters ±0.03–0.07 between readings (A15 edge 0.121 → 0.086 → 0.259; MAIN_long 0.381 → 0.315), and same-seed replicates differ by 0.063 on lane. That is the failure the PI fixed for the box in A17.

**A16's result** (MEASURED, NON-BINDING; the lever-on arm judged as registered, without the decay). **MAIN FAILS** at step 1,000 on 5 of 8 classes: nocls 0.608, drivable 0.813, sidewalk 0.795, lane 0.363, edge 0.054.
- The read-out caught a transient (the train loss went 0.176 → 0.393 over steps 900–1,000).
- At its better readings (800 and 900) it still fails nocls, lane and edge (≈ 0.09). mf did not speed up edge (0.092 / 0.086 against A15's 0.121 / 0.086), and it cost the big classes the predicted gradient share.
- **The launch weights stay sqrt_mf (A8).** A16's must-fail arms were STOPPED to free the GPU, since they cannot change a FAIL; this is recorded.

**The next arm** (the decay protocol, PI-approved; a re-run, not a new lever): **A15's configuration** (near lift 20 + one near refine block, sqrt_mf weights) **+ the A17.1 decay**. Every prereg literal, with A15's must-fail pair (s8_zeros, near_block_zeros) and lane_w0.
- **If edge still fails:** a second near refine block (`--map-hires-near-refine-blocks 2`; the flag already allows up to 4; +4.3 % FLOPs per block) is the proposed next lever. It goes to the PI, since A16 exhausted §9's list.
- **The step budget is the PI's.**

## 23. Amendment A18 (2026-09-27 ~16:55 Berlin, PI in chat, BEFORE its first number): G-MAP-OVERFIT's step budget is 3,000

**PI, 2026-09-27 ~16:53**, choosing from four options (a second decoder block, both, launch with edge below the bar, or this): **"Budget to 3,000 steps"**.

**This is a dated goalpost amendment**, made by the PI under the prereg's own rule ("Changing any literal … is a goalpost move. It needs a dated amendment with the reason, and the run before it stays on the record"). **Every 1,000-step result stays on the record as FAIL:** NEW-2 MAIN, A12, A15, A16 and A17.1.

**The reason (MEASURED, all NON-BINDING).**
- **Across three levers and a decay, 7 of 8 classes pass at step 1,000.** The latest, A17.1 (A15's config + the decay), reads the big classes 0.92–0.96, lane 0.538, crosswalk 0.806, arrow 0.745 and hatched 0.840.
- **Edge is the only failing class:** 0.254 at step 1,000, essentially the same with or without the decay (A15 0.259).
- **Edge is RECALL-limited, not wrong:** P 0.974, R 0.509 at the 0.2 m tolerance.
- **Edge rises with steps:** 0.08 → 0.194 → 0.259 across the levers at 1,000; without any lever it reached 0.43 by step 3,000 (NEW-2 MAIN_long).

The test's purpose (A3, the PI) is that the 10 cm head CAN learn every class. The 1,000-step budget was measuring how FAST it learns the thinnest class.

**The change** (it applies to every G-MAP-OVERFIT arm of the binding run and any re-run):
- **N = 3,000 steps**, the read-out at step 3,000;
- the A17.1 decay moves with it: lr 1e-3 for steps 0–2,699, then a cosine decay to 0 over 2,700–3,000;
- **unchanged:** every bar (0.85 big classes / 0.50 thin classes), every must-fail (s8_zeros, near_block_zeros, lane_w0), C1–C3, the 16 frames, batch 4, the declared rule, the presence floor, §6.4, and the 1 ms guard.

**The configuration** is A15's: near lift 20 + one near refine block + sqrt_mf weights.

**The order:**
1. An EARLY MAIN arm (non-binding) runs now.
2. If it PASSES, the canonical argv becomes final: `--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1` plus sqrt_mf.
3. The BINDING runs follow on the launch closure (A11): box A17 and map A18, with every arm.

**If edge still FAILS at 3,000:** the second near refine block is the next lever, and it goes to the PI.

### 23.1 Correction (2026-09-27; its time is its landing commit): three PI-answer times in A17, A17.1 and A18 were written from memory, and they are wrong

The session transcript's timestamps are the record (UTC; Berlin is UTC+2):

| amendment | as written | the record |
|---|---|---|
| A17 (§22) | "PI, 2026-09-27 ~13:40 in chat" | asked 13:34 Berlin (11:34:03Z), **answered 15:18 Berlin (13:18:44Z)** |
| A17.1 (§22.1) | "PI, 2026-09-27 ~15:44"; heading "~15:45" | asked 15:30 Berlin (13:30:15Z), **answered 15:50 Berlin (13:50:46Z)**; landed 2ac0bfb at 15:51 |
| A18 (§23) | "PI, 2026-09-27 ~16:53"; heading "~16:55" | asked 16:20 Berlin (14:20:43Z), **answered 16:24 Berlin (14:24:23Z)**; landed 37086c3 at 16:25 |

**Nothing else changes.** The PI's words, the options offered, and each amendment's order relative to its arm's first number are as written. RETR-2026-09-27-AMENDMENT-TIMES logs the class.

## 24. Amendment A19 (2026-09-27, PI in chat, BEFORE either binding run's first number): the BINDING overfit runs are MAIN-only

**PI, 17:41 Berlin (15:41:33Z):** "We need to accelerate things to start training of refcv7".
- **Q1** (asked 17:44, 15:44:10Z; four options): keep the plan; share the GPU; map MAIN-only; or MAIN-only for both. **Answer 17:46 (15:46:18Z): "Binding = MAIN only, both"**.
- **Correction, sent before the second question.** Q1's option text said the box must-fails "held in the early runs". For the LAUNCH box head that is not so:
  - the early full-scale `memory_zeros` / `presence_w0` ran on the pre-A14 (unanchored) head, where MAIN also failed (AP@2m 0.013);
  - the early `learned_ref` run was MAIN-only (`…/2026-09-27-refcv7-box-head/raw/thor_gbo_lrp/gbo_learned_ref.json`, arms: main).
- **Q2** (asked 17:47, 15:47:40Z): keep MAIN-only for the box, or add `memory_zeros`. **Answer 17:51 (15:51:29Z): "Keep MAIN-only for both"**.

**The change** (a dated protocol amendment; neither binding run has started):
- **G-BOX-OVERFIT binding** (A11 closure): the MAIN arm only (`--arms main`). Every A9/A10/A13/A14.1/A17 literal is unchanged: 2,000 steps, the A17 decay, `learned_ref`, the 113/77 frame set, the six criteria at step 2,000.
- **G-MAP-OVERFIT binding** (A11 closure): the MAIN arm only (`--arms healthy`), on the A18 protocol: 3,000 steps, decay from 2,700, A15's config, sqrt_mf. C1–C3 and the 1 ms guard are still computed on MAIN's logits.
- **The gate** (`launch_gate.py`): `judge_box_overfit` / `judge_map_overfit` accept a binding record whose must-fail arms are absent, citing A19. MAIN must PASS every registered bar.
- **Unchanged:** the A11 closure binding, the argv sha256 `6402d33d…`, and the host-env equality.

**The must-fail evidence this INHERITS (stated exactly):**
- **Map:** held on the exact launch map config, at 1,000 steps. A17.1 (A15 config + decay) held all three:
  - `s8_zeros` all thin classes 0;
  - `near_block_zeros` edge 0.199;
  - `lane_w0` lane 0.
  A12 and A15 held them too.
- **Box `presence_w0`:** robust by construction. With zero presence-loss weight the presence head receives no gradient.
- **Box `memory_zeros`: NOT measured at full scale on the launch head.** It held in the TOY test on the A17 code (18/18 cells, ap2m 0.0022–0.0171) and on the pre-A14 head. `learned_ref` also reads BEV tokens pooled from the map branch, so a leak around the blinded memory would not be seen before launch. **The PI accepted this knowingly (Q2).**

**Why:**
- The must-fail arms were ~75 % of the binding GPU time: box 2 of 3 arms × ~86 min; map 3 of 4 arms × ~35 min. MAIN-only brings the launch forward by ~4 h.
- A second, MEASURED reason for the map: at A18's 3,000 steps, `near_block_zeros` may itself reach edge ≥ 0.50. The block's effect was +0.055 at 1,000 steps under the decay, equal to MAIN's 3,000-step margin (0.555 vs 0.50). The must-fail could then read "passed" and VOID a binding whose head learns every class.

**The order:** the map MAIN binding first (its edge margin is the thinner one), then the box MAIN binding; then G-LIVE / G-CKPT / G-EVAL on the launch commit; then the token.

### 24.1 A19 in practice (2026-09-27; its time is its landing commit): the map binding runs on an A19 MAIN-only SPEC; the box binding is HELD for a harness fix

- **The map harness refuses `--arms healthy` under the A18 spec.** The frozen `map_hires_overfit.py` (blob 9bec9e88), `main()` lines 650–652, requires every arm the spec's `must_fail` names to be in `--arms`. The map-head builder caught this before the binding started.
  - The fix keeps the harness frozen. The **A19 MAIN-only map spec** `raw/gmo_spec_A19_MAP_MAIN.json` (md5 2d11ba07a19c50726f94e4de3572155e, sha256 56dea067fdef45e9…) is the A18 spec (md5 4eda0636) with `must_fail` / `must_fail_all` removed and nothing else changed. Its script is `run_gmo_binding_a19.sh` (md5 a317bfef).
  - Verified on the frozen harness: the A19 spec is accepted with `--arms healthy`; red arm: the A18 spec is refused.
  - The gate's map judge binds THIS spec's sha256 for the binding record.
- **The map MAIN binding STARTED 16:05:13Z (18:05 Berlin)** on Thor.
  - Tree `/home/nvidia/refcv7_bind/tree_map`: tip 37086c3 + the box A17 overlay + R5's harness 9bec9e88 + the 154-token argv (gate sha 6402d33d…).
  - Wrapped by `closure_run.py --binding`.
- **A defect found in the box harness before its binding started** (the eval-loader agent, MEASURED on a tiny rig built through the real `train()`):
  - `g_box_overfit.py` (A17 blob 3c051db7) replays the 10 cm branch WITHOUT `near_lift_x_m` / `near_refine_blocks`. Under the final argv it therefore builds a model that is not the launch model; that exact replay gives 5 G-DVB mismatches.
  - The box binding is HELD: the Thor chain waits for `BOX_TREE_READY`. It runs on the fixed harness, and the fixed blob lands in the launch commit identically.
  - The fix is harness-only. `stack/tanitad/**` and the trainer, which the running map binding's closure depends on, are untouched.

## 25. PI cost approval (2026-09-27, SPEC 6.2 item 5, the reserved decision (a)): refcv7 runs at ~9.9 s/step

- **MEASURED** (the early cost probe, NON-BINDING):
  - Setup: tip 37086c3 + the box A17 overlay, the intended 154-token argv, batch 16, log 50 / conflict 10 as launched, exclusive GPU.
  - Marginal over steps 50→100: **9.876 s/step**. That is **1.54×** refcv6's 6.41 (measured the same way on refcv6's own metrics.jsonl), above the PI line of 8.0.
  - `cuda_max_mem_gb` 24.83.
  - Record: `/home/nvidia/refcv7_probe/probe_result.json`; its config.json md5 0a659ec8.
  - The trainer logs no per-part timing, so the +3.5 s/step is NOT yet attributed.
- **PI** (asked 18:01 Berlin, 16:01:19Z, with three options: approve and launch; grad-ckpt off first; profile first). **Answered 18:02 (16:02:50Z): "Approve ~9.9 s/step, launch"**.
  - Recorded for the gate as `--pi-cost-approval` with `max_s_per_step` **10.5**. That leaves room for the 30-step smoke's noisier reading.
  - File: `work/refcv7/launch/pi_cost_approval.json`, banked with the launch package.
- **The run length it implies:** 50,400 steps × 9.88 s ≈ 5.8 days.
- **Follow-up** (not a blocker): profile the +3.5 s/step on the dev box. A speed-up found later is applied at a checkpoint with a NEW gate run and new binding runs on the new argv.
