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
