# Proposal-level field loss (L_field): pre-registration (DRAFT)

> ⛔ **DRAFT. It is not in force until all four hold:**
> 1. the PI has decided **D-AUX-4** (`SPEC_AUX_DRIVABLE_HEAD.md` §0); **MET 2026-10-12: "yes" (PI in chat, SPEC §0 "PI DECISIONS": the field loss is registered as its own arm).**
> 2. the label store is built and its **S0 gate passes** (`SPEC_AUX_DRIVABLE_HEAD.md` §7: alignment, no-leakage, exact-0.0 controls, mutation arms RED). The store is shared with `PREREG_AUX_DRIVABLE_DRAFT.md`.
> 3. the wiring of the last section exists, the launch gate's entries for it exist, and a **PASS token** is bound to the arm's commit, argv sha256 and data-manifest sha256s (CLAUDE.md, launch gate).
> 4. this file is finalised, its blanks (last section) are filled, and it is **landed in git before the first training step of the first arm**.
>
> ⛔ **This draft commits NO A40 time.** D-AUX-3: the ~40 A40-h pod slot is decided only after S1 reads. Pod-hour figures below are ESTIMATED, to size a decision, not to make one.
> After landing nothing below changes without a dated amendment written before the read.
> Design: `SPEC_AUX_DRIVABLE_HEAD.md` §8. Module: `refe/field_loss.py`. Proof: `refe/test_field_loss.py` (`ZZTEST_PASS 42`, MEASURED 2026-10-12, CPU, `raw/2026-10-12-field-loss/test_field_loss.out`).
> Host recipes: `eval/PREREG_P2.md`, `eval/PREREG_P2_AMENDMENT_1.md`, `SPEC_P6_PAPER_FAITHFUL.md`. Written 2026-10-12. Evidence classes as in the SPEC.

## The question (one, so the answer can be attributed)

Does a hinge on the **footprint corners of all 64 proposals** against the stored ground-truth signed distance field cut the drivable-area failures of the *picked* plan on navtest, at no cost in progress or fan diversity?
It is a **separate arm from the head** (D-AUX-4). It shares the labels and nothing else; a combined arm is allowed only after each has its own verdict.

## The treatment (identical in every host; both numbers fixed BEFORE any read)

| item | value | status |
|---|---|---|
| loss | `L_field = mean over labelled samples, 64 proposals, 40 steps (t = 0.1 … 4.0 s), 4 footprint corners of relu(m − SDF_gt(corner))`, exactly `refe/field_loss.field_loss` | module sha256 in the blanks |
| **margin m** | **0.25 m** | **FIXED NOW** |
| up-sampling | `interp="5hz"`: 20 poses at 5 Hz → 10 Hz by linear interpolation from the origin pose, wrapped heading (SPEC §8) | FIXED NOW |
| footprint | nuPlan Pacifica corners, equal to `navsim_dac.corners_global` (max deviation **1.4e-14 m** in float64, 1.4e-5 m in float32, 200 × 41 random poses) | MEASURED, `raw/…/test_field_loss.out` |
| lookup | bilinear `grid_sample` on the int8 SDF × 0.1 m; a corner outside the cell-centre hull of the grid (0.25 m inside x ∈ [−8, 72), y ∈ [−32, 32)) carries exactly 0 loss and 0 gradient | MEASURED by the module's tests |
| reduction | sum / (n_labelled × 64 × 40 × 4); a sample without a label contributes exactly 0 (loss and gradient) | – |
| **weight w_field** | **set once, before launch, by a gradient-norm balance, never tuned on PDMS** (below), then frozen | procedure FIXED NOW, value in the blanks |
| total loss | `(l_traj + l_score + w_field · L_field) / accum`; **`--w-field` defaults to 0.0 and every arm's argv names it** (the control names `0.0`) | – |

**Weight rule (committed now).** On 8 real micro-batches of the arm's own bank at the host's init (seeded draw, seed 20261012), measure
`r(w) = ‖∇_{traj_head}(w · L_field)‖ / ‖∇_{traj_head} l_traj‖`. Candidates `w ∈ {1/4, 1/2, 1, 2, 4, 8, 16, 32, 64}`. **Take the smallest w with r(w) ≥ 0.25.**
If r(64) < 0.25 the arm is **not launched** (K2). If r(1/4) > 1.0 the arm is not launched (K2b: the field term would dominate the trajectory loss at the lowest weight; investigate the label).
*Why a rule and not a number:* the mean divides each proposal's share by 64 × 40 × 4, while WTA trains the winner at 1/20 per point, so a unit weight is ESTIMATED to be one to two orders of magnitude weaker per proposal; the published method's weight and tolerance band were **not found** in the extracted text of `2605.04647` §4.4 (UNVERIFIED), so there is no published value to inherit.

**Weight rule, clarification of the reading (2026-10-13, before any measurement on real data):** r(w) is computed on the gradient **accumulated over the 8 micro-batches** (the gradient the optimiser sees under `--accum 8`), not as a mean of 8 per-micro-batch ratios. `raw/2026-10-13-field-loss-wiring/weight_rule.py` implements this reading; it has run on a tiny synthetic bank only (output stamped NOT_A_DECISION), never on real data.

**Gradient reach (required by the gate, not yet measured).** `L_field` depends on `traj` only. It must reach `traj_head`, the queries and proposal decoder, `scene_proj`, `reg_compress` and the LoRA, and must read **exactly 0.0** on `score_q_mlp`, `score_dec`, `score_head`: L_field's graph holds only the ancestors of `traj`, and the scorer sits downstream of `traj` (`model.py:775-779`), not upstream.

## What the published precedent does and does not say

ReflectDrive-2 (`2605.04647`, Table 3 / Table 6 text: DLM 84.8 → DLM + DACF 87.2, DAC 93.9 → 97.0; one run) is the largest controlled published effect on drivable-area compliance, **but it is a different mechanism**.
Its Eq. 12–14 is a **log barrier on the probability mass of a discrete waypoint distribution**, weighted by a BEV cost field `max(0, d_out − safe)`. REFe regresses continuous poses, so L_field is a **hinge on the four footprint corners**, 1-Lipschitz, with an exactly-zero gradient on a clean fan.
Same signal (distance to the drivable area), different mechanism: **+2.4 is context, not a prior for this arm.**
Hydra-MDP++ (`2503.12820`, −0.5 for perception aux tasks) is a feature head, not an output loss, so it does not bear on L_field directly; it bears on the head arm.

## Hosts, arms, and what each can show

⛔ **A host must train the proposal side.** `refe/scorer_finetune.py` (SFT-4, RL-S1) trains only the scorer and leaves the 64 proposals bit-identical (its own docstring), so **L_field cannot act there by construction.** The host is a `refe/train.py` run.

| host | init | arms | status of the init | ESTIMATED pod time (NOT committed) |
|---|---|---|---|---|
| **H-P2** graft on the P2 output | the P2 final checkpoint (`/workspace/data/refe_runs/p2_joint_pdm/`) | **F** (w_field), **C** (`--w-field 0.0`), same init, bank, steps, seed | does not exist: P2 awaits the PI's pod (`P6_GO_SIGNAL.md` status line). **UNVERIFIED** until P2 finishes | ~20 A40-h per arm at P2 scale (P2: ≈1,209 steps, ≈20 h at ~60 s/step, `PREREG_P2.md`) |
| **H-P6** graft on P6's `model_final.pt` | P6's final checkpoint | F, C as above | does not exist: P6 follows only on P2's GO (`PREREG_P2_AMENDMENT_1.md`). **UNVERIFIED** | as above |
| *(my addition, for the Master Mind)* **H-0** P2's recipe from the deployed `model_final.pt` (md5 `b54773f8…`) | the deployed model | **F** only, with **P2 itself as C** | exists on the HF backups (INHERITED, `HANDOFF_2026-10-11.md`; md5 from `PREREG_P2.md`) | ~20 A40-h for F alone. Valid only if the zero-weight exact test below passes (the trainer with `--w-field 0.0` is bit-identical to the one P2 ran) and F matches P2's bank, steps, seed and argv |

Replicate arms (the seed floor, H-ESTIM-SEED-1): **C-seed** (control, seed 1) and **F-seed** (arm, seed 1). A single pair (F, C) is "separated, single seed, not yet a lever claim" at best.
ESTIMATED totals: 2 arms ≈ 40 A40-h, 4 arms ≈ 80 A40-h. **Cost of the loss itself:** 2.8 ms (B = 4) and 4.6 ms (B = 8) forward + backward on 4 CPU threads, random data, `raw/2026-10-12-field-loss/bench_cpu.out`; A40 UNMEASURED (K3 reads it).
⚠️ A graft tests the lever on a converged trunk at a small LR: a **weaker** version of the effect than one from step 0 (SPEC §5). A NOT PROVEN here is not a refutation of the loss at scale.

## Estimator, tier, and what the interval answers

- **Tier:** NAVSIM v1 PDMS, an open-loop plan pseudo-simulated against logged agents. **Not closed loop.**
- **Population:** the full navtest, **12,146 tokens over 136 logs**, NAVSIM v1.1; harness and seam as for `RESULT_SFT4.md` (route fix ON, A7 repair ON, `navsim_v1` selection), on the dev box, never on a training pod. Each arm runs under its own registered selection rule.
- **Estimator:** paired log-cluster bootstrap, **10,000 resamples, seed 20260927**, for every difference and level.
- **It answers one question:** *would another draw of logs say this?* It is blind to a second training run (H-ESTIM-SEED-1, MEASURED: a zero-lever replicate produced "separated" cells at 9.5 %) and, here, to nothing else: REFe inference is deterministic, **asserted** (two evaluations of one checkpoint must be bit-identical per token; a mismatch is a defect).
- **Minimum detectable effect (ESTIMATED):** ≈ **0.75 PDMS**, from SFT-4's paired half-width (`RESULT_SFT4.md`). To ADOPT, the lower bound must also clear the seed floor, so a point Δ of roughly **0.75 + floor** is needed.
- **Ceiling (ESTIMATED arithmetic, upper bound):** the DAC-zero part of the zero tail is **3.26 PDMS** (SPEC §1); the loss acts on the *raw* plan and cannot touch the tracker-made share of the failures (**29.2 %**, INHERITED, SPEC §1), so it can recover at most 3.26 × (1 − 0.292) = **2.31 PDMS**. The plausible range is therefore +0.3 to +2.3 and **NOT PROVEN is a likely outcome, written down as such.**

## Endpoints (paired, same tokens)

| id | endpoint | role |
|---|---|---|
| **E1** | **PDMS** (F − C) | the decision endpoint |
| **E2** | **DAC sub-score** and **DAC-zero share** (u400 reads 3.42 % = ~~415~~ **416** tokens = 3.425 %, corrected 2026-10-13, `eval/PREREG_AUX_DRIVABLE_S1.md` R4; the comparator's own value is read, not inherited) | the **mechanism**: a PDMS gain without fewer DAC zeros is not credited to the loss |
| **E3** | zeroed-token share (NC × DAC = 0) | the zero tail |
| **M1** | **fan cleanliness**: per token, the share of the 64 proposals that are NAVSIM-DAC-compliant; and the same share on the **raw** plan (`navsim_dac_violation(mode="raw")`) | the loss acts on all 64, so this moves first. Diagnostic on the 1,058 C3 tokens (`raw/2026-10-11-c3-selection-headroom/tokens.json`, failure-enriched: **not a rate**) |
| **M2** | **raw → tracked transfer**: ΔDAC on the raw plan against ΔDAC on NAVSIM's LQR-tracked plan | the loss fixes the raw geometry; if raw improves and tracked does not, the loss fixed the wrong thing (routes to tracker-aware compensation) |

All six sub-scores are reported with their paired CI. Nothing is pooled across families.

## Adverse checks (pre-registered; the loss may shorten plans or collapse the fan)

Thresholds are **proposed here; the PI may amend them before landing.**

| id | check | ADVERSE iff |
|---|---|---|
| **A1** | EP sub-score Δ (F − C) | the interval is separated below 0 **and** the point Δ ≤ −1.0 |
| **A2** | pick length: the picked plan's 4 s along-track distance, F / C, mean over tokens (cf. `D-REFE-SLOWPICK-SFT4-1`, proposed, INHERITED) | the ratio's upper bound < 0.97 |
| **A3** | fan spread: mean pairwise distance between the 64 proposals' 4 s endpoints, F / C, on the M1 tokens | the ratio's upper bound < 0.90 |

## Rule (committed now)

**ADOPT** (per host) iff **all** of:
1. E1's lower bound > 0;
2. E2's DAC-zero-share upper bound < 0 (the mechanism moved in the stated direction);
3. none of A1–A3 is ADVERSE, and no other longitudinal or lateral family component separates adversely;
4. the replicate pair (F-seed − C-seed) reproduces both signs: E1 point > 0 **and** DAC-zero-share point < 0;
5. E1's lower bound exceeds the host's **measured seed floor** = |E1(C-seed − C)|. If the floor is unmeasured at read time the verdict is **"separated, single seed, not yet a lever claim"**, never ADOPT.

**REFUTED** iff E1's upper bound < 0, **or** E2's DAC-zero-share lower bound > 0.
**TRADE** iff 1 and 2 hold and at least one of A1–A3 is ADVERSE (the loss converted DAC failures into slow or collapsed plans).
**NOT PROVEN** otherwise.

**Size label (reported with every ADOPT):** **LARGE** iff E1 Δ ≥ **+1.0** and DAC-zero share ≤ **2.39 %** (a ≥ 30 % cut, ≤ 290 tokens); else **SMALL**. Labels, not bars: the verdict is the rule above.

## Kill conditions (each stops the arm and names the next one)

| id | condition | then |
|---|---|---|
| **K1** | the label gate fails (alignment < 95 % on ≥ 64 real frames, or any mutation arm not RED) | no launch; fix the label store |
| **K2 / K2b** | the weight rule: r(64) < 0.25 / r(1/4) > 1.0 | no launch; K2 = the host's fans are too clean for the loss to matter, report the dirty-fan share instead; K2b = check the label sign and extent |
| **K3** | measured step time rises > **3 %** or peak memory > **0.5 GB** at micro-batch 4 | profile the loss on the A40; the CPU cost is 2.8 ms, so a rise most likely means the loader (the label read), not the maths |
| **K4** | at any full-navtest read, E1's upper bound < 0 | REFUTED; stop; §Next arm |
| **K5** | a zero-weight / reach control is not exact (below) | no launch, or the arm is not credited |
| **K6** | the logged `field_viol` (share of in-extent corners with SDF < 0) at 25 % of the steps is > 0.8 × its step-0 value | the loss is not acting: check labels, extent, and that `field_w` was applied |
| **K7** | DAC-zero share falls but A1 or A2 is ADVERSE and E1 Δ ≤ 0 | stop; route to the selection/progress calibration (0 GPU) |

## Deliberate-regression controls

| control | form | expected |
|---|---|---|
| **clean fan** | exact test, in `test_field_loss.py` (MEASURED GREEN): a fan on the road at ≥ margin gives loss **== 0.0** and gradient **== 0.0** | exactly equal |
| **the module's mutation arms** | 14 source mutations (margin sign, x/y swap, align-corners, up-sampling dropped, loss on the best-matching proposal only, int8 unit, hinge, extent mask, yaw wrap, detach, valid mask, rear-axle offset, interp switch, time base), each must fail ≥ 1 check; each of the six the PI listed is caught by at least one hand-computed check (no reference implementation involved) | MEASURED, `test_field_loss.out` |
| **zero weight** | `--w-field 0.0` ⇒ every gradient from the term exactly 0.0 **and the first N optimiser steps bit-identical to the trainer without the flag** | exactly equal (this also licenses H-0's use of P2 as C) |
| **label shuffled across samples** | optional arm, a derangement of the label keys (seed **20261011**, 0 fixed points, same multiset of labels); not budgeted, stated | **no gain**: E1 inside the noise, E2 not moving. A gain here means the lever is regularisation, not map information |
| **label misalignment** | mutation of the alignment gate: y-flip or a one-frame offset | the gate must read RED |
| **frame sanity on real data** | 20 real teacher paths against their own real labels: 1 of 20 leaves the area (the teacher is known to leave on ~5 %), loss 0.0012; the y-mirrored paths: 7 of 20, loss 0.176 (MEASURED, n = 20, `test_field_loss.out` check R8) | small n; the S0 alignment gate is the real test |

## Replicates and the seed floor

- **C-seed** is the zero-lever replicate: same flags as C, seed 1. The floor is |E1(C-seed − C)| and the same for each sub-score. One pair is a **single draw** of the floor (it can understate it); the published TransFuser std 0.56 over 3 seeds (`2406.15349`) is **context only**, a different model.
- **F-seed** is the arm at seed 1. ADOPT needs it to reproduce both signs (Rule 4).
- If only the seed-0 pair is run (≈ 40 A40-h), the best possible verdict is **"separated, single seed, not yet a lever claim"**.

## Four families (binding; per family, never pooled)

Instrument: the one that produced `sft4_B_u400_families.json`, on the identical tokens, each tested component with a paired CI.

| family | components | directional prediction (registered before the read) |
|---|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; EP and TTC sub-scores; headway/time-gap where a lead exists (n and reason where none) | EP is the **adverse risk** (A1); TTC unchanged |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; **DAC sub-score**; last-pose heading error separately (D5) | DAC up; cross-track MAE not worse; the loss gives **no yaw gradient of its own** beyond the corner positions, so yaw-rate MAE should not move |
| **TACTICAL** | lateral / longitudinal / 5-way decision accuracy and κ; command adherence of the selected proposal; **selection quality as random / actual / best-of-64 on the same tokens, the random control mandatory** | all three rise because the fan is cleaner; **actual − random may shrink** (a cleaner fan helps a random pick too), which is not by itself adverse |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Reported as PAUSED with that ruling: not a pass, not omitted, not extended to the other three | – |

## Admissibility (SPEC §6, unchanged)

Training-only: the label is a map-derived **training target** (PI 2026-10-11); it is never an inference input; the planner imports no label path (`grep -c "field_loss\|sdf" refe/model.py refe/planner.py` must stay 0). Navtest labels are never used to train or select (`label_store_keys ∩ navtest_tokens = ∅`, G-LIVE). Goal ⟂ situation (PI 2026-08-03): the loss takes no goal, ego or navigation input; it reads the 64 proposals the model already emits. Anything built on it is "paper recipe + a declared deviation", never "reproduction" (DriveZero uses no annotated aux supervision, `2609.06055` §2.2).

## Next arm, by outcome (Rule Zero; committed now)

| outcome | next lever (executed in the same turn that reads the verdict) |
|---|---|
| **ADOPT, LARGE** | (1) the seed replicate at the host if not run; (2) the next host rung (SPEC §5 S3, declared deviation); (3) the combined arm with the head (H1) on the same labels; (4) the scorer ensemble (P7.2) |
| **ADOPT, SMALL** | the replicate first; do not spend a 7-day run on a lever whose lower bound sits at the floor |
| **TRADE** | the selection/progress calibration on the same proposals (0 GPU, dev box), then a narrower pre-registered variant (loss on the top-k proposals by score only) |
| **REFUTED** | bisect in this order: (i) did the gradient reach (r, K6)? (ii) **M2**: did the raw plan improve but not the tracked one ⇒ tracker-aware compensation (the 29.2 %), not a new loss; (iii) the label-shuffle arm (regularisation?); (iv) a second pre-registered arm at m = 0.5 m, the tracker's tolerance |
| **NOT PROVEN** | the replicate (C-seed first: it prices the floor); more logs are **not** the answer, they narrow an interval that already answers the wrong question |
| **K7** | the selection/progress calibration, 0 GPU |

## Wiring plan (NOT wiring: `train.py`, `model.py`, `scorer_finetune.py`, the launch-gate files, `rl_fan_dump.py`, `rls1_pod.sh` are other streams')

Anchors are the working tree on 2026-10-12 and drift; `SPEC_AUX_DRIVABLE_HEAD.md` §4's line numbers are already stale.

**`refe/train.py`**

| step | anchor | change |
|---|---|---|
| import | `:37` (`from model import … wta_loss`) | `from field_loss import field_loss` |
| flag → namespace | `build_parser` `:1842`; patterns `--score-w` `:1918`, `--yaw-wrap` `:1965` | `--w-field` (float, **default 0.0**), `--field-margin` (0.25), `--field-interp` (`5hz` \| `navsim2hz`, default `5hz`), `--field-labels` (store path), `--field-shuffle-seed` (the control arm) |
| resume identity | `IDENTITY_ARGS` `:963-967`, `IDENTITY_DEFAULTS` `:970-971` | add the five; defaults 0.0 / 0.25 / `"5hz"` / None / None so a pre-existing checkpoint still resumes |
| **loader: deliver the SDF per sample** | `TargetBank.__init__` `:629`; `__getitem__` `:786`; the item tuple `:875-880` | open a memmap `labels.i8 [N, 160, 128]` + a key index `(log_name, token) → row`; append `(sdf_int8 [160, 128], valid)` after the optional `idx` that `online` adds (`:880`). `r["log_name"]`, `r["token"]` are already on the row (`:1557` reads them). A missing key ⇒ zeros and `valid = False`. Goal twins share a key (SPEC §3.1). Default collation stacks it to int8 [B, 160, 128] |
| batch unpack | `:1533-1534` (`batch[:9]`, `ridx = batch[9]`) | read the two new elements by **negative index** (`batch[-2:]`) so the `online`-conditional position of `idx` is untouched; `.to(dev)` beside `:1535-1538` |
| the term | after `:1551` (`wta_loss`); `traj` is already `.float()`, outside autocast (`:1548`) | `l_field = field_loss(traj, sdf, margin=a.field_margin, valid=valid, interp=a.field_interp)` only when `a.w_field > 0` |
| into the loss, three sites | `:1558-1559` (the deferred `cur` dict, add `"l_field"`); `:1683` (`((P["l_traj"] + l_score) / a.accum).backward()`); `:1693-1694` (`loss = (l_traj + l_score) / a.accum`) | add `+ a.w_field * l_field` at the last two; carry `l_field` through `cur` for the online path |
| telemetry | metrics row `:1712-1720` | `field_L` (unweighted), `field_w` (the multiplier actually applied), `field_viol`, `field_inext` (from `return_parts=True`) |
| checkpoint | `_save_ckpt` `:1512` | **nothing**: the loss has no parameters, so G-CKPT's round-trip is unchanged |

**Launch gate (`refe/gate_checks.py`, `refe/launch_gate.py`)**

- **G-DVB:** add `w_field`, `field_margin`, `field_interp`, `field_labels`, `field_shuffle_seed` to `DVB_KINDS` (`gate_checks.py:372-384`), or the gate refuses the new flags (its `stale_registry` arm). Observed values in `dvb_rows` (`:401-446`) by the `rec_wta` pattern (`:200-207`, `:445-446`): wrap `T.field_loss` in the smoke harness and record the `margin` and `interp` it **received**; compare `w_field` with the logged `field_w`. Also: label manifest sha256 and frame count; the grid constants of the manifest equal the module's (`X0, X1, Y0, Y1, RES, NX, NY, SDF_UNIT_M`). Mutation arms: multiplier wired off by a factor ⇒ FAIL; `--w-field` absent from the argv of an arm that means it (default 0.0) ⇒ FAIL.
- **G-LIVE:** add `field_L` to the profile's `live_terms` (`launch_gate.py:41` `P6_TERMS`, or a new `PROFILES["refe_field"]` cloned from `refe_p2` `:62-66`; `check_live` `gate_checks.py:759-800` then requires it present and finite on every logged row). New reach check: an L_field-only backward gives a **non-zero** gradient on `traj_head`, the proposal decoder, `scene_proj` and the LoRA and **exactly 0.0** on `score_q_mlp`, `score_dec`, `score_head`; `--w-field 0` ⇒ exactly 0.0 everywhere; report `r(w)` of the weight rule, label coverage (≥ 99.9 %) and `field_inext`.
- **G-LIVE (alignment / leakage):** shared with the head's S0 gate (`navsim_dac.violations` agreement; `keys ∩ navtest = ∅`).
- **G-SUITE:** add `{"script": "test_field_loss.py", "token": "ZZTEST_PASS"}` to `SUITE` (`launch_gate.py:37`; `RLS1_SUITE` `:56` shows the pattern).
- **G-EVAL:** unchanged by construction (no inference dependency); add the `grep -c` of the admissibility section.
- **Profile `required_flags`:** `--w-field <registered value>`, `--field-margin 0.25`, `--field-interp 5hz`, `--field-labels <path>`; the control profile `--w-field 0.0`.

**Loader change, in one line:** one int8 [160, 128] plus a valid bit per sample, 20,480 B read from a memmap, keyed by `(log_name, token)`; no augmentation to invert (SPEC §3.1).

## Blanks to fill at landing

| blank | value |
|---|---|
| label store: path, frame count (expect 103,288), manifest sha256 | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store/`; **103,139 frames, not 103,288** (100 % of the 103,037-row rank-0 bank, MEASURED `raw/2026-10-13-field-loss-wiring/real_bank_label_coverage.out`; the 149 yaml tokens without a bank row or a local DB are in `missing_tokens.txt`); `store_sha256` `4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a`, `manifest.json` sha256 `5b629af3ca5354ca260bafdcb179ad182a9acb68f8357e119076ee197c3671a9` (filled 2026-10-13) |
| navtest label store (for evaluation only): path, frame count (expect 12,146), sha256 | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest/`, 12,146 frames, `store_sha256` `985a3bc77f3ab7a96b0dceace48e73b80e5b69b70710958635060939bce0d7c1`, 0 overlap with the train store (`raw/2026-10-13-aux-navtest-store/s0_gate_navtest.json`, gate PASS; filled 2026-10-13) |
| `refe/field_loss.py` sha256 and the `test_field_loss.py` output (`ZZTEST_PASS 42`) | `e29d91c349539ba366c73739fd2fe7fc6f1e7a6a015c4c0749df5f15df36c994` at `ee16738` (unchanged since `7da149b`); `ZZTEST_PASS 42` re-run 2026-10-13 by the wiring work (`raw/2026-10-13-field-loss-wiring/`). **Re-hash at launch**: the token binds the shipped bytes |
| host (H-P2 / H-P6 / H-0), its init checkpoint and sha256 | _ |
| **w_field** after the weight rule, with r(w) for every candidate and the 8 micro-batch ids | _ |
| comparator checkpoints and sha256 (C, C-seed) | _ |
| seeds for every arm; the derangement seed (20261011) if the shuffle arm runs | _ |
| the C3 token set used by M1/A3 and the instrument that produced it | `raw/2026-10-11-c3-selection-headroom/tokens.json` (1,058 tokens: u400's 529 zeroed + 529 random), `table.npz` produced by `c3_run.py` (u400 scorer, `navsim_v1_lane`); its scorer logits are reproduced bit for bit on all 1,058 by `eval/aux_navtest_forward.py` (`D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full/identity.json`, 473,984 values, 0 differ; filled 2026-10-13) |
| harness commit and the seam's flags | _ |
| gate PASS token id (commit, argv sha256, data-manifest sha256s) | _ |
