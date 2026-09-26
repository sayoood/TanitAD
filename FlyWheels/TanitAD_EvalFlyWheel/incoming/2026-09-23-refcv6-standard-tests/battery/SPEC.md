# SPEC — refcv6 standard held-out battery (PRE-REGISTRATION)

**Stream:** EvalFlyWheel, refcv6 standard tests, four-family held-out battery (the NavSim suite is a sibling stream and is not covered here).
**Written:** 2026-09-23, finished 23:33 Europe/Berlin (21:33Z). This was **before any refcv6 forward pass by this stream**. The first forward is the loader reproduction in §2. Its sha256 is recorded in `raw/SPEC_SHA256_AT_REGISTRATION.txt` at the moment of writing, and every runner output stamps the sha256 of the SPEC it ran under.
**Refcv6 numbers the author had seen before writing:** only the run's own in-run eval rows in `metrics.jsonl` (step 500: `eval_traj` 1.23225, `eval_loss` 38.16569; step 1000: 1.21898 / 34.26996, plus the other eval_* terms of that row). These are loss diagnostics on 128 windows. `eval_traj` is an oracle-anchor mean L1 per coordinate (`refc_v3_train.py:3580-3598`). None of them is a battery metric and none enters a bar below.

---

## 0. Object under test, code, data

| | |
|---|---|
| run | `refcv6-r101-s0`, Thor, launched 2026-09-23 20:42 Berlin. Commit `284393c` for steps 0–500, then `287d72e` (a logging-only switch) |
| code | `C:/Users/Admin/ev6` = `git archive` of `fe5872f`. Between `287d72e` and `fe5872f` only `taniteval/tools/training_watch/*` changed. The blobs of `refc_v3_train.py`, `refc.py`, `refc_v3.py`, `refcv6_perception_branch.py`, `refcv3_arm.py`, `ci.py` and `four_families.py` were verified equal to `287d72e` (MEASURED, blob comparison, 40-char asserted) |
| model record | the kit's `config.json` (argv, `param_breakdown`, `tac_goal_stats`, `agent_join_stats`, perception stamp) |
| eval clips | kit `refcv6-b1-416x1024-eval139`: 139 clips. These are the files the run's in-run eval reads, copied from Thor. ⚠️ The local rebuild `v2ep-eval139-416x1024cyl` differs from them **byte-wise on 139 of 139 files** (MEASURED md5, `raw/kit_checks/eval139_md5_compare.json`), so it is **NOT used** |
| labels | v8 eval labels, md5 `eefc38d1…`, the same md5 the run stamps |
| max-speed input | `refcv6_speed_max_v8_eval.jsonl` (the run's `--speed-max-sidecar-v6-eval`) |
| SAM3 map GT | 137 files. They are byte-identical by md5 to Thor's `sam3_corpus/semantic_maps/gt` for the eval clips (133 from the local copy + 4 pulled read-only). The same 2 clips are absent there as on Thor (`raw/…/_GT_RECORD.json`) |
| joins | the kit's train+eval 2-D and 3-D joins (the run's `--agent-join` / `--join3d`) |

---

## 1. Tiers, estimator, variance, and what nothing here may be called

* **T1 = self-action OPEN LOOP**: one forward pass at the window origin, and the path is the model's own selection. It is **not closed loop and not driving performance** (EVAL_DOCTRINE, amended 2026-09-02). refcv6 consumes no actions. The tier ruling for an action-free model (BACKLOG R30) is **UNRULED**, so `os` is stamped `T1` with `status UNRULED`, as refcv3/refcv4b/refcv5-v2 were. **T0** (`oracle_sel`, the GT-nearest anchor) is a ceiling diagnostic and is never compared to a T1 number.
* **Point estimate:** the FULL-SET pooled mean over windows.
* **Interval:** `taniteval.ci.episode_cluster_bootstrap`. For two arms on the same windows, `taniteval.ci.paired_episode_cluster_bootstrap`. Settings: `n_boot 2000`, `seed 0`, cluster = episode (clip), 95 % percentile interval. ⛔ **`overlapping_holdout_se` is forbidden** (it biases the point estimate).
* **Which variance each interval answers:**
  * *another draw of EPISODES*: answered by the bootstrap.
  * *another INFERENCE run*: refcv6 samples (`--sampler ddim` draws `eps` at eval, `refc.py:2551`). This is answered by rolling **two inference seeds (0 and 1)**, and every bar must hold at both.
  * *another TRAINING run*: **NOT MEASURED**. There is one training seed (`H-ESTIM-SEED-1`). Replicate false-positive rate for "separated" on the tiny rig: 6/42 = 14.3 %.
* **Four families, never ADE alone, never pooled into one score** (CLAUDE.md binding rule). A family that cannot be computed is reported with its `n` and its reason.

---

## 2. GATE G0: the loader must reproduce the run's own in-run eval (nothing downstream counts until it passes)

**What is reproduced.** `metrics.jsonl`'s eval row at the kit checkpoint's own `step` (step 1000 for the kit). That row is the mean over **8 fixed batches of 16** = 128 windows. The windows are `torch.randperm(len(e_ds), generator=Generator().manual_seed(12345))[:128]` over the eval dataset, iterated in order (`refc_v3_train.py:7561-7567`). Each batch value comes from the trainer's own `compute_losses_v3` under `model.eval()` + `no_grad`, and the row is `round(sum/8, 5)`.

**How** (`code/reproduce_inrun_eval.py`):
* The model and eval dataset are rebuilt with `code/refcv6_loader.py`, which replays `train()` block by block.
* The checkpoint is loaded STRICTLY.
* `--trunk-compile` is dropped (no Triton; it wraps the call only). bf16 + NHWC, fold, dedup and chunk-8 are kept as run.
* ⚠️ Batch 16 cannot sit on the 8 GB RTX 4060 in one forward (the frames alone are 1.96 GB fp32, normalised in place twice). The model's forward is therefore **micro-batched inside `compute_losses_v3`'s single `model(...)` call** (sizes 3,3,3,3,4). The outputs are concatenated before the loss, so **every loss reduction still runs over the same 16-row batch**. Batch-dim detection uses the unequal sizes (a tensor is concatenated only if its dim 0 equals its own micro-batch size in every call). 0-dim outputs are row-weighted means, and every key's merge rule is recorded.
* The LAW target reads only `future_frames[:, LAW_AHEAD-1]` (`:3714`), so only that frame is moved to the device.
* ⭐ **Wrapper control (must read a known value):** on one real batch, the micro-batched and the unwrapped forward are run with the DDIM draw replaced by zeros in both. Every loss term must then agree to relative 1e-3. (This bounds the float noise of different conv batch sizes; a missed merge rule would show up as a gross difference.)

**Classes of term (assigned by name before any value is seen; the stochastic class is assigned by measurement):**

| class | members | tolerance vs the in-run value |
|---|---|---|
| **EXCLUDED** | `eval_goal_gate_grad` | none. It is the `goal_gate.grad` left by the last TRAINING backward (`compute_losses_v3` reads `model.goal_gate.grad`), so a loaded checkpoint has none. It is not an eval quantity |
| **COUNT** | `eval_batches`, `eval_windows`, and every key containing `n_` (e.g. `agent_n_*`, `box3d_n_*`, `map_n_*`, `n_map_cells*`, `tacv6_n_*`), `slot_valid_frac`, `tac_label_rows`, `tac_label_v7`, `nav_injected`, `ego_injected`, `box3d_visible_filter`, `agent_rows_*`, `map_gt_drivable_frac` | **exact** at the row's 5-decimal rounding (abs ≤ 1e-5). These are data and geometry |
| **MATCHED** (Hungarian set losses, discontinuous in the logits) | `agent_centre`, `agent_cls`, `agent_presence`, `agent_size`, `agent_yaw`, `box3d`, `box3d_centre`, `box3d_cls`, `box3d_h`, `box3d_occ`, `box3d_presence`, `box3d_rates`, `box3d_size`, `box3d_yaw`, `box3d_z` | relative ≤ **15 %** each, **and** median over the class ≤ **5 %**. Basis, MEASURED in the run's own smokes: a pure precision lever moved agent yaw 0.848 → 0.679 and box height ~8 % on 2-batch evals (LAUNCH_READINESS_FIXES §9) |
| **STOCHASTIC** | every non-excluded term whose value across the K inference seeds varies by more than relative 1e-5 ((max−min)/\|mean\|) | the in-run value must lie in the **99 % prediction interval** of the K = 8 seed values, `mean ± t(0.995, K−1)·sd·√(1+1/K)` (t = 3.499), widened by 1 % of \|mean\| for the cross-hardware bf16 shift |
| **SMOOTH** (all remaining, deterministic in the seed) | e.g. `lat`, `lon`, `lat_tac`, `lon_tac`, `route`, `law`, `goal_tac`, `goal2s_err_m`, `goal_gate`, `tac_v6`, `tacv6_*` losses, `map`, `map_iou_drivable`, `map_pred_*` | relative ≤ **1 %** (or abs ≤ 1e-3 when \|in-run\| < 0.1). **Median over the class ≤ 0.2 %**. Basis, MEASURED in the smokes: eager vs compiled bf16 in-run eval, median deviation 0.07 %; fold 0.09–0.10 %; bf16 vs fp32 0.03–0.04 % |

**Deliberate regression (the gate must be able to FAIL).** Arm **M1** re-runs one seed with the **lift bank built without `equalize_bottom_rows`**, which is exactly what `taniteval/tools/refcv3_arm.py:1101` builds. It must leave the SMOOTH tolerance on **at least one** term. If M1 passes, the gate has no power and G0 is **VOID**, not PASS.

**Verdict:** G0 **PASS** iff the strict load is clean (0 missing, 0 unexpected), `param_breakdown` equals the stamp, the kit anchor file equals the checkpoint's anchor buffers (max |Δ| 0.0), the wrapper control passes, every term meets its class tolerance, and M1 fails. Otherwise **STOP**: report every differing term with both values, and run no battery number.

---

## 3. The battery

### 3.1 Surfaces (MODEL-FREE facts measured before registration, `raw/pairing_surface.json`)

* **S2, PRIMARY** = the banked refcv4b / refcv5-v2 dev-box 2 s grid (stride 5, instants 0.5/1.0/1.5/2.0 s, model slots 5/10/15/20), restricted to the 139 kit clips: **4,754 windows / 139 episodes**. On all 139 common clips the 256×640 and 416×1024 caches carry **bit-identical `poses` and `actions`**. The GT recomputed from the kit poses through the trainer's own `refb_labels.waypoint_targets` equals the banked `g` **exactly on 4,754 / 4,754** windows (max |Δ| 0.0). The three banked baseline dumps agree bit for bit on (clip, `ws`, `g`, `ha`, `ha0`, `ha0_ext`). ⇒ refcv6 and the baselines are scored on **identical clips, windows and GT futures**, each on its own input geometry.
* **S6, SECONDARY** (refcv6's own 6 s horizon) = the S2 windows whose 60-step future is inside the clip. Instants are 1, 2, 3, 4, 5, 6 s (model slots 10–60). For the banked baselines the full 8-slot plan comes from their decision sidecars (`plan_full_nav_true`). The controls are re-integrated on the same instants.

### 3.2 Arms

| arm | what | tier |
|---|---|---|
| `os` | refcv6: one forward, own selection (`out["traj"]`). Inputs: the true v7 nav token, measured v0 at t0, the ego-history window `poses[t:t+W]`, the eval max-speed sidecar value, and the per-clip lift geometry. It is fed **exactly as `compute_losses_v3` feeds it** (`:3396-3565`) | T1 (UNRULED) |
| `os_navshuf` / `os_navzero` / `os_navflip` | nav paired to the wrong window / nav withheld (`nav_cmd=None`) / left↔right. All other inputs are unchanged | T1 |
| `ha` | hold the last observed (a, steer), `--action-units steer` | T1 |
| `ha0` | **constant velocity** at the measured v0 (a = 0, κ = 0). This is the CV control | T1 |
| `ha0_ext` | the echo control (`refav1_arm.hold_ext_controls`) | T1 |
| **`stop`** | **STOP**: zero displacement at every instant | T1 |
| `oracle_sel` | the GT-nearest anchor of this forward's decoded fan. **T0 ceiling only** | T0 |
| refcv4b `os` | banked dev-box dump of `ckpt_40284` (256×640, deterministic decoder) | T1 |
| refcv5-v2 `os` | banked dumps, inference seeds 0 and 1 (256×640, DDIM) | T1 |

Model facts for the baselines come from `MODEL_REGISTRY.md` §4.6 / §4.8 only.

### 3.3 Metrics per family (the same instrument for every arm)

`refcv3_arm.analyze_refcv3` (imported, not copied) per dump, plus this package's additions:

* **ADE / FDE:** L2 over the grid.
* **LONGITUDINAL:** speed MAE, target-speed accuracy @0.5 m/s, along-track MAE, accel. **Distance keeping** (min headway, time-gap, min TTC) uses the banked B1 eval lead block on the shared {1.0, 2.0} s instants, with the TTC-cap censoring count stated.
* **LATERAL:** heading, **curvature (masked estimator, beside the `ha0` straight-line floor)**, yaw-rate, cross-track.
* **TACTICAL:**
  * lat/lon decision accuracy and Cohen's κ, for both refcv6 surfaces: the z_tac v7 heads **and** the v6 behaviour decoder's `tacv6_lat/lon_logits`. Confusion matrices are included; windows are the in-band (±2 s) labelled ones, with n stated.
  * trajectory-derived manoeuvre agreement.
  * anchor selection accuracy vs chance 1/117.
  * **the 22-token goal selection**: per class AUROC, AP, and precision/recall @0.5 against `tac_goal_y/w`, with n_pos and n_neg per class. Classes under n_pos 200 are reported and flagged UNSCOREABLE. Nothing is pooled over tokens.
* **STRATEGIC:** **NOT APPLICABLE, n = 0.** The strategic layer is OFF (`--no-strategic`, SPEC_REFCV6_V2). Its route CE is gated off in training (`refc_v3_train.py:3736`, `_route_on`), so the route head is untrained and **is not scored**.
* **Frozen refcv6 acceptance instruments** (`taniteval/taniteval/refcv6_acceptance.py`, frozen 2026-09-16): T-FLIP (`follows_FED ≥ 0.50`, `true − shuffled ≥ 0.38`), T-ZERO (reported diagnostic), OBEDIENCE (forced 30 km/h on windows whose GT max > 40 km/h; planned max ≤ limit on ≥ 99 %). These are read on milestone checkpoints if compute allows; if not, they are reported as NOT RUN.

### 3.4 Bars. They apply to MILESTONE checkpoints (≥ step 5,000) and NEVER to the kit's step 1000

All bars are on **S2, ADE (m) over the 4 instants, T1**, paired episode-cluster bootstrap, and must hold at **both inference seeds**.

| id | statement | pass iff |
|---|---|---|
| **BAR-R6-1** (primary) | refcv6 beats the **echo control** | `os − ha0_ext` < 0 and the CI upper bound < 0 |
| BAR-R6-2 | beats **hold-action** | `os − ha` < 0, separated |
| BAR-R6-3 | beats **refcv4b** (the programme's best REF-C arm on this surface) | `refcv6.os − refcv4b.os` < 0, separated |
| BAR-R6-4 | beats **refcv5-v2** | `refcv6.os − refcv5v2.os(seed 0)` < 0, separated |
| BAR-R6-5 (secondary, 6 s) | beats the echo control at its own horizon | S6 `os − ha0_ext` ADE 0–6 s < 0, separated |

Per family, the headline metric of each family is also reported paired against `ha0_ext` and against refcv4b, with its CI. These are reported cells with no pass/fail.

### 3.5 VOID gates (a panel that fails one is VOID, not negative)

1. `ha0`'s own curvature is exactly 0, and its tactical κ is exactly 0.
2. `stop`'s displacement is exactly 0, so its ADE equals the mean GT distance.
3. The model-free arms (`ha`, `ha0`, `ha0_ext`, `stop`) are **bit-identical** between the refcv6 dump and every baseline dump: paired delta 0.0, CI [0, 0].
4. `g` is bit-identical across the dumps.
5. `refcv3_arm`'s trivial profile and selection profile are non-degenerate for `os` (at step 1000 a degenerate profile is expected and is recorded, not gated).

### 3.6 The kit checkpoint (step 1000)

It is **PIPELINE VALIDATION ONLY**. The runner is run end to end on it, and G0 is evaluated on it. **No bar is evaluated and no number from it is a result.** RESULT.md says so above its first table.

---

## 4. Runner

`code/run_battery.py --ckpt <path> [--config <path>] [--infer-seeds 0,1]`: GPU gate → G0 (only when a matching in-run eval row exists for the checkpoint's step) → roll (S2 dump, refcv6 arms) → controls and `stop` → baseline subset dumps → analysis per dump → paired panel and bars → `raw/<tag>/` JSONs.

⛔ It never touches Thor, and it runs one GPU job at a time behind the dev-box gate (used < 4,300 MiB, no other python compute, ≥ 8 GB free RAM).

---

## AMENDMENT A1: registered 2026-09-24 ~01:20 Europe/Berlin, AFTER the as-registered G0 read VOID and BEFORE M2–M4 ran

**What happened.**
* The §2 reproduction criteria all held on the kit checkpoint. The as-run verdict listed two COUNT failures, but those came from an implementation bug in the class rule, not from any reproduction term; the bug is recorded in RESULT.md.
* M1 moved **0 of 20 SMOOTH terms** beyond 1 %. M1 is the lift bank without `equalize_bottom_rows`, i.e. `refcv3_arm.py:1101`'s build.
* Per §2, **G0 as registered is therefore VOID**, and it stays VOID in every report.

**A1 adds a NEW gate, G0-A1.** It uses the same 8-seed reproduction data, unchanged. Only the power demonstration is replaced, by three deliberate regressions. Each reproduces a wiring defect that a refcv6 eval loader can actually have. Each runs at inference seed 0 on the same 128 windows and batches.

| arm | the defect it reintroduces | expected |
|---|---|---|
| **M2** | the max-speed input **withheld** (`v_max_valid = 0` on every row): a loader that never wires the eval sidecar and falls back to "no limit known" | at least one SMOOTH `tacv6_*` term leaves 1 % |
| **M3** | the ego-history window **zeroed** (`pose_hist` = 0): a loader that feeds a wrong or empty ego track | at least one SMOOTH or STOCHASTIC term leaves its tolerance |
| **M4** | tactical-goal `pos_weight` and class mask **absent** (`None`): the literal state `refcv3_arm.load_model` leaves the model in (it sets no model-level loss attribute) | `eval_tacv6_goal_bce` leaves 1 % |

**Verdict rule.**
* **G0-A1 PASS** iff the §2 reproduction criteria hold and **at least one** of M2–M4 moves at least one non-COUNT term outside its class tolerance (SMOOTH: 1 %; STOCHASTIC: the 8-seed 99 % PI).
* Every mutation that stays inside is **named in RESULT.md as a wiring defect this gate cannot see**.
* **G0-A1 is VOID** if none of M2–M4 is detected.

---

## AMENDMENT A2: registered 2026-09-26 ~10:00 Europe/Berlin. The step-5000 G0 read FAIL on the WRAPPER CONTROL ONLY; A2 was written BEFORE any measurement below ran

**What happened.**
* At step 5000 (`ckpt_5000.pt`, md5 `8a1e4da0…`), G0-A1 read **FAIL**, and the only failing clause is the §2 wrapper control: 3.921e-3 against its 1e-3 bar. The worst terms were `box3d_yaw` 3.9e-3, `goal2s_err_m` 2.0e-3 and `goal_tac` 1.5e-3.
* Every one of the 82 reproduction terms was within its class tolerance (SMOOTH median 0.096 %, MATCHED 0.19 %).
* M2, M3 and M4 were all detected (4 / 7 / 4 terms).
* At step 1000 the same control read 8.4e-4.
* **That verdict stands as registered: G0-A1 at step 5000 = FAIL.**

**Why the §2 control is ambiguous.** It compares a plain batch-4 forward with a micro-batched [1,3] one, both under the run's own precision (bf16 trunk, cuDNN TF32 on). A different micro-split changes the kernel batch composition. The control therefore measures **merge correctness plus batch-composition numerics**, and §2 bounds the second by nothing: 1e-3 was an unmeasured choice.

**A2 re-defines the wrapper control so that it isolates the merge.**
* **Batch:** the first 4 windows of the first in-run eval batch, with the DDIM `eps` zeroed in every arm.
* **Three precision conditions:**
  * **P1** `as_run`: backends as G0 ran them (cuDNN TF32 on, matmul TF32 off, cuDNN non-deterministic, benchmark off); trunk bf16 + NHWC.
  * **P2** `cudnn_det`: cuDNN TF32 off, `cudnn.deterministic` on, benchmark off; trunk bf16 + NHWC.
  * **P3** `fp32_det`: P2, plus matmul TF32 off, plus the trunk in **fp32** (bf16 and NHWC off). If P3 does not fit on the card, the backbone chunk drops to 4 for **every** P3 arm, and that is recorded.
* **Five arms per condition:**
  * `plain`;
  * `repeat` (plain again);
  * `permuted` (the 4 rows reversed in every per-row input; the batch-level loss is permutation-invariant by construction);
  * `micro[1,3]`;
  * `micro[3,1]`.
* **Per term** (non-excluded, |plain| > 1e-6):
  * `Floor_P = max rel(|plain − repeat|, |plain − permuted|)`. This is what the arithmetic alone does when the batch composition changes, with no wrapper involved.
  * `Wrapper_P = max rel(|plain − micro13|, |plain − micro31|)`.
* **Deliberate regressions of the wrapper**, all under P3:
  * **W1**: 0-dim and float outputs merged as micro-part 0 only, instead of the row-weighted mean.
  * **W2**: the micro-parts concatenated in REVERSED order.

**The A2 wrapper clause.**
* It **PASSES** iff, under **P3**, every term has `Wrapper ≤ max(3 × Floor, 1e-5)`, **and** W1 and W2 each exceed that same per-term bar on at least one term. Otherwise the control has no power, and it is VOID.
* P1 and P2 are reported, because they attribute the step-5000 3.9e-3 to precision or to merge. They do not gate.
* **G0-A2** = G0-A1 with the §2 wrapper clause replaced by the A2 clause. Everything else is unchanged: the reproduction tolerances, the 8-seed PI, and M2–M4.
* If the wrapper exceeds its bar under P3, the wrapper has a **merge defect**. Then:
  * G0-A2 = FAIL;
  * the term and its merge rule are named;
  * the wrapper is fixed and the reproduction re-run before any battery number.
* **A2 applies to every milestone from step 5000 on.** The step-1000 probe is run too and reported. **Both verdicts are always shown:** G0-A1 as registered, and G0-A2.
