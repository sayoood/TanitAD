# Where we stand — refcv6, refav1, flagship v7 · 2026-09-15 14:45 (Europe/Berlin)

**Master Mind, for the PI** (*"I need a detailed report where we stand in refcv6, refav1 and flagship v7"*).

**Sources.** Repo at origin tip `1638b54`. `MODEL_REGISTRY.md` and raw eval JSON are quoted; prose only where no registry row exists, and then said so. Four read-only research passes cross-checked the registry, `GOALS_AND_CLAIMS.md`, the pre-registrations and the packages. I re-read the decisive numbers from source myself (marked ✔).

**Tiers.** T1 = self-action **open** loop (PI ruling 2026-09-02); nothing below is closed loop. Intervals are the paired episode-cluster bootstrap unless stated.

**Fleet state, MEASURED today.** No training runs anywhere. Thor is producing the SAM3 corpus maps (128/4,719 clips at 12:43, ETA ~2026-09-22 18:30). Pods `pod`, `pod3`, `pod4`, `pod5`, `eval`, `a40` all refuse or time out on SSH; the HF Space is not running.

---

## 0. The short version

1. ⛔ **20 days to the first final evaluation (5 Oct, Mission Plan P7).** The minimum there is "at least very successful phase 0" — a running architecture on real data, "ideally both in open loop and closed loop" (`Mission Plan.md:143,148,181`). **No line has a closed-loop result, and no trained arm beats the hold-action control on the longitudinal family at T1.**
2. **refcv6 has no result.** Built and gated 2026-09-10/11. V0 ran on Thor to **step 250 of 40,284** and was **stopped by your directive of 2026-09-11**; no checkpoint (✔ `PI_DECISION_QUEUE.md:19-40`). The "carries every DiffusionDrive piece" claim was **retracted**: the perception grounding is missing. Its design is what §5 of the companion document finishes.
3. **refav1 is finished and parked.** Training completed 21,109/21,109 (2026-09-04). At T1 its planner **emits an injected do-nothing baseline on 282/282 windows** and never turns (✔ `MODEL_REGISTRY.md:2129-2137`); `cl − ha0` ADE **+0.0158 [+0.0007, +0.0315]**, separated **worse**. No open PI item, no RL work, not on any 5-Oct track.
4. **Flagship v7f was never trained.** The launch line does not run (two flags missing), your hold of 2026-08-31 stands, and the v7-tiny rig shows the model **ignores its actions** — 0 of 32 arms change prediction by more than 0.1 % when actions are replaced by noise (✔ `MODEL_REGISTRY.md:4955-4957`). Its only T1 read is void (decoded through an untrained readout). Not on the 5-Oct path.
5. **The one line that drives is REF-C** (refcv4b ties the controls; refcv5-v2 fails both bars but is the best laterally). The measured failure is **longitudinal and grounding**, and both now have supervision available for the first time: SAM3 maps (in production, corpus-wide), LiDAR BEV ground truth (315 clips), obstacle tracks (4,585 clips), v8.1 max-speed labels.

---

## 1. refcv6

### 1.1 What it is

| | |
|---|---|
| definition | a **configuration** of `stack/scripts/refc_v3_train.py`, not new model code (`PREREG_REFCV6.md:15-20,67-77`) |
| model | identical to refcv5-v2: **108,257,502 params** — core 106,067,312 · `tac_goal_tok_head` 11,286 · `str_goal_head` 771 (`MODEL_REGISTRY.md:3391`) |
| inputs (inference) | front-wide 120° camera, 256×640 cylindrical, 3-frame stack · measured ego state at t0 (dropout 0.5) · oracle v7.2 nav command · **no max-speed channel** in the staged arms |
| decoder | 117 v0-conditioned `alat` anchors · control-space DDIM sampler (`--w-u0 0.5`) · P14 emitted-fan selection · strategic route head · 22-token tactical head · `--agents off` |
| base | refcv5-v2's 65-token argv **plus `--agent-join`** in every arm — so V0 is a fresh control, not a replication of refcv5-v2 (`PREREG_REFCV6.md:93-110`) |

**Arms as pre-registered** (`PREREG_REFCV6.md:133-144`), each with a seed-1 replicate:

| arm | one lever | pairs against | status |
|---|---|---|---|
| V0 | — (control) | — | ran to step 250, **stopped** |
| D | `--w-u0 0.5 → 0` (DiffusionDrive has no denoising loss) | V0 | runnable behind `--ack-ddim-no-u0` (your ruling 09-11) |
| A | `--agents head` (DD's learned detector, Hungarian loss) | V0 | ⛔ failed its two-seed gate at 17 M; untested at 108 M |
| B | `--wp-index on` (trajectory-indexed attention into agent tokens, +2,208 params) | A | built, mutation-proofed, **never run** |
| C | `--w-tac-goal` (the head that took zero gradient in refcv5-v2) | V0 | sweep recommends **0.05**, not ratified (queue item 10) |

**The bar, executable and drop-proof** (`verdict_refcv6.py`; `PREREG_REFCV6.md:294-307`): beat **both** `ha0_ext` and `ha` on ADE, separated **and** ≥ 0.10 relative margin, ≥ 3× the same-panel replicate floor; **no separable lateral regression** (heading, yaw-rate, cross-track, masked curvature); strategic route accuracy not separably worse with n > 0; longitudinal and tactical families **reported**. MEASURED proofs: one-variable check kills 6/6 mutants, verdict blocks 13/13 dropped clauses, coverage gate passes 0.968 / fails 0.040.

### 1.2 What actually ran

| date | event | evidence |
|---|---|---|
| 09-10 | build proofs on CPU; arm D found blocked by a trainer refusal whose stated reason was measured false (`control_head` gets gradient 9.39e4) | `REFCV6_COMPUTE_PLAN.md:138-174` |
| 09-11 | arm C weight sweep, nine 400-step arms on Thor: **0.05** is the largest weight passing all three criteria; it also found arm C would have **died at step 500** (eval set never got its targets) — fixed | `…/2026-09-11-tacgoal-weight-sweep/RESULT.md` |
| 09-11 | V0 launched on Thor at the full 40,284-step budget after the gate caught three launch-killing defects (eval join, `metrics.jsonl` name, a same-basename 3.98 %-coverage join) | `GOALS_AND_CLAIMS.md:11417-11418` |
| 09-11 21:46 | **stopped by PI directive at step 250**, ~33 GPU-min, no checkpoint | ✔ `PI_DECISION_QUEUE.md:19-40` |

**Measured rate on Thor: 4.086–4.102 s/step ⇒ ~46.8 h per 40 k-step arm** (`PI_DECISION_QUEUE.md:64-65`).

### 1.3 The evidence it rests on — refcv5-v2 vs refcv4b vs the controls (MEASURED, T1, 4,823 windows / 141 episodes)

Source: `…/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/raw/` (panel `.txt` + per-arm JSON), as re-read into `PREREG_REFCV6.md:583-593`.

| family | metric | refcv5-v2 | refcv4b | `ha` hold-action | `ha0_ext` |
|---|---|---|---|---|---|
| ADE | m | 0.3079 | 0.2965 | 0.2996 | **0.2874** |
| **LONG** | speed MAE (m/s) | 0.2919 | 0.2900 | **0.2540** | **0.2540** |
| **LONG** | along-track (m) | 0.2655 | 0.2544 | 0.2348 | **0.2341** |
| **LONG** | accel MAE (m/s²) | 0.3596 | 0.4346 | **0.3166** | **0.3166** |
| LAT | heading (°) | **1.2121** | 1.2964 | 1.5489 | 1.4322 |
| LAT | yaw-rate (°/s) | **1.0534** | 1.7368 | 1.4542 | 1.3395 |
| LAT | curvature ÷ straight-line floor | **0.512×** | 1.198× | 0.593× | 0.546× |
| STRAT | route accuracy (n 3,622) | 0.7708 | **0.7791** | — | — |

| bar | refcv5-v2 | refcv4b |
|---|---|---|
| `os − ha0_ext` | **+0.0205 [+0.0043, +0.0390]** ⛔ separated worse (seed 1: +0.0204) | +0.0091 [−0.0055, +0.0254] tie |
| `os − ha` | +0.0082 [−0.0082, +0.0267] tie | −0.0032 [−0.0179, +0.0133] tie |
| refcv5-v2 − refcv4b | **+0.0114 [+0.0059, +0.0172]** — refcv5-v2 separably worse | |

⇒ **Hold-action beats both trained arms on every longitudinal metric; lateral is the only family where training clearly pays.**

⭐ **Two findings in the raw JSON that no steering document carries yet** (✔ read today):

* **refcv5-v2 barely uses its nav command.** `os − os_navzero` ADE: refcv5-v2 **−0.0059 [−0.0125, +0.0005]** (seed 1: −0.0070 [−0.0137, −0.0001]) against refcv4b **−0.0961 [−0.1102, −0.0813]**. The diffusion/selection pipeline lost the conditioning refcv4b had — against your mandate that nav and max speed actually be used.
* **Selection headroom:** `oracle_sel − os` = **−0.0775 [−0.0937, −0.0635]** (stamped `T0minusT1`: an oracle pick from the emitted fan against the model's own pick — the right trajectory is in the fan more often than it is chosen).

**Dead weight, MEASURED from the checkpoint:** 18,472 parameters never trained — `offset_head` 6,160 (bypassed by the DDIM sampler, expected), `tac_goal_tok_head` 11,286 (loss weight defaulted to 0), `scorer.goal_point` 1,026 (exactly zero in refcv3, refcv4b and refcv5-v2). A post-training gate now catches this class.

### 1.4 The grounding evidence (since 09-11)

| probe | result | evidence |
|---|---|---|
| BEV transformer on the **frozen** refcv5-v2 trunk, LiDAR BEV GT (`E-BEVHEAD-FROZEN-1`, pre-registered) | ⛔ **failed all four bars**: test AP 0.4140 vs zero-information 0.3801 / pixels 0.3849. Levers: stride-16 tokens, azimuth-aligned attention, unfrozen last stage — **no effect**; **+181 train clips** — AP **0.4352 [0.3976, 0.4737]**, clears two bars on two seeds; the 0.60 usefulness bar is not met | ✔ `…/2026-09-13-bev-lidar-corpus-and-head/RESULT.md:18-32` |
| WP-D: BEV auxiliary loss while training REF-C (4 k steps, Thor) | after its replicate arms: the aux term **costs lateral accuracy** (heading 3.3×, curvature 6.0×, yaw-rate 3.5× the replicate floor); ADE and longitudinal effects were rig noise; a **shuffled** target trains to the same AP as the real one (0.1525 vs 0.1527); overall **underpowered at 4,000 steps** | ✔ `…/2026-09-07-wpd-bev-aux/PANEL_RESULT.md:22-48` |
| WP-A: readout localisation | the stride-32 token grid (8×20, 6°/column) loses about two-thirds of the available localisation even under a perfect front-end | `…/2026-09-07-wpa-readout-localisation/RESULT.md` |

⇒ **The trunk carries little metric scene structure, reading it out frozen does not work, and the only lever that moved it was more labelled data.** Joint training (`E-BEVLIDAR-CAP-1`) is pre-registered and never run.

### 1.5 Retracted, open, and yours to decide

**Retracted:** "refcv6 includes every DiffusionDrive V1/V2 piece" (perception grounding missing) · "refcv4b has no strategic output" (the column was the `ha` control; the strategic reference is now refcv4b's **0.7791**) · "11,286 dead params" (→ 18,472) · "arm D is free and runnable" · "`--w-agent 1.0` is not pre-registered" (it is, P1 gate).

**Open decisions** (`PI_DECISION_QUEUE.md`): the refcv6 stop (default: stays stopped) · item 13 compute (default one A40, 94 GPU-h) · item 10 tactical weight (0.05 recommended) · items 9/11 agents head and WP-C (default: nothing launches) · 13-Sep report #4 track and #5 closed loop for 5 Oct (defaults: perception only; open-loop panel + videos).

---

## 2. refav1

### 2.1 What it is

| | |
|---|---|
| type | latent world model + test-time iCEM/MPC planner |
| encoder | **frozen DINOv3 ViT-L/16**, fp8-cached features [T, 640, 1024] + WideAdapter |
| predictors | operative 0.2 s × 30, tactical 0.6 s × 10, strategic 1.5 s × 4, FiLM-linked; goals enter the planning cost |
| planner | 300 samples × 30 iterations × 30 elites, K = 10 at 0.2 s; **injects three baselines into its own population** (constant velocity, hold v0, decel 1.5) |
| params | **182,459,701** in the checkpoint (✔ `MODEL_REGISTRY.md:2089`) |
| inputs | fp8 DINOv3 features · oracle v7.2 nav · measured v0 at t0 as a third action channel (PI ruling 09-02) |
| corpus | B1 / v7.2, 168,910 windows over 4,572 episodes |

### 2.2 Training — complete

| run | outcome |
|---|---|
| `refav1-b1-v72-1ep-21109` (09-02) | representation collapse / gradient blow-up; relaunched with frozen targets; **retired at step 1,000** (drift alarm) |
| `refav1-b1-v72-ep2-ema-bf16` | 6.1× faster (bf16 + TF32, EMA teacher); **retired at 6,850** — speed never reached the model |
| `refav1-b1-v72-ep3-speed` (09-03) | ✅ **COMPLETE 21,109 / 21,109** (09-04), 0 non-finite rows, participation 15.95 → 22.79; checkpoint on Thor (md5 `1189bc02…`) and a dev-box copy |

⚠️ Trained from Thor's older code (`refa_v1.py` 1,885 vs 3,030 lines at HEAD) — a re-run from HEAD would not be a replication.

### 2.3 Evaluation (MEASURED, T1, 282 windows / 141 episode clusters, full v7.2 eval split)

| | ADE (m) |
|---|---|
| `cl` (refav1's planner) | 0.5474 [0.4839, 0.6108] |
| `ha0` constant velocity | **0.5316** |
| `ha` hold action | 0.5391 |
| `ol` (T0, teacher-forced) | 0.4237 |

* `cl − ha0`: ADE **+0.0158 [+0.0007, +0.0315]** ⛔ worse, separated; speed MAE +0.0308 [+0.0049, +0.0585] worse; every lateral delta **exactly 0.0000**.
* ⛔ **Why** (✔ `MODEL_REGISTRY.md:2129-2137`): κ is identically zero on **282/282** windows, acceleration constant on 282/282, and **only 2 distinct plans** were emitted in total — `a = 0` (270) and `a = −1.5` (12) — **both baselines the planner injects into its own population.** The world model is never allowed to win the cost.
* What the weights do show (T0): the world model beats persisting the last feature field from 1.0 s (+0.1129) to 6 s (+0.2454); it loses at 0.2 s.
* p4 turn-dense panel (40 windows / 8 clusters, a different denominator): cost variant `wk15` reaches parity with `ha0_ext` (+0.0162 [−0.1648, +0.1980]); `lonshift` improves speed against `ha0` but is still worse than `ha0_ext` on speed (+0.2599 [+0.1290, +0.4265]).
* **Inference seed matters:** iCEM samples, so the same checkpoint differs run to run; measured ADE floors +0.06 to +0.30 m depending on panel. Any refav1 A/B needs a second `--plan-seed`.
* Closed loop: never run.

### 2.4 Status and defects

* **Parked, not failed by a bar of its own design:** the shipped cost's optimum *is* the do-nothing plan; the latent carries **no closing-rate information**; distance keeping works only with the ground-truth gap; left turns are suppressed by the curvature penalty; the seed/goal horizon mismatch fix exists but is off by default.
* **No PI decision item.** The RL work of 09-10/11 (≥GT bar, 2.24× reward efficiency, T0) was all on **REF-C**, not refav1.
* Next steps on record (none scheduled): a κ cost-landscape sweep and cost-side longitudinal levers.

---

## 3. Flagship v7

### 3.1 What it is

| name | what |
|---|---|
| **v7** (recipe, 2026-08-26) | replaces v1.x's self-generated prediction target: O5 L1 rollout + O6 SIGReg, EMA target τ 0.996, conditioning `omega_accel_v`, trainable encoder. It exists because v1.7's S-curve read 0.9785 teacher-forced but **0.0430 at T1** |
| **v7r** (proposal, 08-27) | four brains: trainable distill-init ViT operative model, Alpamayo-trained tactical head, strategic options, anchored-diffusion planner, sub-300 M |
| **v7f** (pre-registered 09-03, `H-V7F-1`) | v7 core + v7r upper levels + trainable DINOv3 ViT-B/16 trunk + LDAD loss + o5_k 60; launch line builds **245,588,568 params** (MEASURED) |
| **v7-tiny** | the validation rig on the real trainer: 19.3 M params, 4×8×64 readout, parity corpus, stage S-W, planner losses at 0 |

### 3.2 Training state

* ⛔ **v7f has never been trained** — no checkpoint, no run directory, no registry row.
* ⛔ **The launch line does not run:** the LDAD flags do not exist and `--horizons` is unset, so the trainer refuses to start (`…/2026-09-05-v7f-state/raw/v7f_launch_line_dryrun.json`). It also passes no `--readout-grid-w`, so the 4×8 readout would silently revert to 4×4.
* ⛔ **Your hold of 2026-08-31** — training waits *"until the remaining problems are solved"* (`V7_LAUNCH_GATE.md:3-4`).
* v7-tiny: 30 k-step arms on Thor (`rdw8p30k`, `postrain30k` ± seed/freeze, `emao14_30k` ± τ-ramp, …); o5_k 60 diverged (grad norm 2.1e9); `o11p30k` stopped by its own degenerate-solution rule.

### 3.3 Results (T0 diagnostics, single seed, v7-tiny)

| gate / probe | result |
|---|---|
| **action use** | ✔ **0 of 32 arms** change prediction nrmse by more than 0.1 % when every action is replaced by noise (max 0.0919 %) |
| action sensitivity | ~**97 % of the response comes from the speed channel**; the command alone contributes almost nothing |
| L3: does the predictor add anything over z_t? | ⛔ **FAIL on all four arms** (max \|t\| 1.08 against a 2.9 bar) |
| participation (val) | 25.6 (`rdw8p30k`) … 6.4 (`splitp30k`); the 8.56 G-RANK bar is **inadmissible** (no instrument reproduces it) |
| decode | `splitp30k` n_agents within-clip r +0.3881 vs DINOv3 +0.2754, but lead range −0.1611 (worst of any arm) |
| **T1** | ⛔ **void** — the only read (ADE ~14 m, 3 arms) decoded through a `step_readout_op` that was never trained and is bit-identical across arms |

### 3.4 Status

* Launch-gate items: P1 undecidable, P2 (the model ignores its actions) **open**, P5 **FAIL**, P4 open.
* PI decisions pending but **not in the queue**: lift the hold for an (a_lon, a_lat) command arm; PRE D1–D9; pull the ViT-B/16 weights.
* **No document names v7 or v7f for 5 October.** The 13-Sep plan runs entirely on REF-C.

---

## 4. The three lines side by side

| | refcv6 | refav1 | flagship v7f |
|---|---|---|---|
| trained | ⛔ no (V0 stopped at 250) | ✅ complete (21,109) | ⛔ never |
| params | 108.3 M | 182.5 M | 245.6 M (built) |
| best T1 read | refcv5-v2 base: lateral wins; fails ADE bars; hold-action wins longitudinal | worse than constant velocity; plans are injected baselines | none (void) |
| perception grounding | ⛔ missing (the open design question) | frozen DINOv3, no BEV | none |
| blocker | your stop; compute; design (grounding) | design of the cost; no item | your hold; launch line broken; ignores actions |
| on the 5-Oct path | ⭐ **yes — the only driving line** | no | no |

---

## 5. Stale records found (to be corrected; none changes a conclusion above)

1. `LEADERBOARD.md` still shows refav1 "TRAINING" / "final read PENDING" and refcv3 PENDING; the registry rows are final.
2. `MODEL_REGISTRY.md` has **no refcv6 row**; its refav1 `ccos` row still reads "IN PROGRESS" (it landed and was refuted); the v7-tiny T1 table still reads "CLOSED-LOOP" and carries the void numbers without a caveat.
3. `verdict_refcv6.py` still holds the mislabelled "refcv4b" literals (really the `ha` control) and route 0.7708; the prereg says to score against the re-read values.
4. `PI_DECISION_QUEUE.md` item 10 still says "being measured"; items describe the A40 as occupied and refcv5-v2 as training; item 14's default predates your switch to SAM3 maps.
5. The 13-Sep report calls Thor idle (now SAM3 until ~22 Sep); `alpasim` / `nurec_scenes` were archived off Thor on 09-14 (the closed-loop feasibility doc still lists them there).
6. The 14.3 % false-positive replicate rig is credited to v7-tiny; it was the REF-C tiny rig (17.0 M).
7. Your D: checkout is 34 commits behind origin.

---

## 6. What this means for the next 20 days

* **Only REF-C can carry 5 October.** refav1's planner cannot beat its own injected baseline without a cost redesign; v7f has no runnable launch line and a model that ignores its actions.
* **The two measured failures of the REF-C line are exactly what DiffusionDrive's missing pieces address:** placement/speed (longitudinal) and environment grounding. And for the first time the supervision exists at corpus scale: SAM3 semantic maps (all clips by ~22 Sep), LiDAR BEV ground truth (315 clips, an independent check), obstacle tracks (97 % of clips), v8.1 max-speed labels.
* ⛔ **Compute is the binding constraint:** Thor is busy until ~22 Sep and every pod is gone. A 40 k-step arm is ~47 h on Thor. Without a pod, at most ~5 arms fit between 23 Sep and 5 Oct — and closed loop needs Thor too.

The companion document **`Project Steering/REFCV6_DESIGN_GROUNDED.md`** finishes the refcv6 design on this basis.

---

## ⛔ CORRECTION (2026-09-16)

The arm-D row above describes `--w-u0 0.5 → 0` as *"DiffusionDrive has no denoising loss"*. **That is misleading.**
DiffusionDrive's matched-anchor L1 **is** its x0 denoising loss (`multimodal_loss.py:133-159`, reg weight 8 with
focal 10, summed per cascade layer and scaled by `trajectory_weight` 12); `diff_loss_weight` 20 multiplies a
`diffusion_loss` key the V1 head never emits, so it contributes nothing. Our `--w-u0` is a **second copy** of DD's
loss in control space, so arm D removes a duplicate rather than adding a paper feature.

Sources and the full correction list: `Project Steering/REFCV6_CLARIFICATION.md` §9 and §5; class entry
`RETR-2026-09-15-DD-PAPER-OVER-CODE` in `Project Steering/RETRACTION_LOG.md`.
