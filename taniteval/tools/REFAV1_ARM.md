# REFAV1_ARM — the T0 / T1 eval adapter for REF-A v1 (`refav1`)

**Status (2026-09-02):** plumbing VALIDATED on a random-init `RefAV1` + a synthetic
3-episode eval slice of the right shapes/dtypes (`stack/tests/test_refav1_arm.py`,
15 tests; the untouched `t1_eval.py --analyze-only` CLI also reads the dump).
⛔ **UNVERIFIED on a real checkpoint** — this box may not contact Thor; the
first real read is the Master Mind's, at the step-1000 pull. Every number in this
file is MEASURED on the synthetic fixture unless marked otherwise.

Files: `taniteval/tools/refav1_arm.py` (the tool), `stack/tests/test_refav1_arm.py`
(the pin), this note. `taniteval/tools/t1_eval.py` is **not modified** and no
patch to it is needed — see "Why a sidecar".

---

## 1. The `t1_eval.py` contract this adapter targets (read end to end, 1,631 lines)

| item | what the code does (file:line) |
|---|---|
| dump | per-episode `ep{fi:03d}.npz`: `g [N,K,2]` GT ego-frame waypoints (x forward, y left) at `dt`; arms `cl` (T1) / `ol` (T0) / `ha` (T1) `[N,K,2]`; `ws [N]` provenance; `eid`, `clip_index`, `v0` are **metadata**, not arms (`_META_KEYS`, t1_eval.py:163) |
| arm enumeration | **every non-suffixed key that is not `g`/`ws`/meta is an arm and MUST resolve to a T0/T1 stamp** (`resolve_tiers`, :301) — an unstamped key is a hard error; extra keys of any other shape break `analyze` (`shape != GT`, :405) |
| tiers | `DEFAULT_TIERS` (:145): `cl/c16/c6/ha/h16` = T1, `ol/o16/o6` = T0; `--tiers name=T0\|T1` adds stamps |
| `analyze()` (:317) | pass 1 per-episode numpy: `fam_row` (VERBATIM `analyze_cl` port — mean-of-episode-means, **reproduction only**), S-curve halves/hits, xcorr lag, event response; pass 2 pooled: `four_families.all_families(win, tactical_from_traj=True, tier=t)` + `ci.bootstrap_metrics` (ADE/FDE/LON/LAT) + paired decision-grade contrasts on `ade_m`, `speed_mae_mps` |
| the `win` dict (:483) | **a whitelist, not a view of the dump** — only `pred_dense/gt_dense/pred/gt/wp_steps/dense_steps/dt_s/eid/v0/lead` reach the families. Anything else in the npz never reaches `all_families` |
| `--with-hold-action` (:1254) | `faH = aw[:, -1:]` expanded over K: the action recorded **at the last observed frame**, held — consumes no recorded future; tier T1 (control) |
| `decode_open` (:731) | unicycle head over pre-rolled transitions: `rows = [v·dt, 0, yaw·dt]`, `v ← clamp(v + a·dt, 0)`, `accumulate_se2` |
| `roll_closed` (:753) | T1: predictor step → head → `(a_j, yaw_j)` → `steer = atan(L·yaw/max(v,0.3))` fed back as the next action; **`ego` = `v0 / SPEED_SCALE` appended as a CONSTANT extra action channel for the whole roll** (:782) — i.e. only the t0 measured speed, never a future speed |
| estimator | full-set pooled point; episode-cluster bootstrap intervals; paired across arms on the same windows; `overlapping_holdout_se` nowhere |
| STRATEGIC | `all_families` leaves it UNAVAILABLE on a trajectory-only dump (`_decision_family` needs `route_pred/route_gt` in `win`, which the whitelist never forwards) |

## 2. Why a sidecar (and why no patch)

The decision heads' outputs, the labels and the nav tokens are per-window
**integers**, not `[N,K,2]` trajectories; putting them in `ep*.npz` would make
`analyze()` enumerate them as arms and refuse. So the roll writes:

```
<dump>/ep{fi:03d}.npz          exactly the t1_eval contract  -> t1_eval.analyze (imported, untouched)
<dump>/decisions/ep{fi:03d}.npz  the refav1 sidecar             -> refav1_arm.analyze_refav1
<dump>/manifest.json           episodes/grid/tiers/plan cfg/model provenance/join report
```

`t1_eval.py --analyze-only <dump> --tiers cl_navshuf=T1,cl_oraclegoal=T0 --dt 0.2`
works unchanged (test-pinned). `refav1_arm.py --analyze-only <dump>` produces the
full record (t1 block + `rec["refav1"]`).

## 3. Arms and tier stamps

| arm | tier | what it is | consumes after frame 2t? |
|---|---|---|---|
| `cl` | **T1** | ONE `plan()` per window at t0 with the TRUE nav token; iCEM rolls the operative/tactical predictor under the **planner's own candidate actions**; trajectory = `unicycle_paths(controls, v0_measured)` | **no** (test-pinned bit-identical under future perturbation) |
| `cl_navshuf` | **T1** | as `cl` with `nav_cmd` permuted across the eval windows (among nav-valid ones) — the D-REFAV1-NAV-DEPTH obligation | no |
| `ha` | **T1** | hold-action control: the last **closed** action `(a, κ)` spanning cache steps `[t−1, t]` = `(v[2t]−v[2t−2])/0.2, kap[2t−2]` held K steps | **no** (pinned) |
| `ha0` | **T1** | ⭐ constant-velocity control: `a = 0, κ = 0` at the measured `v0` — a straight line at constant speed. **The strongest trivial baseline** and the bar the echo test is actually run against (§3a). Consumes strictly less than `ha`: not even the last observed action | **no** |
| `ol` | **T0** | the RECORDED future `(a, κ)` integrated from v0 — for refav1 the **kinematic-contract control**, not a WM diagnostic (see §5) | yes (by definition) |
| `cl_nonav` | T1 (opt-in) | `plan()` with `nav_cmd=None` → index 0 "follow" | no |
| `cl_oraclegoal` | **T0** (opt-in) | `plan()` with the TRUE future field at t+plan_steps as `goal_field` — a goal read from the future is future information | yes |

T1 definition used (EVAL_DOCTRINE): *predictor consumes the planner's own actions;
perception fixed at t0.* The 2.0 s plan horizon (`cfg.plan_steps` = 10 × 0.2 s) equals
the eval horizon (`--horizon-k 10`), so one MPC tick covers the scored path.
`--horizon-k` beyond `plan_steps` is refused (no action exists past the plan).

**v0 rule (PI 2026-09-02):** `v0 = poses[2t, 3]` — the speed MEASURED at t0 — is
the only ego state used; `plan(v0=)`, `forward(v0=)` and the unicycle all integrate
`v_k = v0 + Σ_{j<k} a_j dt` from the arm's OWN actions. The speed channel
(`RefAV1Config.speed_channel`) is derived by the MODEL (`augment_actions`, which
refuses a pre-widened tensor); the adapter passes 2-wide `(a, κ)` + `v0` everywhere,
so there is one convention in the programme, not two.

## 3a. `ha0` and the TRIVIAL-PROFILE INSTRUMENT — why an arm's SHAPE is read before its metrics

⛔ **THE FAILURE THIS EXISTS TO PREVENT, MEASURED 2026-09-03.** The first real T1 read of refav1
(step 1,000, 140 windows) shipped a paragraph saying *"lateral planning already beats holding"* —
heading −2.0°, cross-track −18 cm, yaw-rate −0.048 rad/s, every interval separated, every number
correctly computed. **Every one of the 140 `cl` plans was a straight, constant-speed line.** The
"gain" was a straight line beating `ha`'s held, noisy κ. `cl` was also **bit-identical** to
`cl_navshuf` on 122/140 windows and to `cl_oraclegoal` on 96/140 — two "controls" that measured
nothing on most of the slice. Nothing in a four-families table can say any of that, because a
family table answers *how far off* and never *what shape*.

**Two additions close it, and both are ON by default:**

1. **`ha0`** (`hold_v0_controls`, tier T1): `a = 0, κ = 0` at the measured `v0`. It is trivial **by
   construction** and therefore cannot be broken by a data defect — which matters, because `ha` can:
   with the loader's current κ channel (see §3b) `ha` over-rotates by ~2.9× and reads *worse* than
   `ha0`, and the November-fresh reader concludes "planning beats holding". Repaired, `ha` beats
   `ha0` on both channels. ⇒ **`ha` is not the trivial floor. `ha0` is.** The paired block
   `paired_cl_minus_ha0` is emitted beside `paired_cl_minus_ha`.
2. **`trivial_profile()`**, printed **BEFORE any family row** and banked first in the record
   (`rec['refav1']['trivial_profile']`): per arm the fraction of windows that are straight
   (`max |y| < 1e-6`), constant-speed (chord-length spread `< 1e-4` m), **both** (the
   constant-velocity profile), and the count of windows where it is **bit-identical** to each other
   arm (`< 1e-9` m). Any arm at `trivial_frac > 0.5` is listed in `degenerate_arms` with the
   sentence *"read their LATERAL rows as a control, never as planning skill"*.

On the banked step-1,000 dump it prints, before any metric:

```
  cl             n= 140 straight=1.0000 const_speed=1.0000 CONSTANT-VELOCITY=1.0000  identical_to: cl_navshuf=122/140, cl_oraclegoal=96/140, ...
  ⚠️ CONSTANT-VELOCITY on > 50 % of windows: ['cl', 'cl_navshuf', 'cl_oraclegoal']
```

Pinned by `stack/tests/test_refav1_kin_contract.py` (A1–A6): `ha0` straight and constant-speed on
every window; `ha` not, where the observed κ ≠ 0; the instrument reads 1.0 for `ha0`; two arms that
ARE equal are reported as equal; `ha0` reaches the record as T1 with families and the paired block;
and no existing arm moves (each is rederived from its own documented rule and matched bit for bit).

⚠️ **`t1_eval.DEFAULT_TIERS` does not know `ha0` yet**, so the **standalone** `t1_eval.py
--analyze-only` CLI refuses a dump containing it (`t1_eval.py:307`, and that guard is right).
`refav1_arm.py`'s own analysis path is unaffected. Pass `--tiers ha0=T1`, or land the one-line
`"ha0": "T1"` addition in `DEFAULT_TIERS` — see §8.7.

## 3b. ⛔ THE `ol` / `ha` ARMS ARE CURRENTLY MIS-SCALED — the κ channel is a STEERING ANGLE

MEASURED 2026-09-03, 140 windows / 20 episodes, 0 GPU
(`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refav1-kinematic-contract-lateral/`,
tool `taniteval/tools/refav1_kin_contract_probe.py`):

`physicalai.signals_at` (`physicalai.py:620-630`) writes `steer = atan(L·κ)` — a **road-wheel
angle** — into `v2ep actions[:, 0]`. `refav1_loader.py:264` reads it **as κ**. Every curvature is
therefore inflated by ~**L = 2.9×**, and:

| | `ol` LAT curved (72) | LAT straight (68) | LON all (140) |
|---|---|---|---|
| as shipped | **0.7156** (worse than a straight line's 0.4960) | 0.0685 | 0.2395 |
| with `κ = tan(steer)/2.9` | **0.0600** | 0.0141 | 0.1199 |
| pose-yaw FLOOR | 0.0527 | 0.0096 | 0.1189 |

Integrated-vs-pose yaw ratio **×2.870 → ×0.995**; yaw RMSE **19.81° → 0.63°**. Sign, one-step
timing, mid-frame, integration order and the GT frame are all REFUTED in the same panel.
⇒ **Until the repair lands, every LATERAL row of every refav1 read is off by a known 2.9×
over-rotation.** The fix is PROPOSED, not applied — it changes the live Thor run's training input
distribution and the units of `canonical_controls` / `PlanConfig.kappa_max`. See RESULT.md §6 and
§8.7 below.

## 4. What the record contains

`rec["arms"][arm]` — from `t1_eval.analyze`, unchanged: `four_families`
(LONGITUDINAL incl. `anti_echo` on the dumped `v0`; LATERAL; TACTICAL
trajectory-derived via `refc_tactical.factor_from_kinematics`; STRATEGIC
UNAVAILABLE-by-design), `intervals` (ADE/FDE/LON/LAT, episode-cluster bootstrap),
`s_curve`, `lag`, `response`, `legacy_epmean_row`. The `_protocol` block of every
arm is DECLARED by this tool (inference inputs, vision+v0+nav, goal source,
goal/situation disjointness, corpus, parity note) with `_declared_by` stating so.

`rec["refav1"]`:

| block | content | tier |
|---|---|---|
| `strategic` | route-head argmax vs `route_label` under **three conditionings** (`nav_true`, `nav_shuffled`, `nav_zero`), each an `_agreement_block` (accuracy, κ, per-class, never-predicted, episode-cluster CIs) on windows that are **route-labeled AND nav-valid**; `nav_valid_frac`, exclusion counts, `nav_implied_route_agreement` (the echo index), majority-class rate, `paired_true_minus_shuffled_accuracy`, and the **changed subset** (`route_follows_LABEL` vs `route_follows_SHUFFLED_NAV`, mutually exclusive by construction) | T1 (heads read the observed window only) |
| `tactical_declared` | factored `lat`/`lon` head argmax vs the v7.2 `lat_label`/`lon_label` (v7.0 8-way vocabularies, `-100` excluded) under the three conditionings + paired true−shuffled | T1 |
| `wm_diagnostic_T0` | teacher-forced feature MSE per step (`to_enc(op_pred)` vs `std(future)` — the trainer's own `loss_feat_op`; tripwire asserts the per-step curve averages to `out["loss_feat_op"]`), with two controls: **persist-last-field** (`feat_mse_const`) and **constant-only zero prediction** (`feat_mse_zero`, must read ≈ target variance ≈ 1.0 in frozen space); paired `const − model`, `zero − model` | **T0** |
| `planner` | per planning arm: `source_fractions` (cem vs each injected baseline), `baseline_won_frac`, coarse/fine agreement, mean cost, candidates evaluated | — |
| `families_paired` | per pair (`cl−ol`, `cl−ha`, `cl−cl_navshuf`, …) per family: LON (`speed/along/accel` MAE), LAT (`cross/heading/yaw-rate`), TACTICAL (trajectory-derived lat/lon correctness), ADE/FDE — `paired_episode_cluster_bootstrap` on the same windows | per pair |
| `protocol`, `manifest` | declarations + the roll's provenance (model cfg, strict-load report, eval-time cfg overrides, plan cfg, nav-shuffle stats, join report, cache `index.json`) | — |

Every block prints `n`, `tier`, `estimator`; refusals carry `status: REFUSED/UNAVAILABLE`
+ `reason` + `n`. Distance-keeping stays UNAVAILABLE (no lead block for this grid — a
WORK ITEM: `tools/build_lead_block.py` needs the `obstacle.offline` join on the 141
eval clips).

## 5. MEASURED on the synthetic fixture (random-init tiny model; not model evidence)

* **`ol` reads its known value:** ADE **0.0765 m** [0.0551, 0.1022] over 2 s (n = 27
  windows / 3 episodes, n_boot 50) — the 0.2 s `(a, κ)` contract integrated by the
  programme's unicycle reproduces the 10 Hz GT to ~8 cm. This is what makes the
  other arms' ADE interpretable; a κ sign flip or ego-frame handedness error would
  show here first.
* ⛔ **With no goal, `plan()` returns `baseline:decel_1.5` on 100 % of windows.**
  `plan()`'s docstring says `goal_field` "defaults to the tactical brain's own
  imagined 6 s field"; the code does `search_goal = None if goal_field is None`
  (refa_v1.py, `plan`), so the cost is jerk + curvature only; every zero-curvature
  constant-accel candidate scores exactly 0 and `icem_plan`'s floor loop keeps the
  LAST tie (`<=` over `cv, hold_v0, proposal, decel_1.5`). The "T1 plan" is a
  constant −1.5 m/s² brake. (My first reading was "cv by construction"; the test
  corrected it — the tie-break order decides, and it is load-bearing.)
* Even the **oracle-goal** arm returned a baseline on 100 % of windows
  (`hold_v0` 63 % / `decel_1.5` 37 %) at the test's tiny search (8 samples × 1 iter)
  — a property of that search budget, NOT evidence about the real planner.
* Hold-action / T1 no-future pin: perturbing recorded `v`/`κ` at every frame after
  the last window's t0 leaves `ha`, `cl`, `cl_navshuf` bit-identical and moves `ol`
  on every window whose horizon reaches the perturbed frames.
* Constant-only WM control: zero prediction reads `≈ tgt_std²` (asserted within 0.15).

## 6. Invocation for the real run (Thor — paths marked ⚠️ are UNVERIFIED from this box)

```bash
# on Thor, tanitad-train venv, from the repo checkout
PYTHONPATH=<repo>/stack:<repo>/taniteval python3 taniteval/tools/refav1_arm.py \
  --ckpt   /home/nvidia/experiments/refav1-b1-v72-1ep-21109/ckpt.pt \
  --config /home/nvidia/experiments/refav1-b1-v72-1ep-21109/config.json   # optional; cross-checked vs ckpt['cfg'], contradiction = refusal
  --cache    <refav1-fp8-eval dir: the 141-clip v7.2 EVAL symlink split>   # ⚠️ path not known here
  --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl                # ⚠️ the B1 v2ep dir (dinov3_fp8_encode_ship.py SRC["b1"])
  --labels   <s2_labels_v7.2 EVAL blob, md5 aa12c948f062181c3297265b51526ec5>  # ⚠️ path not known here
  --nav      <same blob> \
  --device cuda --window-stride 10 --episodes-n 0 \
  --with-oracle-goal-arm            # optional T0 arm: search-vs-goal attribution
  --dump-dir /home/nvidia/experiments/refav1-b1-v72-1ep-21109/t1_dump \
  --out      /home/nvidia/experiments/refav1-b1-v72-1ep-21109/refav1_t1.json --arm refav1-21109
```

* ⛔ **Never on the training pod/box while it trains**; the checkpoint is pulled
  by the Master Mind. `RefAV1Windows` REFUSES a mis-gridded cache and a zero-join
  label blob, and the tool refuses a vocabulary mismatch, a non-strict load, an
  unfitted standardizer, and a config that contradicts the weights.
* **Cost (ESTIMATED, from the config's own 4060 measurement of 160 ms/candidate on
  the 640-token field, ~1/10 on the tactical field):** DINO-WM defaults
  (300 × 30, decay 1.25) evaluate ≈ 2,100 candidates per `plan()` ⇒ ~35 s per window
  per planning arm on a 4060-class GPU; 141 episodes × ~86 windows at stride 1
  ≈ 12,000 windows ⇒ **days per arm**. Use `--window-stride 10` (≈ 1,200 windows,
  ~12 h per arm at that rate) and/or `--plan-n-samples/--plan-n-iters`. The tool
  prints an estimate after the first `plan()`; read it before leaving.
* `--wm-k` must be ≥ 15 (the strategic stride) — `forward()` refuses a shorter
  future; default = `cfg.op_steps` (30).

## 7. UNVERIFIED until a real checkpoint is read

1. The real `ckpt.pt` loads STRICT into `RefAV1(RefAV1Config(**cfg))` on the eval
   box's copy of `refa_v1.py` (the trainer's `vars(cfg)` carries dataclass sub-configs;
   a `config.json` with dict sub-configs is handled and cross-checked — but the
   Master Mind's `config.json` format is not known here).
2. The fp8 eval cache opens through `RefAV1Windows` (`torch.load(mmap=True)` on
   `float8_e4m3fn`; the live TRAIN run proves the loader reads fp8, the eval split
   itself is unseen), and its `index.json` geometry matches `DINOV3_GEOMETRY`.
3. The 141-clip eval split's `clip_id`s join the eval labels blob (loader prints the
   join report; `nav_valid_frac` is reported — the loader docstring's obligation).
4. Wall-clock per `plan()` on Thor (only the 4060 number exists).
5. Whether the live run's `target_space` is `frozen` (register says yes; the tool
   reads it from the config and labels the WM space accordingly).
6. `ema_targets` checkpoints: the `ema.*` keys are part of the state_dict and load
   with the same config; not exercised here.

## 8. Decisions for the Master Mind

1. **The T1 arm has no admissible goal source.** With `goal_field=None`,
   `target_speed=None` the deployed `plan()` is a baseline by construction (a
   −1.5 m/s² brake at the current tie-break). Options: (a) accept and report the
   floor honestly (the tool does); (b) define an inference-time goal that does not
   read the future (the docstring's "tactical brain's imagined field" is not
   implemented; `target_latent` is untrained; the proposal head trains only with
   `w_aux_head > 0`); (c) at minimum change `icem_plan`'s tie-break so a tie
   resolves to `cv`/`hold_v0` rather than the last baseline. Any of these is a
   model/planner change, outside this adapter's ownership.
2. **Route accuracy under true nav is an echo index, not skill** (loader docstring;
   D-REFAV1-LAUNCHED caveat). Quote only the nav-shuffled / nav-zero conditionings
   and the changed-subset pair; with a FOLLOW-heavy marginal the changed subset may
   be small — its `n` is printed.
3. **Declared vs executed tactical decisions use different vocabularies** (v7.0
   8-way heads vs the refc 3-way trajectory labeller). Both are reported; no mapping
   is invented. If one is wanted, it is a vocabulary decision, not an eval one.
4. **Distance-keeping** needs a lead block on this grid (WORK ITEM, not a pass).
5. Whether to run `cl_oraclegoal` (T0) on the real checkpoint — it is the cheapest
   discriminator between "the search is the bottleneck" and "the goal is".
6. Compute budget: stride and search size (see §6).
7. ⛔ **THE κ UNIT (§3b) — a PI/Master-Mind call, not a patch.** The corpus stores
   `atan(L·κ)` and three sites disagree about what the channel means: the loader feeds
   it to the model as κ (**this is the live Thor run's training input**),
   `kinematic.rollout_unicycle` integrates it as 1/m, and `refa_v1.canonical_controls`
   / `GOAL_KAPPA_TURN = 0.08` / `PlanConfig.kappa_max = 0.2` mint proposals in true
   1/m — so the tactical proposals are ~2.9× **under-actuated** (a TURN proposal is
   R 36 m in the model's learned action space, not the intended 12.5 m). Options:
   (a) repair at the loader behind a `steer_channel` flag and retrain (the diff is in
   the package's RESULT.md §6 — PROPOSED, not applied); (b) an eval-only stopgap that
   repairs `ol`/`ha` in this adapter — **which does NOT repair `cl`**, whose planned
   actions are in steer units and get integrated as curvature, and would put two
   conventions in one table; (c) accept and stamp the 2.9× on every LATERAL row.
   Not decided here.
8. **One-line integration:** add `"ha0": "T1"` to `t1_eval.DEFAULT_TIERS`
   (`t1_eval.py:145`) so the standalone CLI reads a refav1 dump without
   `--tiers ha0=T1`. Not made here — `t1_eval.py` is outside this change's ownership.
