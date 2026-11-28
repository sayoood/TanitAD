# refcv6 metric-family coverage — will the panel report all four families? — RESULT

**status: DELIVERED (source + artifact audit, 0 GPU, nothing launched). Both dependencies RESOLVED
in-session (§7) — and one of them turned a "just wire it up" into a real blocker: no offline driver in
this tree can load a refcv6 checkpoint at all (§4.0).**

*Opponent Analyzer & Benchmarks FlyWheel, 2026-09-18. Worktree `C:/Users/Admin/tanitad-wt-bevtac`
(detached at `ee1635a`). Artifacts read: `C:/Users/Admin/tanitad-caches/refcv6-pipeval-20260917/`
(14 runs) and `C:/Users/Admin/tanitad-caches/refcv6-zhverify-20260918/` (1 run).*

---

## 0. The answer in one paragraph

**All four families will be missing, and all four will be missing SILENTLY** — not refused with a
reason and an `n`, which is the only admissible form of absence, but simply never mentioned. (Three
of the four are missing because nothing computes them; the fourth, STRATEGIC, is missing because
nothing *declines* it — see §5, where the fix is a refusal rather than a measurement.) The refcv6 eval
path as it exists today is the in-training eval block of `stack/scripts/refc_v3_train.py:7017-7076`.
MEASURED over 15 banked refcv6 runs: it emits **174 distinct keys, of which ZERO map to any of the 16
required criteria** of `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.9.0 — no ADE either. The
emitting script **imports no family instrument at all** (`taniteval.four_families`,
`.lead_metrics`, `.nav_compliance`, `.hierarchy`, `.ci`: all absent from its import list). Worse, the
artifact it writes (`metrics.jsonl`) matches **no in-scope marker** of `tools/criteria_check.py`, so
the machinery built precisely to catch a silently-absent family reads `UNKNOWN_SCOPE` and **scores
nothing** — the omission is invisible to the instrument that exists to see it. And the dump is
**aggregate-only**: every value is a scalar, there is no per-window array and no episode id anywhere,
so `taniteval.ci.paired_episode_cluster_bootstrap(a, b, eid, …)` — the only admissible interval — is
**not reachable from it by any rescore**. ⛔ **None of this is a missing instrument.** Every
LONGITUDINAL, LATERAL and TACTICAL metric the binding rule names is already implemented in
`taniteval/taniteval/four_families.py` and is PROVEN to run on a sibling arm: the banked
`refcv3-40284-openloop.ARM.json` scores **11/16 criteria PRESENT, 0 silent violations**. The gap is
**wiring, not construction** — case (b) on 14 of 16 criteria. The two exceptions are the two
STRATEGIC decision criteria, which for refcv6 are **not-applicable by design** (§5) and must be
reported as a per-family refusal with its reason and its n, not omitted.

⛔ **But the wiring is not a one-line call, and this is the single most actionable finding here.**
**No offline eval driver in this tree can load a refcv6 checkpoint** (§4.0). `refcv3_arm.py` — the
only driver built for `refc_v3` — refuses it at **three independent points**, because the trainer
attaches the refcv6 perception branch as a plain Python **attribute** (`model._perception`,
`refc_v3_train.py:5699-5705`) rather than as a `RefCV3Config` field, so the eval-side rebuild is a
structurally different model than the weights describe. The chain from checkpoint to
`four_families.py` is broken at the **first hop**, and every family downstream of it is blocked by
that one fact. ⇒ work item **W0**, which gates W1–W4.

⚠️ **One thing here is NOT a metric-coverage gap and is escalated rather than worked around:** the
panel's own executable bar, `verdict_refcv6.py`, has clause **S1 requiring `route_acc` with `n > 0`**
and treats `n = 0` as `MISSING_DATA`, *never a pass*. Under `SPEC_REFCV6_V2.md`'s deactivation of the
strategic layer that clause is **unsatisfiable by construction**, so the panel can never emit
`SUCCESS` no matter how good the arm is. That needs a PI ruling, not an engineering fix (§5.3).

---

## 1. What the refcv6 eval path actually emits today — MEASURED from the artifacts

The path is `stack/scripts/refc_v3_train.py`. Its eval block is lines **7017–7076** and its own
comment states what it is not:

> *"⚠️ WHAT THIS IS AND IS NOT: an in-training MONITOR at T0 on the same loss surface … It is NOT the
> four-metric-family result and must never be quoted as one — the binding families (longitudinal /
> lateral / tactical / strategic with paired episode-cluster CIs) are a separate T1 job."*
> — `stack/scripts/refc_v3_train.py:7017-7023`

Mechanically it averages whatever scalars `compute_losses_v3` returned over `--eval-batches` batches
and writes one flat row per eval point (`refc_v3_train.py:7067-7071`).

| fact | value | evidence class |
|---|---|---|
| runs audited | 15 (14 pipeval + 1 zhverify), all `--size tiny`, 1–40 steps, `--w-map 1 --w-box3d 1 --tac-decoder-v6` | **MEASURED** (ours) — `raw/refcv6_family_coverage.json` |
| distinct keys across all runs | **174** | **MEASURED** — same |
| keys mapping to a required criterion | **0 of 16** | **MEASURED** — same |
| keys holding a non-scalar value | **0** | **MEASURED** — same |
| keys holding an episode / clip / window id | **0** | **MEASURED** — same |
| family instruments imported by the emitter | **0 of 12 probed** | **MEASURED** — `emitter_import_test` in the same file |
| eval windows in the banked eval row | **16** (`eval_batches` 8 × `batch` 2) | **MEASURED** — `raw/refcv6_evalrow_as_artifact.json` |
| `criteria_check.py` verdict on the artifact as emitted | **`UNKNOWN_SCOPE` — not scored** | **MEASURED** — `raw/criteria_check_refcv6_AS_EMITTED.txt` |
| same row + only the `ade_0_2s` in-scope marker added | **23 required criteria ABSENT, 0 refused** | **MEASURED** — `raw/criteria_check_refcv6_INSCOPE_CONTROL.txt` |

⛔ **The `--size tiny` stamp matters and is stated rather than glossed.** `refc_v3_train.py:7098-7110`
declares `tiny` the VALIDATION RIG rung and that *"NOTHING measured at it is a model claim or may
enter MODEL_REGISTRY.md"*. These 15 runs are pipeline validations of the refcv6 flag set on the B1
139-clip corpus. They are cited here as evidence about **the emitted KEY SET**, which is a property
of the code path and not of the rung — never as evidence about the model.

### 1.1 The near-misses — keys that look like coverage and are not

Each of these is a **training LOSS term or a perception metric**, pooled over the batch, with no `n`,
no episode id and no interval. Reading any of them as a family metric is the single most likely way
this audit gets mis-reported, so they are listed by name (full list in
`raw/refcv6_family_coverage.json:false_friends`):

| emitted key | what it actually is |
|---|---|
| `lat`, `lon` | the lateral / longitudinal trajectory **loss** terms — not cross-track error, not speed error |
| `lat_tac`, `lon_tac`, `tacv6_lat_ce`, `tacv6_lon_ce` | tactical-head **cross-entropy losses** — a training objective, not manoeuvre-decision agreement |
| `route` | a route **aux-loss** term |
| `goal_tac`, `goal2s_err_m` | a goal-head loss, and the geometric goal POINT's 2 s error — one point, not ADE, not a family |
| `anchor_acc` | selector top-1 over the batch, no `n`, no `eid` |
| `map_iou_drivable`, `box3d_yaw` | **perception** metrics (SAM3 BEV map; other agents' cuboids) — not the ego's lateral yaw-rate error |

---

## 2. The gap table — family → instrument → case

⛔ The three cases are kept apart because they need different actions. **(a)** wired; **(b)** the
instrument exists in `taniteval/` and nothing in the refcv6 path calls it; **(c)** no instrument
anywhere. Case (b) is the cheap win and is the one most easily mis-reported as (a) — the test applied
here is an **import/caller test**, never a file-exists test.

| family | criterion (registry id) | instrument | wired into refcv6? | case | evidence |
|---|---|---|---|---|---|
| **LONG** | `long.target_speed` | `four_families.longitudinal` (speed MAE/bias/RMSE + `target_speed_acc` bands) | ✗ | **(b)** | `taniteval/taniteval/four_families.py:428` |
| **LONG** | `long.along_track_error` | same function | ✗ | **(b)** | `four_families.py:428` |
| **LONG** | `long.progress` | `four_families._ego_progress` | ✗ | **(b)** | `four_families.py:662` |
| **LONG** | `long.distance_keeping` | `four_families._distance_keeping` → `taniteval.lead_metrics.distance_keeping` | ✗ | **(b) + data dep** | `four_families.py:675`; needs a `lead` block (§4.2) |
| **LAT** | `lat.heading` | `four_families.lateral` | ✗ | **(b)** | `four_families.py:754` |
| **LAT** | `lat.curvature` | same | ✗ | **(b)** | `four_families.py:754` |
| **LAT** | `lat.yaw_rate` | same | ✗ | **(b)** | `four_families.py:754` |
| **LAT** | `lat.cross_track` | same | ✗ | **(b)** | `four_families.py:754` |
| **TAC** | `tac.manoeuvre_decision` | `four_families.tactical_from_trajectory` (labeller `refc_tactical.factor_from_kinematics`) | ✗ | **(b)** | `four_families.py:1200` |
| **TAC** | `tac.confusion` | same block (`confusion_gt_rows_pred_cols`, `never_predicted`) | ✗ | **(b)** | reference artifact §3 |
| **TAC** | `tac.goal_selection` | `four_families.tactical`'s `goal_setting` | ✗ | **(b)** | `four_families.py:1372` |
| **TAC** ⭐ | refcv6 decoder's DECLARED decision (22-token set, lat/lon) vs executed | `refcv6_tactical.per_class_report` → `refcv6_acceptance.tzero_nonturn_report` | ✗ | **(b)** | `stack/tanitad/refs/refcv6_tactical.py:852`; `taniteval/taniteval/refcv6_acceptance.py:190`. **MEASURED: `per_class_report` has ZERO production callers** |
| **STRAT** | `strat.decision` | — | **n/a by design** | see §5 | `SPEC_REFCV6_V2.md:51` |
| **STRAT** | `strat.route_goal` | — | **n/a by design** | see §5 | `SPEC_REFCV6_V2.md:51`, `:130` |
| **STRAT** | `strat.nav_compliance` | `nav_compliance.from_refcv3_dump` → `compliance_arm` | ✗ | **(b)** | `taniteval/taniteval/nav_compliance.py:401,846`; driver `taniteval/tools/nav_compliance_report.py:100` |
| **STRAT** | `strat.nav_compliance_ctrl_shuffle` | same (`paired_true_minus_shuffled`) | ✗ | **(b)** | same |
| **STRAT** | `strat.nav_compliance_ctrl_zero` | same (`paired_true_minus_zero`) | ✗ | **(b)** | same |

**⇒ 14 of 16 criteria are case (b). 0 are case (c). 2 are not-applicable by design.**

⛔ **But every one of the 14 is gated by the same precondition.** Case (b) says the instrument exists
and nothing calls it; it does **not** say the call is cheap. For refcv6 all 14 sit behind a single
`refcv3_arm.py` roll, and that roll currently **refuses the checkpoint** (§4.0). ⇒ the gap table's
right-hand column is one work item wide (**W0**) and then fans out into W1–W4. Reporting these as
"case (b), cheap win" without §4.0 attached would be true and would point at the wrong next action.

### 2.1 The proof that these are (b) and not (c) — the same instruments already ran

`taniteval/results/refcv3-40284-openloop.ARM.json` was produced by
`taniteval/tools/refcv3_arm.py` *(trajectory families via `taniteval/tools/t1_eval.py::analyze`,
IMPORTED)* and scores, MEASURED by `tools/criteria_check.py` in this session
(`raw/criteria_check_REFERENCE_refcv3_arm.txt`):

```
LONGITUDINAL  [4/4 present]      LATERAL  [4/4 present]      TACTICAL  [3/3 present]
STRATEGIC     [0/5 present]  — 5 WORK ITEMS (refused with a reason)
VIOLATIONS (silently absent, required): 0
```

including `distance_keeping` at **n = 1,252 windows / 67 episodes** with episode-cluster intervals, and
a tactical block with the full 3×3 confusion, κ = 0.8113 [0.9396, 0.9664] on accuracy, over
**4,823 windows / 141 episodes**. ⇒ **No METRIC has to be built** — every one of these ran, on this
corpus, through this code. What has to be built is the one hop that lets a refcv6 checkpoint reach
them (§4.0). **INHERITED** for the values, **MEASURED** (ours) for the criteria verdict.

### 2.2 ⛔ Zero production callers — the (b) evidence, stated as a measurement

`grep -rn <name> --include=*.py`, excluding each function's own definition file and `tests/`:

| function | production callers |
|---|---|
| `refcv6_acceptance.acceptance_panel` / `tflip_verdict` / `tzero_nonturn_report` / `speed_obedience_report` | **0** |
| `refcv6_tactical.per_class_report` | **0** |
| `refcv6_selection.planned_max_speed` | **0** |

The T-FLIP / T-ZERO / OBEDIENCE panel was frozen before any arm trained (`refcv6_acceptance.py:1-50`)
and its own docstring already says of the nav-flip arm: *"⛔ **IT HAS NEVER BEEN RUN.**"*

---

## 3. The estimator — a BLOCKING gap, answered from the artifact

`taniteval.ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0, alpha=0.05,
reduce="mean")` (`taniteval/taniteval/ci.py:275`) requires **per-window arrays `a` and `b` of equal
shape plus a per-window `eid` of the same length** — it raises on `a.shape != b.shape` and on
`len(eid) != a.size`.

MEASURED over all 15 banked runs (`raw/refcv6_family_coverage.json:estimator_reachability`):

* non-scalar values in `metrics.jsonl`: **0**
* episode / clip / window-id keys: **0**
* per-window dump file anywhere in the run directories: **none** — the directories contain exactly
  `ckpt.pt`, `config.json`, `metrics.jsonl`, `summary.json`
* a `--dump-windows` / `--windows-out` / per-window option in the trainer's parser: **none**
  (`grep` over `refc_v3_train.py`'s `add_argument` calls returns nothing)

⇒ **NOT REACHABLE.** The dump is aggregate-only; no rescore of it can produce an interval. And even
if it dumped per-window, the banked eval row reads **`eval_windows: 16`** — against the reference
arm's 4,823 windows / 141 episodes, 16 windows would not support an episode-cluster bootstrap at all.

⛔ This is the blocking gap named in the brief, and it is blocking in the strict sense: **it cannot be
fixed downstream.** Every other gap in §2 is a call that is not made; this one is data that is not
written.

---

## 4. What stands between the checkpoint and the instruments

### 4.0 ⛔ The chain is broken at the FIRST hop — no driver can load a refcv6 checkpoint

```
refc_v3_train.py  (refcv6 arm: --w-map --w-box3d --tac-decoder-v6 --tac-decoder-d-bev 96)
  └─ torch.save({"model", "opt", "step"})        state_dict CONTAINS _perception.*  (4,815,616 params)
     │
     ▼
  refcv3_arm.py::load_model                       ⛔ REFUSES — three times, see below
     │  (never reached) ▼
  dump/ep*.npz + decisions/ep*.npz ─► t1_eval.analyze ─► four_families.all_families
                                   └─► lead_metrics.distance_keeping
                                   └─► nav_compliance.from_refcv3_dump
                                   └─► refcv6_acceptance.tflip_verdict
```

The root cause is stated in the model file itself: the perception branch is *"still an ATTRIBUTE the
trainer attaches, not a config field"* — `refc_v3.py:1683-1685` reads it as
`br = getattr(self, "_perception", None)`, and the trainer sets it at
`refc_v3_train.py:5699-5705` (`model._perception = _perc.build_perception_branch(...)`). The eval-side
rebuild goes through `refc_v3_train.build_parser` + `_pin_trainer_cfg` (`refcv3_arm.py:998-1026`), so
it reconstructs everything that IS a config field — including `tac_decoder_v6`
(`refc_v3.py:548`) — and nothing that is not.

| # | refusal | site | MEASURED evidence |
|---|---|---|---|
| **1** | `cross_check_config` compares the whole `param_breakdown`, `total` included, and raises `SystemExit` on any conflict | `refcv3_arm.py:1052-1059`, called at `:1102` — **before** the load at `:1104` | On both banked refcv6 configs: named lines sum to **45,703,450**, stamped `total` **50,519,066**, delta **4,815,616** = exactly `refcv6_perception.branch_params.total`; **no `_perception` line in the breakdown**. The rebuilt model reproduces the named sum → conflict. ⛔ `--allow-nonstrict` cannot help: it gates `:1128`, which runs after. |
| **2** | strict-ish load refuses on `res.unexpected_keys` | `refcv3_arm.py:1128-1133` | `_perception` is an `nn.Module` submodule, so `ck["model"]` carries `_perception.*` keys the rebuilt model does not have |
| **3** | the forward itself raises | `refc_v3.py:2002-2012` — *"the behaviour decoder was built with d_bev 96 … but NO perception branch is attached"* | the refcv6 arms run `--tac-decoder-d-bev 96`. `refcv3_arm.py` also never passes `perception_grid` / `perception_valid` (its call is `refcv3_arm.py:2110`) |

The other two drivers do not even get that far: `t1_eval.py:1123-1126` hard-exits on a checkpoint
without a `grounding` key (a refc_v3 checkpoint is `{"model","opt","step"}`), and
`eval_four_families.py` goes through `taniteval/taniteval/loaders.py`, whose `arch` switch has no
`refc_v3` branch at all.

⚠️ **This is still case (b), not (c)** — every instrument exists and every one of them runs on refcv3
arms today. But the precondition is a **model-plumbing change**, not a call site. It is W0 in §6.

### 4.1 The panel has no producer

`verdict_refcv6.py` is invoked as `--panel raw/panel.json` and expects
`panel.families.{ADE,LATERAL,STRATEGIC,LONGITUDINAL,TACTICAL}.<metric> = {arm, reference, separated,
n}` plus `panel.estimator`, `panel.tier`, `panel.replicate_floor` (schema read from
`…/2026-09-10-refcv6-build/code/verdict_dropproof.py:38-60`). **MEASURED: nothing writes that file.**
`chain_refcv6.sh` runs arms strictly sequentially and ends at the training done-marker; a grep for
`panel|four_famil|t1_eval|refcv3_arm|openloop` across `chain_refcv6.sh`, `launch_refcv6.sh` and
`sup_refcv6.sh` returns only pre-**launch**-gate hits. **There is no eval step in the refcv6 chain.**

### 4.2 `distance_keeping` — the lead block EXISTS and is already the default

`four_families._distance_keeping` returns an explicit UNAVAILABLE-with-reason when `lead is None`
(`four_families.py:678-688`) — the *admissible* state, but a WORK ITEM, and the half of LONGITUDINAL
the PI made binding.

⭐ **It is already built.** `…/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/
b1_eval_lead_block.npz` is `refav1_arm.LEAD_BLOCK_DEFAULT` (`taniteval/tools/refav1_arm.py:232-234`)
and `refcv3_arm.py:3500-3510` picks it up automatically unless `--no-lead-block`. **MEASURED (ours,
loaded in this session):** 29,556 rows over **147 distinct `clip_id`** (a superset of the 139-clip
eval corpus), keyed by `(clip_id, frame)` — so the 139-clip subset joins by key, not by position —
carrying every field the metric needs: `leads [29556,10,2]`, `lead_lens`, `speeds`, `state`, `eid`,
`ts_rel_s` = 0.2…2.0 at `dt_s` 0.2. States: `LEAD` 8,341 / `NO_LEAD` 6,480 / `NOT_STRAIGHT` 10,622 /
`NO_LABEL` 4,113.

⚠️ **The horizon caveat travels with it.** `refcv3_arm.lead_block_common_grid` (`:2503-2550`)
index-selects both sides onto shared instants; refcv3's grids are `{"2s": (0.5, 4), "6s": (1.0, 6)}`
(`refcv3_arm.py:244`), so against the block's 0.2 s ladder only **{1.0 s, 2.0 s}** are common — a
uniform 2-instant grid at `dt = 1.0`. Distance-keeping is therefore scoreable, but on 2 of 4 (or 2 of
6) instants unless the block is rebuilt at the dump's own cadence
(`build_lead_block_b1.py --dt <dump dt> --k <dump k>`, named by the record itself at `:2524-2525`).
⇒ That, not the block's existence, is what W3 is now about.

### 4.3 ⛔ `arms.py` predates the spec — the arms it builds are not refcv6

Not a metric-coverage finding, but it decides *what the panel would be scoring*, so it is reported
rather than left for someone to trip over. `…/2026-09-10-refcv6-build/code/arms.py` builds
`BASE = refcv5-v2's argv, re-pointed, plus --agent-join` (its own `base_argv` docstring). MEASURED
flag census over that file:

| flag | occurrences in `arms.py` |
|---|---|
| `--nav-from-v7` | 1 |
| `--w-map`, `--w-box3d`, `--tac-decoder-v6`, `--join3d`, `--map-gt-root`, `--trunk-name`, `--no-strategic` | **0 each** |

Every constitutive flag of `SPEC_REFCV6_V2.md` (2026-09-16) is absent from the arm builder
(2026-09-10). The banked pipeline-validation runs pass those flags **by hand**, not through `arms.py`
(their `config.json:argv`), and all 15 carry `seams.no_strategic: false`.

---

## 5. ⚠️ STRATEGIC — not-applicable by design, and NOT a missing instrument

### 5.1 What the spec says, verbatim

`SPEC_REFCV6_V2.md` **§1**, line 51:

> ⛔ **Deactivated for this experiment:** the whole strategic layer — route head, `g_str`, strategic
> GRU. No head estimates the route (PI). The flags remain but default OFF and the heads are not built.

`SPEC_REFCV6_V2.md` **§9**, risk 4 (line 130):

> **No strategic layer** means no route output at all in this experiment, by the PI's instruction.

and **§0**, the PI's 2026-09-16 directive verbatim (line 18):

> *"Confirm using nav command as mandatory input for tactical and operative planning. The selection
> of the tactical plan and the planing and selection of the trajectory must use the nav command and
> follow it. We dont need any head to estimate the route. Regarding the startegic layer, we will
> deacitivate it in the next training and eval experiment"*

⚠️ **CORRECTION TO THE BRIEF'S PREMISE, stated because a citation must be checkable:** the brief asked
for `SPEC_REFCV6_V2.md` **§0 and §12**. **There is no §12.** The file runs §0–§10 plus ERRATUM-1 and
the 2026-09-17 PI ruling (`grep -n "^#\{1,3\} "` over the file, 275 lines). The strategic content
lives in §0 (the directive), §1 (line 51), §9 risk 4 (line 130) and §10.5 (line 182). Nothing is
missing — the section number was wrong, and the quotations above are the ones intended.

### 5.2 ⇒ The correct report, per family, with its reason and its n

**`strat.decision` and `strat.route_goal` are NOT-APPLICABLE BY DESIGN for refcv6.** Two independent
reasons, and both must travel with the number:

1. **By design.** No route head is built (spec §1). There is no strategic decision to score.
2. **By corpus, pre-existing and settled.** `four_families.STRATEGIC_UNAVAILABLE_REASON`
   (`four_families.py:1277-1293`): PhysicalAI-AV carries no map, no lane graph, no junction label and
   no route signal; both available label sources are inadmissible (a route read off the ego's own
   future yaw cannot see whether the map admitted a choice — that is how the closed-loop harness once
   published `route_head_eq_logged = 1.0000` on a single-continuation clip, and `GATE_PROTOCOL §0.7`
   declares `nonav_route_beats_majority` VOID BY CONSTRUCTION). Marked `_settled` at five independent
   probes. ⇒ **Even with the head switched on, there is no admissible label on this split.**

⛔ **The admissible form is a refusal, not an omission**, and the API for it already exists:
`four_families.strategic(win, no_label={"n": <n_windows>, "tier": "T1"})` →
`strategic_unavailable(n_windows)` (`four_families.py:1305`), which emits `status: UNAVAILABLE`,
`reason`, `n`, `instrument_that_would_close_it` (the PH0→PH1→PH2 VLM strategic-labelling pipeline) and
`_is_a_work_item`.

⚠️ **And the reference artifact gets this WRONG today — do not copy it.**
`refcv3-40284-openloop.ARM.json` reports STRATEGIC via the `_decision_family` fallback with
**`"n": 0`** and the reason *"strategic decisions not present in the scored pass (missing
['route_pred','route_gt'])"*. That reason says *the eval did not traverse the hierarchy*, which reads
as an eval-engineering gap someone should go fix. For refcv6 the true reason is *the layer is off by
design and the corpus has no label*, and `strategic_unavailable`'s own docstring says why `n = 0` is
wrong: *"`n` here is the number of windows the family WOULD have had — it is not zero, and reporting
0 would understate what is missing."* ⇒ work item **W5**.

### 5.3 ⛔ What IS still owed on STRATEGIC, and is not off by design

The PI made nav a **mandatory input** to tactical and operative planning. "Does the emitted behaviour
FOLLOW the commanded route" is therefore not only computable, it is the thing the ruling makes
testable — and it is a *behavioural* readout, immune to the label-bijection defect that voided route
accuracy (`nav_compliance.py:1-25`). The registry requires it as three criteria
(`strat.nav_compliance` + the shuffle and zero controls).

**MEASURED across the whole banked corpus** (`tools/criteria_check.py --all --recursive
taniteval/results/`, 41 in-scope artifacts): nav-compliance reads **0 present / 3 refused / 38
missing** on each of the three. ⇒ It has never been PRESENT on any arm in the programme. This is the
highest-value single gap in this audit.

### 5.4 ⛔ ESCALATION — clause S1 is unsatisfiable under the current spec

`…/2026-09-10-refcv6-build/code/verdict_refcv6.py:284-297`: if `families.STRATEGIC.route_acc` is
absent → `MISSING_DATA` (*"⛔ AN ABSENT STRATEGIC ROW IS THE REGRESSION, NOT AN OMISSION"*); if it is
present with `n <= 0` → `MISSING_DATA` (*"n = 0 is MISSING_DATA, never a pass"*). `MISSING_DATA`
blocks `SUCCESS` by design — and `PREREG_REFCV6.md:320` pre-registers *"omit the whole STRATEGIC
family → MISSING_DATA"* as a proven drop-proof mutant (13/13 blocked,
`raw/verdict_dropproof.json`).

⇒ **With the strategic layer deactivated there is no `route_acc` and S1 can only ever read
`MISSING_DATA`, so the refcv6 panel cannot emit `SUCCESS` however good the arm is.** The prereg
(with corrections to 2026-09-11) and the spec (2026-09-16) disagree, and the spec is the later
binding document. ⛔ **This is a PI decision, not an engineering fix**, and per the operating standard
it is escalated here rather than written into a doc as a merge request. The decision needed: either
(i) amend S1 to score `strat.nav_compliance` — the behavioural readout the same PI ruling makes
mandatory — in place of `route_acc`, or (ii) re-enable the strategic layer for the panel, which
contradicts §0. Option (i) is the only one consistent with both documents.

### 5.5 ⚠️ A second bar/registry mismatch, found while checking S1

`verdict_refcv6.py`'s `REQUIRED_LONGITUDINAL` is `{speed_mae, along_track}` and `REQUIRED_TACTICAL`
is `{tactical_lateral_kappa}` (`verdict_refcv6.py:92-103`). The binding registry requires four
LONGITUDINAL criteria (including **distance-keeping**) and three TACTICAL (including **confusion**
and **goal selection**). ⇒ **A panel can clear `verdict_refcv6.py` and still fail
`tools/criteria_check.py`.** Both must be run; neither subsumes the other. Work item **W6**.

---

## 6. Work items — the smallest thing that closes each gap

Ordered by dependency, not by size: **W0 gates W1–W4**. ⛔ All are 0-GPU except W1's forward pass
(one pass per eval window on the 139-clip corpus; the reference refcv3 roll was ~11 min/arm).

| id | closes | file / function | what it needs |
|---|---|---|---|
| **W0** ⛔ | **gates W1–W4** — nothing downstream can run until this lands | `taniteval/tools/refcv3_arm.py::load_model` + `stack/tanitad/refs/refc_v3.py` | Make the refcv6 perception branch **reconstructible on the eval side**. Two shapes, pick one: **(i)** promote the branch to a `RefCV3Config` field so `RefCV3Model(cfg)` builds it and `param_breakdown_v3` counts it — clean, but touches the training-bit-identity property `refc_v3.py:1683-1685` deliberately protects; or **(ii)** have the trainer stamp a perception record in `config.json` (the `refcv3_arm_model_cfg` escape hatch at `refcv3_arm.py:1012` is the existing precedent) and have `load_model` call `_perc.build_perception_branch` + attach `_lift_bank` after the rebuild. **Either way three further things are needed:** a `_perception` line in `param_breakdown_v3` (`refc_v3.py:2190-2199`) so BLOCK 1 clears legitimately rather than by loosening the check; the `_perception.*` keys accepted by the strict load; and `perception_grid` / `perception_valid` plumbed into `refcv3_arm.py`'s forward call (`:2110`) so BLOCK 3 clears. ⛔ Do **not** close this with `--allow-nonstrict`: it does not reach BLOCK 1, and if it did it would silently score a model missing 4.8 M parameters. |
| **W1** | `long.*` (3), `lat.*` (4), `tac.*` (3) = **10 criteria**, and the estimator | `taniteval/tools/refcv3_arm.py` → dump → `taniteval/tools/t1_eval.py::analyze` → `four_families.all_families` | After W0: roll the refcv6 checkpoint through the **existing** arm driver. The dump is `ep{NNN}.npz` (arm paths keyed by arm name — `os`, `g`, `ha`, `ha0_ext`, … plus `ws`, `eid`, `clip_index`) + `decisions/ep{NNN}.npz`; `t1_eval.analyze` builds the `win` dict (`pred`, `gt`, `eid`, `dt_s`, `wp_steps`) the families consume. **This one roll produces 10 of the 16 criteria AND makes the bootstrap reachable**, because the dump is per-window with an episode id. |
| **W2** | `strat.nav_compliance` ×3 | `taniteval/tools/nav_compliance_report.py --dump-dir <W1 dump> --labels <v8 eval labels>` → `nav_compliance.from_refcv3_dump` | 0 GPU, pure re-analysis of W1's dump. The sidecar keys it needs (`plan_full_*`, `gstr_*`, `sel_bank_*`, `fan_term_heading_*`, `reach_keep_*`, `gt_future_ext`, `pose_last`, `ego_t0`, `ep_poses`) are **already written** by `refcv3_arm.py`'s `decisions/ep*.npz` — so this is a flag, not a change. The nav-FLIP arm needed by T-FLIP is also already implemented (`refcv3_arm.py:1888-1900`, `--with-navflip` at `:3440-3444`) and has **never been run**. |
| **W3** | `long.distance_keeping` | pass `--lead` / let `refcv3_arm.py:3500-3510` pick up the default block | The block **exists** (§4.2) and joins by `(clip_id, frame)`. The only real work is the cadence: either accept the 2-instant common grid {1.0 s, 2.0 s} and **say so with its n**, or rebuild at the dump's cadence with `build_lead_block_b1.py --dt <dump dt> --k <dump k>`. Pass `lead["eid"]` so the speed-stratified read is emitted rather than refused (`four_families.py:735-745`). |
| **W4** | the refcv6 tactical decoder's DECLARED decision | `refcv6_tactical.per_class_report` (has **0** callers) → `refcv6_acceptance.tzero_nonturn_report` / `tflip_verdict` / `speed_obedience_report` → `acceptance_panel` | A driver that runs the decoder on the eval windows under `nav_true` and `nav_cmd=None` (T-ZERO) and under the flip (T-FLIP), and the forced-30 km/h pass (OBEDIENCE). The bars are frozen; only the producer is missing. `acceptance_panel` already refuses to pass with an instrument absent. |
| **W5** | `strat.decision`, `strat.route_goal` — as an **admissible refusal** | call `four_families.strategic(win, no_label={"n": n_windows, "tier": "T1"})` | One argument. Replaces the reference artifact's wrong `n = 0` + wrong reason with the corpus/by-design reason at the true n. ⛔ Not optional: a family reported UNAVAILABLE is a WORK ITEM, a family omitted is a VIOLATION. |
| **W6** | the artifact being *scoreable at all* | the panel writer | Emit the panel in a shape `tools/criteria_check.py` recognises (one of its `in_scope_if_any` markers, e.g. `arms.os.four_families`), carrying `tier: "T1"`, the estimator string, and `n` windows + episodes. Without this the completeness machinery reads `UNKNOWN_SCOPE` and silently scores nothing — **MEASURED in this session**. Run **both** `verdict_refcv6.py` and `criteria_check.py --strict`; §5.5 shows neither subsumes the other. |

---

## 7. The two open dependencies — both RESOLVED in-session

* **D1 — RESOLVED, and it changed the answer.** `refcv3_arm.py`'s **dump writer** is fine: it already
  emits per-window arm paths with `eid`, and the full nav-compliance sidecar. What is **not** fine is
  the hop before it — the **model rebuild** refuses a refcv6 checkpoint three times (§4.0). My earlier
  reading ("argv-driven, so the refcv6 flags reconstruct") was right about `tac_decoder_v6` and
  **wrong about the perception branch**, which is not a config field. ⇒ W1 is gated by the new **W0**.
  Independently verified in this session: the three refusal sites by source read, the refusal
  ORDERING (`cross_check_config` at `:1102` precedes `load_state_dict` at `:1104`, so
  `--allow-nonstrict` cannot reach BLOCK 1), and the parameter arithmetic on **two** banked refcv6
  configs (45,703,450 named vs 50,519,066 total, delta 4,815,616 = the branch exactly).
* **D2 — RESOLVED, favourably.** The B1 lead block exists, is the wired default, and covers 147 clips
  ⊇ the 139-clip corpus, keyed by `(clip_id, frame)` (§4.2). W3 shrinks from "build it" to "match the
  cadence, or state the 2-instant grid with its n".

⇒ **No dependency of this audit is open.** Every claim above is MEASURED (ours) from either the
banked artifacts or the source at `ee1635a`, except the reference arm's own metric VALUES, which are
INHERITED from `taniteval/results/refcv3-40284-openloop.ARM.json` and are cited only to show the
instruments run.

---

## 8. Deliverable manifest

| artifact | location |
|---|---|
| this record | `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-18-refcv6-metric-family-coverage/RESULT.md` |
| reproducible audit tool — emitted keys vs the registry | `…/code/family_coverage_audit.py` |
| key census + import test + estimator verdict | `…/raw/refcv6_family_coverage.json` |
| reproducible probe — the two chain blockers (§4.0, §4.2) | `…/code/chain_blockers_probe.py` |
| param-breakdown arithmetic + lead-block census | `…/raw/chain_blockers.json` |
| the refcv6 eval row exactly as emitted | `…/raw/refcv6_evalrow_as_artifact.json` |
| the same row + only the in-scope marker (constructed control) | `…/raw/refcv6_evalrow_inscope_CONTROL.json` |
| checker verdict, artifact as emitted (`UNKNOWN_SCOPE`) | `…/raw/criteria_check_refcv6_AS_EMITTED.txt` |
| checker verdict, in-scope control (23 ABSENT, 0 refused) | `…/raw/criteria_check_refcv6_INSCOPE_CONTROL.txt` |
| checker verdict, reference refcv3 arm (11/16, 0 violations) | `…/raw/criteria_check_REFERENCE_refcv3_arm.txt` |

**Staged, not committed.** Explicit paths only; nothing pushed, no branch touched, no run launched.

⚠️ **Path note.** The brief named `TanitAD Research Lab/Benchmarks & Eval/Research/…` (singular). The
repo directory is **`Benchmarks & Evals`** (plural) — there is no singular directory, and the plural
form is the binding one (PI 2026-08-27 rename). Banked at the plural path.
