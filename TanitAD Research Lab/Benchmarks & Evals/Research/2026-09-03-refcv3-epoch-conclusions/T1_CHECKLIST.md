# T1_CHECKLIST — the end-of-epoch refcv3 read, in execution order

**For:** the Master Mind, at epoch end (**projected 2026-09-03 22:43Z**, ESTIMATED from the run's own
4.069 s/step over the remaining 14,634 steps; `RESULT.md` §3.6) · **written by:** Benchmarks & Evals
FlyWheel, 2026-09-03 · **basis:** `SPEC.md` / `RESULT.md` in this package.

> ⛔⛔ **READ THIS BEFORE GATE 0. THIS IS A RUNBOOK, NOT AN INSTRUMENT.**
> **refcv3 has no admissible T1 number and no instrument that can produce one**
> (`D-V7-READINESS-2026-09-02` §D). The only draft that exists is
> `Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py` — never executed, never
> tested, no `REFCV3_ARM.md`, no T1 definition. **Nothing below can be run tonight without first
> building the adapter (BACKLOG R20).** What this checklist buys is that when the adapter is built,
> the read is right the first time — because the four failures it is written against each cost
> hours already.

## ⚠️ The two defects this checklist must carry at every step

1. ⛔ **The rescued refcv3 adapter replays a STEER ANGLE as a CURVATURE, and the defect exists in
   BOTH copies of that file** (`C-REFCV3-ARM-SAME-DEFECT`). `recorded_controls` (≈ line 477) sends
   the recorded channel through the unicycle integrator as `kappa` and its docstring calls it *"the
   MEASURED true-kappa channel"*; `stack/tanitad/data/physicalai.py:620` writes
   `steer = arctan(wheelbase · curvature)` into `actions[:, 0]`. It inherits the `action_units`
   kwarg and **nothing passes it**. On refav1's eval slice the identical defect cost **0.716 m** of
   curved-window lateral error against a **0.053 m** pose-yaw floor; the fix took the yaw ratio
   ×2.870 → ×0.995 and yaw RMSE 19.81° → 0.63°. ⇒ **Every path that integrates geometry needs
   `κ = tan(steer)/L`; every planner candidate reaching the model needs `κ → steer = arctan(L·κ)`.
   Follow `refav1_arm.py --action-units`; do NOT invent a second convention.** The wheelbase
   constant is UNVERIFIED (per-clip fits cluster near ~2.85 and ~3.09, none at exactly 2.9) — the PI
   directed: **find the release's own vehicle parameter before any fitted fallback**.
2. ⛔ **`t1_eval.DEFAULT_TIERS` still lacks `"ha0": "T1"`** (`taniteval/tools/t1_eval.py:145`,
   BACKLOG R13). Without it `resolve_tiers` raises *"arms ['ha0'] carry no T0/T1 tier stamp"* and the
   whole analysis aborts **after** the rollout. ⇒ **pass `--tiers ha0=T1` on every invocation**
   until the one-line fix lands.

---

## GATE 0 — the epoch really ended, and the checkpoint cannot move under you

- [ ] `summary.json` in `/workspace/experiments/refcv3-b1-v72-30k/` reads `{"done": true, "step": 40284, ...}`.
      The trainer writes it itself at completion (`refc_v3_train.py`, end of `train()`).
- [ ] `train.log`'s last lines show `ckpt step 40284 -> ckpt.pt` **preceding** `[v3:eval] step 40284`
      (the save-before-eval order, `D-REFCV3-SAVE-BEFORE-EVAL`).
- [ ] **The supervisor has exited.** ⛔ A supervisor whose run never wrote a done-marker resurrects
      the finished run and **overwrites `ckpt.pt` / `config.json` / `metrics.json`** — that is exactly
      how the v5f run was clobbered. `ps -eo args | grep -c 'sup[_]refcv3'` must read **0**, and if a
      lock is left with no holder, name the holder via `/proc/*/fd` before touching it.
- [ ] **Immediately copy the final checkpoint to an immutable name and md5 it.** `ck = out_dir/"ckpt.pt"`
      (`:1090`) is a **rolling** file overwritten every 500 steps — it is not an epoch artifact until
      you make one: `cp ckpt.pt ckpt_40284_FINAL.pt && md5sum ckpt_40284_FINAL.pt`.
- [ ] **Verify the checkpoint by its CONTENT, not its filename:** load it and assert
      `ckpt["step"] == 40284`. Also present, as milestone snapshots: `ckpt_5000.pt`, `ckpt_15000.pt`,
      `ckpt_20000.pt`, `ckpt_30000.pt` (`MILESTONES`, `:103`) — **model-only**, no optimizer.
- [ ] ⚠️ **The prior buffers.** `core.lat_log_prior` / `lon_log_prior` carried the eval-marginal leak
      until the gate at 17,500 (`C-REFCV3-EVAL-PRIOR-LEAK`); the training-only EMA erases it within
      ~460 train batches (0.99^460 ≈ 1 %). At 40,284 that is ~22,800 batches of decay — **clean**.
      State it in the read rather than leaving a reader to wonder.

**Which checkpoint:** `ckpt_40284_FINAL.pt` (the epoch-end weights). ⛔ Do **not** read
`ckpt_30000.pt` for the headline — it is a mid-run snapshot with no optimizer state and predates
2,000+ steps of the clean era. Read it only if a step-matched comparison is wanted **and** say so.

---

## GATE 1 — do not evaluate on the training pod, and do not re-run what is banked

- [ ] The pod is only free **after** GATE 0 passes. **Never add GPU/RAM load to a pod that is
      training** — if any doubt remains that the run is finished, use another device.
- [ ] ⛔ **Before re-running anything, check for `--analyze-only` / an existing dump dir.** An
      analysis-time import failure after a completed rollout reads as `NO_ARMS_PRODUCED` while the
      expensive part is already paid for; re-analysing banked dumps recovered every number with
      **zero GPU** once already.
- [ ] `OMP_NUM_THREADS=6` **before any multi-arm panel.** torch spawns ~113 threads per process; 7
      concurrent arms once sat at 0–6 % GPU for **50 minutes** with no progress, and the same arm
      finished in 232 s with the variable set.

---

## GATE 2 — ⛔ THE BLOCKING DELIVERABLE: define T1 for a ONE-SHOT supervised trajectory model

**This is the deliverable the killed agent never wrote, and no number may be quoted before it
exists.** It is a definition, not compute (`RESULT.md` §5.4, work item W6).

- [ ] Write `taniteval/tools/REFCV3_ARM.md` §1–§3 **with file:line** for every claim, stating:
  - refcv3 emits the **whole 6 s path in one forward pass**: 128 FPS anchors × 8 slots
    (`refc_v3.py::_v3_core_base`, `AnchorConfig(n_anchors=128)`; horizons
    `[5,10,15,20,30,40,50,60]` at 10 Hz = 0.5–6.0 s). **No action input, no rollout, no per-step
    decode.**
  - ⇒ `t1_eval.roll_closed` (`t1_eval.py:753–786`) **cannot be ported**: it closes the loop by
    feeding the head's own `(steer = atan(L·κ), a_j)` back into the predictor, and refcv3 has no
    action to feed back. *(The flagship IS supervised and `roll_closed` handles it — because the
    flagship is also **autoregressive**. That is the difference.)*
  - ⇒ the arm is named **`os`** (one-shot), **NEVER `cl`**, so no reader believes the two arms'
    `cl` columns describe the same procedure.
  - the inputs the arm is allowed at inference: **frames ≤ t0, `nav_cmd`, and the measured `v0` at
    t0** (PI ruling 2026-09-02: velocity at cycle time is a legal initial state; future dynamic
    ground truth is not).
  - ⛔ **the selection gate:** the deployed arm selects with the model's own
    `sel_score_v3` / `anchor_logits`. ⛔ **NEVER with `a_star`** — `a_star = dist.argmin(dim=1)` is
    the anchor nearest the **ground truth** (`refc_v3_train.py:457–459`), so
    `recon = anchor_traj[ar, a_star]` (`:463`) is an **oracle-selected** path. Its number is the
    ceiling, banked as a separate **T0** arm `oracle_sel`, never as the deployed one.
- [ ] Get the **tier ruling** (PI / Master Mind): does the doctrine admit as T1 a model that consumes
      no actions at all? Its T1 text says *"the predictor consumes the decoder/planner's own
      actions"*, which does not literally cover refcv3. Benchmarks' recommendation: **admit it**, with
      the distinct arm name. **Register decision 9 is waiting on this, not on GPU.**
- [ ] `stack/tests/test_refcv3_arm.py` exists and pins, at minimum: the `os` arm consumes nothing
      after frame `2t`; the selection is `sel_score_v3`, not `a_star`; the action-unit conversion is
      applied exactly once; a deliberate-regression arm that must FAIL.

---

## GATE 3 — the eval split and the window grid

- [ ] **Split:** the v7.2 EVAL release is **147 records**; **141 carry pixels** in the B1 epcache
      (the other 6 are the deployed-val40 clips the parity gate drops, `D-REFCV3-EVAL-LEAK`).
      **Score the 141.** Labels: `s2_labels_v7.2_eval.jsonl.gz`, md5 `aa12c948f062181c3297265b51526ec5`
      (the canonical blob, stamped in `config.json::nav_from_v7_stats`); nav distribution
      `follow 90 / left 13 / right 38, missing 0`.
- [ ] **Assert train∩eval = ∅ by episode id** before scoring. The armed chain once pointed
      `--v2-cache` at all 4,713 clips, which would have trained on **141 of the 147 eval clips** —
      *"a held-out split that is not held out measures memorisation"*, and it would not have
      announced itself.
- [ ] **The window grid is shared and asserted equal across arms** (same clips, same stride, same
      `t0`). `D-REFAV1-PAIRED-READ-VOID`'s second read asserted the grid and the ground truth equal
      before pairing; do the same, and print the assertion.
- [ ] ⛔ Do **not** reuse the in-training eval's 160 windows. They are a `manual_seed(12345)` subset
      chosen for a **loss monitor**, they carry no `eid`, and n = 160 is far too small for an
      episode-cluster bootstrap.

---

## GATE 4 — preflight, before the expensive part

- [ ] **Import probe at startup** for every module the analysis will need (`taniteval.ci`,
      `taniteval.four_families`, `taniteval.selgap`, `dump_census`, …). A missing optional module has
      already destroyed a completed 2-arm / 40-episode rollout at `analyze()` time. ⚠️ Note the
      namespace-package shadow: `No module named 'taniteval.ci'` usually means the OUTER `taniteval/`
      (no `__init__.py`) is shadowing the real package — a path problem, not a missing module.
- [ ] `--tiers ha0=T1` present on the command line (GATE-level defect #2).
- [ ] `--lead-block` points at the banked block:
      `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz`
      (29,556 rows = every raw 10 Hz frame of all 147 clips, 2.49 MB; joined by **(clip_id, RAW frame
      2t)**; coverage LEAD 28.2 % / NO_LEAD 21.9 % / NOT_STRAIGHT 35.9 % / NO_LABEL 13.9 %). Without
      it the LONGITUDINAL family's distance-keeping half reads **UNAVAILABLE**, which is a **work
      item, not a pass**.
- [ ] The action-unit conversion is set explicitly (`--action-units`), and the value is **printed**
      in the manifest. A defaulted unit is how defect #1 travels.
- [ ] Dump schema carries per-window **`eid`** (and `v0` as a covariate, not an arm). Without `eid`
      there is **no episode-cluster bootstrap** — the exact wall `metrics.jsonl` hits.

---

## GATE 5 — the arms (all five, in one dump, on one grid)

| arm | tier | definition | why it is here |
|---|---|---|---|
| **`os`** *(refcv3's deployed arm; `cl` for refav1)* | **T1** | one forward pass at t0; anchor chosen by `sel_score_v3`; frames ≤ t0 + nav + measured `v0` only | the thing being measured |
| **`ha`** | **T1** | hold the last **observed** `(a, κ)` for K steps | control that consumes no recorded future — and it can be **worse than trivial**, so it is not the floor |
| **`ha0`** ⭐ | **T1** | **constant velocity**: `a = 0`, `κ = 0` at the measured `v0` — a straight line at constant speed | **THE STRONGEST TRIVIAL BASELINE** (`D-REFAV1-HA0-ARM`). It consumes strictly **less** than `ha`. Against `ha` alone a straight line reads as lateral **skill**; against `ha0` it reads as what it is. Needs `--tiers ha0=T1`. |
| **`<arm>_navshuf`** | **T1** | the same arm with `nav_cmd` **permuted across the eval windows** | mandatory for every nav-conditioned result (`D-REFAV1-NAV-DEPTH`). Nav is an **input**: a route head that echoes it scores well and has learned nothing — flagship v1's scored **1.0000**. |
| **`ol`** | **T0** | the **recorded** future actions integrated from `v0` | ⚠️ **For refcv3 this arm DOES NOT EXIST** — it consumes no actions. Run it for refav1 as the kinematic-contract control; for refcv3 leave it **absent with the reason stated**, never silently dropped. |
| *(beside, optional)* **`oracle_sel`** | **T0** | refcv3's `a_star`-selected path — the GT-nearest anchor's refinement | the ceiling. Banked to show how much of `traj` was selection. **Never compared to a T1 number.** |

---

## GATE 6 — ⛔⛔ THE TRIVIAL-PROFILE INSTRUMENT PRINTS **BEFORE** ANY FAMILY ROW

**Non-negotiable, and it is the cheapest gate in this document.** On 2026-09-03 a paired refav1 read
was rolled for **2.5 h** and then found VOID: `cl` was a straight constant-speed line on **140/140**
windows and bit-identical to `cl_navshuf` on 122/140. The trivial-profile instrument reads that in
**the first minute** (`D-REFAV1-HA0-ARM`; `refav1_arm.py::trivial_profile`, `:1409–1497`).

- [ ] Print, **per arm, before any metric**: `n`, `straight_frac` (|y| < the straight threshold),
      `const_speed_frac`, **`CONSTANT-VELOCITY` = straight AND const-speed**, and `identical_to`
      (windows where two arms agree to < 1e-4 m).
- [ ] `degenerate_arms` = any arm with `trivial_frac > 0.5` — printed as a **warning**, not a
      footnote.
- [ ] ⛔ **A read whose arms are bit-identical to another arm is stamped VOID, never reported as
      "no difference".** Void ≠ negative: the instrument saw the baseline, not the model.
- [ ] For refav1 additionally print **`baseline_won_frac`** before any family row; a read at low
      steps is only honest once it is **falling** and the trivial fraction is **below 1.0**.
- [ ] ⭐ **The echo test's real bar is `os − ha0`** (or `cl − ha0`), never `− ha` alone.

---

## GATE 7 — the four binding metric families, per family, never pooled

ADE stays and these are **ADDED**. An eval reporting ADE alone is INCOMPLETE and must not be
presented as a result. Each family carries its estimator, its CI and its `n`.

| family | must report | refcv3-specific note |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and** distance-keeping (headway / time-gap / TTC to the lead), stratified by the speed bands | needs `--lead-block` (GATE 4). 88.7 % of the oracle gap is longitudinal; ADE hides it. |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross-track | ⛔ every curvature/yaw number is downstream of defect #1 — the conversion must be applied and **stated**. |
| **TACTICAL** | manoeuvre-decision quality **and** tactical goal-setting (selected vs executed, class confusion, goal/anchor selection) | refcv3's `anchor_acc` belongs here (chance **1/128 = 0.0078**), and the factored heads' CE against chance **ln 8 = 2.0794**. |
| **STRATEGIC** | strategic decision + goal/route setting quality | ⛔ inadmissible without the `navshuf` control beside it (nav-echo). Report the `goal_gate` value **and** the score scale it multiplies — the gate alone cannot distinguish *"has not opened yet"* from *"will never open"*. |

- [ ] ⛔ **Never present a horizon sweep of ADE as "the result".** It is one row of four families.
- [ ] Where a family genuinely cannot be computed, say so **per family with the reason and the `n`** —
      a missing metric is a **work item**, not an excuse, and never a silent drop.
- [ ] ⛔ **ADE is an L2 norm.** refcv3's training-time `traj` is a **mean L1 per coordinate**
      (`refc_v3_train.py:462–465`); recompute ADE from the dumped path. **Never convert one into the
      other.**

---

## GATE 8 — the estimator

- [ ] **Point estimates are FULL-SET pooled means over windows.** ⛔ Never the `heldout`
      mean-of-split-means: `overlapping_holdout_se` **biases the point estimate**, not only the
      interval (measured over 27 dumps: headline `ade_0_2s` shifted −6.67 % to +11.69 %,
      bidirectional, up to ×3.3 on hierarchy seams and ×−4.15 **including a sign flip** on paired
      deltas).
- [ ] **Intervals: the episode-cluster bootstrap** — `taniteval.ci.episode_cluster_bootstrap`
      (`ci.py:225`); resample **episodes** with replacement, because windows inside one clip are
      strongly dependent and the episode is the independent unit.
- [ ] **Two arms on the same windows ⇒ the PAIRED form**,
      `taniteval.ci.paired_episode_cluster_bootstrap` (`ci.py:275`). ⛔ Never combine two single-arm
      intervals in quadrature.
- [ ] The headline paired blocks: **`os − ha0`** and **`os − ha`**, per family, with `n` and the
      separation flag. For the H-vs-F claim the statistic is the **difference of each arm's margin
      over the same floor** — `(cl − ha0)_refav1` vs `(os − ha0)_refcv3` — **not** `cl` against `os`
      as levels (`D-HF-COMPARABILITY`).
- [ ] Every emitted number carries its **tier stamp**. `resolve_tiers` raises on an unstamped arm by
      design; do not work around it.

---

## GATE 9 — what to write down, and what to refuse

- [ ] The manifest states: checkpoint md5 + its `step` field, split (141 clips) + label md5, window
      count + episode count, arm list + tiers, action-units, lead-block sha, seeds, and the
      **trivial-profile table**.
- [ ] `metrics.jsonl`'s `eval_*` numbers are **T0** and stay out of the T1 tables. If they appear at
      all, they appear stamped, with `eval_traj` labelled **oracle-anchor-selected L1 per
      coordinate** (`RESULT.md` §2.3).
- [ ] ⛔ **Refuse to publish an H-vs-F number if any of `D-HF-COMPARABILITY`'s six
      inadmissibility conditions holds** — different grids, oracle-selected `traj` compared to `cl`,
      cross-tier or `cl`-vs-`os` as levels, a degenerate trivial profile, the unfixed steer defect,
      or no shared `ha0` floor.
- [ ] Update `GOALS_AND_CLAIMS.md` **in the same turn** as the read, and bank the dump + the JSON in
      the repo — not on the pod, not in a worktree.
- [ ] ⚠️ Report times in **Europe/Berlin** as well as UTC, or label the frame; pods and logs are UTC
      and an unlabelled UTC time reads as a broken clock.
