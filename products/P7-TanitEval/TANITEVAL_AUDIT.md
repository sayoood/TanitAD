# TanitEval capability audit — what the harness can compute TODAY

**Date:** 2026-08-23 · **Scope:** `taniteval/` (+ `stack/scripts/driving_diagnostic.py`,
`stack/tanitad/refs/refc_tactical` where the eval reaches into the stack).
**Evidence class:** MEASURED — every claim below is read out of the source at the cited
`file:line`, or produced by loading a committed artifact on CPU. No doc, changelog or
report was used as a source. No checkpoint was loaded; no GPU was touched.

**Method note (matters for reproduction).** This worktree lives on a Google-Drive mount where a
**whole-file `read()` intermittently fails with `OSError: [Errno 22]` while a chunked read of the
same bytes succeeds.** That is why `pytest` could not even collect `taniteval/conftest.py` in
place. Every file was therefore copied byte-for-byte (64 KiB chunks) to a local mirror and read
there; line numbers are identical to the repo.

---

## 0. Headline

| | |
|---|---|
| ⭐ **The four-family instrument is real and unusually complete.** | `four_families.py` is 1253 lines and implements LONGITUDINAL, LATERAL and TACTICAL with per-family blocks, per-class confusion, `never_predicted`, episode-cluster CIs on the decision families, dt-provenance and honest `UNAVAILABLE` shapes. Nothing is pooled into a composite. |
| ⛔ **ESCALATE — the tool the doctrine names emits an UNSTAMPED tier.** | `taniteval/tools/eval_four_families.py` contains the substring `tier` **zero times**. It calls `ff.all_families(win, hier=hier)` at `:233` with no `tier=`, so no `_tier` key is written (`four_families.py:1247-1251` stamps only when the caller passes one) and the output record `:306-336` has no tier field. Its surface is a teacher-forced `rollout.collect` pass. **This is the one path where a T0 number can leave the harness looking like a result.** |
| ⛔ **ESCALATE — `overlapping_holdout_se` is alive in 5 definitions and 24 product call sites**, and the guard that exists cannot see most of them (it only refuses *verdict-named* bindings). Full list in §B.3. |
| ⛔ **ESCALATE — all 27 committed `windows_*.pt` are sparse-only.** MEASURED by loading them: 7 keys, no `pred_dense`. `tools/eval_four_families.py:263-264` indexes `win["pred_dense"]` unconditionally ⇒ **KeyError on every banked dump**. The rescore path (`ff_rescore`) handles it; the named tool does not. |
| ⭐ **The adapter seam is one function.** `ff_rescore.load_dump` (`tools/ff_rescore.py:150`) is the single place a new corpus format enters. Everything downstream — `score_arm:381`, `all_families:1133`, `ci.py` — is already dataset-agnostic. A NavSim/nuScenes adapter is a *loader*, not a harness rewrite. §C.5 gives the exact contract. |
| ⛔ **STRATEGIC is corpus-blocked on PhysicalAI-AV, but the instrument EXISTS** and has 15 scoreable events banked on 12 admissible NuRec scenes (MEASURED, §A.4). Adopting a map-carrying benchmark is what unblocks the programme's thesis metric. |

---

## A. The four metric families

Everything is assembled by **`four_families.all_families`** — `taniteval/taniteval/four_families.py:1133`.

### A.0 Pooling and completeness (the binding rule's own clauses)

| check | verdict | evidence |
|---|---|---|
| Per-family, never pooled | ✅ **COMPLIANT** | `four_families.py:1186-1194` builds a 4-key dict; grep finds no composite score anywhere in the module. |
| Family carries its `n` when UNAVAILABLE | ✅ **COMPLIANT** | `_rule_satisfied` at `:1238-1242` requires `reason` **and** `n` for any `status == "UNAVAILABLE"`; `strategic_unavailable:889-892` sets `n` to the windows it *would* have had, not 0 (`:886-887`). |
| Family carries its `n` when OK | ⚠️ **PARTIAL** | LONGITUDINAL emits `n_windows` (`:306`) and LATERAL `n_windows` (`:462`) — **neither emits `n`**. TACTICAL emits both (`:809-810`). `t1_eval.py:502` back-fills `n` via `setdefault`; **`ff_rescore.score_arm:425-427` does not**, so a rescored record's LON/LAT blocks have no `n` key. |
| Reason-when-absent | ✅ | `_families_unavailable` `:1213`, `_complete` `:1214-1216`, `_rule_satisfied` `:1238`, and the two-questions note `:1243-1246`. |

⚠️ **`_complete` can never be True on a genuinely empty road.** `:1215` requires
`distance_keeping["status"] == "OK"`, but `lead_metrics.distance_keeping:200-204` returns
`"NOT-APPLICABLE"` (with its reason and n) when no lead is in the corridor in any window —
free flow, not a failure. A free-flow corpus is therefore permanently `_complete: false`.

### A.1 LONGITUDINAL — `longitudinal()` `four_families.py:230`

| metric | status | where |
|---|---|---|
| speed MAE / bias / RMSE | ✅ EXISTS | `:268-270` (bias signed, `+` = too fast) |
| **target-speed ACCURACY** | ✅ EXISTS | `:276-278` — fraction of horizon **steps** with `abs(speed_err) <= b` for `b ∈ (0.5, 1.0, 2.0)` m/s (`TARGET_SPEED_BANDS_MPS:56`). ⚠️ bands self-label as **PROPOSED, not a gate** (`:52-55`, `:279-282`). |
| along-track MAE / bias / final bias | ✅ EXISTS | `:284-286` |
| acceleration MAE | ✅ EXISTS | `:288` |
| ego progress | ✅ EXISTS | `:294` → `taniteval/progress.py`, projected on the human's own direction so GT scores exactly 1.0 (`:330-340`) |
| **distance keeping — headway** | ✅ EXISTS | `lead_metrics.py:170` — per-window **minimum** gap over the horizon |
| **distance keeping — time gap (THW)** | ✅ EXISTS | `lead_metrics.py:171-172` — `headway / v_ego`, **NaN below `MIN_SPEED_MPS = 0.5`** (`:50`), never clamped |
| **distance keeping — min TTC** | ✅ EXISTS | `lead_metrics.py:174-185` — `gap / closing_rate`, rate clipped to ±20 m/s (`:52`), **censored at `TTC_CAP_S = 30` s** (`:48`) with `n_closing` and an explicit `censoring_note` (`:212-215`) |
| speed-stratified distance keeping | ✅ EXISTS | `four_families.py:403-407` → `lead_metrics.distance_keeping_by_speed:255`, 6 bands (`:58-59`), strata below `MIN_STRATUM_N = 30` (`:62`) reported UNPOWERED |
| anti-echo (hold-v0) controls | ✅ EXISTS | `:305` → `taniteval/v0_antiecho.py`; `_longitudinal_claim_admissible` promoted to the block's top level `:1222-1230` and printed by the CLI (`eval_four_families.py:349-350`) |

**How the lead agent is identified** — `lead_source.select_lead_causal` `lead_source.py:235-271`:
**strictly causal** (last cuboid at or before `t0`, staleness `<= MAX_STALE_S = 0.5` s, `:63`),
restricted to `VEHICLE_CLASSES` (`:67`), then the **nearest** candidate with `gap >= 0`,
`gap <= LEAD_MAX_GAP_M = 80` m (`:61`) and `abs(lat) < LEAD_LAT_M = 2.0` m (`:59`).
`gap` is **rig-origin to the lead's REAR face** (`along - size_x/2`), *not* bumper-to-bumper —
stated identically in `lead_metrics.py:26-28` and `lead_source.py:44`.
The lead's future track is composed **rig → world → t0-ego** (`lead_track_in_window:274-323`,
with the caveat at `:307-309` that the ego pose is taken at the *cuboid's* timestamp).
At scoring time the corridor gate uses the **predicted path's own local heading**
(`per_step_gap:115-121`), so an arm that drifts out of lane loses its lead.

**What happens when there is no lead — three states, never two** (`lead_source.py:70-72`, assigned `:355-390`, counted `:398-400`):

| situation | reported |
|---|---|
| no `lead=` block passed at all | `status: UNAVAILABLE` + reason + `n: 0` — `four_families.py:349-359` |
| labels present, road clear | `NO_LEAD` |
| no `obstacle.offline` for the clip, **or** the window's horizon leaves the labelled span | `NO_LABEL` — `lead_source.py:367-371`. ⭐ Never counted as free flow (`:406-407`); collapsing it into `NO_LEAD` is named as the bias this instrument exists to avoid (`:31-36`) |
| a block was passed but no window has a lead in-corridor | `status: NOT-APPLICABLE` + reason + n — `lead_metrics.py:200-204` |
| block passed without `eid` | `by_speed: UNAVAILABLE` with reason — `four_families.py:409-415` |

### A.2 LATERAL — `lateral()` `four_families.py:422`

**All four requested metrics exist.** The past failure ("lateral is fine, from cross-track alone")
is closed:

| metric | status | where |
|---|---|---|
| heading error | ✅ EXISTS | `:445` `heading_mae_deg` (path tangent, wrapped `:436-437`) |
| **curvature error** | ✅ EXISTS | `:447-448` `curvature_mae_1pm` **and** `curvature_bias_1pm` |
| **yaw-rate error** | ✅ EXISTS | `:446` `yaw_rate_mae_degps` |
| cross-track | ✅ EXISTS | `:449-451` MAE, signed bias (`+` = drifts LEFT), final MAE |
| denominator transparency | ✅ | `n_steps_heading`, `n_steps_curvature`, `excluded_below_min_ds` `:453-455`; the `min_ds` gate **scales with dt** `:458` |
| dt-invariance declared | ✅ | `:461` — only `yaw_rate` is 1/dt-sensitive |

⚠️ **DEFECT (small, real): the yaw-rate metric carries no `n` and can emit `NaN` instead of `None`.**
`:439` masks yaw-rate with `both_pair`, but `:446` gates its emission on `n_head` (the *heading*
count) and the pair-count is discarded (`yaw_mae, _ = ...`). If a window set has valid single
steps but no valid pairs, `n_head > 0` while the yaw mean is NaN, and `round(degrees(nan), 4)`
is emitted as `NaN` rather than `None`. The correct denominator is already computed — it is
`n_steps_curvature` (same `both_pair` mask, `:440`) — but it is not labelled as the yaw-rate's n.

⛔ **No interval lives inside the LONGITUDINAL or LATERAL blocks.** Neither function takes
`eid` or produces a `ci` key; every number in them is a bare point estimate. Intervals are
computed by the *callers*, and only for a hand-picked subset:

| tool | components given a CI |
|---|---|
| `tools/eval_four_families.py:289-304` | `ade_mean_4wp_m`, `fde_2s_m`, `LON_speed_mae_mps`, `LON_along_mae_m`, `LAT_cross_mae_m`, `LAT_heading_mae_deg`, `LAT_curvature_mae_1pm` |
| `tools/ff_rescore.py:434-469` | the same 7 + `TAC_goal_point_error_m`, `TAC_lat_decision_correct`, `TAC_lon_decision_correct` |
| `tools/t1_eval.py:518-538` | `ade_dense_m`, `fde_last_m`, `LON_speed_mae_mps`, `LON_along_mae_m`, `LAT_cross_mae_m`, `LAT_heading_mae_deg` |

⇒ **no interval anywhere** for `target_speed_acc`, `speed_bias`, `speed_rmse`, `accel_mae`,
`along_bias`, `along_final_bias`, `ego_progress`, `yaw_rate_mae`, `curvature_bias`,
`cross_bias`, `cross_final_mae`, or the pooled `distance_keeping` scalars (`by_speed` does
bootstrap its strata — `lead_metrics.py:241-253`). This is the largest single hole in the
estimator coverage of the binding families.

### A.3 TACTICAL — `tactical()` `four_families.py:948`

Three input paths, in precedence order:

**(1) `hier` — a `hierarchy.run` result** (`:966-996`). This is the only path that scores a
**DECLARED** decision:
- `maneuver_vs_trajectory_kappa` / `_agreement` `:976-977` — ⭐ **this is "selected vs executed"**
- verdict word via the single published `KAPPA_VERDICT_LADDER` (`:85-86`, bands DECORATIVE < 0.1 ≤ WEAK < 0.4 ≤ SUBSTANTIAL), emitted at `:994-995`. The ladder's "must not be restated" note `:74-82` records a real drift that was fixed.
- `seams_beneficial_of_3` / `seam_verdict` `:979-980`; H18 grounded-vs-ungrounded `:982-984`.

**(2) `traj` — `tactical_from_trajectory` `:776`** (T1 dumps, which carry trajectories only):
- **FACTORED** agreement: `lateral_decision` (3-way) `:814`, `longitudinal_decision` (3-way) `:816`, **beside** the collapsed `maneuver_5way_collapsed` `:818`. ⭐ This is the only place the programme's largest known architectural defect — the 5-way softmax that mixes lat and lon — is visible in a metric (`:822-827`).
- labeller is the programme's own canonical `tanitad.refs.refc_tactical.factor_from_kinematics` (`:786-788`); **no thresholds are re-defined here** (`:493-497`).
- each block via `_agreement_block:628`: accuracy `:651`, **Cohen's κ** `:652-653` (`_kappa_k:512`, generalised to k classes, returns `None` — never a fake 1.0 — when `1-pe` vanishes `:654-657`), **per-class recall/precision/support + full confusion matrix + `never_predicted`** (`_class_report:535-567`), and **episode-cluster-bootstrap CIs on BOTH accuracy and κ** `:677-682` (κ bootstrapped by encoding `gt*k+pred` and decoding in a callable reducer `:666-675`).
- `tactical_goal:686` — goal-setting: `goal_bearing_mae_deg`/`_bias_deg` `:735-738` (gated at 0.5 m human reach, excluded windows **counted** `:726, :739-740`), `goal_range_ratio` `:741` (1.0 = correct reach), signed `goal_long_bias_m`/`goal_lat_bias_m` `:743-744`, sign conventions `:745-747`, and `goal_point_error_m` explicitly labelled as **FDE under another name** `:732-734`. CIs at `:759-767`.
- ⛔ **`anchor_selection` is UNAVAILABLE** `:748-755` with reason + n + the instrument that closes it (`taniteval.selgap` over an `<arm>_fan_err`/`<arm>_sel_idx` surface). `taniteval/selgap.py` **exists** (`selgap:129`, `selgap_report:229`, oracle top-k at `TOPK = (4, 8, 16)` `:63`) and `t1_eval.py:606-629` already wires it — it fires the moment a dump carries a fan.
- ⛔ Explicitly **NOT** "selected vs executed" `:828-833`; both streams are EXECUTED manoeuvres.
- ⛔ T0 echo warning `:840-847` — **fires only when `tier` is passed** (see §D).

**(3) `_decision_family:908`** on pre-decoded `win["maneuver_pred"]/["maneuver_gt"]`:
accuracy + per-class recall + `never_predicted` `:928-944`. ⚠️ **No κ and no CI on this path.**
When absent it returns UNAVAILABLE with reason + n `:915-922` plus `how_to_populate` `:1009-1014`.
`rollout.collect` populates it only when a `decision_fn` is supplied (`rollout.py:222-234, 261-263`).

### A.4 STRATEGIC — `strategic()` `four_families.py:1018`

Four paths, in precedence order:

| path | status |
|---|---|
| `no_label=` → `strategic_unavailable:881` | ✅ the honest n/a: `reason` (`STRATEGIC_UNAVAILABLE_REASON:853-868`) + `n` + `instrument_that_would_close_it:870-878` + `_is_a_work_item:895-898`. **Opt-in, never inferred** (`:1026-1029`) — because the fact is a property of the corpus. |
| `optionset=` → `strategic_optionset.strategic_family:558` | ✅ **EXISTS and is the preferred path.** Scores route class against **map-derived option sets**, refuses single-option junctions, compares against the **best constant predictor** by paired episode-cluster bootstrap, and carries a conditioning **echo control** (`conditioning_echo_control:492`) that leaves `STRATEGIC_SKILL_ADMISSIBLE = None` (UNTESTED) when no sweep is supplied (`four_families.py:1063-1068`). |
| `hier` → `seam_nav_to_strategic` `:1079` | ⚠️ PARTIAL. `route_acc_nav/follow/zeronav` `:1084-1086`, majority + chance baselines `:1088-1089`, paired deltas `:1092-1093`. ⭐ `beats_majority_baseline` is computed from **`route_acc_follow` (vision-only)**, never `route_acc_nav`, with the privileged-input reasoning inline `:1100-1111`. Carries the GATE_PROTOCOL §0.7 **void-by-construction** flag `:1095-1098`. |
| `_decision_family` on `route_pred`/`route_gt` | ⚠️ weakest; `how_to_populate` `:1125-1129` says a future-yaw route label is not a substitute. |

**MEASURED strategic label inventory** (loaded the 14 committed reports on CPU from
`stack/experiments/nurec-gsplat/results/strategic_gt/`):

```
scenes 14   ADMISSIBLE 12   refused-by-SELFCONSISTENCY_CONTROL 2
events 45   SCOREABLE (>=2 options) 18   of which on admissible scenes 15
```
The 2 refusals are `302c5c99` and `d1a25a99` — refused, not scored, by
`load_label_reports:213-220`, and surfaced under `_refused` `:222`.
Builder: `stack/experiments/nurec-gsplat/strategic_gt.py` (exists — the docstring pointer at
`four_families.py:1126-1128` resolves).

⇒ **STRATEGIC is not a missing instrument. It is a missing corpus.** 15 events over 12
scene-clusters is a decision-grade denominator of ~12 — enough to build against, far too small
to publish a leaderboard row on.

---

## B. The estimator layer

### B.1 `taniteval/taniteval/ci.py` (357 lines) — what exists

| estimator | line | notes |
|---|---|---|
| `episode_cluster_bootstrap` | `:225` | **Point estimate = `full_set`** (`:245` `point = red(v)`); the bootstrap supplies only the interval. Emits `estimator`, `n_windows`, `n_episodes`, `n_boot`, `reducer` — so a number cannot be quoted without its construction (`:248-258`). |
| `paired_episode_cluster_bootstrap` | `:261` | ✅ **CONFIRMED PRESENT.** Same resampled episodes for both arms each draw (`:281-282`) — explicitly *not* a quadrature combination (`:265-268`). `separated` is decided on the **UNROUNDED** bounds `:288`. |
| `bootstrap_metrics` | `:320` | Whole suite in ONE resampling so intervals are mutually consistent (`:322-328`). |
| `overlapping_holdout_se` | `:121` | ⛔ **STILL DEFINED AND STILL EXPORTED** in `__all__` (`:44`). Self-labels as DEPRECATED (`:124`) but nothing prevents a call. |
| `_render_bounds` | `:69` | Adaptive display precision so a printed interval can never contradict its printed `separated` verdict; `degenerate` marker below 1e-12 `:310-316`. Fixes a measured `{"delta":0.0,"lo":0.0,"hi":0.0,"separated":true}` record (`:72-83`). |

**Resamples / cluster unit — the answers asked for:**
- **`DEFAULT_N_BOOT = 2000`** (`ci.py:55`), used as the default by every tool
  (`eval_four_families.py:105`, `ff_rescore.score_arm:384`, `t1_eval.analyze:311`,
  `lead_metrics` `:407`).
- **Cluster unit = the EPISODE.** `_draws:158-164` draws `n_ep` episode ids **with replacement**
  and concatenates all window indices of each pick — so a resample has variable window count
  but fixed episode count. `episode_index:145-155` **fails loud on an empty eid** (`:152-153`),
  because a silent zero-episode bootstrap would emit a NaN interval that reads like a pass.
- Reducers: `mean`/`rms`/`median`/`p90`/`p10` (`REDUCERS:196-197`) **plus arbitrary callables**
  (`resolve_reducer:200-213`) — which is what lets κ, macro-F1 and AUC use the same estimator
  instead of inventing their own (`:203-206`).

### B.2 Guards — verified by reading the code, not the docstring

**Guard 1 — `driving.assert_no_deprecated_estimator`, `taniteval/taniteval/driving.py:518-547`.**
This is a **genuine refusal to emit**. It walks an emitted block recursively and **raises**
`ValueError` when (a) any `estimator` string is not in `DECISION_ESTIMATORS`
(`driving.py:154-155` = `{episode_cluster_bootstrap, paired_episode_cluster_bootstrap}`), or
(b) any dict with both `lo` and `hi` carries **no named estimator at all** (`:539-541`).

Call sites (product code only):

| caller | line |
|---|---|
| `driving.tier0` | `driving.py:788` |
| `hierarchy.run` | `hierarchy.py:1147` — run **before** `LEGACY_BLOCK` is attached at `:1148` |
| `closedloop` | `closedloop.py:987` |
| `planner_p2` | `planner_p2.py:645`, `:819` |
| `corridor` | `corridor.py:468` |
| `lateral` (the standalone decomposition block) | `lateral.py:448` |
| `strategic_probes` | `strategic_probes.py:420` |

⛔ **NOT called by** `four_families.all_families`, `tools/eval_four_families.py`,
`tools/ff_rescore.py`, `tools/t1_eval.py`, or `bench.run`. Those paths do not currently call the
banned estimator, but **nothing mechanically stops a future edit from doing so.**

**Guard 2 — `taniteval/taniteval/gate_guard.py` (253 lines) + `tests/test_no_jack_in_gates.py`.**
An **AST taint walk** (deliberately not a regex, `:3-18` — a regex matches its own retirement
notice, the same self-match as `pgrep -f <trainer>`). It propagates taint to a fixpoint
(`_tainted_names:149-183`), resolves import aliases (`banned_import_aliases:100-112`), catches
inlined calls and subscript stores (`_deciding_exprs:186-206`), and exempts only an explicit
`_LEGACY` suffix (`_LEGACY_RE:64`). Eight negative controls and two false-positive controls pin
it (`test_no_jack_in_gates.py:147-202`).

⚠️ **Two structural limits, both verified:**
1. **It only fires on a *verdict-named* binding.** `_DECIDING_RE` (`gate_guard.py:61-62`) matches
   `pass|passed|verdict|gate(d)_ok|admissible` or `^G\d+_pass`. A block that *emits* a banned
   value under any other key passes cleanly. `bench.py:303 "heldout"` passes. So does
   `runner.regression`, whose PASS/FAIL is carried by a local named `ok`
   (`runner.py:425-435`) — not a deciding name — while its input can be the deprecated
   `heldout` mean (`runner.py:409-410`). **A golden-regression verdict computed from the
   deprecated estimator is invisible to this guard.** (It is not silent: `runner.py:416-420`
   prints a loud warning and records the source per arm.)
2. **Scope.** `ENFORCED_ROOTS` (`test_no_jack_in_gates.py:35-38`) is
   `taniteval/taniteval`, `taniteval/tools`, `stack/tanitad`, `stack/scripts`, with
   `SKIP = ("__pycache__", "/.claude/", "/experiments/")` (`:39`). **`TanitAD Research Lab/**` is
   outside the enforced scope entirely** — and that is where 12 of the live call sites live (§B.3).

**Guard 3 — `ff_rescore` CLI flag check, `tools/ff_rescore.py:547-556`.** Refuses any
`--estimator` other than `episode_cluster_bootstrap`, with a tailored message when the requested
name matches `BANNED_ESTIMATORS` (`:80-81`). This is a **CLI argument check only** — it does not
scan output.

### B.3 ⛔ EXHAUSTIVE `overlapping_holdout_se` / `_jack*` INVENTORY

Produced by an **AST walk over every `.py` in the repo** (`ast.FunctionDef` / `ast.Call` /
`ast.Import` on the banned name set from `gate_guard.BANNED_CALLS` + `^_{0,2}jack(_|$)`), so a
comment or docstring can never be miscounted as code. **Classification key:**
**(a)** live code that could emit the value · **(b)** deprecated/labelled output column ·
**(c)** comment/docstring/test-fixture only.

#### Definitions — 5, all class (a)

| file:line | symbol | class | note |
|---|---|---|---|
| `taniteval/taniteval/ci.py:121` | `def overlapping_holdout_se` | **(a)** | the canonical implementation; exported in `__all__:44`. Nothing prevents a call. |
| `taniteval/taniteval/closedloop.py:605` | `def _jack` | **(a)** | wraps `ci.overlapping_holdout_se` at `:615` |
| `taniteval/taniteval/hierarchy.py:440` | `def _jack` | **(a)** | wraps it at `:476` |
| `taniteval/taniteval/planner_p2.py:512` | `def _jack_scalar` | **(a)** | wraps it at `:522` |
| `taniteval/taniteval/planner_p2.py:527` | `def _jack_paired` | **(a)** | wraps it at `:538` |

#### Product call sites — 19

| file:line | class | disposition |
|---|---|---|
| `taniteval/taniteval/bench.py:133` (inside `_agg:115`) | **(a) + (b)** | ⛔ **THE ONE THAT MATTERS.** The result is emitted **twice** by `bench.run`: at `:302` under `LEGACY_BLOCK = "legacy_overlapping_holdout_se"` (`:173`, correctly quarantined + `_quarantine:207` wrapper) **and again at `:303` under the bare key `"heldout"`**. The leaf dicts self-label (`"estimator": "overlapping_holdout_se", "deprecated": True`, `:135-136`), but the key does not end in `_LEGACY`, so `gate_guard` cannot see it — and `runner.py:409-410` reads `d["heldout"]["model"]` into the golden regression comparison. `bench.run` never calls `assert_no_deprecated_estimator`. |
| `taniteval/taniteval/closedloop.py:599` (inside `_agg`) | (b) | reached only from the quarantined `legacy` block, `:881`, `:890-900` |
| `taniteval/taniteval/closedloop.py:615` (inside `_jack`) | (b) | ditto |
| `taniteval/taniteval/closedloop.py:883`, `:886`, `:889` | (b) | all inside the `legacy` dict built at `:879-900`; guard runs at `:987`, block is excluded by name (`:1000`) |
| `taniteval/taniteval/hierarchy.py:476` (inside `_jack`) | (b) | |
| `taniteval/taniteval/hierarchy.py:1200, 1210, 1211, 1212, 1220, 1222, 1226, 1228, 1231` | (b) | **9 call sites**, all inside `_legacy(...)`, attached at `:1148` **after** the guard at `:1147`. Correctly quarantined. |
| `taniteval/taniteval/planner_p2.py:563`, `:610`, `:789` | (b) | the G1/G4 gates were migrated 2026-08-16; `tests/test_no_jack_in_gates.py:57-74` pins that `G1_pass`/`G4_pass` still exist and that both decision-grade estimators are reachable in the file. |
| `taniteval/recompute_ci.py:91` | **(a)** | a **standalone script at the `taniteval/` top level**, i.e. outside `ENFORCED_ROOTS` (`taniteval/taniteval` and `taniteval/tools` only). It can recompute and print the banned value with no guard in the way. |

#### Test call sites — 4, class (c) (positive controls / reproduction pins)

`taniteval/tests/test_ci.py:86`, `:95` · `taniteval/tests/test_driving.py:367` ·
`taniteval/tests/test_hierarchy_ci.py:253`. These *should* exist — they are what proves the
deprecated value still reproduces and that the guards bite.

#### ⛔ Outside the enforced scope — 12 live sites under `TanitAD Research Lab/`

All class **(a)**: real, runnable code that computes the banned statistic, in a tree the
guard test does not scan.

| file:line |
|---|
| `…/Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/closedloop.py:332` (`def _jack`), calls `:420`, `:422`, `:432` |
| `…/incoming/2026-07-25-v16-paired-interval/verify_v16_paired.py:34` (**import**), calls `:117`, `:118` |
| `…/incoming/2026-07-26-closedloop-artifact-rerun/harness_check.py:71` |
| `…/incoming/2026-07-26-closedloop-artifact-rerun/rerun_planner_p2_g4.py:66` |
| `…/incoming/2026-08-16-jack-in-gates/code/recompute_g1_g4.py:75`, `:85` (`def jack_scalar`, `def jack_paired`), calls `:185`, `:245`, `:277`, `:326` |
| `…/incoming/2026-08-18-planner-beats-cv-redrive/code/redrive_planner_vs_cv.py:104` |

Several of these are *deliberate* reproduction harnesses (the 2026-08-16 package is the audit
that found the defect). The finding is not that they exist — it is that **the guard test would
not catch a new one written there.**

#### ✅ Clean

**Zero** definitions, calls or imports of the banned family under `stack/tanitad/` or
`stack/scripts/` — every hit there is prose. Same for `taniteval/taniteval/four_families.py`,
`taniteval/taniteval/ci.py` (beyond the definition), and all three four-family tools.

---

## C. The data adapter surface

### C.1 How evaluation data is loaded today — two entry paths

**Path 1 — inference from an episode corpus** (`tools/eval_four_families.py:137-199`):
```
--corpus <dir of ep_*.pt>
   -> data.load_frames(files)                       taniteval/taniteval/data.py:230
        -> tanitad.data.mixing.load_episode(f, mmap=True)
        -> RawEp                                    data.py:220-227
   -> rollout.collect(model, step_readout, eps, …)  rollout.py:139
   -> hierarchy.run(...)                            (TACTICAL + STRATEGIC)
   -> four_families.all_families(win, hier=hier)    eval_four_families.py:233
```
⚠️ The canonical episode listing is `data.list_val_episodes:76`, which is **PhysicalAI-specific**:
it refuses `LEAKY_VAL = "physicalai-val-f1b378f295ae"` by default (`data.py:37-38`) and calls
`tanitad.data.parity.assert_val_cache` (`:127`) before a single episode is read. `eval_four_families.py`
bypasses it with a raw `glob` (`:137`), which is what lets it score an arbitrary corpus (`:25-28`).

**Path 2 — rescore an already-banked dump, no GPU** (`tools/ff_rescore.py:150 load_dump`):
accepts **either** a directory of per-episode `ep*.npz` (the T1 schema) **or** a
`rollout.collect` `.pt`. This is the seam that matters for NavSim/nuScenes.

### C.2 The input contract

**Episode record** — `RawEp`, `data.py:220-227`:

| attribute | shape / dtype | meaning |
|---|---|---|
| `.feats` | `[T, 9, S, S]` uint8 (mmap) | 3-frame RGB stack; `[T, 256, d]` fp16 for frozen-encoder arms (`FeatEp:140-147`) |
| `.actions` | `[T, A]` float | corpus action contract; `A=2` base, `+v0` with `--speed-input`, `+yaw` with `--dyn-input` (`rollout.ego_action_channels:76-93`) |
| `.poses` | `[T, 4]` float | **`(x, y, yaw, v)`** — verified: `[:2]` and `[2]` in `driving_diagnostic.gt_ego_waypoints:103-105`, `[3:4]` as speed in `rollout.py:88` |
| `.episode_id` | int | the bootstrap **cluster key** |

**Window geometry** — `rollout.collect:139`: `window = 8` context frames, `fwd_k = K_MAX = 20`
steps (`rollout.py:70`), `stride = 8`, `DT = 0.1` s (`:72`). Starts enumerated at
`range(0, T - window - K_MAX, stride)` (`:172`); a window's origin is the pose at
`start + window - 1` (`:175`).

**Horizon and sample rate:** dense = **20 steps at 10 Hz = 2.0 s**; sparse =
`WP_STEPS = (5, 10, 15, 20)` (`driving_diagnostic.py:74`) = **0.5 / 1.0 / 1.5 / 2.0 s**.

**Coordinate frame — the conversion is `driving_diagnostic._ego`, `stack/scripts/driving_diagnostic.py:87-90`:**
```python
def _ego(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[...,0]*c - dxy[...,1]*s,
                        dxy[...,0]*s + dxy[...,1]*c], dim=-1)
```
applied in `gt_ego_waypoints:101-106` as `_ego(poses[last+k, :2] - poses[last, :2], poses[last, 2])`.

| convention | value | stated at |
|---|---|---|
| origin | ego pose at the window's **last observed frame** | `driving_diagnostic.py:103-104` |
| `x` | **forward (along-track)**, metres | `four_families.py:33-34`, `lead_metrics.py:24-25`, `lead_source.py:40` |
| `y` | **left (cross-track)**, metres | same; `cross_bias_m` `+` = drifts LEFT (`four_families.py:450`) |
| heading | `atan2(dy, dx)`, **radians internally**, reported in **degrees** (`four_families.py:445-446`) | `four_families.py:170`, `:39-40` — it is the **path tangent**, not vehicle yaw |
| curvature | `dheading/ds`, **1/m**, dt-invariant | `four_families.py:176-180` |
| yaw rate | rad/s internally, **deg/s** reported; scales as 1/dt | `four_families.py:175`, `:446`, `:461` |
| `along` sign | `+` = model is AHEAD of the human (over-predicted speed) | `four_families.py:36-38` |
| global frame | **clip-local metres. `egomotion` carries no lat/lon/GNSS** | `lead_metrics.py:25-26`, `four_families.py:857-858` |

⭐ **A verification instrument for the frame already exists:** `lateral.assert_axis_convention`
(`taniteval/taniteval/lateral.py:170-205`) raises unless `mean abs(axis0)` dominates
`mean abs(axis1)` at the final step **and** matches `v * K * dt` within `tol = 0.35`. **Any new
dataset adapter should call this first** — it is the cheapest possible catch for a transposed or
`(y, x)` dump.

**dt is DERIVED, never assumed** — `four_families.infer_dt:190-219` reads `win["wp_steps"]` ×
`win["dt_s"]`, returns a provenance string, and refuses to guess silently. The defect this
closes is documented with a negative control at `:130-146`: GT ego speed **12.4565 m/s** read as
**62.9789 m/s** (×5.0559) on 859 real windows, because a 0.5 s sparse grid was divided by 0.1.

### C.3 ⛔ `taniteval/results/windows_*.pt` — MEASURED structure

Loaded on CPU with `torch.load(..., weights_only=False)`. **All three sampled dumps have
exactly 7 keys** — no dense path, no ctrv, no decisions, no `pc2`, no `dt_s`:

```
windows_flagship-30k.pt      (96,104 B)
  'pred'      Tensor (881, 4, 2) float32   min  -9.6848  max  75.4214
  'gt'        Tensor (881, 4, 2) float32   min  -8.8187  max  73.5399
  'cv'        Tensor (881, 4, 2) float32   min  -1.3434  max  73.5826
  'eid'       list  len=881   40 unique    head [0, 0, 0, 0, 0, 0]
  'speed'     Tensor (881,)     float32    min   0.0000  max  36.5479
  'head_deg'  Tensor (881,)     float32    min   0.0000  max  67.8573
  'wp_steps'  list  [5, 10, 15, 20]

windows_refc-v12-smoke-t0.pt (12,191 B)  same 7 keys, N=88, 4 episodes
windows_flagship-v4.2-step4000.pt (98,902 B)  same 7 keys, N=881,
  BUT 'eid' head = [808464434, …]  == big-endian ASCII b'0002'
```

Three consequences, all verified:

1. **Every committed dump is pre-2026-07-25 sparse-only.** `rollout.py:34-39` records that a
   dense-carrying dump measures **378,359 B**; all 27 committed dumps are 96–99 KB. Confirmed by
   content, not by size alone.
2. ⛔ **`tools/eval_four_families.py:263-264` does `win["pred_dense"].float()` unconditionally**
   ⇒ `KeyError` on every one of them. Its `--windows-in` rescore mode
   (`:110-115`, `:149-152`) is therefore **dead against the banked corpus**.
   `ff_rescore.load_dump:195-197` uses `.get` and degrades to the sparse view with `infer_dt`
   (`:210-214`) — that is the working rescore path.
3. **The packed-ASCII `eid` defect is real and still in the bytes.** `808464434 == b'0002'`.
   It is repaired at *both* write and read time — `rollout.normalise_eid:325-347`,
   `save_windows:350-362`, `load_windows:365-380` — and `load_windows` is what
   `ff_rescore.load_dump:191` calls, so a rescore joins correctly. A raw `torch.load` does not.

### C.4 The T1 npz dump schema (`tools/t1_eval.py:91-102`)

```
ep*.npz  (one file per episode; sorted order = episode order)
  g    [N,K,2]  GT ego-frame waypoints                REQUIRED (--gt-key, default "g")
  cl   [N,K,2]  closed-loop arm            tier T1
  ol   [N,K,2]  open-loop teacher-forced   tier T0    (--with-t0-open-loop)
  ha   [N,K,2]  hold-action control        tier T1    (--with-hold-action)
  ws   [N]      window origin frame indices           (provenance, optional)
  eid, clip_index                                     METADATA, not arms (t1_eval.py:157)
  <arm>_fan_err [N,C] / <arm>_sel_idx [N] / <arm>_fan_scores [N,C]   -> selgap fires
```
Any other non-suffixed key is an arm and **MUST** carry a tier or `analyze` raises
(`resolve_tiers:295-305`).

### C.5 ⭐ THE ADAPTER SEAM — what a NavSim/nuScenes adapter must implement

**The whole scoring stack below `load_dump` is dataset-agnostic already.** It consumes a plain
dict of tensors; nothing under `four_families.py`, `ci.py`, `lead_metrics.py` or `selgap.py`
knows what PhysicalAI is.

**The minimum viable adapter is one function** that returns this dict:

```python
win = {
    # REQUIRED
    "pred_dense": [N, K, 2] float,   # arm path, EGO-FRAME METRES, x fwd / y left,
                                     #   origin = ego pose at the window's last observed frame
    "gt_dense":   [N, K, 2] float,   # human path, same frame, same K
    "pred":       [N, M, 2] float,   # sparse companion view (pred_dense[:, idx])
    "gt":         [N, M, 2] float,
    "wp_steps":   [...],             # MODEL-TICK contract of the sparse columns (e.g. [5,10,15,20])
    "dt_s":       0.1,               # spacing of the DENSE columns, seconds
    "eid":        [N],               # episode/scene/log id -> the BOOTSTRAP CLUSTER UNIT
    # OPTIONAL but each unlocks a family
    "speed":      [N] float,         # ego v at t0 -> the anti-echo controls (else UNAVAILABLE)
    "lead":       {...},             # -> distance-keeping (see below)
    "optionset":  {...},             # -> STRATEGIC (see below)
    "maneuver_pred"/"maneuver_gt"/"route_pred":  # -> the declared-decision TACTICAL path
}
four_families.all_families(win, tactical_from_traj=True, tier="T1",
                           strategic_no_label=<bool>, n_boot=2000, seed=0)
```

**The named seams, in the order a NavSim adapter would touch them:**

| # | seam | file:line | what to do |
|---|---|---|---|
| 1 | **`load_dump(label, path, gt_key, dt)`** | `tools/ff_rescore.py:150` | ⭐ **The format seam.** Add a third branch beside `t1_npz_dump` (`:153-185`) and `rollout_windows` (`:189-229`). Must return `{"kind", "gt", "eid", "dt_s", "wp_steps", "source", "n_episodes", "v0", "arms": {name: (key, [N,K,2])}}`. Everything after this is free. |
| 2 | `score_arm(...)` | `tools/ff_rescore.py:381` | **No change needed.** Builds the `win` dict at `:404-406`, forms the sparse companion view at `:399-403`, calls `all_families` at `:425-427`, and bootstraps 10 components at `:434-456`. |
| 3 | `fingerprint` / `same_grid` | `tools/ff_rescore.py:235`, `:257` | **No change needed**, but the adapter must produce a *stable* `eid` order — the fingerprint sha1s the rounded GT bytes **and** the eid sequence, and a mismatch REFUSES the cross-arm comparison. |
| 4 | `canon_eid` / `_runs` | `tools/ff_rescore.py:265`, `:283` | Handles `ep_00000` vs `0` id conventions. A nuScenes `scene_token` (no trailing integer) falls back to raw strings (`:277-278`) — which is fine, but then a lead block keyed differently will not join. |
| 5 | `resolve_tier` / `DEFAULT_TIERS` | `tools/ff_rescore.py:126`, `tools/t1_eval.py:145` | ⛔ **A new arm key is a HARD ERROR until tiered.** Either add it to `DEFAULT_TIERS` or pass `--tier NAME=T0|T1`. Do not add a default. |
| 6 | `lead_block(t0s, ts_rel, obs, ego, …)` | `taniteval/lead_source.py:329` | Distance-keeping seam. Needs `obs = {t, track, center_x, center_y, size_x, is_vehicle}` in the **rig frame** and `ego = {t, x, y, yaw, v}`. NavSim/nuScenes have exactly this (`sample_annotation` + `ego_pose`). Output plugs straight into `win["lead"]`. Set `win["lead"]["path_steps"]` when the lead grid is coarser than the path (`four_families.py:371-390`). |
| 7 | `load_label_reports(source)` | `taniteval/strategic_optionset.py:193` | STRATEGIC seam. Expects `strategic_gt_*.json` with `ADMISSIBLE`, `SELFCONSISTENCY_CONTROL`, and `events[].SCOREABLE`. ⭐ **A map-carrying benchmark is what makes STRATEGIC computable at all** — nuScenes ships a lane graph, NavSim ships route/PDM context. Builder to imitate: `stack/experiments/nurec-gsplat/strategic_gt.py`. |
| 8 | `rollout.collect` | `taniteval/rollout.py:139` | ⛔ **Do NOT route a new benchmark through here.** It hard-depends on `tanitad.models.metric_dynamics.rollout_decode`, `driving_diagnostic.gt_ego_waypoints/baseline_waypoints`, `refb_labels`, and the `RawEp` surface. It is the *inference* seam, not the *data* seam. |
| 9 | `data.list_val_episodes` | `taniteval/data.py:76` | ⛔ **A refusal point.** Hard-codes `CLEAN_VAL`/`LEAKY_VAL` (`:37-38`) and calls `tanitad.data.parity.assert_val_cache` (`:127`). A new corpus needs its own parity record or an explicit bypass (as `eval_four_families.py:137` already does). |
| 10 | `tanitad.refs.refc_tactical.factor_from_kinematics` | imported at `four_families.py:786` | ⛔ **TACTICAL needs `stack/` on `PYTHONPATH`.** Without it the family returns UNAVAILABLE with the reason (`:789-797`). Not a code change — a deployment constraint. |
| 11 | `lateral.assert_axis_convention` | `taniteval/lateral.py:170` | ⭐ **Call it in the adapter's own tests.** Cheapest possible catch for a transposed frame. |

**Effort estimate for a NavSim adapter reaching LONGITUDINAL + LATERAL + TACTICAL:**
one `load_dump` branch + a frame/units converter + tier registration ≈ **1–2 days**, assuming
the NavSim side can emit per-scene ego-frame arrays. Distance-keeping adds ~1 day (seam 6).
STRATEGIC adds the label builder (seam 7), which is the real work.

---

## D. Tier stamps (T0 / T1 / T2)

**T2 (re-perception sim) does not appear in the taniteval tier vocabulary at all.** `DEFAULT_TIERS`
(`tools/t1_eval.py:145-149`) and `_TIER_NOTE` (`:150-153`, `tools/ff_rescore.py:91-97`) know only
`T0` and `T1`, and `resolve_tiers:299` accepts only those two.

### Where the tier IS recorded

| producer | evidence |
|---|---|
| `tools/t1_eval.py` | per-arm `"tier"` + `"tier_note"` `:649-651`; on each family `:500-501`; on `s_curve` `:541`, `lag` `:575`, `response` `:588,603`, `sel_gap` `:620`, `intervals` `:654`; cross-tier deltas labelled `"T1 minus T0"` `:697`. **`resolve_tiers:295-305` raises on an unstamped arm.** |
| `tools/ff_rescore.py` | `"tier"` + `"tier_note"` `:477-478`; stamped onto **every** bootstrap block `:470-471`; `_tier_doctrine` `:495-499`; ⛔ `_cross_tier_warning` `:711-716` when a paired delta crosses tiers. **`resolve_tier:126-141` `_die`s on an unstamped arm.** |
| `four_families.all_families` | `_tier` `:1248` and per-family `tier` `:1249-1251` — **only when the caller passes `tier=`** (default `None`, `:1137`) |
| `p7_strata.py:311, 410` · `tools/p7_per_stratum.py:175` | default `"T0"` |
| `seam.py:619, 743, 811` | literal `"T1"`; `seam.py:875` falls back to the string `"UNSTAMPED"` |
| `seam_dump.py:83, 154` | `tier` is a **required** keyword |
| `degeneracy.py:283` | `"eval_tier": "T0-DIAGNOSTIC"` |

### ⛔ Where a T0 number could be reported as driving performance — three routes

1. ⛔ **`tools/eval_four_families.py` emits an UNSTAMPED record.** The file contains the substring
   `tier` **zero times** (verified). `:233` calls `ff.all_families(win, hier=hier)`; the output
   record `:306-336` has no `tier` key. Its LONGITUDINAL/LATERAL surface is `rollout.collect`,
   which is fed **the expert's true future actions** — `rollout.py:151-160` says so, sets
   `pc2["pc2_pass"] = False` **by construction** (`:236-245`) and names the honest metric
   `"wm_fidelity_ade_2s"` (`:246`). The record does carry `"pc2": win.get("pc2")` (`:322`), so the
   T0-ness is *recoverable* — but only by a reader who knows to look, and no field says "T0".
2. ⛔ **`all_families(tier=None)` silently disables the T0 echo warning.** The
   `⛔_tier_warning` at `four_families.py:840-847` — "*this path is TEACHER-FORCED … substantially
   an ECHO of the label's own source*" — is inside `if tier is not None:` (`:838`). Calling
   `all_families(win, tactical_from_traj=True)` on a T0 dump therefore returns a **tactical
   agreement number with no echo warning and no tier key at all.**
3. ⚠️ **`driving_*.json` encodes the tier only in a block NAME.** MEASURED on
   `taniteval/results/driving_flagship-30k.json`: top-level keys are
   `block, version, spec, arm, n_windows, …` with `block = "taniteval.driving/tier0"`
   (`driving.py:575`, `BLOCK` at `:82`-equivalent). **There is no `tier` key.** The block does
   carry `claim_strength: "open-loop / weak (arXiv:2605.00066)"` (`:585`) and a full `estimator`
   sub-dict (`:586-594`), which is honest — but a machine reading `tier` finds nothing.

**Recommendation (one line of code):** make `tier` a **required** argument of
`four_families.all_families` and raise when it is `None`, mirroring `resolve_tiers:299-305`. That
closes routes 1 and 2 at the same point.

---

## E. Test coverage — RUN, with honest counts

`taniteval/tests/` holds **66 files**, **63** named `test_*.py`, plus `run_all.py` and
`pin_bench_legacy_block.py`. The suite is **CPU-only, needs no GPU and loads no checkpoint** — so
it was run.

| run | environment | result |
|---|---|---|
| 1 | in-repo, on the Drive mount | ⛔ **could not collect** — `ImportError while loading conftest`, `OSError: [Errno 22] Invalid argument` inside `pathlib.read_bytes`. Environment, not harness (whole-file reads fail on this mount; chunked reads of the same bytes succeed). |
| 2 | byte-identical local mirror, default cp1252 console | **28 failed · 1098 passed · 26 skipped · 88.15 s** |
| 3 | same mirror, `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` | ⭐ **6 failed · 1120 passed · 26 skipped · 85.46 s** |

**All 28 failures are explained, and none is a harness defect:**

- **22 of 28 are a cp1252-console defect in three CLIs.** They fail with
  `UnicodeEncodeError`/`UnicodeDecodeError: 'charmap' codec can't … '⛔'` (the ⛔ character)
  when a test either `read_text()`s the tool's source or runs it as a subprocess to print
  `--help`. Affected: `tools/eval_four_families.py` (11 tests), `tools/render_openloop_video.py`
  (10), `tools/t1_eval.py` (1). **Proven environmental:** re-running exactly those three files
  with `PYTHONUTF8=1` gave `52 passed in 128.36s`. ⚠️ It is still a real, if minor, defect —
  **those three CLIs cannot print their own `--help` on a Windows cp1252 console.**
- **6 of 28** are `tests/test_c2_published_policy.py`, which asserts on
  `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-26-v5-imagination-selection/raw/v5_{v1,v4}_windows_reduced.pt`.
  **Those files exist in the repo (14,569,689 B each)**; my mirror omits them (the path exceeds
  Windows `MAX_PATH`). **A mirror artifact, not a harness defect.**

⇒ **Best available evidence: the taniteval suite is green** on a UTF-8-capable host with the full
repo. The one genuine finding is the cp1252 `--help` breakage.

**What the tests actually pin** (a strong signal for benchmark adoption — these are the
invariants a new adapter must not break):
`test_no_jack_in_gates.py` (the AST estimator guard, 8 negative + 2 false-positive controls) ·
`test_ci.py` (point estimate == full_set) · `test_val_parity.py` · `test_warp_geometry.py` ·
`test_lateral.py` · `test_lead_metrics.py` / `test_lead_source.py` / `test_lead_strata.py` ·
`test_four_families_dt.py` (the ×5/×25 dt defect) · `test_eval_four_families_tool.py` ·
`test_t1_eval.py` · `test_eid_normalisation.py` · `test_ego_guard.py` · `test_stack_guard.py` ·
`test_strategic_optionset.py` · `test_selgap.py` · `test_cl_metrics_stamps.py`.

---

## F. Capability matrix (summary)

### Metrics

| capability | verdict | anchor |
|---|---|---|
| LON — target-speed accuracy | ✅ EXISTS | `four_families.py:276-278` |
| LON — speed MAE/bias/RMSE, along, accel | ✅ EXISTS | `:268-288` |
| LON — ego progress | ✅ EXISTS | `:294` |
| LON — headway | ✅ EXISTS | `lead_metrics.py:170` |
| LON — time gap (THW) | ✅ EXISTS | `lead_metrics.py:171-172` |
| LON — min TTC (censored) | ✅ EXISTS | `lead_metrics.py:174-185` |
| LON — lead identification, causal, 3-state | ✅ EXISTS | `lead_source.py:235-271`, `:355-400` |
| LON — anti-echo (hold-v0) | ✅ EXISTS | `four_families.py:305`, `:1222` |
| LAT — heading error | ✅ EXISTS | `:445` |
| LAT — curvature error + bias | ✅ EXISTS | `:447-448` |
| LAT — yaw-rate error | ⚠️ PARTIAL | `:446` — exists, but **no `n`**, and NaN-instead-of-None |
| LAT — cross-track (MAE/bias/final) | ✅ EXISTS | `:449-451` |
| TAC — manoeuvre decision quality (acc + κ + confusion + per-class) | ✅ EXISTS | `_agreement_block:628-683` |
| TAC — factored lat/lon (the 5-way-mixing defect) | ✅ EXISTS | `:814-819` |
| TAC — **selected vs executed** | ⚠️ PARTIAL | only on the `hier` path `:976-977`; UNAVAILABLE on any trajectory dump `:828-833` |
| TAC — goal setting (bearing / range ratio / signed biases) | ✅ EXISTS | `tactical_goal:686-773` |
| TAC — **anchor/goal SELECTION** | ⛔ MISSING (instrument ready) | declared UNAVAILABLE `:748-755`; `selgap.py:129` exists and `t1_eval.py:607-623` wires it — needs a fan surface in the dump |
| STR — decision + route/goal quality (option sets) | ⚠️ PARTIAL | `strategic_optionset.strategic_family:558` works; **15 scoreable events / 12 clusters** banked |
| STR — on PhysicalAI-AV | ⛔ CORPUS-BLOCKED | `STRATEGIC_UNAVAILABLE_REASON:853-868` |
| Per-family, never pooled | ✅ COMPLIANT | `:1186-1194` |
| Reason + n when unavailable | ✅ COMPLIANT | `:1238-1246` |
| `n` when available | ⚠️ PARTIAL | LON/LAT emit `n_windows`, not `n` |

### Estimator

| capability | verdict | anchor |
|---|---|---|
| episode-cluster bootstrap | ✅ EXISTS | `ci.py:225`, n_boot 2000 (`:55`), unit = episode (`:158-164`) |
| **paired** form | ✅ EXISTS | `ci.py:261` |
| point estimate = `full_set` | ✅ EXISTS | `ci.py:245` |
| whole-suite single resampling | ✅ EXISTS | `ci.py:320` |
| callable reducers (κ, F1, AUC) | ✅ EXISTS | `ci.py:200-213` |
| refusal-to-emit guard | ⚠️ PARTIAL | `driving.py:518` is real, but **not wired into the four-family path** |
| gate-decision AST guard | ⚠️ PARTIAL | `gate_guard.py` + `test_no_jack_in_gates.py:45`; blind to non-verdict keys and to `TanitAD Research Lab/**` |
| `overlapping_holdout_se` removed | ⛔ NO | 5 defs, 19 product calls, 12 more out of scope (§B.3) |
| CI on every binding metric | ⛔ MISSING | only 7–10 components per tool (§A.2) |

### Adapter

| capability | verdict | anchor |
|---|---|---|
| single format seam | ✅ EXISTS | `ff_rescore.load_dump:150` |
| dataset-agnostic scoring core | ✅ EXISTS | `four_families.py`, `ci.py`, `lead_metrics.py`, `selgap.py` |
| cross-grid join refusal | ✅ EXISTS | `fingerprint:235`, `same_grid:257` |
| frame-convention verifier | ✅ EXISTS | `lateral.assert_axis_convention:170` |
| dt derived, not assumed | ✅ EXISTS | `infer_dt:190` |
| NavSim / nuScenes loader | ⛔ MISSING | — |
| corpus-agnostic episode listing | ⚠️ PARTIAL | `data.list_val_episodes:76` is PhysicalAI-specific |
| banked dumps usable by the named tool | ⛔ BROKEN | `eval_four_families.py:263` KeyErrors on all 27 |
| T2 tier | ⛔ MISSING | not in `DEFAULT_TIERS:145-149` |

---

## G. Ranked GAP LIST — by value-per-effort for adopting community benchmarks

| # | gap | effort | why this order |
|---|---|---|---|
| **1** | **Make `tier` mandatory in `four_families.all_families`** (`:1137`) — raise when `None`, mirroring `t1_eval.resolve_tiers:299-305` — and pass a tier from `tools/eval_four_families.py:233`, adding it to the record at `:306-336`. | **~1 h** | Closes the single route by which a teacher-forced number can leave the harness unlabelled, and re-arms the T0 echo warning (`:840-847`). Highest value per hour in the whole list. |
| **2** | **Fix `tools/eval_four_families.py:263-264`** to `win.get("pred_dense")` with a sparse fallback (copy `ff_rescore.load_dump:195-214`, which already does it). | **~2 h** | The named four-family tool currently `KeyError`s on **all 27** banked dumps. Without this, "rescore the leaderboard from what we have" is impossible. |
| **3** | **Write the NavSim adapter as one `load_dump` branch** (`ff_rescore.py:150`) + a frame/units converter + a `DEFAULT_TIERS` entry, and call `lateral.assert_axis_convention` in its tests. | **1–2 d** | The scoring core is already dataset-agnostic. This is the whole adoption cost for LON + LAT + TAC + ADE with paired episode-cluster CIs. |
| **4** | **Give every binding metric a CI.** Extend the `bootstrap_metrics` component maps (`eval_four_families.py:289-295`, `ff_rescore.py:434-442`, `t1_eval.py:518-524`) to cover `target_speed_acc`, `speed_bias`, `accel_mae`, `along_bias`, `yaw_rate_mae`, `curvature_bias`, `cross_bias`, `ego_progress`, and the pooled `distance_keeping` scalars. | **1 d** | The binding rule says each family carries its estimator and CI. Today 7–10 components do; ~15 do not. A leaderboard built on point estimates alone is not decision-grade. |
| **5** | **nuScenes adapter** (same shape as #3) **plus a `lead_block` builder** from `sample_annotation` + `ego_pose` (`lead_source.lead_block:329`). | **2–3 d** | nuScenes gives the lead tracks for free, which turns the distance-keeping half of LONGITUDINAL from UNAVAILABLE into a number on a public corpus. |
| **6** | **Wire the refusal guard into the four-family path** — call `driving.assert_no_deprecated_estimator` at the end of `all_families:1252` and in all three tools. | **~2 h** | Cheap, and it makes the binding-family path structurally unable to emit a banned interval, rather than merely not doing so today. |
| **7** | **Retire the bare `"heldout"` key** at `bench.py:303` (keep `LEGACY_BLOCK` at `:302`), and delete the `heldout` fallback at `runner.py:409-410`. | **~3 h** | The only place a deprecated **point estimate** still reaches a gate-facing comparison. Requires regenerating `results/golden.json`. |
| **8** | **Extend `ENFORCED_ROOTS`** (`test_no_jack_in_gates.py:35-38`) to `TanitAD Research Lab/**` with an explicit per-file allowlist for the deliberate reproduction harnesses. | **~4 h** | 12 live banned-estimator sites are currently unguarded. Effort is mostly triaging the allowlist. |
| **9** | **Emit a fan+selector surface** (`<arm>_fan_err`, `<arm>_sel_idx`) from the T1 roll so `selgap` fires and `anchor_selection` stops being UNAVAILABLE (`four_families.py:748-755`). | **1–2 d** | The consumer already exists (`selgap.py:129`, `t1_eval.py:607-623`). This is the last structurally-missing half of TACTICAL. |
| **10** | **Strategic labels on a map-carrying benchmark** — port `stack/experiments/nurec-gsplat/strategic_gt.py` to the nuScenes lane graph / NavSim route context, feeding `load_label_reports:193`. | **3–5 d** | The programme's thesis metric. Today: 15 scoreable events over 12 clusters. A public map corpus is the only path to a publishable denominator. |
| **11** | **Fix the LATERAL yaw-rate `n`** — report `n_steps_yaw_rate` and gate `:446` on the pair count, not `n_head`. | **~30 min** | Small, but it is a metric that can silently emit `NaN` where the contract says `None`. |
| **12** | **Make the three CLIs printable on cp1252** (`PYTHONUTF8` shim or `sys.stdout.reconfigure`) — `tools/eval_four_families.py`, `tools/render_openloop_video.py`, `tools/t1_eval.py`. | **~1 h** | 22 test failures on the dev box; `--help` is unusable there. |
| **13** | **Introduce T2 into the tier vocabulary** (`t1_eval.py:145-153`, `ff_rescore.py:91-97`) before any re-perception sim number is produced. | **~2 h** | Pre-emptive: today an arm scored in a re-perception loop would have to be mislabelled T1 or rejected. |
| **14** | **Give a non-PhysicalAI corpus a parity record** so `data.list_val_episodes:76` is usable, instead of every new tool bypassing it with a raw `glob` (`eval_four_families.py:137`). | **1 d** | The leak-refusal chokepoint (`data.py:37-38`) currently protects only PhysicalAI. A NavSim split adopted through the bypass has no equivalent protection. |

---

## H. Doc-vs-code discrepancies found

| claim | reality |
|---|---|
| `taniteval/README.md` and `rollout.py:20-39` describe `pred_dense`/`gt_dense` as part of the window dump | **True of the code, false of every banked artifact.** All 27 committed `windows_*.pt` predate the dense path (MEASURED, §C.3). Any doc quoting a dense-path number computed from a committed dump is wrong. |
| `four_families.py:1126-1128` points STRATEGIC's label builder at `stack/experiments/nurec-gsplat/strategic_gt.py` | ✅ **Resolves.** (A first `ls` truncated by `head` suggested otherwise; a second probe found it — programme rule 2 in action. Recorded so nobody re-derives the wrong conclusion.) |
| `ci.py:29-33` says `overlapping_holdout_se` is "deprecated for new claims, not deleted" | ✅ Accurate — but the module still **exports** it in `__all__:44`, and 19 product call sites remain (§B.3). "Deprecated" is a label, not an enforcement. |
| `taniteval/taniteval/__init__.py:1-2` calls this "a world-class evaluation harness … v0.1" | The *instrument* is genuinely strong. The **plumbing** is where the gaps are: an unstamped tool, a broken banked-dump path, and no adapter for any public benchmark. |
