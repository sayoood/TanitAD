# Wave-1 A — taniteval metrics integrity (T1-2, T3-11)

**Date:** 2026-07-25 · **Scope:** `taniteval/` only · **Compute:** dev-box CPU only
(no pod, no GPU, no ssh — pod1/pod2/pod3/eval untouched) · **Staging:** nothing
`git add`ed, nothing committed, nothing pushed.

**Suite: `pytest -q` → 177 passed in 18.69 s** (baseline 153 + 24 new; MEASURED
before: `153 passed in 19.94s`).

---

## T1-2 — closed-loop headline CIs routed through the correct estimator

### What was wrong (MEASURED, code audit)

`taniteval/taniteval/closedloop.py` computed its **headline** ADE/FDE
(`_agg`, old `:382`), its **compounding-error deltas** and its **divergence
rate** (`_jack`, old `:393`) as `mean ± 1.96·std/√8` over 8 *overlapping*
random 20 % episode holdouts — i.e. `overlapping_holdout_se`, the estimator the
program deprecated as **1.28–2.06× too narrow**. `driving.py` had been migrated;
`closedloop.py` had not, even though `ci.episode_cluster_bootstrap` was already
imported and used ~40 lines below for the imagination A/B only. Closed-loop is
the axis on which the v4 imagination thesis is judged, so a too-narrow interval
here manufactures false CI-separations on exactly the comparisons that spend
GPU-days.

### What changed — `taniteval/taniteval/closedloop.py`

| line | change |
|--|--|
| `:86–91` | import `taniteval.ci` and `taniteval.driving` at module level |
| `:109–117` | estimator policy **imported** from `driving.py`, not restated (`DECISION_ESTIMATORS`, `DEPRECATED_ESTIMATOR`, `ESTIMATOR_NOTE`, `N_BOOT`, `LEGACY_BLOCK`) |
| `:361–388` | new Aggregation banner: what was replaced, what it cost, what replaced it |
| `:427` | **new** `_suite_components` — per-window decomposition of `_suite` (mirrors `bench._suite_components`) |
| `:449 / :460` | `_interval` / `_paired` **delegate to `driving._interval` / `driving._paired`** — one implementation of the estimator, so the two panels cannot drift |
| `:484 / :502` | `_agg` / `_jack` kept but demoted, ci95 now via the *named* `ci.overlapping_holdout_se`, every node stamped `estimator` + `deprecated: True` |
| `:518` | **new** `_width_ratio` — the artifact re-measures its own narrowing |
| `:580` | `_Draws(eids, n_boot, seed)` — ONE set of episode resamples shared by every interval in the block |
| `:598` | **headline** → `episode_cluster_bootstrap` |
| `:625 / :628` | **compounding** (grounded + bicycle) → `paired_episode_cluster_bootstrap` |
| `:639` | **divergence rate** → `episode_cluster_bootstrap` |
| `:679–681, :712–720` | imagination A/B: `n_boot`/`seed` threaded, components shared (arithmetic unchanged at defaults) |
| `:750–800` | `legacy_overlapping_holdout_se` quarantine block + embedded width ratios |
| `:807–835` | `protocol.ci` corrected; `estimator` provenance stamp + `primary_ci` + gate-readable `cluster_bootstrap["model"]` (mirrors `driving.tier0`) |
| `:864` | `assert_no_deprecated_estimator` — enforced on every returned block, legacy key exempt |
| `:936` | operator print now names the estimator and the measured narrowing |

**Paired, never quadrature.** Both compounding blocks compare two paths scored
on the *same* windows, so they use the paired form. Nothing is combined in
quadrature.

**Nothing is orphaned.** The old numbers are still emitted under
`legacy_overlapping_holdout_se`, self-labelling, excluded from `summary`, and
the single documented exemption from the guard — so every closed-loop interval
published before today stays reproducible.

### Before / after interval widths — MEASURED

Fixture: two **committed, window-aligned** arm dumps over the same **881
windows / 40 val episodes** — `windows_flagship-30k.pt` (ADE 0.4271) and
`windows_flagship-nospeed.pt` (ADE 3.0175). Real errors, real within-episode
correlation, real clustering; only the path *labels* are assigned. B = 2000.

**Headline `closed_bike`, ci95:**

| metric | legacy | new | ratio |
|--|--|--|--|
| de@0.5s / ade@0.5s | 0.1308 | **0.2127** | **1.63×** |
| de@1s | 0.2674 | **0.4050** | **1.51×** |
| ade@1s | 0.1981 | **0.3098** | **1.56×** |
| de@1.5s | 0.4283 | **0.5994** | **1.40×** |
| ade@1.5s | 0.2717 | **0.4052** | **1.49×** |
| **ade@2s / ade_0_2s** | **0.3558** | **0.4997** | **1.40×** |
| de@2s / fde@2s | 0.6246 | **0.8101** | **1.30×** |

**`ade_0_2s` across all five paths:**

| path | legacy ci95 | new ci95 | ratio |
|--|--|--|--|
| closed_bike | 0.3558 | 0.4997 | 1.40× |
| closed_grnd | 0.3558 | 0.4997 | 1.40× |
| open_grnd | 0.0312 | 0.0598 | **1.92×** |
| open_bike | 0.0312 | 0.0598 | 1.92× |
| cv | 0.1035 | 0.2241 | **2.17×** |

**Compounding (paired) + divergence, ci95:**

| block | legacy | new | ratio |
|--|--|--|--|
| compounding Δ@0.5s | 0.1311 | 0.2114 | 1.61× |
| compounding Δ@1s | 0.2727 | 0.4155 | 1.52× |
| compounding Δ@1.5s | 0.4387 | 0.6160 | 1.40× |
| **compounding Δ@2s** | **0.6270** | **0.8217** | **1.31×** |
| **divergence rate >5 m@2s** | **0.0587** | **0.0771** | **1.31×** |

**Every interval got wider. Range 1.30×–2.17×** — an independent reproduction of
the program's 1.28–2.06× finding on a different axis and a different harness.

### Test evidence — `taniteval/tests/test_closedloop_ci.py` (14 tests, new)

* every emitted interval names a decision-grade estimator, **and the guard
  actually bites**: the unfiltered walk over the legacy block raises;
* the headline reproduces **`CI_RECOMPUTE_2026-07-20.json` exactly** —
  `open_grnd.ade_0_2s = (0.4271, 0.3675, 0.4871)`, the same triple
  `test_driving_gate_block.py` pins. The closed loop and the driving panel now
  demonstrably share one estimator, not two lookalikes;
* the migrated headline is **byte-identical** to the already-correct imagination
  block for the same quantity (`heldout.closed_bike.ade_0_2s ==
  imagination.B_closed_bike_ade@2s`, and likewise for divergence) — they can
  never drift apart again;
* the bootstrap supplies the interval and **does not move the mean** (max
  |bootstrap − full-set `_suite`| = 1e-4, i.e. 4th-decimal rounding);
* intervals are wider than the legacy block for **every** metric and block;
* the emitted `ci_width_ratio_new_over_legacy` cannot lie (checked against the
  blocks it summarises);
* every key `closedloop_report.py` and `run_and_save` print still resolves.

Verified end-to-end besides the unit tests: `analyze()` → `json.dumps`
(44 383 B) → `closedloop_report.main()` renders the full 71-line report.
`analyze()` costs **1.64 s** of CPU at the full decision-grade B = 2000 — there
is no cheap-variant trade-off to make.

### ⚠️ Additional finding — the deprecated block also moved the POINT ESTIMATE

Not just the `±`. Averaging 8 random-subset means is not the full-set mean:

| path | true full-set mean | legacy `mean` | error |
|--|--|--|--|
| closed_bike | 3.0175 | 2.9176 | **3.31 %** |
| open_grnd | 0.4271 | 0.4522 | **5.88 %** |
| cv | 0.8377 | 0.8248 | 1.54 % |

`ci.py` documents the SE defect; this consequence was not stated anywhere. **Any
closed-loop *mean* quoted from an artifact written before today is wrong by
1.5–5.9 %, independently of its interval.** Pinned by
`test_deprecated_block_also_moved_the_point_estimate`.

---

## T3-11 — dense-path persistence (the residual open since 2026-07-09)

`rollout.collect` computed the full `[b,20,2]` path and kept 4 of 20 steps,
blocking jerk, comfort bounds, the curvature *profile*, decel-onset lead time
(LAL-v2) and plan stability on one line.

### What changed

* **`taniteval/taniteval/rollout.py`** — `collect` now also returns
  `pred_dense` / `gt_dense` `[N,20,2]` plus `dense_steps` / `dt_s`
  (`:150–177`). **No extra compute**: `wp_full` was already being produced and
  discarded; `gt_dense` is one extra `gt_ego_waypoints` call with 20 steps.
  New `dense_speed_profile()` (`:186`) defines the origin-prepend convention
  once — it is exactly the `ego_v` that the **already-merged**
  `stack/tanitad/eval/metrics.py:202–251` (`decel_onset_index` /
  `compute_lal_v2`) consumes. LAL-v2 was **not** re-merged or modified.
* **Backward compatible.** `pred`/`gt` keep their exact meaning:
  `pred == pred_dense[:, [4,9,14,19]]` and `gt == gt_dense[:, [4,9,14,19]]`,
  both asserted. Pre-2026-07-25 dumps and `refb_eval`/`refc_eval` dumps have no
  dense keys; documented as `win.get("pred_dense")`.
* **`taniteval/taniteval/bench.py`** (`:268–316`) — `collect_full` carries the
  dense keys too, so its documented *"strict superset of rollout.collect"*
  contract stays true.
* **`taniteval/taniteval/driving.py`** — two now-false claims fixed: the module
  docstring's "blocked on rollout.py:94" and `tier0`'s emitted `surface` string.
  New `dense_surface_available` flag so a dump advertises its own surface.
* **No CV-dense stored, deliberately.** Constant velocity is linear in the step
  index (`cv_dense[:,k-1] == cv[:,0]·k/5`), as is hold-v0 from `speed`. Only
  `gt_dense` is irrecoverable, so only it is paid for.

### Storage delta — MEASURED

Committed 881-window / 40-episode `windows_flagship-30k.pt`, re-saved both ways
through `save_windows`:

| | bytes |
|--|--|
| sparse only | 95 950 |
| with dense | **378 359** |
| **delta** | **+282 409 B (+275.8 KiB, 3.94×)** |

**0.378 MB/arm — inside the ~1 MB/arm budget the axis was costed at.** Theory
`2·881·20·2·4 = 281 920 B`; the extra 489 B is pickle overhead.

### Test evidence — `taniteval/tests/test_rollout_dense.py` (10 tests, new)

CPU-only, `rollout_decode` stubbed so the test exercises *persistence and
geometry*, not the predictor. Pins: dense keys with all 20 steps; sparse is a
strict sub-view of dense; `gt_dense` uses the **same ego-frame convention** as
the trusted sparse `gt` on a real curved trajectory; the origin-prepend speed
convention (constant-velocity path ⇒ constant speed at *every* step, including
the first); a jerk-style derivative is now computable and finite over 18 samples
instead of 2; the storage delta; and that pre-dense dumps still load and still
score (`driving.tier0` reports `dense_surface_available: False`).

---

## Incidental fix

`Path.write_text` without an explicit encoding crashed on the dev box (cp1252
vs the `−`/`±` in the artifacts), making the closed-loop JSON and
`CLOSEDLOOP_REPORT.md` unwritable off-pod. Added `encoding="utf-8"` at
`closedloop.py:921` and `closedloop_report.py:62, :180`. Pre-existing, unrelated
to the migration, but it blocked end-to-end verification.

---

## Found but NOT fixed — with recommendations

1. **No closed-loop artifact has been re-run.** `taniteval/results/` contains no
   `closedloop_*.json` on this box, so every closed-loop interval currently in
   circulation is a legacy one — too narrow by ~1.3–2.2× *and* with a mean off
   by 1.5–5.9 %. **Recommend:** re-run `python -m taniteval.runner closedloop
   --model flagship-30k` on the eval pod at the next free slot, then re-check
   any doc quoting a closed-loop number. No GPU work was done here by
   constraint.
2. **29 committed `windows_*.pt` dumps predate the dense fix.** The dense axis
   is unblocked only for arms re-collected from now on. **Recommend:** backfill
   opportunistically during the next eval sweep — it is a re-`run` per arm, not
   new science.
3. **`refb_eval.collect` / `refc_eval.collect` emit no dense keys.** REF-B and
   REF-C own their trajectory surface, so the behavioural axis is still
   unavailable for the direct-trajectory arms. **Recommend:** add the dense
   surface there before any cross-arm comfort comparison, or the comparison
   silently covers only the world-model arms.
4. **Tier-1 behavioural metrics are now UNBLOCKED but UNIMPLEMENTED** — jerk,
   curvature profile, decel-onset lead, plan stability. Inputs and the speed
   convention are in place. **Recommend:** a follow-up that adds them to
   `driving.py` as a tier-1 block, paired against CV / hold-v0 like tier 0.
5. **`bench.run`'s operator print still quotes the deprecated block.**
   `runner.py:153–157` prints `res["heldout"]["model"]`, which in `bench.py` is
   deliberately the legacy block (`bench` emits `cluster_bootstrap` as primary
   alongside it). So the open-loop line an operator reads carries the
   too-narrow ±. **Not fixed** — outside this task and it touches the main
   open-loop print that gates and other agents read. **Recommend:** switch that
   one print to `cluster_bootstrap`, as a change reviewed on its own.
6. **Paired-vs-quadrature was NOT demonstrated narrower on this fixture**
   (paired ci95 0.8217 vs quadrature 0.8193, ≈1.00×) — because the two arms
   paired here are weakly correlated (ADE 0.43 vs 3.02). In the real closed
   loop, `closed_grnd` and `open_grnd` are the *same* model on the *same*
   windows and are strongly correlated, where the paired form is materially
   narrower. The reason to pair is **validity** (the estimates are not
   independent), not width. Stated so nobody quotes "paired ≈ quadrature" as a
   general result. No test asserts it.

---

## Deliverable manifest

All in the working tree, **unstaged** (orchestrator to stage and commit):

| artifact | path |
|--|--|
| closed-loop CI migration | `taniteval/taniteval/closedloop.py` |
| dense-path persistence + speed convention | `taniteval/taniteval/rollout.py` |
| dense keys in the diagnostic collector | `taniteval/taniteval/bench.py` |
| stale-claim fixes + `dense_surface_available` | `taniteval/taniteval/driving.py` |
| utf-8 encoding fix | `taniteval/closedloop_report.py` |
| CI migration tests (14, new) | `taniteval/tests/test_closedloop_ci.py` |
| dense-path tests (10, new) | `taniteval/tests/test_rollout_dense.py` |
| this report | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-wave1-metrics-integrity/WAVE1_A_REPORT.md` |

Nothing is stranded on a pod or in a worktree. `pytest -q` → **177 passed**.

**Escalation (integration, not a note-in-a-README):** items 1 and 5 above need
an owner. Item 1 changes numbers already circulating in closed-loop prose; item
5 leaves the operator-facing open-loop print on the deprecated estimator.
