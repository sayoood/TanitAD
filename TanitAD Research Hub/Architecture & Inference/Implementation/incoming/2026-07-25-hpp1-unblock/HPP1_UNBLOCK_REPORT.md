# HPP-1 unblock — the hierarchy proof's instruments, made admissible

**Date:** 2026-07-25 (Europe/Berlin) · **Agent:** hpp1-unblock · **Scope:** `taniteval/` only
**Compute:** dev box, **zero GPU, zero pod SSH, no checkpoint loaded, no training touched.**
**Spec:** `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/01_EXECUTION_PLAN.md`
Part A (PC1–PC4, HP-1…HP-6) · `…/incoming/2026-07-25-hpp0-confound-audit/HPP0_CONFOUND_AUDIT.md`
**Nothing staged, nothing committed, nothing pushed** — per brief.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + file:line / command output) ·
`PUBLISHED` · `INHERITED` (another doc, NOT re-verified here) · `ESTIMATED` · `HYPOTHESIS`.

**`pytest -q` = 286 passed** (baseline 177 + **109** added). `MEASURED`, full suite, 22.4 s.

---

## 0. The headline, before anything else

Two of the four deliverables produced a **result**, not just an instrument.

| | Finding | Class |
|---|---|---|
| **①** | The deprecated estimator does not merely narrow intervals — **it moves point estimates**, on this panel by up to **2.97×**. `ctx→tactical`'s published `+0.0439` is a `_jack` artifact; the true full-set paired delta is **+0.0148**. | `MEASURED` |
| **②** | Consequently **E2 (`ctx→tactical` is load-bearing) does NOT survive the migration** — it fails on the *point estimate alone*, before any interval question, on all three of its metrics. | `MEASURED` |
| **③** | **E1 / H18 (grounding dominance) DOES survive, and gets bigger**: the corrected delta is **+2.9568 m**, not +2.6979, and it would need an **8.65×** interval widening to un-separate — four times the worst widening ever measured in this program. **This is the hierarchy-supporting result, and it is now admissible.** | `MEASURED` |
| **④** | `corridor_departure_rate` now reproduces **every one of E1a's committed K=20 numbers exactly** from a reconstruction of its own definition (8/8 rates, 4 decimals). | `MEASURED` |
| **⑤** | The 98.6 % longitudinal-energy figure is **clip-specific and does not replicate** at 881 windows: across 8 committed arms the longitudinal share is **0.607–0.976**. The *compounding law* replicates on **8/8**. | `MEASURED` |

---

## 1. TASK 1 (blocking) — `hierarchy.py` ported off the deprecated estimator

### 1.1 What changed, file:line

All in `taniteval/taniteval/hierarchy.py` (1128 lines, was 698).

| Line | Change |
|---|---|
| `1-89` | Module docstring: the migration, why *this* panel mattered most, and the one-sided/two-sided `separated` subtlety spelled out. |
| `112-126` | Estimator policy **imported** from `driving.py`, never restated (`DECISION_ESTIMATORS`, `DEPRECATED_ESTIMATOR`, `ESTIMATOR_NOTE`); `LEGACY_BLOCK`; `MIN_EPISODES_FOR_CI = 2`. |
| `220-248` | `class _Boot` — cached episode-cluster draws for the full window set **and per mask** (`valid`-route subset, turn-active subset), built on `driving._Draws` / `driving._sub_draws`. |
| `250-272` | `_degenerate()` — a subset below 2 episodes emits the point estimate with **no interval** and `insufficient_episodes: True`, never a zero-width one that reads as a certain separation. |
| `274-290` | `_interval()` → delegates to `driving._interval` (rates: agreement, route-follow). |
| `292-321` | `_paired()` → delegates to `driving._paired` (every conditioning contrast). Adds `mean` as an alias of `delta` and `n` of `n_windows` so `report.py`, `runner.py` and `gate_emitters.py` read the block unchanged. |
| `323-363` | `_jack()` **kept**, now calling the named `ci.overlapping_holdout_se`, self-labelling (`estimator`, `deprecated: true`), and called **only** from the quarantine. |
| `365-378` | `_width_ratio()` — new ci95 / legacy ci95, emitted per artifact. |
| `380-392` | `_meaningful()` now reads **`separated_positive`**, not `separated`. |
| `598-847` | `_assemble(rec, n_boot, seed)` — every `_jack` call replaced (seams 1/2/3, consistency, H18). |
| `627-653` | **PC1's exit criterion, newly computable**: `route_skill_vs_majority` (paired vs the always-straight *classifier*, resampled per draw), `route_skill`, `route_skill_ci_separated`, `PC1_route_input_works`. |
| `740-751` | `harmful_if_engaged` now uses the bootstrap's real `hi < 0` instead of assuming a symmetric CI. |
| `830-847` | `assert_no_deprecated_estimator(out)` runs **before** `LEGACY_BLOCK` is attached. |
| `849-857` | Panel-level guard — the omission HPP-0 logged as §7 #6. |
| `860-984` | `_legacy()` — the quarantine: every pre-migration seam number, `ci_width_ratio_new_over_legacy`, and `verdict_flips_vs_legacy`. |
| `986-991` | `_flip()` — records a verdict change as a **correction**, in the artifact. |
| `1010-1030` | `thesis_read.PC1_route_input_works` with its interval and its spec pointer. |

**Tests:** `taniteval/tests/test_hierarchy_ci.py` — **23 tests**.

### 1.2 Before/after interval widths — `MEASURED`

The real arm's per-window arrays are not persisted (`hierarchy_flagship-30k.json` carries aggregates
only), so the widths are measured on an **881-window / 40-episode panel record with planted seam
effects**, 5 seeds, B=2000 — the same shape as the canonical val.

```
seam                                    s0     s1     s2     s3     s4    mean    min    max
nav_delta_nav_vs_follow              1.679  1.848  1.555  1.232  1.309   1.525  1.232  1.848
ctx_maneuver_acc                     2.000  1.866  1.282  1.250  1.481   1.576  1.250  2.000
ctx_goal_latent_cos                  1.419  1.595  0.980  2.190  2.250   1.687  0.980  2.250
intent_cos_real_vs_mean              1.909  2.850  1.081  1.571  1.909   1.864  1.081  2.850
intent_cos_real_vs_none              1.710  2.000  1.258  1.576  1.333   1.575  1.258  2.000
consistency_maneuver_vs_trajectory   1.591  1.190  1.512  1.659  1.556   1.502  1.190  1.659

ALL: n=30  mean 1.621  median 1.574  range 0.980-2.850  wider in 96.7 % of cases
```

**The honest interval is ~1.6× wider**, and 70 % of the ratios land inside the program's
`1.28–2.06×` band — an independent re-measurement of that finding on a panel it was never measured
on. *(`ctx_wp_ade_2s` and `h18` have an undefined ratio on this fixture: their planted effect is a
constant offset, so the legacy ci95 is exactly 0. The emitter returns `None` rather than dividing.)*

### 1.3 ⚠️ The bigger defect: the deprecated estimator **moves the mean**

`_jack`'s point estimate is the mean of **8 overlapping 20 % holdout means**, not the full-set mean.
The holdouts weight episodes unequally, so whenever the per-window value is episode-clustered — which
it always is — the reported point estimate is **biased**. `MEASURED` on the committed artifacts:

| seam (flagship-30k) | full-set paired Δ | published `_jack` Δ | bias |
|---|---:|---:|---:|
| `ctx→tactical` maneuver_acc | **+0.0148** | +0.0439 | **×2.966** |
| `ctx→tactical` goal_latent_cos | +0.0050 | +0.0084 | ×1.680 |
| `ctx→tactical` wp_ade_2s | +0.0437 | +0.0336 | ×0.769 |
| H18 grounded vs ungrounded | **+2.9568** | +2.6979 | ×0.912 |
| `nav→strategic` | +0.3292 | +0.3490 | ×1.060 |

Every full-set value is exact arithmetic on the artifact's own `real` / `mean_ctx` fields
(`hierarchy_flagship-30k.json`), which the pre-migration code computed with `_mean` (full-set nanmean)
over the identical 881 windows. Reproduced on **`hierarchy_flagship-v4.2b-dryrun.json`** (maneuver_acc
bias **×3.275**) and on **`hierarchy_flagship-30k_v1.json`** (×1.758) — three artifacts, same
direction. Mechanism reproduced on synthetic episode-clustered data (bias up to **×4.29**, including a
**sign flip**).

> This is the same root-cause class as the sibling's 1.5–5.9 % finding, an order of magnitude larger
> on a delta between two nearly-equal rates. **`overlapping_holdout_se` is not only a too-narrow
> interval — it is a biased estimator of the quantity itself.**

### 1.4 Do the load-bearing seams survive? — **one does, one does not**

| seam | corrected Δ | legacy ci95 | \|Δ\|/ci95 → widening that un-separates | practical floor | **verdict** |
|---|---:|---:|---:|---:|---|
| **H18 grounding dominance (E1)** | **+2.9568** | 0.3418 | **×8.65** | 0.05 ✅ | **STILL CI-SEPARATED — decisively** |
| `ctx→tactical` maneuver_acc (E2) | +0.0148 | 0.0310 | ×0.48 | 0.02 ❌ | **NOT separated** |
| `ctx→tactical` goal_latent_cos | +0.0050 | 0.0037 | ×1.35 | 0.01 ❌ | **NOT separated** at ≥1.28× |
| `ctx→tactical` wp_ade_2s | +0.0437 | 0.1904 | ×0.23 | 0.05 ❌ | not separated (never was) |
| `nav→strategic` (the echo) | +0.3292 | 0.0970 | ×3.39 | 0.02 ✅ | separated — **by construction**, unchanged |

**H18 — `MEASURED`, and it is evidence FOR the hierarchy.** The grounded operative rollout beats the
ungrounded tactical head by **+2.9568 m** (larger than the +2.6979 the audit quoted). Un-separating it
would need an **8.65×** widening; the worst ever measured in this program is 2.06×, and the fixture
measures ~1.6×. At 2.06× the ratio is still **4.2**. `ESTIMATED` only in that the exact honest ci95
awaits a re-run; the conclusion is robust across the whole plausible range and then some.

**`ctx→tactical` — it does not survive, and it fails for a reason stronger than the interval.**
`seam_ctx["load_bearing"]` is an OR over three metrics, each requiring **CI-separation AND** a
practical floor. With the corrected point estimates **all three fail the floor**: 0.0148 < `MIN_ACC`
0.02, 0.0050 < `MIN_COS` 0.01, 0.0437 < `MIN_ADE_M` 0.05. So the flip from LOAD-BEARING to decorative
happens on the point estimate alone. The interval only confirms it (0.0148 is already **below** the
*deprecated*, too-narrow ci95 of 0.0310).

> **Retraction owed, root-cause class C1-adjacent (*"the estimator, not the metric, produced the
> effect"*).** HPP-0 §5 row **E2** — *"H26 — ctx→tactical is load-bearing, Δ +0.0439, separated,
> ≥ MIN_ACC"* — and every `hierarchy.py` verdict string that has carried "1/3 seams load-bearing"
> since 2026-07-18. The honest statement is **0/3 seams load-bearing**, with `nav→strategic`
> separated by construction. This does **not** weaken the thesis: it removes a number that was
> never real, on a seam that PC1–PC4 say was never fairly tested.

### 1.5 What else the migration changed, deliberately

* **`separated` is now two-sided; `separated_positive` decides load-bearing.** `_jack`'s `separated`
  was `mean − ci95 > 0` (one-sided), so a strongly *negative* delta was never "separated". Emitting
  only the program-standard two-sided flag would have promoted a **harmful** seam to LOAD-BEARING.
  Both are emitted; `_meaningful` reads `separated_positive`. Pinned by
  `test_a_harmful_seam_is_not_load_bearing_end_to_end`.
* **`nav_beneficial` in `thesis_read` now requires CI-separation**, per PC1's actual wording.
  `gate_emitters.nonav_route_beats_majority` still reads the unchanged
  `vision_route_beats_majority`, so no gate behaviour moved.
* **Legacy quarantine** carries every pre-migration number plus `ci_width_ratio_new_over_legacy` and
  `verdict_flips_vs_legacy`, so a flip is visible in the artifact rather than in a reader's memory.
* Non-int episode ids: `_jack` → `gates.split_by_episode` needs `int(e)`, so the legacy block emits
  `not_evaluable: true` instead of crashing. The decision-grade path takes any hashable.

---

## 2. TASK 2 (PC3) — `corridor_departure_rate` is a first-class library metric

`taniteval/taniteval/corridor.py` (474 lines, new) — **lifted from**
`…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py::block` (L302-337), not re-derived.

| API | What |
|---|---|
| `corridor_departure(lat, t)` | **the gate co-primary** — per-window fraction of steps outside the corridor (`e1a:313`) |
| `window_departure` / `peak_xte` / `mean_xte` | `e1a:316/319/320` |
| `junction_mask` / `strata` | E1a's `\|net heading over the FIRST 2 s\| ≥ 10°` + longitudinal/other (`e1a:433-441`) |
| `corridor_block(lat[n,K], eid, …)` | one stratum, **arbitrary K**, episode-cluster bootstrap |
| `stratified(...)` | the `all_windows[K]` shape — overall / junction / longitudinal / other |
| `paired_stratum_delta(...)` | **paired** arm-vs-arm Δ, the only admissible HP-1…HP-6 form |
| `cross_track_from_paths(pred_dense, gt_dense)` | open-loop XTE from the sibling's new dense keys |
| `from_windows(win)` | entry from a persisted dump; `skipped` node without dense keys |
| `horizon_ceiling(T)` | `T−W−1` → **K ≤ 190 (19.0 s)** on PhysicalAI; K=200 is structurally impossible |

### 2.1 E1a reproduction — `MEASURED`, exact

At K=20 the committed common-start block determines its own per-window counts uniquely
(`0.0035 × 43 × 20 = 3` departed steps in `0.0233 × 43 = 1` window). Rebuilding that matrix and
running it through the library:

```
corridor_departure_rate        ours 0.0035  E1a committed 0.0035  match=True
window_departure_rate          ours 0.0233  E1a committed 0.0233  match=True
@    1m  CDR ours 0.0093 / E1a 0.0093 (True)   winDEP ours 0.0465 / E1a 0.0465 (True)
@ 1.75m  CDR ours 0.0035 / E1a 0.0035 (True)   winDEP ours 0.0233 / E1a 0.0233 (True)
@  2.5m  CDR ours 0.0000 / E1a 0.0000 (True)   winDEP ours 0.0000 / E1a 0.0000 (True)
```

**8/8 committed rates reproduced to 4 decimals**, overall and on the junction stratum
(`0.025` / `0.1667`, n=6) — and the junction reconstruction independently confirms E1a's own
statement that the departing windows are junction windows.

Second, independent route: `mean(mean_abs_xte_by_step_m) == mean_xte_m.mean` is checked on **every**
committed block of **both** committed artifacts (`e1a_horizon_heldout44_K185.json`,
`e1a_horizon_heldout44.json`; ≥8 blocks each, worst residual 5.5e-5 = emitted rounding), and on our
implementation. Ordering invariants (CDR monotone in threshold, `winDEP ≥ CDR`, `peak ≥ mean`) verified
on the artifact **and** enforced on our code. The docstring's headline numbers
(`0.0035 → 0.5877`, junction `0.8414`, peak XTE `38.9445`, ratio > 100×) are asserted against the JSON,
so a cited number whose artifact moved fails the build.

**Tests:** `taniteval/tests/test_corridor.py` — **31 tests**.

### 2.2 Refusals kept honest

* **Two surfaces, never pooled.** Closed-loop XTE accumulates control error; open-loop XTE is a
  prediction residual. Every block is stamped `surface`, and `from_windows` emits a
  `_surface_warning`.
* `cross_track_from_paths` reproduces E1a's `dlat` construction with **one documented difference** —
  the reference yaw comes from the polyline tangent because the ego-frame dense dump carries no
  per-step yaw.
* The corridor **is not a lane**: 1.75 m is `PROPOSED`. A `primary` outside the emitted grid raises
  (*"a single-threshold verdict is a knife-edge, not a result"*).
* A stratum with <2 windows or <2 episodes returns `None`, E1a's own refusal — not a NaN interval.

---

## 3. TASK 3 (M1) — every trajectory error carries its lateral + longitudinal split

`taniteval/taniteval/lateral.py` (555 lines, new).

### 3.1 The API

| Call | Returns |
|---|---|
| `decompose(pred, gt, mode)` | signed `(along[N,K], cross[N,K])`. `mode="ego"` = axis0/axis1 (the frame the MEASURED finding used); `mode="frenet"` = GT-tangent projection |
| `assert_axis_convention(gt, speed)` | **verifies** axis0 dominates and matches `v·K·dt`; raises on a transposed dump |
| `tail_stats(v, thresholds)` | **p50/p75/p90/p95/p99/max**, `frac_beyond_m` per threshold, `mean_to_p90_ratio` |
| `energy_share(along, cross)` | longitudinal share of squared error, overall + per step |
| `growth(along, cross, ref_step)` | per-axis growth curve + **`cross_grows_faster_by`** |
| `per_window(pred, gt, steps)` | `de@ / along_abs@ / cross_abs@ / …` at **every dense step** |
| **`decompose_metric(pred, gt, eid, step)`** | **the M1 drop-in**: any ADE row + its split + the cross tail, CI'd |
| `block(win)` | the full block: per-horizon rows, dense aggregate, energy share, growth, verdict |
| `from_sparse_windows(win)` | M1 on the **4-waypoint** surface, so archived dumps work with **zero GPU** |
| `paired_cross_track(a, b, gt, eid, reduce)` | **paired** Δ cross-track — the HP-2/HP-3 channel; `reduce="p90"` tests the tail |

### 3.2 Worked example — MEASURED on the committed 881-window / 40-episode val set

Every archived `results/windows_*.pt` predates the dense-path fix, so this runs on the 4-knot surface
(`from_sparse_windows`, `dt = 0.5 s`, `B=2000`).

**Cross-validation first, because a new metric that disagrees with the leaderboard is worthless:**

```
lateral.from_sparse_windows(flagship-30k, mode="frenet")
  ade_dense      = 0.4271 [0.3675, 0.4871]   ==  driving.tier0 ade_0_2s      (exact, incl. CI)
  de@2s          = 0.9075 [0.7851, 1.0306]   ==  driving.tier0 fde_2s        (exact, incl. CI)
  along_abs@2s   = 0.8412 [0.7293, 0.9591]   ==  driving.tier0 long_abs_2s_m (exact, incl. CI)
  cross_abs@2s   = 0.2369 [0.1820, 0.2960]   ==  driving.tier0 lat_abs_2s_m  (exact, incl. CI)
  max |lateral.frenet_dense − driving.frenet| = 0.0   (along AND cross)
```

`ade_dense` **is** the registry's `0.4271` for flagship v1. Pinned by
`test_sparse_block_reproduces_the_registry_ade_and_driving_split`.

**The decomposition across every committed arm** (ego mode, 881 windows / 40 episodes each):

| arm | lon share | FDE@2s | \|lon\| | \|lat\| | p50 | **p90** | **p95** | **max** | >1.75 m | lon× | lat× | **ratio** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flagship-30k (**v1, deployed**) | 0.873 | 0.908 | 0.826 | 0.274 | 0.161 | **0.712** | 0.920 | 2.29 | 0.8 % | 12.6 | 15.0 | **1.19** |
| flagship-speed | 0.801 | 1.317 | 1.177 | 0.438 | 0.252 | 1.045 | 1.828 | 2.75 | 5.5 % | 12.3 | 15.8 | 1.29 |
| flagship-nospeed (control) | 0.973 | 5.028 | 4.867 | 0.750 | 0.366 | 1.975 | 3.157 | 6.69 | 11.8 % | 4.3 | 24.0 | **5.60** |
| refc-base-30k | 0.837 | 1.003 | 0.866 | 0.305 | 0.129 | 0.759 | 1.212 | **8.05** | 2.6 % | 13.0 | 23.2 | 1.78 |
| refc-xl-30k | 0.826 | 1.006 | 0.871 | 0.307 | 0.130 | 0.723 | 1.205 | **8.02** | 2.7 % | 13.5 | 23.8 | 1.76 |
| refb-v2-30k | 0.823 | 1.243 | 1.051 | 0.411 | 0.144 | 1.075 | 1.676 | 7.21 | 4.4 % | 11.4 | 30.9 | 2.72 |
| refa-dynin-30k | 0.976 | 4.764 | 4.562 | 0.683 | 0.306 | 1.773 | 2.639 | 6.93 | 10.2 % | 3.5 | 16.6 | 4.78 |
| flagship-v4.2-step4000 | **0.607** | 1.749 | 1.273 | **0.861** | 0.500 | **1.908** | 2.773 | 8.69 | **12.0 %** | 6.1 | 7.5 | 1.23 |

*(lon×/lat× = growth of mean \|error\| from 0.5 s to 2.0 s; ratio = lat× ÷ lon×.)*

**Three readings, and the middle one is a correction:**

1. **The compounding law REPLICATES at scale — 8/8 arms.** Lateral error grows faster than
   longitudinal on every arm, by ×1.19 to ×5.60. The source analysis' claim (×4.4/×5.9 on 2 clips)
   is now supported on 881 windows across 8 arms. **Carry this one forward.**
2. ⚠️ **The "98.6 % longitudinal / 1.4 % lateral" figure does NOT replicate.** Across our own arms
   the longitudinal share is **0.607–0.976** (flagship v1: **0.873**, i.e. lateral gets **12.7 %**,
   not 1.4 %). The source's own §5b already showed 84.6 % on the second clip. **98.6 % is an n=1
   clip statistic and must stop being quoted as a program constant.** The *structural* claim —
   longitudinal dominates, so an undecomposed L2 under-weights the axis that ends the drive — stands
   with a corrected magnitude (lateral gets ~13–17 %, not ~1 %).
3. **The concealment mechanism is real and visible on the deployed arm.** flagship v1's mean
   \|XTE\|@2s is 0.274 m — reassuring — while p90 is **0.712 m** (×2.6) and max is **2.29 m**. REF-C
   base/XL carry an **8 m** cross-track error at a 2 s horizon inside an ADE of 1.00 m. And
   flagship-v4.2 has the **lowest** longitudinal share and the **worst** lateral tail (p90 1.91 m,
   12.0 % of windows beyond 1.75 m) at a middling ADE — exactly the arm an ADE-only leaderboard
   would misrank. **This is the case for §M2's p90/p95/max gate, made on our own corpus.**

The axis convention was **verified on the real val set**, not assumed: mean \|axis0\| at 2 s =
**25.396 m** vs **25.273 m** expected from mean speed 12.636 m/s (rel err **0.005**), against mean
\|axis1\| = 1.003 m — a 25× dominance. `test_axis_convention_verified_on_the_real_val_set`.

**Tests:** `taniteval/tests/test_lateral.py` — **38 tests**, including: the split is orthonormal
(`along²+cross² == ‖r‖²`); ego and frenet agree on a straight path and **must diverge in a turn**; a
pure speed error yields **zero** cross-track and a pure lateral offset yields **zero** along-track;
planted energy shares (0.986 / 0.846 / 0.5) and growth ratios recovered to 3 decimals; a transposed
dump is **rejected**; the paired cross-track test is blind to a pure speed difference and its `p90`
form sees a tail its `mean` form dilutes 5×.

---

## 4. TASK 4 (HP-3) — the counterfactual route-swap probe

`taniteval/taniteval/strategic_probes.py` (500 lines, new). **Zero training, one encode per window,
three strategic passes** — built on the hook HPP-0 §3.2 identified.

**What it measures.** For identical observation windows: encode once → branch
`nav ∈ {follow, left, right}` → strategic ctx → tactical head → waypoints. Then

* **(a) divergence**: `wp_l2_mean_m`, `wp_l2_final_m`, and — the headline, per §M6 —
  **`cross_track_2s_m`** with its p90 and full tail, plus `ctx_cosine`, `intent_cosine`,
  `maneuver_changed_rate`, for all three branch pairs;
* **(b) agreement with the command**: `direction.score` — the per-window fraction of the two commands
  whose lateral response points the commanded way (left ⇒ +y, right ⇒ −y), against **chance 0.5**,
  requiring the **CI** to clear it;
* **(c) the echo control**: `route_head_echo` — does the route *logit* follow the command? Under the
  v1 labeler this is ~1.0 **by construction**, and **echo ≈ 1.0 next to divergence ≈ 0 is precisely
  the defect HPP-1 fixes**.

Verdict: `HP3_route_conditional = divergence CI-separated ∧ ≥ MIN_DIVERGENCE_M ∧ direction above chance`.

**Discrimination demonstrated on synthetic fixtures — `MEASURED`, no GPU:**

```
FLAT (output ignores nav_cmd)
  HP-3 FAILS — trajectories are effectively IDENTICAL under left vs right. This is a PC1
  regression (still a command echo), NOT evidence that hierarchy does not help ·
  left-vs-right cross-track@2s 0.0 [0.0, 0.0] m (L2 0.0 m, floor 0.05) ·
  direction score 0.0 vs chance 0.5 · route-logit echo 1.0 · n=48/12 eps

ROUTE-CONDITIONAL (nav_cmd steers laterally)
  HP-3 PASSES — the strategic command causally changes the trajectory, in the commanded
  direction · left-vs-right cross-track@2s 3.0 [3.0, 3.0] m (L2 1.875 m, floor 0.05) ·
  direction score 1.0 vs chance 0.5 · route-logit echo 1.0 · n=48/12 eps

INVERTED (moves, but the wrong way)
  HP-3 PARTIAL — the command changes the trajectory but NOT reliably in the commanded
  direction · direction score 0.0 vs chance 0.5
```

The **flat model scores exactly 0 while echoing the command perfectly** — the probe separates
"the command reached the logits" from "the command reached the trajectory", which is the entire
question. **Tests:** `taniteval/tests/test_strategic_probes.py` — **17 tests**, including
`test_one_encode_three_strategic_passes` (exactly 3 strategic calls per encode — the cost claim),
and `test_an_arm_without_a_strategic_level_is_SKIPPED_not_passed`.

### 4.1 Documented invocation — for when a pod frees

**Not run against any checkpoint**: pod1/2/3 were training and the eval pod mid-transfer, per brief.
`strategic_probes.INVOCATION` (asserted by a test, so it cannot rot):

```
cd /root/taniteval && PYTHONPATH=/root/TanitAD/stack \
  python -m taniteval.strategic_probes \
    --model flagship-30k --episodes 40 --stride 8 \
    --out results/hp3_flagship-30k.json
    [--grounded]   # OUT-OF-REGIME intent-threaded rollout diagnostic, ~3x cost
```

⚠️ **Pre-registered expectation, both outcomes committed in advance.** Every arm trained before
HPP-1 is expected to score `HP3_route_conditional: false`, for three independently MEASURED reasons
carried in the module docstring: (1) `route_target = _NAV_TO_ROUTE[nav_cmd]` ⇒ `route_skill = 0.0` by
construction; (2) `NAV_FOLLOW` is fed on ~73 % of windows; (3) the scored rollout takes no
`intent`/`ctx`/`nav`, so on the deployed surface a route command **structurally cannot** reach the
trajectory — reported as `route_can_reach_scored_trajectory: false` with a `_pc2_note`, so a 0 is
never readable as a model verdict. **A pre-fix null is the baseline this probe exists to establish;
the number that matters is whether it moves after HPP-1's label fix.**

---

## 5. Found but NOT fixed — escalations

1. 🟥 **E2 (`ctx→tactical` load-bearing) needs retracting**, and so does every downstream
   "1/3 seams load-bearing" headline. §1.4. **Root-cause class: the estimator produced the effect.**
   Owner: whoever owns `RETRACTION_LOG.md` + `MODEL_REGISTRY.md`. *This is the same shape as the
   07-25 C4 header-propagation retraction — the correction must reach the headline, not just a body
   paragraph.*
2. 🟥 **`overlapping_holdout_se` biases point estimates, not only intervals** (up to ×2.97 here,
   ×4.29 with a sign flip on clustered synthetic data). Every module still on `_jack`/`_agg` has
   *both* problems. I migrated `hierarchy.py`; `closedloop.py`/`driving.py` were migrated by a
   sibling. **Someone should sweep for remaining callers** — I did not, to stay out of `stack/`.
3. 🟠 **The "98.6 % longitudinal" figure is n=1 and does not replicate** (0.607–0.976 across 8 arms).
   §3.2 reading 2. It is quoted in `LATERAL_VS_LONGITUDINAL_ANALYSIS.md` §1.1/§M3 and will propagate.
   The compounding law is the durable claim; the energy share is not.
4. 🟠 **`corridor.from_windows` cannot run on any archived arm**: no committed
   `results/windows_*.pt` has `pred_dense`. The dense keys need one `rollout.collect` re-run per arm
   (GPU, minutes). **Until then PC3 is unblocked in code but unmeasured on any real arm.**
5. 🟠 **`driving.tier0` does not yet call `lateral.block` / `corridor.from_windows`.** I deliberately
   did **not** edit `driving.py` — a sibling was mid-rewrite. Wiring is two lines
   (`out["lateral"] = lateral.block(win)`, `out["corridor"] = corridor.from_windows(win)`, both
   already guarded and both already passing `assert_no_deprecated_estimator`). **Needs an owner, or
   it becomes the next 10-day orphan.**
6. 🟡 `hierarchy.py`'s `_jack` legacy block needs int-castable episode ids
   (`gates.split_by_episode` does `int(e)`); it self-reports `not_evaluable` otherwise. Pre-existing
   constraint, now visible instead of a crash.
7. 🟡 The E1a *closed-loop* corridor numbers cannot be reproduced bit-for-bit from the library because
   the run's per-window `lat` arrays were not persisted — only the aggregates. **Future one-offs
   should persist the per-window array**; it is a few hundred KB and it is the difference between a
   reproducible metric and a quotable one.

---

## 6. Deliverable manifest

| Artifact | Where it lives | Lines |
|---|---|---|
| `hierarchy.py` — estimator migration + PC1 read + legacy quarantine | `taniteval/taniteval/hierarchy.py` (**modified**) | 1128 (was 698) |
| `corridor.py` — corridor departure, arbitrary K, junction stratum | `taniteval/taniteval/corridor.py` (**new**) | 474 |
| `lateral.py` — M1 lat/lon decomposition + tails + energy + growth | `taniteval/taniteval/lateral.py` (**new**) | 555 |
| `strategic_probes.py` — HP-3 counterfactual route swap | `taniteval/taniteval/strategic_probes.py` (**new**) | 500 |
| `test_hierarchy_ci.py` — 23 tests | `taniteval/tests/test_hierarchy_ci.py` (**new**) | 331 |
| `test_corridor.py` — 31 tests, E1a reproduction | `taniteval/tests/test_corridor.py` (**new**) | 429 |
| `test_lateral.py` — 38 tests, registry pins | `taniteval/tests/test_lateral.py` (**new**) | 462 |
| `test_strategic_probes.py` — 17 tests | `taniteval/tests/test_strategic_probes.py` (**new**) | 336 |
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-hpp1-unblock/HPP1_UNBLOCK_REPORT.md` | — |

**Everything is in the repo working tree. Nothing staged, nothing committed, nothing pushed.**
No file outside `taniteval/` and this bundle was modified. No pod touched, no GPU used, no checkpoint
loaded, no training or eval run.

**Verification command:**
```
cd taniteval && python -m pytest -q      # -> 286 passed
```

**Primary sources read:** `HPP0_CONFOUND_AUDIT.md` · `01_EXECUTION_PLAN.md` Part A ·
`LATERAL_VS_LONGITUDINAL_ANALYSIS.md` · `e1a_horizon.py` + `e1a_horizon_heldout44{,_K185}.json` ·
`taniteval/taniteval/{ci,driving,closedloop,rollout,hierarchy,report,runner}.py` ·
`stack/scripts/{refb_labels,gate_emitters}.py` · `stack/tanitad/eval/gates.py` ·
`hierarchy_flagship-{30k,v4.2b-dryrun}.json` · `hierarchy_flagship-30k_v1.json` ·
`taniteval/results/{windows_*.pt, driving_flagship-30k.json}`.
