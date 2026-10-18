# P6-HN-S implementation, part 2: the per-set export of the trajectory-level family components, and the PAIRED families read (amendment B4)

Tip `363a24ffd753985efd91c857d9453d8d88990cbe`. Prereg `PREREG_P6HN_SCREEN.md` §10 + `PREREG_P6HN_SCREEN.amendmentB.md` B4 (the pairing is kept; "S4 may not read families until the export exists"). No real data, no GPU, no pod, no navtest: every fixture is synthetic.
Not touched (another stream): `refe/scorer_finetune.py`, `refe/gate_checks.py`, `refe/launch_gate.py`.

## Integration needed (escalated, not buried)

1. **Land the 6 code files and the `impl2/` package below.** The families instrument is `taniteval/adapters/navsim.py::four_families_block` (what `hn_screen_families.py` and `families6.py` call); the new module is `taniteval/taniteval/per_window.py`.
2. **The wiring into the S4 chain is NOT done here** (item 8 belongs to the chain stream). `hn_screen_families.py` now REQUIRES `--per-set-out families_per_set.npz`; `hn_screen_read.py` takes `--families-per-set families_per_set.npz`.
   Without it the read still runs, the PDM outcome is unchanged, and `families.published` is `false` with the B4 reason. A caller that does not pass the new flags therefore fails loud on families, never silently on the unpaired reading.
   The "built, tested, unreachable from its caller" rule applies: the chain script must name both flags in its argv, and a launch-gate/chain test must assert it.
3. **Decisions for the PI/Master Mind (each is one line to change, listed under "Interpretations").**

## What was built

| file (repo path) | state | what |
|---|---|---|
| `taniteval/taniteval/per_window.py` | NEW | `export(win, fam)` -> one record per component (kind mean / rms / ratio-of-sums / median), the per-set arrays the block's OWN intervals are made of; `point(c)` = the block's reduction. Nothing in `four_families.py` is edited. |
| `taniteval/adapters/navsim.py` | EDIT (base blob `707904ce12d211a44eac5ce023fb1caafc4b7bce`, CRLF kept) | `four_families_block(..., per_window=False)`; with `True` the block gains ONE key `_per_window`. Default: byte-identical. |
| `taniteval/tests/test_per_window_export.py` | NEW | byte identity, block-reduction of every component, 21-component census, yaw mask, with-lead path, 7 RED mutations. |
| `…/refe-plan/eval/hn_screen_families.py` | EDIT (base `8f2a2cb706c39b146e4855fe4cd211a6425b0f75`) | `build(per_set=True)`, `write_per_set` / `read_per_set` (npz, exact float64, numpy only), `--per-set-out` required; the unpaired legacy reading is not computed when the export is on and is stamped `paired: false` when it is off. |
| `…/refe-plan/eval/hn_screen_read.py` | EDIT (base `4bb60ed846d5036c792adf2ae1cf7a2c6d029fa3`) | `JointBoot.medians`, `comp_draws`, `paired_from_draws`, `decision_draws`, `align_export`, `families_paired`; `read(..., per_set=)`; `--families-per-set`; `families_status` publishes only when paired AND the family IC holds. |
| `…/refe-plan/eval/test_hn_screen_read.py` | EDIT (base `76e8c1ff2000cd2ad82037a4367cfb51cd61af5c`) | suite `P` (paired families), real-adapter `F` checks of the export, CLI with the export, 13 new RED mutations. |
| `…/raw/2026-10-18-p6hn-prereg/impl2/*` | NEW | this file and the logs. |

## What the export carries (per set: its log, token, and)

- **LONGITUDINAL (14 + 2 human-only):** speed MAE / bias / RMSE, target-speed within 0.5 / 1.0 / 2.0 m/s, along-track MAE / bias / final bias, accel MAE, ego-progress ratio mean / median / error / under-progress rate; `gt_progress_mean_m` and `t0_axis_gt_self_ratio` are exported but flagged `human_only` (identical for every arm; `proxy_eval.FAM_SKIP` names the same two) and never compared.
  Headway / time-gap / min-TTC: **UNAVAILABLE with reason and n** when the window has no lead (the RL-val case); with a lead their per-set arrays are exported (tested).
- **LATERAL (7 + 1 declared extra):** heading MAE, yaw-rate MAE, curvature MAE and bias, cross-track MAE / bias / final MAE; the masked four are **ratios of sums** with the standstill mask (`RETR-2026-09-26-YAWMASK`: a stationary window has numerator = denominator = 0, tested). Extra: `last_pose_heading_err_deg` from the NavSim yaw channel (§10 names "last-pose heading error"; it is not one of the block's components, so it is the 22nd, flagged `extra_to_the_four_family_block`).
- **TACTICAL:** the class ids of the EXECUTED lateral / longitudinal / 5-way decisions (the programme's own `refc_tactical` labeller, imported) for the arm's pick and the human, so the correctness flags and kappa can be recomputed on any resample; plus the goal-setting arrays of `tactical_goal`.
- **STRATEGIC:** PAUSED (PI 2026-09-19, item 25), stamped in the export and in the read; never a pass, never omitted.

## How the read pairs them (§10, B4)

`families_paired(JB, D, per_set, seeds)` uses the **same `JointBoot` object** the PDM sub-scores use (10,000, seed 20260927, one log draw per resample applied jointly to every seed and arm). Per component, per seed, the statistic is re-taken on each draw
(mean; rms = sqrt of the mean square; masked components = ratio of sums; median = median of the expanded resample; tactical accuracy / kappa from the draw's confusion table), `B_s - A_s` per draw, mean over the seeds, percentile 2.5 / 97.5.
Rows are aligned to the PDM rows by (log, token); a different set, a duplicate key, a missing arm or an export with no keys is **refused with its reason** (`NOT PAIRABLE` / `NOT COMPUTABLE`), never intersected silently.
Adverse direction = the landed `proxy_eval.fam_direction` table read on the paired interval. **Reported, not gating:** a LONGITUDINAL / LATERAL adverse component is a named watch-item (`watch_items_trajectory`, and `watch_items` = the PDM ones plus these); the outcome row is unchanged by them (tested, and a mutation that lets them gate goes RED).
Families are `published` only when the export paired AND the family IC (an arm minus itself reads exactly 0 with CI [0, 0] through the same functions) holds.

## Evidence (all MEASURED, ours; logs beside this file)

| claim | check | token |
|---|---|---|
| flag OFF is byte-identical to the tip's function | the tip blob `707904ce…` loaded from the git object vs the new default vs `per_window=False`, 3 windows (one with a lead) | `byte_identity_vs_tip_blob.log`: `ZZBYTE_IDENTICAL`; in-repo form: `test_flag_off_is_byte_identical_to_the_function_before_the_flag` (the pre-flag source is rebuilt by removing exactly the lines the flag added, and the removal is asserted) |
| flag ON adds exactly one key | `test_flag_on_adds_exactly_one_key_and_changes_no_value` | 20 passed (`test_per_window_export.log`) |
| every exported component reduces to the block's PRINTED number | dense grid and sparse grid, 21 + 2 components, to the printed rounding | same file |
| the 21 components of §10 are all there | `test_the_21_components…`; and on the real adapter in `F_per_set_export_carries_the_21_components…` | `test_hn_screen_read.log`: `ZZTEST_PASS 31  (1 unmutated suite (incl. items 5, 6, CLI, paired families) + 30 RED mutations)` |
| unpaired vs paired genuinely differ | fixture whose between-log variation (sd 3) swamps a 0.15 shift: the single-arm intervals overlap (the landed unpaired rule sees nothing) and the paired CI is sharp and < 5 % of their width | `P_pairing_is_real…`; RED mutation "B draws meet the REVERSED A draws" |
| identity | every compared component, arm minus itself = 0, CI [0, 0] (synthetic and the real export) | `P_family_IC…`, `F_identical_arms_read_exactly_zero…` |
| same draws as the PDM sub-scores | families' interval == `seedmean_diff` on an independently built `JointBoot(seed 20260927)`, to 1e-12 | `P_family_components_are_read_on_the_SAME_joint_draws…`; RED mutation "another seed" |
| regression | taniteval: 181 passed (tip) -> 201 passed (tip + change, +20 new); `test_hn_screen_read.py` 18 -> 31 | `regression_base_vs_new.log` |

Scale check (MEASURED, interactive, not a test): 3,000 sets, 75 logs, 10,000 resamples, 3 seeds x 2 arms, 9 components + tactical: `families_paired` 4.1 s.

## Interpretations I made (each one line to change; none was chosen after seeing screen data - no screen data exists)

1. **Bias components ("magnitude" in `fam_direction`)** are adverse iff the paired lower bound of `|B| - |A|` is > 0 (B farther from zero). `fam_adverse`'s unpaired rule also requires both intervals to lie on one side of 0; the paired form drops that and counts a sign flip to a larger magnitude as adverse. (`ADVERSE_TF`, `paired_from_draws`.)
2. **Progress ratio ("closer_to_1")**: lower bound of `|B - 1| - |A - 1|` > 0, likewise.
3. **Median** (`ego_progress.progress_ratio_median`) is the median of the EXPANDED resample, as the single-arm bootstrap takes it; tested against a hand-expanded resample on 14 draws.
4. **Kappa undefined** (both streams constant on one class, as in the longitudinal decision of an all-steady fixture) is `NOT COMPUTABLE`, never 1.0 (RED mutation); accuracy is still read.
5. **TACTICAL adverse** (accuracy / kappa / goal-setting) is reported under `TACTICAL.adverse_reported`; §10 names watch-items only for LONGITUDINAL and LATERAL, so these are not in `watch_items`.
6. **Seeds**: the seeds the zeros decision used (3, or 5 after rule E); both arms of every such seed must be in the export. No `base` family export is required (no family vs base comparison is registered).
7. **The family IC is not part of the classifier's IC gate** (§10: families are reported, not gating); it gates the **publication** of the families instead.
8. The one component beyond the 21 (last-pose heading) is read like the rest; it is not masked at standstill because a pose yaw exists there.

## Observations (not changed)

- Pre-existing, in `hn_screen_read.read`: `JB.ratio_interval` for M2's per-arm "own population" emits `RuntimeWarning: All-NaN slice` when an arm has no DAC-only selection-zero set; the value is NaN by construction. Seen in the tip's own fixtures.
- `hn_screen_read.families_paired` imports `proxy_eval` (for `fam_direction`), which imports `eval_checkpoint` and `refe_navtest_seam` at import time. It resolved in this venv with the tip tree; on a machine without those imports the families read fails loud while the PDM read is untouched. `hn_screen_families.adverse` already imported it lazily, so this is no new class of dependency.

## Not done / UNVERIFIED

- Nothing ran on real RL-val data (none exists yet); the export's on-disk layout was exercised through the real adapter and the real writer / reader only.
- The anti-echo controls of the LONGITUDINAL block (hold-v0 baseline) are not exported: they are controls on a lead / v0 reading, not §10 components.
- The chain wiring (see "Integration needed", point 2).
