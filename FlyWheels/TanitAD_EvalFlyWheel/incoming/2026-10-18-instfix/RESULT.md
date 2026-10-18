# RESULT -- INSTFIX: (1) the last silent unit defaults on the rollout CLIs, (2) a gated control must have a producer

**2026-10-18. Agent: INSTFIX (Sonnet), Master Mind brief of 2026-10-18. Base: mirror tip `5448a05` (every repo file read via `git show` / `git archive`, never E:).
Staged under `C:/Users/Admin/instfix/out/`; never committed, never pushed, pod untouched (no pod address or token read or written).**
Evidence class of every number: **MEASURED** from the log / JSON named beside it (`raw/testlogs/*.tsv`, `raw/mutation_each*.json`, `raw/frozen_scan_*.json`).

## 0. Headline (integration request first)
**Land the 12 repo files in s6 plus the package** (`taniteval/` + one `stack/tests/` file; no `stack/tanitad`, no pod parity needed). Land with `git hash-object --no-filters` / the coordinated lander: `taniteval/tools/refcv3_arm.py` is stored **CRLF** at the tip and I preserved that (an LF rewrite would be a whole-file diff).

1. **Fix 1 -- `actdiv_anchored.py --action-units` (was `"kappa"`, the legacy direction) now has NO default.** `--family refav1` without it exits 2 naming `kappa` and `steer` and R46 (`actdiv_anchored.py:836`, declaration `:1402`); `--family v7` and `--selftest` are not asked (v7 still refuses `steer`); `apply_action_units(cands, None)` raises, so a hand-built Namespace is not a way round the gate. **Live callers: none** -- the CLI has zero launchers outside frozen packages (4 frozen sites in 3 files, `FROZEN_CALLERS.md` s1), so no live unit had to be chosen for it.
2. **DECISION FOR YOU TO CHECK (s2): I applied the brief's s5 rule and extended the same change to `refcv3_arm.py` and `openloop_suite.py`** (default `steer`, the safe direction, but *silent*, on the CLIs that integrate the RECORDED channel for the `ha` floors). Two LIVE callers relied on that default and now state it: `taniteval/taniteval/bench/internal_t1.py:58` (`build_cmd`) and `taniteval/tools/run_hierarchy_panel.py:124` (`roll`), both `--action-units steer` = the old default, so a rollout is byte-identical. To revert the extension: restore the three one-line declarations (the exact old text is the `old` side of `code/mutation_each.py::MUTS`) and drop the two caller edits.
3. **Tests (full files, not `-k`; py 3.13.5 / torch 2.11 CPU):** 36 pre-existing files that name or import an edited file, **base 916 passed / 0 failed / 18 skipped -> after 942 / 0 / 18**; the only per-file change is `test_refav1_units_required.py` **121 -> 147** (the 121 untouched). 19 further files that import `taniteval.bench` (shard C): **identical** base and after (315 / 10 / 17 / 98 skipped; the 10 failed + 17 errors are pre-existing navsim/nuscenes environment failures, same on base). **One regression was caught by the full-form run and fixed:** `stack/tests/test_refcv3_ablations.py` 19 passed / 9 FAILED after my first edit (two CLI tests that parse `--ablate` hit the new unit gate before `--out is required`); fixed by stating a unit in those two tests -> 28 / 0 / 1, equal to base (s3).
4. **Mutation RED recorded for every restored default and every deleted gate** (s4): 14 mutations, each in a TEMP COPY, one change at a time, CLEAN control first (24 + 8 + 34 passed).
5. **Fix 2 -- `taniteval/tools/chain_controls.py` (NEW, stdlib only) + `taniteval/tests/test_chain_controls.py` (NEW, 34 tests).** A chain declares per stage what it PRODUCES and GATES ON; `require_producers(plan)` refuses at STARTUP unless every gated control has a producer in an EARLIER stage, or a named `external` (must exist) / `arriving` (checked at the stage that needs it) declaration. Run on the REAL banked chain: the ORIGINAL plan is refused with **exactly one problem, `NO-PRODUCER ... 'D3.json'`**; the fixed plan (a swap-smoke stage + a link stage) is clean; removing the producer declaration goes RED.
6. **The improved copy of the chain is `code/fp_chain_v2.py` in this package; the banked free-point package AND its byte-identical navtestE copy are untouched** and still carry the defect (s7).

## 1. Fix 1 -- what became required, and the evidence for each unit
Evidence that the recorded channel is a STEER angle (why `steer` is the unit a live caller of the recorded channel states): `stack/tanitad/data/physicalai.py:621` (`steer = np.arctan(float(wheelbase) * curv)`), contract `stack/tanitad/models/kinematic.py:20-34` ("COMMAND `steer` ... what `physicalai.signals_at` stores in v2ep `actions[:, 0]`, and what every trained refav1 / v6 / v7 checkpoint has ever consumed"), `refcv3_arm.py` module docstring (ACTION UNITS).
| surface (edited tree, line) | before | now | gate / who is asked | unit stated by live callers |
|---|---|---|---|---|
| `actdiv_anchored.py:1402` `--action-units` | `default="kappa"` (swept level fed to the model UNCHANGED) | `default=None` | `main()` `:836`: `--family refav1` -> `ap.error` (exit 2, names both values + R46). `--family v7` / `--selftest` not asked; v7 accepts absent or `kappa`, still refuses `steer` ("refav1-only"). `apply_action_units(..., None)` -> `ValueError` (`:1036`) | none (0 live launchers). The banked `raw/actdiv_anchored_refav1_step1000*.json` were produced under `kappa` (`2026-09-03-gs9-and-actdiv-units/RESULT.md:306-307`: the `kappa` pass reproduces them exactly); for a new read the repaired reading is `steer` |
| `refcv3_arm.py:3746` `--action-units` | `default="steer"` | `default=None` | after `parse_args` `:3827`: rollout (`--analyze-only` not given) -> `ap.error`; `--analyze-only` reads the unit from the dump manifest. `run_dump` already refused `None` (`units not in (kappa, steer)` -> `SystemExit`) | `bench/internal_t1.py:58`, `run_hierarchy_panel.py:124` -> `steer` |
| `openloop_suite.py:2274` `--action-units` | `default="steer"` | `default=None` | after `parse_args` `:2329`: `--ckpt` (a rollout) -> `ap.error`; `--analyze-only` / `--arm-json` not asked | docstring example `:81` and `OPENLOOP_SUITE.md:150` -> `steer` (the doc's two real commands already did) |
Docs corrected: `REFCV3_ARM.md` s1.5 (a stale line said `kappa` was "kept as the default only so a re-analysis is byte-identical" -- never true for this tool; now states the no-default rule) and `OPENLOOP_SUITE.md:150`.
Stale-prose fix inside the tool: `refcv3_arm.py` module docstring said "THE DEFAULT HERE IS `steer` -- DELIBERATELY DIFFERENT FROM `refav1_arm.py`, whose `kappa` default ..." (already false since HJ-UNITS-2); replaced.

## 2. The decision I made beyond `actdiv_anchored` -- why, and what it costs
The brief: *"Read s5 of that RESULT for any other remaining default in the same class. Fix it if it is a live integration of controls; otherwise list it, with the reason it is safe."* `refcv3_arm.py` and `openloop_suite.py` (which forwards the flag to `refcv3_arm.run_dump`) INTEGRATE the recorded channel (`integrate_select(hold ..., action_units=units)`, the `ha` and `ha0_ext` floors) -- a live integration of controls with a silent default -- so by the brief's own rule they were fixed, not listed. The default was the *correct* reading, so no banked dump is mislabelled; the change removes a choice the operator never made, on the flag whose silent `kappa` twin integrated a steer angle as a curvature.
Cost, MEASURED: 2 live callers needed the flag (fixed, s1), 2 tests needed a unit (s3), and **16 frozen rows now exit 2 if re-run** (`FROZEN_CALLERS.md`: refcv3_arm 10, openloop_suite 2, actdiv 4; of the 16, 8 are launch scripts, 7 are prose commands inside RESULT/SPEC/register documents, 1 is a command string inside a patch script and UNVERIFIED as a rollout). Pod copies of `refcv3_arm.py` / `openloop_suite.py` older than this landing keep their silent default (same value `steer`), so a pod/dev-box mismatch cannot change a number -- it can only be less strict.

## 3. Tests -- base vs after (full files, no `-k`)
py 3.13.5 / torch 2.11 CPU, `CUDA_VISIBLE_DEVICES=-1`, LF trees exported from `5448a05` (`tree_base` vs `tree`), one pytest process per FILE, <= 2 processes at a time. `run_list.py` -> `raw/testlogs/{base,after}_{A,B,C,K}.{log,tsv}`.

| set | files | base passed / failed / errors / skipped | after |
|---|---|---|---|
| shard A+B: files that name or import an edited file | 36 | **916 / 0 / 0 / 18** | **942 / 0 / 0 / 18** |
| shard C: importers of `taniteval.bench` (reach `internal_t1`), incl. `test_refcv3_ablations` | 19 | 315 / 10 / 17 / 98 | 315 / 10 / 17 / 98 (identical per file) |
| NEW `test_chain_controls.py` | 1 | -- | 31 / 0 / 0 / 0 (re-run alone after the last test was added: **34 passed**) |

Per-file differences in shard A+B (everything else identical per file): `taniteval/tests/test_refav1_units_required.py` 121/0/0/0 -> 147/0/0/0.
`test_kingate_contract.py` is an export artefact on the first base pass (it reads `Project Steering/Decisions/2026-09-05-mm-decisions.md`, outside a code export): 2 passed / 1 FAILED; I added that one file to both trees and re-ran it -> 3 passed (`base_K`), which is the base figure used above. Shard C base == after includes 27 PRE-EXISTING environment failures (navsim / nuscenes tests), identical on base: `test_bench_navsim_navtest_single_stage` 6 failed + 5 errors, `test_bench_suite_navsim_offline` 1 + 12, `test_bench_suite_submission` 2, `test_nuscenes_planning` 1.
**The regression the full-form run caught:** after my first edit `stack/tests/test_refcv3_ablations.py` read 19 passed / **9 FAILED** / 1 skipped (`after_A.log`): `test_each_ablation_name_is_accepted_by_the_cli[*]` x8 and `test_the_frame_blind_regression_has_its_own_named_flag` call `rc.main(["--ablate", name])` and expect to die on `--out is required`; they now hit the unit gate first. Fix: both tests state `--action-units steer` (the unit is irrelevant to what they pin). Re-run: 28 / 0 / 1 = base.

## 4. Mutation -- each restored default / deleted gate in a TEMP COPY, one change at a time (`code/mutation_each.py`, `raw/mutation_each_*.log`, `raw/mutation_each_part_*.json`)
Selection per mutation: the units file `-k "test_j or test_k or test_l or test_m"` (the temp-copy sub-tests skip inside, `HJ_UNITS_NESTED=1`), `test_bench_suite_internal_t1.py` FULL, `test_chain_controls.py` FULL. **CLEAN control: 24 passed + 2 skipped / 8 passed + 1 skipped / 34 passed, 0 red.**
| mutation (one change) | file | n RED | RED tests (short ids) |
|---|---|---|---|
| `CLEAN_control` | `(none: CLEAN control)` | **0** | **none -- control, must be 0** |
| `actdiv_default_kappa` | `taniteval/tools/actdiv_anchored.py` | **5** | `test_j5_actdiv_declaration_has_no_default_by_ast`<br>`test_j7_actdiv_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through`<br>`test_j_actdiv_refav1_without_units_is_refused_naming_both_legal_values`<br>`test_l3_census_mutation_restoring_a_default_in_each_REAL_file_is_caught`<br>`test_l_census_every_live_action_units_declaration_has_no_default` |
| `actdiv_gate_deleted` | `taniteval/tools/actdiv_anchored.py` | **2** | `test_j7_actdiv_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through`<br>`test_j_actdiv_refav1_without_units_is_refused_naming_both_legal_values` |
| `actdiv_apply_none_guard_deleted` | `taniteval/tools/actdiv_anchored.py` | **1** | `test_j6_actdiv_a_hand_built_namespace_cannot_slip_a_missing_unit_past_the_library` |
| `refcv3_default_steer` | `taniteval/tools/refcv3_arm.py` | **5** | `test_k3_refcv3_declaration_has_no_default_by_ast`<br>`test_k4_refcv3_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through`<br>`test_k_refcv3_rollout_without_units_is_refused_naming_both_legal_values`<br>`test_l3_census_mutation_restoring_a_default_in_each_REAL_file_is_caught`<br>`test_l_census_every_live_action_units_declaration_has_no_default` |
| `refcv3_gate_deleted` | `taniteval/tools/refcv3_arm.py` | **2** | `test_k4_refcv3_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through`<br>`test_k_refcv3_rollout_without_units_is_refused_naming_both_legal_values` |
| `openloop_default_steer` | `taniteval/tools/openloop_suite.py` | **5** | `test_k6_openloop_rollout_without_units_is_refused_the_other_sources_are_not_asked`<br>`test_k7_openloop_declaration_has_no_default_by_ast_and_the_default_is_the_mutation_target`<br>`test_k8_openloop_mutation_in_process_and_in_a_temp_copy`<br>`test_l3_census_mutation_restoring_a_default_in_each_REAL_file_is_caught`<br>`test_l_census_every_live_action_units_declaration_has_no_default` |
| `openloop_gate_deleted` | `taniteval/tools/openloop_suite.py` | **1** | `test_k6_openloop_rollout_without_units_is_refused_the_other_sources_are_not_asked` |
| `internal_t1_flag_removed` | `taniteval/taniteval/bench/internal_t1.py` | **2** | `test_cmd_is_the_tool_unmodified`<br>`test_m5_live_caller_internal_t1_rollout_states_the_unit_and_analyze_only_does_not` |
| `panel_flag_removed` | `taniteval/tools/run_hierarchy_panel.py` | **1** | `test_m6_live_caller_run_hierarchy_panel_rollout_states_the_unit_and_the_recover_branch_does_not` |
| `doc_flag_removed_openloop_stack_example` | `taniteval/tools/OPENLOOP_SUITE.md` | **2** | `test_m3_cli_census_mutation_removing_the_flag_from_a_REAL_doc_is_caught`<br>`test_m_cli_census_every_live_invocation_of_the_four_rollout_CLIs_states_the_unit` |
| `cc_no_producer_check_deleted` | `taniteval/tools/chain_controls.py` | **13** | `test_a2_no_producer_is_refused_naming_stage_and_control`<br>`test_a3_a_producer_AFTER_its_consumer_is_no_producer`<br>`test_a3b_a_stage_cannot_be_its_own_producer`<br>`test_a4_globs_cover_names_and_a_glob_gate_needs_a_matching_declaration`<br>`test_a9_require_producers_raises_with_every_problem_and_returns_the_ledger`<br>`test_b2_the_ORIGINAL_chain_is_refused_at_startup_for_exactly_one_reason_D3`<br>`test_b6_mutation_dropping_the_d3_stages_from_the_improved_chain_makes_main_refuse_at_startup`<br>`test_c2_with_the_gate_a_missing_producer_is_refused_before_ANY_stage_runs`<br>`test_c4_mutation_removing_the_producer_declaration_from_the_working_chain_turns_it_RED`<br>`test_d1_mutation_removing_the_producer_stage_turns_the_fixed_plan_RED`<br>`test_d2_mutation_removing_only_the_produces_declaration_turns_it_RED`<br>`test_d3_mutation_moving_the_producer_after_its_consumer_turns_it_RED`<br>`test_f1_cli_exit_codes_refused_2_ok_0` |
| `cc_late_producer_counts` | `taniteval/tools/chain_controls.py` | **3** | `test_a3_a_producer_AFTER_its_consumer_is_no_producer`<br>`test_a3b_a_stage_cannot_be_its_own_producer`<br>`test_d3_mutation_moving_the_producer_after_its_consumer_turns_it_RED` |
| `cc_source_check_disabled` | `taniteval/tools/chain_controls.py` | **2** | `test_a8_source_check_a_producer_script_that_never_names_the_file_is_refused`<br>`test_d4_mutation_a_producer_script_that_does_not_name_the_file_turns_it_RED` |
| `cc_ledger_check_deleted` | `taniteval/tools/chain_controls.py` | **1** | `test_e1_verify_stage_outputs_missing_empty_present_and_glob` |
The test file also carries its own mutation arms that run in the normal suite: in-process (`test_j7`, `test_k4`, `test_k8`, `test_l3`, `test_d*` of the chain tests) and self-contained temp-copy subprocesses (`test_j8`, `test_k5`, `test_k8`: clean copy `1 passed`, mutant copy `1 failed`).

## 5. s5 of HJ-UNITS-2 -- every remaining default in the same class, with its disposition
| site | default | disposition | reason (evidence) |
|---|---|---|---|
| `actdiv_anchored.py` `--action-units` | `kappa` | **FIXED** | s1 |
| `refcv3_arm.py` `--action-units` (`integrate_select` itself is already keyword-only, no default) | `steer` | **FIXED** | s1, s2 |
| `openloop_suite.py` `--action-units` | `steer` | **FIXED** | s1, s2 |
| `render_refcv3_video.py:713` `--action-units` | `steer` | **left; SAFE -- the flag is DEAD** | `action_units` is never read in the file, and the only functions in `refcv3_arm.py` that read the unit are `main` (the gate) and `run_dump`, neither called from it (`test_l4_...` proves both from the AST, with a mutation that makes the file read it). A declared-but-unread flag cannot mislabel a run. Closing it = delete the flag |
| `p10_prior.py:148,222` `steer_units=True` | `True` | **left; SAFE** | a torch port of the floor; the only callers are in the same file and pass the unit explicitly (`:228-229`, `:274` `variant == "steer"`); `True` = the recorded steer channel, the correct reading. Closing it = make required (2 call sites, `stack/tanitad/models/`: needs a pod-parity decision) |
| `v8f_mini_data.py:325` `kdx_base(units="steer")`, `ra1_pooled.py:118`, `v8f_mini_data.py:428` | `steer` | **left; SAFE** | training-data construction, not a rollout. The two call sites that rely on the default sit in the `else` (steer-mode) branch of a `kin_mode` switch whose other branch passes `KIN_UNITS["yaw"]` explicitly; `eval_context` refuses `units != KIN_UNITS[kin_mode]` (`v8f_mini_data.py:586-588`). Closing it = make `units` required (2 call sites, `stack/tanitad/data/`) |
| `v8f_mini_fit_p10.py:53,75` `kin_mode="steer"` | `steer` | **left; SAFE** | a mode selector, not a unit; `build_rows(poses, clips, stride, units, ...)` and `eval_rows(eval_kin, grid, units, ...)` take the unit as a REQUIRED positional (no default), so the mode default cannot choose the unit |
| `refc.py:1727`, `refc_sampler.py:350` `control_units="kappa"`; `refc_v3_train.py:8400` `_anchor_stamp(units="kappa")` | `kappa` | **left; SAFE (a different vocabulary)** | `anchor_meta.CONTROL_UNITS = ("kappa", "alat")` (`anchor_meta.py:53`) has **no `steer` member**: the anchor-bank controls are curvature-domain geometry, so the R46 mislabel (steer read as kappa) cannot occur through it; consumers REQUIRE the artifact's own `control_units` (`anchor_meta.py:24`, `:247`). Not audited deeper than the vocabulary |
| `refav1_add_floor.py:62`, `render_refav1_video.py:438` | `... or "kappa"` when the manifest lacks the field | **left; SAFE** | a manifest without `action_units` IS a pre-2026-09-03 dump; `refav1_arm.py:1496` writes the field into every manifest since (a dump read under `kappa` is what it was produced under) |
| `refav1_arm.py --analyze-only --action-units X` | accepted and ignored | **unchanged** | the dump manifest governs (HJ-UNITS-2 decision, `test_h4`) |

## 6. Fix 2 -- a gated control must have a producer
**The defect (RESULT s5 of the free-point read, reproduced from the code).** `fp_checks.py` reads `<work>/D3.json` (`os.path.exists(d3p)`, else `{"missing": True}` -> `D3_teeth None` -> the hard gate `D3_teeth` False -> `INVALID`). `fp_smoke.py --swap-cams` writes `<work>/smoke_swap/smoke_result.json` (`fp_smoke.py`: `sm = f"{a.work}/smoke{'_swap' ...}"`), and `raw/run_d3.sh` linked the two with a hand `cp`. No stage of `fp_chain.py` ran the smoke or the link, so the chain reached its last stage and said *missing*. `fp_read.py:161` has the same shape for the IC controls (`json.load(open(pth)) if os.path.exists(pth) else {"missing": True}`).

**Where it lives, and why.** `taniteval/tools/chain_controls.py` (+ `taniteval/tests/test_chain_controls.py`). The repo had no chain-checks module (looked: `launch_gate.py` / `supervise_run.sh` / `v6_chain.py` are TRAINING-launch machinery with their own contracts; `grep` for "no producer" finds only prose). `taniteval/tools/` is where every chain, arm and instrument of the evaluation side is launched from and already has the import convention (`_HERE` + `sys.path`), test discovery and `conftest.py`; the module is **stdlib-only and ASCII-only** (the free-point chain runs under the `driverl-eval` venv with `-I`, and the dev-box console is cp1252), is **not under a banked package** (those are records), and **not under `stack/`** (no pod landing, no parity question). The improved chain is a COPY in this package -- `code/fp_chain_v2.py` -- so the banked `.../2026-10-17-free-point-navtest/code/fp_chain.py` and `fp_checks.py` stay byte-identical (they are the record of what ran; the navtestE `fp/` copies are md5-identical to them and equally untouched).

**What it checks (before any stage; `check_plan` returns every problem, `require_producers` raises `ChainPlanError` listing all of them):**
| code | refuses when |
|---|---|
| `NO-PRODUCER` | a stage gates on a control that no stage produces and that is not declared `external` / `arriving` |
| `LATE-PRODUCER` | the only producer is the SAME stage or a LATER one (a producer after its consumer is no producer) |
| `SOURCE-SILENT` / `SCRIPT-UNREADABLE` | a producer stage names a `script` that never mentions the control's file name (the declaration without the code), or the script cannot be read (fail closed) |
| `EXTERNAL-ABSENT` / `EXTERNAL-UNNAMED` / `ARRIVING-UNNAMED` | an `external` control (made before the chain) is not in the work dir, or an external / arriving declaration names nobody |
| `DUPLICATE-STAGE` / `DOUBLE-PRODUCER` / `EMPTY-PLAN` | structure |
At run time: `verify_gates_present(stage, work)` before each stage (an `arriving` control that never arrived is refused AT the stage that needs it) and `verify_stage_outputs(stage, work)` after (every declared product exists and is non-empty, so "stage OK" cannot mean "wrote nothing"). A drift tripwire, `scan_reads` / `check_declared_reads`, finds the work-relative files a script READS (AST: f-strings and `os.path.join` rooted at `work`, one level of variable indirection) and flags a read the plan does not declare -- **a LOWER BOUND**: on the real scripts it resolves all 6 reads of `fp_checks.py` (`picks_report.json`, `points.npz`, `seam_*.npz`, `ic5_forward.json`, `IC1.json`, `D3.json`) and the `score_logs/*.status.json` read of `fp_read.py`, and reports the identity-control loop (`f"{work}/{f}.json"`) as UNRESOLVED (`*.json`) instead of dropping it. Controls outside the work dir (the harness CSVs under `DATA`) are not scanned.

**How a future chain adopts it (a dozen lines; the same pattern is `code/fp_chain_v2.py`):**
```python
import chain_controls as CC                                   # sys.path.insert(0, "<repo>/taniteval/tools")
plan = CC.ChainPlan(
    stages=(CC.Stage("picks",  script="fp_picks.py",  produces=("picks_report.json", "points.npz")),
            CC.Stage("checks", script="fp_checks.py", gates_on=("picks_report.json", "D3.json"), produces=("gates_summary.json",))),
    external={"IC1.json": "fp_capture.py check-a9, run before the chain"},     # must exist now
    arriving={"out/chunk_000.npz": "fp_ship.py pull"})                         # delivered while the chain runs
CC.require_producers(plan, work=W, script_root=HERE)            # FIRST: before --wait-for-pull sleeps, before any stage
for st in plan.stages:
    CC.raise_on(CC.verify_gates_present(st, W), "gate missing")        # ... run the stage ...
    CC.raise_on(CC.verify_stage_outputs(st, W), "stage wrote less than it declared")
```
Shell-driven chains: `python taniteval/tools/chain_controls.py check plan.json --work W` (exit 2 = refused). Derive a stage's `gates_on` from the script with `scan_reads` rather than typing it (as `fp_chain_v2.py` does for `checks`), and call `check_declared_reads` at startup.

**Proof, on the real chain (`test_chain_controls.py`, the stage list read from the BANKED `fp_chain.py` by AST, the `checks` gate list from the BANKED `fp_checks.py` by the scanner -- nothing hand-copied):**
| arm | result |
|---|---|
| ORIGINAL plan (the banked five stages, honest declarations, `IC1.json` external and present) | **refused with exactly 1 problem: `NO-PRODUCER ... stage 'checks' gates on 'D3.json'`** (`test_b2`) |
| FIXED plan (+ `d3_smoke` running `fp_smoke.py`, + `d3_link`) | clean; ledger `D3.json -> d3_link`, `IC1.json -> external`, `ic5_forward.json -> ic5_prelude` (`test_b3`) |
| miniature chain, no gate | every stage runs, ends `INVALID` -- the original behaviour (`test_c1`) |
| miniature chain, gate, no producer | **`REFUSED_AT_STARTUP`, 0 stages ran, 0 files written** (`test_c2`) |
| mutation: remove the producer STAGE / only its `produces` | RED, `NO-PRODUCER 'D3.json'` (`test_d1`, `test_d2`) |
| mutation: producer moved AFTER `checks` | RED, `LATE-PRODUCER` (`test_d3`) |
| mutation: producer script that never names the file | RED, `SOURCE-SILENT` (`test_d4`) |
| mutation: an undeclared `D4.json` read injected into `fp_checks.py` | RED, tripwire names `D4.json` (`test_d5`) |
| mutation: the external `IC1.json` removed | RED, `EXTERNAL-ABSENT` (`test_d6`) |
| `fp_chain_v2.py --check-only` through the real `main` | exit 0; dropping the D3 stages -> exit 2, `chain.log` carries `ZZCHAIN_DONE REFUSED_AT_STARTUP` and `D3.json`, no `chain_*.out` (`test_b5`, `test_b6`); a missing dev-box input -> exit 2 (`test_b7`) |
And each check is also mutated in the helper itself (s4: `cc_*`): deleting the no-producer branch, letting a later stage count, disabling the source check, deleting the empty-file check each turn named tests RED.

**Limits (UNVERIFIED / not done):** `fp_chain_v2.py` was exercised only through `--check-only` and its plan tests -- the full chain needs the pod outputs and was not run (it is a copy of logic that ran, with one added startup gate, three added stages and two ledger checks). The `SOURCE-SILENT` check proves a producer script *names* the file, not that it *writes* it on every path; `verify_stage_outputs` is what proves it at run time. The scanner is a lower bound (above).

## 7. What I did NOT change
* The banked free-point package (`.../2026-10-17-free-point-navtest/code/*.py`) and its byte-identical navtestE copies (`.../2026-10-17-safety-objective/navtestE/fp/fp_chain.py`, `fp_checks.py`; `fp_COPIES.md5`) -- records. **Any successor chain that copies them inherits the defect; point it at `fp_chain_v2.py` / `chain_controls.py`.**
* The pod: `FP-POD-CLEAN`'s remaining half (delete `/workspace/ops/freept`) needs pod access and a verified D: copy; not touched.
* Frozen callers (`FROZEN_CALLERS.md`).

## 8. Base blobs (`git rev-parse 5448a05:<path>`) and new blobs (`git hash-object --no-filters`)
| repo path | base blob (5448a05) | new blob | line endings |
|---|---|---|---|
| `taniteval/tools/actdiv_anchored.py` | `aae55fcba37ac77f704b1b123881602c317388bc` | `10908c985f2a83c7a2c85d2672b4674bc617859f` | LF |
| `taniteval/tools/refcv3_arm.py` | `a11f77b9bfab59be7e31d5798607a97799bce3b2` | `151b1d51bb52f245d44322a4fc8baf26177eae8e` | CRLF (preserved from base) |
| `taniteval/tools/openloop_suite.py` | `cee41cf222f33f8be2bf77390c6ab6a4be6258f8` | `9a9a32d9435d1c74e7a2a39d98041b58e77e41df` | LF |
| `taniteval/tools/OPENLOOP_SUITE.md` | `cf99c57669e1a6845b154dbf54aa16f2a90abfed` | `ffef4e52852bc9d18dce1938e9526c8b0bf6cfd1` | LF |
| `taniteval/tools/REFCV3_ARM.md` | `d83dd5c0008f892203a405d7a4236b9691d33282` | `07eadad91284b7d08f9c29562f33c9ee5191c42f` | LF |
| `taniteval/taniteval/bench/internal_t1.py` | `cf9de71c21a83ba8a2457151efb2700607750033` | `8d408747744a40c27905efe44cd261ad4ef35858` | LF |
| `taniteval/tools/run_hierarchy_panel.py` | `2fb088c7491b85b30c2b8d7a4311ae1f8905d977` | `b24e0dd87cedd7ba599cad5e294f77949512af97` | LF |
| `taniteval/tests/test_refav1_units_required.py` | `210513a63e89639c33e632479fa8953b8749f40f` | `ad2e77bf084ded953185840f787f26c647d80178` | LF |
| `taniteval/tests/test_bench_suite_internal_t1.py` | `3191136669dfeaa226b94f1329c5822e8da6771e` | `1ffe81cab3582e8301726df46cfe832577ed0114` | LF |
| `stack/tests/test_refcv3_ablations.py` | `b7f5fd956b513c2b3c531fa87b976fd220728461` | `fc4d69ec6368096106902e8ac09deed573fbcae7` | LF |
| `taniteval/tools/chain_controls.py` | `NEW` | `1d9013fff3cbefc1be5fe6fd3395ce70af81982b` | LF |
| `taniteval/tests/test_chain_controls.py` | `NEW` | `58c7a7ad80a4c45e79dbedfd23ac36af700eb28b` | LF |

New (package) files carry no base blob: everything under `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-instfix/` is NEW. `paths.json` lists every file (repo path, absolute src, `raw`).

## 9. Manifest (all under `C:/Users/Admin/instfix/out/`; nothing committed, nothing pushed; the working trees `C:/Users/Admin/instfix/{tree,tree_base}` are scratch and reproducible from `5448a05`)
| artifact | where |
|---|---|
| 10 edited + 2 new repo files (s8) | `out/<repo path>` (`paths.json` lists `[repo path, abs src, "raw"]`) |
| package: this RESULT, `FROZEN_CALLERS.md`, `code/` (`fp_chain_v2.py`, `mutation_each.py`, `frozen_scan.py`, `run_list.py`), `raw/` (frozen scans, mutation logs + JSON, `testlogs/`) | `out/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-instfix/` |
| test logs base / after | `raw/testlogs/{base,after}_{A,B,C,K}.{log,tsv}` (`after_A` keeps the FIRST-pass `test_refcv3_ablations` regression as evidence; the fixed pass is in `after_C`) |

## 10. Two environment traps met on this box (candidates for `TRAPS.md`; not written there)
* **`grep -c $'\r' <file>` in this Git-bash reported every line of several LF files as CRLF** (it said 2,387/2,387 for `openloop_suite.py`, whose stored blob has 0 CRLF). Count bytes in Python (`data.count(b"\r\n")`) before deciding a file's line endings; the only CRLF file I touched is `refcv3_arm.py`.
* **A Python heredoc given to the shell tool collapsed `\\` (and backslash + newline) before Python saw it**, so a needle containing `\n` or a continuation backslash matched 0 times (`AssertionError: 0`). Build such needles with `chr(92)`, or write the patch script with the file-writing tool.

## 11. Draft BACKLOG lines (I did not edit BACKLOG.md)
`| HJ-UNITS | ... **CLOSED 2026-10-18 (INSTFIX / HJ-UNITS-3, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-instfix/`)**: the last silent unit defaults are gone -- `actdiv_anchored.py --action-units` (was `kappa`; REQUIRED for `--family refav1`, not read by v7/selftest), and, per the s5 rule (live integrations of the recorded channel), `refcv3_arm.py` and `openloop_suite.py` (were `steer`; REQUIRED for a rollout, not for `--analyze-only` / `--arm-json`); the two live callers that relied on it now state `steer` (`bench/internal_t1.py`, `run_hierarchy_panel.py`). `test_refav1_units_required.py` 121->147 (declaration census over every live `--action-units` + invocation census over the four CLIs, positive/negative controls, mutation RED for each restored default and each deleted gate). 36 test files base 916/0/18 -> after 942/0/18 (+26, all in the units file); 19 bench-importer files identical. 16 frozen rows (8 launchers) now exit 2 if re-run (FROZEN_CALLERS.md). LEFT, each with its reason in RESULT s5: `render_refcv3_video` dead flag, `p10_prior` / `v8f_mini` / `ra1_pooled` steer defaults (training-data construction; closing = 2+2 call-site edits in `stack/tanitad/`, a pod-parity call), anchor-bank `control_units` (`kappa|alat`, no `steer`), the manifest-lacks-field `kappa` fallbacks. | nothing / 0 GPU |`
`| FP-POD-CLEAN | **Startup-check half CLOSED 2026-10-18 (INSTFIX)**: `taniteval/tools/chain_controls.py` (+34 tests) refuses a chain plan at startup when a gated control has no producer; on the REAL banked chain the original plan is refused with exactly `NO-PRODUCER 'D3.json'`, the fixed plan is clean, removing the producer goes RED; improved copy `…/2026-10-18-instfix/code/fp_chain_v2.py` (adds `ic1`, `d3_smoke`, `d3_link`; ledger checks). The banked free-point package and its navtestE copies are untouched and still carry the defect -- any successor chain must adopt the helper. **LEFT (needs pod access, not done by this agent):** delete the pod's `/workspace/ops/freept` once the pod is not training AND the D: copy is re-verified against `MANIFEST.sha256` (HANDOFF step 6). | nothing / 0 GPU |`
