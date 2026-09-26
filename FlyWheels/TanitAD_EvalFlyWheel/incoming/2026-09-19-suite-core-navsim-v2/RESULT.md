# RESULT — W1: the one-command suite core, and NAVSIM v2 end to end

**Stream** EvalFlyWheel W1 · **2026-09-19/20** · pre-registered in `SPEC.md` (blob `fa510ef4…`,
`raw/SPEC_PREREG_HASH.txt`, written before any product code) · contract
`../2026-09-19-eval-suite-build/BUILD_PLAN.md` §1.

Every number below is **MEASURED** by this stream unless marked. The suite's own numbers are
REPRODUCTIONS of E1/E2's banked official runs — ⛔ **no new NavSim result is claimed here**, and the
warmup protocol carries **no interval** by construction (7 log groups < the RG-14 floor of 8).

## 1. What exists now

```
python -m taniteval.bench <benchmark> --ckpt <path|none> --split <split> [--arms …] [--device auto|cpu|cuda]
    navsim_v2 ✅ (this package)   internal_t1 ✅ (wraps refcv3_arm.py)
    navsim_v1 → plugins/navsim_v1.py (W3, landed)   nuscenes_ol → plugins/nuscenes_ol.py (W6, landed)
python -m taniteval.bench submit <run_dir> --target … [--pi-approval <id>]   # REFUSED without a recorded PI decision
python -m taniteval.bench validate <run_dir> | gpu-gap | reaggregate --preagg … --split … --out …
python -m taniteval.bench --model <key> …                                    # the PRE-EXISTING diagnostic panel, untouched
```

The run directory is the §1 contract: `bench_run.json`, `scores/<arm>.csv` (the devkit's own output,
unmodified), `artifacts/<arm>.json`, `criteria/<arm>.txt`, `summary.json`, `report/`, `raw/`. Both
JSONs are validated against `taniteval/taniteval/bench/schema/{bench_run,summary}.schema.json`
**at write time** — a run that violates the contract is refused, not published.

## 2. Findings

1. ⛔ **The brief's product path collided with a tracked module, and the collision was silent.**
   `taniteval/taniteval/bench.py` (the diagnostic panel) has existed for months and is imported by
   `runner.py:39`, `refc_rerank.py:78`, `generalization.py:901`, six test files, five stack scripts
   and `rerun_all.sh:7`; CPython prefers a package directory of the same name, so
   `taniteval/taniteval/bench/` **shadows** it. The package now delegates every legacy attribute to
   `bench/_legacy.py`, which executes that file unchanged, and `__main__` routes `--model/--all` to
   its CLI. **MEASURED: the five pre-existing importer test files read 81 passed BEFORE the package
   and 81 passed AFTER** (`raw/legacy_importers_{BEFORE,AFTER}.txt`). ⚠️ W5 had the same collision
   (`report.py`) — the orchestrator renamed their package to `taniteval/taniteval/benchreport/`.
2. ⭐ **A completed scoring run whose AGGREGATION dies is no longer a lost run.** MEASURED by E1 on
   2026-09-19: navhard CV scored **all 5,912 scenarios in 68 min**, then `create_scene_aggregators`
   raised and `calculate_individual_mapping_scores` died on NaN — exit 1, **no CSV**, and because
   that run used a PROCESS POOL the wrapper's in-process `pdm_score` hook recorded **`pdm_calls=0`**,
   so nothing survived at all. The suite now dumps the per-token frame at the **`worker_map`** seam
   (it returns in the PARENT for every worker type) as CSV **and** pickle before any aggregation,
   classifies that case as **`AGGREGATION_FAILED` (scoring_complete = true, retryable = false)**
   rather than a failed run, and can re-run ONLY the aggregation with the devkit's own functions
   (`reaggregate`). Same family as the `t1_eval` analysis-time import that destroyed a finished
   rollout — *check for the banked dump before re-running anything.*
3. ⚠️ **`four_families.lateral` emits `None` (not a refusal) for a stationary plan, and
   `criteria_check.py` reads `None` as ABSENT.** MEASURED on the STOP floor: `heading_mae_deg`,
   `curvature_mae_1pm`, `yaw_rate_mae_degps` came back `None` with `n_steps_heading = 0` and
   `excluded_below_min_ds = 128/128` — **3 silent-omission VIOLATIONS** in the first STOP artifact.
   The suite converts them into refusals carrying the harness's own counts, and a test pins
   `n_violations == 0` for every arm; the honest fix belongs in `four_families.lateral` (W2).
4. ⚠️ **`cross_mae_m` cannot distinguish CV from STOP.** `four_families.py:776` computes
   `ct_err = P["cross"] - G["cross"]` in the EGO frame, so ANY plan with `y ≡ 0` scores the same:
   MEASURED, CV and STOP both read **1.0658 m** cross-track MAE (and −0.1605 bias, 2.8312 final) on
   the same 16 stage-1 scenes, while their longitudinal rows differ enormously (speed MAE 0.8928 vs
   **6.9285 m/s**, along MAE 1.2669 vs **15.5075 m**). Not a suite defect — but a LATERAL row quoted
   alone would read as a tie between a moving arm and a parked one.
5. **The GPU-gap launcher refuses the card exactly as the PI specified, and it is honest about what
   it cannot see.** Windows/WDDM `nvidia-smi --query-compute-apps` reports `used_memory = [N/A]` for
   every process (MEASURED), so per-process GPU memory is unobservable; the launcher therefore uses
   the DEVICE total minus the memory we ourselves added, and treats an unreadable reading as NO GAP.
   Probed live at 09:0x with two CPU trainers alive: `{"gap": false, "reasons": ["training process
   alive: pids [34760, 51068]", …]}`. ⛔ **UNVERIFIED ON HARDWARE**: the GPU belonged to training
   throughout, so the cuda path is mock-tested only (21 literal tests incl. a RED mutation).
6. **The submission gate can pass, and refuses everything real.** MEASURED: no
   `SUBMISSION-APPROVED:` line exists in any PI decision record as committed in HEAD, so
   `submit` exits 3 with the PI's own words; with an injected record the same gate exits 4
   ("approval verified — the upload step is NOT BUILT"). ⛔ Nothing was sent anywhere.
7. ⭐ **The promotion pin earned itself during the build: the ORIGIN MOVED.** E1 edited
   `navsim_win.py` while W1 was building on it (they added their own pre-aggregation dump at the
   `create_scene_aggregators` seam), and `test_bench_suite_promotion.py` went RED with *"the ORIGIN
   moved"* — blob `8590ef84…` → `96d13540…`. The file was re-promoted from E1's current version with
   the W1 addition re-applied, and both dumps now run. ⚠️ The two seams are not equivalent: E1's
   fires inside the aggregation try-block and writes a CSV **without `ego_simulated_states`**, so it
   is readable but NOT re-aggregatable; W1's fires at `worker_map` (earlier, and pool-safe) and
   writes CSV **and pickle**, so `reaggregate` can recompute the two-frame comfort. Keep both.
8. ⚠️ **pandas' default CSV float parser is not round-trip exact, and a headline must never go
   through it.** MEASURED on E2's banked A1 CSV: the devkit's text cell reads
   `0.21845081236026753`, E2's `scores_summary.json` (pandas default) records
   `0.2184508123602675` — one ULP lower. The suite reads every HEADLINE with Python's `float()` on
   the devkit's own text cell; derived statistics still go through pandas exactly as E2 computed
   them, so E2's numbers reproduce. Without that split, "bit-for-bit" would have been false at the
   last digit.
9. **W-25 (W4's ask), implemented and pinned:** an arm whose stage 1 was answered by the devkit CV
   STAND-IN gets **no official two-stage headline** — `headline = {UNAVAILABLE, reason, n}`, the
   devkit's printed combined row is carried only as `statistics.official_combined_row_HYBRID`, and
   the honest stage-2 statistic (S2-EPDMS-u) sits beside it. The stand-in count is read from the
   SEAM ARTIFACT's own `source` column, not from a producer's report. Fixture: E2's real A1 arm
   (16 stand-in rows, devkit combined `0.21845081236026753`, S2-EPDMS-u `0.46702137433213753`).
10. ⭐ **`internal_t1` was RUN, not merely built** (the defect class CLAUDE.md names: *rollable and
   trained are different claims*). `python -m taniteval.bench internal_t1 --ckpt none --split
   v2ep-eval124clean-416x1024cyl-halfB --analyze-only <E9's banked dump> --labels <v8 eval labels>`
   → **COMPLETE in 8.4 s, 0 GPU**, contract- and schema-valid, with the tool's own numbers mapped:
   `os` ADE **2.9098** (T1*), `ha` 0.1946, `ha0` (the floor) 0.5417, `ha0_ext` 0.1884, paired
   `os − ha0` **+2.3682**, and every arm's interval **UNAVAILABLE** because the dump has 2 episodes
   < the RG-14 floor of 8 — the tool prints an interval there and the suite refuses to promote it.
   ⚠️ Not a capability claim: E9's dump is a 2,000-step smoke checkpoint on 9 windows.
11. ⚠️ **W5's report CLI exits 3 when the page renders but its failure GALLERY cannot be built**, and
   the first two runs recorded `REPORT_FAILED` although `report/index.html` was complete and W5's own
   verifier read *"PASS (90 numbers, 0 errors, 0 coverage gaps)"*. Fixed on the W1 side by the
   programme's own rule — **assert on the artifact, not the status**: a non-empty `index.html` is
   RENDERED, and a non-zero exit code is carried beside it as a warning with `report/verify.json`.
12. ⭐ **W2's settled finding travels with every command-conditioned arm** (and is in the summary at
   `controls.route_command_is_a_route_level_oracle` and per arm under `caveats`): NavSim's
   `driving_command` is a **ROUTE-LEVEL ORACLE** — computed from (ego pose now, nuPlan's route, the
   map), but the route is the expert's driven path at roadblock granularity, and on stage 2 the
   command is copied from the expert's own frame. ⇒ a command-conditioned NavSim number is
   *driving with an oracle route*; no route-following or strategic skill may be claimed from it.

## 3. ACCEPTANCE (SPEC §3)

<!-- ACCEPTANCE:BEGIN -->
**PASSED — all six criteria, from a clean shell, on the live devkit.**

```
python -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP
→ BENCH_STATUS=COMPLETE, rc 0, 303 s (CV 143 s + STOP ~150 s), device used "none" (no model inference), 1 worker
run of record: taniteval/results/bench/navsim_v2/warmup_two_stage/20260920T075608Z-navsim_v2-none-5e2458
replicates:    …/9f38f0 (490 s), …/ed36c4 (289 s), …/8fde2f (303 s) — different processes, md5-identical CSVs
report:        RENDERED by taniteval.benchreport (its own verifier: PASS, 178 numbers, 7 refusals, 0 errors)
gallery:       plans/{CV,STOP}.npz + scenes.json (220 tokens) emitted for W5's failure gallery
```

| id | criterion | result |
|---|---|---|
| ACC-1 | exit 0 · both records validate · the six files non-empty | **PASS** |
| ACC-2 | CV headline `0.1853562745165113` from column `score`, row `extended_pdm_score_combined`; ×100 → **18.5356** = the HF warmup LB (INHERITED) | **PASS** |
| ACC-3 | STOP headline `0.3009023137456225` | **PASS** |
| ACC-4 | cell level vs E1/E2's banked CSVs: **4,237 numeric cells, max \|Δ\| = 0.0**, NaN pattern identical, same row order | **PASS** |
| ACC-5 | file level: md5 `b4b33ac29c39c37832371b8b68c9f898` (CV) and `4a15c3a1d59234e518291a1a9e1d5bff` (STOP) — **byte-identical to the banked runs** | **PASS** |
| ACC-6 | the floors are machinery: a summary without STOP is refused | **PASS** |

⚠️ **What the run's own blob record says about drift:** three suite files differ from what that run
executed — `bench/cli.py` and `bench/contract.py` (the append-only helpers and the `tombstone`
subcommand, added minutes later at W4's request) and `bench/plugins/navsim_v1.py` (W3's file, which
W1 does not own). None is on the navsim_v2 scoring path, and the difference is VISIBLE because
`bench_run.json` carries a blob per suite file — that is the mechanism, not a footnote.

Verdict + every number: `raw/acceptance_verdict.json` (written by `code/verify_acceptance.py`, which
asserts on the ARTIFACTS, never on an exit code). Run log: `raw/acceptance_run.log`.

Per-stage, from the official summary rows: CV **0.46028921296917297** / **0.3341286472565139**,
STOP **0.5777242796695368** / **0.5212469877807625**; S2-EPDMS-u CV 0.39713167136274696, STOP
0.5212469877807624; paired CV−STOP headline **−0.11554603922911119** over 220 common tokens.
Controls: C4 (per-token EPDMS identity, 220 rows) PASS; C5 (aggregate recomputed from the frame with
Gaussian weights) PASS at max \|Δ\| **1.11e-16** (E1's C7 float-summation artifact, inside the 1e-9
tolerance); `criteria_check` **0 violations** per arm; the four NavSim gates PASS/NOT_APPLICABLE and
**all five gate mutations RED**; the export reproduction control vs E2's banked export:
**220/220 tokens, 0 fingerprint mismatches**.

⚠️ **The earlier attempts are in the same log and are part of the record.** Two were aborted by the
promoted wrapper's RAM guard (a 5,003 MB window collapsed to **1,176 MB within 20 s** under other
agents' jobs — `raw/ram_guard_abort_CV_counts.json`), which is how `RAM_GUARD_ABORT` + `retryable`
came to exist; the other COMPLETE runs preceded a later integration (W2's estimator, the
report-hook fix, the gallery inputs). **Every COMPLETE run produced md5-identical CSVs** — five
processes, five run ids, one answer.

⛔ **AND I BROKE A CONSUMER DOING THE TIDYING — the record must say so.** I deleted four of those
run directories while **W4's leaderboard was rendering**: the tree went from 4 dirs to 2 in ~20
minutes, W4's round-trip check read DIFFERS for a correct generator, and a published page ended up
citing a directory that no longer existed. Removed, with tombstones now in their place:

| run id | why it was removed |
|---|---|
| `20260920T070300Z-…-b56b47` | RAM_GUARD_ABORT — it scored nothing (evidence kept: `raw/ram_guard_abort_CV_counts.json`) |
| `20260920T070601Z-…-283271` | RAM_GUARD_ABORT, the second one — it scored nothing (its classification line is in `raw/acceptance_run.log`) |
| `20260920T072136Z-…-3123c8` | COMPLETE but superseded (pre-W2-estimator); CSVs md5-identical to the run of record |
| `20260920T072819Z-…-849012` | COMPLETE but superseded (pre-report-fix, pre-gallery); CSVs md5-identical to the run of record |

⚠️ **And the rule bit me back within the hour, which is the useful part.** A throwaway
`ls -dt | head -1` "newest run" line in my own verification chain picked a **TOMBSTONE** directory —
tombstones are written AFTER the run they withdraw, so they are the newest thing in the tree — and
`verify_acceptance.py` died on a `FileNotFoundError` for `bench_run.json` instead of saying so
(the run of record had already been verified separately, so no conclusion moved). ⇒ the verifier now
applies `contract.is_consumable_run()` FIRST and refuses with the reason (exit 2), and the lesson is
the one W4 stated: **a consumer never picks a run by modification time** — it names the run, or it
applies the skip rule.

⇒ **`taniteval/results/bench/**` is now APPEND-ONLY in the code, not only in a habit**
(`contract.py`): a withdrawn run keeps its directory and gains `TOMBSTONE.json` naming what replaced
it, a scratch run goes under `_scratch/` (`--scratch`), `python -m taniteval.bench tombstone <dir>
--superseded-by <run_id> --why …` deletes nothing, and `contract.is_consumable_run()` is the one
rule a consumer applies (skip `_scratch/`, skip a tombstoned directory, skip one still IN FLIGHT).
Four tests pin it, including one that walks the live tree.
<!-- ACCEPTANCE:END -->

**The offline twin (`taniteval/tests/test_bench_suite_navsim_offline.py`) runs the REAL benchmark
code path** — arm resolution, seam construction, artifact building, `criteria_check`, the summary
builder, schema + contract validation, the reference checks — with only the two devkit subprocesses
replaced by E1/E2's banked outputs. **11 tests, all green**, with these literals:

| check | expectation (literal) | result |
|---|---|---|
| CV headline, from column `score`, row `extended_pdm_score_combined` | **0.1853562745165113** | ✅ |
| CV ×100 rounded to 4 dp vs the HF warmup LB `baseline_constant_velocity` (INHERITED) | **18.5356** | ✅ |
| STOP headline | **0.3009023137456225** | ✅ |
| CV vs STOP, S2-EPDMS-u delta (E2's `scores_summary.json`) | **−0.12411531641801543** | ✅ |
| CV vs STOP, stage-2 per-scene W/T/L | **83 / 38 / 83** of 204 | ✅ |
| S2-EPDMS-u: CV / STOP | **0.39713167136274696** / **0.5212469877807624** | ✅ |
| E1 control C4 (per-token EPDMS identity, 220 rows) and C5 (aggregate recompute) | pass, max \|Δ\| ≤ 1e-9 | ✅ |
| four families on CV's 16 stage-1 scenes vs E1's artifact | speed MAE **0.8928**, cross **1.0658**, heading **7.4799** | ✅ |
| `criteria_check` violations per arm | **0** | ✅ |
| the four NavSim BLOCKING gates + their 5 mutations | PASS ×4, **all mutations RED** | ✅ |

## 3b. What a run hands the other packages

| file | consumer | content |
|---|---|---|
| `summary.json` | W4 (leaderboard), W5 (report) | per arm: headline (or its refusal), per stage, per log, submetrics, paired vs STOP and CV (pooled **and** per stage, with `_wtl_scope`), interval (or its refusal, with `pre_csv_frame`), the four families, `modality`, `caveats`, `statistics`, `controls` |
| `bench_run.json` | W4 | identity + provenance: `git_head`, every suite file's blob, `ckpt {path, sha256, registry_key}`, `devkit {repo, sha, patches[]}`, `split`, `arms[]`, `device` (+ `device_policy` when the shared gate ran), `wall_s`, `claim_bearing`, `status`, `report`, `files` (sha256 per file) |
| `scores/<arm>.csv` | W4, W5 | the devkit's own per-token output, UNMODIFIED |
| `artifacts/<arm>.json`, `criteria/<arm>.txt` | audit | the TanitEval artifact + `criteria_check` output (0 violations per arm) |
| **`plans/<arm>.npz`** + **`scenes.json`** | **W5's FAILURE GALLERY** | the poses the OFFICIAL SCORER executed (E1's `pdm_score` hook), with the seam's `source` labels so a `cv_standin` row is excluded; and per token: stage, log, \|v0\|, the NavSim command, scene/pickle ids, the frame bank and maps root |
| `raw/**` | everything | wrapper manifests, hooks, the PRE-AGGREGATION dump (CSV + pickle), the runner logs, the export record, the preflight, the controls |

## 4. Tests (all W1 files green)

`taniteval/tests/test_bench_suite_{legacy_compat,promotion,schema,contract,gpu_gap,submission,internal_t1,navsim_offline}.py`
— literals everywhere, and a RED mutation per binding rule: **M-COL** (reading `pdm_score`),
**M-FLOOR** (a NavSim summary without STOP or CV), **M-GAP** (the gap comparison inverted),
**M-SUBMIT** (a gate that ignores the register), **M-LEGACY** (delegation removed),
**promotion** (a one-token change to a promoted function must break the AST pin).

## 5. Promotion, not rewriting

| origin (E1/E2, unchanged) | product | pinned by |
|---|---|---|
| `navsim_win.py`, `tanitad_seam_agent.py`, `export_agent_inputs.py` | `bench/navsim/devkit_side/` | git blob == origin blob (`PROVENANCE.json`); `navsim_win.py` additionally carries ONE marked W1 ADDITION whose removal reproduces E1's file byte-for-byte |
| `verify_controls.py` (`epdms_formula`, `c4`) · `parse_scores.py` (`s2_group_uniform`) | `bench/navsim/summarize.py` | AST-identical |
| `build_artifacts.py` E1 (`short_row`, `build_win`, `navsim_gates`, `gate_mutations`) · E2 (`_set`, `_get`) | `bench/navsim/artifacts.py` | AST-identical |
| `tanitad_navsim_bridge.py` (16 functions/classes + 10 constants) | `bench/navsim/bridge.py` | AST-identical; exactly ONE changed line (the REPO depth) |
| `score_arm.py` · `make_seam_{stop,cv,echo}.py` · `run_bridge.py` | `bench/navsim/{scoring,seams,model_arms}.py` | the guards and the per-token loop are E2's; glue restructured (declared in the module docstrings) |

## 6. What is NOT done, and what blocks it

| # | item | blocked on |
|---|---|---|
| 1 | the cuda path of the gap launcher | the GPU is training's (A8 → the S1 pass → A7). Mock-tested only (21 literal tests + a RED mutation); **UNVERIFIED ON HARDWARE**, and the suite refuses the card while any training process is alive |
| 2 | navhard (`--split navhard_two_stage`) | E1 owns the aggregation-crash diagnosis (their re-run was live on this box while W1 ran). The suite's handling of that class is implemented and tested — pre-aggregation dump, `AGGREGATION_FAILED`, `reaggregate` — and the profile already points at the waiter's cache |
| 3 | model arms (`--ckpt <path>`) end to end | heavy CPU + a frame bank per split; the code path is E2's, promoted and seam-tested, but no live model run was made here. On warmup it could not produce an official headline anyway (W-25) |
| 4 | a numeric interval on any NavSim row | ≥ 8 log clusters: warmup has 7 and can never carry one. navhard (76) can, through W2's settled estimator, as soon as a navhard run completes |
| 5 | the official two-stage EPDMS of a MODEL arm | a REAL stage 1: warmup ships no stage-1 camera frames, so it needs navhard (whose stage-1 frames exist) — i.e. item 2 first. Until then the suite REFUSES the headline (W-25) rather than printing the devkit's hybrid number |
| 6 | a submission anywhere | the PI. `submit` refuses today and the upload step is deliberately NOT BUILT |

## 7. After the code freeze: two events on a shared tree (2026-09-20, both MEASURED)

⚠️ **A W1-owned file was edited by another stream AFTER my freeze — and my own end-of-turn check is
the only thing that saw it.** `bench/navsim/artifacts.py` read `MISMATCH` at 281/282 (index
`26010543…`, worktree `f87ce653…`, mtime 10:11). The edit is **W2's**, and it is CORRECT: since
W2's source-level fix, `four_families.lateral` emits its own refusal, so `refuse_undefined_lateral`
normally repairs **nothing** — the edit adds an early `continue` so a term already refused upstream
is *counted as refused* instead of being re-written, and the seam survives as the regression guard
for a return to `None`.

⭐ **I did not take that on the diff's word — I exercised BOTH branches, which is the discriminating
control the diff alone cannot give:**

| input artifact | returned | what happened to the term |
|---|---|---|
| legacy, `heading/curvature/yaw = None` | `['heading_mae_deg','curvature_mae_1pm','yaw_rate_mae_degps']` | **repaired here** → `{status: UNAVAILABLE, n: 0, reason: …}` |
| W2-refused at the source | the same three names | **passed through untouched** (already refused) |

Then the four affected test files (`navsim_offline`, `promotion`, `contract`, `schema`) — **91
passed**. ⇒ staged, blob `f87ce653b11858a8db9115329f515d4cc1539c08`, index == worktree. Leaving it
unstaged would have stranded a sibling's verified fix inside my tree; reverting it would have been
the revert-a-sibling failure in `CLAUDE.md`, from the far side.

⛔ **The lesson is not "W2 should have asked".** It is that **an ownership boundary is not a lock**,
so *the blob comparison must be re-run at the END of the turn* — exactly as the operating rule says
— and a MISMATCH on a file you did not touch is **evidence about the tree, never about your edit**.
My freeze was at 09:5x; the edit at 10:11; nothing in my package would have shown it.

⭐ **And the same class, caught from the other direction: W5 renamed `benchreport.html` →
`benchreport.page` mid-stream** (their own retraction records it as an *interface* change, not a
tidy). `report_hook.py` was unaffected — because it shells out to `python -m taniteval.benchreport
<run_dir>` and imports **no internal module of theirs**. Re-verified at the tip today:
`taniteval.benchreport` imports, `__main__` is present, and `page` / `charts` /
`leaderboard_charts` / `render` all import. ⇒ **the subprocess seam was worth its cost**: a
sibling package renamed a module underneath us and the contract held. Any future consumer of W5
should call the CLI, not the submodule.

## 8. E1's four relayed items (2026-09-20) — all done, all MEASURED

| # | item | what landed |
|---|---|---|
| a | **RAM-guard `sustain` 3 → 60** | `navsim_win.py` re-promoted (origin `96d13540` → `11d5e3cc`, product `64186cce` → `935fa44c`). Hard floor (one sample < 2,000 MB) unchanged, so a genuinely dangerous state still aborts at once |
| b | **dry-run vocabulary** | a passing dry run prints **`BENCH_STATUS=DRY_RUN_PASSED`**, exit 0 — normalised at the CLI seam, which fixes W3's plugin too. Mutation **M-DRYRUN** proven RED then restored |
| c | **IDM degenerate-path patch** | added to `devkit_patches()` (6 entries) with a live raw-bytes blob and E1's framing: 166/5,462 stage-2 tokens, one assert takes the run down ⇒ **a precondition of the number existing** |
| d | **stage-1 cross-check** | `summarize.stage1_reference_check` + banked reference + wired into the navhard summary; mutation **M-S1REF** |

⭐ **(a) was re-applied by 3-WAY ALIGNMENT, never by line number.** E1's edit sits ABOVE both W1
ADDITION blocks and shifts them **+7**, so a line-number patch would have silently mis-placed them.
The binding check ran **before** the write, and it is the promotion test's own rule: stripping the two
marked blocks from the product reproduces E1's origin **byte-for-byte**
(`11d5e3cc… == 11d5e3cc…`). Tool: `code/w1_repromote_navsim_win.py`.

### ⚠️ The blob that looked like a mismatch and was not — `git hash-object` NORMALISES CRLF

I first read the IDM file's blob as **`214ef5ee…`** while E1's record says **`b95bcc7f…`**, which
reads exactly like a patch applied twice or a file changed underneath us. **Both numbers are
correct and they answer different questions.** The file is **100 % CRLF (166/166)**, and
`git hash-object` applies the text/eol conversion, so it hashes the **LF-normalised** bytes; E1
hashed the **raw** bytes. `profiles.git_blob()` reads raw bytes and **agrees with E1 exactly**
(MEASURED) — ⇒ **the product was never wrong; my shell probe was.**

⛔ **The trap is live for anyone verifying these entries by hand**, so it is recorded in the entry's
own `reader_note` rather than in a report nobody re-reads. Same family as every scope trap in
`CLAUDE.md` — *a true measurement quoted outside its scope reads exactly like an answer* — with the
scope being **which bytes were hashed**. ⚠️ And note the direction of the damage: the normalising
hash is **blind to a pure line-ending change**, so it is the weaker probe *and* the one that
manufactures false alarms.

### ⭐ E1's published-match claim, re-derived from the PRIMARY rather than inherited

E1 reported that the navhard CV stage-1 sub-metrics match the published leaderboard **under
truncation**. Rather than quote that, I read the column out of the **banked PDF** myself — library
key **`2506.04218`** (*Pseudo-Simulation for Autonomous Driving*, sha256 `a8431697ffa2…`, present on
disk, 5,341,242 B), **p.8 Table 2**, column **`CV [8]`**:

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|---|
| **E1 MEASURED** (n = 450) | 88.89 | 42.89 | 70.67 | 99.33 | 77.53 | 87.33 | 78.67 | 97.11 |
| **PUBLISHED** (1 dp) | 88.8 | 42.8 | 70.6 | 99.3 | 77.5 | 87.3 | 78.6 | 97.1 |
| truncating | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| rounding | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |

**8/8 truncating · 4/8 rounding.** E1's methodological catch reproduces exactly, and the four that
fail under rounding are **precisely the cells whose 2nd decimal is ≥ 5** — which is what makes it a
rule rather than a coincidence. ⚠️ **A reproduction rule written around rounding would mark a
CORRECT reproduction as a MISS on half the metrics.** EC is excluded on purpose: it is **masked** in
the two-stage EPDMS. `_trunc1` uses `Decimal`, not `int(x*10)/10`, so 78.67 truncates **by rule**
rather than by binary-float luck.

⛔ **Our navhard stage-1 numbers must equal E1's EXACTLY** — same split, same scorer, same 450
scenes. That is an **identity** check, not a tolerance, and `M-S1REF` moves one cell by 0.01 to
prove the check can fail. The published comparison is the weaker, independent one.

**Tests: 161 passed** (153 + 4 dry-run + 4 stage-1), all files green at the tip.

## 9. FINDINGS THAT GENERALISE (recorded at the orchestrator's request, 2026-09-20)

### F1 — ⛔ `git hash-object` NORMALISES LINE ENDINGS; RAW-BYTE HASHING DOES NOT. Two "correct" blobs disagree, and the normalising one is BLIND to the change it is most likely to miss

**MEASURED 2026-09-20** on the devkit's IDM file, 100 % CRLF (166/166 lines):

| how the bytes were hashed | blob |
|---|---|
| `git hash-object <path>` (applies the text/eol conversion) | `214ef5ee39944c3584cb16eaaffd53853fc9db85` |
| **raw bytes** (`sha1("blob <len> " + bytes)`) — E1's record, and our `profiles.git_blob()` | `b95bcc7f62246017ea9ab5dca92383c6b50d234b` |

⚠️ **Neither number is wrong. They answer different questions**, and nothing in either output says
which. I read the first, compared it to E1's record, and had a mismatch that looked exactly like a
patch applied twice or a file changed underneath us — a **false alarm about a correct file**.

⛔ **The worse half is the silent one:** the normalising hash is **blind to a pure line-ending
change**. A file converted CRLF→LF hashes *identically* under `git hash-object` while its bytes on
disk are different — so the probe most people reach for is the one that **cannot see** the change a
Windows/MSYS toolchain is most likely to cause.

⇒ **Rule: for any provenance blob, hash RAW BYTES and say so in the record.** Our
`contract.code_blobs()` and `profiles.git_blob()` already do; `_devkit_side_blobs()` now documents
it at the function. ⭐ Same family as every scope trap in `CLAUDE.md` — *a true measurement quoted
outside its scope reads exactly like an answer* — with the scope being **which bytes were hashed**.

### F2 — ⭐ An INDEPENDENT re-derivation, not an inheritance: the published navhard column, read from the banked primary

E1 reported that its measured navhard CV stage-1 sub-metrics match the published leaderboard **under
truncation**. Rather than quote that (which would have been `INHERITED` and inadmissible for a
registry row), I opened the **banked PDF** — library key **`2506.04218`**, sha256 `a8431697ffa2…`,
present on disk — and read **p.8 Table 2, column `CV [8]`** myself with `pypdf`.

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|---|
| **E1 MEASURED** (n = 450) | 88.89 | 42.89 | 70.67 | 99.33 | 77.53 | 87.33 | 78.67 | 97.11 |
| **PUBLISHED**, read by W1 | 88.8 | 42.8 | 70.6 | 99.3 | 77.5 | 87.3 | 78.6 | 97.1 |
| truncating | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| rounding | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |

**8/8 truncating · 4/8 rounding**, and the four that fail under rounding are **exactly the cells
whose second decimal is ≥ 5** (NC .89, DAC .89, DDC .67, LK .67). ⭐ That last clause is what turns a
coincidence into a **rule**: the failures are not scattered, they are the precise set the arithmetic
predicts. ⚠️ **A reproduction rule written around rounding would mark a CORRECT reproduction as a
MISS on half the metrics.** EC is excluded deliberately — it is MASKED in the two-stage EPDMS.
Evidence class: **PUBLISHED (primary, banked by library key)** — no longer `PUBLISHED-SECONDARY`,
and this is exactly what banking the PDF buys.

### F3 — ⭐ PER-ARM provenance, and a correction to my own escalation

**Implemented on the orchestrator's ruling.** `scoring.score_arm` now reads **every devkit-side
file's blob microseconds before each arm's subprocess launches**, in the **same hash space** as
`bench_run.json.git.suite_code_blobs`, and diffs them: `devkit_side_blobs_at_launch`,
`devkit_side_drift` and `devkit_side_blob_basis` land in `<arm>.counts.json`, and a non-empty drift
is **logged as a warning at the moment it happens**. `CODE_CHANGED_MIDRUN.json` is now the fallback,
not the mechanism.

⚠️ **CORRECTION to what I escalated earlier.** I wrote that a mid-run change was invisible to the
run record. **That was too strong.** `score_arm` already captured **`wrapper_sha256`** per arm at
launch — so the wrapper's change *was* recorded. What was genuinely missing was narrower and worth
stating precisely: **(i)** it covered only the wrapper, not the other four devkit-side files;
**(ii)** it is a **SHA-256**, so it is **not comparable** to the run-start snapshot's git blobs —
two provenance fields in **different hash spaces**, which is F1's problem in a second costume; and
**(iii)** nothing ever **diffed** it, so a drift was recorded and never **flagged**. ⛔ *A value that
is stored but never compared is not a guard.* `wrapper_sha256` is kept unchanged — it is E2's banked
record shape — and the new fields sit beside it.

Pinned by **M-DRIFT**, whose positive case is **today's actual event** (`navsim_win.py`
`64186cce` → `935fa44c` while a navhard run was scoring) and whose three negative controls exist
because **a detector that cries wolf on every arm is a detector nobody reads**: identical blobs →
quiet; a path absent from the baseline (`suite_code_blobs` globs `*.py`, so `PROVENANCE.json` is
legitimately missing) → **not** drift; an empty baseline → **no drift CLAIM**.

### The standing rule this session earned (orchestrator, 2026-09-20)

⛔ **Changing a file that another stream's LIVE run depends on requires ALL THREE:** an argument that
**no output can move**, a **provenance record inside the run**, and a **notification to that run's
owner**. Two of the three are things you can do alone; the third is not, and it is the one that
stops the owner learning about it from a hash later.

### F4 — the resolution E1 asked for, and the thing it caught in our own record

**Every file-backed `devkit_patches()` entry now carries BOTH hashes, each labelled with what it is
blind to** (`raw_bytes_git_blob_now`, `normalised_git_blob_now`, `hash_basis`).

⭐ **And carrying both immediately found a live ambiguity in our own record.** The
`dataclasses.py` entry's evidence text cites E1's **`596cb7d`** — that is the **NORMALISED** id. The
raw-byte id for the same file is **`20673663…`**. So the historical evidence note and the hash field
beside it were **in different hash spaces**, and anyone checking one against the other would have
found a "mismatch" on a perfectly correct file. That is the same false alarm F1 cost me, sitting
latent in the record. It is now unambiguous.

⚠️ **One correction to the relayed refinement.** It stated that our `git_blob()` normalises CRLF→LF.
**It does not, and never did** — `profiles.git_blob()` reads raw bytes with no subprocess and returns
`b95bcc7f…`, **agreeing with E1's own patch records**. The normalising number came from
`git hash-object` typed at a shell. The substance of the ask was right and is implemented; only the
attribution needed fixing. ⭐ *Worth noting that this is F1 claiming a third victim — the split is
easy to mis-attribute even by the person who found it.*

**Controls, because one sign proves nothing:** a 100 %-CRLF file must give **two different** ids (or
the normalisation is not happening) **and** an LF-only file must give the **same** id (or the two
functions are not the same hash family at all). `fcntl.py` supplies the second: raw == normalised.

### F5 — a mid-run change has an ASYMMETRY that "no output can move" does not cover (E1)

⭐ **The two arms of the live navhard run ran under DIFFERENT abort policies** — CV under
`sustain=3`, STOP under `sustain=60` — because a running process does not re-read its source. ⇒ if
CV completes, **its number is exactly as valid as STOP's**; but ⛔ **"CV survived" is NOT evidence
that `sustain=3` was safe** — the same policy killed E1's own navhard CV run at **5,576/5,912
(95 %)**. Survival under a policy is a **draw from a distribution, not a property of the policy**.
And had CV died, **it would have been the POLICY, not the DATA** — which is why `scoring.py`
classifies that case as `RAM_GUARD_ABORT` with `retryable: true` rather than a failure.

⇒ Recorded in the run's own `CODE_CHANGED_MIDRUN.json` under `THE_ASYMMETRY_THAT_MATTERS`. ⚠️ **A
note that records only "no score can move" leaves a reader believing the two arms were equivalent in
every respect that matters.** They were equivalent in *value* and not in *exposure*, and only the
first is what that phrase asserts.

## 10. Report curation — the ruling's ⚠️ clause, settled by measurement

The orchestrator's ruling carried a conditional: *stage the run of record's report and regenerate
the rest — **unless the render is NOT deterministic**, in which case "regenerable" is a weaker claim
and we stage more, not fewer.* ⭐ **That clause is the whole decision, so I measured it instead of
assuming it.**

**MEASURED 2026-09-20.** The same run (`…ed36c4`) copied twice to scratch, `report/` deleted from
both, rendered independently: **same file set, 11/11 BYTE-IDENTICAL**. The rendered page carries
**no render-clock stamp** — its only embedded timestamps are `2026-09-20T07:35` (the run's own id)
and one data timestamp, while the renders happened at ~11:20 — and **no absolute paths**.
⇒ **Deterministic. "Regenerable" is a valid claim**, and the ruling's rationale holds.

⚠️ **With the scope stated, because "deterministic" alone would be true-but-misleading:** it is
deterministic **relative to a RENDERER VERSION**. W5's renderer is live code — a module was renamed
mid-stream today — so a regeneration months from now reproduces *that* commit's report, not
necessarily this byte sequence. **The numbers come from `summary.json`, which IS in git**, so the
evidence survives either way; only the pixels are version-bound.

### ⛔ Two corrections to the ruling's premise, both measured

1. **The reports were not "pending" — all 67 were ALREADY STAGED**, by me, as run owner, through my
   blanket `--extra $(find taniteval/results/bench -type f)` staging. W5 staged none, correctly. So
   the action was not *"stage one"* but ***unstage 42***, which is a different operation with a
   different risk profile — an index write on a shared index while W5 was bulk-adding hundreds of
   gallery PNGs.
2. ⛔ **An ignore rule cannot take effect on a file that is already in the index**, so writing
   `.gitignore` first and checking it proves nothing. `git check-ignore` reported the file I intended
   to drop as **TRACKED**, not IGNORED — ⭐ *and that is the negative half of the test doing its job*:
   had I only asserted "the kept file is tracked", I would have shipped a dead rule. Unstage first,
   **then** the rule binds.

### What I did

| | |
|---|---|
| **kept in git** | the NavSim v2 **run of record** (`…5e2458`, 22 files, 2.905 MB) and **internal_t1** (`…04264a`, 3 files, 0.051 MB) — so at least one rendered report is readable straight from the repo, which is what the PI asked for |
| **unstaged + regenerable** | `…9f38f0` (9 files), `…ed36c4` (11), `…8fde2f` (22) — **3.187 MB**, of which `…8fde2f` alone is **2.906 MB (91 %)** |
| **mechanism** | `taniteval/results/bench/.gitignore` — a **local** ignore file inside W1's own tree, ⛔ **not** the shared root `.gitignore` that every stream depends on |
| **recorded in the run** | `REPORT_NOT_IN_GIT.json` in each affected run dir: why, the regenerate command, and the renderer-version caveat |

⚠️ **The pattern excludes `report/**` (contents), not `report/` (the directory), on purpose.** Git
**cannot re-include a file whose parent directory is excluded**, so the directory form would have
made every negation below it **silently dead** — a rule that reads correctly and does nothing. Both
signs are asserted: a dropped file must be ignored **and** a kept file must not be.

⭐ **The run itself is fully in git either way** — `bench_run.json`, `summary.json`, `scores/`,
`artifacts/`, `criteria/`, `raw/`. Only the rendered pixels regenerate.

## 11. The navhard report-hook failure — my defect, and it was hiding behind a default

E1's navhard run scored **COMPLETE** and its report **FAILED**. The cause is mine.

**`navsim/plans.py::write_scenes` emitted `"frame_bank": null`.** W5's gallery reads
`Path(run.scenes.get("frame_bank", ""))` — and ⛔ **a dict default fires only on an ABSENT key, never
on a present-but-null one**, so `.get(k, "")` returned `None`, `Path(None)` raised `TypeError`, and
the **entire report of a complete run died** on a key that was never load-bearing for the numbers.

⭐ **Why it survived every previous run — and why "it worked on warmup" proved nothing.**
`model_arms.DEFAULT_BANKS` has an entry for **warmup** and **none for navhard**. So warmup always
wrote a real string and navhard always wrote `null`. **The defect was in the suite from the first
commit; a default masked it** until the first split without one. *Same family as every scope trap
here: the probe (a warmup run) could not see the failure it was supposed to rule out.*

### Fixed as a CLASS, not as a key

`write_scenes` now drops **every** top-level null before writing (`_drop_top_level_nulls`) and records
which keys it dropped and why, so no future key can repeat this. ⚠️ **Top level only, on purpose** —
`tokens` legitimately carries per-token nulls that W5 reads with `meta.get(...) or <fallback>`, which
*is* null-safe; dropping those would change the token schema to fix a problem that does not exist.

**And the failure mode is now honest at the record level.** It was already `REPORT_FAILED` rather
than a quiet pass — but the record carried only `rc`, `index_exists` and a **log path**, so the cause
was invisible to every consumer unless a reader knew to open a file. `report_hook` now records
**`reason`** (the last real error line) and **`how_to_retry`**. ⚠️ On an unreadable or empty log it
returns a **stated** `UNAVAILABLE…` string, never `""` — an empty reason reads as *no* reason, which
is the same lie as no field at all.

### The run was recovered, not just diagnosed

I repaired that run's `scenes.json` (null key removed, repair recorded inside the file) and
**re-rendered through the real hook**: **`RENDERED`, index.html 204,420 bytes, W5's own verifier
`PASS` — 481 numbers, 5 refusals, 0 errors.** `rc` is **3** because navhard has no frame bank so the
gallery is legitimately unavailable — carried as a `warning` beside the page, which is exactly the
"assert on the ARTIFACT, not the status" rule that was already in the hook. `bench_run.json.report`
now records the render **and** `repaired_after_failure` with the original failure verbatim, whose
defect it was, and that **no score was ever involved** — the hook runs after scoring, in a
subprocess, and the run was `COMPLETE` throughout.

⭐ **W5 can render such a run after the fact** — proven by doing it, on the untouched run directory.

Pinned by **M-NULLSCENE** (re-add the null to any `scenes.json` → RED, verified then restored), which
asserts on the **ARTIFACTS in the tree**, not on the writer: the run that broke was written by an
earlier writer, so testing the writer alone would have passed while the tree stayed broken. It
carries a control requiring that it actually read at least one file.

## 12. navhard landed — and the suite reproduces E1 EXACTLY

⭐ **The cross-check E1 asked for, run the moment the arm landed** (`code/w1_stage1_check.py` on the
suite's own `scores/CV.csv`):

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|---|
| **ours** (n = 450) | 88.8889 | 42.8889 | 70.6667 | 99.3333 | 77.5334 | 87.3333 | 78.6667 | 97.1111 |
| **E1** | 88.89 | 42.89 | 70.67 | 99.33 | 77.53 | 87.33 | 78.67 | 97.11 |

**`identical_to_e1 = True`, 8/8 at 2 dp, n = 450 as expected, 8/8 truncating vs published (4/8
rounding).** Artifact: `raw/navhard_CV_stage1_check.json`.

**Banked in `references.json`:** navhard **CV 11.4816** ×100 with the **log-cluster interval
[8.25, 14.50]** (S1 28.96 · S2 34.25), **Δ 0.0000** against the HF navhard leaderboard under
truncation (11.4816 → **11.4**; rounding would give 11.5 and read as a miss); **STOP 29.8532**;
**HUMAN stage-1 93.4796** (stage 1 only — the human agent is undefined on two-stage splits, which is
why the suite refuses that headline). ⭐ **This is the programme's FIRST quotable NavSim interval**:
warmup has **7** log clusters and can never carry one (the floor is 8); navhard has **76**.
⇒ §6 items **2 (navhard)** and **4 (an interval)** are closed.

⚠️ **And E1's 19/19 SUPERSEDES my 8/8 as the primary read of the truncation convention.** E1
reproduced **all 19 published terms** of [N2] v3 Table 2 exactly under truncation (**9/19** under
rounding). My 8 are the CV stage-1 column minus EC — a **subset** of E1's 19. They agree where they
overlap, and E1's is strictly stronger evidence. Recorded as such in `references.json` rather than
leaving two competing counts in the record.

## 13. `provenance.ckpt` — W4's gap, made structural, and it caught an anonymous row immediately

`summary.json` omitted the checkpoint triple. Harmless on a devkit-floor run; **fatal for the model
arm the PI is waiting for**, because the leaderboard reads `summary.json` and could not say WHICH
checkpoint produced a row.

**Emitted** — `provenance.ckpt` is now **the same object `bench_run.json` carries**
(`ctx.rec["ckpt"]`), never a second triple assembled alongside it: ⭐ *two independently built
triples are two things that can disagree, and only one of them is read.*

**Structural, not documented.** `contract.validate_summary` refuses a summary with a `model` arm
whose checkpoint is not identified — and `write_summary` raises on it, so such a run **FAILS its
contract instead of publishing**. A convention would be a request; this is a refusal.

⚠️ **One refinement, with its reason.** Identity (`path`, `sha256`) is a **hard** refusal. The
**registry key** is a different claim — it *links* an identified checkpoint to `MODEL_REGISTRY.md` —
and a run identified by sha256 is **not anonymous**, so refusing outright would reject a reproducible
result over a missing cross-reference. It must therefore be **present or explicitly accounted for**
(`registry_key_status`, which `cli.py:138` already writes when `--registry-key` is omitted). ⛔ What
is forbidden is **silence**.

### ⭐ It found a real anonymous row in its first run — in a different benchmark

The new check immediately failed the banked **`internal_t1`** run: its **`os` arm is `kind: model`**
and its checkpoint was **null**. The path was not missing from the world — it sat one file down in
`raw/refcv3_arm.json` at **`refcv3.manifest.model.ckpt`** — because `internal_t1.summarize_record`
read **`rec["ckpt"]`**, which is `None` on *every* run of that tool.

⇒ `ckpt_path_from_record` now reads the manifest, `run_benchmark` computes the sha256 and writes the
**same** object into `bench_run.json` and `summary.json`, and the banked run was backfilled:
`C:/Users/Admin/tanitad-caches/a3-heldout-20260919/run/ckpt.pt`, sha256 `54320ec2d72a0b6f…`, both
files agreeing. **That row was on the leaderboard path, anonymous, and nobody had noticed** — found
by making the rule structural, not by inspection.

Pinned by **M-ANONROW** (null the ckpt on a model-bearing summary → RED, verified then restored),
asserted on the **artifacts in the tree**, with a control requiring the tree to contain at least one
model-bearing summary so it cannot pass vacuously.

## 14. ⛔ The promotion pin fired again — and this time the origin DELETED what I promoted

Not caused by my own work: `test_promoted_functions_are_ast_identical` went RED because **E1
restructured `build_artifacts.py`** (mtime 14:09). It now defines only
`{build_win, gate_rows_from_criteria_check, main, read_csv_rows}` — **`short_row`, `navsim_gates`
and `gate_mutations` are gone.**

⛔ **And the version that contained them is UNRECOVERABLE.** That file is **not in `HEAD`** (never
committed) and the index already holds the new blob `53051383`, so there is **no blob to compare
against**. ⇒ **W1's copies are now the only surviving ones.**

⭐ **So the claim changed, and the pin changed with it instead of being quietly deleted.**
"AST-identical to the live origin" is no longer checkable for those three, so they move to
**`ORPHANED_PROMOTIONS`**, pinned on what *is* checkable — that my copy has not drifted since the
moment of orphaning (`sha256(ast.dump)[:16]`). `build_win` is still in the origin and **still
AST-identical**, so it keeps the real pin.

⚠️ **The tempting fix was to drop the three names from `PROMOTED`, and it would have been the worst
one:** the suite would go green while the guarantee silently shrank — precisely the failure this
file exists to prevent. ⭐ The orphan pin therefore ships with a **discriminating control**: a test
asserting those names are *really absent from the live origin*, which goes **RED if E1 restores
one**, forcing a move back under the real AST pin rather than letting an "orphaned" claim outlive the
condition that justified it. *(A stale absence-claim is the exact failure class `CLAUDE.md` logs
against itself four times over.)* It also fails if the origin file disappears, rather than passing
vacuously.

**Tests: 181 passed.**

## 15. FINDING, NAMED: ⛔ **THE PRESENT-BUT-NULL KEY** — a dict default fires only on an ABSENT key

*(Recorded as a named class at the orchestrator's request, 2026-09-20, because it generalises well
past the bug that produced it.)*

```python
{"frame_bank": None}.get("frame_bank", "")   # -> None   ⛔ the default NEVER fires
{}.get("frame_bank", "")                     # -> ""     ✅ what the author expected
```

**The author of `.get(k, default)` wrote that default specifically to avoid the value they got.**
The producer hands over a key that *exists*, so the consumer's guard is bypassed — and the failure
surfaces far from its cause, inside whatever the default was protecting (`Path(None)` here, three
frames deep in a renderer, killing the whole report of a **COMPLETE** run whose scores were fine).

⇒ **The rule: never emit a null into a shared artifact. Absence is the honest encoding of
"not declared".** `write_scenes` now drops every top-level null and records which. ⚠️ Top level only:
`tokens` carries per-token nulls that W5 reads with `meta.get(...) or <fallback>`, which *is*
null-safe — widening the fix would change a schema to solve a problem that does not exist there.

### ⭐ Why nobody caught it, which is the more useful half

`model_arms.DEFAULT_BANKS` has an entry for **warmup** and **none for navhard**. So warmup always
wrote a real string and navhard always wrote `null`. **The defect was in the suite from the first
commit, and every warmup run — including the whole bit-for-bit acceptance — exercised the code path
while being structurally incapable of failing.** ⛔ ***"It works on warmup" could never have found
it.***

⭐ **This is the `df` / Thor `free` / cgroup `usage_in_bytes` family with the object swapped: not a
probe reporting the wrong scope, but a TEST SUITE whose only fixture could not reach the failing
branch.** A default that makes the common case work is exactly what hides the uncommon one. ⇒ when a
lookup table has an entry for some keys and not others, **the missing-entry path is the one that
needs the test**, and "all our runs pass" is a statement about which keys we happened to use.

### And the pin is on the ARTIFACTS, not the writer

**M-NULLSCENE** asserts over every `scenes.json` **in the tree**, not over `write_scenes`. ⚠️ The run
that broke was written by the *old* writer, so a test of the *fixed* writer would have passed **while
the tree stayed broken** — a green suite over a broken artifact. It carries a control requiring that
it actually read at least one file, so it cannot pass vacuously.

## 16. A checkpoint cell is NEVER blank (orchestrator ruling)

`registry_key` stays optional-if-stated — **confirmed as designed**: `path` + `sha256` identify a
checkpoint, so a reproducible result must not be refused over a missing cross-reference, provided the
absence is *stated*. **Added:** wherever a row renders without a key it must show the **sha256 prefix
in the key's place, never a blank** — ⛔ a blank reads as *"no checkpoint"* when the truth is
*"identified, not cross-referenced"*, and those are different claims.

`contract.ckpt_display()` is the single implementation and its result is **written into the
artifact** (`provenance.ckpt.registry_key_display`) rather than re-derived per renderer — ⭐ the same
reasoning that made `provenance.ckpt` the *same object* `bench_run.json` carries: two producers
formatting one row independently are two things that can disagree.

| state | shows |
|---|---|
| key present | the key |
| no key, sha256 present | `sha256:54320ec2d72a` |
| path only | `UNIDENTIFIED (path only, no sha256)` |
| devkit floors | `no checkpoint (devkit floors only)` |

All six banked runs backfilled; the `internal_t1` model row now reads **`sha256:54320ec2d72a`** where
it used to have nothing. Pinned by **M-BLANKKEY** (blank one in the tree → RED, verified then
restored) — again asserted on the artifacts, since the renderer reads the artifact, not the function.

## 17. ⛔ RETRACTION — I merged two artifacts into one reference, and the entry refuted itself on its own face

W3's full-scale v1 result settles the convention question, and settling it **exposed an error in my
own banked record**.

**What I banked (2026-09-20, navhard CV `external`):**

```json
{"name": "HF navhard leaderboard", "value_x100": 11.4, "round_dp": 1,
 "delta": 0.0, "rule": "TRUNCATE to 1 dp"}
```

⛔ **Our value is 11.4816. `11.4816 − 11.4 = 0.0816`, not 0.0000.** The entry carried a stated delta
that its own two other fields contradict. ⭐ **No new data was needed to catch it — only arithmetic
over the fields already in the record — and I banked it anyway.**

**The cause: two artifacts, merged.** E1 reported *"Δ 0.0000 against the HF navhard leaderboard"*,
which is only possible at **4 dp**; separately E1 reproduced the **paper's** Table 2 under
**truncation to 1 dp**. I took the paper's convention and stapled it to the leaderboard's row.

**Corrected — they never disagreed:**

| artifact | prints | dp | rule | ours |
|---|---|---|---|---|
| arXiv 2506.04218 v3, Table 2 | **11.4** | 1 | **TRUNCATES** | 11.4816 → 11.4 ✅ |
| HF navhard leaderboard | **11.4816** | 4 | **ROUNDS** | Δ **0.0000** ✅ |

⭐ **W3's measurement is what settles it, and it supersedes both earlier partial reads** (my 8/8 and
E1's 19/19 were both about the **paper**): **ONE underlying value, `20.65165…`, explains all three
sources** — the paper prints **20.6** (truncated to 1 dp), the leaderboard prints **20.6517**
(rounded to 4 dp). ⚠️ And rounding to 1 dp would give **20.7**, which is *not* what the paper prints —
so the two conventions are genuinely different, not an artefact of precision. W3's value landed
inside a **pre-registered discriminating window**, which is what makes it evidence rather than a
coincidence.

### The rule this earns

⛔ **STATE THE ARTIFACT BEFORE THE CONVENTION.** *"NavSim publishes X"* is not a claim until you say
**which publication**. The paper and the leaderboard are different renderings of the same number, and
quoting one's convention against the other's row manufactures a mismatch — or, as here, a **false
agreement**, which is worse because nothing looks wrong. *Same family as the `df` / cgroup / `step_s`
traps, with the scope being **which artifact printed it**.*

### The check that would have caught it, now in the suite

**M-SELFREF** re-derives every external reference from the entry's **own** fields: our value under
the entry's **own stated rule** at its **own stated dp** must equal the published value, and any
stated delta must equal `|ours − published|`. Reinstating the exact defective entry makes it go RED
with the diagnosis in the message: *"ours 11.4816 under ROUNDS at 1 dp -> 11.5, but the entry says
11.4 — the entry contradicts itself"*. ⭐ **It needs no external data at all** — which is precisely
why it should have existed before the entry was written.

## 18. v1 navtest, full split — banked, and the summary-row stamp confirmed at scale

`summary_row_shape('PDMS_v1_navtest')` **needed no change**; its evidence got stronger, which is the
outcome a pre-registered scale-up is supposed to have. All three **12,147-row** CSVs carry
**12,146 hex token rows + exactly ONE summary row**, token verbatim **`average`**, `valid = True`,
per-file sha256 banked — the original **20-token** smoke read reproduced at **12,146 tokens on three
independent arms**, and corroborated by the code that writes it
(`run_pdm_score.py:144-147`, independent of token count).

| arm | PDMS ×100 | interval ×100 |
|---|---|---|
| CV | **20.6517** | [19.20, 22.22] |
| STOP | **61.8202** | [60.70, 63.08] |
| HUMAN | **94.5514** | [93.79, 95.23] |

Log-cluster estimator over **136 logs**, **12,146/12,146** successful on each arm, PDMS identity
max |Δ| = **0.0** across 3 × 12,146 rows. ⚠️ Note **HUMAN is defined here** — v1 navtest is single
stage — unlike the v2 two-stage splits where the suite refuses that headline.
