# READINESS: can we read P7 on navtest the day it finishes, then P5-C's two arms? (audit, 2026-11-12)

Audit of mirror commit `a3cc6a77` (`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`), read from git, plus a CPU-only test run in a scratch export (`CUDA_VISIBLE_DEVICES=-1`, venvs `tanitad`, `navsim-crun`). No GPU lock, no pod, no launch, nothing staged. Evidence class of every line reference: MEASURED by read/grep of that commit. P7 has no checkpoint yet, so every "would run" is UNVERIFIED end to end.

## HEADLINE

**P7 read: NOT READY. One blocker (silent), three gaps, all small.**

1. **BLOCKER (silent wrong number): the P6 read route would feed P7 a yaw of 0.0.** `eval/stage2_sft.py:150-153` launches `refe_navtest_seam.py` directly. The yaw shim (`refe/ego_yaw.py`, `attach_to_planner` / `patch_planner_class`) is applied only by `python refe/ego_yaw.py --table T -- <script>`; `grep -ci yaw` over stage2_sft.py, stage3_full.py, refe_navtest_seam.py, the pinned harness and `planner.py` finds nothing that installs it (planner.py:162, :263 are unrelated comments). The planner fills ego[4] from `EgoState.angular_velocity`, 0.0 on every DB-built state (ego_yaw.py docstring). Nothing refuses: a 4.3-5.3 h seam would finish and score "P7" at a train/eval mismatch. **FIX BUILT here** (below): `--ego-yaw-table` on stage2_sft.py, refusal of a yaw-trained checkpoint without it (before the GPU lock), table sha256 pin `f5ebe24f`. Tests: new 5/5, old gaps 35/35, selftest 11/11.
2. **GAP: no comparator pin for P6.** `stage3_full.py:737` `--base` choices are `u400 | p2 | deployed`. SPEC_P7 §5 comparator C1 = P6. P6's read exists (seam `E:/.../seams/refe_p6final.npz`, csv `score/refe_p6final/`, `F:/TanitAD_p6read_work/stage3_p6_full/p6_full_families.json`, MEASURED present) but is not a `BASELINES` entry. Until pinned, P7 vs P6 is a by-hand paired compare.
3. **GAP: the E: tree (where `PKG` is hard-coded) lacks the yaw and instrument files**: `refe/ego_yaw.py`, `refe/ego_yaw_table.py`, `eval/instruments/*` (+ 97 refe files; read-route files `planner.py`, `ckpt_io.py`, `refe_navtest_seam.py`, `score_navtest_refe.py`, the 21 harness files, `stage2_sft.py` base, `command_adherence.py` md5-equal to the tip, MEASURED). Sync before the read (step 0).
4. **GAP: P7's headway / time-gap / TTC-to-lead and PAIRED family CIs are not on the stage-2/3 route** (Q3).

Nothing else breaks: P7's aux head sits beside `model` in the checkpoint (`train.py:1688-1692`), so the planner's strict load of `model` (`ckpt_io.py:148,170`) is unchanged; `n_score_components` / `nav_mode` come from meta (`ckpt_io.py:126-134`, same as P6).

**P5-C: read instrument is about 25 % built.** Pieces (tracked harness, W-TRANSP with `--metric-cache-root`, W-FAM `--per-window`, `fam_ci.py`) exist; the read driver, the 20-pose pick dump, the paired C1-C0 estimator on both scorers, the verdict, the printed-twice renderer, the sealed-read and K-ledger writers do not. About **1,000 lines of code + 700 lines of tests, 10 items (B3-B12), ESTIMATED 3-4 agent-days**, plus 2 x ~4.3 h GPU seam and ~6 h CPU transparent pass per read (ESTIMATED).

## Q1. The exact command sequence to read P7

Paths: `PKG=E:/Projects/TanitAD/<PK>` (hard-coded in stage2_sft.py:39), `DATA=E:/Projects/TanitAD/data/refe_navtest`, `OUT=E:/Projects/TanitAD-artifacts/p7_final`, `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `PYD=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, `YAW=E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz` (sha256 begins `f5ebe24fd05a8067`, MEASURED; its report: 12,146 / 12,146 tokens, coverage 1.0).
Always: `export TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1` (see Q2: this makes the first-existing-root trap impossible).

| # | step | command / note | breaks? |
|---|---|---|---|
| 0 | sync E: tree to the tip | export `<PK>/eval`, `<PK>/refe`, plus the hold's `files/eval/*` into `E:/Projects/TanitAD/<PK>/` (the MM lands the hold; the tree is stale). Then `$PY eval/harness_pins.py --verify` -> `ok true, n_files 21, variant NT_FIX1` (MEASURED today); `$PY eval/test_p6_read_gaps.py` 35/35; `eval/selftest_stage3_full.py` 11/11; `eval/test_command_adherence.py` 24/24; `eval/test_p7_read_yaw.py` 5/5 | yes until synced: ego_yaw.py absent on E: |
| 1 | fetch | `Sayood/tanitad-refe-p7-reserve-head`, files `run/model_final.pt` + `run/summary.json` (path derived from `p2r_sync.py:224-230`, UNVERIFIED until P7 pushes). `summary.json` `"done": true`; `sha256(model_final.pt) == model_final_sha256` | runbook still names P6's repo |
| 2 | smoke (navtrain tokens, <= 12) | the runbook smoke (`p6_navtrain_plan.py`, `p6dry_common.py`) hard-codes `D:/` and needs the NAVTRAIN yaw table (`yawrate_wp/tables/ego_yaw_navtrain_devbox.json.gz`, present). Needs a shim hop it does not have. **Skip it; the real read's first 12 tokens + the 36-token frame control are the smoke**, or run `ego_yaw.py --table <navtrain table> -- <smoke script>` after `subst D: E:\` | yes (D:, no shim) |
| 3 | the read | `$PY eval/stage2_sft.py --ckpt $CK --name p7_full --rule navsim_v1 --tokens $DATA/navtest_full_tokens.json --dump-proposals --ego-yaw-table $YAW`; run twice (attempt 1 ends `INCOMPLETE (seam failed)` at the 36-token frame control, EXPECTED, `RESULT_SFT4.md` Deviation 1; attempt 2 scores). Same detached form as runbook step 3. P6's real read took seam 15,638 s (`CHAIN.log`) + score 24 min | **blocker 1 fixed by `--ego-yaw-table`**. Also: if `seams/stage2_p7_full.npz` already exists, stage2 reuses it (`stage2_sft.py:145`) with no yaw record: never reuse a seam of unknown yaw |
| 4 | stage 3 | `$PY eval/stage3_full.py --name p7_full --ckpt $CK --expect-ckpt-sha256 $SHA --base p6 --also p2 --expect-rule navsim_v1` | **`p6` not a choice (gap 2).** Today: `--base p2 --also u400` works and gives C2; P7 vs P6 then needs build item B2. Stage 3's GO/NO-GO words are not quoted; apply SPEC_P7 §5 by hand |
| 5 | tactical | `$PY eval/command_adherence.py read --name p7_full --dump $DATA/seams/stage2_p7_full.proposals.npz --base <p6 seam npz path> --also p2` (accepts a seam path, `command_adherence.py:572,589`, so no pin needed) | no |
| 6 | selection quality | runbook step 5 `selection_quality.py score|table|read`, 4-6.4 h CPU (optional, report-only) | no |
| 7 | determinism | runbook step 6 (`--name p7_full_rep`, WITHOUT `--dump-proposals`, WITH `--ego-yaw-table`), per-token equality 12146 True | no, if the flag is passed |
| 8 | families with CI and distance keeping | see Q3 (new route) | gap 4 |
| 9 | A7 | default ON in the seam (`--no-repair-last-heading` absent); stage 3 expects it; `navsim_v1` selection on the arm's own scorer logits is `--rule navsim_v1` (planner `rule`) | no |

Steps that would NOT break on P7's specifics: the aux head (beside `model`), `nav_mode command`, `--yaw-wrap off` (training-side), the `navsim_v1` rule, A7, the command table (`set_driving_command`, `refe_navtest_seam.py` loop), the harness pin guard.
What I could not test: a real yaw-trained checkpoint through the seam (none exists). The shim's own test (`refe/test_ego_yaw.py`) is the evidence; `EgoYawShim.apply` raises on an unknown token, an un-set token, a scale-less checkpoint and an incoming non-zero ego[4].

## Q2. Hardcoded roots on the read route (the archive is `F:/Archive/devbox-C/navsim`)

State today (MEASURED 2026-11-12): `E:/Archive/devbox-C` exists and is EMPTY (created 15:29 today); `E:/Archive/devbox-C/navsim` does not exist. The resolvers test `isdir(<root>/navsim)` (`stage2_sft.py:53-59`, `stage3_full.py:66`, `selection_quality.py:47`, `harness/score_e.py:13-20`, `harness/w3e/code/run_v1.py:39-45`), so today they pick F:. The trap is armed anyway: anything that creates `E:/Archive/devbox-C/navsim` flips every one of them to a root that lacks `navsim-3e8291…/…/navtest.yaml`. Do not create anything under `E:/Archive`. **Mitigation, zero edits: export `TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim`** (all five resolvers honour it first; stage2's subprocess env copies `os.environ`, `eval_checkpoint.py:40`).

| file:line | path | on read route? | status |
|---|---|---|---|
| `stage2_sft.py:39,49,152` | `E:/Projects/TanitAD/...` PKG, data, db-dir, nuplan splits | yes | E: is canonical for data; the tree is stale (step 0) |
| `stage2_sft.py:150` | `D:/` -> `E:/` replace on `env_driverl()`: backbones, nuplan root, maps | yes | works (E: dirs exist, MEASURED) |
| `eval_checkpoint.py:32-36,41-47` | `C:/.../ev6`, `C:/.../dz`, `D:/` constants | yes (env only) | C: dirs exist; D: replaced above |
| `refe_navtest_seam.py:38,39,145` | `D:/` defaults | yes | stage2 passes `--frames --db-dir --export` explicitly: harmless |
| `score_navtest_refe.py:30,32` | `D:/` W3_CODE, W3_EXPORT | yes | `score_e.py:23` overrides EXPORT; W3_CODE repointed by the banked harness (P6 read ran 2026-11-09 on F: root) |
| `harness/score_e.py:7`, `ev6e/families6.py:34-35` | `E:/Projects/TanitAD/...` PKG, `taniteval/adapters/navsim.py` | yes | E: `taniteval` adapter differs from tip but tip is default-identical (`per_window` opt-in): leave E: adapter, MEASURED |
| `harness/w3e/code/run_v1.py:297`, `write_cache_metadata.py:95` | `E:/Projects/TanitAD/FlyWheels/...` | yes | exists on E: |
| `harness/w3e/e1/navsim_win.py:205-206` | `E:/Archive/devbox-C/navsim/{devkit,nuplan-devkit}` `git_sha` | yes (provenance only) | path absent; UNVERIFIED whether it tolerates that (P6's read passed through it on 2026-11-09, so it does) |
| `stage3_full.py:62,77,86` | `E:/...` data, S2_RAW; `D:/...` families6 fallback (3rd) | yes | fine; banked copy is first |
| `command_adherence.py:51,66` | `E:/` data; REPO from `HERE.parents[4]` | yes | fine on the E: tree |
| `transp_score.py:37,44` | `E:/...` maps, db-root | new route | fine |
| `families_run.py:31` | DEFAULT_REPO = 6 levels up from `eval/instruments` | new route | correct on E: tree; E: has NO `taniteval/taniteval/per_window.py` (MEASURED) -> pass `--tanitad-repo <tip export>` for `--per-window` |
| `aux_navtest_forward.py:58-65` | 5 `D:/` constants | head endpoints only | see Q3 |
| `proxy_eval.py:50-54,233,777` | `D:/` | imported by stage3 for `boot` only | constants unused |
| runbook step 2 | `p6_navtrain_plan.py`, `p6dry_common.py` `D:/` | smoke | skip or `subst` |
| about 20 files under `harness/w3e/` | `E:/Archive` | not on route | fail loudly if run |

G: paths: none found on the route (grep of 22 files).

## Q3. P7's four families on the read route

| family | produced by step 3-5? | CI | missing |
|---|---|---|---|
| LONGITUDINAL (speed, along-track, accel, EP, TTC) | `families6.py` via stage 2 (`p7_full_families.json`) | level CI per metric (log cluster); **stage 3's cross-arm comparison is an UNPAIRED non-overlap** (`stage3_full.py:313-314`, "conservative reading") | headway / time-gap / TTC to the lead: the adapter documents `lead_block` as UNAVAILABLE; only `transp_score.lead_track` builds it |
| LATERAL (heading, cross-track, curvature, yaw rate, DAC, DDC) | families6 + harness sub-scores | same | paired CI |
| TACTICAL (decision accuracy / kappa; command adherence; selection quality) | families6 + `command_adherence.py` (paired vs C1, C2, estimator `proxy_eval.boot`) + `selection_quality.py` | adherence: yes, paired | tactical goal-setting: not computable (stated) |
| STRATEGIC | PAUSED (PI 2026-09-19), print "PAUSED" | n/a | n/a |

SPEC_P7 §5 (ADOPT = lo > 0 and the adverse list empty) is fully readable from stage 3's `families_vs_base.adverse` + `r2_result.json`: no new instrument is needed for the verdict. The paired family CIs and distance keeping need the new route: seam npz `poses` [N,8,3] (already the executed A7 plan in navsim frame) -> `plans8` npz (needs `tokens`, `plans8`, `--token-log`) -> `transp_score.py --split navtest --metric-cache-root F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest --plans P7=.. P6=.. [--human-arm]` -> `families_run.py --per-window --tanitad-repo <tip export>` -> `fam_ci.py --a fam_P6.json --b fam_P7.json --tokens ...`. Cost ESTIMATED 3.4 s/token without the cache flag (INSTRUMENTS.md) = 11.5 CPU-h, ~5.7 h wall on 2 processes, one pass for all arms. Tests MEASURED here: `fam_ci` + `fam_ci_pertoken` 23/23 OK (navsim-crun venv). Missing: the 12,146-token model-arm identity control (tracked column == harness csv, bar 1e-6): only 60 human-arm tokens were checked (`INSTRUMENTS.md` UPDATE 2026-11-12).
**Head endpoints (DAC-zero share vs P6: from the csv, available; head IoU [0,40) m, head-vs-scorer DAC AUC): NOT readable.** `aux_navtest_forward.py` loads only a standalone `aux_drv/1` file (`:234-241`), defaults to a u400 checkpoint and the C3 tokens, and P7's head is a state dict inside `model_final.pt` (`aux_drv`, `aux_drv_meta`). The aux label store refuses navtest by design (`aux_label_store.py:339`), so navtest drivable GT does not exist. Both are reported-not-gating; ESTIMATED 150 lines, and the navtest GT source is an open question for the MM.

## Q5. Tests run (CPU, scratch export of `a3cc6a77` + taniteval + `stack/tanitad` + the two raw dirs G4 reads)

| test | result |
|---|---|
| `harness_pins.py --verify` | `ok true`, `n_files 21`, caller variant `NT_FIX1` |
| `test_p6_read_gaps.py` (tip, full export) | **35/35 PASS** (first run in a thinner export: 25/35, all 10 failures were missing export files; not a defect) |
| `selftest_stage3_full.py` | **11/11 PASS** |
| `test_command_adherence.py` | **24/24 PASS** (4/24 in the thin export: needs `taniteval`) |
| `fam_ci` + `fam_ci_pertoken` (unittest, navsim-crun) | **23 OK** |
| AFTER the fix: `test_p6_read_gaps.py` | **35/35 PASS** (its "seam command = the OLD command exactly" check still passes) |
| AFTER the fix: `selftest_stage3_full.py` | **11/11 PASS** (pin moved) |
| NEW `test_p7_read_yaw.py` | **5/5 PASS** incl. a RED arm (guard dropped -> the yaw checkpoint runs) |
Not run: `stage3_full.py --selfcheck` (~5 min, reads banked tables; unaffected). **Smoke through the harness skipped:** it needs the GPU lock (not allowed), a P7 or P6 checkpoint on E: (`p6_final` absent; P6's is `F:/TanitAD_p6read_work/ckpt`), and the runbook smoke is D:-bound.
Logs: `logs/` beside this file.

## The fix in this hold (stage 2 yaw)

`files/eval/stage2_sft.py` (+ `stage2_sft.diff`, 53 lines): flag `--ego-yaw-table` (default None = the seam command is EXACTLY the old one); a checkpoint whose meta says `ego_yaw` and no table -> `SystemExit` before the lock and the seam; table sha256 must begin `f5ebe24f`; with the flag the seam runs as `PYD refe/ego_yaw.py --table T -- refe_navtest_seam.py <same args>`; `result.json` records `ego_yaw.{table,sha256,checkpoint_wants_yaw}`. `files/eval/stage3_full.py` (+ diff, 1 line): `STAGE2_SFT_MD5` 74c37dd2... -> 3764778748d6ecc02c6569baffa4f758. `files/eval/test_p7_read_yaw.py` (new). Land order: all three together (the pin and the file move as a pair, like P6's layout fix); then step 0's tests. Not covered: the shim is installed in-process, so the seam report does not carry the shim's `n_calls` (add to the report when B3 touches the seam).

## P5-C build list (model: VDAC `a5_draft/READINESS.md`)

Exists (MEASURED): tracked harness (stage 2); W-TRANSP `transp_score.py` with `--metric-cache-root` (harness-identical on 60 human tokens, 0 of 60 over 1e-6, mutation RED); E2's `transparent_states` reused (bit-equal on navtrain); W-FAM `families_run.py --per-window` (frames `executed_tracked` primary, `executed_transparent`, `reference`); `fam_ci.py` paired family CI (23 tests); the yaw launcher (this hold); `command_adherence.py`, `selection_quality.py`.

| # | item | where | ~lines (code / test) | note |
|---|---|---|---|---|
| B1 | stage 2 yaw flag | `eval/stage2_sft.py` | DONE here (35 / 90) | B-P7 arms need it |
| B2 | `p6` entry in `BASELINES` (csv/seam/families md5 of `refe_p6final`, `p6_full_families.json`) | `stage3_full.py` | 25 / 20 | also needed for P7 vs P6 |
| B3 | **20-pose pick dump**: add `pick20` [N,20,3] (REFe frame, post-A7) to the proposals npz. The dump holds 8-pose proposals only (`refe_navtest_seam.py:354-356`); PREREG §5 and `transp_score` want the 20-pose plan (transparent from `plans8` is the dt 0.5 form, ~1.5 cm coarser: not the registered form) | `refe_navtest_seam.py` | 12 / 50 | control: `to_navsim(pick20) == seam.poses` bit-for-bit |
| B4 | detached read chain for P7 and for each P5-C arm: preflight (ckpt sha, yaw table sha, tokens md5 `3a8b3653`), seam, score, stage 3, adherence; markers; `TANITAD_NAVSIM_ROOT` pinned. P6's real chain is `F:/TanitAD_p6read_work/chain_p6.py`, **a single copy outside the repo, bound to P6's sha and paths** | `PK/eval/` | 150 / 60 | bank it |
| B5 | seam/dump -> plans npz converter, one `transp_score` pass over {C1, C0, base, human} | new `p5c_read_prep.py` | 70 / 40 | all arms in ONE pass (cost is arm-independent) |
| B6 | model-arm identity control at scale: tracked PDMS column == harness csv, 12,146 tokens, bar 1e-6, RED arm (one routing line deleted) | in B5 | 45 / 40 | PREREG §10.5 W-TRANSP (i) |
| B7 | `p5c_read.py`: per-token arrays, SHARED draws (10,000, seed 20260927, log clusters), d, d_T, TD = d - d_T with CIs; B = 0.7031; P1-a, P1-b; G1 (five tracked sub-scores, adverse = CI-hi < 0); G2 (LONG/LAT components of the **tracked-executed** frame, via `fam_ci`; transparent frame never a guard); CTRL-1 (C0 - base); P1-c with C0r; deploy clause; goal 94.55 class; verdict table (§6.2) and qualifier TM / TM+C / TI / TU (§7); sub-scores per scorer, DAC / PDMS / NC-zero rates (ratio estimator); tracker-made / tracker-saved DAC zeros | new | 380 / 280 | each clause needs a known-value test and a mutation arm |
| B8 | **printed-twice renderer** (§7): refuses to print a number without its tracked AND transparent twin or the label "no tracker in this number"; refuses the banned wording ("drives better", "safer", "better driving", synonyms) in any rendered text; one qualifier per verdict | new `p5c_render.py` | 130 / 90 | RED arms: drop the transparent twin, insert a banned phrase |
| B9 | families on both frames for C1, C0, base + paired C1 - C0 via `fam_ci` | wrapper | 45 / 30 | transparent families reported, not guards |
| B10 | determinism subset pass: 1,215 tokens, seed 20261112, second seam pass bit-identical per token under both scorers | new | 60 / 30 | +~0.5 h GPU per arm |
| B11 | `SEALED.json` writer (both `model_final.pt` sha256 before either navtest pass), "no per-arm number before both complete" gate, `ledger_K.json` entries (K 41 / 42; `ledger_K.json` exists) | new | 70 / 40 | §11 |
| B12 | tactical rows for C1 / C0 (adherence + selection quality) | existing tools | 0 | pass `--base <P7 seam>` |

Total ~1,000 lines of code, ~700 of tests, 10 open items (B2-B11). Compute per read (ESTIMATED): GPU seam ~4.3 h per arm (P6 measured 15,638 s) + subset 0.5 h; CPU transparent+tracked pass ~5.7 h wall on 2 processes (INSTRUMENTS.md), fewer with the cache flag; selection-quality 4-6 h optional. The GPU part is serial per arm (shared dev-box lock).
Order that costs least: B2, B3 first (they also serve P7), B4, then B5-B8. B7/B8 can be built on P6-vs-P7 data once P7 reads, as a dry run.

## Risks

1. **Silent yaw 0.0** if anyone runs the unpatched stage2 on P7 (blocker 1). Also true for P5-C's B-P7 arms. A seam npz of unknown yaw must never be reused.
2. **E: tree staleness**: `PKG` is a hard-coded E: path in stage2/3, so the read runs E: code. Verify with step 0, not by trusting that "the tip is landed".
3. **First-existing-root trap** armed by the empty `E:/Archive/devbox-C`. Pin `TANITAD_NAVSIM_ROOT`; never create under `E:/Archive`.
4. **P6's chain is not in the repo** (`F:/TanitAD_p6read_work/chain_p6.py`, single copy, F: external drive).
5. **Stage 3's family comparison is unpaired and conservative**; a P7 verdict that rests on "no adverse family" is a log-draw statement only; the training-seed variance is unmeasured (`H-ESTIM-SEED-1`): a separated CI is necessary, not sufficient.
6. **E2 caveat**: transparent comfort depends on how states are derived; never quoted as driving.
7. The seam's 36-token frame control fails attempt 1 by design; a reader who treats `INCOMPLETE (seam failed)` as failure loses 4+ h.
8. HF path and `summary.json` field names for P7 are code-derived, UNVERIFIED until P7 pushes.
9. P7 ends ~11-14/15 (pod); read cost on the dev box ~6-7 h to the verdict (seam 4.3 h, score 0.4 h, stage 3 / adherence minutes), plus the optional columns. P7's verdict therefore lands ~1 day after the pod finishes if the stage-2 fix is landed first.

## Deliverable manifest (all single-copy until landed; hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/read_readiness_p7_p5c/`)

| artifact | location |
|---|---|
| this audit | `READINESS_P7_P5C.md` (hold) |
| patched `stage2_sft.py`, `stage3_full.py`, new `test_p7_read_yaw.py` (PK-relative `eval/`) | `files/eval/` (hold) |
| diffs vs `a3cc6a77` | `stage2_sft.diff`, `stage3_full.diff` (hold) |
| baseline test logs (tip export) | `logs/` (hold) |
| scratch export | `F:/tanitad_scratch/rr_p7p5c/` (deleted at the end of this job, see handoff) |
