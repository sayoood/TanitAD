# P6-HN-S implementation, part 3: the trainer wiring (`--hn-bank`, `--ctx-cache`), preflight gates G14 / G15, and the `refe_hnscreen` launch-gate profile

Tip `bcb9a1f`. Prereg `PREREG_P6HN_SCREEN.md` sections 4.2 / 4.3 / 4.4 / 5 (+ amendment B). Checklist items 3 and 4. Dev box, CPU only, no GPU, no pod, no real data. Nothing staged in git, nothing committed.

> **STATUS (read first):** code complete; the static and wiring tests are green; the base-vs-new regression comparison is **pass 1 done, pass 2 (final tree) pending** - see "Regression" below, the rows marked PENDING are filled by the agent that finishes from `HANDOFF.md`. Nothing below is claimed beyond what the logs under `impl3/logs/` show.

## Integration needed (escalated, not buried)

1. **Land the 7 edited and 4 new files under `refe/` and the `impl3/` package** (manifest below). The trainer edit is **layered**: `refe/hnscreen_layer.py::peel` reverses the 9 edits and `carvefeed_layer.peel` calls it first, so every older pin suite that compares the trainer text keeps seeing the pre-HN trainer. `hnscreen_layer.TIP_TRAINER_BLOB` pins the bcb9a1f blob `e357ede0f538caf7f7bf68278c7382127a8eb96f`; a different tip trainer makes the layer (and the wiring test) fail loud.
2. **Reachability is built, but the CALLERS are not part of this package.** `--hn-bank` and `--ctx-cache` exist in the trainer and in the gate; the S0..S4 chain script (checklist item 8) must name both flags in its argv, and a chain test must assert it ("built, tested, unreachable from its caller" rule). The profile `refe_hnscreen` refuses an argv that does not carry them where the arm requires them (arm B) and refuses them on arm A / S0 as specified.
3. **The pod-side facts the gate needs cannot be minted on the dev box**: `fingerprints_expected` (profile hook, default `{}`) must be filled with the C1/C0 content fingerprints by the Master Mind when the bank and cache are built on the pod (see UNVERIFIED 3).
4. **Decisions for the PI/Master Mind** are one-line parameters (AMBIGUITIES 1-12 below).

## What was built

| file (under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`) | state | what |
|---|---|---|
| `refe/scorer_finetune.py` | EDIT (9 edits, tip blob `e357ede0...`) | flags `--hn-bank`, `--ctx-cache` (default `""`); `HNEnv`, `ScreenItems` (dataset wrapper appending cached context and bank rows as a TAIL after the stock 9-tuple; held-out items get context only, so evaluation reads no bank), `hn_refusals` (the 4.4 refusals, rc 4), `hn_head_rows`, `hn_probe_batch`, `g14_hn_live_and_identity`, `g15_ctx_cache_identity`; data events `hn` and `ctx_cache`; with a flag on, `main()` rebinds `fwd`, `train_losses`, `log` in ONE block (the original nested functions stay AST-identical); arm B only: `lB = lB + hn_out.loss` from the masked two-call term `hn_bank.hn_term`; arm A scores through `model.score_trajectories`. G5's 9-tuple unpack became `batch[:9]`. |
| `refe/hnscreen_layer.py` | NEW (generated) | `HNSCREEN_FILE_REVERSALS` (9 pairs), `peel`, `has_layer`, `MARKER`, `TIP_TRAINER_BLOB`. |
| `refe/carvefeed_layer.py` | EDIT | `peel()` first peels the HN layer. |
| `refe/test_hn_screen_wiring.py` | NEW | the real `main()` on CPU with stub model/bank/targets (cache minted by the real `ctx_cache`, bank by the real `hn_bank.mint_bank`): I3 pre-edit-vs-new bit identity (flags absent, plain and lane-head rigs), B == A with the bank absent and empty, cache-only == live trunk bit-for-bit, bank moves B only, the 4.3 gradient table (exact `0.0` lane row, arm A, base), the 4.4 refusals, G14 / G15 raw fields, 7 mutations each RED. `ZZTEST_PASS 120`. |
| `refe/fixtures/scorer_finetune_pre_hn.py.txt` | NEW | the exact tip-blob text of the trainer (sha256 pinned in the test). |
| `refe/fixtures/hn_wiring_evidence.json` | NEW | raw G14 / G15 of the CPU rig for the real trainer and for the mutated trainers M1..M5 (CPU stub evidence, no paths, no hosts); `test_launch_gate.py` H7 re-judges it with `judge_g14` / `judge_g15`. |
| `refe/gate_checks.py` | EDIT | `DVB_KINDS_SCORER += {hn_bank, ctx_cache: data}`; `SCORER_HN_KEYS` / `SCORER_CTX_KEYS`; `judge_g14`, `judge_g15` (re-judged from RAW fields, never from `ok`); `SCORER_GATES += G14, G15`; `scorer_required_gates` adds G14 iff `--hn-bank`, G15 iff `--ctx-cache`; G-DVB rows (10 hn_bank + 6 ctx_cache, only when the flag is named or the block present: R0's `observed_rows` stays 27); regression arms `lever_ignored`, `stale_bank`, `empty_bank_in_launch` in `check_dvb_scorer`; `HN_ARMS` (12 deliberate-regression arms of 4.4) and `apply_hn_arm`. |
| `refe/launch_gate.py` | EDIT | profile `refe_hnscreen` (base argv = `eval/rls1b_arms.json` arm C1 with ONLY the differences 4.2 allows), `hnscreen_refusals`, `hnscreen_regression_argv`, `HNSCREEN_*` constants, G14 / G15 consumption in `consume_scorer_preflight` (judged from raw fields, cross-checked against the data event's `ctx_cache` trunk fingerprint and manifest sha256), CLI `--arm-name` accepts S0..S4. The `refe_rls1*` / aux profiles now FORBID the hn flags (their previously accepted argvs are accepted unchanged). |
| `refe/test_launch_gate.py` | EDIT | the `hnscreen_static()` block (H1..H7: profile structure, clean S0..S2 argvs, S3/S4 need rule E, moved / added / dropped / doubled tokens, trainer refusals, fingerprint hook, CLI, G-DVB rows and arms, the preflight consumer with all 12 regression arms, the judges field by field, the real-evidence fixture check); gate-key loop and unknown-gate fixture updated for the now-known G14 / G15. |
| `refe/test_dump_scores.py` | EDIT | the parser-dest census excludes `hn_bank`, `ctx_cache` and checks their default `""` (one-line enumeration fix; `--dump-scores` itself is unchanged). |
| `refe/test_carve_feed.py` | EDIT | the B8 pins peel the HN layer first (`HL.peel`) because P6-HN-S now sits on top of B8; the parser comparison uses the peeled B8 trainer. |
| `raw/2026-10-18-p6hn-prereg/impl3/*` | NEW | this file, the generators (`gen_layer.py` = the applier, `gen_blocks/`, `gen/patch_*`), the batch drivers, `regression_base_vs_new.log`, all logs. |

### Tip blob sha of every edited file (and the new blob)

| repo path | kind | tip (base) blob | new blob | bytes |
|---|---|---|---|---|
| `refe/scorer_finetune.py` | EDIT | `e357ede0f538caf7f7bf68278c7382127a8eb96f` | `51a281635d7f01fe8c612ffc57ffab5e524c0ab0` | 167666 |
| `refe/carvefeed_layer.py` | EDIT | `97a0a302d540ae2f5a70f564d0bf03568311701b` | `a14450aa1433f82dbadc344f4c3067e0d7122fb8` | 16271 |
| `refe/gate_checks.py` | EDIT | `86981ceddc63b9766a70853be62639f9c0fcfa9d` | `201006beb82677cbdb3d14237771e7824c68a3b0` | 136892 |
| `refe/launch_gate.py` | EDIT | `3a890085e7b780ff1feb1b23010ca9ec4871880f` | `1121cc960fc1f4e7ab7c46d2ab98ed6e6d9937e1` | 104490 |
| `refe/test_launch_gate.py` | EDIT | `63909985b2137b2b87540c9d808384d4f6d18392` | `c664142412abed8b2b2fa5dbcb53379970d68e3d` | 155244 |
| `refe/test_dump_scores.py` | EDIT | `1be47490b60aa5e22d8f525077eb2ff3502dee67` | `bcb70c9285fe4ec98e61dfd5f62227d27ad5195a` | 33383 |
| `refe/test_carve_feed.py` | EDIT | `0442cfd46fdea200b373a9a1336d227e2f36e83a` | `9fdfa406c7e116b04e79ae15a8c15ac0246a17ad` | 40713 |
| `refe/hnscreen_layer.py` | NEW | `-` | `12c67f76f94da81c74c19fb6d44301de5493660c` | 36004 |
| `refe/test_hn_screen_wiring.py` | NEW | `-` | `55171c24e631f38db5a883f7f88ad92684b4bfcc` | 46922 |
| `refe/fixtures/scorer_finetune_pre_hn.py.txt` | NEW | `-` | `e357ede0f538caf7f7bf68278c7382127a8eb96f` | 135387 |
| `refe/fixtures/hn_wiring_evidence.json` | NEW | `-` | `d52c246567883ec204e1f1e9291f06390e6393ad` | 15978 |

## Evidence (all MEASURED on this dev box, CPU, logs under `impl3/logs/`)

- `test_hn_screen_wiring.py`: `ZZTEST_PASS 120`, 7 mutations each RED (`wiring_new_run3.log`; pass 1 batch `test_hn_screen_wiring.py rc 0, 276.6 s`).
- `test_launch_gate.py --static --no-banked`: `ZZLG_STATIC_PASS 926` with the evidence fixture present (920 without it - the six H7 checks are the difference); full `ZZLG_FULL_PASS 971` (pass 1).
- `test_hn_bank.py` ZZTEST_PASS 25 and `test_ctx_cache.py` ZZTEST_PASS 15: unchanged and green on both trees.
- `--dump-scores` is unchanged (test_dump_scores green; the flags are excluded from its dest census only).

## Regression: base (unedited export) vs new (edited tree)

Method: the same test script, one process at a time, CPU, on the unedited export of the tip (`base/`) and on the edited tree (`tree/`).

**Pass 1 found 6 failures on the new tree; four were mine and are fixed in the final tree:**

| test | cause (mine unless stated) | fix |
|---|---|---|
| `test_aux_g12_bf16.py` (3 `h1old/...` checks) | G14's strict-determinism line duplicated the exact text `torch.use_deterministic_algorithms(True, warn_only=False)`; that test mutates G12's copy and requires exactly ONE site | G14 spells it `**{"warn_only": False}` (same behaviour, G12's mutation site stays unique) |
| `test_carve_feed_gate.py` (mutation GM7) | I had edited the pinned text `"--aux-drv-labels", "--epo-carve-logs")` in `RLS1_DATA` | restored it; the HN data flags are appended on the next line |
| `test_rl_r0c.py` (mutation `lg_cli_arm_choices_without_R0c`) | I had edited the pinned text `\| set(RLS1C_ARM_CLASS)) + sorted(AUXS1_ARM_CLASS)` in the CLI `--arm-name` choices | restored it; the S0..S4 names are appended at the end |
| `test_carve_feed.py` (2 pin checks) | the B8 pins compare the CURRENT trainer with the pre-B8 tip; the HN layer now adds 2 flags and 6 functions on top | the test peels the HN layer first (`test_carve_feed.py` edited, 3 lines) |
| `test_rls1_queue.py` ("returned before S1 was over") | CANDIDATE environmental (a wall-clock assertion); base result decides | see the table below |
| `eval/test_aux_navtest_forward.py` | environmental: asserts the head module lives under `D:/Projects/TanitAD`; the export is at `C:/Users/Admin/hn34/` | see the table below |
| `eval/test_hn_screen_read.py` | environmental: `F_adapter_dir_missing` (needs `TANITEVAL_DIR`, the export has no `taniteval/`) | rerun with `TANITEVAL_DIR=D:/Projects/TanitAD/taniteval` on both trees |

**Final per-test table (pass 2, final tree vs unedited export):** PENDING - generated by `work/compare.py` into `impl3/regression_base_vs_new.log`; the finishing agent pastes it here. Requirement: 0 new failures; a test that fails identically on base is environmental and its log lines are cited.

| test | base (unedited export) | new (edited tree) | verdict |
|---|---|---|---|
| `test_hn_bank.py` | rc 0 (18.4 s) ZZTEST_PASS 25  (1 unmutated suite + 24 RED mutations) | rc 0 (18.1 s) ZZTEST_PASS 25  (1 unmutated suite + 24 RED mutations) | pass |
| `test_ctx_cache.py` | rc 0 (8.8 s) ZZTEST_PASS 15  (1 unmutated suite + 14 RED mutations) | rc 0 (7.4 s) ZZTEST_PASS 15  (1 unmutated suite + 14 RED mutations) | pass |
| `test_hn_screen_wiring.py` | not run | rc 0 (61.0 s) ZZTEST_PASS 120 | pass (no base counterpart: a NEW test, or the base run has not reached it) |
| `test_rl_epo.py` | rc 0 (100.5 s) ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | rc 0 (98.7 s) ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | pass |
| `test_rl_safeobj.py` | rc 0 (178.5 s) ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | rc 0 (162.0 s) ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | pass |
| `test_rl_s1b.py` | rc 0 (16.3 s) ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | rc 0 (16.2 s) ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | pass |
| `test_rl_pickcon.py` | rc 0 (22.4 s) ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | rc 0 (22.2 s) ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | pass |
| `test_rl_r0c.py` | rc 0 (69.9 s) ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | rc 0 (62.6 s) ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | pass |
| `test_carve_feed.py` | rc 0 (53.2 s) ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations) | rc 0 (48.2 s) ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations) | pass |
| `test_carve_feed_gate.py` | rc 0 (13.9 s) ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) | rc 0 (15.3 s) ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations) | pass |
| `test_dump_scores.py` | rc 0 (23.4 s) ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mut | rc 0 (23.7 s) ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mut | pass |
| `test_aux_s1_wiring.py` | rc 0 (170.9 s) ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | rc 0 (149.8 s) ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | pass |
| `test_label_dir_refusal.py` | rc 0 (4.6 s) ZZTEST_PASS 6 | rc 0 (3.9 s) ZZTEST_PASS 6 | pass |
| `test_rl_terms_label.py` | rc 0 (6.7 s) ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + cove | rc 0 (6.6 s) ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + cove | pass |
| `test_rls1_chain.py` | rc 0 (139.7 s) ZZTEST_PASS 522 | rc 0 (141.9 s) ZZTEST_PASS 522 | pass |
| `test_rls1_queue.py` | rc 0 (717.1 s) ZZTEST_PASS 254 | rc 0 (717.6 s) ZZTEST_PASS 254 | pass |
| `test_s1aux_chain.py` | rc 0 (52.9 s) ZZTEST_PASS 256 | rc 0 (48.0 s) ZZTEST_PASS 256 | pass |
| `test_s1c.py` | rc 0 (130.1 s) ZZTEST_PASS 220  (2 skipped, not counted) | rc 0 (144.0 s) ZZTEST_PASS 220  (2 skipped, not counted) | pass |
| `test_sft4r_gate.py` | rc 0 (11.9 s) ZZTEST_PASS 153 | rc 0 (11.7 s) ZZTEST_PASS 153 | pass |
| `test_sft4r_chain.py` | rc 0 (2.7 s) ZZTEST_PASS 31 | rc 0 (2.6 s) ZZTEST_PASS 31 | pass |
| `test_gate_suite_prerun.py` | rc 0 (242.2 s) ZZTEST_PASS 144 | rc 0 (249.6 s) ZZTEST_PASS 144 | pass |
| `test_suite_nesting.py` | rc 0 (9.4 s) ZZTEST_PASS 95 | rc 0 (8.4 s) ZZTEST_PASS 102 | pass |
| `test_p2rep_chain.py` | rc 0 (190.7 s) ZZTEST_PASS 353 | rc 0 (157.3 s) ZZTEST_PASS 353 | pass |
| `test_p2r_chain.py` | rc 0 (5.2 s) ZZTEST_PASS 139 | rc 0 (4.9 s) ZZTEST_PASS 139 | pass |
| `test_aux_navtest_forward.py` | rc 1 (1.8 s) | rc 1 (2.3 s) | same failure on base (environmental) - compare tails |
| `test_r0e_diagnostics.py` | rc 0 (27.1 s) ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI en | rc 0 (26.5 s) ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI en | pass |
| `test_safeobj_outcome.py` | rc 0 (757.1 s) ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI | rc 0 (1022.4 s) ZZTEST_PASS 153  (1 unmutated suite of 189 checks + the CLI | pass (env-fixed / long-timeout rerun) |
| `test_hn_screen_read.py` | rc 0 (20.0 s) ZZTEST_PASS 31  (1 unmutated suite (incl. items 5, 6, CLI, p | rc 0 (21.1 s) ZZTEST_PASS 31  (1 unmutated suite (incl. items 5, 6, CLI, p | pass (env-fixed / long-timeout rerun) |
| `test_p6_cpu_probe.py` | rc 0 (116.1 s) ZZTEST_PASS 111 | rc 0 (132.8 s) ZZTEST_PASS 111 | pass |
| `test_p6_overlap.py` | rc 0 (1.4 s) ZZTEST_PASS 54 | rc 0 (1.6 s) ZZTEST_PASS 54 | pass |
| `test_p6_chain.py` | rc 0 (291.3 s) ZZP6CHAIN_PASS 351 | rc 0 (369.7 s) ZZP6CHAIN_PASS 351 | pass |
| `test_p6_levers.py` | rc 0 (281.0 s) ZZP6_PASS 199 | rc 0 (350.3 s) ZZP6_PASS 199 | pass |
| `test_launch_gate.py::--static --no-banked` | rc 0 (39.8 s) ZZLG_STATIC_PASS 677 | rc 0 (47.4 s) ZZLG_STATIC_PASS 926 | pass |
| `test_launch_gate.py` | rc 0 (400.0 s) ZZLG_FULL_PASS 722 | rc 0 (416.8 s) ZZLG_FULL_PASS 971 | pass |
| `test_aux_g12_bf16.py` | rc 0 (17.0 s) ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutat | rc 0 (18.5 s) ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutat | pass (env-fixed / long-timeout rerun) |
| `test_gate_gpu_probes.py` | rc 0 (2.9 s) ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | rc 0 (3.6 s) ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | pass (env-fixed / long-timeout rerun) |

new failures: 0

(MEASURED by `work/compare.py batch_new batch_base`, 2026-10-19 05:57, after pass 3; CPU batches under `CUDA_VISIBLE_DEVICES=-1`, the two CUDA tests under the shared dev-box lock via `work/gpu_locked.py` (compute-type guard, PI-approved 2026-10-19); logs `logs/batch_{new,base}*/`.)

## AMBIGUITIES (each implemented as an explicit named parameter or documented choice)

1. **G14 `fan_call_max_abs_diff == 0` is redefined** to the comparison that matters: the loss-path two-call term against ONE call; the masked-call fan rows against the 64-row call are informational only, because GEMM shapes differ and the last bits differ (1.8e-7 on CPU). On GPU bf16 this is UNVERIFIED.
2. **Strict deterministic mode in G14 mirrors G12** (`CUBLAS_WORKSPACE_CONFIG` via `setdefault` is a late no-op for cuBLAS itself). Prereg 4.1 says "strict deterministic algorithms are on (`scorer_finetune.py:1369`)"; that does not match the source - they are enabled only inside `g12_probe` and the G14 two-step.
3. **G15 compares 64 train + 64 val scenes**; a side with fewer compares all. The floor is the parameter `min_scenes` (`G15_MIN_SCENES = 64`).
4. **G15 micro-batch dependence**: the identity is stated at the run's own micro-batch; another micro-batch is a different GEMM shape.
5. **`fan_fingerprint` contract** (`keys_fingerprint` over the RL-train fan keys + `ckpt_step`) is mine; the prereg names the quantity, not the construction.
6. **Content-fingerprint equality to the C1 / C0 token fingerprints is not checkable offline**: profile hook `fingerprints_expected` (default `{}`), filled on the pod side.
7. **`--b-mode listnet` is refused** with the hn flags - an extension beyond the four refusals 4.4 lists (the term is defined for the pairwise B).
8. **Arms S3 / S4 are non-mintable** (`arms_need_pi`, rule E): the profile mints S0..S2 only.
9. **Image decode is still done in cache mode** (wasted work): avoiding it needs a `train.py` change, out of scope.
10. **The CPU stub rig uses `ManualMHA`** because stock `nn.MultiheadAttention` under `requires_grad` showed 1.8e-7 numerics on the PRE-edit trainer's stock G3; the module term is proven against the real `nn.MultiheadAttention` in `test_hn_bank.py`.
11. **`refe_rls1` / aux profiles now FORBID the hn flags** (their previously accepted argvs are accepted unchanged; the new flags on them are refused). The sft4r profile's data flags share `RLS1_DATA`; its argv is pinned, so unaffected.
12. **`HN_LABEL_VERSION = 5` and `HN_PARTITION_ID`** (= the pinned RL-S1 partition, equal to `rls1_checks.PARTITION["partition_id"]`) are constants in the trainer; a bank or cache minted under another value is refused.

## UNVERIFIED (not measured here)

1. G14 / G15 on the pod's GPU with bf16 autocast (the dev-box evidence is CPU stub evidence).
2. The real bank and the real cache (no real data was touched); the evidence fixture contains CPU stub values only.
3. The C1 / C0 content fingerprints (AMBIGUITIES 6).
4. The `refe_hnscreen` profile has not been launched; the supervisor path (`supervise_run.sh`) is not exercised.
5. Timing: pass-1 wall-clock numbers are for the dev box only.

## Deliverable manifest

- Everything listed above is staged under `C:/Users/Admin/hn34/out/<repo path>`; `C:/Users/Admin/hn34/out/paths.json` lists `[repo path, absolute source path, "raw"]` for every file. **Only copy: the dev box** (`C:/Users/Admin/hn34/out`, plus the edit tree `C:/Users/Admin/hn34/tree` and the unedited export `C:/Users/Admin/hn34/base`). Nothing is in git, on a pod, or in a worktree.
- Provenance (also in `out/`): `impl3/gen_layer.py` (the applier; asserts the tip blob, writes the trainer and the layer), `impl3/gen_blocks/*.py` (the nine inserted blocks), `impl3/gen/patch_*.py` (the gate / launch-gate / test patchers, `patch_fix1.py` = the three stage-2 fixes), `impl3/gen/run_batch.py`, `run_both.sh`, `pass2.sh`, `stage_out.py`, `impl3/logs/*`.
