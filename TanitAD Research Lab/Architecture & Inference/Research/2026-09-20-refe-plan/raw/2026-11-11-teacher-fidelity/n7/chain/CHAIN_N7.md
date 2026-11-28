# CHAIN_N7: the TF-N7 navtest chain (built and tested, NOT launched; no navtest token was built, scored or read)

Card: `PREREG_TF_N7.md` in this hold (the draft with the MM rulings applied; the MM freezes it and writes no `.sha256` here). Built from the TF-N6 chain (`../n6_tol/chain/`, untouched, and the frozen N6 files are pinned as `n6/` in the run tree). Generator: `../n7_gen/` (untouched; three modules are chain copies, see below). Run tree: copies (never links; NTFS), built by `build_run_tree.sh`. CPU only, dev box, at most 2 processes, detached, marker-based, resumable (`n7run/markers/<stage>.done|.FAIL`, `ZZN7_CHAIN STOP`).

## Arms and the TF-N6 start
| arm | tiers | role | harness scored |
|---|---|---|---|
| **N7-T3** | T0 to T3 | PRIMARY (MM ruling): the TF-N6 tolerance certificate, no human fallback, no T4 | yes, full 12,146 seam |
| **N7-T** | T0 to T4 (T4 = deep slow-down s in {0.7, 0.6, 0.5}, floor 0.5) | SECONDARY, reported beside the primary | yes, full 12,146 seam |
| **N7-X** | T0 to T4, certificate inverted (DAC = 0 required) | deliberate regression: must change 0 tokens, bit-identical to N5-A | no (file identity read, C-REG) |

Start = the banked TF-N5 A1 N5-A plans and seam, read exactly as `n6_nt_build.py` reads them (`inputs/n5a/n5_plans_N5A.npz`, `refe_tfn5_n5a.npz`, `refe_tfn5_n5a.csv`, `n5_build_report.json`). Selection = first tier with a certified candidate, maximum builder PDMS, ties to the earliest generation order (levers in the tier's list order, then iterate / grid index). A token certified at T0 keeps the TF-N6 plan.

## Tree layout (`build_run_tree.sh <dest> [<N6 run tree>]`)
| dir | content |
|---|---|
| `w/`, `repo/`, `n2/`, `n4/`, `n5/` | copies of the TF-N6 run tree (itself byte-identical copies of the pinned TF-N4 A1 tree `F:/tanitad_scratch/n4nt/run_a1` + the TIP `families_run.py`); the A1 tree itself is only read |
| `n6/` | the five PINNED TF-N6 files (`n6_nt_build.py` etc.), byte-identical to the run that produced RESULT_N6; NOT `N6_PINS.json` (it would be an unpinned `.json`) |
| `n7/` | NEW: `n7_nt_build.py`, `n7_nt_chain.py`, `n7_nt_analyze.py`, `n7_make_pins.py`, `test_n7_chain.py`, `fixture_navtrain_tiers.json`, and the generator `n7_lib.py`, `n7_select.py`, `n7_analyze.py` |
| `inputs/` | as TF-N6, plus `inputs/n6t/`: `n6_build_report.json` (the banked N6 report), `refe_tfn6_n6t.csv` (N6-T harness rows, from `harness_csv_n6.tar.xz`), `n6_plans_N6T.npz` (sha256 `1c79202f...`, **fetched from HF by the MM**, see HANDOFF) |

Generator copies: `n7_select.py` and `n7_analyze.py` are byte-identical to `../n7_gen/`; `n7_lib.py` differs in ONE line (line 4: `import n7_env` -> `import n5_env`; `n5_env` already puts the run tree on `sys.path` from `N4_RUN`, whereas `n7_env` hardcodes the F: A1 tree). `n7_run.py`, `n7_env.py`, `n7_analyze` main are not used by the chain.

## Stages and argv
`n7_nt_chain.py --run-dir R --card C --card-sha256 S --pins-sha256 P --ledger L --k 50 --procs 2`. There are no lever flags: the generator parameters and the cascade are pinned code AND hand-written card tables in `n7_nt_build.py`; `reach` reads them back (C-SELECT) so a mutated table, tier list, floor or certificate is refused at the guards.

| stage | what | est. wall (ESTIMATED) |
|---|---|---|
| guards | card sha, ledger `TF-N7:<sha>:E3` n = 50 (TF-N6 n = 49 present, nothing higher), pins sha + every pinned file + REQUIRED pinned + no unpinned .py/.json/.yaml under w/ repo/ n2/ n4/ n5/ n6/ n7/ + metric-cache listing digest + **TIP `families_run.py`** + drive free space + `reach` (card tables, 3 selectors, REPAIR-D + L1, flag audit) | 5 s plus the cache listing |
| reach_real | **every tier T0..T4 executes on a real NAVTRAIN token** (5 tokens, plan sha256 equals the development's, a tier list without the tier no longer chooses at it, N7-T3 / N7-X expectations); fixture `n7/fixture_navtrain_tiers.json` | 4 to 5 min (MEASURED 2026-11-13: smoke/reach_real.json) |
| derive (2 shards) | builder tracked score of the N5-A plan on all 12,146 tokens (C-DACMAP6, `ref`); C-DERIV 200 tokens; on residual tokens: EVERY lever of the cascade (11 levers, 270 scored candidates) run TWICE (C-DET), the three arm selections, the live TF-N6 capture + N6-T selection (C-N6SAME), a separate re-score of each chosen plan | 0.7 to 1.1 h (derive 0.38 CPU-h as TF-N6 + 20 tokens x about 2.2 CPU-min) |
| build | merge: all 15 build controls; plans + seams for N7-T3, N7-T, N7-X; `residual_table` (per token: candidates per tier, blocking class) | 10 min |
| unch | C-UNCH on the files on disk | 5 min |
| score_T3_T | N7-T3 and N7-T FULL 12,146 seams, 2 processes in parallel, through `score_p4_a1.py` (pinned W3 harness); PASS + 12,146 rows. N7-X is not scored | 1.2 to 1.7 h |
| read | `n7_nt_analyze.py read`: controls first (exit 1 before any N7 aggregate), then the bars for N7-T3 (primary) and N7-T (secondary) vs N5-A, the attribution vs N6-T | 2 min |
| families | transp rows (N5A, N7T3, N7T, human) -> famprep -> **rows check** -> families_run `--per-window` (N5A, N7T3, N7T) -> **tokens/logs check** -> fam_ci paired (N5A vs N7T3, N5A vs N7T) -> famread (guard on the primary, the secondary beside it, the changed tokens' speed MAE per token) | 55 to 70 min (one more arm than TF-N6) |

Total ESTIMATE 3.8 to 4.6 h wall. Disk: run tree about 140 MB + outputs about 0.6 GB (three plan + three seam npz, harness score dirs, three families JSONs); `--min-free-gb 5` as TF-N6 (C: about 6 GB free at build time). RAM gate FreeVirtualMemory > 10,240 MB per worker as TF-N5 / N6. Resume: re-issue the same launch; finished stages are skipped, `derive` resumes per token. A `.FAIL` marker must be removed by the MM after the cause is fixed.

## What changed vs TF-N6 (`../n6_tol/chain/`)
| area | TF-N6 | TF-N7 | where |
|---|---|---|---|
| candidates | `Rec` proxy over REPAIR-D (A1 settings) + L1 (TF-N5), cap 5, margin 0.05; the human-log candidate | `n7_lib.run_lever` for 11 levers (REPAIR-D / L1 to cap 20 at margins 0.05 / 0.10 / 0.20, label and target stretch, label offsets, stretch + offset, deep stretch), run twice; **no human candidate anywhere**; the N6 capture is re-run live only for C-N6SAME | `n7_nt_build.py:279-332` (`gen_levers`, `token_row`) |
| arms | N6-H (teacher, then human), N6-T, N6-X | N7-T3 (primary), N7-T, N7-X | `n7_nt_build.py:82-98` (`ARM_SPEC`, `CARD7`) |
| selector | teacher class, then human class; max builder PDMS, ties earliest | **cascade** over tiers (first certified tier, max PDMS, ties earliest in lever-then-index order, cap prefix k <= 5 in T0, T4 floor s >= 0.5) | `n7_nt_build.py:106-155` (`card_cascade`, `make_selector`) |
| card tables | `CARD6` | `LEVERS_CARD`, `TIERS_CARD`, `FLOOR`, `CERT` written by hand; `check_card7` reads `n7_lib.LEVERS`, `n7_select.TIERS`, the grids `n7_select.describe` decodes, `CERT_TOL` and `CARD6["N6T"]` back against them | `n7_nt_build.py:54-81`, `236-277` |
| certificate | `CARD6[arm]` | the same N6-T certificate (imported, and compared three ways: card literal, `CARD6["N6T"]`, `n7_lib.CERT_TOL`, plus behaviour on a 144-point clause grid) | `n7_nt_build.py:236-262` |
| flags | none audited | `audit_flags`: any build flag that is not file / routing plumbing with a disabling default (None, False, 0, "none") must be named in the arm's argv; none exists | `n7_nt_build.py:217-234` |
| mutations (smoke only) | strict, nc_drop, ddc_drop, pdms_drop, human_first, dac_inverted, tie_latest, teacher_off, label_ego_cache | strict, nc_drop, ddc_drop, pdms_drop, dac_inverted, tie_latest, **tier_reverse, floor_off, t4_in_t3, drop_T1, drop_T0**, levers_off | `n7_nt_build.py:94-95`, `128-155` |
| C-CERT-B6 | N6 certificate re-check | **C-CERT-B7** (adds: certificate == CARD6 N6T) | `n7_nt_build.py:667` |
| C-ORDER | N6-H == N6-T off the human class | independent re-derivation of tier order + tie rule; N7-T == N7-T3 where N7-T3 applies; T4 only where N7-T3 took none | `n7_nt_build.py:551-582`, `674` |
| C-N6SAME | (new) | the live N6 pipeline (pinned `B6.capture` + `B6.make_selector("N6T")`) equals the T0 choice on EVERY residual token; the banked N6 report (choices, 6 scores) and the banked N6-T plan file (sha256 from the report, bit equality) agree; read side: N6-T's harness rows == N7-T3 / N7-T rows on its tokens | `n7_nt_build.py:679-742`; `n7_nt_analyze.py:96-104` |
| C-REACH | two real navtrain instances, human path | every tier on a real navtrain token; `tiers_executed` must equal all five tiers | `n7_nt_build.py:333-376`, `759-761` |
| C-SELECT | recompute the card selector on stored candidates | the same, plus the dev `n7_select.cascade` cross-check, plus the card-table readback and the flag audit | `n7_nt_build.py:538-549`, `657` |
| chain | K 49, ledger `TF-N6`, TF-N5 n = 48 present | K **50**, ledger `TF-N7:<sha>:E3`, TF-N6 n = 49 present, `n7/` watched, `n6/` frozen files pinned, `inputs/n6t` required | `n7_nt_chain.py:46-64`, `99-106` |
| scored arms | N6-H, N6-T | N7-T3, N7-T | `n7_nt_chain.py:46` |
| read | bars on N6-H; naming with teacher / human split | bars on N7-T3 (PRIMARY) and N7-T (SECONDARY) vs N5-A, attribution vs N6-T, `rests_on_T4` sentence, per-tier counts, blocking class per remaining zero-DAC token | `n7_nt_analyze.py:45-197` |
| families | N5A + N6H | N5A + N7T3 + N7T; guard on the primary; the secondary beside it; **changed tokens' own speed MAE per token and per tier** from `frames.executed_tracked.block._per_window.longitudinal.speed_mae_mps.value` (UNVERIFIED path against a real `--per-window` JSON: if absent it is reported UNAVAILABLE, never dropped) | `n7_nt_chain.py:247-279`; `n7_nt_analyze.py:199-275` |

## Controls and where each is implemented
Build side (`n7_nt_build.py`): C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET, C-REPLAY, C-DET, C-SELECT, C-CERT-B7, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-N6SAME, C-REACH (lines in the table above; `cmd_merge` 584-806). Read side (`n7_nt_analyze.py`): C-COV, C-BUILD, C-UNCH, C-PIN (+ families TIP), C-CERT, C-ID-C, C-EST, C-REG, C-N6SAME. Each has a deliberate-regression arm in `test_n7_chain.py` (87 tests; the header lists them). Run:
`N4_RUN=<run> CUDA_VISIBLE_DEVICES=-1 python -m pytest -q n7/test_n7_chain.py` (`N7_SKIP_REAL=1` skips the real-token execution test, about 4 min; the dev tarball path is `N7_DEV_TAR`).

Regression arms (each RED): every selector mutation above moves the behavioural signature and is refused by `check_selector`; a mutated `LEVERS` entry, `TIERS` list, `OFF_DY`, `S_DEEP`, `CERT_TOL`, `CARD6["N6T"]` is refused by `check_card7`; a `--floor` flag with default None is found by the flag audit and passes only when named in argv; merge on rows recorded by a mutated selector goes RED at C-SELECT (and C-ORDER / C-CERT-B7 / C-X0 / C-N6SAME where the card names them); non-deterministic generator, levers not run, wrong DAC map, cache ego, missing row, changed residual set, replay violation, `reach` json missing a tier, seam function mismatch, residual count, live N6 selection differs, banked N6 report or plan differs by 1e-9, an N6-T token left out of N7-T's applied set; read side: control exit before any aggregate, N6-T harness row differing, pinned file edited, old `families_run`, wrong draw scheme. Mutation of the **code** (tie `>=`, cap off by one, floor loosened, determinism check disabled, C-N6SAME / C-ORDER / m_recheck check disabled) each turns the suite RED (MEASURED, `SMOKE_N7.md`).

## At the freeze (MM) — exact commands in `HANDOFF.md`
1. `bash build_run_tree.sh C:/Users/Admin/n7nt/run`; fetch `inputs/n6t/n6_plans_N6T.npz` from HF and check its sha256.
2. Copy the frozen card to `<run>/n7/PREREG_TF_N7.md`; its sha256 goes to `--card-sha256`. Append `TF-N7:<sha>:E3` n = 50 to `<run>/inputs/ledger_K.json` (re-read the max n first; the tree's copy holds n up to 49).
3. `python n7/n7_make_pins.py --run-dir R --tip <sha>` (N4_RUN=R, navsim venv): the pins sha goes to `--pins-sha256` (the template pins sha is in `HANDOFF.md`).
4. `n7_nt_chain.py ... --check-only` (guards), then launch with `launch.cmd.template` through WMI. Never `--mutate`: the chain cannot pass it and the build refuses it without `--smoke-allow-mutation`.

## Smoke (navtrain only; nothing on navtest) — `smoke/SMOKE_N7.md`
The 17 real navtrain residual tokens + 3 of the 73 already certified at T0, through the REAL build CLI (`reach --real`, `derive --route navtrain` in 2 shards, `merge`, `unch`), then the selector mutations on the real stored candidates and one real derive with a mutated tier list.
