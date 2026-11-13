# CHAIN_N6: the TF-N6 navtest chain (built and tested, NOT launched, no navtest N6 arm scored)

Card: `../PREREG_TF_N6_DRAFT.md` (draft; the MM freezes). Built from the TF-N5 chain (`../../n5_l1/chain/`, untouched). Run tree: `C:/Users/Admin/n6nt/run` (129 MB, copies, never links; NTFS). CPU only, dev box, at most 2 processes, detached, marker-based, resumable (`n6run/markers/<stage>.done|.FAIL`, `ZZN6_CHAIN STOP`).

## Tree layout (copies of the pinned TF-N4 A1 tree `F:/tanitad_scratch/n4nt/run_a1`, which was not modified, plus the banked TF-N5 A1 outputs from HF)
| dir | content |
|---|---|
| `w/`, `repo/`, `n2/` | A1's harness package, taniteval repo, TF-N1/N2 analyzers, byte-identical, EXCEPT `w/pk/eval/instruments/families_run.py` = the TIP version (sha256 `44c62807f125...`, copied from `../../n3_dose/raw/instr/families_run.py`, hash checked) |
| `n4/` | `n4_geom.py`, `n4_repair.py`, `n4_nt_build.py`, `n4_common.py`, `fam_ci.py` (pinned A1 files the chain imports) |
| `n5/` | the PINNED TF-N5 files, byte-identical to the hold's: `n4_repair_l1.py`, `n5_env.py`, `test_n4_repair_l1.py`, `fixture_real_instances.json`, `n5_nt_build.py`, `n5_nt_analyze.py`, `n5_nt_chain.py` (N6 imports their helpers; nothing copied into n6) |
| `n6/` | NEW: `n6_nt_chain.py`, `n6_nt_build.py`, `n6_nt_analyze.py`, `n6_make_pins.py`, `test_n6_chain.py`, `N6_PINS.json` |
| `inputs/` | A1's inputs, `inputs/n4d/` (banked N4-D plans, csv; secondary contrast only), **`inputs/n5a/`**: `n5_plans_N5A.npz` (9da77557), `refe_tfn5_n5a.npz` (2a7b310a), `refe_tfn5_n5a.csv` (5e0820e0), `n5_build_report.json` (HF copy, a7268635), `replay_expect_navtest.json` (NEW, small: both repairs return no plan on the 20), `n5_result.json`, `residual25.json` |

The N5-A files were read from the HF bank (`Sayood/tanitad-refe-scorer1p6`, `evidence/2026-11-13-tf-n5-a1/n5a1_run.tar`) by seeking inside the tar (no 447 MB download); the plan and seam sha256 equal the hashes in `n5_build_report.json`.

## Stages and argv
`n6_nt_chain.py --run-dir R --card C --card-sha256 S --pins-sha256 P --ledger L --k 49 --procs 2`. There are no L1 flags: the repair objects and the selectors are pinned code, and `reach` reads them back and compares their behavioural signature with the card, so a mutated selector or repair is refused at the guards.

| stage | what | est. wall (ESTIMATED from TF-N5 A1) |
|---|---|---|
| guards | card sha, ledger `TF-N6:<sha>:E3` n = 49 (TF-N5 A1 n = 48 present, nothing higher), pins sha + every pinned file + REQUIRED pinned + no unpinned .py/.json/.yaml under w/ repo/ n2/ n4/ n5/ n6/ + metric-cache listing digest + **TIP `families_run.py`** + run drive free space + `reach` (REPAIR-D, L1, 3 selectors) | 5 s plus the cache listing |
| reach_real | the two real navtrain fixtures: capture, deterministic, selection, N6-X none, the human candidate builds and passes | 65 s |
| derive (2 shards) | builder tracked score of the N5-A plan on all 12,146 tokens (C-DACMAP6, `ref`); C-DERIV 200 tokens; on residual tokens: both repairs recorded twice (C-DET), the human candidate, the three selectors, a separate re-score of each chosen plan | 0.7 to 1.0 h |
| build | merge: all build controls; plans + seams for N6-H, N6-T, N6-X; `residual_table` (per token: candidates, clause failures, blocking class) | 10 min |
| unch | C-UNCH on the files on disk | 5 min |
| score_H_T | N6-H and N6-T FULL 12,146 seams, 2 processes in parallel, through `score_p4_a1.py` (pinned W3 harness); PASS + 12,146 rows. **N6-X is not scored** (bit-identical to N5-A by C-X0 / C-UNCH; the read re-checks it, C-REG) | 1.2 to 1.7 h |
| read | `n6_nt_analyze.py read`: controls first (exit 1 before any N6-H aggregate), then bars 1 to 3 | 2 min |
| families | transp rows (N5A, N6H, human) -> famprep -> **rows check** -> families_run `--per-window` (N5A, N6H) -> **tokens/logs check** -> fam_ci paired (A = N5A, B = N6H) -> famread (bar 4) | 40 to 50 min |

RAM: each worker is gated on FreeVirtualMemory > 10,240 MB (as N5). **Disk:** the run tree is on C: (NTFS, about 6.5 GB free at build time). The chain's own output is about 0.4 GB, but its default `--min-free-gb 20` would WAIT on C:, so the launch template passes `--min-free-gb 5`; alternatively copy the tree to F: when it has >= 21 GB (a launch decision for the MM).
Resume: re-issue the same launch; finished stages are skipped, `derive` resumes per token. A `.FAIL` marker must be removed by the MM after the cause is fixed.

## What the build does that TF-N5's did not
- **Candidate capture without copying a loop.** `Rec` is a recording proxy of the scorer context: every `cx.score(plans)` of `n4_repair.repair_target` (REPAIR-D) and `n4_repair_l1.repair_l1` is logged; the repairs run unchanged. Test: the unproxied L1 plan is among the candidates; iterate 0 is the base label bit for bit.
- **Selector read back + behaviour.** `check_selector` compares the built selector's spec with the card, and its choices on 13 synthetic candidate sets with the card selector's. Every clause moves a signature (`test_every_clause_and_the_ordering_reaches_the_selection`).
- **C-SELECT** re-runs the CARD selector in the merge on the candidates stored in each row and requires the recorded choice to be identical: a derive that ran a different selector goes RED even if every other number looks plausible.
- The human candidate is built from the nuPlan DB ego and scored by the same builder scorer; if the DB has no 4 s future the candidate is absent and the reason is recorded (`n_human_unavailable`), never an error row.

## Controls and where each is implemented
Build side (`n6_nt_build.py`): C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO (+ human ego), C-SET, C-REPLAY, C-DET, C-SELECT, C-CERT-B6, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-REACH. Read side (`n6_nt_analyze.py`): C-COV, C-BUILD, C-UNCH, C-PIN (+ families TIP), C-CERT, C-ID-C, C-EST, C-REG. Mutation arms and their tests: `test_n6_chain.py` (header lists them). Test count: see `test_run_n6_chain.log` (69 chain tests passed, run with `CUDA_VISIBLE_DEVICES=-1`; the 8 pinned L1 tests also pass).
Required mutations, each RED: tolerance off (strict TTC/C), NC guard dropped, DDC guard dropped, builder-PDMS clause dropped, human-first ordering, DAC requirement inverted (H, and X turned to require DAC = 1), tie rule latest. The merge reads each npz once (`test_npz_read_once_not_per_token`, with its per-token mutation arm).

## At the freeze (MM)
1. Copy the frozen card to `<run>/n6/PREREG_TF_N6.md`; its sha256 goes to `--card-sha256`.
2. Copy the banked ledger `../../n5_l1/ledger_K.json` (holds n up to 48) to `<run>/inputs/ledger_K.json` and append the entry `TF-N6:<sha>:E3` n = 49 (re-read the max first).
3. If ANY pinned file changed since this hand-off, re-run `python n6/n6_make_pins.py --run-dir R --tip <sha>` (N4_RUN=R, navsim venv) and use the new pins sha (the template pins sha is in `launch.cmd.template`).
4. `n6_nt_chain.py ... --check-only` (guards), then launch with `launch.cmd.template` through WMI.
Never `--mutate`: the chain cannot pass it and the build refuses it without `--smoke-allow-mutation`.

## Smoke (navtrain only; no navtest N6 arm was scored)
`smoke/`: 7 real navtrain DAC-0 instances (the TF-N5 smoke's DEV30 set) through the REAL build CLI (`reach --real`, `derive --route navtrain`, `merge`, `unch`); inputs by `mk_smoke_inputs.py` (the stand-in "N5-A" is the base label, so all 7 are residual). See `smoke/SMOKE_N6.md` for the numbers and the mutation runs.
