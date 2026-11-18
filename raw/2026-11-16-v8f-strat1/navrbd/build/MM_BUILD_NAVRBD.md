# MM: NAVRBD build record (2026-11-18): the 15-column roundabout channel is built, committed, suite-clean and staged; no arm has trained

- Card: `PREREG_V8F_NAVRBD.md` (frozen sha256 881d7f3d).
- Pre-arm check: RBJK-CARRIES (`rbjk/`).
- Evidence class: MEASURED (ours). Nothing here is a model result.

## Code of record
- **Commit:** `0c3b374fb1036b0ca4ca7620f98137dd6c59cca4`, a child of `e71782a3`, on its own branch `agent/v8f-navrbd-20261118`.
- **Contents:** 16 files (7 modified, 9 new); sha256 list in `b3/TREE_FILES_SHA256.txt`.
- **Why a separate branch:** the refcv9 closure freeze. Nothing under `stack/tanitad/**` lands on arch-inf until K2DC02 trains.
- **What the commit adds:**
  - `--nav-rb-feat state|exitdist`. The default `state` is the N1R path bit for bit (Z0).
  - The exitdist build is step-0 identical (Z1), and each column group gets a gradient while its zero control reads exactly 0.0 (Z2).
  - The loader refuses malformed layouts, and the shuffle derangement has no fixed point.
  - The trainer has a G-DVB entry and width clauses; the checker gains the N1D / N1Dx kinds, the pairing rule and the G-CLOCK-RBD report check.
  - The builder `build_navtext_nav_rbd.py`, and the T1 / T0 study wrappers. The pinned evaluators are byte-untouched.

## Checks (MEASURED)
| check | result | artifact |
|---|---|---|
| builder clock check, 3 tables | PASS; the 1-s-ahead regression arm FAILS on train, eval-141 and 10k | `b2/navtext_nav_rbd*_report.json` |
| held-out counts | 36,951 RB steps; grid 7,594 windows / 545 episodes | same |
| c0 vs the NAVRB2 table | bit-equal (train 29,836 on; eval-141 1,334 on) | same |
| exit / distance columns vs RB-JOINT features | 0 differing windows | `b2/crosscheck_rbjoint.json` |
| mutation arms | each RED on a named test: B1 18 (4 first GREEN, tests added); B2 4; B3 6 | `b1/mutation_results_run*.json`, `b2/mutation_table.json`, `b3/mutation_table_b3.json` |
| integration | checker accepts the real report + npz; a changed sha is refused; the loader loads all real tables; `c13 > 0` with `c14 = 0` occurs 0 times | `b3/proof_integration.json` |
| pod G-SUITE, 0c3b374f vs e71782a3 (same pod) | **0 regressions, 0 failing new tests**; 127 new tests; base 14,994 tests, new 15,121 | `pod/gsuite_result_navrbd.json` |
| pod tree | 16 changed files equal by sha256; 3,425 exported files equal their git blobs | `b4/REPORT_B4.json` |
| binding preflight (CPU half of mint) | no binding, pairing or G-HYG problems for N1D_s0, N1D_s1, N1Dx_s0 | `pod/binding_preflight.json` |
| pairing | each N1D argv equals its N1R argv plus exactly the new flag, npz and sha; two negative controls are refused | `b4/REPORT_B4.json` |

**Failing on both commits.** Two tests fail at e71782a3 as well, and identically: `test_text_encoding_is_explicit` (via `test_r9_e3_make_argv.py:497,502`) and `test_v8f_ego_dagger_compose::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`. They are pre-existing.

## Data
The channel tables are banked on private HF `datasets/Sayood/tanitad-v8f-strat1-runs` as `navrbd_tables/navrbd_tables.tar`. The LFS sha256 equals the local copy; member sha256 values are in `b2/BANKED_TABLES.json`.

## Not yet run
- **Mint:** the GPU half, a `--selfcheck-only` run (G-DVB, G-LIVE) plus a 4-step crash/resume run (G-CKPT, G-EVAL).
- **The arms:** N1D_s0, N1D_s1 and N1Dx_s0, about 0.75 pod-h. Both the mint and the arms wait for the PI's go.
- **Checkpoint-dependent checks:** `navrb_geval_identity.py` on a real N1D checkpoint, and the wrappers on a real eval cache, run after the first arm.
- **Launch line:** in `b4/HANDOFF_B4.md`, step 4.
