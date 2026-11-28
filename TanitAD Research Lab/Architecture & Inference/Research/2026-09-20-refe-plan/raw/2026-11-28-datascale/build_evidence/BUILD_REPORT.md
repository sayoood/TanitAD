# DATA-SCALE build report (PREREG_DATASCALE.md 9), 2026-11-28, REFe build agent

Nothing committed, nothing pushed, no `git add`. All files are staged as copies under `E:/Projects/TanitAD-artifacts/datascale/build/repo/<repo path>`. The MM lands them.
Base: tip 01810a19 (parent build fb686b3a). No real data was read; all tests are synthetic.

## Headline (card / integration)
1. **No card change is needed.** `p5c_read.Boot` fits a per-log table by wrapping (see 2 below); it is used unmodified.
2. **Deviation 1 (placement):** the bootstrap helper is in a NEW module `refe/safehead_cvm.py`, not in `safehead_data.py` as card 9 says. `safehead_data.py` is therefore byte-unchanged, so the parents' `CODE.sha256` pins and `test_l2next_read` are untouched. Say if you want it moved.
3. **Integration to do:** land the 16 files in `build/repo/` (CODE.sha256 lists them). The 8 profile-pin tests moved 49 -> 55 profiles (one-line edits each, same pattern as `test_profile_pins.patch`). `launch_gate.py` is NOT patched (profiles come in through `safehead_gate.profiles()`, as for L2-NEXT).
4. **3 stack tests FAIL here for an environment reason, not this change** (`stack/` is untouched): `test_launch_gate` (1 test), `test_launch_gate_pinned_split` (22 errors), `test_launch_gate_safe_directory` (1 test). Each shells `git add` inside a basetemp on E: (exFAT) and git answers `fatal: detected dubious ownership`. A `GIT_CONFIG_*` safe.directory env and a `GIT_CONFIG_GLOBAL` file did not fix it. UNVERIFIED on a drive git trusts; the other 24 stack launch-gate files pass.

## What was built
| # | item | where |
|---|---|---|
| 1 | `--select cv5m` in the trainer; per-log OOF table in `cv_select.json` | `refe/safehead_train.py` + `refe/safehead_cvm.py` |
| 2 | gate: 6 profiles `refe_ds_{sh,ep}_{s0,s1,red}`, G-DVB check of the built rule, 2 mutation arms | `refe/safehead_gate.py` |
| 3 | learning-curve driver M6 + CTRL-PARENT | `refe/safehead_lc.py` |
| 4 | `ctrl_index(index_185, parent_index)` CTRL-INDEX | `refe/safehead_cvm.py` |
| 5 | tests, CODE.sha256 | `refe/test_safehead_ds.py` (+ `test_safehead_ds_nextfp.json`), `raw/2026-11-28-datascale/CODE.sha256` |
| 6 | read side `--lever ds_sh/ds_ep`, CTRL-CV(e) | `raw/2026-11-28-datascale/ds_read.py` + `test_ds_read.py` (`l2next_read.py` NOT edited) |

**cv5m (card 4.3).** Per fold a fresh model trains on the other 4 folds (unchanged `cv_select`); per log and step the table stores pooled tr_pdms, nc, dac, ttc, ep, c x100 and the token count. The decision is computed FROM that table (`safehead_cvm.decide`), so any re-check reproduces it bit for bit. Gain and the 5 sub-score deltas get paired log-cluster CIs from ONE `p5c_read.Boot` (10,000 draws, seed 20260927, 95 % percentile). Eligible = CI-lo(gain) > 0 and no sub-score CI-hi < 0 (NC, DAC, TTC, EP, C). Choice = eligible step with max gain, ties earliest, else 0. The parent's no-margin argmax is also written (`no_margin_argmax_step`). Boot wrapper: each log's mean is expanded to its n tokens (`np.repeat`) and passed to the unchanged `Boot`; equals Boot on raw tokens to 1e-9 (tested). No estimator code in `safehead_cvm.py` (tested by source grep).
**No new trainer flag**, so no new G-DVB entry was needed: `--select` already had one ("built"); G-DVB now also compares the built `cv_rule` with the registered rule (`REGISTERED_RULE`), checks the cv5m fold map, and refuses a cv5m rule in a non-cv5m arm.
**Mutations** (`TANITAD_GATE_ARM`, gate mode only): `ds_no_margin` (eligible on a positive point gain) and `ds_guard_no_ep`. Both turn G-DVB RED on both levers (and change the chosen step on the constructed fixtures, and `ds_no_margin` changes the trainer's choice on the noise world). CTRL-MARGIN: PASS.
**Profiles** carry `suite` = parent suite + `test_safehead_ds.py --fast` (token `ZZSHDS_PASS`). No existing fingerprint moved (pinned: 40 older via the pre-FP file, 9 SAFE-HEAD/L2-NEXT via the new `test_safehead_ds_nextfp.json`, measured on the tip).
**LC driver:** reuses the trainer's `cv_select` on each nested set. Not token-gated (analysis, writes no head); refuses leaks, FEAT mismatch, red/smoke/build-only. "Best-step gain" (max over steps > 0 of the OOF gain point) with strict and non-strict monotone flags is MY definition of the card's phrase; descriptive.

## Tests (synthetic; logs in `build/test_logs/`; every file run whole, commit-free >= 8,000 MB gate, CPU only)
| test | result |
|---|---|
| `test_safehead_ds.py` (full, incl. all arms + e2e mint/verify/train) | ZZSHDS_PASS 20 (`--fast` form 16) |
| `test_ds_read.py` | ZZDSREAD_PASS 4 |
| `test_safehead.py` (full) | ZZSAFEHEAD_PASS 14 |
| `test_safehead_next.py` (full, unchanged) | ZZSHNEXT_PASS 17 |
| `test_launch_gate.py --static --no-banked` (refe) | ZZLG_STATIC_PASS 1180 |
| `test_hnscreen_chain.py` | ZZTEST_PASS 31 |
| pins: div, dose, rep, s1zsm, scorer1p6, vdac | ZZTEST_PASS each |
| pins: field_p7_profile / scorerft_profile | ZZFIELDP7PROF_PASS 528 / ZZSCORERFTPROF_PASS 526 |
| `stack/tests/test_launch_gate*.py` (27 files) | 24 pass fully; 3 environment failures (headline 4) |

CTRL-SH-UNCHANGED: `--select cv5` is bit-identical to the tip trainer (heads, metrics.jsonl, cv_select.json bytes, meta; probe `be809fc6...` measured on the tip, pinned in the test). SAFE-HEAD heldout path is covered by the existing `test_safehead_next` t01 probe (PASS).
Fixtures: +noise step wins under `cv5` and `cv5m` chooses 0 (constructed tables AND the trainer on a label-permuted world: point gain +0.455, CI-lo -0.587); EP alone separated worse blocks a step (and each of NC, DAC, TTC, C alone does); CTRL-PARENT passes on a banked cv5 table and FAILS on one off by 1e-6; CTRL-INDEX passes, and 7 broken indices fail.

## Deviations from the card
1. Helper in `safehead_cvm.py` (above). 2. `cv5m`'s no-margin argmax is also computed from the table (`..._from_table`) next to the trainer's token-level one; the sub-score means in the existing per-step block stay float32 means (kept bit-identical); the table is float64 (agree to 1e-4). 3. CTRL-INDEX compares token, log, split and file sha256 (file paths differ between indices by design) and also refuses a log carrying two splits. 4. ds_read derives N from the sealed index instead of the parent's 2,272 / 35.

## UNVERIFIED
- Full `read` / `confirm` under `--lever ds_sh|ds_ep` (no end-to-end synthetic read fixture; `ds_read.py` tested for the lever table, CTRL-CV and the head-select check only; the report's `card` label still names the L2-NEXT folder).
- The 3 environment test failures on a drive git trusts. Real-data behaviour, runtime of 2,000-step cv5m x 5 folds on ~12k tokens, and of the LC driver (n x 5 fits per lever) are unmeasured. The card is still DRAFT; freeze after the MM lands this.
- CTRL-PARENT against the real banked EPH_s0 `cv_select.json` (step 500, +0.113 on 2,272 tokens): synthetic only.

## Deletions (mine, literal paths)
`datascale/tip` 3.8 GB, `datascale/work` 3.8 GB (overlay copy for tests), `datascale/tmp` 2.9 GB (pytest basetemps, test dirs, a gitconfig), build scratch scripts and `build/base_tmp` 10 MB, `build/env.sh`, `build/sync.sh`. E: hit 0 bytes free mid-run (my copies); now 12.7 GB. Not mine, untouched: `datascale/stage`, `datascale/run1`.
