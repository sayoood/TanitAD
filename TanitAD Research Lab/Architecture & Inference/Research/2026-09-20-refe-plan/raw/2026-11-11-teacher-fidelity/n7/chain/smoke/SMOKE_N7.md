# SMOKE_N7: navtrain only, through the real build CLI (no navtest N7 build or score)

Evidence class: **MEASURED** (builder scorer = W-TRANSP nuPlan-DB rebuild, LQR tracker, tier T1 builder-side), dev box 2026-11-13, CPU only, 2 processes, `CUDA_VISIBLE_DEVICES=-1`. Artifacts in this folder.

Inputs (`mk_smoke_inputs7.py`): **20 real navtrain tokens** = the 17 residual tokens of `rj5_navtrain/repair_nt/navtrain_residual17.json` + the first 3 (sorted) of the 73 already certified at T0 (`03ab084510af550e`, `0cc59130bd945c39`, `1628652c6dcb5150`). Stand-ins: the RJ5 target rows are `nt_rolls.jsonl`; the "N2-R" and "N5-A" plans are the banked P5-R label of each token (`nt_plans.npz`), so all 20 are builder DAC 0 = residual; the "harness csv" is the development's builder base rows (`ref`); the replay expectation is empty (no banked repair result on navtrain). The N6 report and the N6-T plan file do not exist for navtrain, so `--n6-report` / `--n6t-plans` are not passed and C-N6SAME reads its **live** layer only (see "not exercised").

## What ran
| run | wall | verdict |
|---|---|---|
| `reach --real` (5 real tokens, one per tier) | about 3 min | **PASS**: `reach_real.json`: tiers executed `T0_N6, T1_L2, T2_STR, T3_OFF, T4_DEEP`; on each the chosen tier / lever / k / parameter equals the development's, the chosen plan's sha256 equals the development's plan (bit identical), a tier list without that tier no longer chooses at it, N7-T3 equals N7-T except on the T4 token (nothing), N7-X chooses nothing; T0 token deterministic |
| `derive --route navtrain`, 2 shards, 20 tokens, every lever twice, live N6 | 1076 s and 1009 s (9 and 11 tokens): **mean 102 s, max 159 s per residual token** (builder `timing.cpu_h_sum` 0.569 CPU-h) | rows `derive_clean_s*of2.jsonl` |
| `merge` | seconds | **PASS 15 of 15 controls** (`n7_build_report_clean.json`): C-COVB, C-DACMAP6, C-DERIV (max dev 0.0 m), C-LABELEGO, C-SET, C-REPLAY, C-DET, C-SELECT, C-CERT-B7, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-N6SAME, C-REACH |
| `unch` | seconds | **PASS** (`unch_clean.json`): N7-T3 16 changed (plan and seam), N7-T 19, N7-X 0 and bit-identical |

## Result of the clean run (20 tokens)
- **The build reproduces the development's selection on 20 of 20 tokens** (tier, lever, k; the cascade over the stored candidates is bit-for-bit what `n7_gen` selected): N7-T3 certifies **16** (T0 5, T1 7, T2 2, T3 2), N7-T certifies **19** (+ T4 3), N7-X 0. The unrescued token is `229a59137a705430` (`blocking_class`: no candidate in any tier reaches DAC = 1).
- The 5 T0 tokens (2 of the 17, 3 of the 73) have **the live TF-N6 pipeline (pinned `n6_nt_build.capture` + N6-T selector) choosing the same REPAIR-D / L1 iterate, plan and score**; on the other 15 tokens the live N6 selection certifies nothing (C-N6SAME live layer, 20 of 20 tokens checked).
- No harm by construction and measured: the 3 of the 73 are T0 (TF-N6 plan), later tiers never touched them.

## Deliberate-regression arms on the REAL rows (`mutate_real_rows.py`, `mutations_real_rows.json`)
The stored real candidates of the clean derive are re-selected with the mutated selector (what a derive that ran a different selector would record) and the real merge run:

| mutation | merge | RED controls |
|---|---|---|
| clean | PASS | none |
| strict (TTC / C required) | FAIL | C-ORDER, C-SELECT (N7-T 18) |
| ddc_drop | FAIL | C-CERT-B7, C-N6SAME, C-ORDER, C-SELECT |
| pdms_drop | FAIL | C-CERT-B7, C-SELECT, C-UNCH, C-X0 |
| dac_inverted (all arms) | FAIL | C-CERT-B7, C-N6SAME, C-ORDER, C-SELECT, C-X0 |
| dac_inverted on N7-X only | FAIL | C-CERT-B7, C-SELECT, C-X0 (N7-X changes 19 tokens) |
| tie_latest | FAIL | C-N6SAME, C-ORDER, C-SELECT |
| tier_reverse | FAIL | C-N6SAME, C-ORDER, C-SELECT |
| floor_off | FAIL | C-ORDER, C-SELECT |
| t4_in_t3 | FAIL | C-ORDER, C-SELECT (N7-T3 19) |
| drop_T1 | FAIL | C-ORDER, C-SELECT |
| drop_T0 | FAIL | C-N6SAME, C-ORDER, C-SELECT |
| levers_off (generator not run) | FAIL | C-DET, C-N6SAME, C-REPLAY |
| **real CLI derive with `--mutate drop_T1 --smoke-allow-mutation`** on 4 T1 tokens, rows spliced into the clean set | FAIL | C-ORDER, C-SELECT: the 4 tokens (`437e7aa9`, `7d4286e6`, `c582b495`, `faf94859`) are chosen at T3 (3) and T2 (1) instead of T1 |
| **nc_drop** | **PASS (not detected)** | none: **on these 20 real tokens no candidate with NC worse than the base has DAC = 1 and a builder PDMS above the base**. A DAC-1 candidate with NC = 0 has builder PDMS 0 (NC multiplies the score), so the PDMS clause refuses it anyway; the NC clause binds only for 0 < NC < base NC, which does not occur here. It is exercised on synthetic candidates and through the real repair loops on a fake scorer (`test_nc_ddc_and_pdms_guards_in_the_generated_candidates`, `test_mutation_nc_guard_dropped_is_red`) |

## Mutation of the CODE (not the data): each turned the suite RED
In a scratch copy of `n7_nt_build.py` (restored byte-identical afterwards, `cmp` equal): tie rule `>` -> `>=`, T0 cap `k > cap` -> `k > cap + 1`, T4 floor loosened by 0.11, the C-DET comparison forced True, the C-N6SAME live check disabled, the independent C-ORDER check disabled, the `m_recheck` equality removed from C-CERT-B7: `pytest -k "Select or Real or Card or Capture or Merge"` went from green to RED in all 7 (1 to 7 failing tests each).

## Not exercised here (state it, do not hide it)
- The two banked layers of C-N6SAME (the N6 build report's choices and 6 scores, and bit equality with the banked `n6_plans_N6T.npz`) have no navtrain counterpart; they are exercised on synthetic banked files in `test_n7_chain.py` (`test_n6same_banked_report_and_plans_mismatch_is_red`) and run for the first time on navtest. The N6-T plan file must be fetched from HF (HANDOFF).
- The cache route (`--route cache`, the harness metric cache) and the 12,146-token scoring were not run (navtest is not read); the code path of those parts is the TF-N6 one (pinned `n4_nt_build.ctx_stream`), unchanged.
- The families per-token speed MAE path `frames.executed_tracked.block._per_window.longitudinal.speed_mae_mps.value` is taken from `fam_ci.py` / `per_window.py` and tested on synthetic JSONs; not read from a real `--per-window` JSON (none is on disk): UNVERIFIED, and reported UNAVAILABLE rather than dropped if absent.
- n = 20 tokens on the same 7 navtrain logs the tier order and floor were set on: this smoke checks that the build code path equals the development, not that the generator generalises.
