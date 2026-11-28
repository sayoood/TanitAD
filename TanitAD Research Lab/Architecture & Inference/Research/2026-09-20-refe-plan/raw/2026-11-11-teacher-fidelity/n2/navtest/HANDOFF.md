# HANDOFF: TF-N2 navtest implementation (2026-11-12). Nothing staged, committed or pushed. No navtest token rolled, scored or opened.

The card, the chain, the tests and a navtrain smoke are done. **The MM freezes, appends the K entry (n = 43) and launches.** All files are LF only.

## Files (this folder: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2/navtest/`)
| file | role |
|---|---|
| `PREREG_TF_N2.md` | the card, ready to freeze. Two fields for the MM: the landed tip sha and `N2_PINS.json`'s sha256 (§11) |
| `n2_nt_prep.py` | builds the run tree from the git tip and the hold, verifies the harness pins, writes `n2/N2_PINS.json` |
| `tfn2_nt_chain.py` | the chain: guards, `roll_R0`, `ctl_R0`, `roll_RJ5`, `build`, `score_T`, `cidh`, `cidc`, `score_N2`, `read`, `families` |
| `tfn2_nt_roll.py` | wraps the registered `teacher_navtest_seam.rollouts`; arms R0 and RJ5 only; knob read-back per row |
| `tfn2_nt_build.py` | N2-R (P5-R), N2-C, N2-W and N1-C plans as W3 seams and plan files; C-BUILD, C-DERIV |
| `tfn2_nt_analyze.py` | `read` (controls first, verdicts), `famprep`, `famread` (the longitudinal guard) |
| `launch_tfn2_nt.ps1` | detached launch (`Invoke-CimMethod Win32_Process Create`, `0x08000200`, `< NUL`) |
| `test_tfn2_nt_chain.py` (55 tests), `test_tfn2_nt_roll_reach.py` (6) | refusals, controls and their mutations |
| `raw/smoke/` | the smoke, the mutation arms, the test output (see below) |

The chain needs three files of the hold that are NOT in it, because they are already at the tip: `tf_n1_analyze_a1.py`, `score_p4_a1.py`, and the patched `build_teacher_rollouts.py` (all under `raw/2026-11-11-teacher-fidelity/`), plus `p5r_labels.py` (`raw/2026-11-12-p5r/`). Prep takes them from the tip.

## Where the inputs are (the HF fallback was NOT needed)
The banked teacher rollouts, T seam, CSVs and N1-C's CSV are all in the repo at tip `ed440366`; `n2_nt_prep.py` reads them with `git show`. The N1 run dir is not needed.

## Land first
Land this hold (e.g. at `raw/2026-11-11-teacher-fidelity/n2/navtest/`). The lander rewrites CRLF to LF; the files are already LF, so no sha moves. Then:

## Freeze, ledger, launch (in this order; the chain refuses otherwise)
1. **Prep** (fresh tree on F:, about 30 s, no scoring):
   `python <hold>/n2_nt_prep.py --run-dir F:/tanitad_scratch/tfn2_nt/run --mirror C:/Users/Admin/tanitad-push/.git --tip <landed sha>`
   It prints `N2_PINS.json sha256 <X>`. If the run dir exists it refuses (use a new name; delete the old one by literal path).
2. **Card:** write the tip sha and `<X>` into §11 of `PREREG_TF_N2.md` (the two `<MM fills>` fields). Rule on §6's bars (the +0.5 and the proposed guard §10). Freeze: `sha256sum PREREG_TF_N2.md` = `<C>`. Copy the card to `F:/tanitad_scratch/tfn2_nt/run/n2/PREREG_TF_N2.md`.
3. **Ledger:** append the §9 entry with key `TF-N2:<C>:E3`, n = 43 (the ledger at the tip holds 42, max n 42; if another entry landed first, n and K shift: change `--k` and the constants `K = 43` in `tfn2_nt_analyze.py` and `default=43` in the chain, re-run prep, re-freeze). Land it, then copy the landed `ledger_K.json` to `F:/tanitad_scratch/tfn2_nt/run/inputs/ledger_K.json`.
4. **Check:** `. <hold>/launch_tfn2_nt.ps1 -CardSha <C> -PinsSha <X> -CheckOnly` (guards only; prints `ZZN2 GUARDS PASS`).
5. **Launch:** the same line without `-CheckOnly`. Log: `<run>/chain.log`; markers: `<run>/markers/`. The same command resumes after a Windows-Update kill (rc 1073807364) or a stop. Use `-StopAfter roll_R0` first if you want to see C-R0 before the long roll.
   The launch script copies the chain from `<run>/n2/`, which prep wrote; run it from the MM's PowerShell (no `-ExecutionPolicy Bypass`).

## What to expect and read
- `<run>/out/ctl_R0.json` (C-R0), `<run>/build/out/n2_build_report.json`, `<run>/out/cidh_early.json`, `<run>/out/cidc.json`, `<run>/out/n2_result.json` + `read.log` (`ZZN2_VERDICT PRIMARY / PAPER`), `<run>/out/n2_families.json` + `famread.log` (`ZZN2_GUARD`).
- A stop is always a marker `<stage>.FAIL` and `ZZN2_CHAIN STOP` in `chain.log`. `ctl_R0` or `cidh` or `cidc` FAIL means no later stage ran and no N2-R, N2-C or N2-W PDMS exists.
- Wall about 10 to 11 h on 2 processes (ESTIMATED, §11 cost table). It never runs more than 2 child processes and waits for 10 GB of free virtual memory per child.
- After the read: bank `<run>/out/`, `build/out/n2_build_report.json`, `harness_score/` CSVs, `fam/`, `markers/`, `chain.log` and verify by sha256; then delete the run dir by its literal path and log it.

## Evidence this agent produced (MEASURED, navtrain only; raw in `raw/smoke/`)
- **Smoke, 5 navtrain tokens, the real chain with real guards** on a freshly prepped tree: guards PASS (938 pins); `roll_R0` ok; **`ctl_R0` max |d| 0.0 m** against the re-roll's own R0 rollouts; `roll_RJ5` ok with `knobs` jerk 5.0 on every row and plans moved by 0.11 to 5.46 m against R0; `build` ok with **C-DERIV 0.0** (R and C plans equal the re-roll's scored plans); `families` ok through `famread`. The harness stages are skipped in smoke (navtest metric cache only).
- **Mutations through the chain:** `TFN2_MUTATE_DROP_OVERRIDE=1` → `roll_RJ5` FAIL, 5 of 5 rows are `KNOB-NOT-REACHED`; a banked rollout perturbed by 2e-6 → `ctl_R0` FAIL, no RJ5 row written; build merge with a missing banked token → C-BUILD-COV false; a 1e-6 deviation in a derived row → C-DERIV false.
- **Tests:** 55 + 6 pass (CPU). The 6 roll tests run under `run_p4.py` in the driverl-eval venv.
- **C-DERIV on all 469 navtrain tokens:** max deviation 0.0 m for R and C, 0 errors, 0.48 s/token (`raw/smoke/cderiv469_report.json`).
- **Timings:** RJ5 roll 1.42 s/token (smoke), the bank's own mean 1.35 s/token over 12,146; derive 0.51 s/token; W-TRANSP rows 0.92 s/token for 3 arms + human.

## Open items and caveats (for the MM)
1. **The harness stages (`score_T`, `cidh`, `cidc`, `score_N2`, `read`) were NOT run end to end**: navtest only, and this agent may not score navtest. They are TF-N1-A1's `score_p4_a1.py` unchanged. What is tested: `compare_csv` (1e-6 on 7 metrics, subset mode), the read's controls with a mutation each, the verdict rule branches, CERT2's order. A first full run is therefore also the first run of `cidc`'s subset seam through the harness (`--tokens` with a `token_log` json, as `run_v1.py:split_overrides` reads it). If `cidc` stops the chain, nothing but the 200-token subset has been scored.
2. **Proposals needing a ruling before the freeze:** the primary bar +0.5 with the ΔDAC ≥ −0.25 clause (§6), and the four-check longitudinal guard (§10). The guard is flag-only.
3. **Families use the pinned `transp_score.py`, flagged "not harness-identical"** (the A1 `--metric-cache-root` repair is untested and is not used).
4. `S.rollouts` never retries a token that has any row; the chain therefore drops error rows before every (re)start of a roll stage. A resume after a transient failure re-rolls them.
5. RJ5 is not the registered protocol; the card says so in every quotation rule.

## Deletions (CLAUDE.md rule 4, by literal path)
- `F:	anitad_scratch	fn2_nt` deleted at the end, ~166 MB: two prepped smoke trees (`smoke_run`, `run_s2`, re-creatable by `n2_nt_prep.py` from tip `ed440366`), the smoke and mutation outputs (banked first under `raw/smoke/`), the extracted re-roll tarball `reroll_x` (source `../reroll/run/rolls_scores.tar.xz` still on E:), the 469-token C-DERIV run (report banked), temp test dirs.
- The Sonnet-run `F:/tanitad_scratch/tfn2_reroll/` belongs to the re-roll agent and was not touched (it was already absent).
