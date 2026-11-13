# P5-C navtest read instrument: B5, B6, B7, B8, B9 (built 2026-11-13, CPU only, no P5-C number)

Card implemented as written: `PREREG_P5C.md` (frozen 2026-11-12) section 5, 6, 7, 8.1, 10.5, 11; `PREREG_P5C_A1.md` is the label amendment and adds nothing to the read.
Source tip: `origin/agent/arch-inf-20260803` = `7f7ef7c7` (ls-remote confirmed). Nothing under `stack/`, `refc_v3_train.py`, `eval/harness/*` or `HARNESS_PINS.json` was touched. No `BASELINES` edit (the `p6` entry, B2, is already at the tip).
Evidence class of every reading below: MEASURED (this run, artifacts in `files/raw/2026-11-13-p5c-read/dryrun/`). It is a REHEARSAL: C1 = P6 (`refe_p6final`), C0 = u400, base = deployed route-fix, on a 500-token seeded subset (seed 20261113, sha1 rule). It says nothing about P5-C.

## What was built (repo layout = `files/` relative to PK; all new files)
| item | file | lines (code / test) | note |
|---|---|---|---|
| B5 + B6 | `eval/p5c/p5c_read_prep.py`, `test_p5c_read_prep.py` | 334 / 224 | `convert`, `score`, `identity`, `red` |
| B7 | `eval/p5c/p5c_read.py`, `test_p5c_read.py` | 365 / 272 | verdict table, qualifier, guards, rates, mechanism counts |
| B8 | `eval/p5c/p5c_render.py`, `test_p5c_render.py` | 165 / 133 | printed-twice renderer |
| B9 | `eval/p5c/p5c_families.py`, `test_p5c_families.py` | 121 / 68 | wrapper over `families_run.py --per-window` + `fam_ci.py` |
| dry run | `raw/2026-11-13-p5c-read/dryrun/` | | identity x3, RED, read JSON, rendered report, family CIs, rows |
Tests: 96 passed, 2 skipped (`proxy_eval` import; the dry-run-artifact test skips unless `P5C_DRYRUN_DIR` or the landed `raw/` dir exists, and passes with it), two runs, `logs/TEST_RUN_1.log`, `TEST_RUN_2.log`. Every piece has RED arms (below) and a determinism test. B7 JSON and B8 text are bit-identical on a second run (`cmp`, MEASURED).

## B5: converter and one scoring pass
* `convert --arm NAME=SEAM.npz[,DUMP.npz]`. **Form rule:** `pick20+plans8` when the dump carries `pick20` (the registered form: tracked arm = the seam's poses, transparent arm = E2's spline through the 20 poses at dt 0.2); `plans8` otherwise (transparent from 8 poses at dt 0.5, ~1.5 cm coarser, not the registered form). The report prints the form per arm and flags **mixed forms** (not like-for-like). **Form used in this dry run: `plans8` for all three arms**, because the banked P6/u400/deployed seams predate `pick20`; the real C1/C0/P7 reads will carry `pick20`.
* `score` = one `transp_score.py` pass over all arms plus `--human-arm`, `--split navtest`, `--metric-cache-root` always on (a score command without it is the not-harness-identical DB route, and `build_score_cmds` cannot emit one outside the RED arm), at most 2 processes, `OMP/MKL/OPENBLAS=1`, `TANITAD_NAVSIM_ROOT` pinned to `F:/Archive/devbox-C/navsim` (read only). Resumable (transp_score's own).
* Measured cost: 500 tokens x 4 arms (3 + human, with the family payload) = **3.3 min wall on 2 processes** (18:03 to 18:06:12), so full navtest ~ 80 min ESTIMATED (linear).

## B6: identity control (card 10.5 W-TRANSP i)
Tracked sub-scores (nc, dac, ddc, ttc, ep, c, pdms) of each model arm vs the pinned harness csv, per token, bar 1e-6; PASS needs full coverage, zero error rows, zero tokens over.
| arm (stand-in) | harness csv | n | over 1e-6 | max abs diff | verdict |
|---|---|---|---|---|---|
| C1 = P6 | `p6_navtest_full.csv` | 500 / 500 | 0 | 1.80e-12 | PASS |
| C0 = u400 | `u400_navtest_full.csv` | 500 / 500 | 0 | 1.65e-12 | PASS |
| base = deployed | `deployed_navtest_full.csv` | 500 / 500 | 0 | 1.90e-12 | PASS |
**RED arm (the routing line deleted):** a mutated copy of `transp_score.py` (the line `parts = load_cache_parts(...) if a.metric_cache_root else ...` replaced by the DB rebuild; the mutation asserts exactly one occurrence) on 60 tokens of P6: **26 / 60 tokens over 1e-6, max EP diff 0.0115, PDMS 0.0048: DETECTED** (`red_result.json`). Plus unit RED arms: a 1e-4 perturbation of one tracked value, one metric at a time (all 7), a missing token, an error row, and the bar edge (5e-7 passes, 1.5e-6 fails).
The full 12,146-token B6 needs no extra scoring: it is the identity check on the full B5 pass (command in `HANDOFF.md`).

## B7: the read
Estimator: log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, **one draw matrix shared** by every arm, scorer and contrast, and equal to `fam_ci`'s draws (test: CI equals `fam_ci.cluster_boot` to 1e-12; the instrument sums with `bincount`, so the last ulp differs). TD is computed per draw. Rates use the ratio estimator. Implemented from the card: B = max(2.776 x 0.1791 x sqrt 2, 2 x 0.33) = **0.7031** (asserted); P1-a, P1-b (strict), G1 (NC, DAC, TTC, EP, C; adverse = CI-hi < 0), G2 (families, tracked-executed frame only, via `fam_ci`), CTRL-1 (INVALID iff CI-hi(C0 - base) < 0 and C0 - base < -B), P1-c with C0r (d_r, F_p5c = |C0r - C0|, min(d, d_r) > max(B, 2 F_p5c)), the verdict ladder (INVALID, REFUTED, NULL, SWAMPED, GUARD-FAIL, LEVER / SWAMPED-BY-REPLICATE, SEPARATED-SINGLE), qualifier TM / TM+C / TI / TU, deploy clause, goal 94.55, per-scorer sub-scores, DAC / PDMS / NC-zero rates, tracker-made and tracker-saved DAC zeros, STRATEGIC = PAUSED. Tests: known values (constant shift d = 2.0 exactly, rates 10 % / 5 %), 11 ladder cases, 6 qualifier cases, boundary cases for CTRL-1 and P1-b, RED arm (a per-token bootstrap that ignores logs is > 2x too narrow on clustered data).
**Dry-run reading (rehearsal, 500 tokens, 113 logs):** levels tracked / transparent (PDMS x 100): C1 89.19 / 90.18, C0 88.63 / 88.98, base 84.07 / 85.52, human 94.86 / 92.01. d = +0.56 [-1.77, +2.90], d_T = +1.19 [-0.91, +3.19], TD = -0.63 [-2.77, +1.53] -> verdict `REHEARSAL:NULL`, qualifier TU. G1 flagged TTC (adverse), G2 flagged 8 tracked-frame components (speed MAE / RMSE, target-speed accuracy at 0.5, along-track MAE, progress error, yaw-rate MAE, final cross-track MAE, last-pose heading error) for P6 vs u400 on this subset. These are properties of the two stand-in arms on 500 tokens, not findings.

## B9: families
`families_run.py --per-window` per arm on the rows, `fam_ci.py` per pair (C1:C0, C0:base, C1:base; difference = x - y), summary per frame with its role. Frames: `executed_tracked` (primary, the only guard), `executed_transparent` (reported, never a guard), `reference`. Tactical components included (accuracy, kappa, per-class); STRATEGIC PAUSED. Ran in 56 s for 3 arms + 3 pairs on 500 tokens.
**A bug found and fixed while building B7 (class: a name split at its decimal point):** `fam_direction` was first applied to the last dot-segment, so `target_speed_acc.within_1.0_mps` became `0_mps` and read as LOWER-is-better; a higher accuracy of C1 would have been flagged adverse. `proxy_eval.fam_adverse` passes the full dotted name; the instrument now does too, with a regression test (`test_RED_polarity_of_dotted_names_keeps_the_decimal_point`).

## B8: renderer
Refuses (exit 2, nothing written): a number without both scorers and without the label "no tracker in this number"; banned wording anywhere in the rendered text (patterns for "drives better", "safer", "better driving", "improves driving/safety", "better driver", "drives more safely/smoothly", "drives like a human", ...; 16 RED phrases, also under TI; allowed phrases such as "survive the benchmark's tracker better" and "drivable area compliance" pass); a verdict without exactly one qualifier, or a qualifier its CIs do not support. LEVER prints the card's claim sentence verbatim; SEPARATED-SINGLE and negative verdicts print no "survive ... better" claim. TI adds "not a closed-loop result". A rehearsal banner is the first line.

## Findings that change how the instrument is read
1. **The B3 control (`pick20` bit-for-bit vs the seam) only covers 12 of the 20 poses.** `to_navsim` reads pose indices 1, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19; the other 8 (0, 3, 5, 8, 10, 13, 15, 18) feed only the transparent arm and are unverified by it (test `test_control_limit_only_12_of_20_poses_are_checked`; the convert report lists them as `pick20_unverified_poses`). A `pick20` that is wrong in those 8 would pass B3 and move only the transparent numbers.
2. The polarity bug above (fixed).
3. On these stand-ins the G2 guard fires on a pair where the higher-PDMS arm has worse speed error: G2 is not decorative; a real C1 - C0 will be read against it.

## Card questions (listed, not decided; also in the read JSON `card_questions`)
* Q-1 Qualifier order: TM+C's condition implies TM's; the instrument prints TM+C when both match.
* Q-2 GOAL MET in PREREG_P6 needs "AND ADOPT vs C1"; the P5-C card has no counterpart. Only the level half is computed and labelled "level half only".
* Q-3 G1 lists NC, DAC, TTC, EP, C (no DDC). DDC is reported, not a G1 term.
* Q-4 G2 polarity: directional components use the paired C1 - C0 interval; bias and ratio components use P6's unpaired interval rule (`proxy_eval.fam_adverse`, conservative).
* Q-5 No controls supplied: the card makes a failed control INVALID but is silent on an absent one. The instrument refuses a real read without `--controls` (exit 2; `--rehearsal` overrides with a label) and never issues a positive verdict without them (INCOMPLETE).
* Q-6 SEPARATED-SINGLE's wording is given in the card; whether the "survive ... better" claim sentence applies to it ("any positive verdict") is not stated. The renderer prints the card's SEPARATED-SINGLE wording plus the numbers, without the "better" sentence.

## Not built (stated, not narrowed away)
B10 (determinism subset), B11 (SEALED.json, K ledger), B12 (tactical rows via existing tools) are outside this brief. `--controls` accepts any JSON with a boolean `pass`, so B10 and the W-TRANSP controls (ii), (iii), (iv) plug in; the B7 verdict is INCOMPLETE until they are supplied. W-TRANSP (i) at 12,146 tokens for a model arm is scheduled, not run (HANDOFF).

## Deliverable manifest
Hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_read/` (single copy of everything here; scratch was deleted, see HANDOFF deletion log):
`P5C_READ_BUILD.md`, `HANDOFF.md`, `CHANGED.txt`, `SHA256.txt`, `logs/TEST_RUN_1.log`, `logs/TEST_RUN_2.log`, `files/eval/p5c/*.py` (8 files), `files/raw/2026-11-13-p5c-read/dryrun/*` (11 files, incl. `rows_500.tar.xz`).
