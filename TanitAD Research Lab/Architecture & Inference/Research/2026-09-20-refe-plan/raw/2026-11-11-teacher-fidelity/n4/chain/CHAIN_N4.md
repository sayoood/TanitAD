# CHAIN_N4: the TF-N4 navtest chain, analyzer, pins template, tests and navtrain smoke

2026-11-12, chain agent (Sonnet 5.5). Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n4_dac/chain/`. Nothing staged, committed or pushed. No navtest token of any N4 arm was scored. Tip read: `924901b7` (`ls-remote` verified). Evidence class of every number below: MEASURED (this session, log paths named) unless marked ESTIMATED.

## 1. Files (all new, all in `chain/`; single copy = the hold, until the MM lands them under `n4/`)
| file | what it does |
|---|---|
| `n4_nt_chain.py` | the detached, marker-based, resumable chain (card sec. 11). Guards, then stages `derive`, `build`, `unch`, `score_D_DM`, `score_X`, `read`, `families`. `--dry-run` prints every command. |
| `n4_nt_build.py` | `reach` (read margin / cap / sign / certificate BACK from the three built repair objects, plus a stub-scorer probe that the margin, cap and sign reach the shift and the loop), `derive` (builder tracked score of the N2-R plan on the harness metric cache, REPAIR-D / N4-X / N4-DM where DAC = 0, C-DERIV, C-DET), `merge` (C-COVB, C-DACMAP, C-DERIV, C-UNCH, C-DET, C-CERT-B; writes plans and seams in TF-N2 A1's format), `unch` (C-UNCH again on the files on disk). |
| `n4_nt_analyze.py` | `read`: controls first (C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C full set, C-EST vs `eval/proxy_eval.boot`, C-REG), exit 1 with NO aggregate if any fails; then PRIMARY bars 1 to 4, paper-level tiers, secondary contrasts (N4-D minus N4-DM, N4-DM paper read, N4-P2, CERT2, human-anchored), Bonferroni with `--k` passed in. `famread`: the four-family guard from `fam_ci.py`. |
| `n4_repair.py`, `n4_geom.py`, `n4_common.py` | copies of the dev files with ADDITIVE edits (see sec. 5). The dev originals in the parent folder are untouched. |
| `n4_make_pins.py` | the MM runs it on the frozen run tree to write `n4/N4_PINS.json` (hashes w/, repo/, n2/, n4/, the REQUIRED inputs, the devkit files, the metric-cache listing digest). |
| `N4_PINS.template.json` | pins from A1's `N2_PINS.json` plus the N4 entries; `tip` and `prereg_sha256` left `<FILLED AT FREEZE>`. |
| `n4_smoke.py` | the navtrain smoke of the sec. 7 mutation arms. |
| `test_n4_chain.py`, `testdata/tf_n1_analyze_a1.py` | 29 CPU tests on synthetic data; the second is the unmodified repo file the analyzer's `Boot` comes from (sha256 equals the A1 pin `f0d0a3f3...`). |

Chain design points. The card numbers (margin 0.05, cap 5, sign -1 for N4-X, certificate full / DAC-only) live in ONE place (`n4_nt_build.CARD`). The chain also takes `--margin --iters --sign-n4x` and refuses any value that differs from the card, and refuses if its own build argv names `--mutate`. The build re-checks the read-back before it derives a token, so a bad repair object stops it even if argv is clean. Builder scoring uses `transp_score.load_cache_parts` (the harness's own `metric_cache.pkl`, no nuPlan DB, no fallback).

## 2. Run tree (the MM assembles it; the guard pins every file)
Copy TF-N2 A1's tree (`F:/tanitad_scratch/tfn2_nt/run_a1/`: `w/`, `repo/`, `n2/tf_n1_analyze_a1.py`, `n2/tfn2_nt_analyze.py`, `inputs/` except `rollouts/`), then: replace `w/pk/eval/instruments/transp_score.py` by the TIP copy (A1's has no `--metric-cache-root`; sha256 A1 `e1dc987c...`, tip `a6c727a1...`); add `n4/` (the chain files, `fam_ci.py`, `N4_PINS.json`); add `inputs/n2r/` (sec. 3); put the card at `n4/PREREG_TF_N4.md` and the K ledger at `inputs/ledger_K.json`.

## 3. Banked inputs the card's sec. 3 names (paths and sha256, MEASURED with `sha256sum` 2026-11-12; read-only)
| input | path | sha256 | other copy |
|---|---|---|---|
| N2-R plans (20 poses), 12,146 tokens | `F:/tanitad_scratch/tfn2_nt/run_a1/build/out/n2_plans_N2R.npz` | `b501cc66f33e005a23e4d619b5e8db417b8eb70a8cf388b40abe17e50a7083c8` | **NONE (single copy on F: scratch)** |
| N2-R seam (8 poses, float32) | `.../build/out/refe_tfn2_n2r.npz` | `ade308c86646fa3eea212e0057e16015694b8461d461a7d5659410e3c0fc9249` | **NONE** (re-creatable from the plans by `to_navsim`) |
| RJ5 target rows (`traj20`) | `.../rolls/RJ5/rows_s0of2.jsonl`, `rows_s1of2.jsonl` | `cdd4c1f9...f195`: `cdd4c1f99e8f22f913d166c0cc9e7c1e48dcc7a8cd3d9b0876a262f748a5cda5`, `a5bef38d01bd9ea2a48b4573c2106b9ae4df2645cea2799a46f76093fe55f195` | **NONE**; 35 MB |
| N2-R harness CSV (comparator) | `.../harness_score/score/refe_tfn2_n2r/refe_tfn2_n2r.csv` | `2a2d4cbcc6ee546336847b9b2a19c0590c56f073e3f341c35d38cdcb4009f709` | repo: `PK/raw/2026-11-11-teacher-fidelity/n2/navtest_a1/run/harness_csv_n2.tar.xz`, same sha256 after extraction |
| N2-C harness CSV (CERT2 order) | `.../refe_tfn2_n2c/refe_tfn2_n2c.csv` | `a3cd0218dca596202ca6ac7b8d8e52650725bd9e588cf9c84a53b3fe4d482bb7` | same tar |
| T seam / T CSV / human CSV / N1-C CSV | `.../inputs/` | in `N2_PINS.json` (T seam `29c8129f...`) | repo banks |
| metric cache root | `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest` | listing digest `2ce6c713d4bbe2696990f7e60dd36fa43d1b81bfc46d7331e558084db90ddac6`, 12,146 files (sorted `<log>/unknown/<token>/metric_cache.pkl:<size>` lines), 56 s | the harness's own cache |

**MM action: bank `n2_plans_N2R.npz`, `refe_tfn2_n2r.npz` and the two RJ5 rows files (about 48 MB) before anything sweeps `F:/tanitad_scratch/`.** Re-creation costs the RJ5 roll (hours).

## 3b. Question-by-question differences from the draft card (card NOT edited)
1. Sec. 3 / 11 say the builder reads the map "via W-TRANSP's `--metric-cache-root` route". That route exists only in the tip's `transp_score.py` (A1's pinned copy lacks it). The chain uses `load_cache_parts` from the tip copy for build AND families, so the pin changes (sec. 2). **Edit needed in sec. 13 pins: transp_score sha is the tip one.**
2. Sec. 10 says the families are "labelled not harness-identical as in N2-A1". With `--metric-cache-root` they are on the harness scenery (D-REFE-WTRANSP-CACHE-1/2, 12,146 / 12,146 identity on T). The chain uses the cache route for N2R and N4D (one pass, same scenery). The label in sec. 10 is conservative, not wrong; the MM may relabel.
3. Sec. 7 C-TESTS names `test_n4_repair.py`: it reads navtrain rows from `F:/tanitad_scratch/n4dac/rows`, which no longer exists. `n4_common.ROWS` now honours `N4_ROWS`; the rows come from the repo tar `n2/reroll/run/rolls_scores.tar.xz`.
4. **N4-P2 is not separately scored**: its harness row is CERT2's banked row where a candidate passes DAC, else the N4-D row (REPAIR-D of N2-R is exactly N4-D's plan on those tokens). Derived in the analyzer, labelled so.
5. N4-X is a full 12,146-token seam by the MM decision, but its plan is N2-R's on every token (the flipped shift never certifies: 0 of 5 in the smoke, `smoke_results.json`), so it re-scores ~12,146 rows that should equal N2-R's. It still makes C-REG a real harness check; dropping it would save about 47 min wall (ESTIMATED from A1's W seam, 21:43 to 22:29). MM call.
6. C-CERT and C-ID-C are applied to all three new arms (any failure voids the read). Conservative; the card names N4-D only for C-CERT.
7. "Diagnostic, never voids: human CSV re-read" is not computed: N4 re-scores no human seam, so there is nothing to compare.
8. Margin -0.05 on the smoke: the READ-BACK refuses it (RED) and, with that guard bypassed in the smoke only, the builder controls still pass and the repair rate collapses to 0 / 5. On navtest a collapsed rate is bar 3 (NOT CONFIRMED), not a control void; the analyzer has no builder-rate control by design.

## 4. Exact launch command (for the MM, after freeze, K entry appended, pins written)
```
# 1. guards only (exit 0 = pass; prints ZZN4 GUARDS PASS):
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe <RUN>/n4/n4_nt_chain.py --run-dir <RUN> --card <RUN>/n4/PREREG_TF_N4.md --card-sha256 <SHA> --pins-sha256 <PSHA> --ledger <RUN>/inputs/ledger_K.json --k 45 --procs 2 --check-only
# 2. print every command (writes nothing):   same argv + --dry-run
# 3. detached run (as TF-N2 A1's launch_tfn2_nt.ps1: Win32_Process Create, CreateFlags 0x08000200, stdin NUL):
cmd /c "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe <RUN>/n4/n4_nt_chain.py --run-dir <RUN> --card <RUN>/n4/PREREG_TF_N4.md --card-sha256 <SHA> --pins-sha256 <PSHA> --ledger <RUN>/inputs/ledger_K.json --k 45 --procs 2 >> <RUN>/n4run/chain.log 2>&1 < NUL"
```
`<RUN>/n4run/` must exist first. Outputs: `n4run/build/out/` (plans, seams, `n4_build_report.json`), `n4run/harness_score/`, `n4run/out/n4_result.json` (the read), `n4run/out/n4_families.json`. Resume = rerun the same command (markers). Success line: `ZZN4_CHAIN DONE`; stop line: `ZZN4_CHAIN STOP`. `TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim` is set by the chain (A1 lost an attempt to the E: default). To regenerate pins: `python n4_make_pins.py --run-dir <RUN> --tip <SHA>`.

## 5. Edits to dev code (all additive, defaults reproduce dev behaviour bit for bit)
`n4_repair.py` (dev sha256 `04633326...`): `CAP = 5`; `certified_dac` and `CERTS = {"full", "dac"}`; `repair_target(..., iters=CAP, sign=1.0, cert=certified)` (sign multiplies the shift, `cert` replaces the hard-coded `certified`). `n4_geom.py` (dev `78f7fd7a...`): `Ctx(sc, parts=None)` plus `Ctx.from_parts`; `P5R` honours `N4_P5R`. `n4_common.py` (dev `ae75bfd9...`): `ROWS` honours `N4_ROWS`. The dev folder files were not edited (an in-place overwrite was denied by the harness; copies in `chain/` instead). `PREREG_TF_N4_DRAFT.md`: **no edit.**

## 6. Tests and navtrain smoke (venv `C:/Users/Admin/navsim-crun/venv`, CPU, `CUDA_VISIBLE_DEVICES=-1`, W = pristine tip `924901b7` export of `eval/`, `refe/`, `raw/2026-11-12-p5r/`)
Environment: `PYTHONPATH=<devkit 3e8291b>;<W>/refe`, `N4_W=<W>`, `N4_P5R=<dir with p5r_labels.py>`, `N4_ROWS=<extracted rolls_scores.tar.xz>`, `N4_TMP=<scratch>`; run from `chain/`.

| run | result |
|---|---|
| `python ../test_n4_repair.py` (dev, 5 tests, with the chain/ modules first on the path) | **5 OK**, 18.1 s |
| `python -m unittest test_n4_chain` (29 tests) | **29 OK**, 27.1 s |
| smoke REAL arm (5 RJ5 failures + 20 unchanged navtrain tokens, 2 logs) | **GREEN**: all builder controls PASS; REPAIR-D 5 / 5, N4-DM 5 / 5, N4-X 0 / 5; on-disk C-UNCH PASS |
| smoke certificate disabled (accept first iterate) | **RED**: C-CERT-B and C-UNCH fail; unflagged guard run is REFUSED (read-back) |
| smoke repair applied to ALL tokens | **RED**: C-UNCH fails (20 applied plans on DAC-1 tokens) |
| smoke margin -0.05 | **RED** at the read-back (REFUSING); guard bypassed in the smoke only: repair rate 0 / 5 (a bar-3 outcome on navtest, sec. 3b item 8) |
| smoke 2e-6 perturbation of one unchanged plan in the written files | **RED** (`unch` rc 1); the unperturbed files rc 0 |
| smoke / test: a pinned file edited | **RED**: `TestChain.test_a_pinned_file_edited_is_refused`, `TestRead.test_control_c_pin_pinned_file_edited` |

Smoke caveats. Navtrain has no harness cache: the smoke scenery is the nuPlan-DB rebuild, the "N2-R plan" is the dev R label and the C-DACMAP reference is the banked W-TRANSP tracked row, so C-DACMAP GREEN there is an identity check of the code path, NOT of the harness. The harness-side controls (C-CERT, C-ID-C, C-COV, C-EST, C-REG) are exercised on synthetic CSVs only (each RED in `test_n4_chain.py`, each exits before any N4-D aggregate: output file has no `primary` or `tables` key and stdout no `ZZN4_VERDICT`). C-DACMAP, C-DERIV on real navtest plans, and C-CERT against the real harness are first run in the frozen run. Smoke raw: `F:/tanitad_scratch/n4chain/smoke/` (deleted at the end; the table above is the record; `smoke_results.json` figures are quoted in sec. 6).

## 7. ESTIMATED wall time (2 processes, 0 GPU)
Build: tracked score of 12,146 N2-R plans, about 0.4 CPU-s per token (from `scale/SCALE.md`: 560 s wall on 3 processes for the same scoring route) is about 1.4 CPU-h; repairs on about 120 to 200 tokens at 12 to 21 s each for all three arms (MEASURED on the smoke: 12.5, 15.8, 17.2, 21.1, 21.2 s) is about 0.5 to 1 CPU-h; so **derive about 1 to 1.3 h wall**, merge and unch minutes. Score: A1 measured about 46 to 56 min per full seam (chain.log of `run_a1`); D and DM in parallel, then X: **about 1.7 h**. Families (2 arms + human at 2 procs, then `families_run`, `fam_ci`): about 1 to 1.5 h (A1's was about 80 min with 4 arms). **Total about 4 to 5.5 h wall**, no GPU, no pod. The card's "4 to 8 h" holds.
