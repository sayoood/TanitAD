# HANDOFF: W-TRANSP --metric-cache-root, reachability-proven (2026-11-12). Nothing staged, committed or pushed. MM lands it.

**Headline: ready to land.** Replaces `eval/instruments/transp_score.py` (+ new check and unit test, + INSTRUMENTS.md edit). Integration decision for MM: any navtest
W-TRANSP run that must be harness-identical has to pass `--metric-cache-root`; the callers below do not yet.

## What changed (`transp_score_cachefix.diff`, 56 lines, vs tip e1b3a76 blob)
* `load_cache_parts(root, log, token)`: reads `<root>/<log>/unknown/<token>/metric_cache.pkl` (lzma pickle) -> the 6 parts; missing file raises FileNotFoundError naming the path (no DB fallback).
* `--metric-cache-root` (default empty); one routing line: `parts = load_cache_parts(...) if a.metric_cache_root else PT.metric_cache_parts(sc)`. The DB `sc` is still built, for the human future and lead track.
* `families_run.py`, `validate_controls.py`: build no scenery (read rows) -> no change needed.

## Check results (MEASURED, `results/check_cli_cachefix.json`; human reference seam only, 60 tokens = sorted first 60 of `e0_human_sub200`; CLI subprocess; tracked sub-scores vs `refe_e0b_human_sub200.csv`, tol 1e-6)
| arm | tokens over 1e-6 | max abs diff |
|---|---|---|
| FIX (`--metric-cache-root`) | **0 / 60** | 0.0 on nc dac ddc ep ttc c pdms |
| REGRESS (flag absent, DB route) | **17 / 60** (discriminates) | ep 4.06e-4, pdms 1.69e-4 |
| MUTATION (routing line deleted, flag given) | **17 / 60** (RED) | same as DB route |
`check_cli_cachefix.py` rc 0 (PASS). Default path unchanged: 10 of 10 rows equal (minus `t_s`) to the unmodified tip script. Unit test `test_transp_cache_flag.py`: 3 tests OK (flag default "", default route keeps `metric_cache_parts`, missing cache names path).
Not done: the 12,146-token run (60-token CLI is reachability, not scale). Evidence class: MEASURED. No ledger entry (human reference arm only; no N1/N2/model arm scored).

## Callers that would have to pass the flag (listed, NOT changed; frozen cards / pinned run trees)
* `raw/2026-11-11-teacher-fidelity/n1/tf_n1_chain.py:224-226` (N1, `--split navtest`)
* `raw/2026-11-11-teacher-fidelity/n2/navtest/tfn2_nt_chain.py:487-491` and `n2/navtest_a1/tfn2_nt_chain.py:508-512` (their analyzers label the basis "NOT harness-identical")
* `n2/reroll/tfn2_reroll_score.py:65`, `n2/tfn2_probe.py:158` import `transp_score` as a library (they call `PT.metric_cache_parts` path via score_arms inputs; not the CLI)
* `n1/smoke/run_test.sh:24` is navtrain: flag NOT applicable (no harness cache). Same for any navtrain run.
* pins listing `w/pk/eval/instruments/transp_score.py` (N1_PINS, N2_PINS, `n1_make_pins.py`, `tfn2_nt_chain.py`) change sha on landing: re-pin.
* The A1 `transp_score_a1fix.py` in teacher_fidelity/n1/a1 is superseded by this file (same logic plus error handling).

## Run it
`PYTHONPATH=F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;<refe dir>  C:/Users/Admin/navsim-crun/venv/Scripts/python.exe check_cli_cachefix.py --refe <refe dir> --seam E:/Projects/TanitAD/data/refe_navtest/seams/human_sub200.npz --banked refe_e0b_human_sub200.csv --work <scratch> --out r.json --n 60` (~3 min, 2 procs, CPU).

## Deliverable manifest (all under `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/instruments/wtransp_cachefix/`, repo-relative mirror under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`; each exists ONLY here)
| artifact | path |
|---|---|
| fixed CLI (sha256 a6c727a1) | `<mirror>/eval/instruments/transp_score.py` |
| CLI reachability check (44ea614d) | `<mirror>/eval/instruments/check_cli_cachefix.py` |
| unit test (d3be8c51) | `<mirror>/eval/instruments/test_transp_cache_flag.py` |
| register (be6c0915) | `<mirror>/raw/2026-11-12-instruments/INSTRUMENTS.md` |
| check result, diff, handoff | `results/check_cli_cachefix.json`, `transp_score_cachefix.diff`, `HANDOFF.md` |
Lands: the 4 mirror files (expect.tsv base = tip blob for the two existing files). LF endings verified (0 CR).

## Deletions (CLAUDE.md rule 4)
Scratch `F:/tanitad_scratch/wtransp_cachefix_scratch/` deleted by literal path at the end (size logged in the report). I killed 7 of my own stray processes by explicit PID after launching the check once with the wrong banked CSV (a1 smoke copy); rerun used the tip's `refe_e0b_human_sub200.csv` (sha256 44f3ddde).
