# HANDOFF: RECOVER-NEXT dev-box read (reader `--r-arm`, combiner, chain), BUILT, NOT LAUNCHED; written 2026-11-03 06:10

Package `E:/Projects/TanitAD-artifacts/dagland/recover_next/read/`. Card `../PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6...`) section 3 is what this implements. No read or rollout of RAPR / RTL / RP01 was started (their checkpoints do not exist). Nothing is in git. Nothing was written to F:; scratch is `C:/Users/Admin/rr_scratch/rnext_work/` (extractor rigs, new) and `.../rnext/` (chain write root, created by the chain). The existing `rr_scratch/out` was only read (latest mtime unchanged, 05:25).

## State (MEASURED unless marked)
* **Reader** `code/recover_read.py` = the source reader (`recover_read/code/recover_read.py`, sha `bf533d6b...`) + `--r-arm {R,RAPR,RTL,RP01}` (default R). Rules, bars, rows, `evaluate()` untouched.
* **Known answer PASS** (`raw/KA_RARM_R.json`): `--r-arm R` recomputed from the 15 existing rollouts (file controls, 15 fresh extractions, every row at B = 2000, seed 0) reproduces `recover_read/raw/RECOVER_READ.json` with **0 differences over 2,599 leaves** (ignored: `run_clock`, `reader_sha256`) and all **15 extracted arrays are bit-identical** to the source's. Output `raw/RECOVER_READ_R.json` (label HARMS, as the source).
* **Tests**: the 27 source tests run against the new reader, unchanged: **27 passed** (`logs/pytest_recover_read.log`). New `tests/test_recover_read_arms.py`: **30 passed** (run table per arm incl. arm-R table == source table row for row; ARM_SPEC diff == the shipped argv difference for all 6 argv files; argv audit passes on shipped argv and fails on each break; KR1 band; KR3 a-priori and bank variants; CLI `--list-runs` / `--dry-run`; and a **production-path rehearsal of `file_controls` for each arm on stand-ins**: no crash, arm control names present, json-only controls PASS). `tests/test_recover_next_combine.py`: **18 passed**, 5 mutations go RED (blend/dose swap, Q1 both-seeds, RAPR strict `>`, table order, row-2 closure clause).
* **Combiner** `code/recover_next_combine.py`: per-arm label / composite status; Q1; RAPR secondary (R's shares READ from `recover_read/raw/RECOVER_READ.json`); section-4 next arm. Reads whichever `raw/RECOVER_READ_{RAPR,RTL,RP01}.json` exist (a missing / refused / wrong-arm file is reported, not an error). Schema coupling to the real reader JSON is tested.
* **Chain** `rnext_read_chain.sh <ARM> [--dry-run]` + `rnext_read_chain_<ARM>.cmd` (WMI launcher). Gate 0 verified live (25 s run: `logs/rehearsal_gate0_RAPR.out`: cards, tool, pins, tests, known answer ok; then it waits at gate 1, as designed). `--dry-run` for the three arms: `logs/dryrun_<ARM>.out`. Pins: `PINS.txt`.

## Exact commands for the Master Mind (after the PI's go, the box-2 chain and `pull_rnext.sh`)
Launch ONE arm at a time (two chains started together both pass the CPU gate and then contend). Order = card order: RAPR, RTL, RP01.
```bash
P=E:/Projects/TanitAD-artifacts/dagland/recover_next/read
bash $P/rnext_read_chain.sh RAPR --dry-run          # pulled runs must read present, the 8 reused present
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='E:\Projects\TanitAD-artifacts\dagland\recover_next\read\rnext_read_chain_RAPR.cmd'}"
ls $P/raw/RNEXT_*; tail -n 3 $P/logs/chain_RAPR.out        # ONE poll command (or a script wake on raw/RNEXT_RAPR.done|fail)
# repeat with RTL, RP01 after RNEXT_<prev>.done. RESUME = launch the same .cmd again (finished rollouts are skipped).
# after each arm: raw/RECOVER_READ_<ARM>.json and raw/RECOVER_NEXT.json (rewritten from whichever arms exist).
```
Estimated wall per arm **~45 min (+-12)**: rollouts ~28 min (MEASURED walls of the same kinds on the earlier chain), the rest ESTIMATED. Per arm: 7 rollouts (`<ARM>_s0 KA1<ARM>_s0 <ARM>_s1 FZ<ARM>_s0 FZ<ARM>_s1 <ARM>C_s0 <ARM>C_s1`), the 8 comparators reused (C0, C0REP, FZC0, C0C at s0/s1), 16 threads, one at a time, done markers.

## Controls I parametrised for ARM != R, and why (every other control and every rule is byte-for-byte the source's)
* **KCARD**: composite card now read from `recover_read/` (it lives there); ARM != R adds `KCARD_recover_next_card_hashes_to_FROZEN_SHA`.
* **KR1** (R's selected / eligible in [0.25, 0.35]): the selection is Bernoulli(p) over eligible rows whatever the law (`v8f_recover.py apply: u < p`), so the band scales with p. RAPR / RTL keep [0.25, 0.35]; **RP01 uses [0.0833, 0.1167] = p x [5/6, 7/6]** (the card states no band for p 0.1: my scaling, disclosed). Also requires the recorded `recover_p` to be the arm's.
* **KR2** argv audit: the arm's expected flags come from `ARM_SPEC` (p, target, rejoin flag present or absent, bank flags present or absent, `--recover-dev-law`), plus the EXACT-difference check against R's pulled argv (RAPR: +`--recover-dev-law apriori`, -2 bank flags; RTL: target logged, -`--recover-rejoin-steps`; RP01: p 0.1), plus "the arm's checkpoint differs from R's" (a mis-pull that files R under the arm's name).
* **KR3**: bank arms (RTL, RP01) keep the table / K-B3 / bank-sha check and add "BUILT levers equal the declared ones". **RAPR builds NO table** (read from the shipped trainer: `make_recover` builds a table only for the empirical law with a bank; `info.recover` unset; `levers.recover_table` = None), so KR3 becomes `KR3_apriori_law_builds_no_table_and_reads_no_bank`: `info.recover` absent, `levers.recover_table` None, built law apriori, no bank path / sha, built p / target / rejoin 10 / seed as declared.
* **KMARK** reads `rnext_markers/RUN_<ARM>_<s>.done` for the arm (comparators keep `recft_markers`). **KA1 / KFZ / KF / KK** use the arm's tags (`KA1<ARM>_s0` vs `<ARM>_s0`, `FZ<ARM>` vs `<ARM>`, floor reference `<ARM>_s{s}`).
* **Added (ARM != R only)**: `KREUSE`: all 15 done markers' `ckpt_sha256` equal the pulled checkpoint's today and `tool_sha256` equals the pinned FRAMEFREEZE tool (the 8 reused rollouts were rolled by an earlier chain).
* Paths: `WORK`, default out dir and the arm out dir moved to C: (F: is full; precedent READER_FIX_RREAD2). Extraction is deterministic (the 15 arrays are bit-identical to the source's).

## Interpretation choices in the combiner (disclose, do not hide)
1. Q1 mapping: only RTL's guards pass -> `blend`; only RP01's -> `dose`; both -> `both`; neither -> `neither`. Needs R's label HARMS (read from R's JSON), else `NOT APPLICABLE`; needs both reads, else `INCOMPLETE`.
2. Card section 4 row 4 "does not exceed R's at either seed" is read literally (at NEITHER seed). Exceeding at exactly one seed matches no row: reported `UNCOVERED` with the facts, never filled (Rule Zero: the Master Mind names the next arm).
3. "fails D2 on M1 alone" = D1 and D3 hold at both seeds and the only separated-worse row anywhere is M1; "non-M1 row" = any other row separated-worse. A DOMINANT that is not counted, or a D1 / D3 failure, matches no row (reported UNCOVERED). Rows match in table order; every matching row is listed.
4. Row 3's setting text is taken from the Q1 answer (`blend` -> target logged; `dose` -> p 0.1; `both` -> both; `neither` -> no known-safe setting).

## UNVERIFIED (nothing below has run on a real arm)
1. **Real arm files.** RAPR's `gate_selfcheck.json` shape (no `info.recover`, `levers.recover_table` None, bank fields None) is read from the shipped source, not from a real RAPR file; a different real shape makes KR3 REFUSE the read **after** ~30 min of rollouts. RTL's built `recover_rejoin_steps` = 10 (trainer default) is first seen at the RTL gate. Recommended before launching RAPR: a seconds-long pre-check of the pulled json files with `R.kr1_check / argv_audit / kr3_check` (functions are importable by file path); building it as `code/precheck_arm.py` + a chain gate 1b is the one item NOT done (context stop).
2. The arm checkpoints loading in the dev-box eval tree (the chain's 6-window pre-flight is the first test); `tree_dag` already carries the `ad0bcb72` model module (READER_FIX_RREAD1).
3. Hold_now and kbox rollouts of the arm checkpoints, `KFZ1/3`, `KK2`, `KA1` on real arm ticks (rehearsed only by redirecting the arm tags to the R rollouts).
4. Extraction RAM (9 GB free at build time; 2 workers ran fine on the known answer).
5. The `.cmd` launchers follow the proven `recover_read_chain.cmd` pattern but have not been executed; the chain's gate 1 wait loop and 6 h timeout are untested beyond the first 25 s.
6. The combiner has never seen a real arm JSON (synthetic + the real R JSON as schema).

## Manifest
`MANIFEST.txt` (path | bytes | sha256) lists every file here, including `raw/arrays/` (15 npz + meta, reused by the arm reads as their comparator arrays) and the logs.
