# HANDOFF: STRAT1 read build (2026-11-16, builder agent)

State: the 7 checkpoints are banked and pulled; every reader is built, unit-tested (20 tests green) and mutation-tested (8 of 8 mutations RED, `out/MUTATIONS.json`); a one-arm CPU smoke ran through every reader. The full read is NOT launched.

## Bank (evidence: MEASURED, hub LFS sha256 equals the manifest for every file)
* PRIVATE HF dataset `Sayood/tanitad-v8f-strat1-runs`, rev `332bbeaa`; per arm `<tag>/{ckpt_last.pt, argv.json, train_argv.json, selfcheck.json, gate_token.json, summary.json, train_log.jsonl, ARM_DONE, BANK_MANIFEST.json}`; commit `e81bde01c03a73e811734c088320f101dcec5452`.
* ckpt sha256 prefixes: C0_s0 ad91be31, N1_s0 8081bac2, S1_s0 6c2af475, C0_s1 17c751a0, N1_s1 5d4f4cbf, S1_s1 e51b756f, C0_rep 262b035b.
* Local copies, sha256-checked: `ckpts/<tag>/{ckpt_last.pt, BANK_MANIFEST.json, argv.json}` (87 MB each). Pod uploader (mine, small): pod `/root/strat1/tmp/strat1_bank_pod.py`, log `bank_strat1.log`; copy in `code/`.

## Launch (the MM does this)
`powershell -File F:\TanitAD-scratch\strat1\read\launch_wmi.ps1` (WMI, CreateFlags 0x08000200, stdin NUL; calls Git\usr\bin\bash.exe). Watch `out/strat1_read.log` for `ZZ STRAT1 READ DONE ZZ` or `ZZ STRAT1 READ FAIL ZZ <why>`. Steps: unit tests, 7 T1 dumps (2 threads, one environment), 21 T0 passes plus 2 K4 passes (about 7 min each on the full grid, E), then the three readers. Re-runnable (finished dumps and npz are skipped). Result: `out/RESULT_STRAT1.json`, with `STRAT1_STRATEGIC.json` and `STRAT1_CTX.json` beside.

## Files (all in `F:/TanitAD-scratch/strat1/read/`, not staged, not in git)
`run_dumps_strat1.py` (wrapper over the unedited OTBAND `theory/otband/read/run_dumps.py`; nav args from cfg.nav_source; refusals tested), `run_t0_strat1.py`, `strat1_k4_eval.py`, `strat1_strategic.py`, `strat1_ctx_read.py`, `strat1_read.py`, `strat1_lib.py`, `strategic_cells.py` (audit copy), `theory/` (OTBAND and OBD read files, `read_lib.py` with only its tree path changed), `READ_PINS.json` (tree blobs of e81bde01, inputs, wrapper hashes), `test_strat1_readers.py`, `mutate_check.py`, launcher files, `inputs/` (pinned npz, grid, platform sidecar), `tree/` (stack and taniteval export of e81bde01).

## Findings the MM must see
1. **K9 reads FAIL in the smoke** (T1 tick-0 `s_ckpt` against T0 real `ckpt`, ages 0 at tick 0): 12 of 48 windows differ, max 10.6 m (S1_s0); C0_s0 also differs (4.4 m), so it is not the text nav; `nav_in` at tick 0 equals the navtext row exactly. Differences sit in 2 of 6 episodes and grow with t. By the card's rule the T1 per-tick STRATEGIC row is not quoted unless this is explained. Cause not investigated.
2. **Twilight steps: 240 steps in 5 clips**, not 1 clip (card text). LABELS_REPORT's illumination 0.595 counts twilight steps in the denominator (0.5947); the masked baseline is 0.6049 over steps, 0.6062 over windows. Other four constants match to 3 decimals. Both are printed in `STRAT1_CTX.json`.
3. Card 6 is silent on a both-seed cost plus a one-seed cost on another cell; reading choice r1 is in `strat1_read.py` (both labels print).
4. Smoke caveat: K3 fails on the 6-episode smoke grid by design (no variation); on the full grid K2, K3, K5, K6 pass on the labels alone (MEASURED).

## Not built (named)
Beside rows `heads_read.py` / `topk_read.py` (T's own decisions, top-3) and the cross-venue C0 against box-2 OBD-CTL: those two OTBAND readers hard-code OB tags and need the substitution driver pattern of `beside_obd.py`; copies are in `theory/otband/read/`. The nav-echo split (card 4.3) is also not built. Both are no-bar rows.

## Next
MM launches the read; then `TanitAD_BenchmarkCriteria` on `RESULT_STRAT1.json`; register rows in GOALS_AND_CLAIMS in the same turn.
