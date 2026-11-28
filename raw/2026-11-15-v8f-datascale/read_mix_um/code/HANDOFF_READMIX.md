# HANDOFF_READMIX (UM_s0 read, build agent, 2026-11-18)

State: read_mix BUILT, tested, mutated, smoked. **The real read is NOT launched: F: has 21.8 GB free and the card bar (and the launcher) is 25 GB.** Nothing committed or pushed.

## Next step (exact)
1. Free at least 3.5 GB on F: (the study folders hold only 250 MB; the rest is other sessions), or the MM waives the bar in writing.
2. `python F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 30` (11 GB commit headroom at build time).
3. `powershell -File F:\TanitAD-scratch\datascale\read_mix\launch_wmi_mix.ps1` (WMI, CreateFlags 0x08000200, `< NUL`, Git `usr\bin\bash.exe`).
4. Sentinel: last line of `F:/TanitAD-scratch/datascale/read_mix/out/dscale_mix.log` is `ZZ DSCALE MIX READ DONE ZZ` or `ZZ DSCALE MIX READ FAIL ZZ <why>`. Result: `read_mix/out/RESULT_DSCALE_MIX.json` + `.md`. About 30-45 min (the smokes ran 50-90 s each at B 2,000).
   Re-runnable (finished dumps / evals / route passes skipped). The launcher order: card, amendment UM-A1 and replay-record sha256; block-1/3 K-EVALX records; F: bar; wait_commit; tests whole; make_banker_mix.py; ds_pull; check-only of 6 checkpoints; sync_refs; 1 dump, 4 mech evals, 5 route passes + K-EVALX; reader.

## What to read in the result
`outcome` (every UM label carries "read under UM-A1 (PI-accepted 2026-11-18)" and the as-committed line "tac_src_empty_batches 10 > 5, BLOCKED as committed"); `h_dilution` (TAC-LAT rows UM-UE, UM-BL, F_cell 0.0767); `beside_labels` (STRAT-KEPT, TAC-LON-PRICE, other family costs); `families` (four families per pair, per band, STRATEGIC in full); `controls` (K1, K-RE B3, K-EVALX); `k_tacsrc`; `gate_and_post`; `goal_f1_beside`. ONE seed: not quotable as a lever effect before the replicate.

## Built (all in `F:/TanitAD-scratch/datascale/read_mix/`)
* `ds_read_mix.py` (reader: label_mix, outcome_mix with BLOCKED > CONTROL-FAILED > NOT-YET > label, ktacsrc under UM-A1, gate/Adam-252 check, pins of card 0bc241f5 / amendment b2b2c4f0 / replay record b8b9bc51), `make_banker_mix.py` (accepts exactly: pod `blockum.log` ends in the single K-TACSRC refusal, UM_s0.A BANKED with sha 39de0b05, hub LFS equal; fetched the banked gate checks (20 of 20 pass) and post.json (252 of 252 Adam state) into `evidence/`), `sync_refs.py`, `ds_lib.py` / `ds_run.py` / `ds_pull.py` (block "mix"), `ds_t0.py` (adds `_remap_verified`: the copied UE/BL evals name the deleted read3 checkpoints; the sha-verified re-pull in `ckpts/` is used), launcher `.sh/.cmd/.ps1`, tests, `mutate_check_mix.py`, `smoke_ds_read_mix.py`.
* Checkpoints re-pulled to `read_mix/ckpts/`: UM_s0.A (sha 39de0b05, 61,568 steps), UE_s0.A, BL_s0.A; the six checkpoints verify against their bank records (F: bar waived with `--min-free 5` for the pulls and the check, logged here). UM summary sha 0d3063f0 equals the ledger's.
* Deviation from the card: UE/BL checkpoints live in `read_mix/ckpts` (read3 is read only); references' dumps/route passes are read from read/read3 in place; the six references' mech evals are copied (sha-verified) by `sync_refs.py`.

## Verification (MEASURED, CPU)
* `out/tests_all.txt`: 106 passed, 2 skipped (65 new + block-1 file against this copy).
* Mutations: first run `out/MUTATIONS_MIX.full_first_run.json` 66 of 73 RED; the 7 survivors fixed by stricter tests (one equivalent mutant replaced), re-run `out/MUTATIONS_MIX.partial.json` 8 of 8 RED, baseline GREEN; reader wiring `out/MUTATIONS_MIX_SMOKE.json` 4 of 4 RED.
* Smokes at B 2,000 (`out/smoke_mix_bl.txt`, `out/smoke_mix_ue.txt`): UM := BL reads DILUTION-CONFIRMED, UM-UE TAC-LAT +0.0855 [+0.0492, +0.1216] (the mirror of the banked UE-BL row), UM-BL 0.0 [0, 0] on every cell; UM := UE reads NOT-DILUTION, UM-UE 0.0 [0, 0], UM-BL equals the banked UE-BL row. K1, K-RE (B3: 234 + 234 + 234 T1/S rows, floors, ADE6, mechanism rows, STRATEGIC) and K-EVALX pass, no problems, UM-A1 line printed beside the label.

## Deletions (own temp, literal paths)
`read_mix/ckpts/{UM_s0.A,UE_s0.A,BL_s0.A}/_hf` 8 KB each; `read_mix/bt` 1,094 KB; `read_mix/mut` (mutation copies, removed by the script, one stale 1,754 KB removed by hand); `read_mix/smoke` (stand-in fixtures, about 70 MB each, re-creatable by `smoke_ds_read_mix.py`); `read_mix/evidence/_hf`. Kept: `ckpts/` 250 MB (re-pullable from HF; delete after the result lands if F: is short).
